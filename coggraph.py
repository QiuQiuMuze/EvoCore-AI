# coggraph.py
import uuid
from CogUnit import CogUnit
import torch
import random
from env import GridEnvironment
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque, Counter
from env import logger



# class LimitedDebugHandler(logging.Handler):
#     def __init__(self, capacity=100):
#         super().__init__(level=logging.DEBUG)  # 只处理 DEBUG
#         self.buffer = deque(maxlen=capacity)
#
#     def emit(self, record):
#         if record.levelno == logging.DEBUG:
#             try:
#                 msg = self.format(record)
#                 self.buffer.append(msg)
#             except Exception:
#                 pass  # 防止格式化报错
#
#     def dump_to_console(self):
#         print("\n==== [最近 Debug 日志] ====")
#         for msg in self.buffer:
#             print(msg)
#
# # === 设置 root logger ===
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.handlers.clear()  # ✅ 防止重复打印（关键一步！）
#
# # ✅ 添加 Debug 缓存 Handler（不会显示、不输出、仅内存）
# debug_handler = LimitedDebugHandler(capacity=100)
# debug_handler.setFormatter(logging.Formatter('%(asctime)s [DEBUG] %(message)s', datefmt='%H:%M:%S'))
# logger.addHandler(debug_handler)
#
# # ✅ 添加正常输出 Handler（只显示 INFO 及以上）
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
# logger.addHandler(console_handler)




MAX_CONNECTIONS = 4  # 每个单元最多连接 4 个下游
N_STATE_CHANNELS = 3
N_GOAL_CHANNELS = 1
INPUT_CHANNELS = N_STATE_CHANNELS + N_GOAL_CHANNELS





class TaskInjector:
    def __init__(self, target_position):
        self.target_position = target_position  # 目标坐标 (x, y)

    def encode_goal(self, env_size):
        """将目标位置编码成 one-hot 向量（与输入同维度）"""
        index = self.target_position[1] * env_size + self.target_position[0]
        vec = torch.zeros(env_size * env_size)
        vec[index] = 1.0
        return vec

    def evaluate(self, env, emitter_outputs):
        """评估 emitter 是否“指向”目标位置"""
        if emitter_outputs is None:
            return False

        pred_index = torch.argmax(emitter_outputs.mean(dim=0)).item()
        x, y = pred_index % env.size, pred_index // env.size
        return (x, y) == self.target_position



# 理想比例  emitter : processor : sensor = 1 : 2 : 1
IDEAL_RATIO = {"emitter": 1, "processor": 2, "sensor": 1}
DENOM = sum(IDEAL_RATIO.values())      # =4

# 每轮允许转换的最高比例（15%）
MAX_CONV_FRAC = 0.4

# Δ 容差系数：需要至少 diff ≥ ceil(TOL_FRAC*total) 才触发
TOL_FRAC = 0.05      # 小规模时自动退化成 1


class CogGraph:
    """
    CogGraph 管理所有 CogUnit 的集合和连接关系：
    - 添加 / 删除单元
    - 管理连接（可拓展为图）
    - 调度每一轮所有 CogUnit 的更新、分裂、死亡，并传递输出
    """

    # -------------------------------------------------------------------
    # 自动生成种子细胞（sensor=1, processor=4, emitter=1，可调）
    def _init_seed_units(self,
                         n_sensor: int = 2,
                         n_processor: int = 4,
                         n_emitter: int = 2,
                         device: str = "cpu"):

        expected_input = self.env_size * self.env_size * INPUT_CHANNELS

        # 1) 创建
        sensors = [CogUnit(input_size=expected_input, role="sensor") for _ in range(n_sensor)]
        processors = [CogUnit(input_size=expected_input, role="processor") for _ in range(n_processor)]
        emitters = [CogUnit(input_size=expected_input, role="emitter") for _ in range(n_emitter)]

        # 2) 迁移到目标 device
        for u in sensors + processors + emitters:
            u.to(device)

        # 3) 加入图
        for u in sensors + processors + emitters:
            self.add_unit(u)

        # 4) 连接：sensor → processor → emitter
        for s in sensors:
            for p in processors:
                self.connect(s, p)
        for p in processors:
            for e in emitters:
                self.connect(p, e)

    # -------------------------------------------------------------------
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        # === RL 接口：Processor 输出的统一维度 ===
        self.debug = False
        self.reverse_connections = {}  # to_id -> set(from_ids)
        self.sensor_count = 0
        self.processor_count = 0
        self.emitter_count = 0
        self.energy_pool = 0.0  # 中央能量池
        # self.memory_pool = []  # 存放死亡细胞的 gene + last_output + bias info
        self.env_size = 5  # 初始环境 5x5
        self.env = GridEnvironment(size=self.env_size)  # 创建环境
        self.task = TaskInjector(target_position=(self.env_size - 1, self.env_size - 1))  # 初始目标点
        self.target_vector = self.task.encode_goal(self.env_size)  # 初始目标向量
        self.max_total_energy = 250  # 初始最大总能量
        self.target_vector = self.task.encode_goal(self.env_size)
        self.connection_usage = {}  # {(from_id, to_id): last_used_step}
        self.current_step = 0
        self.units = []
        self.connections = {}  # {from_id: {to_id: strength_float}}
        self.unit_map = {}     # {unit_id: CogUnit 实例} 快速索引单元
        self.processor_hidden_size = self.env_size * self.env_size * INPUT_CHANNELS
        # --- 在 __init__() 的最后调用 ---
        self._init_seed_units(device=device)


    def _update_global_counts(self):
        total = len(self.units)
        self.sensor_count    = sum(1 for u in self.units if u.get_role()=="sensor")
        self.processor_count = sum(1 for u in self.units if u.get_role()=="processor")
        self.emitter_count   = sum(1 for u in self.units if u.get_role()=="emitter")
        # 动态计算目标容量：例如  max(50, total//2)  随细胞数线性增长
        target_mem_cap = max(50, total // 2)
        for u in self.units:
            u.global_sensor_count    = self.sensor_count
            u.global_processor_count = self.processor_count
            u.global_emitter_count   = self.emitter_count
            u.global_unit_count      = total

    def _log_stats_and_conns(self):
        """集中打印一次统计 & 连接强度，避免散落在内层循环里重复计算"""
        # 只有在 debug 模式下才输出
        if not self.debug:
            return
        # 每 50 步 或者前 10 步才打印
        if self.current_step % 50 != 0 and self.current_step >= 10:
            return

        # 快速算一次
        s = sum(1 for u in self.units if u.get_role()=="sensor")
        p = sum(1 for u in self.units if u.get_role()=="processor")
        e = sum(1 for u in self.units if u.get_role()=="emitter")
        logger.info(f"[统计] step={self.current_step} | sensor:{s}, processor:{p}, emitter:{e}")

        # 再把所有连接强度 dump 一遍
        logger.debug("[连接强度]")
        for frm, to_dict in self.connections.items():
            for to, strg in to_dict.items():
                logger.debug(f"  {frm} → {to} = {strg:.3f}")

    def add_unit(self, unit: CogUnit):
        # --- 若图中已有单元，则让新单元跟随它们的 device ---
        if self.units:
            target_device = self.units[0].device
            if unit.device != target_device:
                unit.to(target_device)
        # -----------------------------------------------
        # 将单元加入图结构中
        self.units.append(unit)
        self.unit_map[unit.id] = unit
        self.connections[unit.id] = {}
        self._update_global_counts()

    def _get_min_target_counts(self):
        """
        根据当前 max_total_energy 和角色比例，返回每类角色的最小建议数量。
        """
        total_target = int(self.max_total_energy / 2.6 * 0.9)  # 系统最大细胞数 × 0.9 安全系数

        # 理想比例：1(sensor) : 2(processor) : 1(emitter) → 总共 4 份
        IDEAL_RATIO = {"sensor": 1, "processor": 2, "emitter": 1}
        DENOM = sum(IDEAL_RATIO.values())  # = 4

        target_counts = {
            role: int(total_target * IDEAL_RATIO[role] / DENOM)
            for role in IDEAL_RATIO
        }
        return target_counts

    def remove_unit(self, unit: CogUnit):


        if unit.id not in self.unit_map:
            return  # 已经被删除

        # ✅ 遗产机制：寿终正寝时，能量分配给年轻后辈
        if getattr(unit, "death_by_aging", False) and unit.energy > 0.0:
            heirs = [u for u in self.units if u.role == unit.role and u.age < 240 and u.id != unit.id]
            if heirs:
                per_gain = unit.energy / len(heirs)
                for u in heirs:
                    u.energy += per_gain
                logger.info(
                    f"[寿终能量继承] {unit.id} 死亡（{unit.role}） → 能量 {unit.energy:.2f} 分给 {len(heirs)} 个同类年轻单元，每人 +{per_gain:.2f}")

        # ✅ 加入到同类局部记忆池
        if unit.is_worthy_of_memory():
            for other in self.units:
                if other.role == unit.role:
                    other.local_memory_pool.append({
                        "gene": unit.gene.copy(),
                        "output": unit.last_output.clone(),
                        "role": unit.role,
                        "hidden_size": unit.hidden_size,
                        "score": 0
                    })
                    # 控制大小：每个单元池最多150条
                    if len(other.local_memory_pool) >= 150:
                        other.local_memory_pool.pop(0)

        # 从图中移除单元及其连接
        self.units = [u for u in self.units if u.id != unit.id]
        if unit.id in self.connections:
            del self.connections[unit.id]
        if unit.id in self.unit_map:
            del self.unit_map[unit.id]
        for k in self.connections:
            if unit.id in self.connections[k]:
                del self.connections[k][unit.id]
                self.reverse_connections.get(unit.id, set()).discard(k)
        self._update_global_counts()

        # 把这个被删单元当成“from”的所有反向索引都清理掉
        for to_id, from_set in self.reverse_connections.items():
            if unit.id in from_set:
                from_set.discard(unit.id)
        # 然后再把自己那条 key 删掉
        self.reverse_connections.pop(unit.id, None)

    def connect(self, from_unit: CogUnit, to_unit: CogUnit):
        # 仅允许合法结构连接
        valid_links = {
            "sensor": ["processor"],
            "processor": ["processor", "emitter"],
            "emitter": []
        }
        from_role = from_unit.get_role()
        to_role = to_unit.get_role()

        if to_role not in valid_links.get(from_role, []):
            logger.debug(f"[非法连接阻止] 不允许 {from_role} → {to_role}，跳过连接 {from_unit.id} → {to_unit.id}")
            return  # 🚫 阻止非法连接

        if from_unit.id not in self.connections:
            self.connections[from_unit.id] = {}  # to_id → strength

        if to_unit.id in self.connections[from_unit.id]:
            return

        # 超过上限时，移除 strength 最弱的连接
        if len(self.connections[from_unit.id]) >= MAX_CONNECTIONS:
            weakest_id = min(
                self.connections[from_unit.id],
                key=lambda uid: self.connections[from_unit.id][uid]
            )
            del self.connections[from_unit.id][weakest_id]
            self.reverse_connections.get(weakest_id, set()).discard(from_unit.id)

            logger.debug(f"[连接替换] {from_unit.id} 移除最弱连接 {weakest_id}")

        # 建立新连接，初始权重为 1.0
        self.connections[from_unit.id][to_unit.id] = 1.0
        strength = self.connections[from_unit.id][to_unit.id]
        logger.debug(f"[连接建立] {from_unit.id} → {to_unit.id} (strength={strength:.2f})")
        # 同步维护反向索引
        self.reverse_connections.setdefault(to_unit.id, set()).add(from_unit.id)

    def total_energy(self):
        return sum(unit.energy for unit in self.units if unit.age < 240)

    # ========== 维度适配辅助 ==========
    def _goal_dim(self) -> int:
        """返回当前目标向量长度 (= env_size²)"""
        return self.env_size * self.env_size

    def _align_to_goal_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        把任意长度的向量对齐到 env_size²：
        - 如果恰好相等 → 原样
        - 如果能整除 → reshape(k, goal_dim) 后求均值 → goal_dim
          （默认 4 通道时相当于把 4 个通道压缩成 1 通道）
        - 如果更长但无法整除 → 截断到前 goal_dim
        - 如果更短 → 右侧补零
        """
        goal_dim = self._goal_dim()
        length = tensor.shape[-1]

        if length == goal_dim:
            return tensor

        if length % goal_dim == 0:
            k = length // goal_dim
            return tensor.reshape(-1, k, goal_dim).mean(dim=1).squeeze(0)

        if length > goal_dim:            # 截断
            return tensor[..., :goal_dim]

        # length < goal_dim  → 右补零
        pad = (0, goal_dim - length)
        return torch.nn.functional.pad(tensor, pad)

    # ------------------------------------------------------------------
    # 🆕 供强化学习调用的简化接口
    def reset_state(self):
        """
        每个 episode 开始时调用。这里只清零瞬时计数器，
        不重置能量 / age 等长期指标。
        """
        for u in self.units:
            u.call_history.clear()
            u.inactive_steps = 0
        # —— 若要每个 episode 从头开始，请取消下面注释 ——
        # self.current_step = 0
        # self.energy_pool   = self.initial_energy_pool  # 在 __init__ 中保存初始值
        # self.connections   = {u.id: {} for u in self.units}
        # self.reverse_connections = {u.id: set() for u in self.units}
        # # 如有必要，也重置每个单元的 age / energy / subsystem_id 等
        # for u in self.units:
        # u.age = 0
        # u.energy = u.initial_energy  # 需在 CogUnit 中保存初始能量



    def sensor_forward(self, env_state_np):
        """
        Args:
            env_state_np : np.ndarray 或 torch.Tensor (size=N)
        Returns:
            torch.Tensor (size = env_state_np.size) —— 作为 sensor 输出
        """
        dev = self.device  # ← 统一目标设备
        x = torch.as_tensor(env_state_np, dtype=torch.float32, device=dev).view(-1)
        if x.numel() < self.processor_hidden_size:
            pad = (0, self.processor_hidden_size - x.numel())
            x = torch.nn.functional.pad(x, pad)
        else:
            x = x[: self.processor_hidden_size]
        sensors = [u for u in self.units if u.get_role() == "sensor"]
        if sensors:
            outs = []
            for s in sensors:
                s.update(x.unsqueeze(0))
                outs.append(s.get_output().view(-1))
            return torch.stack(outs).mean(dim=0)
        return x.to(dev)

    def processor_forward(self, sensor_out):
        """
        Args:
            sensor_out : torch.Tensor 1-D
        Returns:
            torch.Tensor (size = self.processor_hidden_size)
        """
        dev = self.device  # ← 统一目标设备
        sensor_out = sensor_out.to(dev)
        processors = [u for u in self.units if u.get_role() == "processor"]
        if processors:
            inp = sensor_out.unsqueeze(0)  # (1,D)
            outs = []
            for p in processors:
                p.update(inp)
                outs.append(p.get_output().view(-1))
            merged = torch.stack(outs).mean(dim=0)
        else:
            merged = sensor_out

        # —— 统一到 processor_hidden_size ——
        if merged.numel() < self.processor_hidden_size:
            pad = (0, self.processor_hidden_size - merged.numel())
            merged = torch.nn.functional.pad(merged, pad)
        else:
            merged = merged[: self.processor_hidden_size]
        return merged.to(dev)

    def emitter_forward(self, processor_out):
        """
        把 processor_out 递给所有 emitter 做一次更新；
        不要求返回值（若你想调试，可 return 平均输出）。
        """
        dev = self.device  # ← 统一目标设备
        processor_out = processor_out.to(dev)
        emitters = [u for u in self.units if u.get_role() == "emitter"]
        if emitters:
            inp = processor_out.unsqueeze(0)  # (1,D)
            for e in emitters:
                e.update(inp)

    def merge_redundant_units(self):
        merged_pairs = set()
        new_units = []

        for i, u1 in enumerate(self.units):
            for j in range(i + 1, len(self.units)):
                u2 = self.units[j]

                # 跳过已标记
                if u1.id in merged_pairs or u2.id in merged_pairs:
                    continue

                # 必须是 processor 或 emitter
                if u1.get_role() != u2.get_role():
                    continue

                # 距离判断
                def euclidean(p1, p2):
                    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

                dist = euclidean(u1.get_position(), u2.get_position())
                if dist > 3.0:
                    continue

                # 输出相似度判断（cosine similarity）
                # === 输出相似度判断（cosine similarity）===
                output1 = u1.get_output().squeeze(0)
                output2 = u2.get_output().squeeze(0)

                # 🔥 自动补零到当前环境预期尺寸
                target_dim = max(output1.shape[0], output2.shape[0])

                if output1.shape[0] < target_dim:
                    padding = (0, target_dim - output1.shape[0])
                    output1 = torch.nn.functional.pad(output1, padding, value=0)

                if output2.shape[0] < target_dim:
                    padding = (0, target_dim - output2.shape[0])
                    output2 = torch.nn.functional.pad(output2, padding, value=0)

                sim = F.cosine_similarity(output1, output2, dim=0).item()

                if sim < 0.95:
                    continue

                # ✅ 满足条件，执行合并
                logger.info(f"[合并触发] {u1.id} 和 {u2.id} 合并为新单元")

                merged = CogUnit(role=u1.get_role())
                merged.position = (
                    (u1.get_position()[0] + u2.get_position()[0]) // 2,
                    (u1.get_position()[1] + u2.get_position()[1]) // 2
                )
                merged.state = (u1.state + u2.state) / 2
                merged.age = int((u1.age + u2.age) / 2)
                merged.energy = u1.energy + u2.energy + 0.1  # 奖励合并能量
                merged.last_output = (u1.last_output + u2.last_output) / 2

                # 加入新单元
                new_units.append(merged)
                merged_pairs.update({u1.id, u2.id})

                # 重定向连接：
                for from_id, to_dict in self.connections.items():
                    if u1.id in to_dict or u2.id in to_dict:
                        if from_id in self.unit_map:  # 防止上游已经被合并删除
                            self.connect(self.unit_map[from_id], merged)

                for to_id in self.connections.get(u1.id, {}):
                    if to_id in self.unit_map:  # ✅ 防止连接到已被删除的单元
                        self.connect(merged, self.unit_map[to_id])
                        logger.debug(f"[连接重定向] {merged.id} → {to_id}（继承自 {u1.id}）")

                for to_id in self.connections.get(u2.id, {}):
                    if to_id in self.unit_map:
                        self.connect(merged, self.unit_map[to_id])
                        logger.debug(f"[连接重定向] {merged.id} → {to_id}（继承自 {u2.id}）")

        # 执行删除 & 添加
        for uid in merged_pairs:
            if uid in self.unit_map:
                logger.info(f"[合并删除] {uid}")
                self.remove_unit(self.unit_map[uid])

        for u in new_units:
            self.add_unit(u)
        self._update_global_counts()

    def restructure_common_subgraphs(self):
        """
        检查并重构高度共现、输出相似的 processor → emitter 子图结构。
        将它们合并为一个新子图：new_processor → new_emitter
        """
        candidates = []

        # 遍历所有 processor → emitter 连接
        for u1 in self.units:
            if u1.get_role() != "processor":
                continue
            for eid in self.connections.get(u1.id, {}):
                if eid not in self.unit_map:
                    continue
                u2 = self.unit_map[eid]
                if u2.get_role() != "emitter":
                    continue
                candidates.append((u1, u2))

        # 检查每对子图是否满足共现与输出相似
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                p1, e1 = candidates[i]
                p2, e2 = candidates[j]

                if p1.id == p2.id or e1.id == e2.id:
                    continue

                # 1. 检查共现（最近 5 步调用频率都不为 0）
                if min(p1.call_history[-3:], default=0) == 0 or min(p2.call_history[-3:], default=0) == 0:
                    continue

                # 2. 检查输出相似性
                import torch.nn.functional as F
                out1 = p1.get_output()
                out2 = p2.get_output()

                # 🔥 自动补零到当前环境 target_dim
                target_dim = self.env_size * self.env_size * INPUT_CHANNELS

                if out1.shape[-1] < target_dim:
                    padding = (0, target_dim - out1.shape[-1])
                    out1 = torch.nn.functional.pad(out1, padding, value=0)

                if out2.shape[-1] < target_dim:
                    padding = (0, target_dim - out2.shape[-1])
                    out2 = torch.nn.functional.pad(out2, padding, value=0)

                sim_p = F.cosine_similarity(out1, out2, dim=-1).item()

                # 统一计算 processor 输出与 emitter 输出的相似性，确保维度一致
                out1 = p1.get_output()
                out2 = p2.get_output()
                out_e1 = e1.get_output()
                out_e2 = e2.get_output()

                max_dim = max(
                    out1.shape[-1], out2.shape[-1],
                    out_e1.shape[-1], out_e2.shape[-1],
                    self.env_size * self.env_size * INPUT_CHANNELS
                )

                def pad_to(tensor, target_dim):
                    if tensor.shape[-1] < target_dim:
                        padding = (0, target_dim - tensor.shape[-1])
                        return F.pad(tensor, padding, value=0)
                    return tensor

                out1 = pad_to(out1, max_dim)
                out2 = pad_to(out2, max_dim)
                out_e1 = pad_to(out_e1, max_dim)
                out_e2 = pad_to(out_e2, max_dim)

                sim_p = F.cosine_similarity(out1, out2, dim=-1).item()
                sim_e = F.cosine_similarity(out_e1, out_e2, dim=-1).item()

                if sim_p > 0.95 and sim_e > 0.95:
                    # ✅ 满足重构条件
                    logger.info(f"[重构触发] 子图 ({p1.id}→{e1.id}) 与 ({p2.id}→{e2.id}) 相似，开始重构")

                    # 创建新单元
                    new_p = CogUnit(role="processor")
                    new_e = CogUnit(role="emitter")
                    new_p.state = (p1.state + p2.state) / 2
                    new_p.last_output = (p1.last_output + p2.last_output) / 2
                    new_e.state = (e1.state + e2.state) / 2
                    new_e.last_output = (e1.last_output + e2.last_output) / 2

                    new_p.energy = p1.energy + p2.energy + 0.1
                    new_e.energy = e1.energy + e2.energy + 0.1

                    # 插入新单元
                    self.add_unit(new_p)
                    self.add_unit(new_e)
                    self.connect(new_p, new_e)

                    # 将所有连接到 p1 / p2 的上游指向 new_p
                    for uid in list(self.unit_map):
                        if p1.id in self.connections.get(uid, {}) or p2.id in self.connections.get(uid, {}):
                            self.connect(self.unit_map[uid], new_p)

                    # 删除原子图
                    logger.info(f"[重构删除] 删除原子图 ({p1.id}→{e1.id}) 和 ({p2.id}→{e2.id})")
                    self.remove_unit(p1)
                    self.remove_unit(p2)
                    self.remove_unit(e1)
                    self.remove_unit(e2)

                    # 重构后更新全局计数
                    self._update_global_counts()
                    return  # 每轮只重构一组…

    def assign_subsystems(self, min_size=3, max_size=10):
        """
        自动发现局部高密度连接区域，标记为子系统
        """
        visited = set()
        subsystem_count = 0

        for unit in self.units:
            if unit.id in visited:
                continue

            # 以当前unit为起点，做局部DFS
            cluster = self._dfs_collect_cluster(unit, max_depth=2)

            if min_size <= len(cluster) <= max_size:
                subsystem_id = f"subsys-{subsystem_count}"
                for u in cluster:
                    u.subsystem_id = subsystem_id
                subsystem_count += 1
                logger.info(f"[子系统生成] 新子系统 {subsystem_id}，包含 {len(cluster)} 个单元")
                visited.update(u.id for u in cluster)

    def _dfs_collect_cluster(self, start_unit, max_depth=2):
        """
        辅助：深度优先搜索，找出局部连接的单元群
        """
        cluster = set()
        stack = [(start_unit, 0)]
        while stack:
            unit, depth = stack.pop()
            if depth > max_depth or unit in cluster:
                continue
            cluster.add(unit)
            for neighbor_id in self.connections.get(unit.id, {}):
                neighbor = self.unit_map.get(neighbor_id)
                if neighbor:
                    stack.append((neighbor, depth + 1))

        return list(cluster)

    def prune_connections(self, prune_ratio=0.2, strengthen_ratio=1.5):
        """
        自动剪掉低效连接，强化高效连接
        :param prune_ratio: 小于全局平均调用频率 * prune_ratio 的连接会被剪掉
        :param strengthen_ratio: 大于全局平均调用频率 * strengthen_ratio 的连接会被强化
        """
        if not self.connection_usage:
            return

        usage_values = list(self.connection_usage.values())
        avg_usage = sum(usage_values) / len(usage_values)

        to_prune = []
        to_strengthen = []

        for conn, usage in self.connection_usage.items():
            if usage < avg_usage * prune_ratio:
                to_prune.append(conn)
            elif usage > avg_usage * strengthen_ratio:
                to_strengthen.append(conn)

        # 剪掉低效连接
        for conn in to_prune:
            from_unit, to_unit = conn
            if to_unit in self.connections.get(from_unit, {}):
                del self.connections[from_unit][to_unit]  # ✅ 删除 dict 的 key
                self.reverse_connections.get(to_unit, set()).discard(from_unit.id)

            logger.debug(f"[剪枝] 连接 {from_unit} → {to_unit} 被剪掉")
            # 也删掉 usage记录
            if conn in self.connection_usage:
                del self.connection_usage[conn]

        # 强化高效连接（可选：比如增加能量传递权重等）
        for conn in to_strengthen:
            # 简单打印标记，可以后续加真实权重系统
            logger.debug(f"[强化] 连接 {conn[0]} → {conn[1]} 被强化")

        logger.info(f"[剪枝] 剪掉 {len(to_prune)} 条弱连接，强化 {len(to_strengthen)} 条强连接")

    def auto_connect(self):
        def euclidean(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        for unit in self.units:
            role = unit.get_role()

            if role == "processor":
                # processor 寻找下游连接对象（processor 或 emitter）
                target_roles = ["processor", "emitter"]
            elif role == "emitter":
                # emitter 不应该主动连接（skip）
                continue
            else:
                continue  # sensor 不参与

            current_connections = self.connections[unit.id]

            if len(current_connections) < 2:
                u_pos = unit.get_position()
                candidates = [
                    u for u in self.units
                    if u.id != unit.id and u.get_role() in target_roles
                       and abs(u.input_size - unit.input_size) <= 100
                       and u.id not in current_connections
                       and euclidean(u.get_position(), u_pos) < 3
                ]

                if not candidates:
                    # 没有近邻 → 全局搜索
                    candidates = [
                        u for u in self.units
                        if u.id != unit.id and u.get_role() in target_roles
                           and u.id not in current_connections
                    ]

                if candidates:
                    def connection_strength(u):
                        incoming_count = sum(u.id in self.connections.get(fid, {}) for fid in self.unit_map)
                        return u.energy + incoming_count * 0.1

                    candidates.sort(key=connection_strength, reverse=True)

                    for target in candidates:
                        if target.id not in current_connections:
                            prev_conn_count = len(self.connections[unit.id])
                            self.connect(unit, target)
                            if len(self.connections[unit.id]) > prev_conn_count:
                                logger.debug(f"[新连接] {unit.id} → {target.id}")
                                break  # ✅ 成功建立连接就跳出

        # === 随机突变连接（只允许 processor 发起） ===
        if random.random() < 0.1:
            from_candidates = [u for u in self.units if u.get_role() == "processor"]
            to_candidates = [u for u in self.units if u.get_role() in ["processor", "emitter"]]

            if from_candidates and to_candidates:
                from_unit = random.choice(from_candidates)
                to_unit = random.choice(to_candidates)

                if to_unit.id not in self.connections.get(from_unit.id, {}):
                    self.connect(from_unit, to_unit)
                    logger.debug(f"[突变连接] {from_unit.id} → {to_unit.id}")

    # === 分化机制：结构失衡时的角色调整 ===
    def rebalance_cell_types(self):
        from collections import Counter
        total = len(self.units)
        if total < 15:
            return  # 太小先自由生长

        # 动态迟滞窗口  ──────────────────────────
        #   总数   <50   <200   <500   500+
        #   hi    1.50  1.30   1.15   1.08
        #   lo    0.50  0.70   0.85   0.92
        if total < 50:
            hi, lo = 1.30, 0.60
        elif total < 200:
            hi, lo = 1.08, 0.92
        elif total < 500:
            hi, lo = 1.05, 0.95
        else:
            hi, lo = 1.03, 0.97

        # Δ 容差（至少相差 Δ_cell 才算“真的多／少”）
        delta_cell = max(1, int(total * TOL_FRAC))

        # 本轮最多转换
        max_conv = max(1, int(total * MAX_CONV_FRAC))
        conv_done = 0

        def pick_weakest(units):
            return min(units, key=lambda u: (u.energy,
                                             getattr(u, "avg_recent_calls", 0.0)))

        while conv_done < max_conv:
            # ── 重新计数
            young_units = [u for u in self.units if u.age < 240]
            cnt = Counter(u.get_role() for u in young_units)
            s_cnt = cnt.get("sensor", 0)
            p_cnt = cnt.get("processor", 0)
            e_cnt = cnt.get("emitter", 0)

            desired = {
                "sensor": total * IDEAL_RATIO["sensor"] / DENOM,
                "processor": total * IDEAL_RATIO["processor"] / DENOM,
                "emitter": total * IDEAL_RATIO["emitter"] / DENOM,
            }

            # ratio & diff
            ratio = {
                "sensor": s_cnt / (desired["sensor"] or 1),
                "processor": p_cnt / (desired["processor"] or 1),
                "emitter": e_cnt / (desired["emitter"] or 1),
            }
            diff = {
                "sensor": s_cnt - desired["sensor"],
                "processor": p_cnt - desired["processor"],
                "emitter": e_cnt - desired["emitter"],
            }

            # 1) 满足 ratio>hi 且 diff≥Δ 才算“over”   2) ratio<lo 且 diff≤-Δ 算“under”
            overs = [r for r in ratio if ratio[r] > hi and diff[r] >= delta_cell]
            unders = [r for r in ratio if ratio[r] < lo and diff[r] <= -delta_cell]

            if not overs or not unders:
                break  # 落入迟滞带 or Δ 太小，结束

            # 选最过量 & 最不足
            giver_role = max(overs, key=lambda r: diff[r])  # diff 最大
            receiver_role = min(unders, key=lambda r: diff[r])  # diff 最小(负数)

            # 取 giver_role 最弱者
            cand = [u for u in self.units if u.get_role() == giver_role]
            if not cand:
                break
            unit = pick_weakest(cand)

            # ── 转化
            old = unit.get_role()
            unit.role = receiver_role
            unit.age = 0
            unit.energy += 0.2
            unit.gene[f"{receiver_role}_bias"] = 1.0
            logger.info(f"[平衡] {old}→{receiver_role} | step={self.current_step}")

            # 清旧连 & 简易新连
            for uid, out_edges in list(self.connections.items()):
                out_edges.pop(unit.id, None)
                self.connection_usage.pop((uid, unit.id), None)
            self.connections[unit.id] = {}

            if receiver_role == "processor":
                tgt = max((u for u in self.units if u.get_role() == "emitter"),
                          default=None, key=lambda u: u.energy)
                if tgt: self.connect(unit, tgt)
            elif receiver_role == "emitter":
                src = max((u for u in self.units if u.get_role() == "processor"),
                          default=None, key=lambda u: u.energy)
                if src: self.connect(src, unit)
            elif receiver_role == "sensor":
                tgt = max((u for u in self.units if u.get_role() == "processor"),
                          default=None, key=lambda u: u.energy)
                if tgt: self.connect(unit, tgt)

            conv_done += 1
            self._update_global_counts()


    def trace_info_paths(self):
        logger.debug(f"[信息路径追踪] 步数 {self.current_step}")
        for emitter in self.units:
            if emitter.get_role() != "emitter":
                continue

            # 追溯上游 processor
            emit_from = [pid for pid in self.unit_map if emitter.id in self.connections.get(pid, {})]
            for pid in emit_from:
                proc_from = [sid for sid in self.unit_map if pid in self.connections.get(sid, {})]
                for sid in proc_from:
                    logger.debug(f"  sensor:{sid} → processor:{pid} → emitter:{emitter.id}")

    def _select_clone_parents(self, pending_by_role):
        """
        从待复制父单元中，按照配额 & 能量/活跃度排序挑出真正允许复制的。
        若细胞能量超过 3.0，强制允许复制，不受比例限制。
        """
        total_cells = len(self.units)
        approved = set()

        if total_cells <= 15:
            for lst in pending_by_role.values():
                approved.update(lst)
        else:
            for role, cand in pending_by_role.items():
                if not cand:
                    continue
                young_units = [u for u in self.units if u.role == role and u.age < 240]
                role_count = len(young_units)
                cap = max(1, (2 * role_count) // 5)  # 40%
                cand.sort(key=lambda u: (u.energy, u.avg_recent_calls), reverse=True)
                approved.update(cand[:cap])

        return list(approved)

    def trim_weak_memories(self):
        """环境发生变化时，清除所有细胞记忆池中的一半最弱记忆"""
        for unit in self.units:
            if hasattr(unit, "local_memory_pool") and unit.local_memory_pool:
                pool = unit.local_memory_pool
                pool.sort(key=lambda m: m["score"])
                half = len(pool) // 2
                del pool[:half]

    def step(self, input_tensor: torch.Tensor):
        self._update_global_counts()
        self.current_step += 1
        env_state = torch.from_numpy(self.env.get_state()).float().to(self.device).unsqueeze(0)
        goal_tensor = self.task.encode_goal(self.env_size).unsqueeze(0).to(self.device)

        if self.current_step == 10000:
            self.subsystem_competition = True
            logger.info("[进化] 子系统竞争机制已激活（Subsystem Competition）")

        if self.current_step == 2000:
            for unit in self.units:
                unit.dynamic_aging = True
            logger.info("[进化] 动态寿命机制已激活（Dynamic Aging）")



        if self.current_step > 200 and self.current_step % 10 == 0:
            total_cell_energy = self.total_energy()
            pool_energy = self.energy_pool
            total_e = total_cell_energy + pool_energy
            max_e = self.max_total_energy

            if total_e > max_e:
                excess = total_e - max_e
                tiers = [
                    (0.00, 0.15, 0.01),  # 超出 0~15% 部分收 1%
                    (0.15, 0.35, 0.05),  # 超出 15~35% 部分收 5%
                    (0.35, 0.55, 0.10),  # 超出 35~55% 部分收 10%
                    (0.50, float("inf"), 0.50)  # 超出 55% 部分收 50%
                ]

                tax = 0.0
                for lower, upper, rate in tiers:
                    lower_abs = max_e * lower
                    upper_abs = max_e * upper
                    if excess > lower_abs:
                        taxed_amount = min(excess, upper_abs) - lower_abs
                        tax += taxed_amount * rate

                if pool_energy >= tax:
                    self.energy_pool -= tax
                    logger.info(
                        f"[能量税] {self.current_step} 步：总能 {total_e:.2f} → 累进税 {tax:.2f}（池足够，剩余池能 {self.energy_pool:.2f}）")
                else:
                    tax_from_cells = tax - self.energy_pool
                    self.energy_pool = 0.0
                    loss_per_unit = tax_from_cells / max(len(self.units), 1)
                    for unit in self.units:
                        unit.energy -= loss_per_unit
                    logger.info(
                        f"[能量税] {self.current_step} 步：总能 {total_e:.2f} → 税 {tax:.2f}，池不足 → 细胞每个扣 {loss_per_unit:.4f}")

        # === Curriculum Learning: 每500步扩展一次环境大小
        if self.current_step > 0 and self.current_step % 500 == 0:
            old_size = self.env_size
            self.env_size = min(self.env_size + 5, 20)  # 每次+5，最大到20x20
            self.env = GridEnvironment(size=self.env_size)  # 重新生成环境
            self.upscale_old_units(self.env_size * self.env_size * INPUT_CHANNELS)
            self.processor_hidden_size = self.env_size * self.env_size * INPUT_CHANNELS

            new_target = (random.randint(0, self.env_size - 1), random.randint(0, self.env_size - 1))
            self.task = TaskInjector(target_position=new_target)
            self.target_vector = self.task.encode_goal(self.env_size)
            logger.info(
                f"[Curriculum升级] 第 {self.current_step} 步：环境大小 {old_size}x{old_size} → {self.env_size}x{self.env_size}，新目标 {new_target}")

        if self.current_step > 0 and self.current_step % 1000 == 0:
            old_max = self.max_total_energy
            self.max_total_energy *= 2
            logger.info(f"[资源扩展] 第 {self.current_step} 步：MAX_TOTAL_ENERGY {old_max:.1f} → {self.max_total_energy:.1f}")

        # 若当前步数非常早期，给予基础能量补偿
        if self.current_step < 10:
            for unit in self.units:
                if unit.get_role() != "sensor":
                    unit.energy += 0.1
                    logger.debug(f"[预热补偿] {unit.id} 初始阶段获得能量 +0.1")

        if self.current_step > 0 and self.current_step % 100 == 0:
            old_target = self.target_vector.clone()
            self.target_vector = torch.rand_like(self.target_vector)

            similarity = torch.cosine_similarity(old_target, self.target_vector, dim=0).item()
            logger.info(f"[目标变化] 第 {self.current_step} 步，target_vector 更新！（相似度 {similarity:.3f}）")

        if self.current_step > 0 and self.current_step % 100 == 0:
            self.prune_connections()

        if self.current_step > 0 and self.current_step % 300 == 0:
            self.assign_subsystems()

        if hasattr(self, "subsystem_competition") and self.subsystem_competition:
            if self.current_step % 100 == 0:
                subsystem_energies = {}
                for unit in self.units:
                    if unit.subsystem_id:
                        subsystem_energies.setdefault(unit.subsystem_id, 0)
                        subsystem_energies[unit.subsystem_id] += unit.energy

                if len(subsystem_energies) >= 5:  # 至少5个子系统才竞争
                    weakest = min(subsystem_energies, key=lambda x: subsystem_energies[x])
                    logger.info(f"[子系统竞争] 淘汰能量最弱的子系统 {weakest}")

                    # 删除弱子系统的所有单元
                    self.units = [u for u in self.units if u.subsystem_id != weakest]
                    self.unit_map = {u.id: u for u in self.units}
                    self.connections = {u.id: {} for u in self.units}


        if self.current_step > 2000 and self.current_step % 40 == 0:
            total = len(self.units)
            max_elites = max(1, int(total * 0.08))  # 最多8%

            # 收集所有有记忆的分数，用于计算阈值
            all_scores = [
                u.local_memory_pool[-1]["score"]
                for u in self.units
                if len(u.local_memory_pool) >= 1
            ]
            score_threshold = np.percentile(all_scores, 90)

            candidates = []
            for u in self.units:
                # 条件1：至少5次记忆
                if len(u.local_memory_pool) < 5:
                    continue
                last_score = u.local_memory_pool[-1]["score"]
                # 条件2：高分门槛
                if last_score < score_threshold:
                    continue
                # 条件3：活跃度
                if getattr(u, "avg_recent_calls", 0) < 2.0:
                    continue

                # 条件4：输出质量 role-specific
                # 先从 local_memory_pool 最近几条里重算 quality
                hist = [m["output"].view(-1) for m in u.local_memory_pool[-5:]]
                # 对齐
                max_len = max(t.numel() for t in hist)
                aligned = [t if t.numel() == max_len else torch.nn.functional.pad(t, (0, max_len - t.numel())) for t in
                           hist]
                diffs = [(aligned[i] - aligned[i + 1]).norm().item() for i in range(len(aligned) - 1)]
                if u.role == "processor":
                    diversity = sum(diffs) / len(diffs)
                    if diversity < 0.1:
                        continue
                elif u.role == "sensor":
                    variation = torch.var(torch.stack(aligned), dim=0).mean().item()
                    if variation < 0.05:
                        continue
                elif u.role == "emitter":
                    avg_diff = sum(diffs) / len(diffs)
                    stability = 1.0 if 0.01 < avg_diff < 0.5 else 0.0
                    if stability < 1.0:
                        continue

                # 全部通过，加入候选
                candidates.append((u, last_score))

            # 按分数降序选 top K
            elites = [u for u, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:max_elites]]

            # 重置旧标记 & 标新精英
            for u in self.units:
                u.is_elite = False
            # 标记新精英 & 重置年龄
            for u in elites:
                u.is_elite = True
                u.age = 0  # ← 关键：清零年龄，让它从头开始，避免进入老化死亡窗口

        # 恢复全局计数更新，避免 should_split 拿到过时值


        new_units = []  # 新生成的单元（复制）
        pending = {"sensor": [], "processor": [], "emitter": []}  # NEW: 待复制父单元
        output_buffer = {}  # 缓存每个单元的输出 {unit_id: output_tensor}

        # === 系统总能量限制，保护 clone ===
        allow_clone = self.total_energy() < self.max_total_energy

        # === 第一阶段：单元更新处理 ===
        # 统计当前 emitter 数量
        emitter_count = sum(1 for u in self.units if u.get_role() == "emitter")

        for unit in self.units:
            unit.global_emitter_count = emitter_count


        for unit in self.units[:]:

            unit_input = torch.cat([env_state, goal_tensor], dim=1)

            # 如果该单元有上游连接（被其他单元指向）
            # O(1) 反向查找所有调用过我的
            # 1. 把 reverse_connections 里 stale 的 uid 丢掉，剩下才是真正有效的 incoming
            raw = list(self.reverse_connections.get(unit.id, set()))
            incoming = []
            for uid in raw:
                # 先检查：uid 还在 unit_map 里？uid→unit.id 这条连边还真在 connections 里？
                if uid in self.unit_map and unit.id in self.connections.get(uid, {}):
                    incoming.append(uid)
                    # 更新 usage & strength
                    self.connection_usage[(uid, unit.id)] = self.current_step
                    self.connections[uid][unit.id] = min(self.connections[uid][unit.id], 5.0)
                else:
                    # 要么单元被删了，要么连边被剪了 —— 顺便清理反向索引
                    self.reverse_connections[unit.id].discard(uid)

            if unit.get_role() == "sensor":
                # 同样使用 dim=1 拼接，再加一个 batch 维度
                unit_input = torch.cat([env_state, goal_tensor], dim=1)

            elif incoming:
                weighted_outputs = []
                total_weight = 0.0
                for uid in incoming:
                    strength = self.connections[uid][unit.id]
                    output = self.unit_map[uid].get_output().squeeze(0)  # 统一为 [8]
                    target_len = self.processor_hidden_size
                    if output.shape[0] != target_len:
                        padding = (0, target_len - output.shape[0])

                        output = torch.nn.functional.pad(output, padding, value=0)

                    weighted_outputs.append(output * strength)
                    total_weight += strength

                if total_weight > 0:
                    unit_input = torch.stack(weighted_outputs).sum(dim=0, keepdim=True) / total_weight
                else:
                    unit_input = torch.zeros_like(input_tensor).unsqueeze(0)
            else:
            # 强制使用零输入触发更新，避免因无输入永远不更新
                unit_input = torch.zeros(unit.input_size).unsqueeze(0)
                logger.debug(f"[零输入] {unit.id} 无上游连接，使用零输入更新")

            # 执行单元的更新逻辑
            # === 统计调用频率（这里可以更精细，比如 sliding window）===
            # O(1) 查 self.reverse_connections
            incoming = self.reverse_connections.get(unit.id, ())
            unit.recent_calls = len(incoming)
            unit.connection_count = len(self.connections.get(unit.id, {}))  # 直接下游连接数量
            # 更新调用历史记录
            unit.call_history.append(unit.recent_calls)
            if len(unit.call_history) > unit.call_window:
                unit.call_history.pop(0)

            # 计算平均调用频率
            unit.avg_recent_calls = sum(unit.call_history) / len(unit.call_history)

            if unit.recent_calls == 0:
                unit.inactive_steps += 1
            else:
                unit.inactive_steps = 0  # 重置

            unit.current_step = self.current_step
            # === 统一动态能量消耗 ===
            var = torch.var(unit_input).item()
            freq = unit.recent_calls
            conn = unit.connection_count
            call_density = freq / (conn + 1)
            conn_strength_sum = sum(self.connections.get(unit.id, {}).values())

            # 统一代谢模型（可调权重）
            dim_scale = 50.0 / unit.input_size  # ← 50 是最初环境(5×5×2)的基准
            bias_factor = 1.0
            if unit.role == "sensor":
                bias_factor = unit.gene.get("sensor_bias", 1.0)
            elif unit.role == "processor":
                bias_factor = unit.gene.get("processor_bias", 1.0)
            elif unit.role == "emitter":
                bias_factor = unit.gene.get("emitter_bias", 1.0)

            step_factor = 1.0 + 0.0005 * max(0, self.current_step - 500)
            unit_factor = 1.0 + 0.005 * max(0, len(self.units) - 50)

            # 代谢公式加入动态因子
            decay = (var * 0.15 + call_density * 0.04 + conn_strength_sum * 0.02) \
                    * dim_scale * bias_factor * step_factor * unit_factor

            unit.energy -= decay
            unit.energy = max(unit.energy, 0.0)

            logger.debug(
                f"[代谢] {unit.id} var={var:.3f}, freq={freq}, conn={conn}, strength_sum={conn_strength_sum:.2f} → -{decay:.3f} 能量")

            unit.update(unit_input)


            # ✅ 加强连接权重（使用次数越多越强）
            for uid in incoming:
                if unit.id in self.connections.get(uid, {}):
                    self.connections[uid][unit.id] *= 1.05  # 增强
                    self.connections[uid][unit.id] = min(self.connections[uid][unit.id], 5.0)
            output_buffer[unit.id] = unit.get_output()
            logger.debug(str(unit))

            # === 判断是否需要复制 ===
            if allow_clone and unit.should_split():
                pending[unit.role].append(unit)   # 只记录父单元，不立即 clone


            else:
                if not allow_clone:
                    logger.debug(f"[系统保护] 总能量过高，禁止 {unit.id} 分裂")

            # === 判断是否死亡 ===
            if unit.should_die():
                logger.debug(f"[死亡] {unit.id} 被移除")
                self.remove_unit(unit)


        self.auto_connect()
        # === 死连接清理 ===
        if self.current_step % 10 == 0:
            threshold = 50
            for from_id in list(self.connections.keys()):
                for to_id in list(self.connections[from_id].keys()):
                    last_used = self.connection_usage.get((from_id, to_id), -1)
                    if self.current_step - last_used > threshold:
                        del self.connections[from_id][to_id]  # ✅ 正确删除方式
                        self.reverse_connections.get(to_id, set()).discard(from_id)

                        logger.debug(f"[死连接清除] {from_id} → {to_id}")
                        # 删除连接后，给 from_unit 轻微能量惩罚
                        if from_id in self.unit_map:
                            self.unit_map[from_id].energy -= 0.015  # 可调参数
                            logger.debug(f"[惩罚] {from_id} 因连接失效，能量 -0.01")

                    else:
                        # ✅ 削弱仍在用但表现差的连接
                        self.connections[from_id][to_id] *= 0.95
                        if self.connections[from_id][to_id] < 0.1:
                            del self.connections[from_id][to_id]
                            self.reverse_connections.get(to_id, set()).discard(from_id)
                            logger.debug(f"[连接衰减清除] {from_id} → {to_id}")

        # 简易任务奖励：如果 emitter 输出靠近某个目标向量，则发放奖励
        target_vector = self.target_vector
        outputs = self.collect_emitter_outputs()
        if outputs is not None:
            avg_output = outputs.mean(dim=0)
            distance = torch.norm(avg_output - target_vector)

            # 线性衰减式奖励分数（距离 0→奖励满分1，距离3→奖励为0）
            reward_score = max(0.0, 1.0 - distance / 5.0)

            if reward_score > 0.0:
                dim_scale = self.target_vector.size(0) / 50  # 50 → 原始基准
                dilution_factor = 1.0
                if self.current_step >= 5000:  # 5000步后开始奖励稀释
                    dilution_factor = max(0.5, 1.0 - 0.00005 * (self.current_step - 5000))

                for unit in self.units:
                    if unit.get_role() == "processor":
                        unit.energy += 0.04 * dim_scale * reward_score * dilution_factor
                    elif unit.get_role() == "emitter":
                        unit.energy += 0.04 * dim_scale * reward_score * dilution_factor

                logger.debug(f"[奖励] 输出接近目标，距离 {distance:.2f}，奖励比率 {reward_score:.2f} → 能量分配完毕")

            # ✅ 增加多样性惩罚（含数量判断）
            action_indices = [torch.argmax(out).item() for out in outputs]

            if len(action_indices) >= 3:  # 至少 3 个 emitter 才有统计意义
                common_action = max(set(action_indices), key=action_indices.count)
                if action_indices.count(common_action) > len(action_indices) * 0.9:
                    for unit in self.units:
                        if unit.get_role() == "emitter":
                            unit.energy -= 0.05
                            logger.debug(f"[惩罚] emitter {unit.id} 因输出单一行为被扣能量")
                elif len(set(action_indices)) > len(action_indices) * 0.6:
                    for unit in self.units:
                        if unit.get_role() == "emitter":
                            unit.energy += 0.01

                    logger.debug(f"[奖励] emitter 输出多样性高 → 所有 emitter +0.01 能量")

            else:
                logger.debug(f"[跳过多样性惩罚] emitter 数量不足，仅 {len(action_indices)} 个")

            if self.task.evaluate(self.env, outputs):
                if self.task.evaluate(self.env, outputs):
                    logger.debug(f"[任务完成] 达成目标位置 {self.task.target_position}，奖励 +0.1")

                    for unit in self.units:
                        if unit.get_role() == "emitter":
                            unit.energy += 0.05  # 提高 emitter 奖励
                        elif unit.get_role() == "processor":
                            unit.energy += 0.06  # 给 processor 更多能量，鼓励参与

        # === 重度维护：只在部分步数执行，避免每步循环开销 ===

        # —— 可选路径追踪（纯调试，不影响状态） ——
        if self.debug and self.current_step % 50 == 0:
            self.trace_info_paths()

        # —— 统一统计 & 连接打印（仅 debug） ——
        self._log_stats_and_conns()

        self.rebalance_cell_types()

        # === 🔁 分裂 or 储能：强制处理能量超标单元 ===
        while True:
            over_energy_units = [u for u in self.units if u.energy > 3.0]
            if not over_energy_units:
                break

            min_counts = self._get_min_target_counts()
            role_counts = Counter(u.get_role() for u in self.units if u.age < 240)

            for unit in over_energy_units:
                role = unit.get_role()
                if role_counts.get(role, 0) < min_counts[role] or self.total_energy() < self.max_total_energy:
                    # ✅ 当前角色数量不足 或 系统能量未超载 → 强制分裂
                    expected_input = self.env_size * self.env_size * INPUT_CHANNELS
                    child = unit.clone(new_input_size=expected_input if unit.input_size != expected_input else None)
                    self.connect(unit, child)
                    self.auto_connect()
                    self.add_unit(child)
                    logger.info(f"[强制分裂] {unit.id} ({role}) → 数量不足/系统未满 → 复制")

                else:
                    # ⚠️ 系统能量过载 & 当前角色数量足够 → 储能
                    contribution = unit.energy * 0.5
                    unit.energy *= 0.5
                    self.energy_pool += contribution
                    logger.debug(
                        f"[能量转移] {unit.id} ({role}) 系统过载 → 存入能量池 {contribution:.2f}，保留 {unit.energy:.2f}")

        # === 40 %-限额复制（>15 细胞才触发） ===
        selected_parents = self._select_clone_parents(pending)
        for parent in selected_parents:
            expected_input = self.env_size * self.env_size * INPUT_CHANNELS

            child = parent.clone(
                new_input_size=expected_input if parent.input_size != expected_input else None
            )
            # 父子连接（含继承上下游）
            self.connect(parent, child)
            self.auto_connect()  # ✅ 让新单元自行寻找连接对象

            new_units.append(child)
            # —— 最终一次性把所有 child 加入图结构 ——
        for unit in new_units:
            self.add_unit(unit)

        # —— 定期合并 & 重构（核心算法，必须保留） ——
        if self.current_step % 100 == 0:
            self.merge_redundant_units()
            self.restructure_common_subgraphs()

        # === 🪫 能量池补给机制：支持能量低的细胞 ===
        if self.energy_pool > 0.0:
            weak_units = [u for u in self.units if u.energy < 0.8]
            if weak_units:
                per_unit = min(0.2, self.energy_pool / len(weak_units))
                for u in weak_units:
                    u.energy += per_unit
                    self.energy_pool -= per_unit
                logger.info(f"[能量补给] 从能量池为 {len(weak_units)} 个弱细胞补充 {per_unit:.2f} 能量")

    def upscale_old_units(self, new_input_size):
        """将所有 input_size 小于当前环境预期尺寸的单元升维（只升不降）"""
        for unit in self.units:
            if unit.input_size < new_input_size:
                logger.info(f"[升维] {unit.id} input_size {unit.input_size} → {new_input_size}")

                # 1. 升维 last_output
                old_output = unit.last_output
                if old_output.dim() == 2 and old_output.shape[0] == 1:
                    old_output = old_output.squeeze(0)
                padded_output = torch.zeros(new_input_size, device=old_output.device)
                padded_output[:old_output.shape[0]] = old_output
                unit.last_output = padded_output

                # 2. 检查 hidden_size 是否也要升高（只升不降）
                if unit.state.shape[0] > unit.hidden_size:
                    unit.hidden_size = unit.state.shape[0]
                    logger.info(f"[升维] {unit.id} hidden_size 升至 {unit.hidden_size}")

                # 3. 升维 state
                old_state = unit.state.squeeze(0) if unit.state.dim() == 2 else unit.state
                padded_state = torch.zeros(unit.hidden_size, device=old_state.device)
                length = min(padded_state.shape[0], old_state.shape[0])
                padded_state[:length] = old_state[:length]
                unit.state = padded_state

                # 4. 重建网络结构
                unit.function = torch.nn.Sequential(
                    torch.nn.Linear(new_input_size, unit.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(unit.hidden_size, new_input_size)
                )

                # 5. 更新 input_size
                unit.input_size = new_input_size

    def summary(self):
        # 打印当前图结构概况

        logger.debug(f"[图结构] 当前单元数: {len(self.units)}")
        for unit in self.units:
            logger.debug(f" - {unit} → 连接数: {len(self.connections[unit.id])}")

    def collect_emitter_outputs(self):
        """收集所有 emitter 输出并自动对齐到目标维度"""
        aligned = []
        for unit in self.units:
            if unit.get_role() != "emitter":
                continue

            raw = unit.get_output().squeeze(0) if unit.get_output().dim() == 2 else unit.get_output()
            vec = self._align_to_goal_dim(raw)

            if vec.shape[-1] != self._goal_dim():
                # 理论不会发生，安全检查
                logger.warning(f"[警告] 对齐失败 {unit.id} 长度 {vec.shape[-1]}")
                continue

            aligned.append(vec.unsqueeze(0))

        if aligned:
            stacked = torch.cat(aligned, dim=0)      # [N, goal_dim]
            logger.debug("[输出检查] Emitter 对齐后均值(前5) :", stacked.mean(dim=0)[:5])
            return stacked
        else:
            logger.debug("[输出检查] 当前没有活跃的 emitter 单元")
            return None



def interpret_emitter_output(output_tensor):
    """
    将 emitter 的输出向量解释为动作。
    """
    action_names = [f"动作{i}" for i in range(50)]
    if output_tensor.dim() == 3:
        output_tensor = output_tensor.squeeze(1)  # 变成 [N, 8]

    for i, out in enumerate(output_tensor):
        raw_index = torch.argmax(out).item()
        action_index = raw_index % 4  # 🌟 折叠到 0~3
        action = ["上", "下", "左", "右"][action_index]  # 或者自定义动作名称
        logger.debug(f"[行为触发] 第 {i + 1} 个 emitter 执行动作: {action}（原始 index = {raw_index}）")


def environment_feedback(output_tensor, graph):
    """
    环境对 emitter 输出的简单反馈机制：
    - 如果输出中出现特定模式（例如 ↑ 和 → 同时较强），奖励对应 emitter
    - 奖励通过提升 energy 实现
    """
    if output_tensor.dim() == 3:
        output_tensor = output_tensor.squeeze(1)  # [N, 8]

    for i, out in enumerate(output_tensor):
        # 示例条件：若 ↑ (index 0) 和 → (index 3) 输出值都大于 0.2
        if out[5] > 0.3 and out[13] < -0.1:  # 自定义规则
            emitter = [u for u in graph.units if u.get_role() == "emitter"][i]
            emitter.energy += 0.05  # 简单奖励
            logger.debug(f"[奖励] emitter {emitter.id} 因 ↑+→ 被奖励 +0.05 能量")



# if __name__ == "__main__":
#     print(logger.handlers)
#     env = GridEnvironment(size=5)
#     # 初始化任务目标（例如目标位置在右下角 (4, 4)）
#     task = TaskInjector(target_position=(4, 4))
#     goal_tensor = task.encode_goal(env.size)  # 生成 25维 one-hot 向量
#     graph = CogGraph()
#
#     # 初始化单元
#     sensor = CogUnit(input_size=graph.env_size * graph.env_size * INPUT_CHANNELS
# , role="sensor")
#     emitter = CogUnit(input_size=graph.env_size * graph.env_size * INPUT_CHANNELS
# , role="emitter")
#
#
#     # 多个 processor
#     processor_list = []
#     for _ in range(4):
#         p = CogUnit(input_size=graph.env_size * graph.env_size * INPUT_CHANNELS
# , role="processor")
#         processor_list.append(p)
#
#
#
#     # 加入图结构
#     graph.add_unit(sensor)
#     graph.add_unit(emitter)
#     for p in processor_list:
#         graph.add_unit(p)
#
#     # 建立连接
#     for p in processor_list:
#         graph.connect(sensor, p)
#         graph.connect(p, emitter)
#
#     # --- 新增：额外种子 ---
#     extra_sensors = [CogUnit(input_size=graph.env_size * graph.env_size * INPUT_CHANNELS,
#                              role="sensor") for _ in range(1)]  # 再补 1 个
#     extra_emitters = [CogUnit(input_size=graph.env_size * graph.env_size * INPUT_CHANNELS,
#                               role="emitter") for _ in range(1)]  # 再补 1 个
#     for u in extra_sensors + extra_emitters:
#         graph.add_unit(u)
#         # 让每个 sensor → processor[0]，processor[-1] → 每个 emitter，保证信息通路
#         graph.connect(u, processor_list[0]) if u.role == "sensor" else graph.connect(processor_list[-1], u)
#
#     # 运行模拟
#     for step in range(20000):
#         logger.info(f"\n==== 第 {step+1} 步 ====")
#         # 1️⃣ 获取环境状态，输入 sensor
#         state = env.get_state()
#         input_tensor = torch.cat([
#             torch.from_numpy(state).float(),
#             goal_tensor
#         ], dim=0)
#
#         # 2️⃣ 启动认知系统（感知 + 决策）
#         graph.step(input_tensor)
#
#         # 3️⃣ 收集 emitter 输出，转换为动作
#         emitter_output = graph.collect_emitter_outputs()
#         if emitter_output is not None:
#             raw_idx = torch.argmax(emitter_output.mean(dim=0)).item()
#             action_index = raw_idx % 4  # 折叠到 0-3
#             logger.debug(f"[行动决策] 执行动作: {action_index}")
#
#             # 4️⃣ 执行动作，改变环境
#             env.step(action_index)
#             # 🔁 将环境能量变化反馈给 emitter
#             for unit in graph.units:
#                 if unit.get_role() == "emitter":
#                     unit.energy += env.agent_energy_gain
#                     unit.energy -= env.agent_energy_penalty
#                     logger.debug(f"[环境反馈] {unit.id} +{env.agent_energy_gain:.2f} -{env.agent_energy_penalty:.2f}")
#
#         # 5️⃣ 打印环境图示
#         env.render()
#
#         # 如果没有单元剩下，退出d
#         if not graph.units:
#             logger.info("[终止] 所有单元死亡。")
#             break
#
# from collections import Counter
# final_counts = Counter([unit.get_role() for unit in graph.units])
# print("\n🧬 最终细胞总数统计：", dict(final_counts))
# print("🔢 总细胞数 =", len(graph.units))
# print(f"\n🧪 模拟结束后能量池剩余：{graph.energy_pool:.2f}")
#

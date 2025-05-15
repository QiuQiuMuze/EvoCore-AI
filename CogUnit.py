# cogunit.py
import torch
import uuid
import random
import logging
import logging
from collections import deque

# ======== CogUnit 全局功能开关 ========
ENABLE_MINI_LEARN = False   # ← 关闭自编码训练
FOLLOW_INPUT_DEVICE = True  # ← 自动把内部张量跟随输入 device（GPU/CPU）
# 如想完全手动控制迁移，改成 False 并仅用 .to() 方法。
MAX_OUTPUT_DIM = None       # ← 若设为 int，则 get_output() 强截断
# ====================================


class LimitedDebugHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__(level=logging.DEBUG)  # 只处理 DEBUG
        self.buffer = deque(maxlen=capacity)

    def emit(self, record):
        if record.levelno == logging.DEBUG:
            try:
                msg = self.format(record)
                self.buffer.append(msg)
            except Exception:
                pass  # 防止格式化报错

    def dump_to_console(self):
        print("\n==== [最近 Debug 日志] ====")
        for msg in self.buffer:
            print(msg)

# === 设置 root logger ===
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers.clear()  # ✅ 防止重复打印（关键一步！）

# ✅ 添加 Debug 缓存 Handler（不会显示、不输出、仅内存）
debug_handler = LimitedDebugHandler(capacity=100)
debug_handler.setFormatter(logging.Formatter('%(asctime)s [DEBUG] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(debug_handler)

# ✅ 添加正常输出 Handler（只显示 INFO 及以上）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(console_handler)



# === Split-Gate 动态阈值表（过量上限）===========================
# 比例 k_es：Emitter <-> Sensor   ／  k_p： 相对 Processor/2
SPLIT_HI_ES_TABLE = { 50: 1.30, 200: 1.20, 500: 1.12, float("inf"): 1.05 }
SPLIT_HI_P_TABLE  = { 50: 1.20, 200: 1.15, 500: 1.08, float("inf"): 1.03 }

TOL_FRAC_SPLIT = 0.05      # 至少差值 Δ≥ceil(total×5 %) （且 ≥1）
# ===============================================================

def _get_hi(table, total):
    """按照总细胞数返回当前阶段的 hi 阈值"""
    for lim, val in table.items():
        if total < lim:
            return val
    return table[float("inf")]



# ── 角色分裂最低能量阈值 以及 最低调用频率 ────────────
ROLE_SPLIT_RULE = {
    "sensor":    {"min_e": 1.2, "min_calls": 0},   # 轻量，几乎不限制调用频率
    "processor": {"min_e": 1.6, "min_calls": 1},   # 中等
    "emitter":   {"min_e": 1.2, "min_calls": 2},   # 最重，门槛最高
}
# ----------------------------------------------------



class CogUnit:
    """
    CogUnit 是 EvoCore 的最小认知单元：
    - 拥有独立状态、能量、年龄
    - 可进行状态更新（update）与输出
    - 可判断是否分裂（should_split）与死亡（should_die）
    - 可克隆生成新单元（clone）
    """

    def __init__(self, input_size=50, hidden_size=16, role="processor"):
        self.is_elite = False
        self.local_memory_pool = []  # 每个单元的私有记忆池
        self.memory_pool_limit = 150  # 每个细胞记忆池最多保留 N 条

        # 基因表达，表示对不同功能的偏好
        self.gene = {
            "sensor_bias": random.uniform(0.8, 1.2),
            "processor_bias": random.uniform(0.8, 1.2),
            "emitter_bias": random.uniform(0.8, 1.2),
            "mutation_rate": 0.001  # 每次复制有0.1%概率突变
        }

        self.death_by_aging = False
        self.subsystem_id = None  # 初始没有子系统归属
        self.output_history = []  # ✅ 用于记录近几次输出，评估是否行为单一
        self.call_history = []  # 记录最近几步的调用次数
        self.call_window = 5  # 窗口长度，过去 5 步
        self.inactive_steps = 0
        self.position = (random.randint(0, 10), random.randint(0, 10))  # 可调范围
        self.state_memory = []  # 记忆队列
        self.memory_limit = 5  # 可调整为 k 步
        self.role = role
        self.id = uuid.uuid4()          # 唯一标识
        self.energy = 1.0               # 初始能量
        self.age = 0                    # 生存步数
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 认知状态向量
        self.state = torch.zeros(hidden_size)

        # 微型前馈网络（输入维度 → 隐藏维度 → 回到输入维度）
        self.function = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size)
        )

        self.last_output = torch.zeros(input_size)

        if "mutation_rate" not in self.gene:
            self.gene["mutation_rate"] = 0.05
        self.device = torch.device("cpu")  # 默认跟随 CPU

    # ---------------- 新增 ----------------
    def to(self, device):
        """把内部权重 & 状态迁移到指定设备（cpu / cuda）"""
        device = torch.device(device)
        self.device = device
        self.function.to(device)
        self.state = self.state.to(device)
        self.last_output = self.last_output.to(device)
        # 若还有其他缓存张量，也一并 .to(device)
        return self
    # -------------------------------------


    def get_position(self):
        return self.position

    def mini_learn(self, input_tensor, target_tensor, lr=0.001):
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)

        # Forward
        output = self.function(input_tensor)

        # Loss
        loss = torch.nn.functional.mse_loss(output, target_tensor)

        # Backward
        self.function.zero_grad()
        loss.backward()

        # Manual parameter update
        with torch.no_grad():
            for param in self.function.parameters():
                if param.grad is not None:
                    param.copy_(param - lr * param.grad)


        logger.debug(f"[Mini-Learn] {self.id} loss={loss.item():.4f} (lr={lr})")


    def compute_self_reward(self, input_tensor, output_tensor):
        """
        简单 self-reward：如果输出能跟输入保持一致性，就获得小奖励
        """
        if input_tensor.shape != output_tensor.shape:
            output_tensor = output_tensor[:, :input_tensor.shape[1]]  # 防止维度不同
        error = torch.mean((input_tensor - output_tensor) ** 2)
        reward = 0.01 * (self.input_size / 50) * (1.0 - error.item())  # error越小奖励越高
        return max(reward, 0.0)  # 不让奖励为负数


    def update(self, input_tensor: torch.Tensor):
        if FOLLOW_INPUT_DEVICE:
            # 若输入在 GPU，但 self.function 还在 CPU，就迁过去
            if self.function[0].weight.device != input_tensor.device:
                self.to(input_tensor.device)

        """更新 CogUnit 状态"""
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # 🚨 先检查 input_size 是否需要扩展（动态适配环境变化）
        current_input_size = input_tensor.shape[-1]
        if current_input_size != self.input_size:
            logger.info(f"[动态扩展] {self.id} 输入尺寸变化 {self.input_size} → {current_input_size}")

            # 重建 function 网络
            self.function = torch.nn.Sequential(
                torch.nn.Linear(current_input_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, current_input_size)
            )
            self.input_size = current_input_size
            self.last_output = torch.zeros(current_input_size, device=input_tensor.device)

        # === Forward: 内部处理 ===
        raw_output = self.function(input_tensor)  # 正常forward
        self.last_output = raw_output.detach().clone()  # ⚡ 关键：detach掉，避免污染计算图
        self.state = self.last_output.clone()

        # ✅ 存储输出历史，供行为质量判断用
        self.output_history.append(self.last_output.detach().clone())
        if len(self.output_history) > 5:
            self.output_history.pop(0)
        self.age += 1

        # === 外部状态记忆（用于后续奖励机制） ===
        self.state_memory.append(self.state.clone())
        if len(self.state_memory) > self.memory_limit:
            self.state_memory.pop(0)

        # ========================
        # 🚨 动态能量消耗逻辑部分
        # ========================

        # 1️⃣ 输入复杂度：使用方差作为熵的近似
        input_var = torch.var(input_tensor).item()

        # 2️⃣ 调用频率：外部由 Graph 写入 recent_calls 属性
        recent_call_freq = getattr(self, "recent_calls", 1)

        # 3️⃣ 活跃连接数：外部由 Graph 写入 connection_count 属性
        connection_count = getattr(self, "connection_count", 1)

        # ⚠️ 代谢已由 CogGraph 控制，这里不再消耗 energy

        # === 高频调用奖励机制 ===
        avg_recent_calls = getattr(self, "avg_recent_calls", 0.0)
        if avg_recent_calls >= 2.0 and self.energy > 0.0:
            self.energy += 0.01
            logger.debug(f"[奖励] {self.id} 平均调用频率 {avg_recent_calls:.2f} → 能量 +0.02")

        # === 输出扰动：模拟早期探索行为（前10步）===
        if hasattr(self, "current_step"):
            if self.get_role() == "emitter" and self.current_step < 20:
                noise = torch.randn_like(self.last_output) * 0.2
                self.last_output += noise
                logger.debug(f"[扰动] emitter {self.id} 输出加入扰动")
            elif self.get_role() == "processor" and self.current_step < 5:
                noise = torch.randn_like(self.last_output) * 0.1
                self.last_output += noise
                logger.debug(f"[扰动] processor {self.id} 输出加入扰动")

        # === ✅ 内部奖励机制 Self-Reward ===
        self_reward = self.compute_self_reward(input_tensor, self.last_output)
        self.energy += self_reward
        if self_reward > 0:
            logger.debug(f"[内部奖励] {self.id} 自评奖励 +{self_reward:.4f} 能量 (现有能量 {self.energy:.2f})")

        # === ✅ 局部微型学习
        if self.get_role() == "emitter":
            bias = self.gene.get("emitter_bias", 1.0)
            lr = 0.001 * (2.0 - min(1.5, bias))
            if ENABLE_MINI_LEARN:
                self.mini_learn(input_tensor, self.last_output.detach(), lr=lr)

        else:
            # processor/sensor 仍是自编码式
            bias = self.gene.get("processor_bias", 1.0) if self.role == "processor" else self.gene.get("sensor_bias",
                                                                                                       1.0)
            lr = 0.001 * (2.0 - min(1.5, bias))  # bias 越高，学习率越低，代表更“稳健”，越低则更易激动
            if ENABLE_MINI_LEARN:
                self.mini_learn(input_tensor, input_tensor, lr=lr)

    def get_output(self) -> torch.Tensor:
        """返回给下游单元使用的输出 (shape=[1, input_size])"""
        if MAX_OUTPUT_DIM is not None and self.last_output.numel() > MAX_OUTPUT_DIM:
            return self.last_output[:MAX_OUTPUT_DIM]
        return self.last_output


    def should_split(self):


        emitter_count = getattr(self, "global_emitter_count", 1)
        processor_count = getattr(self, "global_processor_count", 1)
        sensor_count = getattr(self, "global_sensor_count", 1)
        total = getattr(self, "global_unit_count", 1)

        role = self.get_role()

        # ✅ 各类细胞紧急增殖
        if role == "emitter" and emitter_count <= 1:
            logger.warning(f"[紧急增殖] {self.id} 是唯一 emitter，强制尝试分裂并补给")
            self.energy += 1.5  # 💡 补给能量
            return True

        if role == "processor" and processor_count <= 1:
            logger.warning(f"[紧急增殖] {self.id} 是唯一 processor，强制尝试分裂并补给")
            self.energy += 1.5
            return True

        if role == "sensor" and sensor_count <= 1:
            logger.warning(f"[紧急增殖] {self.id} 是唯一 sensor，强制尝试分裂并补给")
            self.energy += 1.5
            return True

        # ===【Split-Gate : 1 : 2 : 1 动态门槛】===========================
        total = getattr(self, "global_unit_count", sensor_count + processor_count + emitter_count)

        hi_es = _get_hi(SPLIT_HI_ES_TABLE, total)  # emitter <-> sensor
        hi_p = _get_hi(SPLIT_HI_P_TABLE, total)  # 相对 processor/2
        half_p = processor_count / 2

        # 差值必须 ≥1 且 ≥ceil(total×TOL) 才算“真的多”
        def _delta_enough(x, y):
            delta = x - y
            return delta >= max(1, int(total * TOL_FRAC_SPLIT))

        overpop = False
        if role == "emitter":
            if (_delta_enough(emitter_count, sensor_count * hi_es) or
                    _delta_enough(emitter_count, half_p * hi_p)):
                overpop = True
        elif role == "sensor":
            if (_delta_enough(sensor_count, emitter_count * hi_es) or
                    _delta_enough(sensor_count, half_p * hi_p)):
                overpop = True
        elif role == "processor":
            # processor 超标：其一半相对 e/s 也超标
            if (_delta_enough(half_p, emitter_count * hi_p) or
                    _delta_enough(half_p, sensor_count * hi_p)):
                overpop = True

        if overpop:
            return False
        # ================================================================

        # 角色专属能量 + 调用门槛 ----------------------
        rule = ROLE_SPLIT_RULE[role]
        if self.energy < rule["min_e"]:
            return False
        if role != "sensor" and self.avg_recent_calls < rule["min_calls"]:
            return False

        if len(self.output_history) >= 3:
            recent = self.output_history[-3:]
            if all(torch.equal(recent[0], o) for o in recent[1:]):
                return False

        return True

    def is_worthy_of_memory(self):
        """根据不同角色，判断该细胞是否值得加入记忆池"""
        if self.age < 100:
            return False  # 太年轻的不记

        if self.role == "sensor":
            # 感知单元应关注输入响应频率 & 能量利用效率
            return self.avg_recent_calls > 0.6

        elif self.role == "processor":
            # 处理单元应关注调用频率 & 输出多样性
            if getattr(self, "avg_recent_calls", 0) < 0.75:
                return False
            if len(self.output_history) < 1:
                return False
            # variation = sum(
            #     torch.norm(self.output_history[i] - self.output_history[i + 1]).item()
            #     for i in range(len(self.output_history) - 1)
            # ) / (len(self.output_history) - 1)
            # return variation > 0.05  # 输出变化足够丰富
            return True

        elif self.role == "emitter":
            # 行为单元应关注任务完成情况和激活频率（活跃但非重复）
            if self.avg_recent_calls < 1.0:
                return False
            if len(self.output_history) < 2:
                return False
            # diff = sum(
            #     torch.norm(self.output_history[i] - self.output_history[i + 1]).item()
            #     for i in range(len(self.output_history) - 1)
            # ) / (len(self.output_history) - 1)
            # return 0.01 < diff < 0.5  # 太低代表退化，太高可能随机扰动
            return True

        return False

    def add_to_local_memory(self):
        self.local_memory_pool = [m for m in self.local_memory_pool if "score" in m]
        # —— 对齐 output_history 到同一长度 ——
        import torch.nn.functional as F
        aligned_history = None
        if len(self.output_history) >= 3:
            # 取最大元素数量
            max_len = max(t.numel() for t in self.output_history)
            aligned_history = []
            for t in self.output_history:
                vec = t.view(-1)  # 拉平
                if vec.numel() < max_len:
                    # 右侧补 0
                    vec = F.pad(vec, (0, max_len - vec.numel()), value=0)
                else:
                    # 长则截断
                    vec = vec[:max_len]
                aligned_history.append(vec)

        if self.role == "sensor":
            # 感知：稳定性 + 被调用频率（已对齐）
            if aligned_history:
                variation = torch.var(torch.stack(aligned_history), dim=0).mean().item()
            else:
                variation = 0
            score = getattr(self, "avg_recent_calls", 0) * 0.3 + variation * 0.2


        elif self.role == "processor":
            # 处理：输出多样性 + 调用频率（已对齐）
            if aligned_history:
                diffs = [
                    torch.norm(aligned_history[i] - aligned_history[i+1]).item()
                    for i in range(len(aligned_history) - 1)
                ]
                diversity = sum(diffs) / len(diffs)
            else:
                diversity = 0
            score = diversity * 0.5 + getattr(self, "avg_recent_calls", 0) * 0.3


        elif self.role == "emitter":
            # 输出：活跃性 + 输出稳定性（已对齐）
            if aligned_history:
                diffs = [
                    torch.norm(aligned_history[i] - aligned_history[i+1]).item()
                    for i in range(len(aligned_history) - 1)
                ]
                avg_diff = sum(diffs) / len(diffs)
                stability = 1.0 if 0.01 < avg_diff < 0.5 else 0.0
            else:
                stability = 0
            score = getattr(self, "avg_recent_calls", 0) * 0.5 + stability * 0.3


        else:
            score = self.energy + getattr(self, "avg_recent_calls", 0)

        """将自身压缩为记忆格式，加入 local memory pool"""
        mem = {
            "gene": self.gene.copy(),
            "output": self.last_output.clone(),
            "role": self.role,
            "age": self.age,
            "hidden_size": self.hidden_size,
            "score": score

        }
        self.local_memory_pool.append(mem)

        # 控制最大记忆数量，移除最弱
        if len(self.local_memory_pool) > self.memory_pool_limit:
            self.local_memory_pool.sort(key=lambda m: m["score"])
            self.local_memory_pool.pop(0)  # 移除最弱
        logger.info(
            f"[记忆加入] {self.id}（{self.role}，Age={self.age}）加入本地记忆池，评分={mem['score']:.2f}，当前共 {len(self.local_memory_pool)} 条")

    def should_die(self) -> bool:

        if self.role == "emitter" and getattr(self, "global_emitter_count", 1) <= 2:
            if self.age < 600:
                return False  # 不杀唯一 emitter

        elif self.role == "processor" and getattr(self, "global_processor_count", 1) <= 4:
            if self.age < 600:
                return False


        elif self.role == "sensor" and getattr(self, "global_processor_count", 1) <= 2:
            if self.age < 600:
                return False

        if self.role == "processor":
            if self.energy <= 0.0:
                return True
            if self.age > 270:
                return True  # 绝对死亡

            if 250 <= self.age <= 270:
                death_chance = (self.age - 250) / 20  # 随年龄线性增长的死亡概率
                if random.random() < death_chance:
                    logger.info(f"[衰老死亡] {self.id} 年龄={self.age}，概率={death_chance:.2f} → 死亡")
                    self.death_by_aging = True
                    # ✅ 值得记录的细胞才加入 local memory
                    if self.is_worthy_of_memory():
                        self.add_to_local_memory()
                    return True

        if self.energy <= 0.0:
            return True

        if self.age > 270:
            return True  # 绝对死亡

        if 250 <= self.age <= 270:
            death_chance = (self.age - 250) / 20  # 随年龄线性增长的死亡概率
            if random.random() < death_chance:
                logger.info(f"[衰老死亡] {self.id} 年龄={self.age}，概率={death_chance:.2f} → 死亡")
                self.death_by_aging = True
                # ✅ 值得记录的细胞才加入 local memory
                if self.is_worthy_of_memory():
                    self.add_to_local_memory()
                return True

        # 平均调用频率太低（仅针对 emitter）
        if self.role in ["emitter"] and self.inactive_steps > 20:
            return True

        # 输出完全重复（仅针对 processor 和 emitter）
        # if self.role in ["processor", "emitter"] and getattr(self, "current_step", 0) > 600:
        #     if len(self.output_history) >= 4:
        #         diffs = []
        #         for i in range(len(self.output_history) - 1):
        #             a = self.output_history[i]
        #             b = self.output_history[i + 1]
        #             target_dim = max(a.shape[-1], b.shape[-1])
        #             if a.shape[-1] < target_dim:
        #                 padding = (0, target_dim - a.shape[-1])
        #                 a = torch.nn.functional.pad(a, padding, value=0)
        #             if b.shape[-1] < target_dim:
        #                 padding = (0, target_dim - b.shape[-1])
        #                 b = torch.nn.functional.pad(b, padding, value=0)
        #             diffs.append(torch.norm(a - b).item())
        #
        #         if max(diffs) < 0.01:
        #             logger.info(f"[退化死亡] {self.id} 输出变化极小 → 被淘汰")
        #             return True
        return False

    def clone(self, role_override=None, new_input_size=None):
        role = role_override or self.role
        input_size = new_input_size if new_input_size is not None else self.input_size

        clone_unit = CogUnit(
            input_size=input_size,
            hidden_size=self.hidden_size,
            role=role
        )

        # 🔬 基因复制（深拷贝）
        clone_unit.gene = {k: v for k, v in self.gene.items()}

        # 🌱 突变机制（小概率触发）
        if random.random() < self.gene.get("mutation_rate", 0.01):
            # hidden_size 微调 ±2（范围限制）
            delta = random.choice([-2, 2])
            clone_unit.hidden_size = max(4, min(64, self.hidden_size + delta))
            logger.info(f"[突变] hidden_size 突变为 {clone_unit.hidden_size}")

        if random.random() < self.gene.get("mutation_rate", 0.005):

            # 基因突变
            for key in ["sensor_bias", "processor_bias", "emitter_bias"]:
                mutation = random.uniform(-0.1, 0.1)
                clone_unit.gene[key] = max(0.5, min(2.0, clone_unit.gene[key] + mutation))
            logger.info(f"[突变] gene 突变为 {clone_unit.gene}")

        clone_unit.energy = self.energy * 0.6
        clone_unit.age = 0
        clone_unit.state = self.state.clone()

        if input_size != self.input_size:
            # 输入尺寸变化了，新生 last_output 用全0初始化
            clone_unit.last_output = torch.zeros(input_size)
        else:
            clone_unit.last_output = self.last_output.clone()

        self.energy *= 0.4
        # ✅ 继承局部记忆池（只保留最新的 75 条）
        clone_unit.local_memory_pool = [m for m in self.local_memory_pool if "score" in m][-75:]

        # --------------------
        # ⚡ 将子细胞迁移到与母体相同的 device
        clone_unit.to(self.device)
        # --------------------

        # 🎯 改为融合 local memory（局部记忆池）
        if hasattr(self, "local_memory_pool") and len(self.local_memory_pool) >= 1:
            # 可选：更智能挑选最近最活跃的记忆
            memory = random.choice(self.local_memory_pool[-5:])  # 可换成 max(..., key=...)

            for key in ["sensor_bias", "processor_bias", "emitter_bias"]:
                g1 = self.gene.get(key, 1.0)
                g2 = memory["gene"].get(key, 1.0)
                clone_unit.gene[key] = 0.7 * g1 + 0.3 * g2
            logger.debug(f"[记忆融合] {self.id} 结合 local memory 基因 → 子基因：{clone_unit.gene}")

            if self.last_output is None or memory.get("output") is None:
                return clone_unit  # 跳过融合逻辑
            if "output" in memory:
                o1 = self.last_output.squeeze(0) if self.last_output.dim() == 2 else self.last_output
                o2 = memory["output"].squeeze(0) if memory["output"].dim() == 2 else memory["output"]
                target_dim = max(o1.shape[0], o2.shape[0])
                if o1.shape[0] < target_dim:
                    o1 = torch.nn.functional.pad(o1, (0, target_dim - o1.shape[0]), value=0)
                if o2.shape[0] < target_dim:
                    o2 = torch.nn.functional.pad(o2, (0, target_dim - o2.shape[0]), value=0)
                clone_unit.last_output = 0.6 * o1 + 0.4 * o2
                logger.debug(f"[行为融合] 结合输出 → 前5维: {clone_unit.last_output[:5]}")

            if random.random() < self.gene.get("mutation_rate", 0.01) * 2:
                if "hidden_size" in memory:
                    h1 = self.hidden_size
                    h2 = memory.get("hidden_size", h1)
                    new_hidden = int(0.7 * h1 + 0.3 * h2)
                    new_hidden = max(4, min(128, new_hidden))
                    clone_unit.hidden_size = new_hidden
                    clone_unit.function = torch.nn.Sequential(
                        torch.nn.Linear(clone_unit.input_size, new_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(new_hidden, clone_unit.input_size)
                    )
                    clone_unit.gene["hidden_size_tag"] = new_hidden
                    logger.debug(f"[网络融合] hidden_size 融合为 {new_hidden}")
        return clone_unit

    def get_role(self):
        return self.role

    def __str__(self):
        x, y = self.position
        return f"CogUnit<{self.id}> Role:{self.role} Pos:({x},{y}) Age:{self.age} Energy:{self.energy:.2f} Gene:{self.gene}"



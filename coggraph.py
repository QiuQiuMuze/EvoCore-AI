# coggraph.py
import uuid
from CogUnit import CogUnit
import torch
import random
from env import GridEnvironment
import torch.nn.functional as F
import gc, torch
from collections import deque, Counter
from env import logger
from typing import List, Dict
from agents.rl_agent import RLAgent
import copy
# from triton_scatter import scatter_sum
from config_runtime import RF            # ★ 新增
from contextlib import nullcontext       # ★ autocast fallback
try:                                     # Flash-Attn / TE 优先
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
import math
from goal_generator import sample_unvisited, make_onehot
from collections import Counter
from self_model import build_self_model
from energy_policy import EnergyPolicy


HIT_THRESH = 0.15          # 越大越宽松
MAX_CONNECTIONS = 4  # 每个单元最多连接 4 个下游
N_STATE_CHANNELS = 5
N_GOAL_CHANNELS = 3
INPUT_CHANNELS = N_STATE_CHANNELS + N_GOAL_CHANNELS





def _percentile(values, q):
    """轻量级百分位计算，避免依赖 numpy。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    lower = sorted_vals[f]
    upper = sorted_vals[c]
    return float(lower * (c - k) + upper * (k - f))


class TaskInjector:
    def __init__(self, target_position):
        self.target_position = target_position  # 目标坐标 (x, y)

    def encode_goal(self, env_size):
        """将目标位置编码成 one-hot 向量（与输入同维度）"""
        index = self.target_position[1] * env_size + self.target_position[0]
        vec = torch.zeros(2, env_size * env_size)  # 2 通道
        vec[0, index] = 1.0  # 资源层 one-hot
        # vec[1] 先保持全 0（陷阱层以后再写）
        return vec

    def evaluate(self, env, emitter_outputs):
        if emitter_outputs is None:
            return False
        pred_index = torch.argmax(emitter_outputs.mean(dim=0)).item()
        x, y = pred_index % env.size, pred_index // env.size
        return (x, y) in env.resources  # ✅ 只看是否落在资源点上


# 理想比例  emitter : processor : sensor = 1 : 2 : 1
IDEAL_RATIO = {"emitter": 1, "processor": 2, "sensor": 1}
DENOM = sum(IDEAL_RATIO.values())      # =4

# 每轮允许转换的最高比例（60%）
MAX_CONV_FRAC = 0.6

# Δ 容差系数：需要至少 diff ≥ ceil(TOL_FRAC*total) 才触发
TOL_FRAC = 0.05      # 小规模时自动退化成 1


# ------------------------------------------------------------
def _build_transformer_block(D, H, device):
    """
    返回一个 *单层* Transformer（优先 TE / Flash-Attn，CPU 退回官方实现）。
    单独列出来方便 _init_shared_tx() 循环复用。
    """
    if HAS_TE and torch.cuda.is_available():
        import transformer_engine.pytorch as te
        return te.TransformerLayer(
                hidden_size=D,
                num_attention_heads=H,
                mlp_hidden_size=4*D,
                dropout=0.0,
                sequence_parallel=False  # 单卡
        ).to(device)
    else:
        layer = torch.nn.TransformerEncoderLayer(
                d_model=D, nhead=H, dim_feedforward=4*D,
                activation="gelu", batch_first=True)
        return layer.to(device)


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
                         n_sensor: int = 16,
                         n_processor: int = 32,
                         n_emitter: int = 16,
                         device: str = "cpu"):

        expected_input = self.env_size * self.env_size * INPUT_CHANNELS

        # 1) 创建
        sensors = [CogUnit(input_size=expected_input, role="sensor", env_size=self.env_size) for _ in range(n_sensor)]
        processors = [CogUnit(input_size=expected_input, role="processor", env_size=self.env_size) for _ in range(n_processor)]
        emitters = [CogUnit(input_size=expected_input, role="emitter", env_size=self.env_size) for _ in range(n_emitter)]

        # 2) 迁移到目标 device
        for u in sensors + processors + emitters:
            u.to(device)

        # 3) 加入图
        for u in sensors + processors + emitters:
            self.add_unit(u)

        # --- 初始化内在目标状态 ---
        # 保证初始 personal_goal 在不同 emitter 之间不重复
        taken = set()
        for e in emitters:
            e.visit_counts = Counter()
            # 循环采样，直到拿到一个还没被占用的点
            while True:
                cand = sample_unvisited(self.env_size, e.visit_counts)
                if cand not in taken:
                    break
            e.personal_goal = cand
            e.visit_counts[cand] = 0
            taken.add(cand)

            e.intrinsic_cooldown = 20  # 冷却 20 步
            e._last_intrinsic_step = -1e9  # 初始化为极早之前

        # 4) 连接：sensor→processor（保持全连），processor→emitter 随机连几条
        for s in sensors:
            for p in processors:
                self.connect(s, p)

        # 每个 emitter 只挑几个 processor 做上游

        for e in emitters:
            # 如果处理器太少，就连所有；否则随机取 3 条
            ups = processors if len(processors) <= 3 else random.sample(processors, 3)
            for p in ups:
                self.connect(p, e)


    # -------------------------------------------------------------------
    def __init__(self, rl_agent: RLAgent, device: str = "cpu", env: GridEnvironment | None = None):
        # 保存传入的 RLAgent 实例
        self.rl_agent = rl_agent
        self.device = torch.device(device)
        # ==== 环境注入 ====
        if env is not None:
            self.env = env
            self.env_size = env.size
        else:
            self.env_size = 5
            self.env = GridEnvironment(size=self.env_size)

        self._pending_deaths: List[CogUnit] = []
        self._death_energy_sum: Dict[str, float] = {}  # role -> total energy
        self.debug = False
        self.reverse_connections = {}  # to_id -> set(from_ids)
        self.static_mode_exit_step = -9999  # 初始化为很早以前
        self.sensor_count = 0
        self.processor_count = 0
        self.emitter_count = 0
        self.energy_pool = 0.0  # 中央能量池
        # self.memory_pool = []  # 存放死亡细胞的 gene + last_output + bias info
        initial_target = (random.randint(0, self.env_size - 1), random.randint(0, self.env_size - 1))
        self.task = TaskInjector(target_position=initial_target)
        self.target_vector = self.task.encode_goal(self.env_size).to(self.device)
        self.target_vector = torch.zeros((2, self.env_size * self.env_size), device=self.device)  # 初始化为空目标
        self.max_total_energy = 500  # 初始最大总能量
        self.removed_hazards_by_reward = 0
        self.connection_usage = {}  # {(from_id, to_id): last_used_step}
        self.current_step = 0
        self.units = []
        self.removed_resources_count = 0
        self.removed_hazards_count = 0
        self.active_units = set()
        self.energy_policy = EnergyPolicy()
        self.connections = {}  # {from_id: {to_id: strength_float}}
        self.unit_map = {}     # {unit_id: CogUnit 实例} 快速索引单元
        self.processor_hidden_size = self.env_size * self.env_size * INPUT_CHANNELS
        self._target_buf = torch.zeros((2, self.env_size * self.env_size),
                                       device=self.device)
        self.steps_since_last_reward = 0
        self.static_mode = False
        self.static_mode_entry_step = None
        self.static_mode_max_duration = 200
        self._orig_metabolic = {}   # 存储进入静吸模式前的速率
        self.static_mode_allowed = False
        self._static_mode_forbid_step = 0
        # —— 缓存最近一次各角色的原始输出，供拓扑加权聚合使用 ——
        self._last_sensor_outputs: Dict[uuid.UUID, torch.Tensor] = {}
        self._last_processor_outputs: Dict[uuid.UUID, torch.Tensor] = {}
        # —— 存储最近一次聚合时的拓扑统计，便于调试与自省 ——
        self._role_topology_snapshots: Dict[str, Dict[str, Dict[str, float]]] = {
            "sensor": {},
            "processor": {},
            "emitter": {},
        }
        # —— 追踪各角色的代谢趋势（指数滑动平均） ——
        self._metabolic_ema_beta = 0.85
        self._role_metabolic_stats: Dict[str, Dict[str, float]] = {
            role: self._make_metabolic_bucket() for role in ("sensor", "processor", "emitter")
        }
        self._role_feature_dim_ema: Dict[str, float] = {
            "sensor": 0.0,
            "processor": 0.0,
            "emitter": 0.0,
        }
        self._feature_dim_ema_beta = 0.9
        self._emitter_input_rms_cap = 1.25
        self._emitter_input_value_cap = 3.5
        self._emitter_metabolic_floor = 0.06
        self._emitter_metabolic_ceiling = 0.18
        self._emitter_refuel_share_cap = 0.28
        # --- 在 __init__() 的最后调用 ---
        self._init_seed_units(device=device)

        # ---------- 🆕 把初始化代码折到一个函数里 ----------
        if RF.use_shared_tx:
            self._init_shared_tx()  # 第一次建好共享 Tx

    def _mark_connection_used(self, from_id: uuid.UUID | str, to_id: uuid.UUID | str):
        """记录连接在当前步被使用。"""
        self.connection_usage[(from_id, to_id)] = self.current_step

    def _clear_connection_usage(self, from_id: uuid.UUID | str, to_id: uuid.UUID | str):
        """安全移除某条连接的使用记录。"""
        self.connection_usage.pop((from_id, to_id), None)

    # ============================================================
    #                   Shared-Tx 初始化封装
    # ============================================================
    def _init_shared_tx(self):
        """
        根据 **当前** self.processor_hidden_size 重新创建
        shared_encoder / role_embed / id_embed。
        在   1) __init__   2) env 扩容后   调一次即可。
        """
        D = self.processor_hidden_size
        H = math.gcd(D, RF.shared_tx_heads) or 1
        if H != RF.shared_tx_heads:
            logger.warning(f"[Shared-Tx] embed_dim={D} 不整除 {RF.shared_tx_heads}；自动改为 {H} 头")

        L = RF.shared_tx_layers

        blocks = [_build_transformer_block(D, H, self.device) for _ in range(L)]
        # TE 分支已是单层，官方分支需要包装成 Encoder
        if isinstance(blocks[0], torch.nn.TransformerEncoderLayer):
            self.shared_encoder = torch.nn.TransformerEncoder(
                blocks[0], num_layers=L).to(self.device).eval()
        else:  # TE
            self.shared_encoder = torch.nn.Sequential(*blocks).to(self.device).eval()

        # 重新生成 embedding，长度=新 D
        self.role2id = {"sensor": 0, "processor": 1, "emitter": 2}
        self.role_embed = torch.nn.Embedding(3, D).to(self.device)
        self.id_embed = torch.nn.Embedding(1_000_000, D).to(self.device)
        # —— 新增：把 6 维 self_model 投影到 D 维 —— #
        self.self_model_dim  = 6
        self.self_model_proj = torch.nn.Linear(self.self_model_dim, D).to(self.device)


        if RF.use_compile and torch.cuda.is_available():
            # ⚠️ 千万别再写 “import torch._dynamo” —— 那会把 torch 当作局部变量
            from torch import _dynamo as torch_dynamo      # 只绑定 torch_dynamo，不碰 torch
            torch_dynamo.config.suppress_errors = True

            if RF.use_compile and torch.cuda.is_available():
                try:
                    self.shared_encoder = torch.compile(self.shared_encoder, mode=RF.compile_mode, fullgraph=False)
                    logger.info("[Compile] shared_encoder 已 JIT 编译")
                except Exception as e:
                    logger.warning(f"[Compile] shared_encoder 编译失败：{e}")

            def _compile(m):
                if getattr(m, "_compiled", False): return m
                m._compiled = torch.compile(m, mode=RF.compile_mode, fullgraph=False)
                return m._compiled

            for u in self.units:
                u.function = _compile(u.function)

    @torch.no_grad()
    def _run_shared_transformer(self):
        """
        把所有 unit.state → [1,N,D] tokens，
        role / id 做 embedding，加一次 MHA，
        再写回各自 unit.state。只做前向，不反传梯度。
        """
        if not RF.use_shared_tx or len(self.units) == 0:
            return

        toks = []
        for u in self.units:

            role_id = self.role2id.get(u.role, 0)
            # UUID → int → 0‥999 999
            uid_idx = u.int_id % 1_000_000
            # 1) 原始 state 向量
            vec = u.state.view(-1)

            # 2) pad / truncate 到 D
            if vec.numel() < self.processor_hidden_size:
                vec = F.pad(vec, (0, self.processor_hidden_size - vec.numel()))
            elif vec.numel() > self.processor_hidden_size:
                vec = vec[: self.processor_hidden_size]

            # 3) 自我模型嵌入 （6→D）并相加
            sm = build_self_model(u)               # [6]
            sm_emb = self.self_model_proj(sm)      # [D]
            vec = vec + sm_emb

            # 4) 加上 role/id embedding
            tok = vec \
                + self.role_embed.weight[role_id] \
                + self.id_embed.weight[uid_idx]


            toks.append(tok)

        tokens = torch.stack(toks, 0).unsqueeze(0).to(self.device)   # [1,N,D]

        ctx = (torch.autocast("cuda", dtype=torch.float16)
               if (RF.use_fp16 and self.device.type == "cuda") else nullcontext())
        with ctx:
            out = self.shared_encoder(tokens)                        # [1,N,D]

        for i, u in enumerate(self.units):
            u.state = out[0, i].detach()     # 写回


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
        logger.warning(f"[统计] step={self.current_step} | sensor:{s}, processor:{p}, emitter:{e}")

        # 再把所有连接强度 dump 一遍
        logger.debug("[连接强度]")
        for frm, to_dict in self.connections.items():
            for to, strg in to_dict.items():
                logger.debug(f"  {frm} → {to} = {strg:.3f}")

    def add_unit(self, unit: CogUnit):
        # --- 若图中已有单元，则让新单元跟随它们的 device ---
        # if self.units:
        #     target_device = self.units[0].device
        #     if unit.device != target_device:
        #         unit.to(target_device)
        # -----------------------------------------------
        # 全局统一：始终迁移到 graph.device（在 __init__ 设定）,启用gpu的时候使用
        if unit.device != self.device:
            unit.to(self.device)
        unit.graph = self
        if hasattr(unit, "attach_energy_policy"):
            unit.attach_energy_policy(self.energy_policy)
        # 将单元加入图结构中
        self.units.append(unit)
        self.unit_map[unit.id] = unit
        self.connections[unit.id] = {}
        # # --- ❶ 维护 id⇄index ---
        # self.id2idx[unit.id] = len(self.idx2id)  # 顺序追加
        # self.idx2id.append(unit.id)
        # self.edge_dirty = True

        self._update_global_counts()

    def _get_min_target_counts(self):
        """
        根据当前 max_total_energy 和角色比例，返回每类角色的最小建议数量。
        只有低于各自阈值的角色才会触发强制分裂。
        """
        # 系统希望的最小总细胞数（原逻辑）
        total_target = int(self.max_total_energy / 2.5 * 0.8)

        # 理想比例：sensor:processor:emitter = 1:2:1
        IDEAL_RATIO = {"sensor": 1, "processor": 2, "emitter": 1}
        DENOM = sum(IDEAL_RATIO.values())  # = 4

        target_counts = {}
        for role, weight in IDEAL_RATIO.items():
            # 按比例分配，并且至少保留 1 个
            cnt = math.ceil(total_target * weight / DENOM)
            target_counts[role] = max(1, cnt)

        return target_counts
    def finalize_deaths(self):
        # 遍历每个 role，分配它们累加的能量
        for role, total_energy in self._death_energy_sum.items():
            heirs = [
                u for u in self.units
                if u.role == role and u.age < 240
            ]
            if heirs and total_energy > 0.0:
                per_gain = total_energy / len(heirs)
                for u in heirs:
                    u.energy += per_gain
                logger.info(
                    f"[寿终能量继承] 本步共 {len(self._pending_deaths)} 个{role}死亡，"
                    f"合计能量 {total_energy:.2f} → 平分给 {len(heirs)} 个后辈，每人 +{per_gain:.2f}"
                )

        # 清空，准备下一步
        self._pending_deaths.clear()
        self._death_energy_sum.clear()

    def remove_unit(self, unit: CogUnit):


        if unit.id not in self.unit_map:
            return  # 已经被删除

        # ✅ 遗产机制：寿终正寝时，能量分配给年轻后辈
        # —— 收集“寿终”死亡单元能量 ——
        if getattr(unit, "death_by_aging", False) and unit.energy > 0.0:
            role = unit.role
            self._pending_deaths.append(unit)
            self._death_energy_sum[role] = (
                    self._death_energy_sum.get(role, 0.0) + unit.energy
            )

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
                    if len(other.local_memory_pool) > other.memory_pool_limit:
                        other.local_memory_pool.sort(key=lambda m: m["score"])
                        other.local_memory_pool.pop(0)

        # 从图中移除单元及其连接
        self.units = [u for u in self.units if u.id != unit.id]
        if unit.id in self.connections:
            for to_id in list(self.connections[unit.id].keys()):
                self._clear_connection_usage(unit.id, to_id)
            del self.connections[unit.id]
        if unit.id in self.unit_map:
            del self.unit_map[unit.id]
        for k in self.connections:
            if unit.id in self.connections[k]:
                del self.connections[k][unit.id]
                self._clear_connection_usage(k, unit.id)
                self.reverse_connections.get(unit.id, set()).discard(k)
        self._update_global_counts()

        # 把这个被删单元当成“from”的所有反向索引都清理掉
        for to_id, from_set in self.reverse_connections.items():
            if unit.id in from_set:
                from_set.discard(unit.id)
        # 然后再把自己那条 key 删掉
        self.reverse_connections.pop(unit.id, None)
        # self.edge_dirty = True  # ❷ 告诉后面“邻接表需要重建”

    def connect(self, from_unit: CogUnit, to_unit: CogUnit):
        # 仅允许合法结构连接
        valid_links = {
            "sensor": ["processor"],
            "processor": ["emitter"],
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
            self._clear_connection_usage(from_unit.id, weakest_id)

            logger.debug(f"[连接替换] {from_unit.id} 移除最弱连接 {weakest_id}")

        # 建立新连接，初始权重为 1.0
        self.connections[from_unit.id][to_unit.id] = 1.0
        strength = self.connections[from_unit.id][to_unit.id]
        logger.debug(f"[连接建立] {from_unit.id} → {to_unit.id} (strength={strength:.2f})")
        # 同步维护反向索引
        self.reverse_connections.setdefault(to_unit.id, set()).add(from_unit.id)
        self.edge_dirty = True
        self._mark_connection_used(from_unit.id, to_unit.id)

    def total_energy(self):
        return sum(unit.energy for unit in self.units if unit.age < 240)

    # ========== 维度适配辅助 ==========
    def _goal_dim(self) -> int:
        """返回当前目标向量长度 (= env_size²)"""
        return self.env_size * self.env_size

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


    def merge_redundant_units(self, max_merge_cells=100):
        merged_pairs = set()
        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"

        new_units = []

        # —— STEP1：构建空间哈希（按位置格子划分） ——
        grid = {}  # (gx,gy) -> [unit,...]
        cell_size = 4  # 半径 3 内的邻居一定在同格或相邻格
        for u in self.units:
            gx, gy = u.position[0] // cell_size, u.position[1] // cell_size
            grid.setdefault((gx, gy), []).append(u)

        # ✅ 限制参与者：根据能量+活跃度挑 top-N
        scored = [(u, u.energy + getattr(u, "avg_recent_calls", 0)) for u in self.units]
        scored.sort(key=lambda x: x[1], reverse=True)
        limited_units = [u for u, _ in scored[:max_merge_cells]]
        # —— STEP1.5：提前缓存所有输出并 pad 到相同维度 ——
        target_dim = self.env_size * self.env_size
        output_cache = {}

        for u in limited_units:
            out = u.get_output().squeeze(0)
            if out.shape[0] < target_dim:
                out = F.pad(out, (0, target_dim - out.shape[0]))
            elif out.shape[0] > target_dim:
                out = out[:target_dim]
            out = out.to(self.device, non_blocking=True)
            output_cache[u.id] = out

        if use_cuda:
            unit_ids = [u.id for u in limited_units]
            output_tensor = torch.stack([output_cache[uid] for uid in unit_ids], dim=0)  # [N,D]
            output_tensor = F.normalize(output_tensor, dim=1)  # 单位化以计算 cosine
            similarity_matrix = output_tensor @ output_tensor.T  # [N,N] cosine sim 矩阵

        for u1 in limited_units:
            gx, gy = u1.position[0] // cell_size, u1.position[1] // cell_size
            # 只拿本格和 8 个邻格里的单元做两两
            neighbor_cells = [
                grid.get((gx + dx, gy + dy), [])
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            ]
            neighbors = set().union(*neighbor_cells)  # 去重
            for u2 in neighbors:
                if u2 not in limited_units:
                    continue

                if u2 is u1:
                    continue

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
                output1 = output_cache[u1.id]
                output2 = output_cache[u2.id]

                if use_cuda:
                    idx1 = unit_ids.index(u1.id)
                    idx2 = unit_ids.index(u2.id)
                    sim = similarity_matrix[idx1, idx2].item()
                else:
                    sim = F.cosine_similarity(output1, output2, dim=0).item()

                if sim < 0.95:
                    continue

                # ✅ 满足条件，执行合并
                logger.info(f"[合并触发] {u1.id} 和 {u2.id} 合并为新单元")

                # —— 1) 先计算当前期望输入维度
                expected_input = self.env_size * self.env_size * INPUT_CHANNELS
                # —— 2) 用父体的 hidden_size 保持隐层容量，并继承权重
                merged = CogUnit(
                    input_size=expected_input,
                    hidden_size=u1.hidden_size,
                    role=u1.get_role(),
                    env_size=self.env_size
                )
                # 深拷贝 u1 的 network 权重到 merged
                import copy
                merged.function = copy.deepcopy(u1.function)

                # —— 3) 下面所有融合逻辑都作用在这个 merged 上
                merged.position = (
                    (u1.get_position()[0] + u2.get_position()[0]) // 2,
                    (u1.get_position()[1] + u2.get_position()[1]) // 2
                )
                merged.state = (u1.state + u2.state) / 2
                merged.age = int((u1.age + u2.age) / 2)
                merged.energy = u1.energy + u2.energy + 0.02  # 奖励合并能量
                # merged.meta.record(action="Combine", reward=+0.02)
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
        # —— STEP1：先只挑最近活跃 top-K 对子图 ——
        # 记录每条 processor→emitter 边的 last call step
        calls = sorted(self.connection_usage.items(), key=lambda x: x[1], reverse=True)
        # 只保留调用最频繁的前 K 条
        topk = min(len(calls), 100)
        candidates = [(self.unit_map[p], self.unit_map[e]) for ((p, e), _) in calls[:topk]
                      if p in self.unit_map and e in self.unit_map]

        limit = len(candidates)

        # 再在这 top-K 里两重比较
        for i in range(limit):
            for j in range(i + 1, limit):

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
                    # 创建新单元（带上正确维度）
                    expected_input = self.env_size * self.env_size * INPUT_CHANNELS
                    new_p = CogUnit(
                        input_size=expected_input,
                        hidden_size=p1.hidden_size,  # 或融合 p1,p2 的 hidden_size
                        role="processor",
                        env_size=self.env_size
                    )
                    new_e = CogUnit(
                        input_size=expected_input,
                        hidden_size=e1.hidden_size,  # 或融合 e1,e2 的 hidden_size
                        role="emitter",
                        env_size=self.env_size
                    )
                    # 如果要继承父权重：
                    new_p.function = copy.deepcopy(p1.function)
                    new_e.function = copy.deepcopy(e1.function)

                    new_p.state = (p1.state + p2.state) / 2
                    new_p.last_output = (p1.last_output + p2.last_output) / 2
                    new_e.state = (e1.state + e2.state) / 2
                    new_e.last_output = (e1.last_output + e2.last_output) / 2

                    new_p.energy = p1.energy + p2.energy + 0.05
                    # new_p.meta.record(action="Combine", reward=+0.05)
                    new_e.energy = e1.energy + e2.energy + 0.05
                    # new_e.meta.record(action="Combine", reward=+0.05)

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

    def assign_subsystems(self, min_size=3, max_size=20):
        """
        自动发现局部高密度连接区域，标记为子系统
        """
        if self.current_step > 0 and self.current_step % 300 == 0:
            visited = set()
            subsystem_count = 0

            for unit in self.units:
                if unit.id in visited:
                    continue

                # 以当前unit为起点，做局部DFS
                cluster = self._dfs_collect_cluster(unit, max_depth=4)

                if min_size <= len(cluster) <= max_size:
                    subsystem_id = f"subsys-{subsystem_count}"
                    for u in cluster:
                        u.subsystem_id = subsystem_id
                    subsystem_count += 1
                    logger.info(f"[子系统生成] 新子系统 {subsystem_id}，包含 {len(cluster)} 个单元")
                    visited.update(u.id for u in cluster)

    def _dfs_collect_cluster(self, start_unit, max_depth=4):
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
        # --- 清理失效反向索引（O(total_edges)) ---
        if self.current_step > 0 and self.current_step % 100 == 0:
            for to_id, from_set in list(self.reverse_connections.items()):
                for frm in list(from_set):
                    if frm not in self.unit_map or \
                            to_id not in self.connections.get(frm, {}):
                        from_set.discard(frm)
            # ✅ 清除连接记录中，指向已不存在或失效连接的条目
            self.connection_usage = {
                k: v for k, v in self.connection_usage.items()
                if k[0] in self.unit_map and k[1] in self.unit_map and
                   k[1] in self.connections.get(k[0], {})
            }

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
                    self.reverse_connections.get(to_unit, set()).discard(from_unit)
                logger.debug(f"[剪枝] 连接 {from_unit} → {to_unit} 被剪掉")
                self._clear_connection_usage(from_unit, to_unit)

            # 强化高效连接（可选：比如增加能量传递权重等）
            for conn in to_strengthen:
                from_unit, to_unit = conn
                if from_unit in self.connections and to_unit in self.connections[from_unit]:
                    self.connections[from_unit][to_unit] *= 1.1  # 每次乘以 1.1
                    self.connections[from_unit][to_unit] = min(self.connections[from_unit][to_unit], 3.0)  # 上限 cap
                    logger.debug(
                        f"[强化] 连接 {from_unit} → {to_unit} 权重提升为 {self.connections[from_unit][to_unit]:.2f}")

            logger.info(f"[剪枝] 剪掉 {len(to_prune)} 条弱连接，强化 {len(to_strengthen)} 条强连接")

    def auto_connect(self):
        # 退火：单元越多，触发间隔越长，避免 O(N²) 每步扫描
        if self.current_step % max(60, len(self.units)//100) != 0:
            return

        def euclidean(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        # 只对最近 step 内更新过且能量高的前 M 个单元做新连尝试
        # 优先使用活跃单元缓存（如为空则 fallback 全局）
        active_units = list(self.active_units) if self.active_units else self.units
        scored = [(u, u.energy + u.avg_recent_calls) for u in active_units]
        scored.sort(key=lambda x: x[1], reverse=True)
        hot_units = [u for u, _ in scored[: min(len(scored), 50)]]
        for unit in hot_units:
            if unit.id not in self.connections:
                self.connections[unit.id] = {}  # ✅ 加这个防御性初始化

            role = unit.get_role()

            if role == "processor":
                # processor 寻找下游连接对象（processor 或 emitter）
                target_roles = [ "emitter"]
            elif role == "sensor":
                target_roles = ["processor"]
            else:
                continue  # sensor 不参与

            current_connections = self.connections[unit.id]

            if len(current_connections) < MAX_CONNECTIONS:
                u_pos = unit.get_position()

                # 前半数连接使用“近邻优先”
                if len(current_connections) < MAX_CONNECTIONS / 2:
                    candidates = [
                        u for u in self.units
                        if u.id != unit.id
                           and u.get_role() in target_roles
                           and abs(u.input_size - unit.input_size) <= 100
                           and u.id not in current_connections
                           and euclidean(u.get_position(), u_pos) < 3
                    ]
                    if not candidates:
                        # 若附近无候选，则全局搜索
                        candidates = [
                            u for u in self.units
                            if u.id != unit.id
                               and u.get_role() in target_roles
                               and u.id not in current_connections
                        ]
                else:
                    # 后半数连接使用“远程优先”
                    candidates = [
                        u for u in self.units
                        if u.id != unit.id
                           and u.get_role() in target_roles
                           and abs(u.input_size - unit.input_size) <= 100
                           and u.id not in current_connections
                           and euclidean(u.get_position(), u_pos) >= 3
                    ]

                if candidates:
                    def connection_strength(u):
                        incoming_count = sum(u.id in self.connections.get(fid, {}) for fid in self.unit_map)
                        return u.energy + incoming_count * 0.1

                    candidates.sort(key=connection_strength, reverse=True)

                    for target in candidates:
                        if target.id not in self.connections.get(unit.id, {}):
                            prev_conn_count = len(self.connections[unit.id])
                            self.connect(unit, target)
                            if len(self.connections[unit.id]) > prev_conn_count:
                                logger.debug(f"[新连接] {unit.id} → {target.id}")
                                break  # ✅ 成功建立连接就跳出

        # === 随机突变连接（只允许 processor 发起） ===
        if random.random() < 0.1:
            from_candidates = [u for u in self.units if u.get_role() == "processor"]
            to_candidates = [u for u in self.units if u.get_role() in ["emitter"]]

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
            hi, lo = 1.20, 0.80
        elif total < 200:
            hi, lo = 1.12, 0.88
        elif total < 500:
            hi, lo = 1.08, 0.92
        else:
            hi, lo = 1.05, 0.95

        # Δ 容差（至少相差 Δ_cell 才算“真的多／少”）
        delta_cell = max(1, int(total * TOL_FRAC))

        # 本轮最多转换
        max_conv = max(1, int(total * MAX_CONV_FRAC))
        conv_done = 0

        def pick_weakest(units):
            # ✅ 优先选择年龄在 5-30 的弱细胞（非永生），否则全局选弱者
            young_candidates = [u for u in self.units if u.get_role() == giver_role and 5 <= u.age <= 30 and not getattr(u, "is_elite", False)]
            other_candidates = [u for u in units if u not in young_candidates and not getattr(u, "is_elite", False)]

            def sort_key(u):
                return (u.energy, getattr(u, "avg_recent_calls", 0.0))

            if young_candidates:
                return min(young_candidates, key=sort_key)
            elif other_candidates:
                return min(other_candidates, key=sort_key)
            else:
                return None  # 没有可用细胞，返回 None
        # def pick_weakest(units):
        #     return min(units, key=lambda u: (u.energy, getattr(u, "avg_recent_calls", 0.0)))

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
            unit.energy += 0
            unit.gene[f"{receiver_role}_bias"] = 1.0
            logger.info(f"[平衡] {old}→{receiver_role} | step={self.current_step}")

            # 清旧连 & 简易新连
            for uid, out_edges in list(self.connections.items()):
                if unit.id in out_edges:
                    out_edges.pop(unit.id, None)
                    self._clear_connection_usage(uid, unit.id)
            for to_id in list(self.connections.get(unit.id, {}).keys()):
                self._clear_connection_usage(unit.id, to_id)
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


    def expand_unit_dim(self, unit: CogUnit, new_input_size: int):
        """仅将 *一个* unit 升维到 new_input_size（只升不降）"""
        import torch, gc
        if unit.input_size >= new_input_size:
            return  # 安全回退

        logger.info(f"[懒升维] {unit.id}: {unit.input_size} → {new_input_size}")

        # === 以下逻辑直接复制自旧 upscale_old_units，注意把 `unit` 循环去掉 ===
        old_out = unit.last_output
        if old_out.dim() == 2 and old_out.shape[0] == 1:
            old_out = old_out.squeeze(0)
        new_out = torch.zeros(new_input_size, device=old_out.device)
        new_out[: old_out.shape[0]] = old_out
        unit.last_output = new_out

        new_history = []
        for out in unit.output_history:
            v = out.squeeze(0) if out.dim() == 2 else out
            p = torch.zeros(new_input_size, device=v.device)
            p[: v.shape[0]] = v
            new_history.append(p.unsqueeze(0))
        unit.output_history = new_history

        old_state = unit.state.squeeze(0) if unit.state.dim() == 2 else unit.state
        new_state = torch.zeros(new_input_size, device=old_state.device)
        new_state[: old_state.shape[0]] = old_state
        unit.state = new_state

        new_mem = []
        for mem in unit.state_memory:
            p = torch.zeros(new_input_size, device=mem.device)
            p[: mem.shape[-1]] = mem
            new_mem.append(p)
        unit.state_memory = new_mem

        old_l1, old_l2 = unit.function[0], unit.function[2]
        w1, b1 = old_l1.weight.data.clone(), old_l1.bias.data.clone()
        w2, b2 = old_l2.weight.data.clone(), old_l2.bias.data.clone()

        h = unit.hidden_size
        new_l1 = torch.nn.Linear(new_input_size, h, device=w1.device)
        new_l2 = torch.nn.Linear(h, new_input_size, device=w1.device)
        new_func = torch.nn.Sequential(new_l1, torch.nn.ReLU(), new_l2)
        with torch.no_grad():
            new_l1.weight[:, : w1.shape[1]].copy_(w1)
            new_l1.bias.copy_(b1)
            new_l2.weight[: w2.shape[0], : w2.shape[1]].copy_(w2)
            new_l2.bias[: b2.shape[0]].copy_(b2)
        unit.function = new_func
        unit.input_size = new_input_size

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def upscale_old_units(self, new_input_size):
        """将所有 input_size < new_input_size 的单元升维（只升不降）。"""
        import torch

        for unit in self.units:
            if unit.input_size >= new_input_size:
                continue

            logger.info(f"[升维] {unit.id} input_size {unit.input_size} → {new_input_size}")

            # —— 1. 升维 last_output ——
            old_out = unit.last_output
            if old_out.dim() == 2 and old_out.shape[0] == 1:
                old_out = old_out.squeeze(0)
            new_out = torch.zeros(new_input_size, device=old_out.device)
            new_out[: old_out.shape[0]] = old_out
            unit.last_output = new_out

            # —— 2. 升维 output_history ——
            new_history = []
            for out in unit.output_history:
                v = out.squeeze(0) if out.dim() == 2 else out
                p = torch.zeros(new_input_size, device=v.device)
                p[: v.shape[0]] = v
                new_history.append(p.unsqueeze(0))
            unit.output_history = new_history

            # —— 3. 升维 state ——
            # 注意：这里 state 的维度一直等同于 input_size，所以用 new_input_size
            old_state = unit.state.squeeze(0) if unit.state.dim() == 2 else unit.state
            new_state = torch.zeros(new_input_size, device=old_state.device)
            new_state[: old_state.shape[0]] = old_state
            unit.state = new_state

            # —— 4. 升维 state_memory ——
            new_mem = []
            for mem in unit.state_memory:
                p = torch.zeros(new_input_size, device=mem.device)
                p[: mem.shape[-1]] = mem
                new_mem.append(p)
            unit.state_memory = new_mem

            # —— 5. 重建 function（只升不降）并拷贝旧权重 ——
            # 保存旧权重
            old_l1, old_l2 = unit.function[0], unit.function[2]
            w1, b1 = old_l1.weight.data.clone(), old_l1.bias.data.clone()
            w2, b2 = old_l2.weight.data.clone(), old_l2.bias.data.clone()

            # 新网络：输入 new_input_size，隐藏层保持原 hidden_size
            h = unit.hidden_size
            new_l1 = torch.nn.Linear(new_input_size, h, device=w1.device)
            new_l2 = torch.nn.Linear(h, new_input_size, device=w1.device)
            new_func = torch.nn.Sequential(new_l1, torch.nn.ReLU(), new_l2)

            # 无梯度拷贝旧参数
            with torch.no_grad():
                new_l1.weight[:, : w1.shape[1]].copy_(w1)
                new_l1.bias.copy_(b1)
                new_l2.weight[: w2.shape[0], : w2.shape[1]].copy_(w2)
                new_l2.bias[: b2.shape[0]].copy_(b2)

            unit.function = new_func

            # —— 6. 更新 input_size ——
            unit.input_size = new_input_size

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 立即释放旧显存


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

    def _apply_warmup_and_energy_tax(self):
        if self.current_step == 10000:
            self.subsystem_competition = True
            logger.info("[进化] 子系统竞争机制已激活（Subsystem Competition）")

        # 🟡 预热补偿（前 500 步）
        warmup_bonus = self.energy_policy.warmup_bonus(self.current_step)
        if warmup_bonus > 0.0:
            for unit in self.units:
                unit.energy += warmup_bonus
                logger.debug(
                    f"[预热补偿] {unit.id} 初始阶段获得能量 +{warmup_bonus:.3f}"
                )

        # 🟠 能量税（每 10 步）
        if self.current_step > 200 and self.current_step % 100 == 0:
            total_cell_energy = self.total_energy()
            pool_energy = self.energy_pool
            total_e = total_cell_energy + pool_energy
            max_e = self.max_total_energy

            if total_e > max_e:
                excess = total_e - max_e
                tiers = [
                    (0.00, 0.10, 0.05),
                    (0.10, 0.15, 0.10),
                    (0.15, 0.35, 0.15),
                    (0.35, 0.55, 0.20),
                    (0.50, float("inf"), 0.25)
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
                    logger.warning(
                        f"[能量税] {self.current_step} 步：总能 {total_e:.2f} → 累进税 {tax:.2f}（池足够，剩余池能 {self.energy_pool:.2f}）")
                else:
                    tax_from_cells = tax - self.energy_pool
                    self.energy_pool = 0.0
                    loss_per_unit = tax_from_cells / max(len(self.units), 1)
                    for unit in self.units:
                        unit.energy -= loss_per_unit
                    logger.warning(
                        f"[能量税] {self.current_step} 步：总能 {total_e:.2f} → 税 {tax:.2f}，池不足 → 细胞每个扣 {loss_per_unit:.4f}")

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

    def _expand_energy_cap_if_needed(self):
        if self.current_step > 0 and self.current_step % 1000 == 0 and self.max_total_energy < 8000:
            old_max = self.max_total_energy
            self.max_total_energy *= 2
            logger.info(
                f"[资源扩展] 第 {self.current_step} 步：MAX_TOTAL_ENERGY {old_max:.1f} → {self.max_total_energy:.1f}")

    def run_subsystem_competition(self):
        """执行子系统竞争淘汰机制（每 150 步触发一次，淘汰能量最弱子系统）"""
        if not getattr(self, "subsystem_competition", False):
            return  # 若未启用则直接返回

        if self.current_step % 150 != 0:
            return  # 非触发步数则不执行

        # 统计各 subsystem 的总能量
        subsystem_energies = {}
        for unit in self.units:
            if unit.subsystem_id:
                subsystem_energies.setdefault(unit.subsystem_id, 0)
                subsystem_energies[unit.subsystem_id] += unit.energy

        # 若 subsystem 数不足 5 个，则不触发淘汰
        if len(subsystem_energies) < 5:
            return

        # 找出能量最少的 subsystem
        weakest = min(subsystem_energies, key=subsystem_energies.get)
        logger.info(f"[子系统竞争] 淘汰能量最弱的子系统 {weakest}")

        # 删除该 subsystem 的所有细胞单元
        self.units = [u for u in self.units if u.subsystem_id != weakest]
        self.unit_map = {u.id: u for u in self.units}
        self.connections = {u.id: {} for u in self.units}

    def select_elites(self):
        """从当前单元中评选出表现优异的精英个体，并设置 is_elite 标志与重置 age"""
        if self.current_step <= 2000 or self.current_step % 80 != 0:
            return  # 仅每 80 步、步数超过 2000 才执行

        total = len(self.units)
        max_elites = max(1, int(total * 0.08))  # 最多 8%

        # 收集所有最近一次评分
        all_scores = [
            u.local_memory_pool[-1]["score"]
            for u in self.units
            if len(u.local_memory_pool) >= 1
        ]
        if not all_scores:
            return  # 若没有评分数据，则跳过

        score_threshold = _percentile(all_scores, 90)  # 取前 10% 的分数为门槛
        candidates = []

        for u in self.units:
            if len(u.local_memory_pool) < 5:
                continue
            last_score = u.local_memory_pool[-1]["score"]
            if last_score < score_threshold:
                continue

            # ——— 输出质量判断（按角色）———
            hist = [m["output"].view(-1) for m in u.local_memory_pool[-5:]]
            max_len = max(t.numel() for t in hist)
            aligned = [t if t.numel() == max_len else torch.nn.functional.pad(t, (0, max_len - t.numel())) for t in
                       hist]
            diffs = [(aligned[i] - aligned[i + 1]).norm().item() for i in range(len(aligned) - 1)]

            if u.role == "sensor":
                variation = torch.var(torch.stack(aligned), dim=0).mean().item()
                if variation < 0.05:
                    continue
            elif u.role == "processor":
                diversity = sum(diffs) / len(diffs)
                if diversity < 0.1 and getattr(u, "avg_recent_calls", 0) < 2.0:
                    continue
            elif u.role == "emitter":
                avg_diff = sum(diffs) / len(diffs)
                stability = 1.0 if 0.01 < avg_diff < 0.5 else 0.0
                if stability < 1.0 and getattr(u, "avg_recent_calls", 0) < 2.0:
                    continue

            candidates.append((u, last_score))

        # 选出前 max_elites 个单元作为精英
        elites = [u for u, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:max_elites]]

        for u in self.units:
            u.is_elite = False  # 清除所有旧精英标记
        for u in elites:
            u.is_elite = True
            u.age = 0  # 重置年龄，避免误判死亡

    def prune_dead_connections(self):
        """定期清理死连接与表现差的连接，并给予能量惩罚。"""
        if self.current_step % 60 != 0:
            return

        threshold = 50  # 超过多少步未使用则视为死连接

        for from_id in list(self.connections.keys()):
            for to_id in list(self.connections[from_id].keys()):
                last_used = self.connection_usage.get((from_id, to_id), -1)

                # === 情况 1：连接长时间未使用 → 直接删除 ===
                if self.current_step - last_used > threshold:
                    del self.connections[from_id][to_id]
                    self.reverse_connections.get(to_id, set()).discard(from_id)
                    self._clear_connection_usage(from_id, to_id)
                    logger.debug(f"[死连接清除] {from_id} → {to_id}")

                    # 能量惩罚
                    if from_id in self.unit_map:
                        self.unit_map[from_id].energy -= 0.015
                        logger.debug(f"[惩罚] {from_id} 因连接失效，能量 -0.015")

                else:
                    # === 情况 2：使用频率下降的连接 → 权重衰减 ===
                    self.connections[from_id][to_id] *= 0.95
                    if self.connections[from_id][to_id] < 0.1:
                        del self.connections[from_id][to_id]
                        self.reverse_connections.get(to_id, set()).discard(from_id)
                        self._clear_connection_usage(from_id, to_id)
                        logger.debug(f"[连接衰减清除] {from_id} → {to_id}")

    def handle_energy_overflow(self) -> object:
        """处理能量超标的细胞，优先强制分裂，否则将能量存入能量池。"""
        if self.current_step % 40 != 0:
            return

        over_energy_units = [u for u in self.units if u.energy > 3.0]
        if not over_energy_units:
            return

        min_counts = self._get_min_target_counts()
        role_counts = Counter(u.get_role() for u in self.units if u.age < 240)

        for unit in over_energy_units:
            role = unit.get_role()

            # ✅ 分裂条件：当前角色不足 or 总能量未超标
            if (
                    role_counts.get(role, 0) < min_counts[role] and
                    (self.total_energy() < self.max_total_energy)
            ):

                expected_input = self.env_size * self.env_size * INPUT_CHANNELS
                child = unit.clone(
                    new_input_size=expected_input,
                    global_resources=set(self.env.resources.keys()),
                    global_hazards=set(self.env.hazards.keys()),
                    free_positions=self.free_positions
                )

                if self.device.type == "cuda":
                    child.to(self.device)

                split_bonus = 0.35 if self.current_step < 2000 else 0.0
                unit.energy += split_bonus
                child.energy += split_bonus

                self.connect(unit, child)
                self.auto_connect()
                self.add_unit(child)

                logger.info(f"[强制分裂] {unit.id} ({role}) → 数量不足/系统未满 → 复制")
            else:
                # ⚠️ 不满足分裂条件 → 转为能量池
                contribution = unit.energy * 0.2
                unit.energy *= 0.8
                self.energy_pool += contribution

                logger.debug(
                    f"[能量转移] {unit.id} ({role}) 系统过载 → 存入能量池 {contribution:.2f}，保留 {unit.energy:.2f}")

    def _target_energy_for_unit(self, unit: CogUnit) -> float:
        bias = float(unit.gene.get(f"{unit.role}_bias", 1.0))
        bias = min(max(bias, 0.3), 2.0)

        if unit.role == "emitter":
            preferred = 0.78 + 0.28 * math.sqrt(bias)
            return max(1.05, min(1.35, preferred))

        base = 0.46 if unit.role == "processor" else 0.44
        growth = 0.16 * math.sqrt(bias)
        ceiling = 1.18 if unit.role == "processor" else 1.08
        floor = 0.58 if unit.role == "processor" else 0.52
        return max(floor, min(ceiling, base + growth))

    def _emitter_priority_score(self, unit: CogUnit) -> float:
        recent = getattr(unit, "avg_recent_calls", 0.0)
        success = 0.0
        if hasattr(unit, "meta"):
            rate = unit.meta.recent_success_rate()
            if rate is not None:
                success = rate
        maturity = min(unit.age / 260.0, 1.0)
        return (
            0.55 * recent
            + 0.22 * success
            + 0.15 * float(unit.energy)
            + 0.08 * maturity
        )

    def _rapid_emitter_refuel(self):
        """保持兼容性的占位方法，现阶段 emitter 不再享受额外补能。"""
        return

    def supply_energy_from_pool(self):
        """
        每一步都根据个体属性评估是否需要从能量池补给，避免统一撒网。

        - 首先按角色偏好和当前缺口计算目标能量；仅低于目标的细胞会参与分配。
        - 若系统总能量仍低于上限的 90%，优先使用能量池填补缺口。
        - 若总能量已达上限但仍存在低能量细胞，则提供小额稳态补给，防止个体瞬时死亡。
        """
        if not self.units or self.energy_pool <= 1e-6:
            return

        total_cell_energy = self.total_energy()
        max_allowable = self.max_total_energy * 0.9

        deficits = []
        for unit in self.units:
            target = self._target_energy_for_unit(unit)
            gap = target - float(unit.energy)
            if gap <= 0.0:
                continue
            recent = getattr(unit, "avg_recent_calls", 0.0)
            activity_bonus = 1.0 + 0.3 * min(recent, 4.0)
            bias = float(unit.gene.get(f"{unit.role}_bias", 1.0))
            resilience = 1.0 / max(0.5, math.sqrt(max(bias, 0.3)))
            weight = gap * activity_bonus * resilience
            if unit.role == "processor":
                weight *= 1.0 + 0.05 * min(unit.age / 360.0, 1.0)
            deficits.append({"unit": unit, "gap": gap, "weight": weight})

        active_deficits = [d for d in deficits if d["gap"] > 1e-6 and d["weight"] > 0.0]
        if not active_deficits:
            return

        pool_before = self.energy_pool

        def distribute(budget: float, *, cap: float) -> float:
            if budget <= 1e-6:
                return 0.0
            eligible = [d for d in active_deficits if d["gap"] > 1e-6]
            if not eligible:
                return 0.0

            eligible.sort(key=lambda item: item["weight"], reverse=True)
            total_weight = sum(max(d["weight"], 1e-6) for d in eligible)
            if total_weight <= 0.0:
                return 0.0

            consumed = 0.0
            shares = []
            for data in eligible:
                weight = max(data["weight"], 1e-6)
                proportional = budget * (weight / total_weight)
                share = min(cap, data["gap"], proportional)
                if share > 0.0:
                    data["unit"].energy += share
                    data["gap"] -= share
                    consumed += share
                shares.append(max(share, 0.0))

            remaining_budget = budget - consumed
            if remaining_budget > 1e-6:
                for idx, data in enumerate(eligible):
                    if remaining_budget <= 1e-6:
                        break
                    if data["gap"] <= 1e-6:
                        continue
                    room = max(cap - shares[idx], 0.0)
                    if room <= 0.0:
                        continue
                    extra = min(room, data["gap"], remaining_budget)
                    if extra <= 0.0:
                        continue
                    data["unit"].energy += extra
                    data["gap"] -= extra
                    consumed += extra
                    remaining_budget -= extra

            return consumed

        primary_cap, secondary_cap = self.energy_policy.pool_caps()
        cap_gap = max(0.0, max_allowable - total_cell_energy)
        primary_budget = min(self.energy_pool, cap_gap)
        consumed_primary = distribute(primary_budget, cap=primary_cap)
        self.energy_pool -= consumed_primary

        remaining_needy = [d for d in active_deficits if d["gap"] > 1e-6]
        consumed_secondary = 0.0
        if remaining_needy and self.energy_pool > 1e-6:
            stability_factor = self.energy_policy.pool_stability_factor()
            stability_budget = min(self.energy_pool, stability_factor * len(remaining_needy))
            consumed_secondary = distribute(stability_budget, cap=secondary_cap)
            self.energy_pool -= consumed_secondary

        consumed_total = (pool_before - self.energy_pool)
        if consumed_total > 1e-6:
            logger.debug(
                f"[能量补给] 消耗 {consumed_total:.2f}（主补 {consumed_primary:.2f} / 稳态 {consumed_secondary:.2f}），"
                f" 剩余池 {self.energy_pool:.2f} → 当前细胞总能量 {self.total_energy():.2f}"
            )

    def clone_and_connect(self, parent):
        """
        给定一个 parent 单元，克隆其子单元并建立连接。
        返回新生成的 child 单元。
        """
        expected_input = self.env_size * self.env_size * INPUT_CHANNELS

        child = parent.clone(
            new_input_size=expected_input if parent.input_size != expected_input else None,
            global_resources=self.env.resources,
            global_hazards=self.env.hazards,
            free_positions=self.free_positions
        )
        if hasattr(child, "attach_energy_policy"):
            child.attach_energy_policy(self.energy_policy)

        if getattr(parent, "role", None) == "emitter":
            cap = getattr(parent, "emitter_split_cap", None)
            if cap is None:
                cap = random.randint(3, 6)
                parent.emitter_split_cap = cap
            parent.emitter_splits_done = getattr(parent, "emitter_splits_done", 0) + 1
            if parent.emitter_splits_done >= cap:
                logger.debug(
                    f"[分裂封顶] {parent.id} 达到 {parent.emitter_splits_done}/{cap} 次限制，后续将作为工作细胞")

        self.connect(parent, child)
        self.auto_connect()  # 让新单元主动寻找连接

        return child

    def meta_self_evaluation(self):
        """
        每 250 步触发一次：非 sensor 单元根据成功率评估自己，低于阈值则申请升级。
        """
        if self.current_step % 250 != 0:
            return

        for unit in self.units:
            if unit.role == "sensor":
                continue
            if unit.evaluate_self(min_rate=0.5):
                unit.request_upgrade(
                    target_role=unit.get_role(),
                    reason="low_success_rate"
                )

    def record_long_term_memory(self, prev_energies: dict, state_snapshot: torch.Tensor):
        """
        每 20 步触发一次：记录所有单元的长期记忆，包括状态、动作、奖励和结果（成功或失败）。
        """
        if self.current_step % 20 != 0 or self.current_step < 500:
            return

        for u in self.units:
            reward = u.energy - prev_energies.get(u.id, 0.0)
            action = u.last_output.clone().detach() if hasattr(u, "last_output") else None
            outcome = "success" if reward > 0 else "fail"
            unit_state = getattr(u, "last_input_snapshot", state_snapshot)
            u.record_memory(unit_state, action, reward, outcome)

    def _maintain_explorer_emitter_ratio(self):
        emitters = [u for u in self.units if u.role == "emitter"]
        total = len(emitters)
        if total == 0:
            return

        target = max(1, int(total * 0.1))  # 目标探索者数量

        explorers = [e for e in emitters if getattr(e, "is_permanent_explorer", False)]
        non_explorers = [e for e in emitters if not getattr(e, "is_permanent_explorer", False)]

        # === 补齐探索者 ===
        if len(explorers) < target:
            need = target - len(explorers)
            non_explorers.sort(key=lambda u: -u.energy)  # 按能量从高到低挑
            for u in non_explorers[:need]:
                u.is_permanent_explorer = True
                logger.info(f"[开拓者补齐] {u.id} 被标记为开拓 emitter，充满了决心！")

        # === 削减多余探索者 ===
        elif len(explorers) > target:
            excess = len(explorers) - target
            explorers.sort(key=lambda u: u.energy)  # 能量最低者先剔除
            for u in explorers[:excess]:
                delattr(u, "is_permanent_explorer")
                logger.info(f"[开拓者削减] {u.id} 被取消开拓身份。列车人太多啦，下去点下去点。")

    def _compute_emitter_metabolic_scalar(self, unit: CogUnit, *, adjusted_var: float, call_density: float) -> float:
        bias = max(0.3, float(unit.gene.get("emitter_bias", 1.0)))
        energy = max(unit.energy, 0.0)
        preferred = 0.92 + 0.24 * math.sqrt(bias)
        energy_gap = preferred - energy
        energy_term = 1.0 - 0.28 * math.tanh(energy_gap)
        energy_term = max(0.75, min(1.22, energy_term))

        maturity = min(unit.age / 220.0, 1.0)
        activity = min(call_density, 1.5)
        stability = 1.0 - min(adjusted_var / (adjusted_var + 1.0), 0.7)

        composite = (0.86 + 0.05 * activity + 0.045 * maturity) * (0.92 + 0.08 * stability)
        trait_term = 0.92 / math.sqrt(bias)
        scalar = 0.08 * composite * trait_term * energy_term
        return max(self._emitter_metabolic_floor, min(self._emitter_metabolic_ceiling, scalar))

    def _apply_unit_metabolism(self, unit, unit_input):
        if getattr(unit, "resting", False):
            return
        incoming = self.reverse_connections.get(unit.id, ())
        unit.recent_calls = len(incoming)
        unit.connection_count = len(self.connections.get(unit.id, {}))
        unit.call_history.append(unit.recent_calls)
        if len(unit.call_history) > unit.call_window:
            unit.call_history.pop(0)
        unit.avg_recent_calls = sum(unit.call_history) / len(unit.call_history)
        unit.inactive_steps = unit.inactive_steps + 1 if unit.recent_calls == 0 else 0
        unit.current_step = self.current_step

        raw_var = float(unit_input.var(unbiased=False))
        freq = unit.avg_recent_calls
        conn = unit.connection_count
        call_density = min(freq / (conn + 1), 3.0)
        conn_strength_sum = sum(self.connections.get(unit.id, {}).values())
        conn_strength_sum = min(conn_strength_sum, 6.0)

        if unit.role == "emitter" and raw_var > 0.0:
            adjusted_var = math.sqrt(raw_var)
        else:
            adjusted_var = raw_var

        if unit_input.dim() == 0:
            feature_dim = 1.0
        elif unit_input.dim() == 1:
            feature_dim = float(unit_input.shape[0])
        else:
            feature_dim = float(unit_input.shape[-1])
        feature_dim = max(feature_dim, 1.0)

        dim_baseline = self._role_feature_dim_ema.get(unit.role, 0.0)
        if dim_baseline <= 0.0:
            dim_baseline = feature_dim

        dim_scale = math.sqrt(feature_dim / max(dim_baseline, 1.0))
        dim_scale = max(0.65, min(1.45, dim_scale))
        updated_baseline = (
            self._feature_dim_ema_beta * dim_baseline
            + (1.0 - self._feature_dim_ema_beta) * feature_dim
        )
        self._role_feature_dim_ema[unit.role] = max(updated_baseline, 1.0)

        trait_bias = float(unit.gene.get(f"{unit.role}_bias", 1.0))
        bias_factor = max(0.6, min(1.4, trait_bias))
        preferred_energy = 0.75 + 0.2 * math.sqrt(max(trait_bias, 0.3))
        energy_gap = preferred_energy - float(unit.energy)
        energy_mod = 1.0 - 0.25 * math.tanh(energy_gap)
        energy_mod = max(0.65, min(1.35, energy_mod))
        step_factor = 1.0 + 0.00001 * max(0, self.current_step - 2000)
        unit_factor = 1.0 + 0.00005 * max(0, len(self.units) - 150)

        adjusted_term = adjusted_var * (0.33 if unit.role == "emitter" else 0.35)
        decay_base = adjusted_term + call_density * 0.14 + conn_strength_sum * 0.12
        decay = decay_base * dim_scale * bias_factor * step_factor * unit_factor * energy_mod
        # honor 单元自己的 metabolic_rate
        progress = min(max(self.current_step - 500, 0) / 4000.0, 1.0)
        base_factor = 0.018 + 0.012 * progress
        if unit.role == "emitter":
            base_factor += 0.002
        metabolic_rate = getattr(unit, "metabolic_rate", 1.0)
        drain = decay * base_factor * metabolic_rate
        scalar = 1.0
        if unit.role == "emitter":
            scalar = self._compute_emitter_metabolic_scalar(
                unit, adjusted_var=adjusted_var, call_density=call_density
            )
            drain *= scalar

        unit.energy -= drain

        unit.energy = max(unit.energy, 0.0)

        rms = float(unit_input.pow(2).mean().sqrt().item()) if unit_input.numel() else 0.0
        self._update_metabolic_stats(
            unit.role,
            input_var=raw_var,
            adjusted_var=adjusted_var,
            input_rms=rms,
            energy=unit.energy,
            drain=drain,
            scalar=scalar,
            dim_scale=dim_scale,
            feature_dim=feature_dim,
        )

        logger.debug("[代谢] %s var=%.3f adj=%.3f conn_sum=%.2f", unit.id, raw_var, adjusted_var, conn_strength_sum)

        logger.debug("[代谢] %s var=%.3f adj=%.3f conn_sum=%.2f", unit.id, raw_var, adjusted_var, conn_strength_sum)
    def _finalize_unit_update(self, unit, unit_input, state_snapshot, output_buffer, pending, allow_clone):
        try:
            recs = unit.recall(state_snapshot, k=5, metric='cosine')
            if recs:
                rewards = [r['reward'] for r in recs]
                top_reward = rewards[0]
                avg_reward = sum(rewards) / len(rewards)
                if top_reward > avg_reward + 1:
                    unit.last_output = recs[0]['action'].to(self.device).clone()
        except RuntimeError:
            pass

        unit.update(unit_input)
        if hasattr(self, "active_units"):
            self.active_units.add(unit)
        output_buffer[unit.id] = unit.get_output()

        for uid in self.reverse_connections.get(unit.id, ()):
            if unit.id not in self.connections.get(uid, {}):
                continue
            self.connections[uid][unit.id] *= 1.05
            self.connections[uid][unit.id] = min(self.connections[uid][unit.id], 5.0)

        wants_split = unit.should_split()
        if wants_split:
            if allow_clone:
                pending[unit.role].append(unit)
            else:
                logger.debug(f"[系统保护] 总能量过高，暂缓 {unit.id} 分裂")

        if unit.should_die():
            logger.debug(f"[死亡] {unit.id} 被移除")
            self.remove_unit(unit)


    # —— 2) 判断是否进入静吸模式 —— #
    def _check_enter_static_mode(self):
        if self.current_step == self._static_mode_forbid_step:
            return
        if self.current_step >= 1000 and self.steps_since_last_reward >= 100 and not self.static_mode:
            self._enter_static_mode()

    # —— 3) 进入静吸模式 —— #
    def _enter_static_mode(self):
        """
        静息模式：只有 10% 的 emitter 及其上游 processor+sensor 活跃。
        """
        # 1) 随机选 10% emitter
        logger.warning("没奖励，没动力了，睡大觉！")
        # ❌ 排除探索者
        emitters = [
            u for u in self.units
            if u.role == "emitter" and not getattr(u, "is_permanent_explorer", False)
        ]
        n_active = max(1, math.ceil(0.1 * len(emitters)))
        if n_active > len(emitters):
            logger.warning("[静息模式] 可选的普通 emitter 不足，启用全部普通 emitter")
            active_emitters = set(emitters)
        else:
            active_emitters = set(random.sample(emitters, n_active))

        # 2) 扩展到上游 processor 和最强 sensor
        active = set(active_emitters)
        for e in active_emitters:
            # 找到上游 processor
            proc_ids = self.reverse_connections.get(e.id, ())
            procs = {self.unit_map[p] for p in proc_ids
                     if p in self.unit_map and self.unit_map[p].role == "processor"}
            active |= procs
            # 每个 processor 找一个能量最高的 sensor
            for p in procs:
                sensor_ids = self.reverse_connections.get(p.id, ())
                best = max(
                    (self.unit_map[s] for s in sensor_ids
                     if s in self.unit_map and self.unit_map[s].role == "sensor"),
                    key=lambda u: u.energy, default=None
                )
                if best:
                    active.add(best)

        # 3) 记录原始 age，并设置 resting / metabolic_rate
        self._orig_age = {u.id: u.age for u in self.units}  # ✅ 不只是 active，而是所有细胞
        for u in self.units:
            u.resting = (u not in active)
            # 记录原始代谢率
            self._orig_metabolic[u.id] = getattr(u, "metabolic_rate", 1.0)
            if u in active_emitters:
                u.metabolic_rate = 0.1  # emitter 只消耗 10%
            else:
                u.metabolic_rate = 0.0  # 上游 processor/sensor 不消耗

        self.static_mode = True
        self.static_mode_entry_step = self.current_step

    # —— 4) 退出静吸模式 —— #
    def _exit_static_mode(self):
        logger.warning("每日刷新了，集美们动起来动起来")
        self.static_mode = False
        self.static_mode_exit_step = self.current_step  # ✅ 记录当前步数
        self.static_mode_entry_step = None
        for u in self.units:
            # 如果这个单元进入了 resting，它才应该有一个原始 metabolic_rate
            if u.id in self._orig_metabolic:
                u.metabolic_rate = self._orig_metabolic[u.id]
            else:
                # 没有原始记录时，设置一个安全的默认值
                u.metabolic_rate = getattr(u, "metabolic_rate", 1.0)
            # 清除 resting 标记
            if hasattr(u, "resting"):
                delattr(u, "resting")

        self._orig_metabolic.clear()
        self._orig_age.clear()

    def _static_step(self, input_tensor: torch.Tensor):
        """
        静息模式下的单步，只对 active_ids 单元做“追目标→更新→奖励/惩罚”，跳过其它逻辑。
        """
        # —— 拆 env_state + goal ——
        env_dim = self.env_size * self.env_size * N_STATE_CHANNELS
        batch = input_tensor
        env_state = batch[:, :env_dim]
        res_map   = self.target_vector[0].unsqueeze(0)
        hzd_map   = self.target_vector[1].unsqueeze(0)
        full_state = torch.cat([env_state, torch.cat([res_map, hzd_map], dim=1)], dim=1)

        # —— ⚙️ 静息模式下也要给 emitter 初始化 goal_vec —— #
        for u in self.units:
            if u.get_role() == "emitter":
                ext = self.target_vector.clone()            # [2, env²]
                int_zero = torch.zeros(1, self.env_size*self.env_size, device=self.device)
                u.goal_vec = torch.cat([ext, int_zero], dim=0)
                u.current_hazard_xy = getattr(self, "current_hazard_xy", None)

        # upstream logic 重用 snapshot/prev_energies（如果需要 record_memory）
        state_snapshot = full_state.clone().squeeze(0).detach().to(self.device)
        prev_energies  = {u.id: u.energy for u in self.units}

        # 只更新 active 的单元
        active_ids = {u.id for u in self.units if not getattr(u, "resting", False)}
        expected_in = self.env_size * self.env_size * INPUT_CHANNELS
        for u in self.units:
            if u.id not in active_ids:
                continue
            inp = self._prepare_unit_before_update(u, full_state, expected_in)
            self._apply_unit_metabolism(u, inp)
            u.update(inp)
            # —— 恢复 age，不让寿命增加 —— #
            u.age = self._orig_age.get(u.id, u.age)

        # 奖励/惩罚 + 统计连续无奖励步数
        self.reward_emitter_grid_environment()
        self._handle_reward_and_penalty()

        # 如果刚获得奖励，会在 _handle_reward_and_penalty 内 exit static
        # ✅ 所有静息单元的 age 保持静止
        for u in self.units:
            if getattr(u, "resting", False):
                u.age = self._orig_age.get(u.id, u.age)

        self._perform_system_maintenance()

        self._check_static_timeout()

        return

    def _check_static_timeout(self):
        if not self.static_mode:
            return
        if self.static_mode_entry_step is None:
            return
        if self.current_step - self.static_mode_entry_step < self.static_mode_max_duration:
            return
        logger.warning("[静息模式] 已连续休息 %d 步，强制苏醒", self.static_mode_max_duration)
        self.steps_since_last_reward = 0
        self._exit_static_mode()

    def _perform_system_maintenance(self):
        self.supply_energy_from_pool()
        self._rapid_emitter_refuel()

        if self.current_step > 0 and self.current_step % 40 == 0:
            self.rebalance_cell_types()

    def _ensure_minimum_population(self):
        if self.units:
            return
        if self.static_mode:
            self._exit_static_mode()
        logger.warning("[紧急增殖] 细胞数量降为 0，触发紧急补种")
        self._init_seed_units(n_sensor=6, n_processor=12, n_emitter=6, device=self.device)
        self.steps_since_last_reward = 0
        self._static_mode_forbid_step = self.current_step


    def step(self, input_tensor: torch.Tensor):
        if self.current_step % 1000 == 0:
            # 重置计数
            self.removed_resources_count = 0
            self.removed_hazards_count  = 0
            self.removed_hazards_by_reward = 0
        if self.current_step % 200 == 0:
            self.active_units.clear()

        self.current_step += 1
        if self.current_step % 1000 == 0:
            self.steps_since_last_reward = 0
            self._static_mode_forbid_step = self.current_step
            if self.static_mode:
                self._exit_static_mode()
        self._update_global_counts()

        # === Transformer 一网打尽 ===
        if RF.use_shared_tx and (self.current_step % RF.shared_tx_interval == 0):
            self._run_shared_transformer()

        # 同步环境尺寸
        self._sync_environment_dimensions()

        # —— 改为拆分外部传入的 input_tensor ——
        # 假设调用方已经做了 torch.cat([env_state, goal_vec], dim=1)
        batch = input_tensor             # (1, env_dim+goal_dim)
        env_dim  = self.env_size * self.env_size * N_STATE_CHANNELS
        env_state = batch[:, :env_dim]                    # (1, env_dim)
        # ---------- NEW: 把目标 one-hot 变成 2 个平面 ----------
        res_map = self.target_vector[0].unsqueeze(0)  # (1, env²)  资源
        hzd_map = self.target_vector[1].unsqueeze(0)  # (1, env²)  陷阱
        goal_flat = torch.cat([res_map, hzd_map], dim=1)  # (1, 2·env²)

        # 6 通道打包
        full_state = torch.cat([env_state, goal_flat], dim=1)  # (1, 6·env²)
        # ─── 新增：长期记忆准备 ───
        # 1) 把 state snapshot 存下来（去掉 batch 维）
        state_snapshot = full_state.clone().squeeze(0).detach().to(self.device)
        # 2) 记录下此刻每个 unit 的能量，用来算 reward
        prev_energies = {u.id: u.energy for u in self.units}

        # 准备目标向量
        self._update_target_vector()

        self._expand_environment_curriculum()

        self._rebuild_free_positions()
        
        self._expand_energy_cap_if_needed()

        if self.static_mode:
            return self._static_step(input_tensor)

        self._apply_warmup_and_energy_tax()

        # 每次循环时，根据当前步数决定更新间隔
        step = self.current_step

        if step < 1000:
            interval = 1000
        elif step < 1500:
            interval = 500
        elif step < 3000:
            interval = 250
        elif step < 5000:
            interval = 200
        else:
            interval = 100

        # # 然后用这个 interval 来判断是否需要更新 target_vector
        self.prune_connections()

        self.assign_subsystems()

        self.run_subsystem_competition()

        self.select_elites()

        new_units = []  # 新生成的单元（复制）
        pending = {"sensor": [], "processor": [], "emitter": []}  # NEW: 待复制父单元
        output_buffer = {}  # 缓存每个单元的输出 {unit_id: output_tensor}

        # === 系统总能量限制，保护 clone ===
        cell_energy = self.total_energy()
        allow_clone = cell_energy < self.max_total_energy * 0.98

        # === 第一阶段：单元更新处理 ===
        # 统计当前 emitter 数量
        expected_input = self.env_size * self.env_size * INPUT_CHANNELS

        # —— 如果处于静吸模式，先筛掉“休眠”的单元 —— #
        active_ids = None
        if self.static_mode:
            active_ids = {u.id for u in self.units if not getattr(u, "resting", False)}
        for unit in self.units[:]:
            pos = tuple(self.env.agent_pos)
            for u in self.units:
                if u.get_role() == "emitter":
                    u.visit_counts.setdefault(pos, 0)
                    u.visit_counts[pos] += 1
            if active_ids is not None and unit.id not in active_ids:
                continue
            unit_input = self._prepare_unit_before_update(unit, full_state, expected_input)
            unit_input = unit_input.to(self.device)
            self._apply_unit_metabolism(unit, unit_input)
            self._finalize_unit_update(unit, unit_input, state_snapshot, output_buffer, pending, allow_clone)


        if self.current_step > 0 and self.current_step % 50 == 0:
            self.finalize_deaths()

        self._ensure_minimum_population()

        self.auto_connect()

        self.prune_dead_connections()

        # 在执行环境奖励逻辑之后：
        self.reward_emitter_grid_environment()
        # 加入这一行：处理删除惩罚 & 计数
        self._handle_reward_and_penalty()
        # 再判断是否要进静吸
        self._check_enter_static_mode()

        # === 重度维护：只在部分步数执行，避免每步循环开销 ===

        # —— 可选路径追踪（纯调试，不影响状态） ——
        if self.debug and self.current_step % 100 == 0:
            self.trace_info_paths()

        # —— 统一统计 & 连接打印（仅 debug） ——
        self._log_stats_and_conns()

        # === 40 %-限额复制（>15 细胞才触发） ===
        selected_parents = self._select_clone_parents(pending)
        for parent in selected_parents:
            child = self.clone_and_connect(parent)
            new_units.append(child)

            # —— 最终一次性把所有 child 加入图结构 ——
        for unit in new_units:
            self.add_unit(unit)

        self._perform_system_maintenance()

        self.handle_energy_overflow()

        interval = max(200, len(self.units) // 8)
        if self.current_step % interval == 0:
            self.merge_redundant_units()
            self.restructure_common_subgraphs()

        # —— 新增：周期性清理统计 ——
        if self.current_step % 50 == 0:
            res_cleared = self.removed_resources_count
            haz_by_reward = self.removed_hazards_by_reward
            haz_cleared = self.removed_hazards_count
            res_left = sum(self.env.resources.values())
            haz_left = sum(self.env.hazards.values())
            logger.warning(
                f"[清理统计] 已清理资源 {res_cleared} 个，"
                f"已清理危险 {haz_cleared} 个，因资源奖励移除危险 {haz_by_reward} 个；"
                f"剩余资源 {res_left} 个，剩余危险 {haz_left} 个"
            )


        self.meta_self_evaluation()
        self.record_long_term_memory(prev_energies, state_snapshot)
        if self.current_step % 50 == 0:
            self._maintain_explorer_emitter_ratio()

    def _handle_reward_and_penalty(self):
        # 本步是否有 emitter 刚获得奖励？
        found = any(
            getattr(u, "last_reward_step", None) == self.current_step
            for u in self.units if u.role == "emitter"
        )

        if found:
            self.steps_since_last_reward = 0
            # 找到这一步获得奖励的 emitter
            last = max(
                (u for u in self.units
                 if u.role == "emitter"
                 and u.last_reward_step == self.current_step),
                key=lambda u: u.last_reward_step,
                default=None
            )
            if self.static_mode:
                self._exit_static_mode()
        else:
            self.steps_since_last_reward += 1

    def _sync_environment_dimensions(self):
        if self.env.size != self.env_size:
            old_size = self.env_size
            self.env_size = self.env.size
            self.processor_hidden_size = self.env_size * self.env_size * INPUT_CHANNELS
            full_state_dim = self.processor_hidden_size + 2 * (self.env_size * self.env_size)
            self.rl_agent.resize_state_dim(full_state_dim)
            self._target_buf = self._target_buf.new_zeros((2, self.env_size * self.env_size))
            if RF.use_shared_tx:
                self._init_shared_tx()
            logger.info(f"[Env Resize] {old_size}→{self.env_size}, synced CogGraph dims")

    def _update_target_vector(self):
        agent_pos = tuple(self.env.agent_pos)
        nearest_res = self.env.get_nearest_known_resource(agent_pos)
        nearest_hzd = self.env.get_nearest_known_danger(agent_pos)

        target_xy = nearest_res
        self.current_hazard_xy = nearest_hzd

        target_vec = self._target_buf
        target_vec.zero_()

        if target_xy is not None:
            idx = target_xy[1] * self.env_size + target_xy[0]
            target_vec[0, idx] = 1.0

        if nearest_hzd is not None:
            hidx = nearest_hzd[1] * self.env_size + nearest_hzd[0]
            target_vec[1, hidx] = 1.0

        self.target_vector = target_vec


    def _prepare_unit_before_update(self, unit, full_state, expected_input):
        if unit.input_size < expected_input:
            self.expand_unit_dim(unit, expected_input)

        if unit.get_role() == "emitter":
            # === 判断使用最近资源 / 陷阱 / 好奇点 ===
            use_curiosity = getattr(unit, "is_permanent_explorer", False) and self.current_step >= 2000

            if not use_curiosity:
                pos = unit.get_position()
                nearest_res = self.env.get_nearest_known_resource(pos)
                nearest_hzd = self.env.get_nearest_known_danger(pos)

                if nearest_res is not None:
                    unit.personal_goal = nearest_res
                    unit.goal_type = "resource"
                elif nearest_hzd is not None:
                    unit.personal_goal = nearest_hzd
                    unit.goal_type = "hazard"
                else:
                    use_curiosity = True

            if use_curiosity:
                unit.goal_type = "curiosity"
                taken = {
                    e.personal_goal for e in self.units
                    if e is not unit and e.get_role() == "emitter" and e.personal_goal is not None
                }
                unit.visit_counts = getattr(unit, "visit_counts", Counter())
                unit.personal_goal = sample_unvisited(self.env_size, unit.visit_counts, exclude=taken)
                unit.visit_counts.setdefault(unit.personal_goal, 0)

            # === 构造目标向量 ===
            res_map = torch.zeros(1, self.env_size * self.env_size, device=self.device)
            hz_map = torch.zeros_like(res_map)
            cu_map = torch.zeros_like(res_map)

            if unit.personal_goal is not None:
                idx = unit.personal_goal[1] * self.env_size + unit.personal_goal[0]
                if unit.goal_type == "resource":
                    res_map[0, idx] = 1.0
                elif unit.goal_type == "hazard":
                    hz_map[0, idx] = 1.0
                elif unit.goal_type == "curiosity":
                    cu_map[0, idx] = 1.0

            unit.goal_vec = torch.cat([res_map, hz_map, cu_map], dim=0)
            unit.current_hazard_xy = getattr(self, "current_hazard_xy", None)

        unit.global_emitter_count = sum(1 for u in self.units if u.get_role() == "emitter")
        incoming = self.reverse_connections.get(unit.id, ())
        unit.recent_calls = len(incoming)

        if unit.get_role() == "sensor":
            return full_state
        elif incoming:
            weighted_outputs = []
            total_weight = 0.0
            for uid in incoming:
                if unit.id not in self.connections.get(uid, {}):
                    continue
                if uid not in self.unit_map:
                    continue  # 已被删除的单元，跳过
                strength = self.connections[uid][unit.id]
                self._mark_connection_used(uid, unit.id)
                output = self.unit_map[uid].get_output().squeeze(0)
                target_len = self.processor_hidden_size
                if output.shape[0] != target_len:
                    padding = (0, target_len - output.shape[0])
                    output = torch.nn.functional.pad(output, padding, value=0)
                weighted_outputs.append(output * strength)
                total_weight += strength
            if total_weight > 0:
                merged = torch.stack(weighted_outputs).sum(dim=0) / total_weight
                return merged.unsqueeze(0)
            else:
                return torch.zeros(unit.input_size, device=self.device).unsqueeze(0)
        else:
            logger.debug(f"[零输入] {unit.id} 无上游连接，使用零输入更新")
            return torch.zeros(unit.input_size).unsqueeze(0)

    def reward_emitter_grid_environment(self):
        policy = self.energy_policy
        decay_threshold = policy.inactivity_threshold()
        decay_amount = policy.inactivity_decay()
        """基于 emitter 输出，在网格环境中执行资源 / 陷阱 奖励与惩罚逻辑。"""
        outputs = self.collect_emitter_outputs()
        if outputs is None:
            return

        action_indices = [torch.argmax(out).item() for out in outputs]
        emitters = [u for u in self.units if u.get_role() == "emitter"]

        for idx, unit in enumerate(emitters):
            out = outputs[idx]
            pred = torch.argmax(self._align_to_goal_dim(out)).item()
            px, py = pred % self.env_size, pred // self.env_size
            if (px, py) == unit.personal_goal:
                intrinsic_bonus = policy.intrinsic_completion_bonus()
                unit.intrinsic_reward = intrinsic_bonus
                unit.energy += intrinsic_bonus
                unit.meta.record(action="intrinsic", reward=+intrinsic_bonus)
                logger.info("你达到了你好奇的地方，心中充满了决心")
                unit.visit_counts.setdefault((px, py), 0)
                unit.visit_counts[(px, py)] += 1
                if not getattr(unit, "is_permanent_explorer", False):
                    unit._last_intrinsic_step = self.current_step
                unit.personal_goal = None
                if getattr(unit, "goal_vec", None) is not None \
                        and unit.goal_vec.dim() == 2 \
                        and unit.goal_vec.size(0) >= 3:
                    unit.goal_vec[2].zero_()

        for i, unit in enumerate(emitters):
            out_vec = outputs[i]
            pred = self._align_to_goal_dim(out_vec)
            pred = torch.softmax(pred, dim=0)

            if unit.goal_vec.dim() == 2 and unit.goal_vec.size(0) >= 2:
                # 用 personal_goal 而不是全局最近资源
                if unit.personal_goal is not None:
                    idx = unit.personal_goal[1] * self.env_size + unit.personal_goal[0]
                    res_vec = torch.zeros_like(unit.goal_vec[0])
                    res_vec[idx] = 1.0
                else:
                    res_vec = unit.goal_vec[0]

                hz_vec = unit.goal_vec[1]
            else:
                res_vec = unit.goal_vec.view(-1) if unit.goal_vec.dim() == 1 else unit.goal_vec[0]
                hz_vec = torch.zeros_like(res_vec)

            goal_vec = res_vec
            res_dist = float((pred - res_vec).pow(2).mean().sqrt())
            is_res_hit = res_dist < HIT_THRESH
            is_res_near = HIT_THRESH <= res_dist <= 1.5
            cur_idx = torch.argmax(res_vec).item()
            pred_idx = torch.argmax(pred).item()

            pred_idx = torch.argmax(pred).item()

            hazard = getattr(unit, "current_hazard_xy", None)
            if hazard is not None:
                hx, hy = hazard
                hazard_idx = hy * self.env_size + hx
                is_hz_hit = (pred_idx == hazard_idx)
            else:
                is_hz_hit = False

            upstream_processors = [
                self.unit_map[pid]
                for pid in self.reverse_connections.get(unit.id, set())
                if pid in self.unit_map and self.unit_map[pid].get_role() == "processor"
            ]

            # === 靠近陷阱后又撤退，触发好奇点切换 ===
            if getattr(unit, "goal_type", "") == "hazard" and hazard is not None:
                px, py = pred_idx % self.env_size, pred_idx // self.env_size
                hz_dist = math.hypot(px - hx, py - hy)

                prev_dist = getattr(unit, "_last_hazard_dist", float("inf"))
                unit._last_hazard_dist = hz_dist  # 更新距离记录

                if prev_dist <= 3.0 and hz_dist > 4.0:
                    unit.goal_type = "curiosity"
                    unit.personal_goal = sample_unvisited(self.env_size, unit.visit_counts)
                    unit.visit_counts[unit.personal_goal] = 0
                    logger.info(f"[目标切换] emitter {unit.id} 接近陷阱后撤退，发现不对劲，有歹徒要害我！ → 切换为好奇点 {unit.personal_goal}")

            if is_hz_hit and (hx, hy) in self.env.hazards:
                hazard_loss, hazard_share = policy.hazard_penalties()
                unit.energy -= hazard_loss
                unit.meta.record(action=pred_idx, reward=-hazard_loss)
                for p in upstream_processors:
                    if hazard_share > 0.0:
                        p.energy -= hazard_share
                        p.meta.record(action=pred_idx, reward=-hazard_share)
                unit.is_hazard_confirmed = True
                unit.last_action_rewarded = False
                if self.env.hazards[(hx, hy)] > 0:
                    self.env.hazards[(hx, hy)] -= 1
                    if self.env.hazards[(hx, hy)] == 0:
                        del self.env.hazards[(hx, hy)]
                self.env.update_known_cell((hx, hy))
                self.removed_hazards_count += 1
                unit.goal_vec[1, hazard_idx] = 0.0
                # —— 吃完这个资源之后，重新选最近的资源和惩罚点 —— #
                if getattr(unit, "is_permanent_explorer", False):
                    continue  # 永久探索者不应被重设为资源目标

                next_res = self.env.get_nearest_known_resource(unit.get_position())
                if next_res is not None:
                    ridx = next_res[1] * self.env_size + next_res[0]
                    unit.goal_vec[0].zero_()
                    unit.goal_vec[0, ridx] = 1.0
                next_hz = self.env.get_nearest_known_danger(unit.get_position())
                if next_hz is not None:
                    hidx = next_hz[1] * self.env_size + next_hz[0]
                    unit.goal_vec[1].zero_()
                    unit.goal_vec[1, hidx] = 1.0

                continue

            hz_dist = float("inf")
            if hazard is not None:
                px, py = pred_idx % self.env_size, pred_idx // self.env_size
                hz_dist = math.hypot(px - hx, py - hy)

            if unit.is_hazard_confirmed and hz_dist > 3.0:
                escape_bonus = policy.hazard_escape_bonus()
                if escape_bonus > 0.0:
                    unit.energy += escape_bonus
                    unit.meta.record(action=pred_idx, reward=+escape_bonus)
                unit.is_hazard_confirmed = False
                unit.last_reward_step = self.current_step
                unit.last_action_rewarded = True

            x_res, y_res = cur_idx % self.env_size, cur_idx // self.env_size
            if (x_res, y_res) not in self.env.resources:
                continue

            if (unit.last_rewarded_target_idx != cur_idx) and (is_res_hit or is_res_near):
                base_r = policy.resource_base_reward(res_dist)
                share_ratio = policy.resource_upstream_share()
                if base_r > 0.0:
                    unit.energy += base_r
                    unit.meta.record(action=cur_idx, reward=+base_r)
                    for p in upstream_processors:
                        share = base_r * share_ratio
                        if share > 0.0:
                            p.energy += share
                            p.meta.record(action=cur_idx, reward=+share)

                if is_res_hit:
                    hit_bonus = policy.resource_hit_bonus()
                    unit.energy += hit_bonus
                    unit.meta.record(action=cur_idx, reward=+hit_bonus)
                    if self.env.resources[(x_res, y_res)] > 0:
                        self.env.resources[(x_res, y_res)] -= 1
                        if self.env.resources[(x_res, y_res)] == 0:
                            del self.env.resources[(x_res, y_res)]
                    self.env.update_known_cell((x_res, y_res))
                    self.removed_resources_count += 1
                    # 2) 因资源奖励，额外删一个最远的坑
                    if self.env.hazards:
                        # 最远距离可以根据当前 (x_res,y_res) 算，也可以随意取
                        far = max(
                            self.env.hazards.keys(),
                            key=lambda p: (p[0] - x_res) ** 2 + (p[1] - y_res) ** 2
                        )
                        del self.env.hazards[far]
                        self.env.update_known_cell(far)
                        self.removed_hazards_by_reward += 1
                    for p in upstream_processors:
                        share = hit_bonus * share_ratio
                        if share > 0.0:
                            p.energy += share
                            p.meta.record(action=cur_idx, reward=+share)
                    if getattr(unit, "is_permanent_explorer", False):
                        continue  # 永久探索者不应被重设为资源目标

                    # —— 吃完资源后，重新选最近的资源&惩罚点 —— #
                    next_res = self.env.get_nearest_known_resource(unit.get_position())
                    if next_res is not None:
                        ridx = next_res[1] * self.env_size + next_res[0]
                        unit.goal_vec[0].zero_()
                        unit.goal_vec[0, ridx] = 1.0
                    next_hz = self.env.get_nearest_known_danger(unit.get_position())
                    if next_hz is not None:
                        hidx = next_hz[1] * self.env_size + next_hz[0]
                        unit.goal_vec[1].zero_()
                        unit.goal_vec[1, hidx] = 1.0

                    unit.last_rewarded_target_idx = None
                    unit.linger_steps = 0
                    unit.last_reward_amount = 0.0

                unit.last_rewarded_target_idx = cur_idx
                unit.last_reward_amount = base_r
                unit.linger_steps = 0
                unit.last_reward_step = self.current_step
                unit.last_action_rewarded = True
                continue

            if (unit.last_rewarded_target_idx == cur_idx) and res_dist <= 2.0 and self.current_step >= 1500:
                unit.linger_steps = min(unit.linger_steps + 1, 20)
                if unit.linger_steps > 3:
                    linger_penalty = policy.linger_penalty()
                    if linger_penalty > 0.0:
                        unit.energy -= linger_penalty
                        unit.meta.record(action=cur_idx, reward=-linger_penalty)
                continue

            if (unit.last_rewarded_target_idx == cur_idx) and res_dist > 4.0:
                unit.energy -= unit.last_reward_amount
                unit.meta.record(action=cur_idx, reward=-unit.last_reward_amount)
                unit.last_rewarded_target_idx = None
                unit.linger_steps = 0
                unit.last_reward_amount = 0.0

            if len(action_indices) >= 3:
                most_common = max(set(action_indices), key=action_indices.count)
                if action_indices.count(most_common) > len(action_indices) * 0.9:
                    penalty = policy.diversity_penalty()
                    if penalty > 0.0:
                        unit.energy -= penalty
                        unit.meta.record(action="diversity_penalty", reward=-penalty)
                elif len(set(action_indices)) > len(action_indices) * 0.6:
                    bonus = policy.diversity_bonus()
                    if bonus > 0.0:
                        unit.energy += bonus
                        unit.meta.record(action="diversity_penalty", reward=+bonus)

            if self.current_step > 1500:
                inactive_steps = self.current_step - unit.last_reward_step
                if inactive_steps > decay_threshold:
                    unit.energy -= decay_amount
                    unit.meta.record(action="round", reward=-decay_amount)

                if (hasattr(unit, "output_positions")
                        and len(unit.output_positions) >= 10
                        and self.current_step % 10 == 0):
                    start = unit.output_positions[0]
                    end = unit.output_positions[-1]
                    manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
                    penalty = policy.movement_penalty(manhattan)
                    if penalty > 0.0:
                        unit.energy -= penalty
                        unit.meta.record(action="move less", reward=-penalty)

        if hasattr(policy, "update_environment_feedback"):
            policy.update_environment_feedback(
                reward_hits=self.env.reward_hit_count,
                danger_hits=self.env.danger_hit_count,
                processor_count=self.processor_count,
                emitter_count=self.emitter_count,
                exploration_ratio=getattr(self.env, "_last_cycle_exploration_ratio", None),
                last_cycle_success=getattr(self.env, "_last_cycle_success_ratio", None),
            )

    def _expand_environment_curriculum(self):
        if self.current_step >= 1000 and self.current_step % 1000 == 0:
            old_size = self.env_size
            self.env_size = min(self.env_size + 5, 500)
            self.env.resize(self.env_size)

            self.processor_hidden_size = self.env_size * self.env_size * INPUT_CHANNELS
            full_dim = self.processor_hidden_size + self.env_size * self.env_size * 2
            self.rl_agent.resize_state_dim(full_dim)

            if RF.use_shared_tx:
                self._init_shared_tx()

            new_target = (random.randint(0, self.env_size - 1), random.randint(0, self.env_size - 1))
            self.task = TaskInjector(target_position=new_target)
            self.target_vector = self.task.encode_goal(self.env_size).to(self.device)

            logger.info(
                f"[Curriculum升级] 第 {self.current_step} 步：环境大小 {old_size}x{old_size} → {self.env_size}x{self.env_size}，新目标 {new_target}")

            for u in self.units:
                u.env_size = self.env_size
            self._target_buf = self._target_buf.new_zeros((2, self.env_size * self.env_size))
            for u in self.units:
                u.memory_buffer.clear()


    def trim_weak_memories(self):
        """环境发生变化时，清除所有细胞记忆池中的一半最弱记忆"""
        if (
                self.units
                and hasattr(self.units[0], "local_memory_pool")
                and len(self.units[0].local_memory_pool) < int(0.75 * self.units[0].memory_pool_limit)
        ):
            return  # ✅ 未达触发条件，直接退出

        for unit in self.units:
            if hasattr(unit, "local_memory_pool") and unit.local_memory_pool:
                pool = unit.local_memory_pool
                pool.sort(key=lambda m: m["score"])
                half = len(pool) // 2
                del pool[:half]

    def is_current_target_hazard(self) -> bool:
        """判断当前目标是否是陷阱（指向陷阱的 one-hot）"""
        index = torch.argmax(self.target_vector[1]).item()
        x, y = index % self.env_size, index // self.env_size
        return (x, y) in self.env.hazards

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
        if tensor.shape[-1] == goal_dim:  # ⚡ 已经是一维 env²，直接返回
            return tensor
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
        return torch.nn.functional.pad(tensor, pad, value=0).to(tensor.device)

    # ------------------------------------------------------------------

    def _align_to_hidden_dim(self, tensor: torch.Tensor, *, target: int | None = None) -> torch.Tensor:
        """
        把输入张量对齐到 transformer 期望的隐藏维度（processor_hidden_size）。
        """
        if tensor.dim() > 1:
            tensor = tensor.view(-1)
        dim = tensor.numel()
        target_dim = target if target is not None else self.processor_hidden_size
        if dim == target_dim:
            return tensor
        if dim > target_dim:
            return tensor[:target_dim]
        pad = (0, target_dim - dim)
        return torch.nn.functional.pad(tensor, pad, value=0.0)

    def _role_outgoing_strength(self, unit: CogUnit) -> float:
        """计算单元的对外连接强度总和（用于作为加权因子）。"""
        strengths = self.connections.get(unit.id, {})
        if not strengths:
            return 0.0
        return float(sum(strengths.values()))

    def _role_outgoing_degree(self, unit: CogUnit) -> int:
        return len(self.connections.get(unit.id, {}))

    def _normalize_weight_tensor(self, values: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        total = values.sum()
        if not torch.isfinite(total) or total.abs() < eps:
            return torch.full_like(values, 1.0 / max(values.numel(), 1))
        return values / total

    def _summarize_topology_tensor(self, tensor: torch.Tensor) -> Dict[str, float]:
        if tensor.numel() == 0:
            return {"mean": 0.0, "abs_mean": 0.0, "var": 0.0}
        det = tensor.detach()
        mean = det.mean()
        abs_mean = det.abs().mean()
        var = det.var(unbiased=False)
        return {
            "mean": float(mean.item()),
            "abs_mean": float(abs_mean.item()),
            "var": float(var.item()),
        }

    def _record_topology_snapshot(
        self,
        role: str,
        *,
        weighted_mean: torch.Tensor,
        degree_mean: torch.Tensor,
        plain_mean: torch.Tensor,
        spread: torch.Tensor,
        std: torch.Tensor,
        central_mean: torch.Tensor,
        count: int,
    ) -> None:
        self._role_topology_snapshots[role] = {
            "count": {"mean": float(count), "abs_mean": float(count), "var": 0.0},
            "weighted_mean": self._summarize_topology_tensor(weighted_mean),
            "degree_mean": self._summarize_topology_tensor(degree_mean),
            "plain_mean": self._summarize_topology_tensor(plain_mean),
            "spread": self._summarize_topology_tensor(spread),
            "std": self._summarize_topology_tensor(std),
            "central_mean": self._summarize_topology_tensor(central_mean),
        }

    def _make_metabolic_bucket(self) -> Dict[str, float]:
        return {
            "ema_var": 0.0,
            "ema_adjusted_var": 0.0,
            "ema_rms": 0.0,
            "ema_energy": 0.0,
            "ema_drain": 0.0,
            "ema_scalar": 0.0,
            "ema_dim_scale": 0.0,
            "ema_feature_dim": 0.0,
            "last_var": 0.0,
            "last_adjusted_var": 0.0,
            "last_rms": 0.0,
            "last_energy": 0.0,
            "last_drain": 0.0,
            "last_scalar": 0.0,
            "last_dim_scale": 0.0,
            "last_feature_dim": 0.0,
            "steps": 0,
        }

    def _update_metabolic_stats(
        self,
        role: str,
        *,
        input_var: float,
        adjusted_var: float,
        input_rms: float,
        energy: float,
        drain: float,
        scalar: float,
        dim_scale: float,
        feature_dim: float,
    ) -> None:
        stats = self._role_metabolic_stats.setdefault(role, self._make_metabolic_bucket())
        beta = self._metabolic_ema_beta

        if stats["steps"] == 0:
            stats["ema_var"] = input_var
            stats["ema_adjusted_var"] = adjusted_var
            stats["ema_rms"] = input_rms
            stats["ema_energy"] = energy
            stats["ema_drain"] = drain
            stats["ema_scalar"] = scalar
            stats["ema_dim_scale"] = dim_scale
            stats["ema_feature_dim"] = feature_dim
        else:
            inv = 1.0 - beta
            stats["ema_var"] = beta * stats["ema_var"] + inv * input_var
            stats["ema_adjusted_var"] = beta * stats["ema_adjusted_var"] + inv * adjusted_var
            stats["ema_rms"] = beta * stats["ema_rms"] + inv * input_rms
            stats["ema_energy"] = beta * stats["ema_energy"] + inv * energy
            stats["ema_drain"] = beta * stats["ema_drain"] + inv * drain
            stats["ema_scalar"] = beta * stats["ema_scalar"] + inv * scalar
            stats["ema_dim_scale"] = beta * stats["ema_dim_scale"] + inv * dim_scale
            stats["ema_feature_dim"] = beta * stats["ema_feature_dim"] + inv * feature_dim

        stats["last_var"] = input_var
        stats["last_adjusted_var"] = adjusted_var
        stats["last_rms"] = input_rms
        stats["last_energy"] = energy
        stats["last_drain"] = drain
        stats["last_scalar"] = scalar
        stats["last_dim_scale"] = dim_scale
        stats["last_feature_dim"] = feature_dim
        stats["steps"] += 1

    def get_role_metabolic_snapshot(self, role: str) -> Dict[str, float]:
        stats = self._role_metabolic_stats.get(role)
        if not stats:
            return {}
        snapshot: Dict[str, float] = {}
        for key, value in stats.items():
            if key == "steps":
                snapshot[key] = int(value)
            elif isinstance(value, (int, float)):
                snapshot[key] = float(value)
        return snapshot

    def _aggregate_role_outputs(
        self,
        role: str,
        stacked: torch.Tensor,
        strength_weights: torch.Tensor,
        degree_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据连接强度 / 度作为权重，并融合离散度信息。
        返回与单个单元输出同维度的聚合向量。
        """
        if stacked.dim() == 1:
            stacked = stacked.unsqueeze(0)

        dtype = stacked.dtype
        device = stacked.device

        s_w = self._normalize_weight_tensor(strength_weights.to(device=device, dtype=dtype))
        d_w = self._normalize_weight_tensor(degree_weights.to(device=device, dtype=dtype))

        weighted_mean = (s_w.unsqueeze(-1) * stacked).sum(dim=0)
        degree_mean = (d_w.unsqueeze(-1) * stacked).sum(dim=0)
        plain_mean = stacked.mean(dim=0)
        max_vals, _ = stacked.max(dim=0)
        min_vals, _ = stacked.min(dim=0)
        spread = max_vals - min_vals
        std = stacked.var(dim=0, unbiased=False).sqrt()

        central_mean = torch.stack([weighted_mean, degree_mean, plain_mean], dim=0).mean(dim=0)
        imbalance = (weighted_mean - degree_mean + weighted_mean - plain_mean) * (1.0 / 3.0)

        spread_denom = spread.abs().mean().clamp(min=1e-6)
        std_denom = std.abs().mean().clamp(min=1e-6)
        dispersion = 0.5 * (torch.tanh(spread / spread_denom) + torch.tanh(std / std_denom))

        summary = central_mean + 0.35 * dispersion + 0.15 * imbalance
        summary = self._align_to_hidden_dim(summary)

        self._record_topology_snapshot(
            role,
            weighted_mean=weighted_mean,
            degree_mean=degree_mean,
            plain_mean=plain_mean,
            spread=spread,
            std=std,
            central_mean=central_mean,
            count=stacked.shape[0],
        )
        return summary

    def get_role_topology_snapshot(self, role: str) -> Dict[str, Dict[str, float]]:
        """返回指定角色最近一次聚合时的拓扑统计（浮点摘要）。"""
        stats = self._role_topology_snapshots.get(role, {})
        return {name: values.copy() for name, values in stats.items()}


    def sensor_forward(self, env_state_np):
        """
        Args:
            env_state_np : np.ndarray 或 torch.Tensor (size=N)
        Returns:
            torch.Tensor (size = env_state_np.size) —— 作为 sensor 输出
        """
        dev = self.device  # ← 统一目标设备
        x = torch.as_tensor(env_state_np, dtype=torch.float32, device=dev).view(-1)
        x = self._align_to_hidden_dim(x)

        sensors = [u for u in self.units if u.get_role() == "sensor"]
        if not sensors:
            return x

        outputs = []
        strengths = []
        degrees = []
        active_ids = set()

        for sensor in sensors:
            sensor.update(x.unsqueeze(0))
            raw = sensor.get_output().detach().to(dev)
            aligned = self._align_to_hidden_dim(raw)
            outputs.append(aligned)
            strengths.append(self._role_outgoing_strength(sensor))
            degrees.append(float(self._role_outgoing_degree(sensor)))
            self._last_sensor_outputs[sensor.id] = aligned
            active_ids.add(sensor.id)

        # 清理已经死亡的 sensor 输出缓存
        self._last_sensor_outputs = {
            uid: self._last_sensor_outputs[uid]
            for uid in active_ids
        }

        stacked = torch.stack(outputs, dim=0)
        strength_tensor = stacked.new_tensor(strengths)
        degree_tensor = stacked.new_tensor(degrees)
        summary = self._aggregate_role_outputs(
            "sensor", stacked, strength_tensor, degree_tensor
        )
        summary = self._align_to_hidden_dim(summary)
        return summary


    def processor_forward(self, sensor_out):
        """
        Args:
            sensor_out : torch.Tensor 1-D
        Returns:
            torch.Tensor (size = self.processor_hidden_size)
        """
        """批量或逐个执行 processor.update()."""
        dev = self.device
        sensor_out = self._align_to_hidden_dim(sensor_out.to(dev))

        procs = [u for u in self.units if u.role == "processor"]
        if not procs:
            return sensor_out

        outputs = []
        strengths = []
        degrees = []
        active_ids = set()

        for proc in procs:
            incoming_vecs = []
            incoming_weights = []
            for sid, vec in self._last_sensor_outputs.items():
                if proc.id in self.connections.get(sid, {}):
                    incoming_vecs.append(vec.to(dev))
                    incoming_weights.append(self.connections[sid][proc.id])

            if incoming_vecs:
                stacked_in = torch.stack(incoming_vecs, dim=0)
                weight_tensor = stacked_in.new_tensor(incoming_weights)
                weight_tensor = self._normalize_weight_tensor(weight_tensor)
                proc_input = (weight_tensor.unsqueeze(-1) * stacked_in).sum(dim=0)
            else:
                proc_input = sensor_out

            proc.update(proc_input.unsqueeze(0))
            raw = proc.get_output().detach().to(dev)
            aligned = self._align_to_hidden_dim(raw)
            outputs.append(aligned)
            strengths.append(self._role_outgoing_strength(proc))
            degrees.append(float(self._role_outgoing_degree(proc)))
            self._last_processor_outputs[proc.id] = aligned
            active_ids.add(proc.id)

        self._last_processor_outputs = {
            uid: self._last_processor_outputs[uid]
            for uid in active_ids
        }

        stacked = torch.stack(outputs, dim=0)
        strength_tensor = stacked.new_tensor(strengths)
        degree_tensor = stacked.new_tensor(degrees)
        summary = self._aggregate_role_outputs(
            "processor", stacked, strength_tensor, degree_tensor
        )
        summary = self._align_to_hidden_dim(summary)
        return summary


    def emitter_forward(self, processor_out):
        """
        把 processor_out 递给所有 emitter 做一次更新；
        不要求返回值（若你想调试，可 return 平均输出）。
        """
        dev = self.device
        processor_vec = self._align_to_hidden_dim(processor_out.to(dev))
        emitters = [u for u in self.units if u.role == "emitter"]
        if not emitters:
            return

        emitter_outputs = []
        emitter_strengths = []
        emitter_degrees = []

        for emitter in emitters:
            incoming_vecs = []
            incoming_weights = []
            for pid, vec in self._last_processor_outputs.items():
                if emitter.id in self.connections.get(pid, {}):
                    incoming_vecs.append(vec.to(dev))
                    incoming_weights.append(self.connections[pid][emitter.id])

            if incoming_vecs:
                stacked_in = torch.stack(incoming_vecs, dim=0)
                weight_tensor = stacked_in.new_tensor(incoming_weights)
                weight_tensor = self._normalize_weight_tensor(weight_tensor)
                emitter_input = (weight_tensor.unsqueeze(-1) * stacked_in).sum(dim=0)
            else:
                emitter_input = processor_vec

            centered = emitter_input - emitter_input.mean()
            rms = centered.pow(2).mean().sqrt()
            rms_value = float(rms.item()) if torch.isfinite(rms).item() else 0.0
            cap = self._emitter_input_rms_cap
            if rms_value > cap:
                scale = cap / (rms_value + 1e-6)
                emitter_input = centered * scale + emitter_input.mean()
            emitter_input = emitter_input.clamp(
                min=-self._emitter_input_value_cap, max=self._emitter_input_value_cap
            )

            emitter.update(emitter_input.unsqueeze(0))
            last = emitter.get_output().detach()
            emitter.last_output = last
            aligned = self._align_to_hidden_dim(last.to(dev))
            emitter_outputs.append(aligned)
            emitter_strengths.append(self._role_outgoing_strength(emitter))
            emitter_degrees.append(float(self._role_outgoing_degree(emitter)))

        if emitter_outputs:
            stacked = torch.stack(emitter_outputs, dim=0)
            strength_tensor = stacked.new_tensor(emitter_strengths)
            degree_tensor = stacked.new_tensor(emitter_degrees)
            # 仅为更新拓扑快照，emitters 本身不需要返回摘要
            self._aggregate_role_outputs("emitter", stacked, strength_tensor, degree_tensor)

    def _rebuild_free_positions(self):
        """一次性扫描所有格子，生成安全出生点列表"""
        occupied = set(self.env.resources.keys()) | set(self.env.hazards.keys())
        size = self.env_size
        self.free_positions = [
            (x, y)
            for x in range(size)
            for y in range(size)
            if (x, y) not in occupied
        ]

    def summary(self):
        # # 打印当前图结构概况
        #
        # logger.debug(f"[图结构] 当前单元数: {len(self.units)}")
        # for unit in self.units:
        #     logger.debug(f" - {unit} → 连接数: {len(self.connections[unit.id])}")
        return

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
            stacked = torch.cat(aligned, dim=0).to(self.device, non_blocking=True)      # [N, goal_dim]
            logger.debug(
                "[输出检查] Emitter 对齐后均值(前5) : %s",
                stacked.mean(dim=0)[:5]
            )

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

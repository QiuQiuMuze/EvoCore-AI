import torch
import math

from contextlib import nullcontext
from collections import deque, Counter
from coggraph import CogGraph
import copy
from env_net import GridSecurityEnv
from torch.nn.functional import conv2d
from emitter_actions import EmitterActions
from memory_net_unit import MemoryNetUnit
from processor_immune import ImmuneProcessor
from special_emitter import SpecialEmitter
from emitter_actions import ACTION_BLOCK, ACTION_QUARANTINE, ACTION_HACK_DEFENSE
from adaptive_guidance import (
    AdaptiveGuidanceModule,
    ASSIGNMENT_LEARNED,
    ASSIGNMENT_SELF,
)
from config_runtime import RF
from typing import List
import random
from per_memory import PrioritizedReplayBuffer
from meta_trainer import MetaTrainer
from types import MethodType
import torch.nn.functional as F
import torch.optim as optim
import types, torch
import torch.nn as nn
from models.transformer_policy import TransformerPolicyNetwork
from env import logger
from torch.optim import AdamW
from CogUnit import CogUnit
from sensor_module import SensorMLP

from .cells import (
    SensorCellController,
    ProcessorCellController,
    EmitterCellController,
)

from .constants import (
    HIT_THRESH,
    MAX_CONNECTIONS,
    N_GOAL_CHANNELS,
    FLASH_ATTN_AVAILABLE,
    MIN_PATROL_DIST,
    HIT_BONUS,
    MISS_PENALTY,
    LEARNED_FACTOR,
)

class ImmuneCogGraph(CogGraph):
    """
    最大化整合网络安全免疫系统：
    - 病毒环境：GridSecurityEnv
    - 免疫识别：ImmuneProcessor
    - 抗体记忆：MemoryNetUnit
    - 特攻单元：SpecialEmitter
    - 继承原 CogGraph 自组织、好奇探索、代谢、分裂、死亡、奖励/惩罚、结构维护
    """

    def __init__(self, rl_agent, device="cpu", env=None):
        self.device = device
        self.target_weights = torch.tensor([1.0, 1.0, 1.0], device=self.device)  # [探索, 感染, 提权]

        # 环境替换
        if env is None:
            env = GridSecurityEnv(size=10, device=device)
        self.last_flat_idx = 0
        # 自适应调度模块：通过学习选择目标而非硬编码指导
        super().__init__(rl_agent, device=device, env=env)

        # --- 1) 定义 hack channels 数量 & 总输入通道数 ---
        self._HACK_CHANNELS = len(self.env.attack_types)      # T 个 hack 类型
        # —— 动态获取环境状态通道数 —— #
        state_tensor = env.get_state_tensor()
        self._STATE_CHANNELS = state_tensor.shape[0]  # C
        self._INPUT_CHANNELS = (
                self._STATE_CHANNELS
                + self._HACK_CHANNELS
                + N_GOAL_CHANNELS
        )

        # —— 初始化 cleared_positions —— #
        for u in self.units:
            if u.role == "emitter":
                u.cleared_positions = set()
        self.env: GridSecurityEnv = env
        self.emitter_hidden_size  = self.processor_hidden_size
        self.antibody_success_count = 0
        self.current_hazard_xy: tuple[int, int] | None = None
        self.goal_vec = None

        self._rng = torch.Generator(device=device)

        # Cell controllers encapsulate role-specific logic
        self.sensor_controller = SensorCellController(self)
        self.processor_controller = ProcessorCellController(self)
        self.emitter_controller = EmitterCellController(self)

        # ——— 在这里 env 已经创建，self.env.size 已经可用 ———
        H, W = self.env.size, self.env.size

        # # ① 实例化可学习的 SensorMLP
        # self.learnable_sensor = SensorMLP(H, W).to(self.device)
        # # ② 为它单独创建一个优化器
        # self.sensor_optimizer = torch.optim.Adam(
        #     self.learnable_sensor.parameters(), lr=1e-3
        # )

        # self._rng.manual_seed(42)  # 任选一个固定种子，方便复现
        # 用当前 env.size 构造一次全局网络
        seq_len = self.env.size * self.env.size
        self._build_global_nets(seq_len)
        self._dilate = nn.MaxPool2d(kernel_size=9, stride=1, padding=4).to(self.device)

        # —— 1.5) 单元网络共享开关 ——
        # 只有当 FLASH_ATTN_AVAILABLE=True 时，才所有单元共用同一套 processor_net/emitter_net。
        self.use_shared_unit_nets = FLASH_ATTN_AVAILABLE

        # —— 1.6) 如果不共享，就给每个 processor/emitter 拷贝一份 ——
        if not self.use_shared_unit_nets:
            for u in self.units:
                if u.role == "processor":
                    u.processor_net = copy.deepcopy(self.processor_net)
                elif u.role == "emitter":
                    u.emitter_net  = copy.deepcopy(self.emitter_net)

        self.known_infections = set()
        self.known_hacks      = set()

        self.detect_thresh = 0.12  # 初始阈值，越大=越保守

        self._last_seq_len = seq_len
        self.antibody_failure_count = 0
        self.env.bind_units_reference(self.units)
        self.visit_age_map = torch.zeros_like(
            self.env.infected_map, dtype=torch.float16, device=self.device
        )
        self._update_target_vector()
        self.adaptive_guidance = AdaptiveGuidanceModule(self.env.size, device=self.device)
        self.last_report_stage = None
        self.hack_kill_stats = {ASSIGNMENT_SELF: 0, ASSIGNMENT_LEARNED: 0}
        # —— 新增：连续无感染计数器 —— #
        self.zero_infection_counter = 0
        # 追踪各黑客类型的击杀数
        self.hack_kill_stats_by_type = {}
        # --- 可学习的黑客类型权重 embedding ---
        # 利用 env.attack_types 列表初始化 mapping
        self.hack_types = list(self.env.hack_types)
        self.hack_type_to_idx = {t: i for i, t in enumerate(self.hack_types)}
        # 每个类型一个可训练的标量 weight
        self.hack_type_embedding = nn.Embedding(len(self.hack_types), 1).to(self.device)

        # 追踪各病毒类型的击杀数（如果只有一种病毒，可统一用 'virus' 作为 key）
        self.virus_kill_stats_by_type = {}
        self.env.resources = {}  # 空字典，保证 .resources 存在
        self.env.hazards = {}  # handle_energy_overflow 里可能也用到 .hazards
        self.punished_map = torch.zeros_like(self.env.infected_duration_map, dtype=torch.bool)
        self.prev_infected_map = torch.zeros_like(self.env.infected_map)
        self.prev_priv = torch.zeros_like(self.env.privilege_level)
        self.prev_vuln = torch.zeros_like(self.env.vulnerability)
        self.prev_fail = torch.zeros_like(self.env.login_failures)
        self.kill_stats = {ASSIGNMENT_SELF: 0, ASSIGNMENT_LEARNED: 0, "last_reset": 0}
        seq_len = self.env.size * self.env.size

        # ─────────── 静息模式相关字段 ─────────── #
        # 统计连续多少步内环境没有任何感染 & 没有任何黑客
        self.no_threat_steps = 0
        # 连续多少步无威胁就进入静息：可以调整，比如设 100
        self.static_threshold = 100
        # 标记是否在静息模式下
        self.static_mode = False
        # 记录进入静息模式时的 step，用于保护单元 age 不增长
        self.static_mode_entry_step = None
        # 记录退出静息模式时的 step，用于短期保护 CogUnit 不被误杀
        self.static_mode_exit_step = -1
        self._orig_age = {}

        # ──────────────────────────────────────── #
        # —— 使用带 n_step 的 PER ——（n=3, γ=0.99）
        self.gamma = 0.99
        # 专门给 RL 用
        self.rl_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, n_step=3, gamma=self.gamma)
        # 专门给 antibody 用（如果你也想用 PER）
        self.anti_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, n_step=1, gamma=1.0)

        self.transformer = TransformerPolicyNetwork(
            input_dim=self._STATE_CHANNELS,
            num_actions=seq_len,
            d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
            max_seq_len=seq_len,
            use_action_noise=False,
            use_flash_attn=FLASH_ATTN_AVAILABLE
        ).to(self.device)

        # —— 2) replay_head 接收 emitter 输出的 flat logits（seq_len 维）—— #
        seq_len = self.env.size * self.env.size
        self.replay_head = nn.Linear(seq_len, 2).to(self.device)

        # 防御接口
        self.emitter_actions = EmitterActions(self.env)
        self.long_virus_tracker = torch.zeros_like(self.env.infected_duration_map, dtype=torch.bool)

        # —— 3) 用 Transformer 作为 feature_extractor 构造 memory —— #
        self.memory = MemoryNetUnit(
            maxlen=1000,
            device=device,
            use_faiss=False,
            feature_extractor=self.transformer  # <— 传给 memory
        )

        # —— 4) 构造 ImmuneProcessor —— #
        self.immune_processor = ImmuneProcessor(
            memory_pool=self.memory,
            feature_extractor=self.transformer,
            similarity_threshold=0.5
        )
        # —— 抗体分类头 & 优化器 —— #
        feat_dim = self.transformer.fc_out.in_features
        self.immune_clf = nn.Linear(feat_dim, 1).to(self.device)
        self.immune_opt = optim.Adam(self.immune_clf.parameters(), lr=1e-4)

        # 特攻单元
        for atk in self.env.attack_types:
            sp = SpecialEmitter(
                unit_id=str(len(self.units)),
                attack_type=atk,
                strategy={"type":"quarantine"},
                env=self.env,
                clone_threshold=5
            )
            sp.cleared_positions = set()

            if not self.use_shared_unit_nets:
                # 只给这个新加的特攻单元拷一份网络
                sp.processor_net = copy.deepcopy(self.processor_net)
                sp.emitter_net  = copy.deepcopy(self.emitter_net)

            self.prev_dur = self.env.infected_duration_map.clone()
            sp.cleared_positions = set()
            self.add_unit(sp)

        # —— 新增：GoalNet —— #
        # 输入维度：seq_len（= env.size * env.size），输出维度也是 seq_len
        # 代表“当前有哪些格子是威胁（感染/提权），学会去哪个格子打击”
        seq_len = self.env.size * self.env.size
        self.goal_net = nn.Sequential(
            nn.Linear(seq_len, seq_len),    # 也可加激活层，或者多层网络；这里只做示例
            # 注：在前向时做 softmax，我们会在 _assign_emitter_goal 里处理
        ).to(self.device)

        # —— 若要给 GoalNet 加一个优化器 —— #
        # 可以在已有的 policy_optimizer 里加，也可以单独新建一个
        self.goal_optimizer = optim.Adam(self.goal_net.parameters(), lr=5e-4)

        # —— 记得把它加入整体调度器（可选） —— #
        # 例如：
        # self.scheduler_goal = optim.lr_scheduler.StepLR(self.goal_optimizer, step_size=200, gamma=0.9)


        # —— 优化 Transformer —— #
        self.optimizer = optim.Adam(
            list(self.transformer.parameters()) +
            list(self.replay_head.parameters()),
            lr=5e-4
        )
        # —— **1) 全局 policy (actor) —— #
        # 包含 hack_type_embedding 参数，使其可被 policy 更新
        self.policy_optimizer = AdamW(
            # list(self.learnable_sensor.parameters()) +
            list(self.sensor_net.parameters()) +
            list(self.processor_net.parameters()) +
            list(self.emitter_net.parameters()) +
            list(self.hack_type_embedding.parameters()) +
            list(self.goal_net.parameters()),
            lr=5e-4,
            weight_decay=1e-2
        )


        # —— **2) 新增 baseline / critic —— #
        # full_state_dim = (env_state_channels + goal_channels) * grid_area
        size2 = self.env.size * self.env.size
        D_env = size2 * self._STATE_CHANNELS
        D_hack = size2 * self._HACK_CHANNELS
        D_goal = size2 * N_GOAL_CHANNELS
        full_state_dim = D_env + D_hack + D_goal
        self.value_head = nn.Linear(full_state_dim, 1).to(self.device)
        self.value_optimizer = optim.Adam(
            self.value_head.parameters(), lr=2e-4
        )


        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,  # 作用对象
            step_size=200,
            gamma=0.9  # 学习率乘 0.8
        )
        # ===================================================================
        self._rebuild_free_positions()            # 保证有 free_positions
        for u in self.units:
            if u.role == "emitter":
                if not hasattr(u, "position"):
                    idx = torch.randint(0, len(self.free_positions), (1,),
                        generator = self._rng, device = self.device).item()
                    u.position = self.free_positions[idx]
                u.output_positions = deque(maxlen=20)   # 轨迹用于惩罚

        # --------------------------------------------------------------
        # --- 元学习 trainer

        # --- 同步状态：只做一次分配＋编译 （删掉上面所有与 _full_state_buf/_D_*/_hack_onehot/_last_hack_keys 重复的初始化） ---
        self._last_seq_len = self.env.size * self.env.size
        self._last_hack_keys = set(self.env.hacks.keys())

        size2 = self.env.size * self.env.size
        self._D_env = size2 * self._STATE_CHANNELS
        self._D_hack = size2 * self._HACK_CHANNELS
        self._D_goal = size2 * N_GOAL_CHANNELS

        self._S2 = size2
        # 预分配一次性缓冲区
        self._full_state_buf = torch.empty(1, self._D_env + self._D_hack + self._D_goal,
                                           device=self.device)
        self._hack_onehot = torch.zeros(1, self._D_hack, device=self.device)

        # 预编译 sensor_net（只做一次），并预定义膨胀卷积
        try:
            self._sensor_net = torch.compile(self.sensor_net,
                                             fullgraph=False, dynamic=True)
        except:
            self._sensor_net = self.sensor_net

        # 熵正则系数，用于 entropy bonus
        self.entropy_coef = max(0.01, 1.0 / (1 + self.current_step / 2000))

        self.meta_trainer = MetaTrainer(self.immune_processor, device=self.device)
        def _safe_compute(self, inp, out):
            if inp is None or out is None:
                return torch.tensor(0.0, device="cpu")

            if inp.shape[-1] != out.shape[-1]:
                L = min(inp.shape[-1], out.shape[-1])
                inp = inp[..., :L]
                out = out[..., :L]
            return torch.mean((inp - out) ** 2)

        for u in self.units:
            if hasattr(u, "compute_self_reward"):
                u.compute_self_reward = types.MethodType(_safe_compute, u)

        _orig_clone = CogUnit.clone  # 备份原方法

        def _clone_patched(self, *args, **kwargs):
            child = _orig_clone(self, *args, **kwargs)  # 这里 self 是实例

            src_l1 = getattr(self, "l1", None) or getattr(self, "linear1", None)
            tgt_l1 = getattr(child, "l1", None) or getattr(child, "linear1", None)
            if isinstance(src_l1, nn.Linear) and isinstance(tgt_l1, nn.Linear):
                need = src_l1.in_features
                if tgt_l1.in_features < need:
                    new_l1 = nn.Linear(need, tgt_l1.out_features,
                                       bias=tgt_l1.bias is not None).to(tgt_l1.weight.device)
                    # 仅拷贝已有权重
                    new_l1.weight.data[:, :tgt_l1.in_features].copy_(tgt_l1.weight.data)
                    if tgt_l1.bias is not None:
                        new_l1.bias.data.copy_(tgt_l1.bias.data)
                    # 真正替换
                    if hasattr(child, "l1"):
                        child.l1 = new_l1
                    else:
                        child.linear1 = new_l1
                    child.input_size = need
            return child

        CogUnit.clone = _clone_patched
        self._last_seq_len = self.env.size * self.env.size
        self._last_hack_keys = set(self.env.hacks.keys())


        # -----------------------------------------------------------

    def _should_evolve(self) -> bool:
        # 基于规模 + 病毒压力双阈值
        base = 80  # 最低间隔
        scale_factor = max(1, self.env.size / 10)
        pressure = (self.env.infected_map > 0).float().mean().item()  # 0~1
        interval = int(base * scale_factor / max(0.5, pressure))  # 压力越大，间隔越短
        return self.current_step % interval == 0

    def _build_global_nets(self, seq_len: int):
        """
        构建（或重建）CogGraph 的三条主干网络，并自动同步 hidden_size。
        这里把 emitter_net 输出维度从 seq_len → 3*seq_len，分别对应
        [MOVE, BLOCK, HACK_DEFENSE] × 每个格子。
        """
        # 新：用实例属性中计算好的总输入通道数
        D_in = seq_len * self._INPUT_CHANNELS

        # 1) sensor_net：D_in → processor_hidden_size
        self.sensor_net = nn.Sequential(
            nn.Linear(D_in, self.processor_hidden_size),
            nn.ReLU(),
        ).to(self.device)

        # 2) processor_net：processor_hidden_size → emitter_hidden_size
        self.processor_net = nn.Sequential(
            nn.Linear(self.sensor_net[0].out_features, self.emitter_hidden_size),
            nn.ReLU(),
        ).to(self.device)

        # 3) emitter_net：emitter_hidden_size → 3 * seq_len （三种动作 × 每个格子）
        self.emitter_net = nn.Sequential(
            nn.Linear(self.processor_net[0].out_features, 3 * seq_len),
        ).to(self.device)

        # —— 同步子类的 hidden_size 属性 —— #
        self.processor_hidden_size = self.sensor_net[0].out_features
        self.emitter_hidden_size = self.processor_net[0].out_features

        # 如果之前编译版存在，删掉
        if hasattr(self, "_compiled_sensor_net"):
            del self._compiled_sensor_net
        if hasattr(self, "_compiled_processor_net"):
            del self._compiled_processor_net

        # 重新编译（可选）
        try:
            self._sensor_net = torch.compile(self.sensor_net, fullgraph=False, dynamic=True)
        except:
            self._sensor_net = self.sensor_net

    import math
    import torch

    def _run_sensor_scans(self):
        """Delegate sensor scanning to the sensor controller."""
        return self.sensor_controller.run_scans()


    def _handle_reward_and_penalty(self):
        """
        重写版本：允许基于“病毒痕迹”直接退出静息模式，而不仅仅是 reward。
        """
        # —— 1. 原 reward 唤醒机制——
        found = any(
            getattr(u, "last_reward_step", None) == self.current_step
            for u in self.units if u.role == "emitter"
        )
        if found:
            self.steps_since_last_reward = 0
            if self.static_mode:
                self._exit_static_mode()
            return

        self.steps_since_last_reward += 1

        # —— 2. 病毒痕迹触发唤醒——
        tensor_state = self.env.get_state_tensor()
        infected_count = (tensor_state[0] > 0.04).sum().item()  # 通道0：是否感染
        if self.static_mode and infected_count >= 1:
            logger.warning(f"[静息唤醒] 来敌人啦，快干他！检测到病毒痕迹，感染点数 = {infected_count}")
            self._exit_static_mode()

    def _step_toward(self, start, goal):
        sx, sy = start
        gx, gy = goal
        if sx == gx and sy == gy:
            return (sx, sy)
        if abs(gx - sx) >= abs(gy - sy):  # 先走 x 轴
            sx += 1 if gx > sx else -1
        else:
            sy += 1 if gy > sy else -1
        return (sx, sy)

    def _on_env_resize(self):
        """
        只要 env.size 改变，就调用此函数。它会：
          1) 把 sensor_net、emitter_net、replay_head、value_head、transformer.head 等
             所有跟 grid_size * grid_size 相关的层都做「增量扩张／拷贝旧权重」；
          2) 重建 full_state_buf、hack_onehot、visit_age_map 这些 buffer；
          3) 刷新一次 target_vector (未访问图 / 感染图 / 提权图)；
        调用完毕之后，所有部件都已经是「新尺寸」了，后续 step() 里再也不需要检查任何维度。
        """
        new_seq = self.env.size * self.env.size
        if new_seq == self._last_seq_len:
            return

        ############################
        #（1）sensor_net 第一层 (Linear: old_in → hidden) 要扩到 new_in → hidden
        old_layer = self.sensor_net[0]  # 假设 sensor_net[0] = nn.Linear(old_in, old_out)
        old_in, old_out = old_layer.in_features, old_layer.out_features
        new_in = new_seq * self._INPUT_CHANNELS  # _INPUT_CHANNELS = STATE_CHANNELS + HACK_CHANNELS + N_GOAL_CHANNELS
        if new_in > old_in:
            new_sensor = nn.Linear(new_in, old_out, bias=(old_layer.bias is not None)).to(self.device)
            with torch.no_grad():
                # 先把旧权重前 old_in 列拷过去
                new_sensor.weight[:, :old_in].copy_(old_layer.weight)
                if old_layer.bias is not None:
                    new_sensor.bias.copy_(old_layer.bias)
            self.sensor_net[0] = new_sensor

        ############################
        # —— （2）emitter_net 第一层要从 hidden → (3*old_seq) 扩到 hidden → (3*new_seq) —— #
        emit_layer = self.emitter_net[0]
        old_in_e, old_out_e = emit_layer.in_features, emit_layer.out_features
        # 旧的 seq_len 大小 = old_out_e // 3
        old_seq = old_out_e // 3
        # 如果 new_seq > old_seq，说明要扩容
        if new_seq > old_seq:
            new_out = 3 * new_seq
            new_emit = nn.Linear(old_in_e, new_out, bias=(emit_layer.bias is not None)).to(self.device)
            with torch.no_grad():
                # 把原先 3*old_seq 行的权重全部 copy 过去
                new_emit.weight[:old_out_e, :].copy_(emit_layer.weight)
                if emit_layer.bias is not None:
                    new_emit.bias[:old_out_e].copy_(emit_layer.bias)
            self.emitter_net[0] = new_emit

        ############################
        #（3）replay_head (Linear: old_seq → 2) 扩到 new_seq → 2
        old_r = self.replay_head
        old_in_r, out_r = old_r.in_features, old_r.out_features  # out_r == 2
        if new_seq > old_in_r:
            new_r = nn.Linear(new_seq, out_r, bias=(old_r.bias is not None)).to(self.device)
            with torch.no_grad():
                new_r.weight[:, :old_in_r].copy_(old_r.weight)
                if old_r.bias is not None:
                    new_r.bias.copy_(old_r.bias)
            self.replay_head = new_r

        ############################
        #（4）value_head (Linear: old_full → 1) 扩到 new_full → 1
        old_v = self.value_head
        old_in_v, out_v = old_v.in_features, old_v.out_features  # out_v = 1
        new_full = new_seq * (self._STATE_CHANNELS + self._HACK_CHANNELS + N_GOAL_CHANNELS)
        if new_full > old_in_v:
            new_v = nn.Linear(new_full, out_v, bias=(old_v.bias is not None)).to(self.device)
            with torch.no_grad():
                new_v.weight[:, :old_in_v].copy_(old_v.weight)
                if old_v.bias is not None:
                    new_v.bias.copy_(old_v.bias)
            self.value_head = new_v

        ############################
        #（5）TransformerPolicyNetwork 自带的 "resize_head"，把内部最后一层 (d_model→old_seq) 扩到 (d_model→new_seq)
        self.transformer.resize_head(new_seq)

        # —— (X) 对 GoalNet 做扩容 —— #
        # 先把旧的线性层提取出来
        old_g = self.goal_net[0]  # 假设 goal_net = nn.Sequential(nn.Linear(old_in, old_out))
        old_in_g, old_out_g = old_g.in_features, old_g.out_features
        # old_in_g == old_out_g == old_seq_len

        # 如果 new_seq > old_seq_len，就新建一个 new_linear，再把旧权重 copy 过去
        if new_seq > old_out_g:
            new_g = nn.Linear(new_seq, new_seq, bias=(old_g.bias is not None)).to(self.device)
            with torch.no_grad():
                # 拷贝旧的 weights
                # 只拷贝前 old_out_g 行、old_out_g 列
                new_g.weight[:old_out_g, :old_out_g].copy_(old_g.weight)
                if old_g.bias is not None:
                    new_g.bias[:old_out_g].copy_(old_g.bias)

            self.goal_net = nn.Sequential(new_g).to(self.device)

            # —— 同步调整优化器 —— #
            # 如果你把 goal_net 放到单独的 goal_optimizer，需要把新的参数加进来
            for g in self.goal_optimizer.param_groups:
                g['params'] = list(self.goal_net.parameters())
            # 如果你用的是 policy_optimizer，也要 update param_groups，否则会漏掉新权重
            for g in self.policy_optimizer.param_groups:
                # 将新的 goal_net.parameters() 并入
                g['params'] = [p for p in g['params'] if p is not old_g.weight and p is not old_g.bias] + list(
                    self.goal_net.parameters())

        # —— (X) 记得更新 self._last_seq_len —— #
        self._last_seq_len = new_seq

        ############################
        #（6）如果不使用“共享子网”的话，还要把每个子单元的 processor_net / emitter_net 也用 deepcopy 扩一下
        if not self.use_shared_unit_nets:
            for u in self.units:
                if u.role == "processor":
                    u.processor_net = copy.deepcopy(self.processor_net)
                elif u.role == "emitter":
                    u.emitter_net = copy.deepcopy(self.emitter_net)

        ############################
        #（7）重建 buffer：_full_state_buf、_hack_onehot、visit_age_map
        D_env  = new_seq * self._STATE_CHANNELS
        D_hack = new_seq * self._HACK_CHANNELS
        D_goal = new_seq * N_GOAL_CHANNELS
        total  = D_env + D_hack + D_goal

        self._full_state_buf = torch.empty(1, total, device=self.device)
        self._hack_onehot    = torch.zeros(1, D_hack, device=self.device)
        # 同步 visit_age_map 至新的方格大小
        self.visit_age_map   = torch.zeros_like(self.env.infected_map, dtype=torch.float16, device=self.device)

        # 记录这些新值，供 step() 里的填表使用
        self._D_env       = D_env
        self._D_hack      = D_hack
        self._D_goal      = D_goal
        self._S2          = new_seq
        self._last_seq_len= new_seq

        ############################
        #（8）刷新一次目标向量，保证 target_vector 也是“新尺寸”的
        self._update_target_vector()
        self.tv_cached = self.target_vector.detach()

        ############################
        #（9）—— 新增 —— 把历史张量也扩成“新尺寸” —— #
        #      这样一来，prev_dur、punished_map、prev_infected_map、prev_priv、prev_vuln、
        #      prev_fail 全都能立刻和扩容后的环境同步，避免越界。
        #
        new_h, new_w = self.env.infected_duration_map.shape  # 例如 (12, 12)

        # 1) prev_dur
        if hasattr(self, "prev_dur"):
            old = self.prev_dur
            h0, w0 = old.shape
            # 重新创建一个全 0 的张量，并把 old copy 到左上角
            new_prev_dur = torch.zeros((new_h, new_w), device=old.device, dtype=old.dtype)
            new_prev_dur[:h0, :w0] = old
            self.prev_dur = new_prev_dur

        # 2) punished_map
        if hasattr(self, "punished_map"):
            old_pm = self.punished_map
            h0, w0 = old_pm.shape
            new_pm  = torch.zeros((new_h, new_w), device=old_pm.device, dtype=torch.bool)
            new_pm[:h0, :w0] = old_pm
            self.punished_map = new_pm

        # 3) prev_infected_map
        if hasattr(self, "prev_infected_map"):
            old_inf = self.prev_infected_map
            h0, w0 = old_inf.shape
            new_inf = torch.zeros((new_h, new_w), device=old_inf.device, dtype=old_inf.dtype)
            new_inf[:h0, :w0] = old_inf
            self.prev_infected_map = new_inf

        # 4) prev_priv
        if hasattr(self, "prev_priv"):
            old_priv = self.prev_priv
            h0, w0 = old_priv.shape
            new_priv = torch.zeros((new_h, new_w), device=old_priv.device, dtype=old_priv.dtype)
            new_priv[:h0, :w0] = old_priv
            self.prev_priv = new_priv

        # 5) prev_vuln
        if hasattr(self, "prev_vuln"):
            old_vuln = self.prev_vuln
            h0, w0 = old_vuln.shape
            new_vuln = torch.zeros((new_h, new_w), device=old_vuln.device, dtype=old_vuln.dtype)
            new_vuln[:h0, :w0] = old_vuln
            self.prev_vuln = new_vuln

        # 6) prev_fail
        if hasattr(self, "prev_fail"):
            old_fail = self.prev_fail
            h0, w0 = old_fail.shape
            new_fail = torch.zeros((new_h, new_w), device=old_fail.device, dtype=old_fail.dtype)
            new_fail[:h0, :w0] = old_fail
            self.prev_fail = new_fail

        ############################
        #（7）—— 新增 —— 把 long_virus_tracker 同步扩到新尺寸 —— #
        new_h, new_w = self.env.infected_duration_map.shape  # e.g. (12, 12)

        if hasattr(self, "long_virus_tracker"):
            old_lvt = self.long_virus_tracker
            h0, w0 = old_lvt.shape
            new_lvt = torch.zeros((new_h, new_w), device=old_lvt.device, dtype=old_lvt.dtype)
            new_lvt[:h0, :w0] = old_lvt
            self.long_virus_tracker = new_lvt
        ############################

        # # （11）—— 新增 —— 把 learnable_sensor 一并扩容 —— #
        # new_H, new_W = self.env.size, self.env.size
        # self.learnable_sensor.resize(new_H, new_W)

    def _update_hack_channels(self):
        """每 step 调用：检查 hack 点变化，更新 self._hack_onehot"""
        # 检查是否有新的黑客类型出现，若有则扩充 embedding
        for hack_name in self.env.hack_types.keys():
            if hack_name not in self.hack_type_to_idx:
                new_idx = len(self.hack_types)
                self.hack_types.append(hack_name)
                self.hack_type_to_idx[hack_name] = new_idx
                old_weight = self.hack_type_embedding.weight.data
                new_embed = nn.Embedding(len(self.hack_types), 1).to(self.device)
                with torch.no_grad():
                    new_embed.weight[:-1].copy_(old_weight)
                    new_embed.weight[-1].uniform_(-0.05, 0.05)
                self.hack_type_embedding = new_embed
                logger.info(f"[Hack类型扩展] 新增类型 {hack_name}，embedding size → {len(self.hack_types)}")
        new_keys = set(self.env.hacks.keys())
        if new_keys == self._last_hack_keys:
            return

        size = self.env.size
        coords = list(self.env.hacks.keys())
        if not coords:
            # 没有任何 hack 点，直接清零并更新 last_hack_keys
            self._hack_onehot.zero_()
            self._last_hack_keys = new_keys
            return

        types = [self.env.hacks[xy]['type'] for xy in coords]
        hx_list, hy_list = zip(*coords)
        hx_tensor = torch.tensor(hx_list, dtype=torch.long, device=self.device)
        hy_tensor = torch.tensor(hy_list, dtype=torch.long, device=self.device)
        type_indices = [self.hack_type_to_idx[t] for t in types]
        type_tensor = torch.tensor(type_indices, dtype=torch.long, device=self.device)

        flat_position = hy_tensor * size + hx_tensor
        flat_idxs = type_tensor * (size * size) + flat_position

        self._hack_onehot.zero_()
        self._hack_onehot[0].scatter_(0, flat_idxs, 1.0)
        self._last_hack_keys = new_keys

    def _update_target_vector(self):
        """
        ImmuneCogGraph 的目标结构：
        - 通道 0：未访问区域（好奇探索）
        - 通道 1：感染区域（病毒热点）
        - 通道 2：提权区域（黑客目标）
        """
        size = self.env.size
        size2 = size * size

        # —— 0) 通道 0：好奇点 = 1 - visited_map —— #
        # —— 0) 通道 0：巡逻优先级 = min(1 , age / 20) —— #
        #     age = 距离上次被任何 emitter 访问的步数
        age_map = self.visit_age_map  # (H,W)
        prio = torch.clamp(age_map / 20.0, 0.0, 1.0)  # 0~1 归一化
        unvisited = prio.view(1, -1)  # shape:[1 , size²]

        # —— 1) 通道 1：感染点 —— #
        # 直接不做“距离 emitter ≤4”的局部过滤，只要 env.infected_map>0 就当感染
        if hasattr(self.env, "infected_map"):
            full_inf = self.env.infected_map.clone()  # (H, W)
            infected = (full_inf > 0.04).float().view(1, -1)  # 只要 env.infected_map[y,x]>0.04 就算 1
        else:
            infected = torch.zeros_like(unvisited)

        # —— 2) 通道 2：提权点 —— #
        if hasattr(self.env, "privilege_level"):
            privileged = torch.clamp(self.env.privilege_level * 2.0, 0.0, 1.0).view(1, -1)
        else:
            privileged = torch.zeros_like(unvisited)

        # —— 拼接为目标向量 —— #
        tv = torch.cat([unvisited, infected, privileged], dim=0)
        self.target_vector = self.target_weights.view(3, 1) * tv

    def _prepare_unit_before_update(self, unit, full_state, expected_input):
        """Delegate processor preparation to the processor controller."""
        return self.processor_controller.prepare_before_update(unit, full_state, expected_input)


    def _rebuild_free_positions(self):
        """
        一次性在 GPU 上生成可出生坐标列表：
          未感染 & 未隔离 & 当前无单位。
        对应的坐标 (x,y) 必须在 [0, size) 范围内才会被考虑，否则跳过。
        """
        # 1) 获取两张布尔图：infected_map、is_quarantined
        inf = (self.env.infected_map > 0).to(torch.bool)       # 形状 (H, W)
        quar = (self.env.is_quarantined > 0).to(torch.bool)    # 形状 (H, W)

        H, W = inf.shape  # 网格大小
        # 2) “占用”图：首先全 0，然后把所有单元的 position 标为 True（如果在范围内）
        occ = torch.zeros_like(inf)
        for u in self.units:
            if hasattr(u, "position"):
                x, y = u.position
                # 只在有效范围内才标记
                if 0 <= x < W and 0 <= y < H:
                    occ[y, x] = True
                else:
                    # 如果确实出现越界的坐标，可以打印一条 debug/warning
                    # logger.warning(f"[_rebuild_free_positions] 单元 {u.id} 位置 {(x,y)} 越界，跳过标记")
                    pass

        # 3) “free = 未感染 & 未隔离 & 当前无单位”
        free = ~(inf | quar | occ)

        # 4) 把 free 的每个 (y,x) 提取出来，转成 [(x,y), ...] 列表
        ys, xs = torch.nonzero(free, as_tuple=True)  # 返回一对张量：ys, xs
        self.free_positions = [(int(xx), int(yy)) for xx, yy in zip(xs, ys)]


    def sensor_forward(self, flat_state: torch.Tensor):
        """Forward pass for sensor cells delegated to the controller."""
        return self.sensor_controller.forward(flat_state)


    def _finalize_unit_update(self, unit, full_state, extra_dict, pending_dict, allow_clone=True):
        """Finalize updates via the processor controller for consistent handling."""
        return self.processor_controller.finalize_unit_update(
            unit, full_state, extra_dict, pending_dict, allow_clone=allow_clone
        )


    def _assign_emitter_goal(self, u):
        """Delegate emitter goal assignment to the emitter controller."""
        return self.emitter_controller.assign_goal(u)


    def _assign_curiosity_goal(self, unit):
        """Delegate curiosity goal assignment to the emitter controller."""
        return self.emitter_controller.assign_curiosity_goal(unit)


    def processor_forward(self, sensor_out: torch.Tensor):
        """Delegate processor forward pass to the processor controller."""
        return self.processor_controller.forward(sensor_out)


    def emitter_forward(self, proc_out: torch.Tensor):
        """Delegate emitter forward pass to the emitter controller."""
        return self.emitter_controller.forward(proc_out)


    def expand_unit_dim(self, unit, new_in: int):
        """Delegate expansion of unit dimensions to the processor controller."""
        return self.processor_controller.expand_unit_dim(unit, new_in)


    def _expand_environment_curriculum(self):
        """
        只做内部网络、head、buffer 的维度扩容，
        保留已有权重和优化器状态，不重建整条网络。
        """
        seq_len = self.env.size * self.env.size

        # 1) Transformer head 只做 resize (它自身会保留 trunk 权重)
        self.transformer.resize_head(seq_len)

        # 2) 扩容 replay_head (in_features -> seq_len)
        old = self.replay_head
        new = nn.Linear(seq_len, old.out_features, bias=(old.bias is not None)).to(self.device)
        with torch.no_grad():
            new.weight[:, :old.in_features].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.replay_head = new
        # 更新 optimizer param_groups（假设 replay_head 在 policy_optimizer 或 optimizer 里）
        for g in self.optimizer.param_groups:
            g['params'] = [p for p in list(self.optimizer.param_groups[0]['params'])
                           if p is not old.weight and p is not old.bias] + list(new.parameters())

        # 3) 扩容 value_head (in_features -> D_env+D_hack+D_goal)
        D_env = seq_len * self._STATE_CHANNELS
        D_hack = seq_len * self._HACK_CHANNELS
        D_goal = seq_len * N_GOAL_CHANNELS
        full_state_dim = D_env + D_hack + D_goal

        old = self.value_head
        new = nn.Linear(full_state_dim, old.out_features, bias=(old.bias is not None)).to(self.device)
        with torch.no_grad():
            new.weight[:, :old.in_features].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.value_head = new
        for g in self.value_optimizer.param_groups:
            g['params'] = list(self.value_head.parameters())

        # 4) 按需扩容主干网络的 Linear 层（sensor, processor, emitter）
        def expand_linear(layer, new_in=None, new_out=None):
            old_w, old_b = layer.weight.data, layer.bias.data if layer.bias is not None else None
            in_f, out_f = layer.in_features, layer.out_features
            target_in = new_in if new_in is not None else in_f
            target_out = new_out if new_out is not None else out_f
            new_layer = nn.Linear(target_in, target_out, bias=(layer.bias is not None)).to(self.device)
            with torch.no_grad():
                new_layer.weight[:out_f, :in_f].copy_(old_w)
                if old_b is not None:
                    new_layer.bias[:out_f].copy_(old_b)
            return new_layer

        # sensor_net: 扩 input
        C = self._STATE_CHANNELS + self._HACK_CHANNELS + N_GOAL_CHANNELS
        new_in = C * seq_len
        self.sensor_net[0] = expand_linear(self.sensor_net[0], new_in=new_in)

        # processor_net: 第一层 in_features 扩 to sensor_net.out, 保持 out
        first_proc = next(m for m in self.processor_net.modules() if isinstance(m, nn.Linear))
        self.processor_net[0] = expand_linear(first_proc, new_in=self.sensor_net[0].out_features)

        # emitter_net: 第一个 Linear 从 (in→3*old_seq) 扩到 (in→3*new_seq)
        first_emit = next(m for m in self.emitter_net.modules() if isinstance(m, nn.Linear))
        old_out = first_emit.out_features  # = 3 * old_seq
        new_out = 3 * seq_len  # seq_len = new 环境的 size²
        self.emitter_net[0] = expand_linear(
            first_emit,
            new_in=self.processor_net[0].out_features,
            new_out=new_out
        )

        # 5) buffer 重分配
        self._D_env = D_env
        self._D_hack = D_hack
        self._D_goal = D_goal
        self._S2 = seq_len
        total = D_env + D_hack + D_goal

        self._full_state_buf = torch.empty(1, total, device=self.device)
        self._hack_onehot = torch.zeros(1, D_hack, device=self.device)

        # 6) 保持原有 scheduler 策略 不重建，只 reset last_epoch
        try:
            self.scheduler.last_epoch = 0
        except Exception:
            pass

        # 7) 更新目标向量
        self._update_target_vector()
        self.adaptive_guidance.resize(self.env.size)

        logger.warning(
            f"[Curriculum升级] 第 {self.current_step} 步："
            f"环境大小 → {self.env.size}x{self.env.size} ({seq_len} cells)"
        )

        self.policy_optimizer.param_groups[0]['params'] = (
                list(self.sensor_net.parameters()) +
                list(self.processor_net.parameters()) +
                list(self.emitter_net.parameters()) +
                list(self.hack_type_embedding.parameters()) +
                list(self.goal_net.parameters())
        )

    def _check_enter_static_mode(self):
        """
        每一步在 step() 末尾调用：
          1) 如果不在静息模式：只要环境（known_infections, known_hacks）都为空，就 no_threat_steps += 1；
             一旦 no_threat_steps >= static_threshold，就进入静息模式 (_enter_static_mode)。
          2) 如果已经在静息模式：只要出现 任意（感染 or 黑客），就退出静息模式 (_exit_static_mode)。
        """
        no_infections = len(self.known_infections) == 0
        no_hacks = len(self.known_hacks) == 0

        if not self.static_mode:
            # 如果当前不在静息模式
            if no_infections and no_hacks:
                self.no_threat_steps += 1
            else:
                self.no_threat_steps = 0

            if self.no_threat_steps >= self.static_threshold:
                # 连续多步“无感染 & 无黑客”，触发静息
                self._enter_static_mode()
        else:
            # 已在静息模式：只要出现任一感染 or 黑客，就退出
            if not (no_infections and no_hacks):
                self._exit_static_mode()

    def _enter_static_mode(self):
        """
        切换到静息模式：
          • 标记 static_mode=True
          • 记录进入静息模式的 step
          • 冻结所有单元的 age（存到 self._orig_age 里）
        """
        self.static_mode = True
        self.static_mode_entry_step = self.current_step

        # 存储每个单元当前 age，后续在静息中一直保持这个值
        for u in self.units:
            self._orig_age[u.id] = u.age

        logger.warning(
            f"[进入静息模式] step={self.current_step}；"
            f"已连续 {self.static_threshold} 步无感染／无黑客，开始休眠"
        )


    def _exit_static_mode(self):
        """
        退出静息模式，恢复正常流程：
          • static_mode=False
          • 重置 no_threat_steps
          • 记录退出静息的 step，用于短期保护 CogUnit 不被误杀
        """
        self.static_mode = False
        self.no_threat_steps = 0
        self.static_mode_exit_step = self.current_step
        logger.warning(
            f"[退出静息模式] step={self.current_step}；"
            f"检测到新威胁，重新上线"
        )

    def _static_step(self, full_state: torch.Tensor):
        """
        静息模式下的一步：只让 sensor 扫描环境（更新 known_infections/known_hacks），
        其它单元完全冻结（age/energy 不变），也不做任何结构维护。
        """
        # —— 1) 冻结所有单元的 age —— #
        for u in self.units:
            u.age = self._orig_age.get(u.id, u.age)

        # —— 2) 扫描一次环境，把 known_infections/known_hacks 都更新 —— #
        self._threshold_scan()

        # —— 3) 如果这一步发现了任何感染 or 黑客，就立即退出静息 —— #
        if len(self.known_infections) > 0 or len(self.known_hacks) > 0:
            self._exit_static_mode()
            return

        # —— 4) 如果依旧无威胁，一切保持冻结，直接返回 —— #
        return

    def step(self, input_tensor: torch.Tensor):
        self.current_step += 1

        # ----- Sensor 阈值 warm-up & 衰减 -----
        warmup = 3000
        start, end = 0.5, 0.5
        if self.current_step < warmup:
            ratio = self.current_step / warmup  # 0→1
            self.detect_thresh = start * (1 - ratio) + end * ratio
        else:
            self.detect_thresh = end
        # --------------------------------------

        # 每 1000 步扩张环境
        if self.current_step % 1000 == 0 and self.current_step >= 1000 and self.env.size <= 40:
            self.env._expand_environment()
            self._on_env_resize()
            super().trim_weak_memories()
            input_tensor = self.env.get_state_tensor().view(1, -1)

        # —— 每步更新 hack 通道 —— #
        self._update_hack_channels()

        # 巡逻计时器递增
        self.visit_age_map += 1

        # --- 全局计数 & 能量扩张 --- #
        if self.current_step % 200 == 0:
            self.active_units.clear()
        self._update_global_counts()
        self._expand_energy_cap_if_needed()

        # ——— 用私有方法代替之前那段“解包+推进环境” —— #
        (
            full_state,
            prev_infected,
            prev_dur,
            prev_priv,
            prev_vuln,
            prev_fail
        ) = self._prepare_and_step_env(input_tensor)

        if RF.use_shared_tx and self.current_step % RF.shared_tx_interval == 0:
            self._run_shared_transformer()

        # --- 3) 好奇点奖励（emitter 达到个人目标） --- #
        self._reward_curiosity()

        state_snapshot = full_state.detach().squeeze(0).to(self.device)
        prev_energies  = {u.id: u.energy for u in self.units}


        # —— 如果当前处于静息模式，直接调用 _static_step 并 return —— #
        if self.static_mode:
            # full_state 参数这里不实际用到，传 None 也可以
            return self._static_step(None)


        self.tv_cached = self.target_vector.detach()
        self._rebuild_free_positions()
        self._apply_warmup_and_energy_tax()

        if self.current_step % 40 == 0:
            self.rebalance_cell_types()

        # ——— 用私有方法代替阈值扫描 ——— #
        self._threshold_scan()

        # —— 用私有方法处理单元逻辑 & 防御奖励 —— #
        emitter_outs = self._process_units_and_defense(
            full_state,
            prev_infected,
            prev_dur,
            prev_priv,
            prev_vuln,
            prev_fail
        )


        # --- 8) 记录长期记忆 --- #
        if self.current_step % 100 == 0:
            self.record_long_term_memory(prev_energies, state_snapshot)

        # --- 9) 代谢 & 死亡 & 重生 --- #
        self._metabolism_and_death(full_state)

        # --- 10) 结构维护 & 清理 --- #
        self.auto_connect()
        self.prune_dead_connections()

        # --- 11) 静息 & 系统维护 --- #
        self._check_enter_static_mode()
        self.supply_energy_from_pool()
        self.handle_energy_overflow()

        self.select_elites()
        self.run_subsystem_competition()
        self.assign_subsystems()

        if not self.static_mode and self._should_evolve():
            self._perform_structural_evolution()

        # 保持探索型 emitter 始终占据 10% 比例
        if self.current_step % 50 == 0:
            self._maintain_explorer_emitter_ratio()

        if self.current_step % 50 == 0:
            logger.warning(f"[病毒阶段] 当前为 {self._get_virus_stage()} 阶段")
            self.report_antibody_stats()
        self._maybe_report_by_stage()

        if self.current_step % 500 == 0:
            # ——— 用私有方法处理 PER 训练 —— #
            self._train_from_replay_buffer(batch_size=10, beta=0.4)

        # === meta-learning 更新 === #
        if self.current_step % 1000 == 0 and self.current_step > 1000:
            tasks = self._sample_meta_tasks()
            if tasks:
                self.meta_trainer.meta_update(tasks)

        self.meta_self_evaluation()

        # === 定期修剪 Memory === #
        if self.current_step % 2000 == 0:
            self.memory.trim(keep_last=800)

        # ——— 更新 target_weights ——— #
        if self.current_step % 5 == 0:
            self._update_target_weights()

        # —— 打印击杀统计 & 单元数统计 —— #
        if self.current_step % 50 == 0:
            span = self.current_step - self.kill_stats["last_reset"]
            logger.warning(
                f"[击杀统计] 过去 {span} 步："
                f"病毒-自主={self.kill_stats[ASSIGNMENT_SELF]}, 病毒-学习={self.kill_stats[ASSIGNMENT_LEARNED]} | "
                f"Hack-自主={self.hack_kill_stats[ASSIGNMENT_SELF]}, Hack-学习={self.hack_kill_stats[ASSIGNMENT_LEARNED]}"
            )
            logger.warning(
                f"Sensor 数量：{self.sensor_count}, Processor 数量：{self.processor_count}, "
                f"Emitter 数量：{self.emitter_count}, 总单元数：{len(self.units)}"
            )

        if self.current_step - self.kill_stats["last_reset"] >= 1000:
            self.kill_stats.update({ASSIGNMENT_SELF: 0, ASSIGNMENT_LEARNED: 0, "last_reset": self.current_step})
            self.hack_kill_stats.update({ASSIGNMENT_SELF: 0, ASSIGNMENT_LEARNED: 0})


    def _update_target_weights(self):
        """
        根据当前 env.infected_map 和 env.privilege_level 计算
        infected_intensity 和 hack_intensity 并更新 self.target_weights：
          • w0 恒为 1.0
          • w1 = 0               if 没有感染
                 = clamp(infected_intensity/total_area*10, 1.1, 2.0)
          • w2 = 0               if 没有 hack
                 = clamp(hack_intensity/5, 1.2, 2.0)
        """
        # 1) 计算感染强度
        inf_mask = (self.env.infected_map > 0.04).float()
        infected_intensity = (self.env.infected_map * inf_mask).sum().item()

        # 2) 计算黑客强度
        hack_mask2 = (self.env.privilege_level > 0.04).float()
        hack_intensity = (self.env.privilege_level * hack_mask2).sum().item()

        total_area = self.env.size ** 2
        w0 = 1.0

        if infected_intensity == 0:
            w1 = 0.0
        else:
            w1 = min(2.0, max(1.1, infected_intensity / total_area * 10))

        if hack_intensity == 0:
            w2 = 0.0
        else:
            w2 = min(2.0, max(1.2, hack_intensity / 5.0))

        # 3) 更新 target_weights 张量
        self.target_weights = torch.tensor([w0, w1, w2], device=self.device)

    def _train_from_replay_buffer(self, batch_size: int = 10, beta: float = 0.4):
        """
        从 self.rl_buffer 中采样 batch_size 个样本，用 PER 更新 policy：
          1) 检查 buffer 大小，如果不足就直接跳过并打印警告
          2) sample(batch_size, beta)，得到 samples、indices、is_weights
          3) 对每个样本的 'state' 做 pad/截断，保证维度等于 replay_head.in_features
          4) 构造 labels、is_weights，计算交叉熵 loss，并反向更新 optimizer + scheduler
          5) 用当前 batch 的 td_errors 更新 replay_buffer 的优先级
        """
        if len(self.rl_buffer) < batch_size:
            logger.warning(f"[PER 跳过] RL buffer 大小 {len(self.rl_buffer)} < {batch_size}")
            return

        # 2) 采样
        samples, indices, is_weights = self.rl_buffer.sample(batch_size, beta=beta)

        # 3) pad / truncate 保证 state 维度一致
        feat_dim = self.replay_head.in_features
        padded_states = []
        for tr in samples:
            st = tr["state"]
            if st.shape[0] < feat_dim:
                pad = torch.zeros(feat_dim - st.shape[0], dtype=st.dtype, device=st.device)
                st = torch.cat([st, pad], dim=0)
            elif st.shape[0] > feat_dim:
                st = st[:feat_dim]
            padded_states.append(st)
        states = torch.stack(padded_states, dim=0).to(self.device)  # [B, feat_dim]

        labels = torch.tensor([s["label"] for s in samples], device=self.device)       # [B]
        is_weights = is_weights.to(self.device)                                          # [B]

        # 4) 计算 loss 并反向更新
        self.optimizer.zero_grad()
        logits = self.replay_head(states)                # [B, 2]
        per_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]
        loss = (per_loss * is_weights).mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # 5) 更新 priority
        with torch.no_grad():
            # 计算 td_errors = per_loss.detach().cpu() + 1e-6
            td_errors = (per_loss.detach().cpu() + 1e-6).tolist()
        self.rl_buffer.update_priorities(indices, td_errors)

    def _process_units_and_defense(
            self,
            full_state: torch.Tensor,
            prev_infected: torch.Tensor,
            prev_dur: torch.Tensor,
            prev_priv: torch.Tensor,
            prev_vuln: torch.Tensor,
            prev_fail: torch.Tensor
    ):
        """
        1) 为所有单元分配目标并更新访问记录（SpecialEmitter 用 step() 处理）
        2) 执行 _run_emitter_actions() → 发射动作 + A2C 更新
        3) 再次把 goal_vec 同步给所有 emitter
        4) 收集 emitter 输出
        5) 调用 _apply_defense_rewards(...)，传入推进前的环境快照
        返回：
            emitter_outs: collect_emitter_outputs() 的结果（如果为空则为 []）
        """
        # —— 1) 给各单元分配目标、更新访问记录 —— #
        tensor_state = self.env.get_state_tensor()
        for u in self.units:
            if u.role == "emitter":
                # 同步最新的全局目标向量
                u.goal_vec = self.tv_cached
                u.assignment_source = ASSIGNMENT_SELF
                u.assignment_trace = None

            # 计算本轮 expected 输入维度（虽然这里没真正用到 exp_in 做后续处理，但保留原注释意图）
            if u.role == "sensor":
                exp_in = full_state.shape[1]
            elif u.role == "processor":
                exp_in = self.processor_hidden_size
            elif u.role == "emitter":
                exp_in = self.emitter_hidden_size
            else:
                exp_in = self.processor_hidden_size

            # 如果是 SpecialEmitter，则单独调用它的 step(tensor_state)
            if isinstance(u, SpecialEmitter):
                u.step(tensor_state)

            # 如果是 emitter，就分配个人目标，并把访问 visited_map/visit_age_map 置零
            if u.role == "emitter" and hasattr(u, "position"):
                self._assign_emitter_goal(u)
                x, y = u.position
                if 0 <= x < self.env.size and 0 <= y < self.env.size:
                    self.env.visited_map[y, x] = True
                    self.visit_age_map[y, x] = 0

        # —— 2) 发射动作 & A2C 更新 —— #
        self._run_emitter_actions()

        # —— 3) 再次把最新目标同步给所有 emitter —— #
        for u in self.units:
            if u.role == "emitter":
                u.goal_vec = self.tv_cached

        # —— 4) 收集 emitter 输出 —— #
        outs = self.collect_emitter_outputs()
        emitter_outs = outs if (outs is not None and isinstance(outs, list) and len(outs) > 0) else []

        # —— 5) 应用防御奖励 & 惩罚 —— #
        self._apply_defense_rewards(prev_infected, prev_dur, prev_priv, prev_vuln, prev_fail)

        return emitter_outs

    def _threshold_scan(self):
        """
        用传感器分片扫描（run_sensor_scans），然后再打印“真实 vs 预测”。
        如果当前没有任何 role="sensor" 的单元，_run_sensor_scans() 会退化成原先的全图一次性阈值扫描。
        """

        # ——— 1) 先调用 _run_sensor_scans()，让各个传感器单元去负责自己那块扫描 ———
        # 这样就把 self.known_infections / self.known_hacks 更新好了
        self._run_sensor_scans()

        # ——— 2) 清空一下原来的记录（这里只是演示，你也可以把“清空”逻辑挪到 run_sensor_scans 里） ———
        #    如果你希望直接依赖 run_sensor_scans 里清空的逻辑，就不要重复清空。
        #    下面两行可以注释掉或者删除：
        # self.known_infections.clear()
        # self.known_hacks.clear()
        #
        #    因为 _run_sensor_scans() 本身已经在一开始清空了两者，所以这里不用重清空。

        # ——— 3) 打印“真实 vs 预测”的逻辑完全保留 ———
        #     这里的“真实”就是直接从 env.infected_map / env.privilege_level 门槛筛选出来的
        thr = 0.04
        # 3.1 扫描真实感染格
        inf_mask = (self.env.infected_map > thr)
        ys_true, xs_true = torch.nonzero(inf_mask, as_tuple=True)
        true_coords = [(int(x), int(y)) for x, y in zip(xs_true.tolist(), ys_true.tolist())]

        # 3.2 扫描真实提权格
        hack_mask = (self.env.privilege_level > thr)
        ys_hack, xs_hack = torch.nonzero(hack_mask, as_tuple=True)
        true_hack_coords = [(int(x), int(y)) for x, y in zip(xs_hack.tolist(), ys_hack.tolist())]

        # 3.3 打印日志：真实 VS 预测
        if self.current_step % 50 == 0:
            logger.warning(f"[Step {self.current_step}] 真实感染坐标：{true_coords}")
            logger.warning(f"[Step {self.current_step}] 预测感染坐标：{list(self.known_infections)}")
            logger.warning(f"[Step {self.current_step}] 真实提权坐标：{true_hack_coords}")
            logger.warning(f"[Step {self.current_step}] 预测提权坐标：{list(self.known_hacks)}")

    def _prepare_and_step_env(self, input_tensor: torch.Tensor):
        """
        将以下逻辑打包：
          1) 解包环境状态 + 目标向量 → 构造 full_state
          2) 保存 prev_infected / prev_dur / prev_priv / prev_vuln / prev_fail
          3) 推进环境 self.env.step()
        返回：
            full_state: [1, D_env + D_hack + D_goal] 的张量
            prev_infected: 推进前的 self.env.infected_map.clone()
            prev_dur: 推进前的 self.prev_dur.clone() 或全零张量
            prev_priv: 推进前的 self.env.privilege_level.clone()
            prev_vuln: 推进前的 self.env.vulnerability.clone()
            prev_fail: 推进前的 self.env.login_failures.clone()
        """
        # 1) 解包环境状态 + 目标（未访问、感染、提权）
        size = self.env.size
        # env_dim = size * size * STATE_CHANNELS
        env_dim = size * size * self._STATE_CHANNELS
        env_state = input_tensor[:, :env_dim]  # 通道0～(_STATE_CHANNELS-1)

        unvisited_map = self.target_vector[0].unsqueeze(0)    # [1, size²]
        infected_map   = self.target_vector[1].unsqueeze(0)    # [1, size²]
        privileged_map = self.target_vector[2].unsqueeze(0)    # [1, size²]

        # 填充 full_state 缓冲区
        fs = self._full_state_buf
        fs[:, :self._D_env] = env_state
        fs[:, self._D_env: self._D_env + self._D_hack] = self._hack_onehot

        offset = self._D_env + self._D_hack
        # unvisited → 通道0
        fs[:, offset: offset + self._S2] = unvisited_map
        # infected → 通道1
        fs[:, offset + self._S2: offset + 2 * self._S2] = infected_map
        # privileged → 通道2
        fs[:, offset + 2 * self._S2: offset + 3 * self._S2] = privileged_map

        full_state = fs  # [1, D_env + D_hack + D_goal]

        # 2) 保存环境推进前的快照
        prev_infected = self.env.infected_map.clone()
        prev_dur      = (
            self.prev_dur.clone()
            if hasattr(self, "prev_dur")
            else torch.zeros_like(self.env.infected_duration_map)
        )
        prev_priv = self.env.privilege_level.clone()
        prev_vuln = self.env.vulnerability.clone()
        prev_fail = self.env.login_failures.clone()

        # 3) 推进环境
        self.env.step()

        return full_state, prev_infected, prev_dur, prev_priv, prev_vuln, prev_fail

    def _sample_meta_tasks(self, num_tasks=5, k_support=4, k_query=4):
        """
        从 replay_buffer 中采样元学习任务（support + query），
        只挑那些已经带 raw_state 的 transition。
        """
        # 只考虑含 raw_state 的 entries
        entries = [e for e in self.rl_buffer if "raw_state" in e]
        # 条件不足就返回空
        if len(entries) < num_tasks * (k_support + k_query):
            return []

        # 按 priority 计算采样概率
        prios = [e["priority"] for e in entries]
        total = sum(prios)
        probs = [p/total for p in prios] if total > 0 else None

        tasks = []
        for _ in range(num_tasks):
            sampled = random.choices(entries, weights=probs, k=(k_support + k_query))
            # 用 e["state"]（flat vector）作为 MAML 的输入
            support = [(e["state"], e["action"], e["label"])
                       for e in sampled[:k_support]]
            query = [(e["state"], e["action"], e["label"])
                     for e in sampled[k_support:]]

            tasks.append({"support": support, "query": query})
        return tasks



    def _maybe_report_by_stage(self):
        stage = self._get_virus_stage()
        if stage != self.last_report_stage:
            logger.warning(f"[阶段切换] 进入 {stage} 阶段")
            self.report_antibody_stats()
            self.last_report_stage = stage

    def _apply_and_reward(self, unit, action):
        """Delegate emitter reward application to the emitter controller."""
        return self.emitter_controller.apply_and_reward(unit, action)


    def _record_antibody_effectiveness(self, action: dict, feat: torch.Tensor):
        return
        # """判断抗体动作是否成功清除感染，并更新计数器"""
        #
        # # --- 1) 拷贝一份环境 ---
        # env_clone = self.env.clone()
        #
        # # --- 2) 读 before ---
        # x, y = action["target"]
        # infected_before = env_clone.infected_map[y, x].item()
        #
        # # --- 3) 在 clone 上执行动作（注意：EmitterActions.perform 需要
        # #     接受一个 env 参数；如果你没写，要改它支持传 env_clone）---
        # self.emitter_actions.perform(action, env=env_clone)
        #
        # # --- 4) 读 after ---
        # infected_after = env_clone.infected_map[y, x].item()
        #
        # # --- 5) 根据 before/after 判定 success ---
        # success = (infected_before > 0.04 and infected_after == 0.0)
        # if success:
        #     self.antibody_success_count += 1
        # else:
        #     self.antibody_failure_count += 1
        #
        # # --- 6) 构造 transition，一定要带上 label 字段 ---
        # next_state_vec = env_clone.get_state_tensor().view(-1).cpu()
        # transition = {
        #     "state": feat.detach().view(-1).cpu(),
        #     "action": action,
        #     "reward": float(success),
        #     "next_state": next_state_vec,
        #     "done": False,
        #     "label": int(success),
        # }
        # self.anti_buffer.append(transition, priority=1.0)
        #
        # # --- 7) 对抗体分类头微调（保持不变） ---
        # logits = self.immune_clf(feat.view(1, -1))
        # label = torch.tensor([[float(success)]], device=self.device)
        # loss = F.binary_cross_entropy_with_logits(logits, label)
        # self.immune_opt.zero_grad()
        # loss.backward()
        # self.immune_opt.step()

    def _get_virus_stage(self) -> str:
        """根据感染演化进度判断阶段：early / middle / outbreak"""
        ratio = self.env.step_count / self.env.difficulty_ramp
        if ratio < 0.3:
            return "early"
        elif ratio < 0.6:
            return "middle"
        else:
            return "outbreak"

    def report_antibody_stats(self):
        return
        # mem_size = len(self.memory.buffer)
        # # 如果 last_similarity 还没设，就默认为 0.0
        # last_sim = getattr(self.immune_processor, "last_similarity", None)
        # if last_sim is None:
        #     last_sim = 0.0
        #
        # total = self.antibody_success_count + self.antibody_failure_count
        # success_rate = (self.antibody_success_count / total) if total > 0 else 0.0
        # if self.current_step % 100 == 0:
        #     logger.warning(
        #         f"[抗体统计] 成功 {self.antibody_success_count} 次，失败 {self.antibody_failure_count} 次，"
        #         f"成功率 {success_rate:.2%}, Memory={mem_size}, last_sim={last_sim:.3f}"
        #     )

    def _reward_curiosity(self):
        """
        只有当个人目标（goal_type）确实是 'curiosity'，
        并且到达了这个 curiosity 点后，才发放好奇奖励。
        """
        for u in self.units:
            # 1) 先判断：如果当前并不是“好奇目标”，直接跳过
            if getattr(u, "goal_type", None) != "curiosity":
                continue

            # 2) 只有在“goal_type == 'curiosity'”时，
            #    我们才给 intrinsic_reward （你可以把数值调到想要的大小）
            #    下面把它设为 0.2（或其他正数）
            u.intrinsic_reward = 0.2

            # 3) 如果 emitter 确实存在 personal_goal，再检查是否到达
            if u.role == "emitter" and hasattr(u, "personal_goal") and u.personal_goal:
                out = u.get_output().flatten()
                seq_len = self.env.size * self.env.size
                # 只对 “MOVE” 那前 seq_len 维做 argmax
                move_logits = out[:seq_len]
                pred = torch.argmax(move_logits).item()
                # 把 flat 索引转换成 (x,y)
                goal_x, goal_y = u.personal_goal
                px = pred % self.env.size
                py = pred // self.env.size

                # 4) 只有当网络预测的位置正好就是那“好奇点”时，才给奖励
                if (px, py) == (goal_x, goal_y):
                    r = getattr(u, "intrinsic_reward", 0.0)
                    u.energy += r
                    u.meta.record(action="intrinsic", reward=+r)
                    u.visit_counts[u.personal_goal] = u.visit_counts.get(u.personal_goal, 0) + 1
                    # 清空 personal_goal 以便下次重新指派
                    u.personal_goal = None
                    u.goal_type = None
                    logger.info("你达到了好奇点，心中充满了决心")

    def _apply_defense_rewards(self,
                               prev_infected, prev_dur,
                               prev_priv, prev_vuln, prev_fail):
        """
        更精准版本：
          - 只奖励有清除行为的 emitter
          - 上游 processor 奖励按比例反馈
          - 全局新增感染可作为环境惩罚处理（非每个 emitter 扣分）
          - 对未清除提权的 emitter 单独扣能量
          - 对刚连续3回合未被清理的病毒点，扣最近 emitter 能量（并在 env.infected_duration_map 上清零）
          - 对持续超过50回合的“真正”未被清理的病毒点，进行全局惩罚
          - 黑客防御奖励也按剩余 hack 数量缩放
        """
        # —— 新增：在此函数内部维护“真实感染计时” —— #
        # 第一次调用时初始化：
        curr_dur = self.env.infected_duration_map
        H, W = curr_dur.shape
        if not hasattr(self, "true_inf_age_map"):
            # 第一次创建：跟环境短期计时图同尺寸，int32 类型
            self.true_inf_age_map = torch.zeros_like(curr_dur, dtype=torch.int32, device=self.device)
        else:
            old = self.true_inf_age_map
            h0, w0 = old.shape
            if (h0, w0) != (H, W):
                # 扩到新尺寸：先生成全 0，再拷贝旧数据到左上角
                new = torch.zeros((H, W), dtype=old.dtype, device=old.device)
                new[:h0, :w0] = old
                self.true_inf_age_map = new

        # “真实感染计时”更新：只要 env.infected_map[y,x] >0.04，就累加；否则清零
        curr_inf_mask = (self.env.infected_map > 0.04)
        self.true_inf_age_map = torch.where(
            curr_inf_mask,
            self.true_inf_age_map + 1,
            torch.zeros_like(self.true_inf_age_map)
        )
        # =============================================================

        # ==== 连续 100（这里你改成200）步“感染格数为 0”时，全体 +0.5 能量奖励 ====
        curr_inf_count = int((self.env.infected_map > 0.04).sum().item())
        if curr_inf_count == 0:
            self.zero_infection_counter += 1
        else:
            self.zero_infection_counter = 0

        if self.zero_infection_counter >= 200:
            for u in self.units:
                u.energy += 0.5
                u.meta.record(action="zero_infection_bonus", reward=1.0)
            self.zero_infection_counter = 0
        # ==================================================================================

        # --- A. 靠近病毒微奖励 / 远离扣分（向量化后，只对发生变化的 emitter 做 Python 操作） ---
        emitter_units = [u for u in self.units if u.role == "emitter" and hasattr(u, "position")]
        M = len(emitter_units)
        if M > 0:
            emit_pos = torch.tensor(
                [u.position for u in emitter_units],
                device=self.device, dtype=torch.float32
            )[:, [0, 1]]  # [M, 2]

            virus_idx = torch.nonzero(self.env.infected_map > 0.04)
            if virus_idx.numel() > 0:
                virus_xy = virus_idx[:, [1, 0]].to(torch.float32)  # [N, 2]
                dists = torch.cdist(emit_pos, virus_xy, p=1)        # [M, N]
                min_dists, min_idxs = dists.min(dim=1)             # [M]
                close_mask = min_dists <= 4.0                       # [M] 布尔向量

                # 初始化历史字段（若尚未创建）
                for u in emitter_units:
                    if not hasattr(u, "latest_base_reward"):
                        u.latest_base_reward = 0.0
                        u.last_rewarded_target_idx = None

                # 筛选需要给奖励和需要撤销奖励的索引
                need_reward_idx = [i for i, flag in enumerate(close_mask.tolist()) if flag]
                need_undo_idx = [
                    i for i, flag in enumerate(close_mask.tolist())
                    if (not flag and emitter_units[i].latest_base_reward > 0)
                ]

                # 给奖励
                for i in need_reward_idx:
                    u_i = emitter_units[i]
                    idx = int(min_idxs[i].item())
                    if u_i.last_rewarded_target_idx != idx:
                        u_i.energy += 0.08
                        u_i.meta.record(action="approach_bonus", reward=0.05)
                        u_i.latest_base_reward = 0.08
                        u_i.last_rewarded_target_idx = idx

                # 撤销之前的奖励
                for i in need_undo_idx:
                    u_i = emitter_units[i]
                    u_i.energy = max(0.0, u_i.energy - u_i.latest_base_reward)
                    u_i.meta.record(action="leave_penalty", reward=-u_i.latest_base_reward)
                    u_i.latest_base_reward = 0.0
                    u_i.last_rewarded_target_idx = None

        # -----------------------------------------------------------------

        curr_dur = self.env.infected_duration_map
        if self.punished_map.shape != curr_dur.shape:
            aligned = torch.zeros_like(curr_dur, dtype=torch.bool)
            h, w = self.punished_map.shape
            aligned[:h, :w] = self.punished_map
            self.punished_map = aligned

        if prev_dur.shape != curr_dur.shape:
            aligned = torch.zeros_like(curr_dur)
            h, w = prev_dur.shape
            aligned[:h, :w] = prev_dur
            prev_dur = aligned

        # —— B. “3 回合滞留”惩罚 —— #
        just_stale = torch.nonzero((curr_dur >= 3) & (prev_dur < 3))
        if emitter_units and just_stale.numel() > 0:
            stale_xy = just_stale[:, [1, 0]].to(torch.float32)  # [S, 2]
            d2 = torch.cdist(stale_xy, emit_pos, p=1)           # [S, M]
            far_idxs = d2.min(dim=1).indices                     # [S]
            penalty = 0.5
            for sid, fidx in enumerate(far_idxs):
                u_j = emitter_units[fidx.item()]
                if not getattr(u_j, "is_permanent_explorer", False):
                    u_j.energy = max(0.0, u_j.energy - penalty)
                    u_j.meta.record(action="persistence_penalty", reward=-penalty)
                    logger.info(f"[滞留惩罚] emitter {u_j.id} 扣能量 {penalty}，让你摸鱼！")
            for y, x in just_stale.tolist():
                # 3 回合滞留惩罚：重置 env.infected_duration_map，开始重新计数
                self.env.infected_duration_map[y, x] = 0
                if hasattr(self, "prev_dur"):
                    self.prev_dur[y, x] = 0

        # —— 2) 计算全局病毒传播惩罚 —— #
        curr = self.env.infected_map

        def align(old, target_shape):
            if old.shape == target_shape:
                return old
            new = torch.zeros(target_shape, device=old.device, dtype=old.dtype)
            h, w = old.shape
            new[:h, :w] = old
            return new

        prev_infected = align(prev_infected, curr.shape)
        prev_priv = align(prev_priv, curr.shape)
        prev_vuln = align(prev_vuln, curr.shape)
        prev_fail = align(prev_fail, curr.shape)

        new_inf = ((curr > 0) & (prev_infected == 0)).sum().item()

        infected_points = torch.nonzero(curr).tolist()

        # —— 3) 当前提权位置 —— #
        priv_positions = {
            (x.item(), y.item())
            for y, x in torch.nonzero(self.env.privilege_level > 0.04)
        }
        penalty_per_node = 0.5

        total_cleared = 0
        curr_inf_count = int((self.env.infected_map > 0.04).sum().item())

        # —— 4) 遍历所有 emitter，处理病毒奖励 & 黑客惩罚 —— #
        for u_k in [u for u in self.units if u.role == "emitter"]:
            # —— 4.1) 病毒清理奖励 —— #
            cleared = len(getattr(u_k, "cleared_positions", set()))
            if cleared > 0:
                total_cleared += cleared
                base_reward = 0.6
                scale = 1.0 / (1.0 + curr_inf_count)
                reward = base_reward * scale
                if getattr(u_k, "assignment_source", ASSIGNMENT_SELF) == ASSIGNMENT_LEARNED:
                    reward *= LEARNED_FACTOR

                u_k.energy += reward
                u_k.meta.record(action="defense", reward=reward)
                logger.warning(f"真棒，干掉了个病毒，场上感染点越多，奖励越低，给奖励{reward}")
                u_k.cleared_positions.clear()

                if infected_points and hasattr(u_k, "position"):
                    dists = [math.hypot(u_k.position[0] - x, u_k.position[1] - y) for x, y in infected_points]
                    bonus = max(0, (5 - min(dists)) / 5) * 0.05 * cleared
                    u_k.energy += bonus
                    u_k.meta.record(action="distance_bonus", reward=bonus)

                for pid in self.reverse_connections.get(u_k.id, ()):
                    p = self.unit_map.get(pid)
                    if p and p.role == "processor":
                        fb = reward * 0.6
                        p.energy += fb
                        p.meta.record(action="upstream", reward=fb)

                u_k.cleared_positions.clear()

            # ==== NEW：hack 清理奖励（若 cleared_hack 属性存在）====
            if hasattr(u_k, "cleared_hack") and u_k.cleared_hack:
                hack_r = 1.2
                if getattr(u_k, "assignment_source", ASSIGNMENT_SELF) == ASSIGNMENT_LEARNED:
                    hack_r *= LEARNED_FACTOR

                curr_hack_count = len(self.env.hacks)
                scale = 1.0 / (1 + curr_hack_count)
                hack_r *= scale

                u_k.energy += hack_r
                u_k.meta.record(action="hack_defense", reward=hack_r)
                u_k.cleared_hack.clear()

                for pid in self.reverse_connections.get(u_k.id, ()):
                    p = self.unit_map.get(pid)
                    if p and p.role == "processor":
                        fb = hack_r * 0.6
                        p.energy += fb
                        p.meta.record(action="hack_defense", reward=fb)

            # —— 4.2) 黑客惩罚 —— #
            if hasattr(u_k, "position") and u_k.position in priv_positions:
                u_k.energy = max(0.0, u_k.energy - penalty_per_node)
                u_k.meta.record(action="hack_failure", reward=-penalty_per_node)
                logger.info(
                    f"[黑客惩罚] emitter {u_k.id} 在 {u_k.position} 未清除提权，扣能量 {penalty_per_node}，菜就多练!"
                )
            # —— 不动/打圈惩罚 —— #
            if not hasattr(u_k, "output_positions"):
                u_k.output_positions = deque(maxlen=20)
            if (len(u_k.output_positions) == u_k.output_positions.maxlen
                    and self.current_step % 10 == 0):
                start = u_k.output_positions[0]
                end = u_k.output_positions[-1]
                manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])

                if manhattan < 3:
                    u_k.energy -= 0.3
                    u_k.meta.record(action="idle_penalty", reward=-0.1)

                uniq = len(set(u_k.output_positions))
                if uniq <= 3:
                    u_k.energy -= 0.5
                    u_k.meta.record(action="loop_penalty", reward=-0.14)
        # —— 6) 全局黑客防御奖励 —— #
        cleared_priv = (prev_priv > 0.04).sum() - (self.env.privilege_level > 0.04).sum()
        reduced_vuln = (prev_vuln - self.env.vulnerability).clamp(min=0).sum()
        reduced_fail = (prev_fail - self.env.login_failures).clamp(min=0).sum()
        base_hack_reward = (
            0.3 * cleared_priv.item()
            + 0.15 * reduced_vuln.item()
            + 0.1 * reduced_fail.item()
        )
        if base_hack_reward > 0:
            curr_hack_count = len(self.env.hacks)
            scale = 1.0 / (1 + curr_hack_count)
            hack_reward = base_hack_reward * scale

            self.energy_pool += hack_reward
            logger.warning(
                f"[黑客防御奖励] 降权 {cleared_priv.item()}，"
                f"修复 {reduced_vuln.item():.1f}，"
                f"重置登录失败 {reduced_fail.item():.1f}，"
                f"场上剩余 hack={curr_hack_count}，缩放后奖励={hack_reward:.2f}"
            )

        self.energy_pool = max(self.energy_pool, 0.0)

        # —— 7) 超时病毒惩罚（使用 true_inf_age_map） —— #
        overlong_mask = (self.true_inf_age_map > 100)
        num_overlong = overlong_mask.sum().item()

        if num_overlong > 1 and self.current_step > 3000:
            penalty = 0.01 * num_overlong
            logger.warning(f"[超时病毒惩罚] 有 {num_overlong} 个格子持续感染超 100 回合，全体每个细胞扣能量 {penalty:.2f}")
            for u in self.units:
                u.energy = max(0.0, u.energy - penalty)
                u.meta.record(action="timeout_penalty", reward=-penalty)

            # 同时，把这些格子的计时都重置
            self.env.infected_duration_map[overlong_mask] = 0
            self.true_inf_age_map[overlong_mask] = 0

        self.punished_map &= (self.env.infected_map > 0)

    def _run_emitter_actions(self):
        """Delegate emitter action execution to the emitter controller."""
        return self.emitter_controller.run_actions()


    def _decode_action_from_output(self, unit, output_vec):
        """Delegate decoding of emitter outputs to the emitter controller."""
        return self.emitter_controller.decode_action(unit, output_vec)


    def _argmax_position(self, output_vec):
        """Delegate argmax computation to the emitter controller."""
        return self.emitter_controller.argmax_position(output_vec)


    def _metabolism_and_death(self, full_state):
        """代谢、分裂准备、死亡清理"""
        new_units=[]
        pending={"sensor":[], "processor":[], "emitter":[]}
        for u in list(self.units):
            if u.role == "sensor":
                expected_in = self.env.size * self.env.size * self._INPUT_CHANNELS
            elif u.role == "processor":
                expected_in = self.processor_hidden_size
            elif u.role == "emitter":
                expected_in = self.emitter_hidden_size
            else:
                expected_in = self.processor_hidden_size

            inp = self._prepare_unit_before_update(u, full_state, expected_in)

            self._apply_unit_metabolism(u, inp)
            self._finalize_unit_update(u, full_state, {}, pending, allow_clone=True)
            if u.should_die():
                u._allow_external_kill = True
                super().remove_unit(u)
        self.finalize_deaths()
        # 复制逻辑与原版一致
        parents=self._select_clone_parents(pending)
        for p in parents:
            c=self.clone_and_connect(p)
            new_units.append(c)
        for c in new_units:
            self.add_unit(c)

    def _perform_structural_evolution(self):
        """周期性结构演化：合并、重构、子系统、精英"""
        self.merge_redundant_units()
        self.restructure_common_subgraphs()
        self.prune_connections()

    def policy_update(self, env_state: torch.Tensor, flat_idx: int, reward: float):
        # 1) 扁平化环境状态到 [1, D_env]
        if env_state.dim() > 2:
            env_flat = env_state.flatten().unsqueeze(0)
        elif env_state.dim() == 1:
            env_flat = env_state.unsqueeze(0)
        else:
            env_flat = env_state

        # 2) 扁平化目标向量到 [1, D_goal]
        goal_flat = self.tv_cached.reshape(1, -1)

        hack_maps = []
        for t in self.env.attack_types:
            mask = torch.zeros_like(self.env.privilege_level, dtype=torch.float32, device=self.device)
            for (hx, hy), info in self.env.hacks.items():
                if info.get('type') == t:
                    mask[hy, hx] = 1.0
            hack_maps.append(mask)
        hack_flat = torch.stack(hack_maps, dim=0).view(1, -1)

        # 3) 拼接为 full_state
        full_state = torch.cat([env_flat, hack_flat, goal_flat], dim=1)

        # 4) 前向算 logits，这里 emitter_net 输出 [1, 3*seq_len]
        sensor_out = self.sensor_forward(full_state)
        proc_out = self.processor_forward(sensor_out)
        logits = self.emitter_net(proc_out)  # [1, 3*seq_len]
        probs = torch.softmax(logits, dim=-1)  # [1, 3*seq_len]

        # —— 保证索引合法 —— #
        max_len = probs.size(1)
        flat_idx = flat_idx % max_len  # 限制在 [0..3*seq_len-1]

        # —— A2C 更新 —— #
        # 5.1) 计算 log π(a|s)
        log_p = torch.log(probs[0, flat_idx] + 1e-8)

        # 5.2) Critic：估计 V(s) 并计算优势 A = R − V(s)
        D_in = full_state.size(1)
        old_v = self.value_head
        old_in, _ = old_v.in_features, old_v.out_features
        new_in = D_in
        if new_in > old_in:
            # 如果维度不匹配，就扩容 value_head
            new_v = nn.Linear(new_in, 1, bias=(old_v.bias is not None)).to(self.device)
            with torch.no_grad():
                new_v.weight[:, :old_in].copy_(old_v.weight)
                if old_v.bias is not None:
                    new_v.bias.copy_(old_v.bias)
            self.value_head = new_v
            for group in self.value_optimizer.param_groups:
                group['params'] = [self.value_head.weight, self.value_head.bias]

        value = self.value_head(full_state).squeeze(0)  # [1] → scalar
        advantage = reward - value.detach()

        # 5.3) Actor loss & Critic loss
        actor_loss = -log_p * advantage
        critic_loss = F.mse_loss(value, torch.tensor([reward], device=self.device))

        # —— 熵正则 —— #
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # 5.4) 反向更新
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

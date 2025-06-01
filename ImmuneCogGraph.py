import torch
import math

from contextlib import nullcontext
from collections import deque, Counter
from coggraph import CogGraph
import copy
from env_net import GridSecurityEnv
from emitter_actions import EmitterActions
from memory_net_unit import MemoryNetUnit
from processor_immune import ImmuneProcessor
from special_emitter import SpecialEmitter
from emitter_actions import ACTION_BLOCK, ACTION_QUARANTINE, ACTION_HACK_DEFENSE
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

HIT_THRESH = 0.15          # 越大越宽松
MAX_CONNECTIONS = 4  # 每个单元最多连接 4 个下游
N_GOAL_CHANNELS = 3
FLASH_ATTN_AVAILABLE = False
MIN_PATROL_DIST = 3      # 巡逻目标与当前位置的最小曼哈顿距离
HIT_BONUS       = 1.0     # 命中立刻奖励
MISS_PENALTY    = 0.00    # 打空扣分
GUIDED_FACTOR   = 0.3      # guided 奖励折扣
MAX_GUIDED_DIST = 5

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
        # 混合策略控制：guided 的比例，逐步衰减到 0
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

        self._last_seq_len = seq_len
        self.antibody_failure_count = 0
        self.env.bind_units_reference(self.units)
        self.visit_age_map = torch.zeros_like(
            self.env.infected_map, dtype=torch.float16, device=self.device
        )
        self._update_target_vector()
        self.last_report_stage = None
        self.hack_kill_stats = {"self_direct": 0, "guided": 0}
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
        self.kill_stats = {"self_direct": 0, "guided": 0, "last_reset": 0}
        self.guided_prob = 0.4
        self.guided_decay = 0.0001  # 每 step 衰减量，可按需调整
        seq_len = self.env.size * self.env.size

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
        """
        # 新：用实例属性中计算好的总输入通道数
        D_in = seq_len * self._INPUT_CHANNELS

        # 1) sensor_net：D_in → processor_hidden_size
        self.sensor_net = nn.Sequential(
            nn.Linear(D_in, self.processor_hidden_size),
            nn.ReLU(),
        ).to(self.device)

        # 2) processor_net：刚从 sensor_net 拿到的 out_features → emitter_hidden_size
        self.processor_net = nn.Sequential(
            nn.Linear(self.sensor_net[0].out_features, self.emitter_hidden_size),
            nn.ReLU(),
        ).to(self.device)

        # 3) emitter_net：processor_net 的 out_features → seq_len
        self.emitter_net = nn.Sequential(
            nn.Linear(self.processor_net[0].out_features, seq_len),
        ).to(self.device)

        # —— 同步子类的 hidden_size 属性 —— #
        self.processor_hidden_size = self.sensor_net[0].out_features
        self.emitter_hidden_size  = self.processor_net[0].out_features

        if hasattr(self, "_compiled_sensor_net"):
            del self._compiled_sensor_net
        if hasattr(self, "_compiled_processor_net"):
            del self._compiled_processor_net


        try:
            self._sensor_net = torch.compile(self.sensor_net, fullgraph=False, dynamic=True)
        except:
            self._sensor_net = self.sensor_net


    def _run_sensor_scans(self):
        """
        让所有的 sensor 单元“平分”当前整张网格中的(H*W)个格子：
        - 假设有 S 个 sensor，先把所有格子按行优先打平到一个列表 cells；
        - 给第 i（0 ≤ i < S）个 sensor 扫 cells[i], cells[i+S], cells[i+2S], … 这些坐标。
        扫到病毒就放进 self.known_infections；扫到黑客就放进 self.known_hacks。
        """
        # 先收集所有 active 的 sensor 单元
        sensors = [u for u in self.units if u.role == "sensor" and hasattr(u, "position")]
        S = len(sensors)
        if S == 0:
            return  # 没有 sensor，就不用扫描了

        # 把当前网格 flatten 成一个列表
        H, W = self.env.infected_map.shape
        cells = [(x, y) for y in range(H) for x in range(W)]  # 长度 = H*W

        # 遍历每个 sensor，给它一个“索引 i”，让它负责扫 cells[i : : S]
        # 为了让“哪些格子分给哪个 sensor”是固定顺序，可以根据 sensors 列表的顺序来编号
        for i, s in enumerate(sensors):
            # i 是当前 sensor 在 sensors 列表中的“索引”
            # 它要扫描 cells[i], cells[i+S], cells[i+2S], ... 直到越界
            idx = i
            while idx < len(cells):
                x, y = cells[idx]
                # 只要 env.infected_map[y,x] > 0.5，就认为这里是“感染格”
                if self.env.infected_map[y, x] > 0.04:
                    self.known_infections.add((x, y))
                # 只要 env.privilege_level[y,x] > 0.05，就认为这里是“提权（黑客）格”
                if self.env.privilege_level[y, x] > 0.04:
                    self.known_hacks.add((x, y))
                idx += S

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
        #（2）emitter_net 第一层 (Linear: hidden → old_seq) 要扩到 hidden → new_seq
        emit_layer = self.emitter_net[0]
        old_in_e, old_out_e = emit_layer.in_features, emit_layer.out_features
        if new_seq > old_out_e:
            new_emit = nn.Linear(old_in_e, new_seq, bias=(emit_layer.bias is not None)).to(self.device)
            with torch.no_grad():
                # 把旧权重前 old_out_e 行拷过去
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


    def _update_hack_channels(self):
        """每 step 调用：检查 hack 点变化，更新 self._hack_onehot"""
        new_keys = set(self.env.hacks.keys())
        if new_keys == self._last_hack_keys:
            return

        size = self.env.size
        # 重建 flat indices
        flat_idxs = [
            self.hack_type_to_idx[info['type']] * (size*size) + (hy*size + hx)
            for (hx, hy), info in self.env.hacks.items()
        ]
        self._hack_flat_indices = torch.tensor(flat_idxs, dtype=torch.long, device=self.device)

        # zero + scatter
        self._hack_onehot.zero_()
        self._hack_onehot[0].scatter_(0, self._hack_flat_indices, 1.0)

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
        if hasattr(self.env, "infected_map"):
            full_inf = self.env.infected_map.clone()    # (H, W)
            local_inf = torch.zeros_like(full_inf)  # (H,W)
            emitters_xy = [
                u.position for u in self.units
                if u.role == "emitter" and hasattr(u, "position")
            ]
            if emitters_xy:  # 可能当前没有 emitter
                pos = torch.as_tensor(emitters_xy, device=full_inf.device, dtype=torch.long)
                ys, xs = pos[:, 1], pos[:, 0]  # 注意 (x,y) 顺序
                mask = torch.zeros_like(full_inf)
                mask[ys, xs] = 1.0  # 发射 1-hot
                # 9×9 = (2*4+1) 的全部 1 Kernel，相当于把 4-Manhattan 近邻全 Dilate
                dilated = self._dilate(mask.unsqueeze(0).unsqueeze(0))[0, 0]

                local_inf = (dilated > 0).float() * full_inf  # 只保留真正感染格
            else:
                local_inf.zero_()
            infected = local_inf.view(1, -1)
            # <<< END PATCH


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
        # —— 1) 扩张输入维度，跟父类保持一致 —— #
        if unit.input_size < expected_input:
            self.expand_unit_dim(unit, expected_input)

        # —— 2) 对 emitter 统一设置三通道目标 —— #
        if unit.role == "emitter":
            # 直接把全局 target_vector 当成 goal_vec
            # 形状：[3 * size²]，后面 concat 到状态里
            vec = getattr(self, "tv_cached", self.target_vector)
            unit.goal_vec = vec
            unit.current_hazard_xy = getattr(self, "current_hazard_xy", None)

        # —— 3) 全局统计 —— #
        unit.global_emitter_count = sum(1 for u in self.units if u.role == "emitter")
        incoming = self.reverse_connections.get(unit.id, ())
        unit.recent_calls = len(incoming)

        # —— 4) 根据 role 决定输入来源 —— #
        if unit.role == "sensor":
            # 传入完整的环境+目标状态
            return full_state

        # 如果有上游连接，就把上游输出按权重聚合
        if incoming:
            weighted, total_w = [], 0.0
            for uid in incoming:
                strength = self.connections.get(uid, {}).get(unit.id, 0.0)
                if strength == 0.0 or uid not in self.unit_map:
                    continue
                out = self.unit_map[uid].get_output().squeeze(0)
                if out.shape[0] < self.processor_hidden_size:
                    out = F.pad(out, (0, self.processor_hidden_size - out.shape[0]))
                elif out.shape[0] > self.processor_hidden_size:
                    out = out[: self.processor_hidden_size]
                weighted.append(out * strength)
                total_w += strength

            if total_w > 0:
                agg = torch.stack(weighted, dim=0).sum(dim=0) / total_w
                return agg.unsqueeze(0)

        # 没有任何上游连接，返回零输入
        return torch.zeros(1, expected_input, device=self.device)

    # --- 定位：CogGraph._rebuild_free_positions() 全函数 ---
    #   把整个函数替换为 ↓
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


    def _static_step(self, full_state: torch.Tensor):
        """
        静息模式下的一步：只对 active emitter 走一次 sense→act→update 流程，
        其它单元完全冻结，并执行奖励/惩罚 & 系统维护。
        """
        # —— 1) 更新所有 emitter 的 goal_vec —— #
        for u in self.units:
            if u.role == "emitter":
                # tv_cached 在 step() 里已设为最新 target_vector
                u.goal_vec = self.tv_cached
                u.current_hazard_xy = getattr(self, "current_hazard_xy", None)

        # —— 2) 发射动作 —— #
        self._run_emitter_actions()

        # —— 3) 让所有未 resting 的 emitter 真正跑一次 update —— #
        active_ids = {u.id for u in self.units if u.role == "emitter" and not getattr(u, "resting", False)}
        expected_in = full_state.shape[1]
        for u in self.units:
            if u.id in active_ids:
                inp = self._prepare_unit_before_update(u, full_state, expected_in)
                self._apply_unit_metabolism(u, inp)
                u.update(inp)
                # 恢复 age（让‘静息’之外的 emitter 也能正确地冻结/唤醒）
                u.age = self._orig_age.get(u.id, u.age)

        # 奖励 / 惩罚（是否触发退出静息）
        self._handle_reward_and_penalty()

        # 所有 resting 单元的 age 冻结
        for u in self.units:
            if getattr(u, "resting", False):
                u.age = self._orig_age.get(u.id, u.age)

        # 最后做一次全局维护
        self._perform_system_maintenance()

    def sensor_forward(self, flat_state: torch.Tensor):
        # 保证 [B, D_full]
        if flat_state.dim() == 1:
            flat_state = flat_state.unsqueeze(0)
        D_full = self._D_env + self._D_hack + self._D_goal
        L = flat_state.shape[1]
        if L < D_full:
            flat_state = F.pad(flat_state, (0, D_full - L))
        elif L > D_full:
            flat_state = flat_state[:, :D_full]

        # 直接调用原始 sensor_net，不再使用 torch.compile
        return self.sensor_net(flat_state)


    def _finalize_unit_update(
        self, unit, full_state, extra_dict, pending_dict, allow_clone=True
    ):
        """
        在把 last_output 再送回单元网络之前：
        1. 找到该单元(或其 .function)内部第一层 Linear；
        2. 若输入维度不足 → 调用 expand_unit_dim 扩宽；
        3. 同时把 last_output 的长度对齐到线性层 in_features。
        """
        vec = getattr(unit, "last_output", None)
        if vec is not None:
            need = vec.shape[-1]
            # 动态扩宽（内部已兼容非 Module）
            self.expand_unit_dim(unit, need)

            # ---- 获取当前第一层 Linear 的 in_features ----
            def first_linear(root):
                if isinstance(root, nn.Module):
                    for m in root.modules():
                        if isinstance(m, nn.Linear):
                            return m
                return None

            lin = first_linear(unit) or first_linear(getattr(unit, "function", None))
            layer_in = lin.in_features if lin else need

            # ---- 调整向量长度与层对齐 ----
            if need < layer_in:
                vec = F.pad(vec, (0, layer_in - need))
            elif need > layer_in:
                vec = vec[..., : layer_in]
            unit.last_output = vec

        # 调用父类逻辑
        return super()._finalize_unit_update(
            unit, full_state, extra_dict, pending_dict, allow_clone=allow_clone
        )

    def _assign_emitter_goal(self, u):
        """
        把“从 known_infections ∪ known_hacks 里挑最近优先点”这一逻辑，改成：
          1) 用 goal_net(threat_vec) 计算每个威胁点的得分；
          2) 在所有存在的威胁 flat_index 中取 top_k 得分最高的候选；
          3) 在 top_k 中找跟 emitter 最近的那个，设为 personal_goal。
        其余到达后 fallback 回好奇点的逻辑不变。
        """
        size = self.env.size
        seq_len = size * size

        # —— ①（可选）保留“旧目标还在集合里就不改变”—— #
        old_goal = getattr(u, "personal_goal", None)
        old_type = getattr(u, "goal_type", None)
        if old_type in ("infection", "hack") and old_goal is not None:
            # 如果旧目标依然在 known_infections 或 known_hacks 里，就维持原目标
            if old_type == "infection" and old_goal in self.known_infections:
                return
            if old_type == "hack" and old_goal in self.known_hacks:
                return
            # 否则清除 old_goal，接着重新选
            u.personal_goal = None
            u.goal_type = None

        # —— ② 构造“威胁 flat index 列表” threat_list —— #
        threat_list = []
        # 把 known_infections 内每个 (x,y) 转成 flat_idx = y * size + x
        for (x_inf, y_inf) in self.known_infections:
            threat_list.append(y_inf * size + x_inf)
        # 把 known_hacks 内每个 (x,y) 转成 flat_idx
        for (x_hk, y_hk) in self.known_hacks:
            threat_list.append(y_hk * size + x_hk)

        if threat_list:
            # —— 2.1) 构造二值向量 threat_vec，shape=[1, seq_len] —— #
            device = self.device
            threat_vec = torch.zeros((1, seq_len), device=device)
            indices = torch.tensor(threat_list, dtype=torch.long, device=device)
            threat_vec[0].scatter_(0, indices, 1.0)

            # —— 2.2) 用 goal_net 输出 logits，再 Softmax 得到 probs —— #
            logits = self.goal_net(threat_vec)    # [1, seq_len]
            probs = torch.softmax(logits, dim=-1) # [1, seq_len]

            # —— 2.3) 从所有 `threat_list` 中取 top_k 得分最高的索引 —— #
            top_k = 5  # 你可以改成 1、3、5 或更大
            # 从 probs[0, threat_list] 里找前 top_k 大的
            # 先取出 threat_list 对应位置的概率
            threat_probs = probs[0, indices]     # shape = [len(threat_list)]
            # 找到这几个概率里最大的 top_k 个在 threat_list 里的相对位置
            if threat_probs.numel() <= top_k:
                # 如果威胁点总数 ≤ top_k，就直接把所有直接作为候选
                top_indices_in_threat = torch.arange(threat_probs.numel(), device=device)
            else:
                # 否则取前 top_k 大的那几个
                _, top_indices_in_threat = torch.topk(threat_probs, k=top_k, largest=True)

            # 真正的候选 flat idx，就用 indices[top_indices_in_threat]
            candidates = indices[top_indices_in_threat]  # shape = [ ≤ top_k ]
            # 把它变成 Python list 的 flat_idx
            candidates = candidates.tolist()  # e.g. [flat1, flat2, flat3, ...]

            # —— 2.4) 在候选里，选一个与 emitter 当前位置曼哈顿距离最小的 —— #
            # 先拿出 emitter 当前位置 (ux,uy)
            ux, uy = u.position
            best_flat = None
            best_dist = None
            for flat_idx in candidates:
                cx = flat_idx % size
                cy = flat_idx // size
                d = abs(cx - ux) + abs(cy - uy)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_flat = flat_idx

            # best_flat 就是最终选中的 flat index
            goal_x, goal_y = best_flat % size, best_flat // size

            # —— 2.5) 标记 goal_type：优先判断是不是黑客，否则当病毒 —— #
            if (goal_x, goal_y) in self.known_hacks:
                u.goal_type = "hack"
            else:
                u.goal_type = "infection"

            u.personal_goal = (goal_x, goal_y)
            # 重置好奇冷却（如果你留了这段逻辑）
            cooldown = getattr(u, "intrinsic_cooldown", 0)
            u._last_intrinsic_step = self.current_step - cooldown

            logger.info(f"[学习+距离] 给 emitter {u.id} 分配目标 → ({goal_x},{goal_y}), 距离={best_dist}")
            return

        # —— ③ 如果没有任何威胁，就走“好奇点”逻辑 —— #
        if getattr(u, "goal_type", None) is None:
            visited = self.env.visited_map
            age_map = self.visit_age_map

            never_visited_mask = ~visited
            cooldown = getattr(u, "intrinsic_cooldown", 0)
            long_time_mask    = (age_map >= cooldown)

            candidate_mask = never_visited_mask | long_time_mask
            cand = torch.nonzero(candidate_mask)

            if cand.numel() == 0:
                return

            if hasattr(u, "position"):
                ex, ey = u.position
                mask_far = []
                for (yy, xx) in cand.tolist():
                    if abs(xx - ex) + abs(yy - ey) >= MIN_PATROL_DIST:
                        mask_far.append(True)
                    else:
                        mask_far.append(False)
                mask_far = torch.tensor(mask_far, dtype=torch.bool, device=cand.device)
                if mask_far.any():
                    cand = cand[mask_far]

            sel = torch.randint(0, cand.size(0), (1,), generator=self._rng).item()
            ty, tx = cand[sel].tolist()

            u.personal_goal      = (tx, ty)
            u.goal_type          = "curiosity"
            u._last_intrinsic_step = self.current_step
            logger.info(f"[好奇点] 给 emitter {u.id} 分配新好奇点：({tx},{ty})")




    def processor_forward(self, sensor_out: torch.Tensor):
        if sensor_out.dim() == 1:
            sensor_out = sensor_out.unsqueeze(0)

        # —— 共享网络模式 —— #
        if self.use_shared_unit_nets:
            if not hasattr(self, "_compiled_processor_net"):
                try:
                    self._compiled_processor_net = torch.compile(
                        self.processor_net, fullgraph=False, dynamic=True
                    )
                except:
                    self._compiled_processor_net = self.processor_net
            out = self._compiled_processor_net(sensor_out)

            # hotspot 合并
            if not hasattr(self, "_hotspot_queue"):
                self._hotspot_queue = deque(maxlen=20)
            for p in (u for u in self.units if u.role == "processor"):
                for pos in getattr(p, "hotspots", set()):
                    if pos not in self._hotspot_queue:
                        self._hotspot_queue.append(pos)
                p.hotspots = set()
            return out

        # —— 独立网络模式 —— #
        proc_units = [u for u in self.units if u.role == "processor"]
        outs = []
        for idx, u in enumerate(proc_units):
            inp = sensor_out[idx:idx+1]
            # 如果 u 没有自己的网络，就降级用全局的 self.processor_net
            net = u.processor_net if hasattr(u, "processor_net") else self.processor_net
            po  = net(inp)
            outs.append(po)
        proc_out = torch.cat(outs, dim=0) if outs else None

        # hotspot 合并
        if not hasattr(self, "_hotspot_queue"):
            self._hotspot_queue = deque(maxlen=20)
        for p in proc_units:
            for pos in getattr(p, "hotspots", set()):
                if pos not in self._hotspot_queue:
                    self._hotspot_queue.append(pos)
            p.hotspots = set()
        return proc_out



    def emitter_forward(self, proc_out: torch.Tensor):
        """
        支持「全局共享」或「各自网络」两种模式。
        """
        if proc_out.dim() == 1:
            proc_out = proc_out.unsqueeze(0)

        # 先对 vec 做对齐（pad/trunc）
        vec = proc_out[0]
        H_e = self.emitter_hidden_size
        if vec.shape[0] < H_e:
            vec = F.pad(vec, (0, H_e - vec.shape[0]))
        elif vec.shape[0] > H_e:
            vec = vec[:H_e]

        emitters = [u for u in self.units if u.role == "emitter"]

        # —— 共享 emitter_net —— #
        if self.use_shared_unit_nets:
            batch_in = []
            for e in emitters:
                self.expand_unit_dim(e, H_e)
                batch_in.append(vec.unsqueeze(0))
            if not batch_in:
                return None
            batch = torch.cat(batch_in, dim=0)
            logits = self.emitter_net(batch)
            for e, lg in zip(emitters, logits):
                e.last_output = lg.detach()
            return logits

        # —— 独立 emitter_net —— #
        logits_list = []
        for e in emitters:
            self.expand_unit_dim(e, H_e)
            # 如果 e 没有 emitter_net，就降级用全局的 self.emitter_net
            net = e.emitter_net if hasattr(e, "emitter_net") else self.emitter_net
            lg  = net(vec.unsqueeze(0))  # [1, seq_len]
            e.last_output = lg.detach().squeeze(0)
            logits_list.append(lg)
        if not logits_list:
            return None
        logits = torch.cat(logits_list, dim=0)
        return logits


    def expand_unit_dim(self, unit, new_in: int):
        """
        把第一层 Linear 的 in_features 和最后一层 Linear 的 out_features
        都扩到 new_in。兼容 unit/function 双路径；若找不到 Linear 则跳过。
        """

        def first_linear(root):
            if isinstance(root, nn.Module):
                for m in root.modules():
                    if isinstance(m, nn.Linear):
                        return m
            return None

        def last_linear(root):
            if isinstance(root, nn.Module):
                for m in reversed(list(root.modules())):
                    if isinstance(m, nn.Linear):
                        return m
            return None


        # ---------- 获取首层 ----------
        net_root = unit if isinstance(unit, nn.Module) else getattr(unit, "function", None)
        lin1 = first_linear(net_root)
        if lin1 is None:
            unit.input_size = max(getattr(unit, "input_size", 0), new_in)
            return

        # 若首层已够宽 → 直接返回（输出层之前必一致）
        if lin1.in_features >= new_in:
            unit.input_size = lin1.in_features
            return

        # ---------- 构建新首层 ----------
        old_w, old_b = lin1.weight.data, lin1.bias.data if lin1.bias is not None else None
        out_f, old_in = old_w.shape
        new_l1 = nn.Linear(new_in, out_f, bias=lin1.bias is not None).to(old_w.device)
        new_l1.weight.data[:, :old_in].copy_(old_w)
        if old_b is not None:
            new_l1.bias.data.copy_(old_b)

        # ---------- 替换首层 ----------
        replaced = False

        def _replace_first(root):
            nonlocal replaced
            for name, mod in root.named_children():
                if mod is lin1:
                    setattr(root, name, new_l1)
                    replaced = True
                    return
                _replace_first(mod)

        _replace_first(net_root)


        # ---------- 同步输出层 ----------
        lin_last = last_linear(net_root)
        if lin_last is not None and lin_last.out_features < new_in:
            w_last, b_last = lin_last.weight.data, lin_last.bias.data if lin_last.bias is not None else None
            old_out, old_in = w_last.shape  # ← 行=old_out, 列=old_in **顺序要对**
            # 1) 新层：in_features = old_in, out_features = new_in
            new_last = nn.Linear(old_in, new_in, bias=lin_last.bias is not None).to(w_last.device)
            # 2) 复制 “行”
            new_last.weight.data[:old_out, :].copy_(w_last)
            if b_last is not None:
                new_last.bias.data[:old_out].copy_(b_last)

            # 3) 把新层替换回网络
            def _replace_last(root):
                for name, mod in root.named_children():
                    if mod is lin_last:
                        setattr(root, name, new_last)
                        return
                    _replace_last(mod)

            _replace_last(net_root)

        unit.input_size = new_in
        unit.l1 = new_l1

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

        # emitter_net: 第一层 in_features 扩 to processor_net.out
        first_emit = next(m for m in self.emitter_net.modules() if isinstance(m, nn.Linear))
        self.emitter_net[0] = expand_linear(first_emit, new_in=self.processor_net[0].out_features)

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

    def step(self, input_tensor: torch.Tensor):
        self.current_step += 1

        if self.current_step % 1000 == 0 and self.current_step >= 1000 and self.env.size <= 40:
            # 扩张环境，每1000步扩展2
            self.env._expand_environment()
            self._on_env_resize()
            # 同步清理 local_memory_pool 中最弱 50%
            super().trim_weak_memories()
            # —— 环境、buffer 已扩容完毕，input_tensor 已过时 —— #
            #    重新从新环境拿一次最新的状态张量并展平
            input_tensor = self.env.get_state_tensor().view(1, -1)

        # —— 然后是每步都要的 hack 通道更新 ——
        self._update_hack_channels()

        # # --- 巡逻计时器递增，必要时自动扩张 -------------------
        # if self.visit_age_map.shape != self.env.infected_map.shape:
        #     # 地图扩容后同步 shape
        #     self.visit_age_map = torch.zeros_like(self.env.infected_map, dtype=torch.float16)

        # 每格 +1 计时
        self.visit_age_map += 1

        # # —— 保证 buffer 维度与当前 env.size 同步 ——
        # curr_size2 = self.env.size * self.env.size  # H*W
        # # 重新计算各通道长度
        # self._D_env = curr_size2 * self._STATE_CHANNELS
        # self._D_hack = curr_size2 * self._HACK_CHANNELS
        # self._D_goal = curr_size2 * N_GOAL_CHANNELS
        # self._S2 = curr_size2
        # # 如果 buffer 尺寸不对，就重建
        # total_dim = self._D_env + self._D_hack + self._D_goal
        # if self._full_state_buf.shape[1] != total_dim:
        #     self._full_state_buf = torch.empty(1, total_dim, device=self.device)
        # if self._hack_onehot.shape[1] != self._D_hack:
        #     self._hack_onehot = torch.zeros(1, self._D_hack, device=self.device)
        #
        # # 下面再拿 env_state 填充 fs，就不会维度错乱了
        #
        # # ------------------------------------------------------------

        # —— 缓存 hack 点 & 它们的 learnable 风险权重 ——
        if self.env.hacks.keys() != self._last_hack_keys:
            size = self.env.size
            # 1) 重新生成所有扁平索引
            flat_idxs = [
                self.hack_type_to_idx[info['type']] * (size * size) + (hy * size + hx)
                for (hx, hy), info in self.env.hacks.items()
            ]
            self._hack_flat_indices = torch.tensor(flat_idxs, dtype=torch.long, device=self.device)
            self._last_hack_keys = set(self.env.hacks.keys())

        # --- 1) 前置计数器 & 同步 ---
        if self.current_step % 200 == 0:
            self.active_units.clear()

        self._update_global_counts()

        self._expand_energy_cap_if_needed()

        # 解包环境状态 + 目标（当前设计中：未访问区域 + 感染热图）
        env_dim = self.env.size * self.env.size * self._STATE_CHANNELS
        env_state = input_tensor[:, :env_dim]  # 通道 0~3：感染、强度、历史、评分

        # 从 target_vector 中提取：未访问图 + 感染图（每个是 [1, size²]）
        unvisited_map = self.target_vector[0].unsqueeze(0)  # 替代旧资源目标
        infected_map = self.target_vector[1].unsqueeze(0)  # 替代旧陷阱目标

        # 拼接为 [B, 状态 + 目标]：最终 full_state 送入前向
        privileged_map = self.target_vector[2].unsqueeze(0)

        # -------- 填充缓冲区，比 cat 省一次内存分配 --------
        fs = self._full_state_buf  # 本地 alias

        # 1) 环境状态
        fs[:, :self._D_env] = env_state

        # 写回缓冲
        fs[:, self._D_env: self._D_env + self._D_hack] = self._hack_onehot

        # 偏移量
        offset = self._D_env + self._D_hack

        # 3) 未访问
        fs[:, offset: offset + self._S2] = unvisited_map

        # 4) 感染
        fs[:, offset + self._S2: offset + 2 * self._S2] = infected_map

        # 5) 提权
        fs[:, offset + 2 * self._S2: offset + 3 * self._S2] = privileged_map

        full_state = fs

        # --- 2) 病毒环境推进 ---
        prev_infected = self.env.infected_map.clone()
        prev_dur = self.prev_dur if hasattr(self, "prev_dur") else torch.zeros_like(self.env.infected_duration_map)
        prev_priv = self.env.privilege_level.clone()
        prev_vuln = self.env.vulnerability.clone()
        prev_fail = self.env.login_failures.clone()

        self.env.step()

        if RF.use_shared_tx and self.current_step % RF.shared_tx_interval == 0:
            self._run_shared_transformer()

        # --- 3) 好奇点奖励（emitter 达到个人目标） ---
        self._reward_curiosity()
        state_snapshot = full_state.detach().squeeze(0).to(self.device)
        prev_energies = {u.id: u.energy for u in self.units}

        if self.static_mode:
            return self._static_step(full_state)

        self.tv_cached = self.target_vector.detach()
        self._rebuild_free_positions()
        self._apply_warmup_and_energy_tax()

        if self.current_step % 40 == 0:
            self.rebalance_cell_types()

        # 每一步开始时，清空已知集合，让 sensor 重新扫描
        self.known_infections.clear()
        self.known_hacks.clear()

        # 所有 sensor 并行平分扫描整张地图
        self._run_sensor_scans()


        # —— 统一遍历所有单元，一次完成下列任务 ——
        tensor_state = self.env.get_state_tensor()
        for u in self.units:
            # 1) 给所有 emitter 写入最新目标向量
            if u.role == "emitter":
                u.goal_vec = self.tv_cached

            # 2) 所有单元都要做 metabolism + update，但 expected 输入维度按角色决定
            if u.role == "sensor":
                exp_in = full_state.shape[1]  # 传给 sensor 的是整个 flat state
            elif u.role == "processor":
                exp_in = self.processor_hidden_size  # processor_hidden_size
            elif u.role == "emitter":
                exp_in = self.emitter_hidden_size  # emitter_hidden_size
            else:
                exp_in = self.processor_hidden_size  # fallback

            # 3) SpecialEmitter 独立执行 step
            if isinstance(u, SpecialEmitter):
                u.step(tensor_state)

            # 4) 给 emitter 分配个人 goal 并重置巡逻计时
            if u.role == "emitter" and hasattr(u, "position"):
                self._assign_emitter_goal(u)
                x, y = u.position
                if 0 <= x < self.env.size and 0 <= y < self.env.size:
                    self.env.visited_map[y, x] = True
                    self.visit_age_map[y, x] = 0

        self._run_emitter_actions()

        for u in self.units:
            if u.role == "emitter":
                u.goal_vec = self.tv_cached


        # --- 4) 免疫识别 & 抗体响应 ---
        tensor_state = self.env.get_state_tensor()
        action = self.immune_processor.classify_and_match(tensor_state)
        if action:
            feat = self.immune_processor._extract_features(tensor_state)
            self.memory.store_attack(feat, action)
            # 修改调用
            self._record_antibody_effectiveness(action, feat)

        outs = self.collect_emitter_outputs()
        emitter_outs = outs if outs is not None and isinstance(outs, list) and len(outs) > 0 else []



        self._apply_defense_rewards(
            prev_infected, prev_dur,
            prev_priv, prev_vuln, prev_fail
        )

        # --- 8) 记录长期记忆 ---
        if self.current_step % 100 == 0:
            self.record_long_term_memory(prev_energies, state_snapshot)

        # --- 9) 代谢 & 死亡 & 重生 ---
        self._metabolism_and_death(full_state) ##########

        # --- 10) 结构维护 & 清理 ---
        self.auto_connect()
        self.prune_dead_connections()

        # --- 11) 静息 & 系统维护 ---
        self._check_enter_static_mode()
        self.supply_energy_from_pool()
        self.handle_energy_overflow()

        self.select_elites()
        self.run_subsystem_competition()
        self.assign_subsystems()

        if not self.static_mode and self._should_evolve():
            self._perform_structural_evolution()

        # 保持探索型 emitter 始终占据 10% 比例（否则自动补充）
        if self.current_step % 50 == 0:
            self._maintain_explorer_emitter_ratio()

        if self.current_step % 50 == 0:
            logger.warning(f"[病毒阶段] 当前为 {self._get_virus_stage()} 阶段")
            self.report_antibody_stats()
        self._maybe_report_by_stage()

        if self.current_step % 500 == 0:
            batch_size = 32
            # buffer 不够就跳过
            if len(self.rl_buffer) < batch_size:
                logger.warning(f"[PER 跳过] RL buffer 大小 {len(self.rl_buffer)} < {batch_size}")
            else:
                samples, indices, is_weights = self.rl_buffer.sample(batch_size, beta=0.4)

                # --- 先 pad / truncate 保证同样长度 ---
                feat_dim = self.replay_head.in_features
                padded = []
                for tr in samples:
                    st = tr["state"]
                    if st.shape[0] < feat_dim:
                        pad = torch.zeros(feat_dim - st.shape[0], dtype=st.dtype, device=st.device)
                        st = torch.cat([st, pad], dim=0)
                    elif st.shape[0] > feat_dim:
                        st = st[:feat_dim]
                    padded.append(st)
                states = torch.stack(padded, dim=0).to(self.device)

                # 同 label 一起推到 GPU
                labels = torch.tensor([s["label"] for s in samples], device=self.device)

                is_weights = is_weights.to(self.device)

                self.optimizer.zero_grad()
                logits = self.replay_head(states)  # (B,2)
                # per‐sample loss
                per_loss = F.cross_entropy(logits, labels, reduction='none')
                loss = (per_loss * is_weights).mean()  # 加权平均
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 计算新的 TD‐error 作为 priority
                with torch.no_grad():
                    # 取 logits 对应 action idx
                    probs = torch.softmax(logits, dim=-1)
                    logp = torch.log(probs[range(batch_size), labels] + 1e-6)
                    td_errors = (per_loss.detach().cpu() + 1e-6).tolist()
                self.rl_buffer.update_priorities(indices, td_errors)

        # === meta-learning ===
        # if self.current_step % 1 == 0:
        if self.current_step % 1000 == 0 and self.current_step > 1000:
            tasks = self._sample_meta_tasks()  # 从 replay_buffer 中采 support/query
            if tasks:  #  非空才更新
                self.meta_trainer.meta_update(tasks)

        self.meta_self_evaluation()

        # === 定期修剪 Memory ===
        if self.current_step % 2000 == 0:
            self.memory.trim(keep_last=800)
        self.guided_prob = max(0.0, self.guided_prob - self.guided_decay)

        if self.current_step % 100 == 0:
            infected_count = self.env.infected_map.sum().item()
            hack_count = sum(1 for v in self.env.hacks.values() if v['type'])
            total_area = self.env.size ** 2
            w0 = 1.0
            w1 = min(2.0, infected_count / total_area * 10)  # 感染密度
            w2 = min(2.0, hack_count / 5.0)  # 黑客数量稀疏
            self.target_weights = torch.tensor([w0, w1, w2], device=self.device)

        if self.current_step % 50 == 0:
            span = self.current_step - self.kill_stats["last_reset"]
            logger.warning(
                f"[击杀统计] 过去 {span} 步："
                f"病毒-自主={self.kill_stats['self_direct']}, 病毒-指引={self.kill_stats['guided']} | "
                f"Hack-自主={self.hack_kill_stats['self_direct']}, Hack-指引={self.hack_kill_stats['guided']}"
            )

            logger.warning(f"Sensor 数量：{self.sensor_count},Processor 数量：{self.processor_count}Emitter 数量：{self.emitter_count}总单元数：{len(self.units)}")


        if self.current_step - self.kill_stats["last_reset"] >= 1000:
            self.kill_stats.update(self_direct=0, guided=0, last_reset=self.current_step)
            self.hack_kill_stats.update(self_direct=0, guided=0)

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
        """
        如果 action['type'] 是 ACTION_BLOCK，就把 action['target'] 作为 3×3 中心：
          1) 基于 orig_inf 里该 3×3 区域内所有 orig_inf>0.04 的格子，构造一个 all_to_clear 列表（去重后）；
          2) hits = len(all_to_clear)，并更新 kill_stats、virus_kill_stats_by_type，以及 unit.cleared_positions；
          3) 对 all_to_clear 中每个坐标依次 perform() 清理；
          4) 统一根据 hits 发放能量奖励。
        其余类型（ACTION_HACK_DEFENSE、QUARANTINE）的逻辑不变。
        """
        size = self.env.size
        H, W = self.env.infected_map.shape

        # === 1) 先做“打击前”的感染图快照，供本次收集使用 ===
        orig_inf = self.env.infected_map.clone()

        # --- 保存“打”动作之前的状态，供 A2C 用 ---
        raw_state = self.env.get_state_tensor().cpu()
        state_tensor = raw_state.view(1, -1).to(self.device)
        state_vec = unit.get_output().detach().view(-1).cpu()
        goal_flat = self.tv_cached.view(1, -1)

        # hack 通道准备（与原来保持一致）
        hack_maps = []
        for t in self.env.attack_types:
            mask = torch.zeros_like(self.env.privilege_level, dtype=torch.float32, device=self.device)
            for (hx, hy), info in self.env.hacks.items():
                if info.get('type') == t:
                    mask[hy, hx] = 1.0
            hack_maps.append(mask)
        hack_flat = torch.stack(hack_maps, dim=0).view(1, -1)

        full_state = torch.cat([state_tensor, hack_flat, goal_flat], dim=1)

        # —— 自动扩容 value_head，如果 D_in 不匹配也会在这里重建 value_head —— #
        with torch.no_grad():
            value_s = self.value_head(full_state).squeeze(0)

        # === 2) 区分 action 类型 ===
        hits = 0
        total_reward = 0.0

        if action["type"] == ACTION_BLOCK:
            cx0, cy0 = action["target"]

            # —— ① 基于 orig_inf 构造 to_clear（中心 3×3 区域内的感染点）—— #
            to_clear = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cx = cx0 + dx
                    cy = cy0 + dy
                    if 0 <= cx < W and 0 <= cy < H and orig_inf[cy, cx] > 0.04:
                        v_info = self.env.attacks.get((cx, cy), None)
                        virus_type = "扩散点" if v_info is None else v_info.get("type", "virus")
                        to_clear.append((cx, cy, virus_type))

            # —— ② 基于 to_clear 构造 all_to_clear（去重后的实际 3×3 覆盖集合）—— #
            clear_dict = {}  # key=(x,y), value=virus_type
            for (cx, cy, _) in to_clear:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        x = cx + dx
                        y = cy + dy
                        if 0 <= x < W and 0 <= y < H and orig_inf[y, x] > 0.04:
                            # 重新获取病毒类型，保证按实际存储
                            v_info2 = self.env.attacks.get((x, y), None)
                            virus_type2 = "扩散点" if v_info2 is None else v_info2.get("type", "virus")
                            clear_dict[(x, y)] = virus_type2
            # all_to_clear 是一个列表：[(x, y, virus_type), ...]
            all_to_clear = [(x, y, vt) for ((x, y), vt) in clear_dict.items()]

            # —— ③ hits = len(all_to_clear)，更新 kill_stats、virus_kill_stats_by_type、unit.cleared_positions —— #
            hits = len(all_to_clear)
            if hits > 0:
                src = "guided" if getattr(unit, "guided_this_round", False) else "self_direct"
                unit.cleared_positions = set()  # 重置为本轮待清理集合
                for (x, y, virus_type) in all_to_clear:
                    # 全局累计命中次数
                    self.kill_stats[src] += 1
                    # 按“病毒子类型”统计
                    bucket_v = self.virus_kill_stats_by_type.setdefault(
                        virus_type, {"self_direct": 0, "guided": 0}
                    )
                    bucket_v[src] += 1
                    # 记到 unit.cleared_positions
                    unit.cleared_positions.add((x, y))

                # —— ④ 对 all_to_clear 中每个坐标执行一次 perform() —— #
                for (x, y, _) in all_to_clear:
                    self.emitter_actions.perform({
                        "type": ACTION_BLOCK,
                        "target": (x, y)
                    })

                # —— ⑤ 统一根据 hits 发放能量奖励 —— #
                curr_inf_count = int((self.env.infected_map > 0.04).sum().item())
                scale = 1.0 / (1 + curr_inf_count)
                factor = GUIDED_FACTOR if getattr(unit, "guided_this_round", False) else 0.2
                reward = HIT_BONUS * hits * factor * scale

                unit.energy += reward
                unit.meta.record(action="hit_bonus", reward=reward)
                total_reward += reward
                logger.warning(f"[清理奖励] 本次共清除 {hits} 个病毒，获得能量 {reward:.2f}")

                # —— ⑥ 距离 bonus & 上游 processor 分成逻辑 —— #
                infected_points = torch.nonzero(self.env.infected_map > 0).tolist()
                if infected_points and hasattr(unit, "position"):
                    ux, uy = unit.position
                    dists = [
                        math.hypot(ux - px, uy - py)
                        for (py, px) in infected_points
                    ]
                    bonus = max(0, (5 - min(dists)) / 5) * 0.05 * hits
                    unit.energy += bonus
                    unit.meta.record(action="distance_bonus", reward=bonus)

                for pid in self.reverse_connections.get(unit.id, ()):
                    p = self.unit_map.get(pid)
                    if p and p.role == "processor":
                        fb = reward * 0.6
                        p.energy += fb
                        p.meta.record(action="upstream", reward=fb)

                # —— ⑦ 本轮结束后清空 unit.cleared_positions —— #
                unit.cleared_positions.clear()

            hit = (hits > 0)

        elif action["type"] == ACTION_HACK_DEFENSE:
            x, y = action["target"]
            # —— ① 打之前先读一次原始 hack 类型 —— #
            info = self.env.hacks.get((x, y), None)
            hack_type = "unknown" if info is None else info.get("type", "unknown")

            before_priv = self.env.privilege_level[y, x].item()
            before_stealth = self.env.hack_strength[y, x].item()

            # —— ② 执行清理 —— #
            self.emitter_actions.perform(action)

            after_priv = self.env.privilege_level[y, x].item()
            after_stealth = self.env.hack_strength[y, x].item()
            hit = ((before_priv > 0.04 and after_priv == 0.0) or
                   (before_stealth > 0.04 and after_stealth == 0.0))
            if hit:
                src = "guided" if getattr(unit, "guided_this_round", False) else "self_direct"
                self.hack_kill_stats[src] += 1

                bucket_h = self.hack_kill_stats_by_type.setdefault(
                    hack_type, {"self_direct": 0, "guided": 0}
                )
                bucket_h[src] += 1

                hit_reward = HIT_BONUS * (GUIDED_FACTOR if getattr(unit, "guided_this_round", False) else 0.2)
                unit.energy += hit_reward
                unit.meta.record(action="hack_defense", reward=hit_reward)
                total_reward += hit_reward
                logger.warning(f"[黑客奖励] 本次清除 hack 类型 {hack_type}，获得能量 {hit_reward:.2f}")
            else:
                total_reward += -MISS_PENALTY
                unit.energy = max(0.0, unit.energy - MISS_PENALTY)
                unit.meta.record(action="miss_penalty", reward=-MISS_PENALTY)

        else:
            # 其余类型（如 QUARANTINE），只 perform 一次，不给奖励/惩罚
            self.emitter_actions.perform(action)
            hit = False
            total_reward += 0.0

        # === 3) A2C 更新、PER 更新等逻辑保持不变 ===
        next_raw_state = self.env.get_state_tensor().to(self.device)
        next_state_tensor = next_raw_state.view(1, -1)
        goal_flat = self.tv_cached.view(1, -1)
        hack_flat = torch.stack(hack_maps, dim=0).view(1, -1)
        full_next_state = torch.cat([next_state_tensor, hack_flat, goal_flat], dim=1)

        with torch.no_grad():
            D_in2 = full_state.size(1)
            old_v2 = self.value_head
            old_in2 = old_v2.in_features
            if D_in2 != old_in2:
                new_v2 = nn.Linear(D_in2, 1, bias=(old_v2.bias is not None)).to(self.device)
                with torch.no_grad():
                    new_v2.weight[:, :old_in2].copy_(old_v2.weight)
                    if old_v2.bias is not None:
                        new_v2.bias.copy_(old_v2.bias)
                self.value_head = new_v2
                for g in self.value_optimizer.param_groups:
                    g['params'] = [self.value_head.weight, self.value_head.bias]

            value_s_next = self.value_head(full_next_state).squeeze(0)

        next_state_vec = next_state_tensor.view(-1).cpu()
        delta = total_reward + self.gamma * value_s_next - value_s
        priority = (delta.abs() + 1e-6).pow(self.rl_buffer.alpha).item()

        transition = {
            "state": state_vec,
            "raw_state": raw_state,
            "action": action,
            "reward": total_reward,
            "next_state": next_state_vec,
            "done": False,
            "label": int(hit),
        }
        if action["type"] in (ACTION_BLOCK, ACTION_HACK_DEFENSE):
            self.rl_buffer.append(transition, priority)

        # A2C 更新
        self.policy_update(state_tensor, self.last_flat_idx, total_reward)

        return hit

    def _record_antibody_effectiveness(self, action: dict, feat: torch.Tensor):
        """判断抗体动作是否成功清除感染，并更新计数器"""

        # --- 1) 拷贝一份环境 ---
        env_clone = self.env.clone()

        # --- 2) 读 before ---
        x, y = action["target"]
        infected_before = env_clone.infected_map[y, x].item()

        # --- 3) 在 clone 上执行动作（注意：EmitterActions.perform 需要
        #     接受一个 env 参数；如果你没写，要改它支持传 env_clone）---
        self.emitter_actions.perform(action, env=env_clone)

        # --- 4) 读 after ---
        infected_after = env_clone.infected_map[y, x].item()

        # --- 5) 根据 before/after 判定 success ---
        success = (infected_before > 0.04 and infected_after == 0.0)
        if success:
            self.antibody_success_count += 1
        else:
            self.antibody_failure_count += 1

        # --- 6) 构造 transition，一定要带上 label 字段 ---
        next_state_vec = env_clone.get_state_tensor().view(-1).cpu()
        transition = {
            "state": feat.detach().view(-1).cpu(),
            "action": action,
            "reward": float(success),
            "next_state": next_state_vec,
            "done": False,
            "label": int(success),
        }
        self.anti_buffer.append(transition, priority=1.0)

        # --- 7) 对抗体分类头微调（保持不变） ---
        logits = self.immune_clf(feat.view(1, -1))
        label = torch.tensor([[float(success)]], device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        self.immune_opt.zero_grad()
        loss.backward()
        self.immune_opt.step()

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
        mem_size = len(self.memory.buffer)
        # 如果 last_similarity 还没设，就默认为 0.0
        last_sim = getattr(self.immune_processor, "last_similarity", None)
        if last_sim is None:
            last_sim = 0.0

        total = self.antibody_success_count + self.antibody_failure_count
        success_rate = (self.antibody_success_count / total) if total > 0 else 0.0
        if self.current_step % 100 == 0:
            logger.warning(
                f"[抗体统计] 成功 {self.antibody_success_count} 次，失败 {self.antibody_failure_count} 次，"
                f"成功率 {success_rate:.2%}, Memory={mem_size}, last_sim={last_sim:.3f}"
            )

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
                pred = torch.argmax(out).item()
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
          - 在同一次循环里，对未清除提权的 emitter 单独扣能量
          - 新增：对刚连续3回合未被清理的病毒点，扣最近 emitter 能量
        """
        # ==== 新增：如果连续 100 步“感染格数为 0”，给全体细胞（所有单元）+1.0 能量奖励 ====
        curr_inf_count = int((self.env.infected_map > 0.04).sum().item())
        if curr_inf_count == 0:
            self.zero_infection_counter += 1
        else:
            self.zero_infection_counter = 0

        if self.zero_infection_counter >= 100:
            for u in self.units:
                u.energy += 1.0
                u.meta.record(action="zero_infection_bonus", reward=1.0)
            self.zero_infection_counter = 0
        # ====================================================================================

        # --- A. 靠近病毒微奖励 / 远离扣分 ---------------------------------
        emitters = [u for u in self.units if u.role == "emitter" and hasattr(u, "position")]
        # 预先构造 emitter 坐标张量，后面 B/C 段都能重用
        if emitters:
            emit_pos = torch.tensor(
                [u.position for u in emitters],
                device=self.device, dtype=torch.float32
            )[:, [0, 1]]  # (x,y)


        virus_idx = torch.nonzero(self.env.infected_map > 0)  # [V,2]
        if emitters and virus_idx.numel() > 0:
            virus_xy = virus_idx[:, [1, 0]].to(torch.float32)  # (x,y)
            # cdist → [E,V]
            dists = torch.cdist(emit_pos, virus_xy, p=1)
            min_dists, min_idxs = dists.min(dim=1)  # [E], [E]
            mask = min_dists <= 4.0
            # 只遍历需要更新的小集合
            for i, u in enumerate(emitters):
                if not hasattr(u, "latest_base_reward"):
                    u.latest_base_reward = 0.0
                    u.last_rewarded_target_idx = None
                if mask[i]:
                    idx = min_idxs[i].item()
                    if u.last_rewarded_target_idx != idx:
                        u.energy += 0.08
                        u.meta.record(action="approach_bonus", reward=0.05)
                        u.latest_base_reward = 0.08
                        u.last_rewarded_target_idx = idx
                else:
                    if u.latest_base_reward > 0:
                        u.energy = max(0.0, u.energy - u.latest_base_reward)
                        u.meta.record(action="leave_penalty", reward=-u.latest_base_reward)
                        u.latest_base_reward = 0.0
                        u.last_rewarded_target_idx = None

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
        # —— 0) 病毒持续3回合惩罚 —— #

        curr_dur = self.env.infected_duration_map
        # 只找那些刚跨过 3 回合，且之前没被惩罚过的
        # —— B. “3 回合滞留”惩罚 ——
        just_stale = torch.nonzero((curr_dur >= 3) & (prev_dur < 3))  # [S,2]
        if emitters and just_stale.numel() > 0:
            stale_xy = just_stale[:, [1, 0]].to(torch.float32)  # (x,y)
            # reuse emit_pos
            d2 = torch.cdist(stale_xy, emit_pos, p=1)  # [S,E]
            far_idxs = d2.min(dim=1).indices  # 每 stale 对应的最近 emitter idx
            penalty = 0.2
            for sid, fidx in enumerate(far_idxs):
                u = emitters[fidx.item()]
                if not getattr(u, "is_permanent_explorer", False):
                    u.energy = max(0.0, u.energy - penalty)
                    u.meta.record(action="persistence_penalty", reward=-penalty)
                    logger.info(f"[滞留惩罚] emitter {u.id} 扣能量 {penalty}，让你摸鱼！")

            # 清零持续计数（只保留一次）
            for y, x in just_stale.tolist():
                self.env.infected_duration_map[y, x] = 0
                if hasattr(self, "prev_dur"):
                    self.prev_dur[y, x] = 0

        # —— 2) 计算病毒传播带来的全局惩罚量 —— #
        curr = self.env.infected_map

        # —— 对齐所有 prev_* 到 curr.shape —— #
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

        # —— 3) 收集当前仍处于提权状态的位置 —— #
        priv_positions = {
            (x.item(), y.item())
            for y, x in torch.nonzero(self.env.privilege_level > 0.04)
        }
        penalty_per_node = 0.2  # 每个未清除提权节点，对应 emitter 扣的能量

        total_cleared = 0
        # 在进入循环之前，先算一下当前场上有多少感染点
        curr_inf_count = int((self.env.infected_map > 0.04).sum().item())

        # —— 4) 遍历所有 emitter，一次性处理 清理奖励 + 黑客惩罚 —— #
        for u in [u for u in self.units if u.role == "emitter"]:
            # —— 4.1) 病毒清理奖励 —— #
            cleared = len(getattr(u, "cleared_positions", set()))
            if cleared > 0:
                total_cleared += cleared

                # 先算一个“基础奖励”，不考虑场上病毒数量
                base_reward = 1 * cleared

                # 然后按场上病毒数量做衰减：病毒越多，实际给到每次清理的奖励越少
                # 这里示例用 1 / (1 + curr_inf_count) 作为缩放因子，你可以根据实际需求调整
                scale = 1.0 / (1.0 + curr_inf_count)
                reward = base_reward * scale

                # 如果是 guided 状态，则再乘以 GUIDED_FACTOR
                if getattr(u, "guided_this_round", False):
                    reward *= GUIDED_FACTOR

                u.energy += reward
                u.meta.record(action="defense", reward=reward)
                logger.warning(f"真棒，干掉了个病毒，场上感染点越多，奖励越低，给奖励{reward}")
                u.cleared_positions.clear()

                # 距离 bonus
                if infected_points and hasattr(u, "position"):
                    dists = [
                        math.hypot(u.position[0] - x, u.position[1] - y)
                        for x, y in infected_points
                    ]
                    bonus = max(0, (5 - min(dists)) / 5) * 0.05 * cleared
                    u.energy += bonus
                    u.meta.record(action="distance_bonus", reward=bonus)

                # 上游 processor 分一部分
                for pid in self.reverse_connections.get(u.id, ()):
                    p = self.unit_map.get(pid)
                    if p and p.role == "processor":
                        fb = reward * 0.6
                        p.energy += fb
                        p.meta.record(action="upstream", reward=fb)

                u.cleared_positions.clear()

            # ==== NEW：hack 清理奖励 ====
            if hasattr(u, "cleared_hack") and u.cleared_hack:
                hack_r = 2 * len(u.cleared_hack)
                if getattr(u, "guided_this_round", False):
                    hack_r *= GUIDED_FACTOR

                u.energy += hack_r
                u.meta.record(action="hack_defense", reward=hack_r)
                u.cleared_hack.clear()

                for pid in self.reverse_connections.get(u.id, ()):
                    p = self.unit_map.get(pid)
                    if p and p.role == "processor":
                        fb = hack_r * 0.6
                        p.energy += fb
                        p.meta.record(action="hack_defense", reward=fb)

            # —— 4.2) 黑客惩罚 —— #
            # 如果 emitter 当前所在位置仍在提权列表里，就扣能量
            if hasattr(u, "position") and u.position in priv_positions:
                u.energy = max(0.0, u.energy - penalty_per_node)
                u.meta.record(action="hack_failure", reward=-penalty_per_node)
                logger.info(
                    f"[黑客惩罚] emitter {u.id} 在 {u.position} 未清除提权，扣能量 {penalty_per_node}，菜就多练!"
                )
            # ---------- “不动 / 原地打圈” 惩罚 ------------------
            # 如果之前还没创建 output_positions（可能这一局从未遇到过病毒）
            if not hasattr(u, "output_positions"):
                u.output_positions = deque(maxlen=20)
            # 每 10 步检查一次最近 20 步的活动范围
            if (len(u.output_positions) == u.output_positions.maxlen
                    and self.current_step % 10 == 0):
                start = u.output_positions[0]
                end   = u.output_positions[-1]
                manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])

                if manhattan < 3:                 # 几乎没动
                    u.energy -= 0.1
                    u.meta.record(action="idle_penalty", reward=-0.1)

                # 打圈：20 步内只覆盖 ≤3 个不同格子
                uniq = len(set(u.output_positions))
                if uniq <= 3:
                    u.energy -= 0.14
                    u.meta.record(action="loop_penalty", reward=-0.14)
            # ----------------------------------------------------------

        # —— 6) 黑客防御奖励 —— #
        cleared_priv = (prev_priv > 0.04).sum() - (self.env.privilege_level > 0.04).sum()
        reduced_vuln = (prev_vuln - self.env.vulnerability).clamp(min=0).sum()
        reduced_fail = (prev_fail - self.env.login_failures).clamp(min=0).sum()
        hack_reward = (
            0.3 * cleared_priv.item()
            + 0.15 * reduced_vuln.item()
            + 0.1 * reduced_fail.item()
        )
        if hack_reward > 0:
            self.energy_pool += hack_reward
            logger.warning(
                f"[黑客防御奖励] 降权 {cleared_priv.item()}，"
                f"修复 {reduced_vuln.item():.1f}，"
                f"重置登录失败 {reduced_fail.item():.1f}"
            )

        self.energy_pool = max(self.energy_pool, 0.0)

        # —— 7) 超时病毒惩罚（>50回合，每个病毒每步都扣 0.1）—— #
        dur_map = self.env.infected_duration_map
        overlong_mask = (dur_map > 100)
        num_overlong = overlong_mask.sum().item()

        if num_overlong > 0 and self.current_step > 2000:
            penalty = 0.05 * num_overlong
            logger.warning(f"[超时病毒惩罚] 有 {num_overlong} 个病毒超过 50 回合，全体每个细胞扣能量 {penalty:.2f}")
            for u in self.units:
                u.energy = max(0.0, u.energy - penalty)
                u.meta.record(action="timeout_penalty", reward=-penalty)
        self.punished_map &= (self.env.infected_map > 0)

    def _run_emitter_actions(self):
        for unit in self.units:
            if unit.role != "emitter" or not hasattr(unit, "get_output"):
                continue

            unit.guided_this_round = False
            action_vec = unit.get_output()
            action = self._decode_action_from_output(unit, action_vec)

            # 如果是“移动”动作，先移动并重新解码
            if action and action["type"] == "move":
                unit.position = action["target"]
                action = self._decode_action_from_output(unit, action_vec)

            if not action or action["type"] == "move":
                continue

            # 先把 flat_idx 记录下来（供 A2C 用）
            x, y = action["target"]
            size = self.env.size
            self.last_flat_idx = y * size + x

            # —— 直接把所有的清理/统计工作交给 _apply_and_reward —— #
            hit = self._apply_and_reward(unit, action)

            # -------- 注意：**这里不再做任何额外的“中心统计”** --------
            # （原来在这里对 action["type"]==ACTION_BLOCK／HACK_DEFENSE、hit==True 的
            #  再次把 kill_stats/virus_kill_stats_by_type++，已经全部迁移到 _apply_and_reward 里了）
            #
            # 例如，原来可能有：
            #     if action["type"] == ACTION_BLOCK and hit:
            #         self.kill_stats[...] +=1
            #         ...
            #         unit.cleared_positions.add((x, y))
            #
            # 这一整段都要删掉，确保“一次清理 → 一处统计”。




    def _decode_action_from_output(self, unit, output_vec):
        """
        让Emitter“学会”自己根据网络输出决定下一步移动：
          1. 先根据 output_vec.argmax() 得到一个 flat_idx → (tx, ty)；
          2. 如果 unit.personal_goal 存在并且距离 personal_goal ≤1，进入硬编码的 3×3 清理/隔离 逻辑；
             （发现附近 hack → hack_defense；否则发现附近 virus → block；都没发现 → quarantine，同时把个人目标清空）
          3. 否则，无论有没有 personal_goal，都让网络输出的 (tx,ty) 决定下一步“朝它”迈一步。
        """
        size = self.env.size
        ux, uy = unit.position  # 当前 Emitter 位置

        # —— 1) 先用网络输出选一个“意图目标” flat_idx —— #
        flat_idx = torch.argmax(output_vec).item()
        tx = flat_idx % size
        ty = flat_idx // size

        # # —— 2) 如果 personal_goal 存在，且当前离它的曼哈顿距离 ≤1，就进入硬编码清理/隔离 —— #
        # if hasattr(unit, "personal_goal") and unit.personal_goal is not None:
        #     goal_x, goal_y = unit.personal_goal
        #     dist_to_goal = abs(ux - goal_x) + abs(uy - goal_y)
        #
        #     if dist_to_goal <= 1:
        #         # 到达 personal_goal 附近，先在 3×3 区域寻找 hack
        #         H, W = self.env.infected_map.shape  # 地图尺寸
        #         # 优先查 hack
        #         for dy in (-1, 0, 1):
        #             for dx in (-1, 0, 1):
        #                 cx = ux + dx
        #                 cy = uy + dy
        #                 if 0 <= cx < W and 0 <= cy < H and self.env.privilege_level[cy, cx] > 0.004:
        #                     # 在周围找到一个 hack，就执行 hack_defense
        #                     # 并把 personal_goal 清空，下轮重新分配
        #                     unit.personal_goal = None
        #                     unit.goal_type = None
        #                     return {"type": ACTION_HACK_DEFENSE, "target": (cx, cy)}
        #         # 如果没找到 hack，再在 3×3 区域找 virus
        #         for dy in (-1, 0, 1):
        #             for dx in (-1, 0, 1):
        #                 cx = ux + dx
        #                 cy = uy + dy
        #                 if 0 <= cx < W and 0 <= cy < H and self.env.infected_map[cy, cx] > 0.004:
        #                     # 找到 virus，就执行 block
        #                     unit.personal_goal = None
        #                     unit.goal_type = None
        #                     return {"type": ACTION_BLOCK, "target": (cx, cy)}
        #         # 既没 hack 也没 virus，就在原地隔离一次，并把 personal_goal 清空
        #         unit.personal_goal = None
        #         unit.goal_type = None
        #         return {"type": ACTION_QUARANTINE, "target": (ux, uy)}

        # —— 3) 其余情况：由网络输出 (tx,ty) 决定下一步朝它迈一步 —— #
        # 如果网络预测的 (tx,ty) 与当前位置相距 >1，就只走一步
        if abs(tx - ux) + abs(ty - uy) > 1:
            # 按原逻辑先计算出 nx, ny
            nx = ux + (1 if tx > ux else -1) if tx != ux else ux
            ny = uy + (1 if ty > uy else -1) if ty != uy else uy

            # —— clamp 到合法范围 —— #
            size = self.env.size
            nx = max(0, min(nx, size - 1))
            ny = max(0, min(ny, size - 1))

            return {"type": "move", "target": (nx, ny)}

        else:
            # —— 3) 否则（距离 ≤ 1），以 (tx,ty) 为中心扫描它的 3×3 区域 —— #
            H, W = self.env.infected_map.shape
            # 先看 hack
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cx = tx + dx
                    cy = ty + dy
                    if 0 <= cx < W and 0 <= cy < H and self.env.privilege_level[cy, cx] > 0.004:
                        return {"type": ACTION_HACK_DEFENSE, "target": (cx, cy)}

            # 再看 virus
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cx = tx + dx
                    cy = ty + dy
                    if 0 <= cx < W and 0 <= cy < H and self.env.infected_map[cy, cx] > 0.004:
                        return {"type": ACTION_BLOCK, "target": (cx, cy)}

            # 九宫格里都没有 hack/virus，就隔离当前位置
            unit.personal_goal = None
            unit.goal_type = None
            return {"type": ACTION_QUARANTINE, "target": (ux, uy)}

    def _argmax_position(self, output_vec):
        """
        将输出向量转为 (x, y) 坐标
        假设 output_vec 是 (size²,) one-hot 或 logits
        """
        flat_idx = torch.argmax(output_vec).item()
        size = self.env.size
        y, x = divmod(flat_idx, size)
        return x, y

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

        # 3) 拼接
        full_state = torch.cat([env_flat, hack_flat, goal_flat], dim=1)

        # 4) 前向算 logits
        sensor_out = self.sensor_forward(full_state)
        proc_out = self.processor_forward(sensor_out)
        logits = self.emitter_net(proc_out)  # [1, seq_len]
        probs = torch.softmax(logits, dim=-1)

        # —— 新增：保证索引合法 —— #
        seq_len = probs.size(1)
        flat_idx = flat_idx % seq_len  # 或者用 min(flat_idx, seq_len-1)

        # —— 5) A2C 更新 —— #
        # 5.1) 计算 log π(a|s)
        log_p = torch.log(probs[0, flat_idx] + 1e-8)

        D_in = full_state.size(1)
        # 先拿到旧的层
        old_v = self.value_head
        old_in, _ = old_v.in_features, old_v.out_features
        new_in = D_in
        if new_in > old_in:
            # 构造新的更大输入维度层
            new_v = nn.Linear(new_in, 1, bias=(old_v.bias is not None)).to(self.device)
            # 把旧权重和偏置拷过去
            with torch.no_grad():
                new_v.weight[:, :old_in].copy_(old_v.weight)
                if old_v.bias is not None:
                    new_v.bias.copy_(old_v.bias)
            self.value_head = new_v

            # 把 optimizer 的参数列表替换成新的 layer，保留 state（动量等）
            for group in self.value_optimizer.param_groups:
                group['params'] = [self.value_head.weight, self.value_head.bias]

        # 5.2) 估计 V(s) 并计算优势 A = R − V(s)
        # full_state 在前面已经拼接过：[1, full_state_dim]
        value = self.value_head(full_state).squeeze(0)       # shape=[1] → scalar
        advantage = reward - value.detach()

        # 5.3) Actor loss 与 Critic loss
        actor_loss  = -log_p * advantage
        critic_loss = F.mse_loss(value, torch.tensor([reward], device=self.device))

        # —— 新增 entropy bonus —— #
        # 这里我们需要重新构造分布，以便计算熵
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()  # 标量

        # 合成总 loss
        total_loss = actor_loss \
                     + 0.5 * critic_loss \
                     - self.entropy_coef * entropy

        # 5.4) 联合反向、更新
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()




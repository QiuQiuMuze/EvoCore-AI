import torch
import math

from contextlib import nullcontext
from collections import deque, Counter
from coggraph import CogGraph
from env_net import GridSecurityEnv
from emitter_actions import EmitterActions
from memory_net_unit import MemoryNetUnit
from processor_immune import ImmuneProcessor
from special_emitter import SpecialEmitter
from emitter_actions import ACTION_BLOCK, ACTION_QUARANTINE, ACTION_HACK_DEFENSE
from config_runtime import RF
from typing import List
import random
from meta_trainer import MetaTrainer
from types import MethodType
import torch.nn.functional as F
import torch.optim as optim
import types, torch
import torch.nn as nn
from models.transformer_policy import TransformerPolicyNetwork
from env import logger
from CogUnit import CogUnit

HIT_THRESH = 0.15          # 越大越宽松
MAX_CONNECTIONS = 4  # 每个单元最多连接 4 个下游
N_STATE_CHANNELS = 14
N_GOAL_CHANNELS = 3
INPUT_CHANNELS = N_STATE_CHANNELS + N_GOAL_CHANNELS
FLASH_ATTN_AVAILABLE = False
MIN_PATROL_DIST = 3      # 巡逻目标与当前位置的最小曼哈顿距离
HIT_BONUS       = 0.10     # 命中立刻奖励
MISS_PENALTY    = 0.02     # 打空扣分
GUIDED_FACTOR   = 0.7      # guided 奖励折扣

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
        # 环境替换
        if env is None:
            env = GridSecurityEnv(size=10, device=device)
        # 混合策略控制：guided 的比例，逐步衰减到 0
        super().__init__(rl_agent, device=device, env=env)
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
        self.antibody_failure_count = 0
        self.env.bind_units_reference(self.units)
        self.visit_age_map = torch.zeros_like(
            self.env.infected_map, dtype=torch.float16, device=self.device
        )
        self._update_target_vector()
        self.replay_buffer = deque(maxlen=10000)
        self.last_report_stage = None
        self.hack_kill_stats = {"self_direct": 0, "guided": 0}
        # 追踪各黑客类型的击杀数
        self.hack_kill_stats_by_type = {}
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
        self.guided_prob = 1.0
        self.guided_decay = 0.0001  # 每 step 衰减量，可按需调整
        seq_len = self.env.size * self.env.size
        self.transformer = TransformerPolicyNetwork(
            input_dim=N_STATE_CHANNELS,
            num_actions=seq_len,
            d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
            max_seq_len=seq_len,
            use_action_noise=False,
            use_flash_attn=FLASH_ATTN_AVAILABLE
        ).to(self.device)

        # —— 2) 用 Transformer 的输出维度初始化 replay_head —— #
        self.replay_head = nn.Linear(self.transformer.fc_out.in_features, 2)

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
            self.prev_dur = self.env.infected_duration_map.clone()
            sp.cleared_positions = set()
            self.add_unit(sp)
        # —— 优化 Transformer —— #
        self.optimizer = optim.Adam(
            list(self.transformer.parameters()) +
            list(self.replay_head.parameters()),
            lr=2e-4
        )
        # —— **新增**：把全局 MLP 主干也加入训练 —— #
        self.policy_optimizer = optim.Adam(
            list(self.sensor_net.parameters()) +
            list(self.processor_net.parameters()) +
            list(self.emitter_net.parameters()),
            lr=1e-4
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,  # 作用对象
            step_size=5_000,  # 每 5 k 步衰减一次
            gamma=0.8  # 学习率乘 0.8
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
        # ---------- 预分配一次 full_state 缓冲 ----------
        size2 = self.env.size * self.env.size
        D_env = size2 * N_STATE_CHANNELS
        D_goal = size2 * 3  # 3 个目标通道
        self._full_state_buf = torch.empty(1, D_env + D_goal, device=self.device)
        self._D_env = D_env
        self._S2 = size2

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
        # -----------------------------------------------------------

    def _should_evolve(self) -> bool:
        # 基于规模 + 病毒压力双阈值
        base = 50  # 最低间隔
        scale_factor = max(1, self.env.size / 10)
        pressure = (self.env.infected_map > 0).float().mean().item()  # 0~1
        interval = int(base * scale_factor / max(0.3, pressure))  # 压力越大，间隔越短
        return self.current_step % interval == 0

    def _build_global_nets(self, seq_len: int):
        """
        构建（或重建）CogGraph 的三条主干网络，并自动同步 hidden_size。
        """
        D_in = seq_len * INPUT_CHANNELS

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
        infected_count = (tensor_state[0] > 0.5).sum().item()  # 通道0：是否感染
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
                kernel = torch.ones((1, 1, 9, 9), device=full_inf.device)
                dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0),
                                   kernel, padding=4).squeeze()
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
        self.target_vector = torch.cat([
            unvisited,  # 通道 0：好奇探索点
            infected,  # 通道 1：病毒热点
            privileged  # 通道 2：黑客提权目标
        ], dim=0)

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
        """
        # 1) 获取三张布尔图
        inf = (self.env.infected_map > 0).to(torch.bool)  # (H,W)
        quar = (self.env.is_quarantined > 0).to(torch.bool)  # (H,W)

        occ = torch.zeros_like(inf)
        for u in self.units:
            if hasattr(u, "position"):
                x, y = u.position
                occ[y, x] = True

        # 2) 可用 = 全 False
        free = ~(inf | quar | occ)

        # 3) 转坐标列表
        ys, xs = torch.nonzero(free, as_tuple=True)
        self.free_positions = [(int(x), int(y)) for x, y in zip(xs, ys)]

    def _static_step(self, input_tensor: torch.Tensor):
        """
        静息模式下的单步，只对 active emitter 执行：
          - 感知目标向量
          - 运行动作（quarantine 等）
          - 保持年龄 & 状态冻结
        """
        # —— 解包状态 —— #
        env_dim = self.env_size * self.env_size * N_STATE_CHANNELS
        env_state = input_tensor[:, :env_dim]
        unvisited = self.target_vector[0].unsqueeze(0)  # [1, size²]
        infected = self.target_vector[1].unsqueeze(0)  # [1, size²]
        privileged = self.target_vector[2].unsqueeze(0)  # 新增
        self.tv_cached = self.target_vector.detach()
        goal_map = torch.cat([unvisited, infected, privileged], dim=0).unsqueeze(0)
        full_state = torch.cat([env_state, goal_map], dim=1)

        # —— 设置 goal_vec（和正常模式保持一致）—— #
        for u in self.units:
            if u.role == "emitter":
                u.goal_vec = self.target_vector.clone()
                u.current_hazard_xy = getattr(self, "current_hazard_xy", None)

        state_snapshot = full_state.detach().squeeze(0).to(self.device)
        prev_energies = {u.id: u.energy for u in self.units}
        self._update_target_vector()
        #  正式执行动作（静息中的探索者 emitter）
        self._run_emitter_actions()

        #  更新 active 单元（静息者不动）
        active_ids = {u.id for u in self.units if not getattr(u, "resting", False)}
        expected_in = self.env_size * self.env_size * INPUT_CHANNELS
        for u in self.units:
            if u.id not in active_ids:
                continue
            inp = self._prepare_unit_before_update(u, full_state, expected_in)
            self._apply_unit_metabolism(u, inp)
            u.update(inp)
            u.age = self._orig_age.get(u.id, u.age)

        #  奖励 / 惩罚（是否触发退出静息）
        self._handle_reward_and_penalty()

        #  所有 resting 单元的 age 冻结
        for u in self.units:
            if getattr(u, "resting", False):
                u.age = self._orig_age.get(u.id, u.age)

        self._perform_system_maintenance()

    def sensor_forward(self, flat_state: torch.Tensor):
        """
        子类完全接管：不再调用父类的 patch 逻辑，而是把整幅
        (env_state + goal_map) 展平后送入自定义 sensor_net。
        """
        # ---------- GUARD: 主干自动对齐 ----------
        cur_D_in = flat_state.numel() if flat_state.dim() == 1 else flat_state.shape[1]
        # --------- JIT compile 缓存 ---------
        if not hasattr(self, "_compiled_sensor_net"):
            try:
                self._compiled_sensor_net = torch.compile(
                    self.sensor_net, fullgraph=False, dynamic=True
                )
            except Exception:
                self._compiled_sensor_net = self.sensor_net  # 回退
        net_use = self._compiled_sensor_net
        # -----------------------------------

        if cur_D_in != self.sensor_net[0].in_features:
            # env.size 已经变了 / 或 INPUT_CHANNELS 被调整
            logger.warning(
                f"[自动重建主干] detected D_in={cur_D_in} "
                f"(old={self.sensor_net[0].in_features}); "
                f"re-building global nets …"
            )
            seq_len = self.env.size * self.env.size  # ← 最新的格子数
            self._build_global_nets(seq_len)  # ← 重新生成三条 MLP
            # 重新 JIT 编译一次，并覆盖 net_use
            try:
                self._compiled_sensor_net = torch.compile(
                    self.sensor_net, fullgraph=False, dynamic=True
                )
            except Exception:
                self._compiled_sensor_net = self.sensor_net  # 回退 CPU Interpreter
            net_use = self._compiled_sensor_net  #  新 Callable OK

        # ----------------------------------------
        # 1) 保证形状 [B, D_in]
        if flat_state.dim() == 1:
            flat_state = flat_state.unsqueeze(0)

        D_in = self.env_size * self.env_size * INPUT_CHANNELS
        # 2) Pad / 截断到 D_in
        if flat_state.shape[1] < D_in:
            flat_state = F.pad(flat_state, (0, D_in - flat_state.shape[1]))
        elif flat_state.shape[1] > D_in:
            flat_state = flat_state[:, :D_in]

        out = net_use(flat_state)

        # === 生成协助请求 ===
        # 仅第 0 个 batch（本实现 1×B），检查 5×5 区域病毒计数
        if out.shape[0] == 1:
            x0, y0 = self.units[0].position if self.units and hasattr(self.units[0], "position") else (0, 0)
            xs = slice(max(0, x0-3), min(self.env.size, x0+4))
            ys = slice(max(0, y0-3), min(self.env.size, y0+4))

            local_stealth = self.env.hack_strength[ys, xs].sum().item()
            if local_stealth > 1.0:  # 阈值可调
                # 通知所有 processor 前往 (0,0)
                for p in (u for u in self.units if u.role == "processor"):
                    p.hotspots = getattr(p, "hotspots", set())
                    p.hotspots.add((0, 0))

            local_cnt = self.env.infected_map[ys, xs].sum().item()
            if local_cnt > 2:
                # infection centroid → 方向向量 (-1/0/1 , -1/0/1)
                coords = torch.nonzero(self.env.infected_map[ys, xs])  # 相对窗口
                if len(coords) > 0:
                    cy, cx = coords.float().mean(0)         # 质心
                    dx = int(torch.sign(cx - 2))            # -2~+2 → -1/0/1
                    dy = int(torch.sign(cy - 2))
                    direction = (dx, dy)
                    for p in (u for u in self.units if u.role == "processor"):
                        p.hotspots = getattr(p, "hotspots", set())
                        p.hotspots.add(direction)

        # === 新增：hack 热点广播 ===
        hack_cnt = self.env.privilege_level[ys, xs].gt(0.05).sum().item()
        if hack_cnt > 0:
            coords = torch.nonzero(self.env.privilege_level[ys, xs] > 0.05)
            cy, cx = coords.float().mean(0)
            dx = int(torch.sign(cx - 2))
            dy = int(torch.sign(cy - 2))
            direction = (dx, dy)
            for p in (u for u in self.units if u.role == "processor"):
                p.hotspots = getattr(p, "hotspots", set())
                p.hotspots.add(direction)  # <-- 复用同一个 hotspot 队列

        # --- 扩大到 13×13 环（±6） ---
        XS2 = slice(max(0, x0 - 6), min(self.env.size, x0 + 7))
        YS2 = slice(max(0, y0 - 6), min(self.env.size, y0 + 7))
        hack_cnt2 = self.env.privilege_level[YS2, XS2].gt(0.02).sum().item()
        if hack_cnt2 > 0:
            coords2 = torch.nonzero(self.env.privilege_level[YS2, XS2] > 0.02)
            cy2, cx2 = coords2.float().mean(0)
            dx2 = int(torch.sign(cx2 - 6))
            dy2 = int(torch.sign(cy2 - 6))
            direction2 = (dx2, dy2)
            for p in (u for u in self.units if u.role == "processor"):
                p.hotspots = getattr(p, "hotspots", set())
                p.hotspots.add(direction2)

        # --- 扩大到 19×19 环（±9） for hack hotspots ---
        XS3 = slice(max(0, x0 - 9), min(self.env.size, x0 + 10))
        YS3 = slice(max(0, y0 - 9), min(self.env.size, y0 + 10))
        if self.env.privilege_level[YS3, XS3].gt(0.015).any():
            coords3 = torch.nonzero(self.env.privilege_level[YS3, XS3] > 0.015)
            cy3, cx3 = coords3.float().mean(0)
            dx3 = int(torch.sign(cx3 - 9))
            dy3 = int(torch.sign(cy3 - 9))
            direction3 = (dx3, dy3)
            for p in (u for u in self.units if u.role == "processor"):
                p.hotspots = getattr(p, "hotspots", set())
                p.hotspots.add(direction3)

        return out


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
        为 emitter 分配个人目标 (personal_goal)

        1. 若本地能看到提权/感染 → 立即锁定最近的该类坐标；
        2. 否则若队列里有 “支援方向 (dx,dy)” 并且 emitter 目前
           goal_type ∈ {None, 'curiosity'} → 生成支援坐标；
        3. 若仍没有目标 → 继续好奇探索（unvisited）。
        """
        vec_len = self.target_vector.shape[1]
        size    = int(vec_len ** 0.5)

        # 解析三张图
        unvisited = self.target_vector[0].view(size, size)
        infected  = self.target_vector[1].view(size, size)
        hacked    = self.target_vector[2].view(size, size)

        # --- 1) 本地可见目标 ---
        if hacked.sum() > 0:
            pos_list = torch.nonzero(hacked > 0.05)
            goal_type = "hack"
        elif infected.sum() > 0:
            pos_list = torch.nonzero(infected > 0.5)
            goal_type = "infection"
        else:
            pos_list = torch.nonzero(unvisited > 0.5)
            goal_type = "curiosity"
        # --- 过滤距离太近的巡逻点 -------------------------
        if goal_type == "curiosity" and hasattr(u, "position"):
            ex, ey = u.position
            mask = [
                (abs(x - ex) + abs(y - ey) >= MIN_PATROL_DIST)
                for y, x in pos_list.tolist()
            ]
            if any(mask):  # 若还有满足距离要求的
                pos_list = pos_list[torch.tensor(mask, dtype=torch.bool)]
        # ---------------------------------------------------------

        # ---------------------- 2) 支援逻辑 ----------------------
        if len(pos_list) == 0:
            idle = getattr(u, "goal_type", None) in (None, "curiosity")
            if idle and hasattr(self, "_hotspot_queue") and self._hotspot_queue:
                dx, dy = self._hotspot_queue.popleft()        # (-1/0/1, -1/0/1)
                ex, ey = getattr(u, "position", (size // 2, size // 2))
                tx = min(max(0, ex + dx * 3), size - 1)       # 向方向前进 3 格
                ty = min(max(0, ey + dy * 3), size - 1)
                u.personal_goal = (tx, ty)
                u.goal_type     = "support"
                return
            else:
                # 无支援且无可见目标——保持原目标或置空
                if getattr(u, "personal_goal", None) is None:
                    u.personal_goal = None
                    u.goal_type     = None
                return

        # ---------------------- 3) 正常选目标 --------------------
        sel = torch.randint(0, pos_list.size(0), (1,), generator=self._rng, device=self.device).item()
        y, x = pos_list[sel].tolist()
        u.personal_goal = (x, y)
        u.goal_type     = goal_type



    def processor_forward(self, sensor_out: torch.Tensor):
        if sensor_out.dim() == 1:
            sensor_out = sensor_out.unsqueeze(0)

        # --------- JIT compile 缓存 ---------
        if not hasattr(self, "_compiled_processor_net"):
            try:
                self._compiled_processor_net = torch.compile(
                    self.processor_net, fullgraph=False, dynamic=True
                )
            except Exception:
                self._compiled_processor_net = self.processor_net
        net_proc = self._compiled_processor_net
        # -----------------------------------

        proc_out = net_proc(sensor_out)


        # 将每个 processor.hotspots 合并到全局 self._hotspot_queue
        if not hasattr(self, "_hotspot_queue"):
            self._hotspot_queue = deque(maxlen=20)
        for p in (u for u in self.units if u.role == "processor"):
            for pos in getattr(p, "hotspots", set()):
                if pos not in self._hotspot_queue:
                    self._hotspot_queue.append(pos)
            p.hotspots = set()        # 清空

        return proc_out


    def emitter_forward(self, proc_out: torch.Tensor):
        """
        子类完全接管 emitter 前向：
        1. proc_out → align 到 self.emitter_hidden_size
        2. 按 emitter 个数复制
        3. 送入子网 self.emitter_net，返回 logits 列表
        """
        if proc_out.dim() == 1:
            proc_out = proc_out.unsqueeze(0)          # [1, H_p]

        # 1) 裁 / 补到 H_e
        vec = proc_out[0]
        H_e = self.emitter_hidden_size
        if vec.shape[0] < H_e:
            vec = F.pad(vec, (0, H_e - vec.shape[0]))
        elif vec.shape[0] > H_e:
            vec = vec[:H_e]

        # 2) 复制给所有 emitter，并确保它们的 l1 输入维度 >= H_e
        emitters = [u for u in self.units if u.role == "emitter"]
        batch_in = []
        for e in emitters:
            self.expand_unit_dim(e, H_e)              # 升维若需要
            batch_in.append(vec.unsqueeze(0))
        if not batch_in:                              # 没有 emitter
            return None
        batch_in = torch.cat(batch_in, dim=0)         # [N_emitters, H_e]

        # 3) 送入自定义 emitter_net
        logits = self.emitter_net(batch_in)           # [N_emitters, seq_len]

        # 4) 把结果写回各 emitter
        for e, lg in zip(emitters, logits):
            e.last_output = lg.detach()
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

    def step(self, input_tensor: torch.Tensor):
        # --- 巡逻计时器递增，必要时自动扩张 -------------------
        if self.visit_age_map.shape != self.env.infected_map.shape:
            # 地图扩容后同步 shape
            self.visit_age_map = torch.zeros_like(self.env.infected_map, dtype=torch.float16)
        # 每格 +1 计时
        self.visit_age_map += 1

        curr_size2 = self.env.size * self.env.size  # H*W
        curr_D_env = curr_size2 * N_STATE_CHANNELS  # 状态通道
        if curr_D_env != getattr(self, "_D_env", -1):
            D_goal = curr_size2 * 3  # 目标向量 3 通道
            self._full_state_buf = torch.empty(1, curr_D_env + D_goal,
                                               device=self.device)
            self._D_env = curr_D_env
            self._S2 = curr_size2
            logger.warning(
                f"[full_state_buf] 环境尺寸变为 {self.env.size}×{self.env.size}，"
                f"已重新分配缓冲区 (D_env={curr_D_env})"
            )

        # ------------------------------------------------------------

        if self.current_step % 50 == 0:
            span = self.current_step - self.kill_stats["last_reset"]
            logger.warning(
                f"[击杀统计] 过去 {span} 步："
                f"病毒-自主={self.kill_stats['self_direct']}, 病毒-指引={self.kill_stats['guided']} | "
                f"Hack-自主={self.hack_kill_stats['self_direct']}, Hack-指引={self.hack_kill_stats['guided']}"
            )
        if self.current_step - self.kill_stats["last_reset"] >= 1000:
            self.kill_stats.update(self_direct=0, guided=0, last_reset=self.current_step)
            self.hack_kill_stats.update(self_direct=0, guided=0)


        self._update_target_vector()

        # —— 缓存一次，全局共享 —— #
        self.tv_cached = self.target_vector.detach()
        for u in self.units:
            if u.role == "emitter":
                u.goal_vec = self.tv_cached


        if self.current_step % 50 == 0:
            logger.warning(f"Sensor 数量：{self.sensor_count},Processor 数量：{self.processor_count}Emitter 数量：{self.emitter_count}总单元数：{len(self.units)}")
        # --- 1) 前置计数器 & 同步 ---
        if self.current_step % 200 == 0:
            self.active_units.clear()
        self.current_step += 1
        self._update_global_counts()
        if RF.use_shared_tx and self.current_step % RF.shared_tx_interval == 0:
            self._run_shared_transformer()
        self._sync_environment_dimensions()

        # 解包环境状态 + 目标（当前设计中：未访问区域 + 感染热图）
        env_dim = self.env_size * self.env_size * N_STATE_CHANNELS
        env_state = input_tensor[:, :env_dim]  # 通道 0~3：感染、强度、历史、评分

        # 从 target_vector 中提取：未访问图 + 感染图（每个是 [1, size²]）
        unvisited_map = self.target_vector[0].unsqueeze(0)  # 替代旧资源目标
        infected_map = self.target_vector[1].unsqueeze(0)  # 替代旧陷阱目标

        # 拼接为 [B, 状态 + 目标]：最终 full_state 送入前向
        privileged_map = self.target_vector[2].unsqueeze(0)
        # -------- 填充缓冲区，比 cat 省一次内存分配 --------
        fs = self._full_state_buf  # 本地 alias
        fs[:, :self._D_env] = env_state
        fs[:, self._D_env: self._D_env + self._S2] = unvisited_map
        fs[:, self._D_env + self._S2: self._D_env + 2 * self._S2] = infected_map
        fs[:, self._D_env + 2 * self._S2:] = privileged_map
        full_state = fs

        # --- 3) 好奇点奖励（emitter 达到个人目标） ---
        self._reward_curiosity()
        state_snapshot = full_state.detach().squeeze(0).to(self.device)
        prev_energies = {u.id: u.energy for u in self.units}


        # 替换掉原来的无条件扩容：
        # self.env._expand_environment()

        #  改为：仅当 size 小于阈值，且间隔一定步数再扩一次
        if self.current_step % 1000 == 0 and self.env.size < 20:
            # --- 新增：同步巡逻计时表尺寸 --------------------------
            self.visit_age_map = torch.zeros_like(
                self.env.infected_map, dtype=torch.int16
            )
            # -------------------------------------------------------
            super()._expand_environment_curriculum()
            # 同步清理 local_memory_pool 中最弱 50%
            super().trim_weak_memories()
            # 目标向量已在 _expand 中更新，无需再手动调用
            if self.visit_age_map.shape != self.env.infected_map.shape:
                new_map = torch.zeros_like(self.env.infected_map, dtype=torch.int16)
                h, w = self.visit_age_map.shape
                new_map[:h, :w] = self.visit_age_map  # 迁移旧计时
                self.visit_age_map = new_map

            self._update_target_vector()

            new_seq_len = self.env.size * self.env.size
            # (1) 重建三大 MLP 主干
            self._build_global_nets(new_seq_len)

            # (2) 只替换 transformer 的输出头
            self.transformer.resize_head(new_seq_len)
            self.optimizer.add_param_group(
                {"params": self.transformer.fc_out.parameters()}
            )

            # (3) 重建 replay_head 对齐新 in_features
            self.replay_head = nn.Linear(
                self.transformer.fc_out.in_features, 2, device=self.device
            )
            self.optimizer.add_param_group({"params": self.replay_head.parameters()})

        self._expand_energy_cap_if_needed()

        if self.static_mode:
            return self._static_step(input_tensor)

        self._run_emitter_actions()
        self._rebuild_free_positions()
        self._apply_warmup_and_energy_tax()

        # --- 2) 病毒环境推进 ---
        prev_infected = self.env.infected_map.clone()
        prev_dur = self.prev_dur if hasattr(self, "prev_dur") else torch.zeros_like(self.env.infected_duration_map)
        prev_priv = self.env.privilege_level.clone()
        prev_vuln = self.env.vulnerability.clone()
        prev_fail = self.env.login_failures.clone()

        self.env.step()
        if self.current_step % 40 == 0:
            self.rebalance_cell_types()


        # --- 4) 免疫识别 & 抗体响应 ---
        tensor_state = self.env.get_state_tensor()
        action = self.immune_processor.classify_and_match(tensor_state)
        if action:
            feat = self.immune_processor._extract_features(tensor_state)
            self.memory.store_attack(feat, action)
            # 修改调用
            self._record_antibody_effectiveness(action, feat)

        # --- 5) 特攻单元响应 ---
        for u in list(self.units):
            if isinstance(u, SpecialEmitter):
                u.step(tensor_state)

        # --- 6) 自组织网络 update ---
        sensor_out = self.sensor_forward(full_state)
        proc_out   = self.processor_forward(sensor_out)
        self.emitter_forward(proc_out)
        env_dim = self.env.size * self.env.size
        for u in self.units:
            if u.role == "emitter":
                self._assign_emitter_goal(u)
            if u.role == "emitter" and hasattr(u, "position"):
                x, y = u.position
                if 0 <= x < self.env.size and 0 <= y < self.env.size:
                    self.env.visited_map[y, x] = True
                    self.visit_age_map[y, x] = 0  # PATCH: 重置该格巡逻时间
        outs = self.collect_emitter_outputs()
        emitter_outs = outs if outs is not None and isinstance(outs, list) and len(outs) > 0 else []

        # --- 7) 丰富奖励/惩罚（带持久惩罚） ---
        # 记录一下上一轮的持续时间，用于检测“刚跨过3回合”
        # --- 7) 丰富奖励/惩罚（带持久惩罚）——传入前后黑客状态—
        self._apply_defense_rewards(
            prev_infected, prev_dur,
            prev_priv, prev_vuln, prev_fail
        )

        self.prev_infected_map = self.env.infected_map.clone()
        self.prev_dur = self.env.infected_duration_map.clone()
        self.prev_priv = self.env.privilege_level.clone()
        self.prev_vuln = self.env.vulnerability.clone()
        self.prev_fail = self.env.login_failures.clone()

        # --- 8) 记录长期记忆 ---
        self.record_long_term_memory(prev_energies, state_snapshot)

        # --- 9) 代谢 & 死亡 & 重生 ---
        self._metabolism_and_death(full_state)

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
        if self.current_step % 50 == 0:
            logger.warning(f"[病毒阶段] 当前为 {self._get_virus_stage()} 阶段")
            self.report_antibody_stats()
        self._maybe_report_by_stage()

        # === replay Training ===
        if self.current_step % 500 == 0 and len(self.replay_buffer) >= 100:
            # 1) 随机采 64 条经验
            # ---- 按成功 / 失败拆分 ---------------------------------
            successes = [s for s in self.replay_buffer if s[2] == 1]
            fails = [s for s in self.replay_buffer if s[2] == 0]

            # ---- 至多 1/4 为正样本 --------------------------------
            k_succ = min(len(successes), 16)  # 上限 16
            k_fail = 64 - k_succ
            batch = random.sample(successes, k_succ) + random.sample(fails, k_fail)
            random.shuffle(batch)  # 打乱次序
            # --------------------------------------------------------

            states, actions, labels = zip(*batch)

            # 强制扁平化
            states = [s.view(-1) for s in states]

            # 裁剪为最短长度（避免 stack 报错）
            min_len = min(s.shape[0] for s in states)
            states = [s[:min_len] for s in states]

            # 统一 stack
            states = torch.stack(states, dim=0).to(self.device)

            # labels: List[int] -> (B,)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            # 2) 前置清梯度
            self.optimizer.zero_grad()
            # 3) 前向 + 反向
            logits = self.replay_head(states)  # (B, 2)
            loss = F.cross_entropy(logits, labels)  # 两类: 0=fail,1=success
            loss.backward()
            # 4) 更新参数
            self.optimizer.step()
            self.scheduler.step()

        # 保持探索型 emitter 始终占据 10% 比例（否则自动补充）
        if self.current_step % 50 == 0:
            self._maintain_explorer_emitter_ratio()

        # === meta-learning ===
        if self.current_step % 1000 == 0:
            tasks = self._sample_meta_tasks()  # 从 replay_buffer 中采 support/query
            if tasks:  #  非空才更新
                self.meta_trainer.meta_update(tasks)
        self.meta_self_evaluation()

        # === 定期修剪 Memory ===
        if self.current_step % 2000 == 0:
            self.memory.trim(keep_last=800)
        self.guided_prob = max(0.0, self.guided_prob - self.guided_decay)

    def _sample_meta_tasks(self, num_tasks=5, k_support=4, k_query=4):
        """
        从 replay_buffer 中采样元学习任务（support + query）
        每个任务是一个字典：{ 'support': List[(x, y)], 'query': List[(x, y)] }
        """
        tasks = []
        if len(self.replay_buffer) < (num_tasks * (k_support + k_query)):
            return []

        for _ in range(num_tasks):
            batch = random.sample(self.replay_buffer, k_support + k_query)
            support = batch[:k_support]
            query = batch[k_support:]
            tasks.append({
                "support": support,
                "query": query
            })

        return tasks

    def _maybe_report_by_stage(self):
        stage = self._get_virus_stage()
        if stage != self.last_report_stage:
            logger.info(f"[阶段切换] 进入 {stage} 阶段")
            self.report_antibody_stats()
            self.last_report_stage = stage

    def _apply_and_reward(self, unit, action):
        """
        对单个 emitter 动作：
          1. 读 before
          2. perform(action)
          3. 读 after
          4. 根据 hit 给 energy + / -
        返回 hit(bool)
        """
        x, y = action["target"]
        # 1) before
        if action["type"] == ACTION_HACK_DEFENSE:
            before = self.env.privilege_level[y, x].item()
            stealth_before = self.env.hack_strength[y, x].item()
        else:
            before = self.env.infected_map[y, x].item()

        # 2) 真正执行一次 perform
        self.emitter_actions.perform(action)

        # 3) after + hit 判定
        if action["type"] == ACTION_HACK_DEFENSE:
            after = self.env.privilege_level[y, x].item()
            hit = (before > 0.05 and after == 0.0)
            # === 新增：隐蔽黑客清除奖励 ===
            # 把执行 perform 之前的 hack_strength 记录下来
            stealth_after = self.env.hack_strength[y, x].item()
            stealth_hit = (stealth_before > 0.05 and stealth_after == 0.0)
            if stealth_hit:
                bonus = 1.5  # 隐蔽清除奖励，可根据情况调整
                unit.energy += bonus
                unit.meta.record(action="stealth_kill", reward=bonus)

        else:
            after = self.env.infected_map[y, x].item()
            hit = (before > 0.5 and after == 0.0)

        # 4) 奖励 / 惩罚
        factor = GUIDED_FACTOR if getattr(unit, "guided_this_round", False) else 1.0
        if hit:
            bonus = HIT_BONUS * factor
            unit.energy += bonus
            unit.meta.record(action="hit_bonus", reward=bonus)
            step_reward = bonus
            logger.debug("恭喜你打中啦")
        else:
            unit.energy = max(0.0, unit.energy - MISS_PENALTY)
            unit.meta.record(action="miss_penalty", reward=-MISS_PENALTY)
            step_reward = -MISS_PENALTY
        # === 新增：把 hack_defense 的成功/失败都存回放池 ===
        if action["type"] == ACTION_HACK_DEFENSE:
            # state_vec 可用 unit.get_output() 或者从外层传入
            state_vec = unit.get_output().detach().view(-1).cpu()
            # 成功的定义：privilege_level 由 >0.05 变为 0 OR hack_strength 由 >0.05 变为 0
            success = ((before > 0.05 and after == 0.0) or
                       (stealth_before > 0.05 and stealth_after == 0.0))
            self.replay_buffer.append((state_vec, action, int(success)))

        # —— 同步策略网络学习 —— #
        # 用当前 env_state + last_flat_idx + 这一步 reward 直接更新
        # 先拿到最新的 env_state tensor
        state = self.env.get_state_tensor().view(1, -1).to(self.device)
        self.policy_update(state, self.last_flat_idx, step_reward)

        return hit

    def _record_antibody_effectiveness(self, action: dict, feat: torch.Tensor):
        """判断抗体动作是否成功清除感染，并更新计数器"""
        if not action or "target" not in action:
            return

        x, y = action["target"]
        if not (0 <= x < self.env.size and 0 <= y < self.env.size):
            return

        infected_before = self.env.infected_map[y, x].item()
        self.emitter_actions.perform(action)
        infected_after = self.env.infected_map[y, x].item()

        if infected_before > 0.5 and self.env.infected_map[y, x].item() == 0.0:
            self.antibody_success_count += 1
        else:
            self.antibody_failure_count += 1
        # 把本次防御经验加入回放池
        state_vec = feat.detach().view(-1).cpu()
        success = (infected_before > 0.5 and infected_after == 0.0)
        self.replay_buffer.append((state_vec, action, success))

        # —— 2) 对抗体分类头做一次微调 —— #
        logits = self.immune_clf(feat.view(1, -1))          # [1,1]
        label  = torch.tensor([[float(success)]], device=self.device)
        loss   = F.binary_cross_entropy_with_logits(logits, label)
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

        """Emitter 达到 personal_goal 时的内在奖励"""
        for u in self.units:
            if getattr(u, "goal_type", "") == "hack":
                setattr(u, "intrinsic_reward", 0.3)
            else:
                setattr(u, "intrinsic_reward", 0.2)

            if u.role=="emitter" and hasattr(u,"personal_goal") and u.personal_goal:
                out = u.get_output().flatten()
                pred = torch.argmax(out).item()
                if (pred % self.env.size, pred//self.env.size)==u.personal_goal:
                    r = getattr(u,"intrinsic_reward",0.2)
                    u.energy += r
                    u.meta.record(action="intrinsic", reward=+r)
                    u.visit_counts[u.personal_goal]=u.visit_counts.get(u.personal_goal,0)+1
                    u.personal_goal = None
                    u.goal_type = None

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
        # --- A. 靠近病毒微奖励 / 远离扣分 ---------------------------------

        virus_coords = torch.nonzero(self.env.infected_map > 0)
        if len(virus_coords) > 0:
            for u in (x for x in self.units if x.role == "emitter"):
                if not hasattr(u, "output_positions"):  # ★ 保证存在
                    u.output_positions = deque(maxlen=20)

                # ---------- NEW: 初始化并记录最近 20 步坐标轨迹 ----------
                if hasattr(u, "position"):  # 当前位置落网格
                    u.output_positions.append(u.position)
                # ----------------------------------------------------------

                if not hasattr(u, "latest_base_reward"):
                    u.latest_base_reward = 0.0
                    u.last_rewarded_target_idx = None

                # 最近病毒的曼哈顿距离
                dists = [
                    abs(u.position[0] - x.item()) + abs(u.position[1] - y.item())
                    for y, x in virus_coords
                ]
                idx_min, dmin = min(enumerate(dists), key=lambda t: t[1])

                # 距离阈值  ⇒  小奖 0.05
                if dists[idx_min] <= 4:
                    if u.last_rewarded_target_idx != idx_min:
                        u.energy += 0.08
                        u.meta.record(action="approach_bonus", reward=+0.05)
                        u.latest_base_reward = 0.08
                        u.last_rewarded_target_idx = idx_min
                else:
                    # 离开后把 base 奖扣回去（一次性）
                    if u.latest_base_reward > 0:
                        u.energy = max(0.0, u.energy - u.latest_base_reward)
                        u.meta.record(action="leave_penalty",
                                      reward=-u.latest_base_reward)
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
        just_stale = torch.nonzero((curr_dur >= 5) & (prev_dur < 5))

        for y, x in just_stale:
            y, x = int(y), int(x)
            emitters = [
                u for u in self.units
                if u.role == "emitter" and hasattr(u, "position")
                   and not getattr(u, "is_permanent_explorer", False)
            ]
            if emitters:
                # 改成 “最远” 而不是最近
                farthest = max(
                    emitters,
                    key=lambda u: abs(u.position[0] - x) + abs(u.position[1] - y)
                )
                penalty = 0.1
                farthest.energy = max(0.0, farthest.energy - penalty)
                farthest.meta.record(action="persistence_penalty", reward=-penalty)
                logger.info(
                    f"[滞留惩罚] emitter {farthest.id} 扣能量 {penalty},摸鱼的下场！"
                )

            # 标记该点已惩罚
            # self.punished_map[y, x] = True

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
            for y, x in torch.nonzero(self.env.privilege_level > 0.05)
        }
        penalty_per_node = 0.1  # 每个未清除提权节点，对应 emitter 扣的能量

        total_cleared = 0
        # —— 4) 遍历所有 emitter，一次性处理 清理奖励 + 黑客惩罚 —— #
        for u in [u for u in self.units if u.role == "emitter"]:
            # —— 4.1) 病毒清理奖励 —— #
            cleared = len(getattr(u, "cleared_positions", set()))
            if cleared > 0:
                total_cleared += cleared
                reward = 1.0 * cleared
                if getattr(u, "guided_this_round", False):
                    reward *= GUIDED_FACTOR  # ← guided 击杀折扣
                u.energy += reward

                u.meta.record(action="defense", reward=reward)
                logger.debug("真棒，干掉了个病毒")

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
                    u.energy -= 0.06
                    u.meta.record(action="idle_penalty", reward=-0.06)

                # 打圈：20 步内只覆盖 ≤3 个不同格子
                uniq = len(set(u.output_positions))
                if uniq <= 3:
                    u.energy -= 0.08
                    u.meta.record(action="loop_penalty", reward=-0.08)
            # ----------------------------------------------------------

        # —— 6) 黑客防御奖励 —— #
        cleared_priv = (prev_priv > 0.05).sum() - (self.env.privilege_level > 0.05).sum()
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


            # 先处理“移动”动作
            if action and action["type"] == "move":
                unit.position = action["target"]
                action = self._decode_action_from_output(unit, action_vec)

            if not action or action["type"] == "move":
                continue

            hit = self._apply_and_reward(unit, action)
            x, y = action["target"]

            # 记录 hack 清理集合 & 统计
            if action["type"] == ACTION_HACK_DEFENSE and hit:
                unit.cleared_hack = getattr(unit, "cleared_hack", set())
                unit.cleared_hack.add((x, y))
                src = "guided" if getattr(unit, "guided_this_round", False) else "self_direct"
                # 全局累计
                self.hack_kill_stats[src] += 1
                # —— 新增：按黑客类型分类统计 —— #
                hack_type = self.env.hacks.get((x, y), {}).get("type", "unknown")
                bucket = self.hack_kill_stats_by_type.setdefault(
                    hack_type, {"self_direct": 0, "guided": 0}
                )
                bucket[src] += 1


            # 记录 virus 清理
            if action["type"] == ACTION_BLOCK and hit:
                src = "guided" if getattr(unit, "guided_this_round", False) else "self_direct"
                # 全局累计
                self.kill_stats[src] += 1
                # —— 新增：按病毒类型分类统计 —— #
                virus_type = "virus"  # 如有多种，可从环境或 action 中提取
                bucket_v = self.virus_kill_stats_by_type.setdefault(
                    virus_type, {"self_direct": 0, "guided": 0}
                )
                bucket_v[src] += 1
                unit.cleared_positions.add((x, y))
                logger.debug(f"[清除记录] emitter {unit.id} 在 {(x, y)} 清除了一个病毒 ({src})")

    def _decode_action_from_output(self, unit, output_vec):
        # === hacker 类型优先级（越大越危险） ===
        risk_weight = {
            "privilege_escalation": 3.0,
            "lateral_move":        2.5,
            "bruteforce":          2.0,
            "phishing":            1.5
        }
        # 1) 网络给出的 raw index
        raw_idx = int(torch.argmax(output_vec).item())
        size    = self.env.size
        ry, rx  = divmod(raw_idx, size)

        # 2) 在 guided_prob 范围内尝试硬编码覆盖
        guided_choice = None
        if random.random() < self.guided_prob:
            # —— 高风险黑客覆盖 —— #
            if self.env.hacks:
                cx, cy = unit.position
                items = list(self.env.hacks.items())
                items.sort(key=lambda it: (
                    -risk_weight.get(it[1]['type'], 1.0),
                    abs(cx - it[0][0]) + abs(cy - it[0][1])
                ))
                (gx, gy), _ = random.choice(items[:3])
                guided_choice = (gy, gx)
            # —— 病毒覆盖 (40% 概率) —— #
            if guided_choice is None and self.env.infected_map.sum() > 0 and random.random() < 0.4:
                cx, cy = unit.position
                virus_coords = torch.nonzero(self.env.infected_map).tolist()
                if virus_coords:
                    gy, gx = min(virus_coords, key=lambda c: abs(c[1] - cx) + abs(c[0] - cy))
                    guided_choice = (gy, gx)

        # 3) 最终坐标：guided 优先，否则用网络
        if guided_choice is not None:
            y, x = guided_choice
            flat_idx = y * size + x
            unit.guided_this_round = True
        else:
            y, x = ry, rx
            flat_idx = raw_idx
            unit.guided_this_round = False


        # ★FIX: 先裁剪一次，保证后面第一次索引安全
        H, W = self.env.infected_map.shape  # 用 infected_map 的真实尺寸
        y = min(y, H - 1)
        x = min(x, W - 1)

        # 如果黑客存在，从“Top-3 高风险”中随机选一个，而不是永远第1
        if self.env.hacks:
            cx, cy = unit.position
            items = list(self.env.hacks.items())
            # 先按 (risk, -distance) 排序，risk 越大排越前
            items.sort(key=lambda it: (
                -risk_weight.get(it[1]['type'], 1.0),
                abs(cx - it[0][0]) + abs(cy - it[0][1])
            ))
            top_k = items[:min(3, len(items))]
            (x_sel, y_sel), _ = random.choice(top_k)
            y, x = y_sel, x_sel
            unit.guided_this_round = True


        # ★PATCH 3-A: 若网络选的格子没病毒…
        if self.env.infected_map.sum() > 0 and self.env.infected_map[y, x] == 0:
            if random.random() < 0.4:
                cx, cy = unit.position
                virus_coords = torch.nonzero(self.env.infected_map).tolist()
                if virus_coords:
                    vy, vx = min(virus_coords,
                                 key=lambda c: abs(c[1] - cx) + abs(c[0] - cy))
                    y, x = vy, vx
                unit.guided_this_round = True  # ← 标记“被系统指引”
        else:
            unit.guided_this_round = False  # ← 正常自主
            # ---------- 若格子也不是 hack，但仍有提权节点 → 指向最近 hack ---
            if not unit.guided_this_round and self.env.privilege_level.sum() > 0 \
                    and self.env.privilege_level[y, x] <= 0.02:
                cx, cy = unit.position
                hack_coords = torch.nonzero(self.env.privilege_level > 0.05).tolist()
                if hack_coords:
                    hy, hx = min(hack_coords,
                                 key=lambda c: abs(c[1] - cx) + abs(c[0] - cy))
                    y, x = hy, hx
                    unit.guided_this_round = True

        if not unit.guided_this_round and self.env.privilege_level.sum() > 0:
            if random.random() < 0.4:
                cx, cy = unit.position
                hack_coords = torch.nonzero(self.env.privilege_level > 0.05).tolist()
                if hack_coords:
                    hy, hx = min(hack_coords,
                                 key=lambda c: abs(c[1] - cx) + abs(c[0] - cy))
                    y, x = hy, hx
                    unit.guided_this_round = True

        # -----------
        # —— 防止张量实际 shape 比 size 小（扩容延迟同步）——
        H, W = self.env.privilege_level.shape
        y = min(y, H - 1)
        x = min(x, W - 1)

        # ---------- 距离 >1 先移动 ----------
        cx, cy = unit.position
        if abs(cx - x) + abs(cy - y) > 1:
            nx, ny = self._step_toward((cx, cy), (x, y))
            self.last_flat_idx = flat_idx
            return {"type": "move", "target": (nx, ny)}
        # ------------------------------------

        if self.env.privilege_level[y, x] > 0.02:
            a_type = ACTION_HACK_DEFENSE
        elif self.env.infected_map[y, x] > 0.5:
            a_type = ACTION_BLOCK
        else:
            a_type = ACTION_QUARANTINE
        self.last_flat_idx = flat_idx
        return {"type": a_type, "target": (x, y)}

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
                expected_in = self.env_size * self.env_size * INPUT_CHANNELS
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

        # 3) 拼接
        full_state = torch.cat([env_flat, goal_flat], dim=1)

        # 4) 前向算 logits
        sensor_out = self.sensor_forward(full_state)
        proc_out = self.processor_forward(sensor_out)
        logits = self.emitter_net(proc_out)  # [1, seq_len]
        probs = torch.softmax(logits, dim=-1)

        # —— 新增：保证索引合法 —— #
        seq_len = probs.size(1)
        flat_idx = flat_idx % seq_len  # 或者用 min(flat_idx, seq_len-1)

        # 5) REINFORCE 更新
        log_p = torch.log(probs[0, flat_idx] + 1e-8)
        loss = - log_p * reward

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


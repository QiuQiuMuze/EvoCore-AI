import torch
import random
from collections import deque
import torch.nn.functional as F

class GridSecurityEnv:
    def __init__(self, size=10, device='cpu', difficulty_ramp=5000, spawn_interval=200):
        """
        初始化环境。
        - size：网格大小（默认 10×10）
        - device：张量所在设备
        - difficulty_ramp：从友好期过渡到完全攻击状态所需步数
        - spawn_interval：每隔多少步新增一个入侵事件
        """
        self.size = size
        self.device = device
        self.difficulty_ramp = difficulty_ramp
        self.spawn_interval = spawn_interval
        self.infected_duration_map = torch.zeros((self.size, self.size), dtype=torch.int32, device=self.device)
        self.reset()
        self.hacks = {}

    def reset(self):
        """
        重置整个环境，包括感染图、攻击历史、行为评分等。
        初始化第一波攻击。
        """
        self.traceback_log = []
        self.step_count = 0
        self.infected_map = torch.zeros((self.size, self.size), dtype=torch.float32, device=self.device)
        self.infection_strength = torch.zeros_like(self.infected_map)
        self.attack_history = torch.zeros_like(self.infected_map)
        self.behavior_score = torch.rand_like(self.infected_map) * 0.1  # 初始行为评分较低
        self.is_quarantined = torch.zeros_like(self.infected_map)
        self.visited_map = torch.zeros((self.size, self.size), dtype=torch.bool, device=self.device)
        # —— 新增多模态：网络流量 & 权限等级 —— #
        self.net_traffic = torch.zeros_like(self.infected_map)  # 模拟流量热图
        self.perm_level = torch.zeros_like(self.infected_map)  # 模拟权限等级分布

        # —— 新增：系统调用历史（用于入侵检测） —— #
        self.syscall_history = deque(maxlen=20)  # 保存最近 20 步的 syscall 特征
        # —— 新增：黑客入侵相关 —— #
        # 节点被黑客攻破的次数 & 强度
        self.hack_history    = torch.zeros_like(self.infected_map)
        self.hack_strength   = torch.zeros_like(self.infected_map)
        # 每个节点的“脆弱度”（越高越容易被攻破）
        self.vulnerability   = torch.rand_like(self.infected_map) * 0.5
        # 模拟提权后的权限等级（0=普通，1=root）
        self.privilege_level = torch.zeros_like(self.infected_map)
        # 登录失败次数（暴力破解检测）
        self.login_failures  = torch.zeros_like(self.infected_map)

        # 支持的攻击类型及参数
        self.attack_types = {
            'worm':    {'spread_prob': 0.4, 'stealth': 0.0, 'burst': False},
            'trojan':  {'spread_prob': 0.1, 'stealth': 0.6, 'burst': False},
            'scan':    {'spread_prob': 0.0, 'stealth': 1.0, 'burst': True,  'burst_chance': 0.3, 'burst_area': 2},
            'ransom':  {'spread_prob': 0.2, 'stealth': 0.3, 'burst': True,  'burst_chance': 0.1, 'burst_area': 1},
            'apt':     {'spread_prob': 0.3, 'stealth': 0.8, 'burst': True,  'burst_chance': 0.1, 'burst_area': 3},
        }

        self.attacks = {}
        self.hack_spawn_interval = 10
        self.hack_types = {
            'bruteforce':      {'spawn_prob': 0.02, 'max_fail': 5},
            'phishing':        {'spawn_prob': 0.01, 'stealth':0.8},
            'lateral_move':    {'spawn_prob': 0.005},
            'privilege_escalation': {'spawn_prob': 0.003}
        }
        self.hacks = {}  # 当前活跃的黑客事件: dict[(x,y)]→{type,…}
        # if self.step_count >= 0:  # 比如只在后期才允许初始病毒出现
        #     self._spawn_attack(initial=True)


    # === 新增：统计当前活跃黑客 ===
    def get_hack_stats(self):
        """
        返回一个 dict：
        {
            'per_type': {'bruteforce':3, 'phishing':1, ...},
            'total_priv': float,          # ∑ privilege_level
            'threat_score': float         # ∑ hack_strength
        }
        """
        per_type = {t: 0 for t in self.hack_types}
        for (_, _), info in self.hacks.items():
            per_type[info['type']] += 1

        total_priv = float(self.privilege_level.sum().item())
        threat     = float(self.hack_strength.sum().item())
        total_events = int((self.hack_strength > 0).sum().item())
        return {'per_type': per_type,
                'total_priv': total_priv,
                'threat_score': threat,
                'total_events': total_events}


    def _spawn_attack(self, initial=False):
        """
        创建新的攻击点。
        - 会根据当前 step_count 决定允许的攻击类型（早期较弱）
        - 随机选择攻击类型与位置
        """
        if self.step_count < self.difficulty_ramp * 0.3:
            available = ['worm', 'trojan']
        elif self.step_count < self.difficulty_ramp * 0.6:
            available = ['worm', 'trojan', 'scan', 'ransom']
        else:
            available = list(self.attack_types.keys())

        attack_type = random.choice(available)
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        params = self.attack_types[attack_type]
        self.attacks[(x, y)] = {'type': attack_type, 'power': params.get('power', 1.0)}

        self.infected_map[y, x] = 1.0
        self.infection_strength[y, x] = params.get('power', 1.0)

    def _expand_environment(self, growth=2):
        old_size = self.size
        new_size = old_size + growth
        new_size = min(40, new_size)
        self.size = new_size

        def expand(tensor, fill=0.0):
            new_t = torch.full(
                (new_size, new_size),
                fill,
                dtype=tensor.dtype,
                device=tensor.device
            )
            new_t[:old_size, :old_size] = tensor
            return new_t

        # —— 扩展原有的各种图 —— #
        self.infected_map = expand(self.infected_map)
        self.infected_duration_map = expand(self.infected_duration_map, fill=0)

        self.infection_strength = expand(self.infection_strength)
        self.attack_history = expand(self.attack_history)
        self.behavior_score = expand(self.behavior_score)
        self.is_quarantined = expand(self.is_quarantined)
        self.visited_map = expand(self.visited_map, fill=False)
        self.net_traffic = expand(self.net_traffic)
        self.perm_level = expand(self.perm_level)

        # —— 一并扩展所有“黑客”相关张量 —— #
        # 已有历史要保留，新节点 vulnerability 可以给个随机初始
        self.hack_history = expand(self.hack_history)
        self.hack_strength = expand(self.hack_strength)
        self.vulnerability = expand(self.vulnerability, fill=random.random() * 0.5)
        self.privilege_level = expand(self.privilege_level)
        self.login_failures = expand(self.login_failures)

        # 保持 device 一致
        self.device = self.infected_map.device

        # —— 扩展 syscall_history 中的历史帧 —— #
        new_syscalls = deque(maxlen=20)
        for t in self.syscall_history:
            pad_t = torch.full((self.size, self.size), 0.0, device=self.device)
            old_size = t.shape[0]
            pad_t[:old_size, :old_size] = t
            new_syscalls.append(pad_t)
        self.syscall_history = new_syscalls

    def resize(self, new_size):
        growth = new_size - self.size
        if growth > 0:
            self._expand_environment(growth=growth)

    def get_nearest_resource_to(self, pos):
        # 兼容 CogGraph 的调用，直接返回一个“无资源”值
        return None, float('inf')

    def get_nearest_danger_to(self, pos):
        return None, float('inf')

    def remove_unit(self, unit):
        # 只有 unit.should_die() 返回 True 时才能被真正移除
        if getattr(unit, "_allow_external_kill", False):
            # 如果未来有必要的外部 kill，再打开这行
            super().remove_unit(unit)
        else:
            # 否则忽略所有外部 remove_unit 调用
            return
    def step(self):
        """
        执行一步攻击扩散与演化。
        - 根据当前攻击类型传播或爆发
        - 自动更新攻击强度、感染图、历史攻击记录
        - 每 N 步生成一个新攻击
        """

        # if self.step_count % 1 == 0:
        if self.step_count % 1000 == 0 and self.step_count >= 1000:
            decay_mask = torch.rand_like(self.visited_map, dtype=torch.float32) < 0.1  # 10% decay
            self.visited_map[decay_mask] = False
        # 更新感染持续时间：+1 或清零
        new_infected = self.infected_map > 0.5
        self.infected_duration_map[new_infected] += 1
        self.infected_duration_map[~new_infected] = 0

        # —— 新增：记录本步系统调用特征 —— #
        self.syscall_history.append(self._extract_syscall_vector())

        self.step_count += 1
        difficulty = min(1.0, self.step_count / self.difficulty_ramp)
        new_attacks = {}

        # 先算当前感染点数 & 分段上限
        curr = int((self.infected_map > 0.5).sum().item())
        if   self.step_count < 1000: max_inf = 10
        elif self.step_count < 2000: max_inf = 20
        elif self.step_count < 5000: max_inf = 40
        else:                        max_inf = float('inf')
        if self.step_count >= 0:
            # ────────── 1) 4-邻扩散：一次卷积完成 ──────────
            infected = (self.infected_map > 0.5).float().unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            kernel = torch.tensor([[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]],
                                  dtype=infected.dtype,  # <-- 关键
                                  device=self.device).view(1, 1, 3, 3)

            nbr_cnt  = F.conv2d(infected, kernel, padding=1).squeeze()               # [H,W]
            cand     = (nbr_cnt > 0) & (self.infected_map == 0) & (self.is_quarantined == 0)

            # spread_prob_map：同一格只要有邻居感染，就取邻居里**最大的** spread_prob
            spread_prob_map = torch.zeros_like(self.infected_map)
            if self.attacks:                                  # self.attacks 是 dict[(x,y)]→info
                # 先准备一张 “当前每格攻击类型 id” 的矩阵
                type_map = torch.full_like(self.infected_map, -1, dtype=torch.long)
                for (x, y), info in self.attacks.items():
                    type_idx = list(self.attack_types).index(info['type'])   # 0,1,2…
                    type_map[y, x] = type_idx

                for idx, (t, p) in enumerate(self.attack_types.items()):
                    if p['spread_prob'] == 0:        # 没有扩散能力
                        continue
                    mask = (type_map == idx)
                    if mask.any():
                        spr = p['spread_prob'] * (1 + difficulty * 0.5)
                        spread_prob_map = torch.where(mask, torch.full_like(spread_prob_map, spr), spread_prob_map)

                # ---------- 计算 spread_prob_map（同一格取邻居中最大的 spread_prob） ----------
                if self.attacks:
                    type_map = torch.full_like(self.infected_map, -1, dtype=torch.long)
                    for (x, y), info in self.attacks.items():
                        type_idx = list(self.attack_types).index(info['type'])
                        type_map[y, x] = type_idx

                    prob_seed = torch.zeros_like(self.infected_map)  # 先只在感染源格写 spr
                    for idx, (t, p) in enumerate(self.attack_types.items()):
                        if p['spread_prob'] == 0:
                            continue
                        spr = p['spread_prob'] * (1 + difficulty * 0.5)
                        prob_seed[type_map == idx] = spr

                    # >>> PATCH begin —— 把 spr 扩散到 4-邻域，邻居取最大的 spread_prob
                    kernel_4 = torch.tensor([[0, 1, 0],
                                             [1, 1, 1],
                                             [0, 1, 0]],
                                            dtype=infected.dtype,
                                            device=self.device).view(1, 1, 3, 3)

                    spread_prob_map = F.conv2d(prob_seed.unsqueeze(0).unsqueeze(0),
                                               kernel_4, padding=1).squeeze()
                else:
                    spread_prob_map = torch.zeros_like(self.infected_map)

                rand_mat = torch.rand_like(self.infected_map)
                new_inf = cand & (rand_mat < spread_prob_map)

                self.infected_map[new_inf]       = 1.0
                self.infection_strength[new_inf] = 1.0

            # ────────── 2) burst 爆发传播 ──────────
            for t, p in self.attack_types.items():
                if not p.get('burst'):
                    continue
                area  = p['burst_area']
                bmask = torch.zeros_like(self.infected_map, dtype=torch.bool)
                for (x, y), info in self.attacks.items():
                    if info['type'] == t:
                        bmask[y, x] = True
                if not bmask.any():
                    continue

                # 对 bmask 膨胀 (2*area+1)^2
                k = torch.ones(1,1, 2*area+1, 2*area+1, device=self.device)
                burst_area = (F.conv2d(bmask.float()[None,None], k, padding=area) > 0).squeeze()
                m1   = burst_area & (self.is_quarantined == 0) & (self.infected_map == 0)
                m2   = torch.rand_like(self.infected_map) < p['burst_chance'] * (1 + difficulty)
                burst_new = m1 & m2
                self.infected_map[burst_new]       = 1.0
                self.infection_strength[burst_new] = 1.0

        self.attacks.update(new_attacks)

        self.attack_history += self.infected_map
        # —— 模拟网络流量 & 权限变化 —— #
        self.net_traffic = torch.rand_like(self.net_traffic) * 0.05 + self.infected_map * 0.1
        self.perm_level = torch.clamp(self.perm_level + (self.infected_map * 0.01), 0.0, 1.0)

        self.attack_history.clamp_(0, 10)

        curr = int((self.infected_map > 0.5).sum().item())

        # 代替固定间隔触发入侵
        spawn_chance = 20 / self.spawn_interval  # e.g. 每步有 20/200 概率
        if curr < max_inf and random.random() < spawn_chance and self.step_count >= 0:
            self._spawn_attack()

        # —— 调整：每 hack_spawn_interval 步才做一次 spawn —— #
        if self.step_count >= 1000 and self.step_count % self.hack_spawn_interval == 0:
            for ht, params in self.hack_types.items():
                if random.random() < params['spawn_prob']:
                    x, y = random.randrange(self.size), random.randrange(self.size)
                    self.hacks[(x,y)] = {'type': ht, 'power': params.get('stealth',1.0)}
                    # 马上标记一次
                    self.hack_history[y,x]  += 1
                    self.hack_strength[y,x]  = params.get('stealth',1.0)
                    self.privilege_level[y, x] = 1.0

            # 传播 or 行动
            new_hacks = {}
            if self.hacks:
                idx = torch.tensor(list(self.hacks.keys()), device=self.device).T  # [2,N]
                neigh = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]],
                                     device=self.device, dtype=torch.long).T  # ➜ [2,4]
                dst = (idx.unsqueeze(1) + neigh.unsqueeze(2)).reshape(2, -1)

                inb = (dst[0] >= 0) & (dst[0] < self.size) & (dst[1] >= 0) & (dst[1] < self.size)
                dst = dst[:, inb]
                free = (self.is_quarantined[dst[1], dst[0]] == 0)
                dst = dst[:, free]
                if dst.numel():
                    vul = self.vulnerability[dst[1], dst[0]]
                    p = 0.1  # 可以根据 hack 类型决定
                    keep = (torch.rand_like(vul) < vul * p)
                    dst = dst[:, keep]
                    self.hack_history.index_put_((dst[1], dst[0]),
                                                 torch.ones(dst.shape[1], device=self.device),
                                                 accumulate=True)
                    self.hack_strength[dst[1], dst[0]] = p
                    for x, y in zip(dst[0].tolist(), dst[1].tolist()):
                        self.hacks[(x, y)] = {'type': 'lateral_move', 'power': p}

    def bind_units_reference(self, units):
        """
        安全绑定：将 CogGraph 中的 self.units 映射进来，供环境判断位置占用等使用。
        """
        self._external_units = units

    def get_state_tensor(self):
        """
        返回状态张量，供 sensor 感知。
        通道结构：
        0: 是否感染
        1: 感染强度
        2: 累计感染频次
        3: 当前行为评分
        4: 是否被 quarantine
        """
        # 将 syscalls 序列做平均，得到 [H, W] 特征
        if not self.syscall_history:
            self.syscall_history.append(torch.zeros((self.size, self.size), dtype=torch.float32, device=self.device))
        syscall_feat = torch.stack(list(self.syscall_history), dim=0).mean(dim=0)

        return torch.stack([
            self.infected_map,
            self.infection_strength,
            self.attack_history,
            self.behavior_score,
            self.is_quarantined,
            self.visited_map.float(),
            self.net_traffic,
            self.perm_level,
            syscall_feat,
            # —— 新增黑客相关 —— #
            self.hack_history,
            self.hack_strength,
            self.vulnerability,
            self.privilege_level,
            self.login_failures
        ], dim=0).clone()

    def block_connection(self, pos):
        """
        emitter 执行 block 动作：清除该格子感染
        """
        x, y = pos
        self.infected_map[y, x] = 0.0
        self.infection_strength[y, x] = 0.0
        self.attacks.pop((x, y), None)

    def quarantine_zone(self, pos):
        """
        emitter 执行 quarantine：标记该格子不可传播
        """
        x, y = pos
        self.is_quarantined[y, x] = 1.0

    def mark_suspicious(self, pos):
        """
        emitter 执行 mark：将某格子行为评分增加
        """
        x, y = pos
        self.behavior_score[y, x] += 0.2

    def is_attack_successful(self, targets=None):
        """
        判断攻击是否成功抵达关键区域（默认右下角）。
        可用于训练奖励的惩罚判断。
        """
        if targets is None:
            targets = [(self.size - 1, self.size - 1)]
        return any(self.infected_map[y, x] > 0 for x, y in targets)

    def _extract_syscall_vector(self) -> torch.Tensor:
        """
        模拟获取当前系统调用特征向量，shape=[H, W]。
        """
        # 这里用随机噪声做示例
        return torch.randn((self.size, self.size), device=self.device) * 0.01

    def clone(self):
        # 1) 新建一个空白 env
        new = GridSecurityEnv(
            size=self.size,
            device=self.device,
            difficulty_ramp=self.difficulty_ramp,
            spawn_interval=self.spawn_interval
        )
        # 2) 把所有“运行时”张量都一一 clone
        for name in (
            "infected_map", "infected_duration_map", "infection_strength",
            "attack_history", "behavior_score", "is_quarantined",
            "visited_map", "net_traffic", "perm_level",
            "hack_history", "hack_strength", "vulnerability",
            "privilege_level", "login_failures"
        ):
            setattr(new, name, getattr(self, name).clone())

        # 3) 深拷贝 deque 里的历史帧
        from collections import deque
        new.syscall_history = deque(
            (t.clone() for t in self.syscall_history),
            maxlen=self.syscall_history.maxlen
        )

        # 4) 普通字典也一并拷贝
        new.attacks = dict(self.attacks)
        new.hacks   = dict(self.hacks)

        return new



# 安全包装器：屏蔽旧接口
class SecureGridEnv(GridSecurityEnv):
    def consume_resource(self, *args, **kwargs): return 0.0
    def get_nearest_resource_to(self, *args, **kwargs): return None
    def get_nearest_danger_to(self, *args, **kwargs): return None
    def get_agent_pos(self): return (-1, -1)
    def has_resource(self, *args, **kwargs): return False
    def has_hazard(self, *args, **kwargs): return False
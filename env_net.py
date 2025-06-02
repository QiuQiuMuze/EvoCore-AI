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
        # 用于 4-邻卷积的固定 kernel，只需创建一次
        self._kernel4 = torch.tensor([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]],
                                     dtype=torch.float32,
                                     device=self.device).view(1, 1, 3, 3)
        # 用于 burst 膨胀的最大可能 kernel（pad area 最大为 4 时，kernel 大小 = 2*4+1 = 9）
        # 这里提前分配一个 9×9 的 all-ones，用于后续任意 area ≤ 4 的 burst
        self._max_burst_kernel = torch.ones(1, 1, 9, 9, device=self.device)

        # —— 扁平 one-hot，每个细胞在黑客字典里就置 1，否则 0 —— #
        self._hack_onehot = torch.zeros(self.size * self.size, dtype=torch.float32, device=self.device)
        # 记录上一次 keys，用于增量更新 one-hot
        self._last_hack_keys = set()

        # —— 扁平方式存储“攻击类型索引”（–1 表示无攻击） —— #
        self._attack_type_flat = torch.full((self.size * self.size,),
                                            -1, dtype=torch.long, device=self.device)
        self._last_attack_keys = set()

        # 统计“随机更新”步数，用于降低频率
        self._rand_update_counter = 0

        self.infected_duration_map = torch.zeros((self.size, self.size), dtype=torch.int32, device=self.device)
        self.reset()
        self.hacks = {}

    def reset(self):
        """
        重置整个环境，包括感染图、攻击历史、行为评分等。
        初始化第一波攻击。
        """
        self.infected_duration_map.zero_()
        self._rand_update_counter = 0
        # 重置黑客 one-hot 缓冲及其“上一次键集”
        self._hack_onehot.zero_()
        self._last_hack_keys.clear()
        # 重置“攻击类型扁平张量”（全 –1）及它的键集合
        self._attack_type_flat.fill_(-1)
        self._last_attack_keys.clear()
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
            'worm':    {'spread_prob': 0.2, 'stealth': 0.0, 'burst': False},
            'trojan':  {'spread_prob': 0.05, 'stealth': 0.6, 'burst': False},
            'scan':    {'spread_prob': 0.0, 'stealth': 1.0, 'burst': True,  'burst_chance': 0.5, 'burst_area': 3},
            'ransom':  {'spread_prob': 0.1, 'stealth': 0.3, 'burst': True,  'burst_chance': 0.3, 'burst_area': 2},
            'apt':     {'spread_prob': 0.15, 'stealth': 0.8, 'burst': True,  'burst_chance': 0.4, 'burst_area': 4},
        }

        self.attacks = {}
        self.hack_spawn_interval = 50
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

        # —— 新增：把 (x,y) 对应的扁平索引设为 attack_type 的序号 —— #
        flat_idx = y * self.size + x
        type_idx = list(self.attack_types).index(attack_type)
        self._attack_type_flat[flat_idx] = type_idx
        # 同时把新键加入 _last_attack_keys（如果 _spawn_attack 也可能在 step() 里调用多次，保证最终增量同步）
        self._last_attack_keys.add((x, y))

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

        # 假设 new_size = self.size 已经更新
        old_flat_len = old_size * old_size
        new_flat_len = new_size * new_size

        # 先备份旧的 one-hot / 类型索引
        old_hack = self._hack_onehot
        old_type = self._attack_type_flat

        # 重新分配新的向量，先全部清零（或 -1）
        self._hack_onehot = torch.zeros(new_flat_len, dtype=torch.float32, device=self.device)
        self._attack_type_flat = torch.full((new_flat_len,), -1, dtype=torch.long, device=self.device)

        # 把原来对应的 [0:old_flat_len] 区域拷贝回去
        self._hack_onehot[:old_flat_len] = old_hack
        self._attack_type_flat[:old_flat_len] = old_type

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

        # —— 每帧先让随机更新计数器+1 —— #
        self._rand_update_counter += 1

        # 每 1000 步才对 visited_map 做一次衰减
        if self.step_count % 1000 == 0 and self.step_count >= 1000:
            decay_mask = torch.rand_like(self.visited_map, dtype=torch.float32) < 0.1
            self.visited_map[decay_mask] = False

        # —— 降低 net_traffic 和 perm_level 的更新频率 —— #
        # 例如设定“每 1 步 更新一次”，你可以把 1 调整成更大
        imp_count = 1
        if self._rand_update_counter % imp_count == 0:
            # 在这 1 帧里，只有第 1 帧会做一次更新——
            # 先给 visited_map 再做一次小概率衰减（可选，也可以只做 net_traffic & perm_level）
            decay_mask2 = torch.rand_like(self.visited_map, dtype=torch.float32) < 0.1
            self.visited_map[decay_mask2] = False

        # 更新感染持续时间：+1 或清零
        new_infected = self.infected_map > 0.04
        self.infected_duration_map[new_infected] += 1
        self.infected_duration_map[~new_infected] = 0

        # # —— 新增：记录本步系统调用特征 —— #
        # self.syscall_history.append(self._extract_syscall_vector())

        self.step_count += 1
        difficulty = min(1.0, self.step_count / self.difficulty_ramp)
        new_attacks = {}

        # 先算当前感染点数 & 分段上限
        curr = int((self.infected_map > 0.04).sum().item())
        if   self.step_count < 1000: max_inf = 10
        elif self.step_count < 2000: max_inf = 20
        elif self.step_count < 5000: max_inf = 40
        else:                        max_inf = float('inf')
        if self.step_count >= 0:
            # ────────── 1) 4-邻扩散：一次卷积完成 ──────────
            infected = (self.infected_map > 0.04).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            # kernel = torch.tensor([[0, 1, 0],
            #                        [1, 1, 1],
            #                        [0, 1, 0]],
            #                       dtype=infected.dtype,
            #                       device=self.device).view(1, 1, 3, 3)
            kernel = self._kernel4

            nbr_cnt = F.conv2d(infected, kernel, padding=1).squeeze()  # [H,W]
            cand = (nbr_cnt > 0) & (self.infected_map == 0) & (self.is_quarantined == 0)

            # —— 按阶段设定扩散倍率 factor_spread —— #
            if self.step_count < 1000:
                factor_spread = 0.5
            elif self.step_count < 2000:
                factor_spread = 1.0
            else:
                factor_spread = 2.0

            # ────────── 2) 计算 spread_prob_map —— 增量化版本 ──────────
            spread_prob_map = torch.zeros_like(self.infected_map)
            if self.attacks:
                # 直接从扁平向量 self._attack_type_flat 得到每个细胞的类型索引
                flat_type = self._attack_type_flat  # [size*size], 值 ∈ [-1..num_attack_types-1]
                # 先构建一个扁平 prob_seed_flat
                prob_seed_flat = torch.zeros_like(flat_type, dtype=self.infected_map.dtype)
                for idx, (_, p_dict) in enumerate(self.attack_types.items()):
                    base_p = p_dict['spread_prob']
                    if base_p <= 0.0:
                        continue
                    spr = base_p * (1 + difficulty * 0.5) * factor_spread
                    # 在扁平张量中，类型等于 idx 的位置全部置为 spr
                    mask_idx = (flat_type == idx)
                    prob_seed_flat[mask_idx] = spr
                # reshape 成 [H, W]
                prob_seed = prob_seed_flat.view(self.size, self.size)

                # 用预分配的 4-邻卷积 kernel 做扩散
                spread_prob_map = F.conv2d(prob_seed.unsqueeze(0).unsqueeze(0),
                                           self._kernel4, padding=1).squeeze()
            else:
                spread_prob_map = torch.zeros_like(self.infected_map)

            rand_mat = torch.rand_like(self.infected_map)
            raw_new_inf = cand & (rand_mat < spread_prob_map)  # 候选新增感染格

            if self.step_count < 2000:
                curr_inf = int((self.infected_map > 0.04).sum().item())
                remain_slots = 20 - curr_inf
                if remain_slots <= 0:
                    new_inf = torch.zeros_like(raw_new_inf)
                else:
                    candidates = torch.nonzero(raw_new_inf, as_tuple=False)  # [N,2]
                    N = candidates.size(0)
                    if N <= remain_slots:
                        new_inf = raw_new_inf.clone()
                    else:
                        # 在 GPU 上为每个候选点生成一个随机 score，然后 topk 取最小前 remain_slots 个
                        scores = torch.rand(N, device=self.device)
                        _, topk_idx = torch.topk(scores, k=remain_slots, largest=False)
                        chosen = candidates[topk_idx]  # [remain_slots, 2]
                        new_inf = torch.zeros_like(raw_new_inf)
                        new_inf[chosen[:, 0], chosen[:, 1]] = True
            else:
                new_inf = raw_new_inf

            self.infected_map[new_inf] = 1.0
            self.infection_strength[new_inf] = 1.0

            # ────────── 3) burst 爆发传播 ──────────
            for t, p in self.attack_types.items():
                if not p.get('burst'):
                    continue
                area = p['burst_area']
                bmask = torch.zeros_like(self.infected_map, dtype=torch.bool)
                for (x, y), info in self.attacks.items():
                    if info['type'] == t:
                        bmask[y, x] = True
                if not bmask.any():
                    continue

                # 膨胀 (2*area+1)^2 区域
                # k = torch.ones(1, 1, 2 * area + 1, 2 * area + 1, device=self.device)
                # burst_area = (F.conv2d(bmask.float()[None, None], k, padding=area) > 0).squeeze()
                # 从 self._max_burst_kernel 中切片得到 (2*area+1)×(2*area+1)
                k = self._max_burst_kernel[:, :, 4 - area:5 + area, 4 - area:5 + area]
                burst_area = (F.conv2d(bmask.float()[None, None], k, padding=area) > 0).squeeze()

                m1 = burst_area & (self.is_quarantined == 0) & (self.infected_map == 0)
                m2 = torch.rand_like(self.infected_map) < (p['burst_chance'] * (1 + difficulty))
                raw_burst = m1 & m2

                if self.step_count < 2000:
                    curr_inf2 = int((self.infected_map > 0.04).sum().item())
                    remain2 = 20 - curr_inf2
                    if remain2 <= 0:
                        burst_new = torch.zeros_like(raw_burst)
                    else:
                        burst_candidates = torch.nonzero(raw_burst, as_tuple=False)  # [Nb,2]
                        Nb = burst_candidates.size(0)
                        if Nb <= remain2:
                            burst_new = raw_burst.clone()
                        else:
                            scores2 = torch.rand(Nb, device=self.device)
                            _, topk2 = torch.topk(scores2, k=remain2, largest=False)
                            chosen2 = burst_candidates[topk2]  # [remain2, 2]
                            burst_new = torch.zeros_like(raw_burst)
                            burst_new[chosen2[:, 0], chosen2[:, 1]] = True
                else:
                    burst_new = raw_burst

                self.infected_map[burst_new] = 1.0
                self.infection_strength[burst_new] = 1.0

        self.attacks.update(new_attacks)

        self.attack_history += self.infected_map
        # —— 模拟网络流量 & 权限变化 —— #
        if self._rand_update_counter % imp_count == 0:
            self.net_traffic = torch.rand_like(self.net_traffic) * 0.05 + self.infected_map * 0.1
            self.perm_level = torch.clamp(self.perm_level + (self.infected_map * 0.01), 0.0, 1.0)

        self.attack_history.clamp_(0, 10)

        curr = int((self.infected_map > 0.04).sum().item())

        # 代替固定间隔触发入侵
        spawn_chance = 1 / self.spawn_interval  # e.g. 每步有 1/200 概率
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
                    self.privilege_level[dst[1], dst[0]] = p
                    for x, y in zip(dst[0].tolist(), dst[1].tolist()):
                        self.hacks[(x, y)] = {'type': 'lateral_move', 'power': p}

        # —— 更新黑客 one-hot —— #
        new_keys = set(self.hacks.keys())
        if new_keys != self._last_hack_keys:
            # 只有真正改动时才重置 one-hot，这样生成次数少、性能好
            self._hack_onehot.zero_()
            size = self.size
            # 假设 hack_keys = [(x1,y1), (x2,y2), ...]
            if new_keys:
                xs, ys = zip(*new_keys)  # Python list of coords
                # 转变成 flat idx
                flat = torch.tensor([y * size + x for (x, y) in new_keys],
                                    dtype=torch.long, device=self.device)
                self._hack_onehot[flat] = 1.0
            self._last_hack_keys = new_keys

        # —— 更新攻击类型扁平张量 _attack_type_flat —— #
        new_atk_keys = set(self.attacks.keys())
        if new_atk_keys != self._last_attack_keys:
            # 先将被移除的旧攻击恢为 -1
            removed = self._last_attack_keys - new_atk_keys
            for (x_old, y_old) in removed:
                flat_old = y_old * self.size + x_old
                self._attack_type_flat[flat_old] = -1
            # 再将新增或类型发生变化的点写入
            added = new_atk_keys - self._last_attack_keys
            for (x2, y2) in added:
                flat2 = y2 * self.size + x2
                t2 = self.attacks[(x2, y2)]['type']
                idx2 = list(self.attack_types).index(t2)
                self._attack_type_flat[flat2] = idx2
            self._last_attack_keys = new_atk_keys

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
        # —— 只有在真正需要“状态”时，才补齐 syscall_history —— #
        # 如果队列还没满，则一次性往里 append 随机 noise，直到长度 = maxlen
        if len(self.syscall_history) < self.syscall_history.maxlen:
            needed = self.syscall_history.maxlen - len(self.syscall_history)
            for _ in range(needed):
                self.syscall_history.append(self._extract_syscall_vector())
        else:
            # 如果已经满，要“滚动”一次：pop 左，再 append 一个新向量
            self.syscall_history.popleft()
            self.syscall_history.append(self._extract_syscall_vector())

        # 将 syscalls 队列做平均，得到 [H, W]
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

        # ========= 下面是“补齐”需要额外复制的那些缓存字段 =========

        # 5) 把 hack_onehot 也 clone 一份
        new._hack_onehot = self._hack_onehot.clone()
        # last_hack_keys：拷贝一份 set
        new._last_hack_keys = set(self._last_hack_keys)

        # 6) 把攻击类型扁平张量也 clone
        new._attack_type_flat = self._attack_type_flat.clone()
        new._last_attack_keys = set(self._last_attack_keys)

        # 7) 把 rand_update_counter 也拷过去
        new._rand_update_counter = self._rand_update_counter

        return new



# 安全包装器：屏蔽旧接口
class SecureGridEnv(GridSecurityEnv):
    def consume_resource(self, *args, **kwargs): return 0.0
    def get_nearest_resource_to(self, *args, **kwargs): return None
    def get_nearest_danger_to(self, *args, **kwargs): return None
    def get_agent_pos(self): return (-1, -1)
    def has_resource(self, *args, **kwargs): return False
    def has_hazard(self, *args, **kwargs): return False
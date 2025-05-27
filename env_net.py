import torch
import random
from collections import deque


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
        self.hack_types = {
            'bruteforce':      {'spawn_prob': 0.02, 'max_fail': 5},
            'phishing':        {'spawn_prob': 0.01, 'stealth':0.8},
            'lateral_move':    {'spawn_prob': 0.005},
            'privilege_escalation': {'spawn_prob': 0.003}
        }
        self.hacks = {}  # 当前活跃的黑客事件: dict[(x,y)]→{type,…}
        if self.step_count >= 0:  # 比如只在后期才允许初始病毒出现
            self._spawn_attack(initial=True)

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
        new_size = min(25, new_size)  # 你可以把最大尺寸设为 25
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
        if self.step_count % 1000 == 0 and len(self.attacks) > 10:
            self._expand_environment()

        if self.step_count % 1000 == 0:
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

        for (x, y), info in list(self.attacks.items()):
            atype = info['type']
            params = self.attack_types[atype]
            spread_prob = params['spread_prob'] * (1 + difficulty * 0.5)
            if self.step_count < 1000:
                spread_prob *= 0.5
            elif self.step_count < 2000:
                spread_prob *= 0.8


            # 传播到邻居格子
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.is_quarantined[ny, nx] == 0 and self.infected_map[ny, nx] == 0:
                        if random.random() < spread_prob:
                            # 修复：让新增的攻击点能继续传播
                            self.attacks[(nx, ny)] = {'type': atype, 'power': info['power']}

                            self.infected_map[ny, nx] = 1.0
                            self.infection_strength[ny, nx] = info['power']
                            self.behavior_score[ny, nx] += 0.1 + params['stealth'] * 0.1

            # 某些类型具备爆发传播能力（如 APT、scan）
            if params.get('burst') and random.random() < params['burst_chance'] * (1 + difficulty):
                area = params['burst_area']
                for _ in range(area * area):
                    bx = random.randint(max(0, x - area), min(self.size - 1, x + area))
                    by = random.randint(max(0, y - area), min(self.size - 1, y + area))
                    if self.is_quarantined[by, bx] == 0:
                        new_attacks[(bx, by)] = {'type': atype, 'power': info['power']}
                        self.infected_map[by, bx] = 1.0
                        self.infection_strength[by, bx] = info['power']
                        self.behavior_score[by, bx] += 0.1

        self.attacks.update(new_attacks)
        self.attack_history += self.infected_map
        # —— 模拟网络流量 & 权限变化 —— #
        self.net_traffic = torch.rand_like(self.net_traffic) * 0.05 + self.infected_map * 0.1
        self.perm_level = torch.clamp(self.perm_level + (self.infected_map * 0.01), 0.0, 1.0)

        self.attack_history.clamp_(0, 10)

        # 🔁 代替固定间隔触发入侵
        spawn_chance = 20.0 / self.spawn_interval  # e.g. 每步有 4/200 概率
        if random.random() < spawn_chance and self.step_count >= 0:
            self._spawn_attack()

        if self.step_count >= 2000:
            # —— 模拟一次黑客事件 —— #
            # 随机 spawn
            for ht, params in self.hack_types.items():
                if random.random() < params['spawn_prob']:
                    x, y = random.randrange(self.size), random.randrange(self.size)
                    self.hacks[(x,y)] = {'type': ht, 'power': params.get('stealth',1.0)}
                    # 马上标记一次
                    self.hack_history[y,x]  += 1
                    self.hack_strength[y,x]  = params.get('stealth',1.0)

            # 传播 or 行动
            new_hacks = {}
            for (x,y), info in list(self.hacks.items()):
                ht = info['type']
                p  = info['power']
                # 暴力破解：累积失败次数
                if ht=='bruteforce':
                    self.login_failures[y,x] += 1
                    if self.login_failures[y,x] > self.hack_types['bruteforce']['max_fail']:
                        # 一旦破解成功，提权为 root
                        self.privilege_level[y,x] = 1.0
                # 钓鱼/横向/提权等可以像病毒一样传播
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx,ny = x+dx, y+dy
                    if 0<=nx<self.size and 0<=ny<self.size:
                        # 根据脆弱度和 stealth 决定是否攻破
                        prob = self.vulnerability[ny,nx].item() * p
                        if random.random() < prob:
                            new_hacks[(nx,ny)] = {'type':ht,'power':p}
                            self.hack_history[ny,nx]  += 1
                            self.hack_strength[ny,nx]  = p
            self.hacks.update(new_hacks)

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
        你可以替换为实际的调用序列统计。
        """
        # 这里用随机噪声做示例
        return torch.randn((self.size, self.size), device=self.device) * 0.01


# ⚠️ 安全包装器：屏蔽旧接口
class SecureGridEnv(GridSecurityEnv):
    def consume_resource(self, *args, **kwargs): return 0.0
    def get_nearest_resource_to(self, *args, **kwargs): return None
    def get_nearest_danger_to(self, *args, **kwargs): return None
    def get_agent_pos(self): return (-1, -1)
    def has_resource(self, *args, **kwargs): return False
    def has_hazard(self, *args, **kwargs): return False
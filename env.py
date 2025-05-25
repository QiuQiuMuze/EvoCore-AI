# env.py
import numpy as np
import random
import logging
from collections import deque
from collections import Counter

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
logger.setLevel(logging.WARNING)
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


class GridEnvironment:
    action_space_n = 4  # 上/下/左/右
    def __init__(self, size=5, max_steps: int | None = None):
        self.size = size
        self.max_steps = max_steps
        self.resources = Counter()  # 同一格子可以有多份资源
        self.hazards = Counter()  # 同一格子可以有多份陷阱
        self.step_count = 0
        self.explored_cells_count = 0
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        exclude = {tuple(self.agent_pos)}
        self.refresh_environment(step=0, explored_cells_count=0, exclude_positions=exclude)
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)  # ✅ 初始化探索标记
        self.reset()
        self.reward_hit_count = 0
        self.danger_hit_count = 0

    def refresh_environment(self, step: int, explored_cells_count: int, exclude_positions: set = None):
        exclude_positions = exclude_positions or set()

        extra = min(100,max(((self.step_count - 1500) // 250), 0))

        # —— 资源点 —— #
        total_res = int(self.size * self.size / 5) + extra
        self.resources.clear()
        candidates = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in exclude_positions
        ]
        random.shuffle(candidates)
        for pos in candidates[: total_res]:
            self.resources[pos] = 1

        # —— 危险点 —— #
        total_haz = int(self.size * self.size / 3) + extra
        self.hazards.clear()
        random.shuffle(candidates)
        for pos in candidates[: total_haz]:
            self.hazards[pos] = 1

        # 刷新探索标记
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)
        self.explored_cells_count = 0

    def reset(self):
        # agent 初始化在随机位置
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]

        # --- 重置计数 & 能量 & 命中 & 访问记录 ---
        self.step_count = 0
        self.agent_energy_gain = 0.0
        self.agent_energy_penalty = 0.0
        self.prev_dist_resource = self.distance_to_nearest_resource(tuple(self.agent_pos))
        self.prev_danger_dist = self.distance_to_nearest_danger(tuple(self.agent_pos))

        # 清空访问轨迹
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)
        self.explored_cells_count = 0

        # 清空命中统计
        self.reward_hit_count = 0
        self.danger_hit_count = 0

        return self.get_state()

    def step(self, action, cog_step: int | None = None):
        """
        动作定义：
        0: 上 (y-1)
        1: 下 (y+1)
        2: 左 (x-1)
        3: 右 (x+1)
        """
        x, y = self.agent_pos
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.size - 1:
            x += 1
        self.agent_pos = [x, y]

        # 环境交互逻辑
        pos = (x, y)
        # —— 资源命中 —— #
        if self.resources[pos] > 0:
            # 只发信号，不删 data，让 CogGraph 来处理
            self.reward_hit_count += 1
            self.agent_energy_gain = 0.1
        else:
            self.agent_energy_gain = 0.0

        if self.hazards[pos] > 0:
            # 同上，仅发信号
            self.danger_hit_count += 1
            self.agent_energy_penalty = 0.1

        else:
            self.agent_energy_penalty = 0.0

        # 更新环境状态
        self.step_count += 1
        # 刷新周期基于 CogGraph 的轮数，否则退回用本地 step_count
        step_for_refresh = cog_step if cog_step is not None else self.step_count
        if step_for_refresh % 1000 == 0 and step_for_refresh >= 1000:
            self.refresh_environment(step_for_refresh, self.explored_cells_count)
            self.prev_dist_resource = self.distance_to_nearest_resource(tuple(self.agent_pos))
            self.prev_danger_dist = self.distance_to_nearest_danger(tuple(self.agent_pos))

        # --- 计算奖励 & 下一状态 ---
        # —— 基础奖励 ——
        base = self.agent_energy_gain - self.agent_energy_penalty

        # —— reward shaping ——
        pos = (x, y)
        # 1) 资源引导：走得更近 +0.001, 走远了 –0.001
        dist_res = self.distance_to_nearest_resource(pos)
        delta_res = self.prev_dist_resource - dist_res
        resource_shaping = 0.001 if delta_res > 0 else (-0.001 if delta_res < 0 else 0.0)
        self.prev_dist_resource = dist_res

        # 2) 危险引导（delta 版本）
        danger_dist = self.distance_to_nearest_danger(pos)
        delta_danger = self.prev_danger_dist - danger_dist
        # 如果比上一帧更远则 +0.001，离得更近则 –0.001 否则 0
        danger_shaping = 0.001 if delta_danger > 0 else (-0.001 if delta_danger < 0 else 0.0)
        # 更新 prev_danger_dist 供下次比较
        self.prev_danger_dist = danger_dist

        # 合成最终 reward
        # ✅ 探索奖励逻辑
        explore_bonus = 0.0
        if not self.visited_map[y, x]:
            self.visited_map[y, x] = True
            self.explored_cells_count += 1
            explore_bonus = 0.0001  # 或你想给的探索分数

        reward = base + resource_shaping + danger_shaping + explore_bonus

        next_state = self.get_state()
        # --- 终止条件 (done) ---
        done = False
        # ① 资源全部收集完
        if not self.resources:
            done = True
        # ② 超过 max_steps（若指定）
        elif self.max_steps is not None and self.step_count >= self.max_steps:
            done = True

        return next_state, reward, done, {}

    def get_nearest_resource_to(self, pos):
        """返回 pos 最近的资源位置"""
        if not self.resources:
            return None
        return min(self.resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def get_state(self):
        """
        返回包含 3 层信息的感知向量：
        - agent 位置（one-hot）
        - 资源分布（1 表示有资源）
        - 危险分布（1 表示有危险）
        - 是否访问过（1 表示已访问）
        最终返回 size * size * 3 的 1D 向量
        """
        agent_layer = np.zeros((self.size, self.size), dtype=np.float32)
        resource_layer = np.zeros((self.size, self.size), dtype=np.float32)
        hazard_layer = np.zeros((self.size, self.size), dtype=np.float32)
        visited_layer = self.visited_map.astype(np.float32)  # ✅ 新增

        x, y = self.agent_pos
        agent_layer[y, x] = 1.0

        for (rx, ry), cnt in self.resources.items():
            if cnt > 0:
                resource_layer[ry, rx] = 1.0

        for (hx, hy), cnt in self.hazards.items():
            if cnt > 0:
                hazard_layer[hy, hx] = 1.0

        # ✅ 现在堆叠四层
        stacked = np.stack([agent_layer, resource_layer, hazard_layer, visited_layer], axis=0)
        return stacked.flatten().astype(np.float32)

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        x, y = self.agent_pos
        grid[y, x] = 'A'

        logger.debug('\n' + '\n'.join(' '.join(row) for row in grid) + '\n')

    # ---------------------------------------------------------
    def distance_to_nearest_danger(self, pos):
        """曼哈顿距离最近危险格；若无危险返回 +∞"""
        if not self.hazards:
            return float("inf")
        return min(abs(pos[0] - hx) + abs(pos[1] - hy) for hx, hy in self.hazards)
    # ---------------------------------------------------------

    def distance_to_nearest_resource(self, pos):
        """曼哈顿距离最近资源格；若无资源返回 +∞"""
        if not self.resources:
            return float("inf")
        return min(abs(pos[0] - rx) + abs(pos[1] - ry)
                   for rx, ry in self.resources)

    def get_nearest_danger_to(self, pos):
        """返回 pos 最近的危险位置"""
        if not self.hazards:
            return None
        return min(self.hazards, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def resize(self, new_size: int):
        """
        平滑扩张环境到 new_size×new_size：
        - 保留 agent_pos, step_count, prev_dist_… 等状态
        - 更新 size，然后增量 flush 资源 & 危险
        """
        # 1) 更新 size
        old_size = self.size
        self.size = new_size

        # 2) 保留当前位置：如果超出边界，裁剪到合法范围
        x, y = self.agent_pos
        x = min(x, new_size - 1)
        y = min(y, new_size - 1)
        self.agent_pos = [x, y]

        # 3) 重新生成资源/危险，但不重置 step_count 等
        #    这里直接调用 refresh_environment，让它基于当前 step_count 重绘
        self.refresh_environment(step=self.step_count, explored_cells_count=self.explored_cells_count)

    def reset_with_size(self, new_size: int):
        import numpy as np
        # —— 更新维度 ——#
        self.size = new_size
        # —— 随机初始化 agent_pos ——#
        self.agent_pos = [
            np.random.randint(0, self.size),
            np.random.randint(0, self.size)
        ]
        # —— 清空资源/危险 ——#
        self.resources = Counter()
        self.hazards   = Counter()
        # —— 清空统计 & 访问记录 ——#
        self.step_count = 0
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)
        self.explored_cells_count = 0
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self.refresh_environment(step=0, explored_cells_count=0, exclude_positions={tuple(self.agent_pos)})

    def inject_reward_map(self,
                          reward_points: list[tuple[int,int]],
                          punishment_points: list[tuple[int,int]]):
        # "直接用外部生成的点来覆盖资源/危险集合。
        self.resources = Counter(reward_points)
        self.hazards   = Counter(punishment_points)
        # 清零统计与访问
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)
        self.explored_cells_count = 0



if __name__ == "__main__":
    env = GridEnvironment(size=10)
    env.render()
    for _ in range(5):
        action = np.random.choice(4)
        env.step(action)
        env.render()
        print("State vector:", env.get_state())

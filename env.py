import numpy as np
import random
import logging
import torch
from collections import deque, Counter

class LimitedDebugHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__(level=logging.DEBUG)
        self.buffer = deque(maxlen=capacity)

    def emit(self, record):
        if record.levelno == logging.DEBUG:
            try:
                msg = self.format(record)
                self.buffer.append(msg)
            except Exception:
                pass

    def dump_to_console(self):
        print("\n==== [最近 Debug 日志] ====")
        for msg in self.buffer:
            print(msg)

# === 设置 root logger ===
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.handlers.clear()

# Debug 缓存 Handler
debug_handler = LimitedDebugHandler(capacity=100)
debug_handler.setFormatter(logging.Formatter('%(asctime)s [DEBUG] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(debug_handler)

# 普通输出 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(console_handler)

class GridEnvironment:
    action_space_n = 4  # 上/下/左/右

    def __init__(self, size=5, max_steps: int | None = None):
        self.size = size
        self.max_steps = max_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 经验点和危险点
        self.resources = Counter()
        self.hazards = Counter()

        # 计数和位置
        self.step_count = 0
        self.explored_cells_count = 0
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]

        # 初始化探索标记和状态缓冲
        self.visited_map = torch.zeros((self.size, self.size), dtype=torch.bool, device=self.device)
        self._state_buf = torch.zeros((4, self.size, self.size), dtype=torch.float32, device=self.device)

        # 初始化环境
        exclude = {tuple(self.agent_pos)}
        self.refresh_environment(step=0, explored_cells_count=0, exclude_positions=exclude)
        self.reset()

        self.reward_hit_count = 0
        self.danger_hit_count = 0

    def refresh_environment(self, step: int, explored_cells_count: int, exclude_positions: set = None):
        exclude_positions = exclude_positions or set()
        extra = min(100, max(((self.step_count - 1500) // 250), 0))

        # 资源点
        total_res = int(self.size * self.size / 5) + extra
        self.resources.clear()
        candidates = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in exclude_positions]
        random.shuffle(candidates)
        for pos in candidates[:total_res]:
            self.resources[pos] = 1

        # 危险点
        total_haz = int(self.size * self.size / 3) + extra
        self.hazards.clear()
        random.shuffle(candidates)
        for pos in candidates[:total_haz]:
            self.hazards[pos] = 1

        # 清空探索标记
        self.visited_map.fill_(False)
        self.explored_cells_count = 0

    def reset(self):
        # 随机初始化 agent
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        self.step_count = 0
        self.agent_energy_gain = 0.0
        self.agent_energy_penalty = 0.0

        # 最近距离初始化
        self.prev_dist_resource = self.distance_to_nearest_resource(tuple(self.agent_pos))
        self.prev_danger_dist = self.distance_to_nearest_danger(tuple(self.agent_pos))

        # 清空标记
        self.visited_map.fill_(False)
        self.explored_cells_count = 0

        # 清空命中统计
        self.reward_hit_count = 0
        self.danger_hit_count = 0

        return self.get_state()

    def step(self, action, cog_step: int | None = None):
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

        pos = (x, y)
        # 资源命中
        if self.resources[pos] > 0:
            self.reward_hit_count += 1
            self.agent_energy_gain = 0.1
        else:
            self.agent_energy_gain = 0.0

        # 危险命中
        if self.hazards[pos] > 0:
            self.danger_hit_count += 1
            self.agent_energy_penalty = 0.1
        else:
            self.agent_energy_penalty = 0.0

        # 更新步数
        self.step_count += 1
        step_for_refresh = cog_step if cog_step is not None else self.step_count
        if step_for_refresh % 1000 == 0 and step_for_refresh >= 1000:
            self.refresh_environment(step_for_refresh, self.explored_cells_count)
            self.prev_dist_resource = self.distance_to_nearest_resource(tuple(self.agent_pos))
            self.prev_danger_dist = self.distance_to_nearest_danger(tuple(self.agent_pos))

        # 计算 reward shaping
        base = self.agent_energy_gain - self.agent_energy_penalty
        dist_res = self.distance_to_nearest_resource(pos)
        delta_res = self.prev_dist_resource - dist_res
        resource_shaping = 0.001 if delta_res > 0 else (-0.001 if delta_res < 0 else 0.0)
        self.prev_dist_resource = dist_res

        danger_dist = self.distance_to_nearest_danger(pos)
        delta_danger = self.prev_danger_dist - danger_dist
        danger_shaping = 0.001 if delta_danger > 0 else (-0.001 if delta_danger < 0 else 0.0)
        self.prev_danger_dist = danger_dist

        explore_bonus = 0.0
        if not self.visited_map[y, x]:
            self.visited_map[y, x] = True
            self.explored_cells_count += 1
            explore_bonus = 0.0001

        reward = base + resource_shaping + danger_shaping + explore_bonus
        next_state = self.get_state()

        done = False
        if not self.resources:
            done = True
        elif self.max_steps is not None and self.step_count >= self.max_steps:
            done = True

        return next_state, reward, done, {}

    def get_nearest_resource_to(self, pos):
        if not self.resources:
            return None
        return min(self.resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def get_state(self):
        # 就地更新状态缓冲

        buf = self._state_buf
        buf.fill_(0.0)

        x, y = self.agent_pos
        buf[0, y, x] = 1.0
        for (rx, ry), cnt in self.resources.items():
            # ← 边界检查，确保 0 <= rx,ry < size
            if cnt > 0 and 0 <= rx < self.size and 0 <= ry < self.size:
                buf[1, ry, rx] = 1.0

        for (hx, hy), cnt in self.hazards.items():
            if cnt > 0 and 0 <= hx < self.size and 0 <= hy < self.size:
                buf[2, hy, hx] = 1.0
        buf[3].copy_(self.visited_map.to(torch.float32))

        return buf.view(-1)

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        x, y = self.agent_pos
        grid[y, x] = 'A'
        logger.debug('\n' + '\n'.join(' '.join(row) for row in grid) + '\n')

    def distance_to_nearest_danger(self, pos):
        if not self.hazards:
            return float("inf")
        return min(abs(pos[0] - hx) + abs(pos[1] - hy) for hx, hy in self.hazards)

    def distance_to_nearest_resource(self, pos):
        if not self.resources:
            return float("inf")
        return min(abs(pos[0] - rx) + abs(pos[1] - ry) for rx, ry in self.resources)

    def get_nearest_danger_to(self, pos):
        if not self.hazards:
            return None
        return min(self.hazards, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def resize(self, new_size: int):
        self.size = new_size
        # 2. 重新分配 visited_map 和 状态缓冲，保证和新 size 匹配
        self.visited_map = torch.zeros((new_size, new_size),
                                       dtype=torch.bool,
                                       device=self.device)
        self._state_buf = torch.zeros((4, new_size, new_size),
                                      dtype=torch.float32,
                                      device=self.device)

        x, y = self.agent_pos
        x = min(x, new_size - 1)
        y = min(y, new_size - 1)
        self.agent_pos = [x, y]
        self.refresh_environment(step=self.step_count,explored_cells_count = self.explored_cells_count)

    def reset_with_size(self, new_size: int):
        self.size = new_size
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        self.visited_map = torch.zeros((self.size, self.size), dtype=torch.bool, device=self.device)
        self._state_buf = torch.zeros((4, self.size, self.size), dtype=torch.float32, device=self.device)
        self.resources = Counter()
        self.hazards = Counter()
        self.step_count = 0
        self.explored_cells_count = 0
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self.refresh_environment(step=0, explored_cells_count=0, exclude_positions={tuple(self.agent_pos)})

    def inject_reward_map(self, reward_points: list[tuple[int,int]], punishment_points: list[tuple[int,int]]):
        self.resources = Counter(reward_points)
        self.hazards = Counter(punishment_points)
        self.visited_map = torch.zeros((self.size, self.size), dtype=torch.bool, device=self.device)
        self.explored_cells_count = 0
        self.reward_hit_count = 0
        self.danger_hit_count = 0

if __name__ == "__main__":
    env = GridEnvironment(size=10)
    env.render()
    for _ in range(5):
        action = np.random.choice(4)
        env.step(action)
        env.render()
        print("State vector:", env.get_state())

import numpy as np
import random

class GridEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.resources = set()
        self.hazards = set()
        self.step_count = 0
        self.refresh_environment()
        self.reset()

    def refresh_environment(self):
        """刷新资源与危险格子的位置（每隔一段时间）"""
        self.resources.clear()
        self.hazards.clear()
        for _ in range(self.size // 2):  # 生成资源
            self.resources.add((random.randint(0, self.size - 1), random.randint(0, self.size - 1)))
        for _ in range(self.size // 3):  # 生成危险区域
            self.hazards.add((random.randint(0, self.size - 1), random.randint(0, self.size - 1)))

    def reset(self):
        # agent 初始化在随机位置
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]

    def step(self, action):
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
        if pos in self.resources:
            self.resources.remove(pos)
            self.agent_energy_gain = 0.2  # 单步奖励
        else:
            self.agent_energy_gain = 0.0

        if pos in self.hazards:
            self.agent_energy_penalty = 0.1
        else:
            self.agent_energy_penalty = 0.0

        # 更新环境状态
        self.step_count += 1
        if self.step_count % 100 == 0:
            self.refresh_environment()

    def get_state(self):
        """
        返回包含 3 层信息的感知向量：
        - agent 位置（one-hot）
        - 资源分布（1 表示有资源）
        - 危险分布（1 表示有危险）
        最终返回 size * size * 3 的 1D 向量
        """
        agent_layer = np.zeros((self.size, self.size), dtype=np.float32)
        resource_layer = np.zeros((self.size, self.size), dtype=np.float32)
        hazard_layer = np.zeros((self.size, self.size), dtype=np.float32)

        x, y = self.agent_pos
        agent_layer[y, x] = 1.0

        for rx, ry in self.resources:
            resource_layer[ry, rx] = 1.0
        for hx, hy in self.hazards:
            hazard_layer[hy, hx] = 1.0

        # 堆叠三层，并展开为 1D 向量
        stacked = np.stack([agent_layer, resource_layer, hazard_layer], axis=0)
        return stacked.flatten()

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        x, y = self.agent_pos
        grid[y, x] = 'A'

        print('\n'.join(' '.join(row) for row in grid))
        print()

if __name__ == "__main__":
    env = GridEnvironment(size=10)
    env.render()
    for _ in range(5):
        action = np.random.choice(4)
        env.step(action)
        env.render()
        print("State vector:", env.get_state())

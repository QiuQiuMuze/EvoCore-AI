import numpy as np

class GridEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

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

    def get_state(self):
        """
        返回 1D 感知向量（size*size 维度，当前位置为 1，其它为 0）
        可作为 sensor 输入
        """
        state = np.zeros((self.size, self.size), dtype=np.float32)
        x, y = self.agent_pos
        state[y, x] = 1.0
        return state.flatten()

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

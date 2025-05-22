import os
import math
from self_feedback_module import SelfFeedback

class InputProcessor:
    def __init__(self, input_dir, strategy_pool, env_class, env_kwargs=None, device="cpu"):
        # 输入文件夹和策略池
        self.input_dir = input_dir
        self.strategy_pool = strategy_pool
        # 自省打分模块
        self.self_feedback = SelfFeedback(device=device)
        # 环境类和构造参数
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        # 预列出所有 .txt 文件，按文件名排序
        self.files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith(".txt")]
        )
        self.counter = 0

    def step(self):
        # 全部文件处理完后返回 None
        if self.counter >= len(self.files):
            return None

        # 读取当前文本
        filename = self.files[self.counter]
        filepath = os.path.join(self.input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # —— 1) 动态决定地图大小 ——#
        text_len = len(content)
        # 地图边长 = 文本长度开根号，最小 10，最大 100
        grid_size = max(10, min(100, int(math.ceil(text_len ** 0.5))))

        # —— 2) 重置或新建环境 ——#
        try:
            self.env.reset_with_size(grid_size)
        except AttributeError:
            # 第一次调用时没有 env，直接创建新环境
            self.env = self.env_class(size=grid_size, **self.env_kwargs)

        # —— 3) 自省模块打分 ——#
        result = self.self_feedback.process(content, self.strategy_pool)

        # —— 4) 根据 score 规划奖励点和惩罚点数量 ——#
        score = result.get("reward", 0.0)
        total_points = int(grid_size ** 2 * 0.1)  # 10% 的格子用于 reward+punish
        score = 1
        num_rewards = int(score * total_points)
        num_punishments = total_points - num_rewards

        # —— 新增：保存该文本的总 resource 点数 ——#
        self.num_rewards = num_rewards

        # —— 5) 生成确定性的位置 ——#
        reward_points = [(i, i) for i in range(num_rewards)]
        punishment_points = [(i, grid_size - 1 - i) for i in range(num_punishments)]

        # —— 6) 注入地图 ——#
        self.env.inject_reward_map(reward_points, punishment_points)

        # —— 7) 动态决定学习步数 ——#
        # 示例：每个字符 1 步
        learning_steps = text_len

        self.counter += 1
        return learning_steps

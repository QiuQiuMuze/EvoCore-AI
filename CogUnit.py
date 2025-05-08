# cogunit.py
import torch
import uuid
import random

class CogUnit:
    """
    CogUnit 是 EvoCore 的最小认知单元：
    - 拥有独立状态、能量、年龄
    - 可进行状态更新（update）与输出
    - 可判断是否分裂（should_split）与死亡（should_die）
    - 可克隆生成新单元（clone）
    """

    def __init__(self, input_size=50, hidden_size=16, role="processor"):
        self.position = (random.randint(0, 10), random.randint(0, 10))  # 可调范围
        self.state_memory = []  # 记忆队列
        self.memory_limit = 5  # 可调整为 k 步
        self.role = role
        self.id = uuid.uuid4()          # 唯一标识
        self.energy = 1.0               # 初始能量
        self.age = 0                    # 生存步数
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 认知状态向量
        self.state = torch.zeros(hidden_size)

        # 微型前馈网络（输入维度 → 隐藏维度 → 回到输入维度）
        self.function = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size)
        )

        self.last_output = torch.zeros(input_size)

    def get_position(self):
        return self.position

    def update(self, input_tensor: torch.Tensor):
        """使用输入信息更新状态，同时消耗能量。"""
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # 统一为 [1, input_size]

        self.last_output = self.function(input_tensor)  # 输出同样是 [1, input_size]
        self.state = self.last_output.clone()

        # 能量代谢与奖赏
        self.energy -= 0.01 + random.uniform(0, 0.02)
        if random.random() < 0.5:
            gain = random.uniform(0.05, 0.1)
            self.energy += gain
            print(f"[奖励] {self.id} 获得能量 +{gain:.2f}")

        self.age += 1
        self.state_memory.append(self.state.clone())
        if len(self.state_memory) > self.memory_limit:
            self.state_memory.pop(0)

    def get_output(self) -> torch.Tensor:
        """返回给下游单元使用的输出 (shape=[1, input_size])"""
        return self.last_output

    def should_split(self):
        if self.energy > 1.5 and self.age >= 10:
            # 增加被奖励次数统计后，可以用次数阈值判断
            return True
        return False

    def should_die(self) -> bool:
        return self.energy <= 0.0 or self.age > 100

    def clone(self, role_override=None):
        role = role_override or self.role  # 支持指定角色
        clone_unit = CogUnit(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            role=role
        )
        clone_unit.energy = self.energy / 2
        clone_unit.age = 0  # ✅ 分裂重置年龄
        clone_unit.state = self.state.clone()
        clone_unit.last_output = self.last_output.clone()
        self.energy /= 2
        print(f"[分裂] {self.id} → {clone_unit.id} (能量平均分配, 角色: {role})")
        return clone_unit

    def get_role(self):
        return self.role

    def __str__(self):
        x, y = self.position
        return f"CogUnit<{self.id}> Role:{self.role} Pos:({x},{y}) Age:{self.age} Energy:{self.energy:.2f}"

# TODO: 支持 get_role(), route_info(), register_trigger() 等接口

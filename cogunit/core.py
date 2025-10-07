# cogunit.py
import torch
import uuid
import random
from env import logger
from collections import deque
import torch.nn as nn
from collections import defaultdict
from config_runtime import RF            # ★ 新增
from contextlib import nullcontext       # ★ autocast fallback
from meta_cognition import MetaCognition
from memory_unit import MemoryBuffer

from . import settings as unit_settings
from .settings import (
    ENABLE_MINI_LEARN,
    FOLLOW_INPUT_DEVICE,
    MAX_OUTPUT_DIM,
    SPLIT_HI_ES_TABLE,
    SPLIT_HI_P_TABLE,
    TOL_FRAC_SPLIT,
    ROLE_SPLIT_RULE,
    get_hi_threshold,
)


from .device import DeviceMixin
from .learning import LearningMixin
from .lifecycle import LifecycleMixin
from .memory import MemoryMixin

class CogUnit(DeviceMixin, LearningMixin, LifecycleMixin, MemoryMixin):
    MAX_OUTPUT_DIM = MAX_OUTPUT_DIM

    @classmethod
    def configure_max_output_dim(cls, value):
        """Synchronize the max output dimension across all CogUnit components."""
        unit_settings.MAX_OUTPUT_DIM = value
        cls.MAX_OUTPUT_DIM = value

    """
    CogUnit 是 EvoCore 的最小认知单元：
    - 拥有独立状态、能量、年龄
    - 可进行状态更新（update）与输出
    - 可判断是否分裂（should_split）与死亡（should_die）
    - 可克隆生成新单元（clone）
    """

    def __init__(self, input_size=50, hidden_size=16, role="processor",env_size=5, id=None, **kwargs):
        self.is_elite = False
        self.local_memory_pool = []  # 每个单元的私有记忆池

        # 基因表达，表示对不同功能的偏好
        self.gene = {
            "sensor_bias": random.uniform(0.5, 1.5),
            "processor_bias": random.uniform(0.5, 1.5),
            "emitter_bias": random.uniform(0.5, 1.5),
            "mutation_rate": 0.01 # 每次复制有1%概率突变
        }

        self.death_by_aging = False
        self.subsystem_id = None  # 初始没有子系统归属
        self.output_history = []  # ✅ 用于记录近几次输出，评估是否行为单一
        self.call_history = []  # 记录最近几步的调用次数
        self.call_window = 5  # 窗口长度，过去 5 步
        self.inactive_steps = 0
        # 位置总在 [0, env_size) 范围内随机
        self.env_size = env_size
        self.position = (
            random.randint(0, env_size - 1),
            random.randint(0, env_size - 1),
        )
        self.output_history_tensor = torch.zeros((5, input_size), device="cpu")
        self.output_history_ptr = 0
        # self._rebuild_safe_positions()
        self.state_memory = []  # 记忆队列
        self.memory_limit = 5
        self.memory_pool_limit = 50
        self.role = role
        self.cleared_positions = set()
        self.uuid = uuid.uuid4()
        self.id = id or str(uuid.uuid4())
        self.int_id = self.uuid.int & 0xFFFFFFFF
        self.energy = 1.0               # 初始能量
        self.age = 0                    # 生存步数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.avg_recent_calls = 0.0
        # 认知状态向量
        self.state = torch.zeros(hidden_size)
        self.output_positions = deque(maxlen=10)
        self.is_hazard_confirmed = False
        # 元认知记录器
        self.meta = MetaCognition(history_len=100)
        self.personal_goal = None            # 当前内在目标 (x,y)
        self.visit_counts = {}               # {(x,y): 次数}
        self.intrinsic_reward = 0.1        # 达到内在目标奖励能量
        # 微型前馈网络（输入维度 → 隐藏维度 → 回到输入维度）
        self.intrinsic_cooldown = 0  # 冷却步数
        self._last_intrinsic_step = -float("inf")

        self.function = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size)
        )

        self.last_output = torch.zeros(input_size)
        # ---------- 加速格式 ---------- #
        if RF.use_channels_last and torch.cuda.is_available():
            self.function = self.function.to(memory_format=torch.channels_last)
        if RF.use_fp16 and torch.cuda.is_available():
            self.function = self.function.half()   # 权重转 FP16

        if "mutation_rate" not in self.gene:
            self.gene["mutation_rate"] = 0.05
        self.device = torch.device("cpu")  # 默认跟随 CPU
        self.last_action_rewarded = False
        self.last_reward_step = 0  # 记录上次获得 env 奖励的 step
        # ----- 资源点打转管控 -----
        self.last_rewarded_target_idx = None   # 上一次领奖的资源索引
        self.linger_steps             = 0      # 在同一目标附近逗留的帧数
        self.latest_base_reward       = 0.0    # 上一次靠近时发的 base 奖励，用来扣回
        self.is_permanent_explorer = False
        self.visit_counts = defaultdict(int)
        self.memory_buffer = MemoryBuffer(maxlen=200)



    # ---------------- 新增 ----------------
    # -------------------------------------
























import random
import uuid
from collections import defaultdict, deque

import torch
import torch.nn as nn

from config_runtime import RF
from meta_cognition import MetaCognition
from memory_unit import MemoryBuffer


class CogUnitBase:
    """核心属性与通用操作"""

    def __init__(self, input_size=50, hidden_size=16, role="processor", env_size=5, id=None, **kwargs):
        self.is_elite = False
        self.local_memory_pool = []
        self.gene = {
            "sensor_bias": random.uniform(0.5, 1.5),
            "processor_bias": random.uniform(0.5, 1.5),
            "emitter_bias": random.uniform(0.5, 1.5),
            "mutation_rate": 0.01,
        }
        self.death_by_aging = False
        self.subsystem_id = None
        self.output_history = []
        self.call_history = []
        self.call_window = 5
        self.inactive_steps = 0
        self.env_size = env_size
        self.position = (
            random.randint(0, env_size - 1),
            random.randint(0, env_size - 1),
        )
        self.output_history_tensor = torch.zeros((5, input_size), device="cpu")
        self.output_history_ptr = 0
        self.state_memory = []
        self.memory_limit = 5
        self.memory_pool_limit = 50
        self.role = role
        self.cleared_positions = set()
        self.uuid = uuid.uuid4()
        self.id = id or str(uuid.uuid4())
        self.int_id = self.uuid.int & 0xFFFFFFFF
        self.energy = 1.0
        self.age = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.avg_recent_calls = 0.0
        self.state = torch.zeros(hidden_size)
        self.output_positions = deque(maxlen=10)
        self.is_hazard_confirmed = False
        self.meta = MetaCognition(history_len=100)
        self.personal_goal = None
        self.visit_counts = {}
        self.intrinsic_reward = 0.1
        self.intrinsic_cooldown = 0
        self._last_intrinsic_step = -float("inf")
        self.function = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.last_output = torch.zeros(input_size)
        if RF.use_channels_last and torch.cuda.is_available():
            self.function = self.function.to(memory_format=torch.channels_last)
        if RF.use_fp16 and torch.cuda.is_available():
            self.function = self.function.half()
        if "mutation_rate" not in self.gene:
            self.gene["mutation_rate"] = 0.05
        self.device = torch.device("cpu")
        self.last_action_rewarded = False
        self.last_reward_step = 0
        self.last_rewarded_target_idx = None
        self.linger_steps = 0
        self.latest_base_reward = 0.0
        self.is_permanent_explorer = False
        self.visit_counts = defaultdict(int)
        self.memory_buffer = MemoryBuffer(maxlen=200)

    def to(self, device):
        device = torch.device(device)
        if device == getattr(self, "device", torch.device("cpu")):
            return self
        self.device = device
        self.function.to(device)
        self.state = self.state.to(device)
        self.last_output = self.last_output.to(device)
        return self

    def get_position(self):
        return self.position

    def get_recent_outputs(self, n=5):
        L = min(n, self.output_history_tensor.size(0))
        idxs = [(self.output_history_ptr - i - 1) % L for i in reversed(range(L))]
        return [self.output_history_tensor[i] for i in idxs]

    def get_role(self):
        return self.role

    def __str__(self):
        return f"CogUnit(id={self.id}, role={self.role}, energy={self.energy:.2f}, age={self.age})"

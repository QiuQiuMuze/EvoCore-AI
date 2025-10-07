"""Core :class:`CogUnit` implementation built from modular helpers."""
from __future__ import annotations

import random
import uuid
from collections import defaultdict, deque

import torch
import torch.nn as nn

from config_runtime import RF
from meta_cognition import MetaCognition
from memory_unit import MemoryBuffer

from cogunit_components.constants import MAX_OUTPUT_DIM
from cogunit_components.experience import record_memory as record_experience, recall as recall_experience
from cogunit_components.learning import (
    compute_self_reward as learning_compute_self_reward,
    perform_mini_learn as learning_mini_learn,
    perform_update,
)
from cogunit_components.lifecycle import (
    clone_unit as lifecycle_clone,
    evaluate_self as lifecycle_evaluate_self,
    request_upgrade as lifecycle_request_upgrade,
    should_die as lifecycle_should_die,
    should_split as lifecycle_should_split,
)
from cogunit_components.memory import (
    add_to_local_memory as memory_add_to_local,
    get_recent_outputs as memory_recent_outputs,
    is_worthy_of_memory as memory_is_worthy,
)


class CogUnit:
    """最小认知单元，封装为独立模块化实现。"""

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

        self.function = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size),
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

    def mini_learn(self, input_tensor, target_tensor, lr=0.001):
        learning_mini_learn(self, input_tensor, target_tensor, lr)

    def compute_self_reward(self, input_tensor, output_tensor):
        return learning_compute_self_reward(self, input_tensor, output_tensor)

    def get_recent_outputs(self, n=5):
        return memory_recent_outputs(self, n)

    def update(self, input_tensor: torch.Tensor):
        perform_update(self, input_tensor)

    def get_output(self) -> torch.Tensor:
        if MAX_OUTPUT_DIM is not None and self.last_output.numel() > MAX_OUTPUT_DIM:
            return self.last_output[:MAX_OUTPUT_DIM]
        return self.last_output

    def should_split(self):
        return lifecycle_should_split(self)

    def evaluate_self(self, min_rate=0.3):
        return lifecycle_evaluate_self(self, min_rate)

    def request_upgrade(self, target_role=None, reason=""):
        lifecycle_request_upgrade(self, target_role, reason)

    def is_worthy_of_memory(self):
        return memory_is_worthy(self)

    def add_to_local_memory(self):
        memory_add_to_local(self)

    def should_die(self) -> bool:
        return lifecycle_should_die(self)

    def clone(
        self,
        role_override=None,
        new_input_size=None,
        global_resources=None,
        global_hazards=None,
        free_positions=None,
    ):
        return lifecycle_clone(
            self,
            role_override=role_override,
            new_input_size=new_input_size,
            global_resources=global_resources,
            global_hazards=global_hazards,
            free_positions=free_positions,
        )

    def record_memory(self, state: torch.Tensor, action, reward: float, outcome: str):
        record_experience(self, state, action, reward, outcome)

    def recall(self, query_state: torch.Tensor, k: int = 5, metric: str = "cosine"):
        return recall_experience(self, query_state, k, metric)

    def get_role(self):
        return self.role

    def __str__(self):
        x, y = self.position
        return f"CogUnit<{self.id}> Role:{self.role} Pos:({x},{y}) Age:{self.age} Energy:{self.energy:.2f} Gene:{self.gene}"

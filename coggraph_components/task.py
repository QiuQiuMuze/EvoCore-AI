"""Goal injection helpers for the cognitive graph."""
from __future__ import annotations

import torch


class TaskInjector:
    def __init__(self, target_position):
        self.target_position = target_position

    def encode_goal(self, env_size):
        index = self.target_position[1] * env_size + self.target_position[0]
        vec = torch.zeros(2, env_size * env_size)
        vec[0, index] = 1.0
        return vec

    def evaluate(self, env, emitter_outputs):
        if emitter_outputs is None:
            return False
        pred_index = torch.argmax(emitter_outputs.mean(dim=0)).item()
        x, y = pred_index % env.size, pred_index // env.size
        return (x, y) in env.resources

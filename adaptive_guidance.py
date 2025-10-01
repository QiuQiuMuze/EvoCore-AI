"""Learning-based emitter coordination utilities."""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Optional, Tuple
import random

import torch


ASSIGNMENT_SELF = "self_direct"
ASSIGNMENT_LEARNED = "learned"


class AdaptiveGuidanceModule:
    """Contextual bandit that learns which cells to assign to emitters."""

    def __init__(
        self,
        grid_size: int,
        device: torch.device | str = "cpu",
        lr: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        history: int = 2048,
    ) -> None:
        self.device = torch.device(device)
        self.grid_size = grid_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_map = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=self.device)
        self.visit_counts = torch.zeros_like(self.value_map)
        self.history = deque(maxlen=history)
        self._rng = random.Random()

    def resize(self, new_size: int) -> None:
        if new_size == self.grid_size:
            return
        new_values = torch.zeros((new_size, new_size), dtype=self.value_map.dtype, device=self.device)
        new_counts = torch.zeros_like(new_values)
        min_size = min(new_size, self.grid_size)
        new_values[:min_size, :min_size] = self.value_map[:min_size, :min_size]
        new_counts[:min_size, :min_size] = self.visit_counts[:min_size, :min_size]
        self.value_map = new_values
        self.visit_counts = new_counts
        self.grid_size = new_size

    def select_goal(
        self,
        emitter_id: int,
        emitter_pos: Tuple[int, int],
        candidates: Iterable[int],
        threat_scores: Dict[int, float],
        step: int,
    ) -> Tuple[Optional[int], Dict[str, float]]:
        cand_list = list(candidates)
        if not cand_list:
            return None, {}

        if self._rng.random() < self.epsilon:
            choice = self._rng.choice(cand_list)
            return choice, {"strategy": "explore", "score": 0.0}

        ex, ey = emitter_pos
        best_flat = None
        best_score = None
        for flat in cand_list:
            x = flat % self.grid_size
            y = flat // self.grid_size
            distance = abs(x - ex) + abs(y - ey) + 1
            learned = float(self.value_map[y, x].item())
            prior = threat_scores.get(flat, 0.0)
            visits = float(self.visit_counts[y, x].item())
            confidence = 1.0 / (1.0 + visits)
            score = learned + confidence * prior - 0.05 * distance
            if best_score is None or score > best_score:
                best_score = score
                best_flat = flat

        return best_flat, {"strategy": "exploit", "score": best_score or 0.0}

    def register_feedback(
        self,
        position: Tuple[int, int],
        reward: float,
        latency: int,
    ) -> None:
        x, y = position
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return
        decay = self.gamma ** max(latency, 1)
        target = reward * decay
        old_value = self.value_map[y, x]
        self.value_map[y, x] = old_value + self.lr * (target - old_value)
        self.visit_counts[y, x] += 1

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "values": self.value_map.detach().clone(),
            "counts": self.visit_counts.detach().clone(),
        }


__all__ = [
    "AdaptiveGuidanceModule",
    "ASSIGNMENT_SELF",
    "ASSIGNMENT_LEARNED",
]

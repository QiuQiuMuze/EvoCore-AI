"""Experience buffer helpers for :class:`CogUnit`."""
from __future__ import annotations


def record_memory(unit, state, action, reward: float, outcome: str) -> None:
    unit.memory_buffer.add(state, action, reward, outcome)


def recall(unit, query_state, k: int = 5, metric: str = "cosine"):
    return unit.memory_buffer.recall(query_state, k, metric)

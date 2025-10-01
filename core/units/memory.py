import torch


class MemoryMixin:
    def record_memory(self, state: torch.Tensor, action, reward: float, outcome: str):
        self.memory_buffer.add(state, action, reward, outcome)

    def recall(self, query_state: torch.Tensor, k: int = 5, metric: str = "cosine"):
        return self.memory_buffer.recall(query_state, k, metric)

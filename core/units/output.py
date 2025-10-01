import torch

from .constants import MAX_OUTPUT_DIM


class OutputMixin:
    def get_output(self) -> torch.Tensor:
        if MAX_OUTPUT_DIM is not None and self.last_output.numel() > MAX_OUTPUT_DIM:
            return self.last_output[:MAX_OUTPUT_DIM]
        return self.last_output

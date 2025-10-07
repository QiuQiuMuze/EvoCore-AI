"""CogUnit mixin module generated from the legacy monolith."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CogUnit

import torch

from . import settings as unit_settings


class DeviceMixin:
    def to(self, device):
        """把内部权重 & 状态迁移到指定设备（cpu / cuda）"""
        device = torch.device(device)
        if device == getattr(self, "device", torch.device("cpu")):
            return self  # 已在目标 device，直接返回
        self.device = device
        self.function.to(device)
        self.state = self.state.to(device)
        self.last_output = self.last_output.to(device)
        # 若还有其他缓存张量，也一并 .to(device)
        return self

    def get_position(self):
        return self.position

    def get_output(self) -> torch.Tensor:
        """返回给下游单元使用的输出 (shape=[1, input_size])"""
        if unit_settings.MAX_OUTPUT_DIM is not None and self.last_output.numel() > unit_settings.MAX_OUTPUT_DIM:
            return self.last_output[:unit_settings.MAX_OUTPUT_DIM]
        return self.last_output

    def get_role(self):
        return self.role

    def __str__(self):
        x, y = self.position
        return f"CogUnit<{self.id}> Role:{self.role} Pos:({x},{y}) Age:{self.age} Energy:{self.energy:.2f} Gene:{self.gene}"

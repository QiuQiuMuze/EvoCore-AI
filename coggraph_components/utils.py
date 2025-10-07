"""Utility helpers for coggraph operations."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def align_vec(a: torch.Tensor, b: torch.Tensor, mode: str = "pad"):
    la, lb = a.shape[-1], b.shape[-1]
    if la == lb:
        return a, b, la
    if mode == "pad":
        L = max(la, lb)
        if la < L:
            a = F.pad(a, (0, L - la))
        if lb < L:
            b = F.pad(b, (0, L - lb))
        return a, b, L
    else:
        L = min(la, lb)
        return a[..., :L], b[..., :L], L

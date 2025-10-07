"""Factory helpers for shared transformer blocks."""
from __future__ import annotations

import torch

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TE = False


def build_transformer_block(D, H, device):
    if HAS_TE and torch.cuda.is_available():
        return te.TransformerLayer(
            hidden_size=D,
            num_attention_heads=H,
            mlp_hidden_size=4 * D,
            dropout=0.0,
            sequence_parallel=False,
        ).to(device)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=D,
        nhead=H,
        dim_feedforward=4 * D,
        activation="gelu",
        batch_first=True,
    )
    return layer.to(device)

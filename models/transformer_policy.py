import torch
import torch.nn as nn
from typing import Optional

# 尝试导入 Flash-Attn 的高效 Transformer 实现
try:
    from flash_attn.modules.transformer import TransformerLayer as FlashTransformerLayer
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# 兼容性的正余弦位置编码
from utils import sinusoidal_positional_encoding

class TransformerPolicyNetwork(nn.Module):
    """
    轻量级策略网络：支持 Flash-Attn + RoPE 或 原生 TransformerEncoder。
    输入 (batch, seq_len, input_dim) → logits (batch, num_actions)
    """
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_seq_len: int = 16,
        use_action_noise: bool = True,
        use_flash_attn: bool = True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.noise_std = 0.2
        self.use_action_noise = use_action_noise

        # 线性投影到 d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 选择 Flash-Attn 或 原生 PyTorch Transformer
        self.use_flash = use_flash_attn and FLASH_ATTN_AVAILABLE
        if self.use_flash:
            # Flash-Attn 的 TransformerLayer 支持 RoPE
            self.transformer_encoder = nn.Sequential(*[
                FlashTransformerLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.1,
                    layer_norm_eps=1e-5,
                    causal=False,
                    use_flash_attn=True,
                    rotary_emb=True
                ) for _ in range(num_layers)
            ])
        else:
            # 原生 TransformerEncoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

        # 输出头
        self.fc_out = nn.Linear(d_model, num_actions)
        self._reset_parameters()

    def _reset_parameters(self):
        # 官方推荐的 Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: optional padding mask of shape (batch, seq_len)
        Returns:
            logits: (batch, num_actions)
        """
        # 投影
        h = self.input_proj(x)  # (B, L, d_model)

        if not self.use_flash:
            # 手动加位置编码
            seq_len = h.size(1)
            pos_emb = sinusoidal_positional_encoding(
                seq_len, h.size(-1), device=h.device
            )
            h = h + pos_emb
            # 支持可变长度填充 mask
            h = self.transformer_encoder(h, src_key_padding_mask=mask)
        else:
            # Flash-Attn 内部已处理 RoPE，无需手动编码
            h = self.transformer_encoder(h)

        # 池化
        h = h.mean(dim=1)  # (B, d_model)
        logits = self.fc_out(h)

        # 动作噪声
        if self.training and self.use_action_noise:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        return logits

"""
# 安装 PyTorch（确保你用的是 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后安装 flash-attn
pip install flash-attn --no-build-isolation
"""
"""
TransformerPolicyNetwork
------------------------
将 (batch, seq_len, input_dim) 的状态序列 → (batch, num_actions) 的 logits。

✦ 设计要点
1. 先用 Linear 把输入映射到 d_model 维度；
2. 加可学习的位置编码（支持最长 max_seq_len）；
3. 使用 nn.TransformerEncoder 做时序特征提取；
4. 对时序维做 mean-pool，接全连接层得到动作 logits。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerPolicyNetwork(nn.Module):
    """轻量级策略网络：TransformerEncoder ➜ 池化 ➜ 动作 logits"""

    def __init__(
        self,
        input_dim: int,            # 传入 sensor / processor 的 state 向量维度（已统一）
        num_actions: int,          # 环境动作数量（你的 GridEnvironment.action_space_n）
        d_model: int = 128,        # Transformer 隐藏维度
        nhead: int = 4,            # 多头注意力头数
        num_layers: int = 2,       # Encoder 层数
        dim_feedforward: int = 256,
        max_seq_len: int = 16      # 允许的最大序列长度（默认足够你当前 2-token 输入）
    ) -> None:
        super().__init__()

        # 1) 输入映射到 d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2) 可学习的位置编码
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_model))

        # 3) TransformerEncoder
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

        # 4) 输出头
        self.fc_out = nn.Linear(d_model, num_actions)

        # 初始化（官方推荐方式）
        self._reset_parameters()

    # -------------------------------------------------------------

    def _reset_parameters(self):
        """按照 PyTorch 官方初始化 Transformer 的做法轻微调优"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = (batch_size, seq_len, input_dim)

        Returns:
            logits: Tensor, shape = (batch_size, num_actions)
        """
        # step-1 线性投影到 d_model
        h = self.input_proj(x)  # (B, L, d_model)

        # step-2 加上位置编码（只取前 L 个位置）
        seq_len = h.size(1)
        h = h + self.pos_emb[:seq_len]

        # step-3 TransformerEncoder
        h = self.transformer_encoder(h)  # (B, L, d_model)

        # step-4 时序池化（平均）
        h = h.mean(dim=1)  # (B, d_model)

        # step-5 分类头
        logits = self.fc_out(h)  # (B, num_actions)
        return logits


"""
⚙️ 使用方式简述
from models.transformer_policy import TransformerPolicyNetwork

net = TransformerPolicyNetwork(
    input_dim=graph.processor_hidden_size,   # 或你在 graph 中对齐的统一维度
    num_actions=env.action_space_n
)

seq = torch.randn(8, 2, graph.processor_hidden_size)  # (batch=8, seq_len=2, dim)
logits = net(seq)  # (8, num_actions)
注意

如果后续你想把序列长度扩展到 >16（例如加入时间窗口），把 max_seq_len 调大或改成动态注册方式即可；

本实现没有做软性约束（mask）。如果需要处理可变长度/填充，可在调用处自带 src_key_padding_mask 传入 self.transformer_encoder。

完成后，RLAgent 中即可 from models.transformer_policy import TransformerPolicyNetwork 正常使用。
"""
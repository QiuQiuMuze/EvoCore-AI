"""
utils.py
========
通用小工具函数集合
"""

from __future__ import annotations
import torch
import math


# ---------------------------------------------------------------------- #
# 1. 折扣回报
# ---------------------------------------------------------------------- #
def discounted_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """
    REINFORCE / PPO 常用：把一段 reward 序列折算成回报 G_t。

    Args
    ----
    rewards : list[float]
        r_0, r_1, …, r_{T-1}
    gamma : float
        折扣因子

    Returns
    -------
    torch.Tensor
        shape = (T,)
    """
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------------- #
# 2. 简易正余弦位置编码
# ---------------------------------------------------------------------- #
def sinusoidal_positional_encoding(seq_len: int, d_model: int, device="cpu") -> torch.Tensor:
    """
    生成与 Transformer 论文同款的 sin/cos 位置编码。

    说明：
    - 若你改用 learnable embedding，可忽略本函数；
    - 返回 shape = (seq_len, d_model)。

    Example
    -------
    >>> pos_emb = sinusoidal_positional_encoding(100, 128)
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


"""
如何使用
from utils import discounted_returns, sinusoidal_positional_encoding

# 1) 计算一条轨迹的回报
G = discounted_returns(reward_list, gamma=0.98)

# 2) 若你想用固定 sin/cos pos_emb 而不是 nn.Parameter：
pos_emb = sinusoidal_positional_encoding(seq_len=2, d_model=128, device="cuda")
transformer_input = tokens + pos_emb  # 自行广播 / 对齐
如果后续发现别的公共函数（如张量对齐、日志工具等）多次被用到，继续追加到 utils.py 即可，不必另开文件。
"""
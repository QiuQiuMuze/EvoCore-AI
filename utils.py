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



import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------- #
# 3. Intrinsic Curiosity Module (ICM)
# ---------------------------------------------------------------------- #
class IntrinsicCuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module:
    - inverse_net：预测 (s, s_next) → a，用于学习可逆特征
    - forward_net：预测 (s, a) → s_next_embedding，用于计算预测误差奖励

    Buffers:
    - self.states, self.next_states, self.actions 存储每步 transition
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-4
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 逆模型：拼接 state & next_state
        self.inverse_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # 前向模型：拼接 state & action(one-hot)
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 缓存
        self.states: list[torch.Tensor] = []
        self.next_states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def expand_state_dim(self, new_state_dim: int) -> None:
        """
        当环境／模型的 state_dim 扩大时，扩张 inverse_net 和 forward_net
        第一层 Linear 的输入维度，并保留旧权重。
        """
        old_dim = self.state_dim
        if new_state_dim <= old_dim:
            return

        # —— 扩张 inverse_net 的第一层（Linear(state_dim*2 → hidden)） ——
        old_lin = self.inverse_net[0]  # nn.Linear(state_dim*2, hidden_dim)
        new_lin = nn.Linear(new_state_dim * 2, old_lin.out_features).to(old_lin.weight.device)
        with torch.no_grad():
            # 复制旧权重到新 weight 的前 self.state_dim*2 列
            new_lin.weight[:, : self.state_dim * 2].copy_(old_lin.weight)
            new_lin.bias.copy_(old_lin.bias)
        self.inverse_net[0] = new_lin

        # —— 扩张 forward_net 的第一层（Linear(state_dim+action_dim → hidden)） ——
        old_lin = self.forward_net[0]  # nn.Linear(state_dim+action_dim, hidden_dim)
        new_lin = nn.Linear(new_state_dim + self.action_dim, old_lin.out_features).to(old_lin.weight.device)
        with torch.no_grad():
            # 复制旧 weight 的 state 部分
            new_lin.weight[:, : self.state_dim].copy_(old_lin.weight[:, : self.state_dim])
            # 复制旧 weight 的 action 部分
            new_lin.weight[:, self.state_dim : self.state_dim + self.action_dim]\
                .copy_(old_lin.weight[:, self.state_dim : self.state_dim + self.action_dim])
            new_lin.bias.copy_(old_lin.bias)
        self.forward_net[0] = new_lin

        # —— 扩张 forward_net 的最后一层 ——
        # old_final: nn.Linear(hidden_dim, old_dim)
        old_final = self.forward_net[2]
        # new_final: nn.Linear(hidden_dim, new_state_dim)
        new_final = nn.Linear(old_final.in_features, new_state_dim).to(old_final.weight.device)
        with torch.no_grad():
            # 复制旧权重到 new_final 的前 old_dim 行
            new_final.weight[:old_dim, :].copy_(old_final.weight)
            new_final.bias[:old_dim].copy_(old_final.bias)
        self.forward_net[2] = new_final


        # 更新记录
        self.state_dim = new_state_dim
        # ⚠️ 扩维后清空所有旧 transition 缓存，避免不同维度冲突
        self.states.clear()
        self.next_states.clear()
        self.actions.clear()


    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,       # shape=(state_dim,)
        next_state: torch.Tensor,  # shape=(state_dim,)
        action: torch.Tensor       # shape=() or (1,)
    ) -> float:
        """
        计算预测误差作为内在奖励，并缓存 transition。
        """
        # one-hot 编码 action
        a_onehot = F.one_hot(
            action.long().squeeze(),
            num_classes=self.action_dim
        ).to(dtype=state.dtype)
        # 前向预测下一 state
        fwd_input = torch.cat([state, a_onehot], dim=-1)
        pred_next = self.forward_net(fwd_input)
        # 预测误差
        intrinsic_reward = 0.5 * (pred_next - next_state).pow(2).sum().item()
        # 缓存 transition（detach 保持无梯度）
        self.states.append(state.detach())
        self.next_states.append(next_state.detach())
        self.actions.append(action.detach().squeeze().long())
        return intrinsic_reward

    def update_parameters(self) -> None:
        """
        基于缓存 transitions，计算 inverse & forward loss 并更新模型参数。
        """
        if not self.states:
            return
        # 堆叠 batch
        states = torch.stack(self.states, dim=0)         # (B, state_dim)
        next_states = torch.stack(self.next_states, dim=0)
        actions = torch.stack(self.actions, dim=0)       # (B,)
        # --- inverse loss ---
        inv_input = torch.cat([states, next_states], dim=1)  # (B, 2*state_dim)
        inv_logits = self.inverse_net(inv_input)             # (B, action_dim)
        inv_loss = F.cross_entropy(inv_logits, actions)
        # --- forward loss ---
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim)
        actions_onehot = actions_onehot.to(dtype=states.dtype)
        fwd_input = torch.cat([states, actions_onehot], dim=1)  # (B, state_dim+action_dim)
        pred_next = self.forward_net(fwd_input)                 # (B, state_dim)
        fwd_loss = 0.5 * (pred_next - next_states).pow(2).sum(dim=1).mean()
        # 总 loss
        loss = inv_loss + fwd_loss
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 清空缓存
        self.states.clear()
        self.next_states.clear()
        self.actions.clear()

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
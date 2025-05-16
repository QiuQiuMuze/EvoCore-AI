"""
RLAgent
=======
轻量级策略梯度代理（默认 REINFORCE）。

✦ 功能
1. 包装 TransformerPolicyNetwork；
2. 负责 action 采样、轨迹缓存；
3. 在 episode 结束后执行策略梯度更新；
4. 提供 save / load 接口方便断点续训。

✦ 依赖
- models.transformer_policy.TransformerPolicyNetwork
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List

from models.transformer_policy import TransformerPolicyNetwork



class RLAgent:
    """无值函数的纯策略代理（REINFORCE）。"""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        d_model: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        # 1) 先构造策略网
        self.policy_net = TransformerPolicyNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            d_model=d_model  # ← 真正用到传入的 d_model
        ).to(self.device)

        # 2) 再构造值函数
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 1)
        ).to(self.device)



        # 3) 优化器把两部分参数都加进来
        params = list(self.policy_net.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.gamma = gamma  # ← 保存折扣因子

        # —— 轨迹缓存 ——
        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float]       = []
        self.saved_states = []  # baseline 需要状态输入


    # --------------------------------------------------------------------- #
    #                           交互接口                                     #
    # --------------------------------------------------------------------- #

    def expand_value_head(self, new_input_dim):
        old_layer = self.value_head[0]  # 取 Sequential 中的 Linear 层
        old_input_dim = old_layer.in_features

        if new_input_dim <= old_input_dim:
            return  # 不需要扩展

        # 构建新 Linear 层（保留旧参数）
        new_layer = torch.nn.Linear(new_input_dim, 1).to(old_layer.weight.device)

        with torch.no_grad():
            # 拷贝旧权重（只拷贝前 old_input_dim 部分）
            new_layer.weight[:, :old_input_dim] = old_layer.weight
            new_layer.bias = old_layer.bias

        self.value_head = torch.nn.Sequential(new_layer)
        print(f"[🔁 升维] value_head 输入维度 {old_input_dim} → {new_input_dim}")

    def select_action(self, state_seq: torch.Tensor) -> int:
        """
        给定状态序列，采样一个动作。

        Args:
            state_seq: Tensor, shape=(1, seq_len, input_dim)

        Returns:
            int: 动作索引
        """
        state_seq = state_seq.to(self.device)
        logits = self.policy_net(state_seq)          # (1, num_actions)
        dist = Categorical(logits=logits)
        action = dist.sample()                       # Tensor([])
        self.log_probs.append(dist.log_prob(action)) # 缓存本步 log π(a|s)
        state_feat = state_seq.detach().mean(dim=1)  # (1, input_dim)
        self.saved_states.append(state_feat)  # ← 保存一份状态序列，用于 baseline 值函数
        return action.item()

    def store_reward(self, r: float) -> None:
        """在环境步结束后调用，缓存即时回报。"""
        self.rewards.append(r)

    # --------------------------------------------------------------------- #
    #                           学习更新                                     #
    # --------------------------------------------------------------------- #

    def _compute_returns(self) -> torch.Tensor:
        """
        折扣回报 G_t = Σ γ^k r_{t+k}

        Returns:
            Tensor, shape=(T,)
        """
        R = 0.0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # 归一化，提升数值稳定性
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 新 Actor-Critic 损失
        policy_loss = []
        value_loss = []

        for log_prob, state_feat, R in zip(self.log_probs, self.saved_states, returns):
            self.expand_value_head(state_feat.shape[-1])  # 确保 value_head 支持当前输入维度
            value = self.value_head(state_feat).squeeze()
            advantage = R - value.detach()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(torch.nn.functional.mse_loss(value, R))

        loss = torch.stack(policy_loss).sum() + 0.5 * torch.stack(value_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        self.saved_states.clear()

    # --------------------------------------------------------------------- #
    #                          模型持久化                                    #
    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_head.state_dict(),  # ← 新增
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str, map_location: str | torch.device | None = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_head.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


"""
说明要点

模块	             作用
select_action	前向推断 → Categorical → 采样 → 记录 log_prob
store_reward	在每个 env.step() 后调用，缓存即时奖励
finish_episode	计算折扣回报 → 标准化 → REINFORCE 更新
save / load	便于长时间训练中断点续训

如果后续改用 PPO / A2C 等算法，只需：

替换策略更新部分；

增加值函数或旧策略缓存。其他接口保持不变。
"""
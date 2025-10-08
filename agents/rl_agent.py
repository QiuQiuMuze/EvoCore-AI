# rl_agent.py

from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Optional
from env import logger
from models.transformer_policy import TransformerPolicyNetwork


class RLAgent:
    """
    支持 REINFORCE 和 PPO 的策略代理。

    参数：
      input_dim: 状态维度
      num_actions: 动作数量
      lr: 学习率
      gamma: 折扣因子
      d_model: Transformer 内部维度
      use_ppo: 是否使用 PPO（否则用 REINFORCE）
      ppo_epochs: PPO 更新轮数
      clip_epsilon: PPO 剪切范围
      value_coef: value loss 权重
      entropy_coef: entropy 正则权重
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        d_model: int = 128,
        use_ppo: bool = False,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.1,
        gae_lambda: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = input_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 策略网络 & 值网络
        self.policy_net = TransformerPolicyNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            d_model=d_model
        ).to(self.device)
        self.value_head = nn.Linear(input_dim, 1).to(self.device)

        # 优化器
        params = list(self.policy_net.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # PPO 参数
        self.use_ppo = use_ppo
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 经验缓存
        self.log_probs: List[torch.Tensor] = []
        self.saved_states: List[torch.Tensor] = []
        self.saved_actions: List[int] = []
        self.saved_values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []


    def expand_value_head(self, new_input_dim):
        old_layer = self.value_head
        old_input_dim = old_layer.in_features

        if new_input_dim <= old_input_dim:
            return

        # 构造新 Linear（保留旧权重）
        new_layer = torch.nn.Linear(new_input_dim, 1).to(self.device)
        with torch.no_grad():
            new_layer.weight[:, :old_input_dim] = old_layer.weight
            new_layer.bias = old_layer.bias
        self.value_head = new_layer
        logger.info(f"[RLAgent] expand value_head: {old_input_dim}→{new_input_dim}")

    def resize_state_dim(self, new_input_dim: int):
        """
        环境扩张后调用，重建 policy_net 并拷贝 input_proj 的重叠权重，
        再扩展 value_head。
        """
        old_policy = self.policy_net
        old_dim = self.state_dim

        # 1) 构建新策略网络（参数同原来）
        self.policy_net = TransformerPolicyNetwork(
            input_dim=new_input_dim,
            num_actions=self.num_actions,
            d_model=old_policy.transformer_encoder.layers[0].self_attn.embed_dim,
            nhead=old_policy.transformer_encoder.layers[0].self_attn.num_heads,
            num_layers=len(old_policy.transformer_encoder.layers),
            dim_feedforward=old_policy.transformer_encoder.layers[0].linear1.out_features,
            max_seq_len=old_policy.max_seq_len,
            use_action_noise=old_policy.use_action_noise
        ).to(self.device)

        # 2) 拷贝 input_proj 权重
        with torch.no_grad():
            old_proj = old_policy.input_proj
            new_proj = self.policy_net.input_proj
            share = min(old_dim, new_input_dim)
            new_proj.weight[:, :share].copy_(old_proj.weight[:, :share])
            new_proj.bias.copy_(old_proj.bias)

        logger.info(f"[RLAgent] policy_net input_dim {old_dim}→{new_input_dim}, weights copied")

        # 3) 扩展 value_head
        self.expand_value_head(new_input_dim)
        self.state_dim = new_input_dim


    def select_action(self, state_seq: torch.Tensor) -> int:
        """
        输入 state_seq (1, seq_len, input_dim)，返回动作索引，并缓存 log_prob、state_feat、value。
        """
        state_seq = state_seq.to(self.device)
        logits = self.policy_net(state_seq)  # (1, num_actions)
        dist = Categorical(logits=logits)

        action = dist.sample()                # Tensor([a])
        log_prob = dist.log_prob(action)      # Tensor([logp])

        # 缓存
        # state_feat: 简化后的状态表示 (input_dim,)
        state_feat = state_seq.detach().mean(dim=1).squeeze(0)  # (input_dim,)
        value = self.value_head(state_feat.unsqueeze(0)).squeeze()

        self.saved_states.append(state_feat)
        self.saved_actions.append(action.item())
        self.log_probs.append(log_prob.squeeze(0))
        self.saved_values.append(value)
        return action.item()

    def store_reward(self, r: float, done: bool = False) -> None:
        """在每个 env.step() 后调用，缓存即时回报和终止标记。"""
        self.rewards.append(r)
        self.dones.append(done)

    def _compute_returns_and_advantages(self, bootstrap_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """使用 GAE 计算回报与优势。"""
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        values = torch.stack(self.saved_values + [bootstrap_value]).view(-1)

        gae = torch.tensor(0.0, device=self.device)
        returns: List[torch.Tensor] = []
        advantages: List[torch.Tensor] = []

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * values[step + 1] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        returns_tensor = torch.stack(returns).view(-1)
        advantages_tensor = torch.stack(advantages).view(-1)

        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std(unbiased=False)
        if torch.isnan(adv_std) or adv_std < 1e-8:
            advantages_tensor = advantages_tensor - adv_mean
        else:
            advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)

        return returns_tensor.detach(), advantages_tensor.detach()

    def finish_episode(self, last_state_seq: torch.Tensor | None = None):
        """
        根据 use_ppo 决定使用 REINFORCE 或 PPO 更新，并清空缓存。
        """
        if last_state_seq is not None:
            last_state_seq = last_state_seq.to(self.device)
            if last_state_seq.dim() == 3:  # (1, L, D)
                last_feat = last_state_seq.mean(dim=1).squeeze(0)
            elif last_state_seq.dim() == 2:
                last_feat = last_state_seq.mean(dim=0)
            else:
                last_feat = last_state_seq

            cur_dim = last_feat.shape[-1]
            if cur_dim < self.state_dim:
                last_feat = F.pad(last_feat, (0, self.state_dim - cur_dim))
            elif cur_dim > self.state_dim:
                last_feat = last_feat[:self.state_dim]

            bootstrap_value = self.value_head(last_feat.unsqueeze(0)).squeeze()
        else:
            bootstrap_value = torch.tensor(0.0, device=self.device)

        returns, advantages = self._compute_returns_and_advantages(bootstrap_value)
        old_log_probs = torch.stack(self.log_probs)             # (T,)
        # —— 补丁：把所有 saved_states pad 到当前 state_dim 再 stack —— #
        padded_states = []
        for s in self.saved_states:
            cur_d = s.shape[-1]
            if cur_d < self.state_dim:
                # 右侧补零
                pad_sz = self.state_dim - cur_d
                s = F.pad(s, (0, pad_sz), value=0.0)
            elif cur_d > self.state_dim:
                # 截断多余维度（一般不会出现）
                s = s[:self.state_dim]
            padded_states.append(s)
        states = torch.stack(padded_states)  # (T, state_dim)

        actions = torch.tensor(self.saved_actions, device=self.device)  # (T,)
        old_values = torch.stack(self.saved_values).view(-1)        # (T,)

        # 多次 PPO 更新
        if self.use_ppo:
            for _ in range(self.ppo_epochs):
                # 新 log_probs & values
                # 把 states 视作 seq_len=1 序列
                seq = states.unsqueeze(1)  # (T,1,input_dim)
                logits = self.policy_net(seq)  # (T, num_actions)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)  # (T,)
                entropy = dist.entropy().mean()

                # 值函数预测
                values = self.value_head(states).squeeze(1)  # (T,)

                # PPO 损失
                ratio = torch.exp(new_log_probs - old_log_probs)
                s1 = ratio * advantages
                s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(values, returns)

                loss = policy_loss \
                       + self.value_coef * value_loss \
                       - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_head.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
        else:
            # REINFORCE 更新
            policy_losses = []
            value_losses = []
            # 重算 entropy
            logits = self.policy_net(states.unsqueeze(1))
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()

            for log_prob, value, R in zip(old_log_probs, old_values, returns):
                advantage = R - value.detach()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(F.mse_loss(value, R))

            loss = torch.stack(policy_losses).sum() \
                   + self.value_coef * torch.stack(value_losses).sum() \
                   - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_head.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

        # 清空缓存
        self.log_probs.clear()
        self.saved_states.clear()
        self.saved_actions.clear()
        self.saved_values.clear()
        self.rewards.clear()
        self.dones.clear()

    def save(self, path: str) -> None:
        """保存策略状态、值网络和优化器状态。"""
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict":  self.value_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str, map_location: Optional[str | torch.device] = None) -> None:
        """加载模型与优化器状态，支持断点续训。"""
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        self.value_head.load_state_dict(checkpoint["value_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            logger.warning("Optimizer state 与当前模型不匹配，已跳过加载")

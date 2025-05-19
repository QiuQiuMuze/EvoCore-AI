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
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List
from env import logger
from models.transformer_policy import TransformerPolicyNetwork



class RLAgent:
    """无值函数的纯策略代理（REINFORCE）。"""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        lr: float = 3e-4,
        use_epsilon: bool = False,
        gamma: float = 0.99,
        d_model: int = 128,
        use_entropy: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = input_dim
        self.num_actions = num_actions
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

        # —— 探索 & 正则超参数 ——
        self.use_epsilon = use_epsilon
        self.epsilon = 0.2 if use_epsilon else 0.0            # ε-greedy 探索率
        self.entropy_coef = 0.025      # Entropy 正则系数
        self.use_entropy = use_entropy

        # —— 轨迹缓存 ——
        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float]       = []
        self.saved_states = []            # baseline 需要状态输入
        self.saved_logits: List[torch.Tensor] = []  # 用于计算 entropy



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

    def resize_state_dim(self, new_input_dim: int):
        """
        环境扩张后调用，重建 policy_net 并拷贝 input_proj 的重叠部分权重，
        再扩展 value_head。
        """
        old_policy = self.policy_net
        old_dim = self.state_dim
        # 1) 构建新网络
        self.policy_net = TransformerPolicyNetwork(
            input_dim=new_input_dim,
            num_actions=self.num_actions,
            d_model=old_policy.transformer_encoder.layers[0].self_attn.embed_dim,
            nhead=old_policy.transformer_encoder.layers[0].self_attn.num_heads,
            num_layers=len(old_policy.transformer_encoder.layers),
            dim_feedforward=old_policy.transformer_encoder.layers[0].linear1.out_features,
            max_seq_len=old_policy.pos_emb.shape[0],
            use_action_noise=old_policy.use_action_noise
        ).to(self.device)

        # 2) 拷贝重叠的 input_proj 权重和 bias
        with torch.no_grad():
            old_proj = old_policy.input_proj    # Linear(old_dim → d_model)
            new_proj = self.policy_net.input_proj
            shared = min(old_dim, new_input_dim)
            # weight shape = [d_model, input_dim]
            new_proj.weight[:, :shared].copy_(old_proj.weight[:, :shared])
            new_proj.bias.copy_(old_proj.bias)

        logger.info(f"[RLAgent] policy_net input_dim {old_dim}→{new_input_dim}, 权重已拷贝")

        # 3) 扩展 value_head
        self.expand_value_head(new_input_dim)

        # 4) 更新内部记录
        self.state_dim = new_input_dim
    def select_action(self, state_seq: torch.Tensor) -> int:
        """
        给定状态序列，采样一个动作。

        Args:
            state_seq: Tensor, shape=(1, seq_len, input_dim)

        Returns:
            int: 动作索引
        """
        state_seq = state_seq.to(self.device)
        logits = self.policy_net(state_seq)  # (1, num_actions)
        # —— 如果是 eval 模式，直接贪心取 max ——
        if not self.policy_net.training:
            return int(torch.argmax(logits, dim=-1).item())

        # — ε-greedy 行为 —
        if self.policy_net.training and self.use_epsilon and random.random() < self.epsilon:
            # 随机动作
            action = torch.randint(
                low=0,
                high=logits.size(-1),
                size=(1,),
                device=self.device
            )
            # uniform log_prob = log(1/num_actions)
            log_prob = torch.log(
                torch.ones_like(action, dtype=torch.float, device=self.device)
                / logits.size(-1)
            )
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()                   # Tensor([a])
            log_prob = dist.log_prob(action)         # 缓存 log π(a|s)
        # —— 只有在训练模式下，才把 log_prob、state、logits 存入缓存 ——
        if self.policy_net.training:
            self.log_probs.append(log_prob)
            self.saved_logits.append(logits.squeeze(0))
            state_feat = state_seq.detach().mean(dim=1)  # (1, dim)
            self.saved_states.append(state_feat)
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

        import torch.nn.functional as F
        for log_prob, state_feat, R in zip(self.log_probs, self.saved_states, returns):
            # 1) 拿到当前 value_head 的输入维度
            linear = self.value_head[0]  # Sequential 中的 Linear 层
            in_dim = linear.in_features

            # 2) 如果 state_feat 比网络输入大，先扩维网络（只升不缩）
            if state_feat.shape[-1] > in_dim:
                self.expand_value_head(state_feat.shape[-1])
                linear = self.value_head[0]
                in_dim = linear.in_features

            # 3) 如果 state_feat 比网络输入小，pad 零补齐
            cur_dim = state_feat.shape[-1]
            if cur_dim < in_dim:
                pad_size = in_dim - cur_dim
                # pad 格式 (left, right)，这里只有最后一维右侧补 pad_size
                state_feat = F.pad(state_feat, (0, pad_size), value=0.0)

            # 4) 现在 state_feat.shape[-1] == in_dim，安全调用
            value = self.value_head(state_feat).squeeze()

            advantage = R - value.detach()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(torch.nn.functional.mse_loss(value, R))

        # — 计算 Entropy bonus —
        if self.saved_logits:
            L = torch.stack(self.saved_logits)
            P = torch.softmax(L, dim=-1)
            entropy = -(P * torch.log(P + 1e-8)).sum(dim=-1).mean()
        else:
            entropy = torch.tensor(0.0, device=self.device)

        # 总损失 = policy + 0.5 * value - entropy_coef * entropy
        if self.use_entropy:
            loss = (
                    torch.stack(policy_loss).sum()
                    + 0.5 * torch.stack(value_loss).sum()
                    - self.entropy_coef * entropy
            )
        else:
            loss = (
                    torch.stack(policy_loss).sum()
                    + 0.5 * torch.stack(value_loss).sum()
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        self.saved_states.clear()
        self.saved_logits.clear()  # 清空 entropy 缓存

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
        self.policy_net.load_state_dict(
            checkpoint["policy_state_dict"],
            strict=False
        )

        self.value_head.load_state_dict(checkpoint["value_state_dict"])
        # 评估模式下，optimizer state 可能因参数变动无法完全匹配；捕获错误并跳过
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            # 可选：打印一条 warning
            logger.warning("optimizer state 与当前模型不匹配，已跳过 optimizer 加载")


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
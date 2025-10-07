"""CogUnit mixin module generated from the legacy monolith."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CogUnit

import random
import torch

from env import logger
from meta_cognition import MetaCognition


class LearningMixin:
    def mini_learn(self, input_tensor, target_tensor, lr=0.001):
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)

        # Forward
        output = self.function(input_tensor)

        # Loss
        loss = torch.nn.functional.mse_loss(output, target_tensor)

        # Backward
        self.function.zero_grad()
        loss.backward()

        # Manual parameter update
        with torch.no_grad():
            for param in self.function.parameters():
                if param.grad is not None:
                    param.copy_(param - lr * param.grad)


        logger.debug(f"[Mini-Learn] {self.id} loss={loss.item():.4f} (lr={lr})")

    def compute_self_reward(self, input_tensor, output_tensor):
        """
        简单 self-reward：如果输出能跟输入保持一致性，就获得小奖励
        """
        if input_tensor.shape != output_tensor.shape:
            output_tensor = output_tensor[:, :input_tensor.shape[1]]  # 防止维度不同
        error = torch.mean((input_tensor - output_tensor) ** 2)
        reward = 0.01 * (self.input_size / 50) * (1.0 - error.item())  # error越小奖励越高
        return max(reward, 0.0)  # 不让奖励为负数

    def evaluate_self(self, min_rate=0.3):
        """
        检查最近表现，若低于 min_rate 返回 True 表示需要变异/调整。
        """
        rate = self.meta.recent_success_rate()
        if rate is None:
            return False
        return rate < min_rate

    def request_upgrade(self, target_role=None, reason=""):
        """
        元认知评估后触发：记录一次升级意图，
        CogGraph 后续可以检测到并执行真正的变异/重构。
        """
        # 1) 清空 MetaCognition 历史
        self.meta = MetaCognition(history_len=self.meta.reward_trace.maxlen)
        # 2) 基因轻扰动
        for k in ["sensor_bias", "processor_bias", "emitter_bias"]:
            self.gene[k] += random.gauss(0, 0.05)
        # 3) 网络参数加小噪声
        for p in self.function.parameters():
            p.data += torch.randn_like(p) * 0.01
        logger.info(f"[Meta-升级] {self.id}, {self.role} 因“{reason}”触发自我进化，开始思考赛博人生，觉得自己又行了。新gene={self.gene}")

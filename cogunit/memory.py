"""CogUnit mixin module generated from the legacy monolith."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CogUnit

import torch

from env import logger


class MemoryMixin:
    def get_recent_outputs(self, n=5):
        """按时间顺序返回最近 n 帧输出（张量列表）"""
        L = min(n, self.output_history_tensor.size(0))
        idxs = [(self.output_history_ptr - i - 1) % L for i in reversed(range(L))]
        return [self.output_history_tensor[i] for i in idxs]

    def is_worthy_of_memory(self):
        """根据不同角色，判断该细胞是否值得加入记忆池"""
        if self.age < 100:
            return False  # 太年轻的不记

        if self.role == "sensor":
            # 感知单元：至少要有两帧输出才能计算变化
            if len(self.output_history) < 2:
                return False

            # 1) 计算 L1 变化量，先对齐到最小公共长度
            changes = []
            for prev, curr in zip(self.output_history, self.output_history[1:]):
                p = prev.view(-1)
                c = curr.view(-1)
                L = min(p.numel(), c.numel())
                changes.append((c[:L] - p[:L]).abs().sum().item())

            if not changes:
                return False

            avg_change = sum(changes) / len(changes)
            SENSOR_CHANGE_THRESHOLD = 5.0
            return avg_change > SENSOR_CHANGE_THRESHOLD



        elif self.role == "processor":
            # 处理单元应关注调用频率 & 输出多样性
            if getattr(self, "avg_recent_calls", 0) < 0.75:
                return False
            if len(self.output_history) < 2:
                return False
            total_diff = 0.0
            count = 0
            for prev, curr in zip(self.output_history, self.output_history[1:]):
                # 只比较 shape 相同的
                if prev.shape == curr.shape:
                    total_diff += torch.norm(curr - prev).item()
                    count += 1

            if count == 0:
                return False

            variation = total_diff / count
            return variation > 0.05  # 输出变化足够丰富


        elif self.role == "emitter":
            # 行为单元应关注任务完成情况和激活频率（活跃但非重复）
            if self.avg_recent_calls < 2.0:
                return False
            if len(self.output_history) < 2:
                return False
            diff = sum(
                torch.norm(self.output_history[i] - self.output_history[i + 1]).item()
                for i in range(len(self.output_history) - 1)
            ) / (len(self.output_history) - 1)
            return 0.01 < diff < 0.5  # 太低代表退化，太高可能随机扰动


        return False

    def add_to_local_memory(self):
        self.local_memory_pool = [m for m in self.local_memory_pool if "score" in m]
        # —— 对齐 output_history 到同一长度 ——
        import torch.nn.functional as F
        aligned_history = None
        if len(self.output_history) >= 3:
            # 取最大元素数量
            max_len = max(t.numel() for t in self.output_history)
            aligned_history = []
            for t in self.output_history:
                vec = t.view(-1)  # 拉平
                if vec.numel() < max_len:
                    # 右侧补 0
                    vec = F.pad(vec, (0, max_len - vec.numel()), value=0)
                else:
                    # 长则截断
                    vec = vec[:max_len]
                aligned_history.append(vec)

        if self.role == "sensor":
            if len(aligned_history) >= 2:
                # 假设 output_history 至少有 2 帧
                hist = [t.view(-1) for t in self.output_history]
                diffs = []
                for prev, curr in zip(self.output_history, self.output_history[1:]):
                    p = prev.view(-1)
                    c = curr.view(-1)
                    L = min(p.numel(), c.numel())
                    diffs.append((c[:L] - p[:L]).abs().sum().item())

                variation = sum(diffs) / len(diffs)

                score = variation
            else:
                score = 0

        elif self.role == "processor":
            # 处理：输出多样性 + 调用频率（已对齐）
            if aligned_history:
                diffs = [
                    torch.norm(aligned_history[i] - aligned_history[i+1]).item()
                    for i in range(len(aligned_history) - 1)
                ]
                diversity = sum(diffs) / len(diffs)
            else:
                diversity = 0
            score = diversity * 0.5 + getattr(self, "avg_recent_calls", 0) * 0.3


        elif self.role == "emitter":
            # 输出：活跃性 + 输出稳定性（已对齐）
            if aligned_history:
                diffs = [
                    torch.norm(aligned_history[i] - aligned_history[i+1]).item()
                    for i in range(len(aligned_history) - 1)
                ]
                avg_diff = sum(diffs) / len(diffs)
                stability = 1.0 if 0.01 < avg_diff < 0.5 else 0.0
            else:
                stability = 0
            score = getattr(self, "avg_recent_calls", 0) * 0.5 + stability * 0.3


        else:
            score = self.energy + getattr(self, "avg_recent_calls", 0)

        """将自身压缩为记忆格式，加入 local memory pool"""
        mem = {
            "gene": self.gene.copy(),
            "output": self.last_output.clone(),
            "role": self.role,
            "age": self.age,
            "hidden_size": self.hidden_size,
            "score": score

        }
        self.local_memory_pool.append(mem)

        # 控制最大记忆数量，移除最弱
        if len(self.local_memory_pool) > self.memory_pool_limit:
            self.local_memory_pool.sort(key=lambda m: m["score"])
            self.local_memory_pool.pop(0)  # 移除最弱
        logger.info(
            f"[记忆加入] {self.id}（{self.role}，Age={self.age}）加入本地记忆池，评分={mem['score']:.2f}，当前共 {len(self.local_memory_pool)} 条")

    def record_memory(self, state: torch.Tensor, action, reward: float, outcome: str):
        """
        外部在每步做完 reward 计算后调用：
        state: 当时传入 update() 的输入张量（含 env+goal）
        action: 本轮 emitter/processor 选的动作标识
        reward: 本轮环境＋自我评价总 reward
        outcome: 'success' or 'fail' or 自定义标签
        """
        self.memory_buffer.add(state, action, reward, outcome)

    def recall(self, query_state: torch.Tensor, k: int = 5, metric: str = 'cosine'):
        """
        查历史经验，返回字典列表：
        [{'state':…, 'action':…, 'reward':…, 'outcome':…}, …]
        """
        return self.memory_buffer.recall(query_state, k, metric)

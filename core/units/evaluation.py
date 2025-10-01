import random

import torch
import torch.nn.functional as F

from env import logger
from meta_cognition import MetaCognition


class EvaluationMixin:
    def evaluate_self(self, min_rate=0.3):
        rate = self.meta.recent_success_rate()
        if rate is None:
            return False
        return rate < min_rate

    def request_upgrade(self, target_role=None, reason=""):
        self.meta = MetaCognition(history_len=self.meta.reward_trace.maxlen)
        for k in ["sensor_bias", "processor_bias", "emitter_bias"]:
            self.gene[k] += random.gauss(0, 0.05)
        for p in self.function.parameters():
            p.data += torch.randn_like(p) * 0.01
        logger.info(
            f"[Meta-升级] {self.id}, {self.role} 因“{reason}”触发自我进化，开始思考赛博人生，觉得自己又行了。新gene={self.gene}"
        )

    def is_worthy_of_memory(self):
        if self.age < 100:
            return False
        if self.role == "sensor":
            if len(self.output_history) < 2:
                return False
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
            if getattr(self, "avg_recent_calls", 0) < 0.75:
                return False
            if len(self.output_history) < 2:
                return False
            total_diff = 0.0
            count = 0
            for prev, curr in zip(self.output_history, self.output_history[1:]):
                if prev.shape == curr.shape:
                    total_diff += torch.norm(curr - prev).item()
                    count += 1
            if count == 0:
                return False
            variation = total_diff / count
            return variation > 0.05
        elif self.role == "emitter":
            if self.avg_recent_calls < 2.0:
                return False
            if len(self.output_history) < 2:
                return False
            diff = sum(
                torch.norm(self.output_history[i] - self.output_history[i + 1]).item()
                for i in range(len(self.output_history) - 1)
            ) / (len(self.output_history) - 1)
            return 0.01 < diff < 0.5
        return False

    def add_to_local_memory(self):
        self.local_memory_pool = [m for m in self.local_memory_pool if "score" in m]
        aligned_history = None
        if len(self.output_history) >= 3:
            max_len = max(t.numel() for t in self.output_history)
            aligned_history = []
            for t in self.output_history:
                vec = t.view(-1)
                if vec.numel() < max_len:
                    vec = F.pad(vec, (0, max_len - vec.numel()), value=0)
                else:
                    vec = vec[:max_len]
                aligned_history.append(vec)
        if self.role == "sensor":
            if len(aligned_history or []) >= 2:
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
            if aligned_history:
                diffs = [
                    torch.norm(aligned_history[i] - aligned_history[i + 1]).item()
                    for i in range(len(aligned_history) - 1)
                ]
                diversity = sum(diffs) / len(diffs)
            else:
                diversity = 0
            score = diversity * 0.5 + getattr(self, "avg_recent_calls", 0) * 0.3
        elif self.role == "emitter":
            if aligned_history:
                diffs = [
                    torch.norm(aligned_history[i] - aligned_history[i + 1]).item()
                    for i in range(len(aligned_history) - 1)
                ]
                avg_diff = sum(diffs) / len(diffs)
                stability = 1.0 if 0.01 < avg_diff < 0.5 else 0.0
            else:
                stability = 0
            score = getattr(self, "avg_recent_calls", 0) * 0.5 + stability * 0.3
        else:
            score = self.energy + getattr(self, "avg_recent_calls", 0)
        mem = {
            "gene": self.gene.copy(),
            "output": self.last_output.clone(),
            "role": self.role,
            "age": self.age,
            "hidden_size": self.hidden_size,
            "score": score,
        }
        self.local_memory_pool.append(mem)
        if len(self.local_memory_pool) > self.memory_pool_limit:
            self.local_memory_pool.sort(key=lambda m: m["score"])
            self.local_memory_pool.pop(0)
        logger.info(
            f"[记忆加入] {self.id}（{self.role}，Age={self.age}）加入本地记忆池，评分={mem['score']:.2f}，当前共 {len(self.local_memory_pool)} 条"
        )

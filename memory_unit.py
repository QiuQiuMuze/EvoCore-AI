import random
import torch
import torch.nn.functional as F
from collections import deque

try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class MemoryBuffer:
    """
    经验回放缓冲池：
      ✔ 精确 KNN（支持 GPU）
      ✔ 可选 FAISS 加速（近似 ANN）
      ✔ 支持 reward filter / embedding 输出
    """

    def __init__(self, maxlen=200, device=None, use_faiss=False):
        self.buffer = deque(maxlen=maxlen)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.dim = None

        if self.use_faiss:
            res = faiss.StandardGpuResources()
            self.faiss_cpu_index = faiss.IndexFlatL2(0)
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_cpu_index)
            self._id_map = []

        self._next_id = 0

    def add(self, state: torch.Tensor, action, reward: float, outcome: str):
        s_cpu = state.detach().cpu().clone()
        record = {
            "id": self._next_id,
            "state": s_cpu,
            "action": action,
            "reward": reward,
            "outcome": outcome
        }

        if len(self.buffer) == self.buffer.maxlen and self.use_faiss:
            old_id = self._id_map.pop(0)
            self.faiss_index.remove_ids(faiss.IDSelectorBatch(np.array([old_id], dtype=np.int64)))

        self.buffer.append(record)

        if self.use_faiss:
            vec = s_cpu.numpy().astype("float32").reshape(1, -1)
            if self.dim is None:
                self.dim = vec.shape[1]
                self.faiss_cpu_index = faiss.IndexFlatL2(self.dim)
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_cpu_index)
                self._id_map = []
            self.faiss_index.add_with_ids(vec, np.array([self._next_id], dtype=np.int64))
            self._id_map.append(self._next_id)

        self._next_id += 1

    def recall(self, query_state: torch.Tensor, k: int = 5, metric: str = "cosine",
               reward_filter: float = None, return_embedding=False):
        """
        返回与 query_state 最相似的 top-k 条记录。
        支持：
          - metric: "cosine" 或 "l2"
          - reward_filter: 仅回忆 reward ≥ 该值的经验
          - return_embedding: 返回 (record, embedding) 元组列表
        """
        if not self.buffer:
            return []

        # — 过滤满足条件的记录 —
        records = list(self.buffer)
        if reward_filter is not None:
            records = [rec for rec in records if rec["reward"] >= reward_filter]

        if not records:
            return []

        # — FAISS 优先 —
        if self.use_faiss and reward_filter is None:
            q = query_state.detach().cpu().numpy().astype("float32").reshape(1, -1)
            D, I = self.faiss_index.search(q, min(k, len(records)))
            ids = I[0].tolist()
            results = [rec for rec in self.buffer if rec["id"] in ids]
        else:
            states = torch.stack([rec["state"].to(self.device).view(-1) for rec in records])
            q = query_state.detach().to(self.device).view(-1).unsqueeze(0).expand_as(states)

            if metric == "cosine":
                sims = F.cosine_similarity(q, states, dim=1)
            else:
                sims = -torch.norm(states - q, dim=1)

            topk_idx = sims.topk(min(k, len(sims))).indices.tolist()
            results = [records[i] for i in topk_idx]

        if return_embedding:
            embeddings = [rec["state"].view(-1) for rec in results]
            return list(zip(results, embeddings))
        else:
            return results

    def get_all_records(self):
        """返回当前所有记忆内容的副本列表。"""
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
        if self.use_faiss:
            self.faiss_index.reset()
            self._id_map.clear()

    def sample(self, batch_size: int, min_reward: float = None, require_action: bool = True):
        """返回经过过滤的随机样本列表。

        Args:
            batch_size: 期望返回的记录数量（若不足则返回全部）。
            min_reward: 仅保留 reward ≥ min_reward 的记录；为 None 时不限制。
            require_action: 若为 True，仅保留包含 action 的记录。
        """
        if batch_size <= 0 or not self.buffer:
            return []

        if min_reward is None and not require_action:
            candidates = list(self.buffer)
        else:
            candidates = [
                rec for rec in self.buffer
                if (not require_action or rec.get("action") is not None)
                and (min_reward is None or rec.get("reward", 0.0) >= min_reward)
            ]

        if not candidates:
            return []

        if len(candidates) <= batch_size:
            return list(candidates)

        return random.sample(candidates, batch_size)


"""
# GPU 版本（适用于 CUDA 环境）
pip install faiss-gpu
"""
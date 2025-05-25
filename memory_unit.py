# memory_unit.py

import torch
import torch.nn.functional as F
from collections import deque

# Optional: 如果想用 FAISS 做近似检索，加速海量数据场景
try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class MemoryBuffer:
    """
    经验回放缓冲池，支持：
      1) PyTorch GPU 上的精确 KNN 检索（cosine 或 L2）
      2) 可选的 FAISS 近似 ANN 检索（GPU / CPU）

    用法不变：
      buf = MemoryBuffer(maxlen=200, use_faiss=True)
      buf.add(state, action, reward, outcome)
      recs = buf.recall(query_state, k=5)
    """

    def __init__(self, maxlen=200, device=None, use_faiss=False):
        self.buffer = deque(maxlen=maxlen)
        # 自动选择 device：优先 CUDA
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # FAISS 近似检索开关
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.dim = None
        if self.use_faiss:
            # 初始化一个空的 FAISS GPU index（后续补维度）
            res = faiss.StandardGpuResources()
            self.faiss_cpu_index = faiss.IndexFlatL2(0)
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_cpu_index)
            self._id_map = []  # 保存插入顺序对应的 record id
        self._next_id = 0

    def add(self, state: torch.Tensor, action, reward: float, outcome: str):
        """
        将经验存入缓冲池，并同步更新 FAISS（如果启用）。
        state: 任意形状 Tensor
        """
        # 保留原始 state 在 CPU
        s_cpu = state.detach().cpu().clone()
        record = {
            "id": self._next_id,
            "state": s_cpu,
            "action": action,
            "reward": reward,
            "outcome": outcome
        }
        # 如满则自动出队
        if len(self.buffer) == self.buffer.maxlen and self.use_faiss:
            # 删除最旧一条在 FAISS 中的索引
            old_id = self._id_map.pop(0)
            self.faiss_index.remove_ids(faiss.IDSelectorBatch(np.array([old_id], dtype=np.int64)))
        self.buffer.append(record)

        # 更新 FAISS index
        if self.use_faiss:
            vec = s_cpu.numpy().astype("float32").reshape(1, -1)
            if self.dim is None:
                # 首次插入时定维度并重建 index
                self.dim = vec.shape[1]
                self.faiss_cpu_index = faiss.IndexFlatL2(self.dim)
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_cpu_index)
                self._id_map = []
            self.faiss_index.add_with_ids(vec, np.array([self._next_id], dtype=np.int64))
            self._id_map.append(self._next_id)

        self._next_id += 1

    def recall(self, query_state: torch.Tensor, k: int = 5, metric: str = "cosine"):
        """
        返回与 query_state 最相似的 top-k 条记录列表。
        metric: "cosine" 或 "l2"
        """
        if not self.buffer:
            return []

        # 1) 优先走 FAISS 近似检索
        if self.use_faiss:
            q = query_state.detach().cpu().numpy().astype("float32").reshape(1, -1)
            D, I = self.faiss_index.search(q, min(k, len(self.buffer)))
            ids = I[0].tolist()
            # 按原 buffer 顺序过滤
            return [rec for rec in self.buffer if rec["id"] in ids]

        # 2) PyTorch 精确检索（在 GPU 上计算 similarity/dist）
        # 拼状态矩阵 [N, D]
        states = torch.stack([rec["state"].to(self.device).view(-1) for rec in self.buffer])
        q = query_state.detach().to(self.device).view(-1).unsqueeze(0).expand_as(states)

        if metric == "cosine":
            sims = F.cosine_similarity(q, states, dim=1)         # 越大越相似
        else:
            sims = -torch.norm(states - q, dim=1)                # 负值越大越相似

        topk_idx = sims.topk(min(k, len(sims))).indices.tolist()
        return [self.buffer[i] for i in topk_idx]

    def clear(self):
        """清空所有历史记忆。"""
        self.buffer.clear()
        if self.use_faiss:
            self.faiss_index.reset()
            self._id_map.clear()

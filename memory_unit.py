# memory_unit.py
import torch
import torch.nn.functional as F
from collections import deque

class MemoryBuffer:
    """
    一个简单的环形经验池，
    支持按状态相似度检索 top-k 条历史经验。
    """
    def __init__(self, maxlen=200):
        self.buffer = deque(maxlen=maxlen)

    def add(self, state: torch.Tensor, action, reward: float, outcome: str):
        # 一律保存到 CPU，避免显存爆炸
        self.buffer.append({
            'state': state.detach().cpu().clone(),
            'action': action,
            'reward': reward,
            'outcome': outcome
        })

    def recall(self, query_state: torch.Tensor, k: int = 5, metric: str = 'cosine'):
        """
        返回与 query_state 最相似的 k 条记录。
        metric: 'cosine' 或 'l2'
        """
        if not self.buffer:
            return []

        # 构造状态矩阵 [N, D]
        states = torch.stack([m['state'].view(-1) for m in self.buffer])
        q = query_state.detach().cpu().view(-1).unsqueeze(0).expand_as(states)

        if metric == 'cosine':
            sims = F.cosine_similarity(q, states, dim=1)  # 越大越相似
        else:
            sims = -torch.norm(states - q, dim=1)        # 负的 L2 距离

        topk = sims.topk(min(k, len(sims))).indices.tolist()
        return [self.buffer[i] for i in topk]

    def clear(self):
        """清空所有历史记忆。"""
        self.buffer.clear()
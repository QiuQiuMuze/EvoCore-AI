# memory_net_unit.py

import torch
from memory_unit import MemoryBuffer
from models.transformer_policy import TransformerPolicyNetwork
from typing import Optional
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch import cdist


def fgsm_attack(model, loss_fn, data_vec, epsilon=0.05):
    """
    兼容 Transformer 和普通 MLP/CNN 的 FGSM。
    - model: feature_extractor，可能是 TransformerPolicyNetwork
    - data_vec: 特征向量，Transformer 情况下是扁平化后再池化的向量，
                我们把它 reshape 回 (1, L, C) 才能送回 transformer。
    """
    device = data_vec.device
    # 如果是 TransformerPolicyNetwork，则把 data_vec 恢复成 (1, L, C)
    if isinstance(model, TransformerPolicyNetwork):
        seq_len = model.max_seq_len
        C = model.input_proj.in_features
        # data_vec 长度应该是 d_model（池化后维度），这里我们直接对 Transformer 的 head 做对抗
        # 所以直接把 data_vec 当成 logits 前激活输入，这里简单求梯度：
        x = data_vec.detach().clone().unsqueeze(0)           # (1, d_model)
        x.requires_grad_(True)
        logits = model.fc_out(x) if hasattr(model, 'fc_out') else model(x)
        # 造一个“正确”label 1（抗体）
        label = torch.tensor([1], device=device)
        loss = loss_fn(logits, label)
        loss.backward()
        adv = x + epsilon * x.grad.sign()
        return adv.detach().squeeze(0)

    # 否则，data_vec 本身就是模型的直接输入 (1, D)／(D,)
    v = data_vec.detach().clone().requires_grad_(True)
    logits = model(v.unsqueeze(0))
    label = torch.tensor([1], device=device)
    loss = loss_fn(logits, label)
    loss.backward()
    adv = v + epsilon * v.grad.sign()
    return adv.detach()

class MemoryNetUnit(MemoryBuffer):
    """
    记忆网络单元：继承自 MemoryBuffer，
    专门用于存储“攻击签名 → 防御策略”（抗体）并进行匹配。
    """

    def __init__(self, maxlen=500, device=None, use_faiss=False, feature_extractor: Optional[torch.nn.Module]=None):
        """
        初始化 MemoryNetUnit
        :param maxlen: 最大记忆容量
        :param device: 张量存储设备
        :param use_faiss: 是否启用 FAISS 加速
        """
        super().__init__(maxlen=maxlen, device=device, use_faiss=use_faiss)
        # —— 新增这一行 ——
        try:
            import faiss
            self._FAISS_AVAILABLE = True
        except ImportError:
            self._FAISS_AVAILABLE = False

        self.feature_extractor = feature_extractor



    # -------- GPU 距离矩阵版 KNN --------
    def _gpu_knn(self, query_vec: torch.Tensor, k: int = 8):
        """
        不依赖 faiss，用 torch.cdist 在 GPU 上一次性算完 L2 距离。
        返回 buffer 中最相近的 k 条记录（list[dict]，已按距离升序）。
        """
        if len(self.buffer) == 0:
            return []

        # [M, D] 记忆矩阵
        mem = torch.stack([rec["state"].to(query_vec.device)
                           for rec in self.buffer])          # (M, D)
        q   = query_vec.view(1, -1)                          # (1, D)
        dist = cdist(q, mem).squeeze(0)                      # (M,)

        k = min(k, mem.size(0))
        topk_idx = torch.topk(dist, k, largest=False).indices  # 最小距离
        return [self.buffer[i] for i in topk_idx.cpu()]

    def store_attack(self, attack_vec: torch.Tensor, defense_action: dict):
        # 原有存抗体
        self.add(state=attack_vec, action=defense_action, reward=1.0, outcome="antibody")

        if self.feature_extractor is not None:
            # 临时切到 eval()，在 enable_grad() 下计算对抗样本
            prev_mode = self.feature_extractor.training
            self.feature_extractor.eval()
            with torch.enable_grad():
                self.feature_extractor.zero_grad()
                loss_fn = torch.nn.CrossEntropyLoss()
                adv_vec = fgsm_attack(
                    self.feature_extractor,
                    loss_fn,
                    attack_vec.to(self.device),
                    epsilon=0.05
                )
            # 存对抗样本
            self.add(state=adv_vec.cpu(), action=defense_action,
                     reward=1.0, outcome="antibody_adv")
            # 恢复原来的训练/推理模式
            self.feature_extractor.train(prev_mode)
    def trim(self, keep_last: int = 800):
        """保留最近 keep_last 条，其余丢弃（含 faiss index 对齐）"""
        if len(self.buffer) <= keep_last:
            return
        # 简单 FIFO
        while len(self.buffer) > keep_last:
            self.buffer.popleft()
        if self._FAISS_AVAILABLE and hasattr(self, "_index"):
            self._rebuild_faiss_index()

    # -------- 新增 --------
    def _rebuild_faiss_index(self):
        """将当前 buffer 重灌到 faiss 索引；仅当 _FAISS_AVAILABLE=True 时调用"""
        if not self._FAISS_AVAILABLE:
            return
        import faiss, numpy as np
        vecs = np.stack([rec["state"].cpu().numpy().astype("float32")
                         for rec in self.buffer])
        d = vecs.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(vecs)

    def match(self, query_vec: torch.Tensor, k: int = 3, similarity_threshold: float = 0.8):
        """
        匹配最相似的历史攻击签名，并返回对应防御策略
        :param query_vec: 当前待匹配的攻击特征向量
        :param k: 检索最近邻数量
        :param similarity_threshold: 相似度阈值，低于则返回 None
        :return: 最匹配的防御策略(dict)或 None
        """
        # 仅匹配 reward >= 1.0 的抗体记录
        # ① 优先用 faiss（若已构建）；② 否则走 GPU KNN
        if getattr(self, "_FAISS_AVAILABLE", False) and hasattr(self, "_index"):
            import faiss, numpy as np
            q = query_vec.detach().cpu().view(1, -1).numpy().astype("float32")
            D, I = self._index.search(q, min(k, len(self.buffer)))   # I: (1, k)
            records = [self.buffer[i] for i in I[0] if i != -1]
        else:
            records = self._gpu_knn(query_vec, k=k)

        if not records:
            return None

        # 计算相似度并选出最高
        states = torch.stack([rec["state"].to(self.device).view(-1) for rec in records])
        q = query_vec.detach().to(self.device).view(-1).unsqueeze(0).expand_as(states)
        sims = torch.nn.functional.cosine_similarity(q, states, dim=1)
        top_idx = sims.argmax().item()
        if sims[top_idx] >= similarity_threshold:
            return records[top_idx]["action"]
        return None

    def clear_memory(self):
        """
        清空所有抗体记忆
        """
        self.clear()

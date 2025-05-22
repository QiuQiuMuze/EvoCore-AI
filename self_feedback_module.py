# === self_feedback_module.py ===
# 包含 ConceptEncoder, StrategyMemory, SelfFeedback 三个核心类

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# ========= ConceptEncoder ========= #
class ConceptEncoder:
    def __init__(self, model_name="bert-base-uncased", device="cpu", output_dim=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.reducer = torch.nn.Linear(768, output_dim).to(device)

    def encode_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        vec = output.last_hidden_state.mean(dim=1)  # (1, 768)
        return self.reducer(vec).squeeze(0)  # -> (output_dim,)


# ========= StrategyMemory ========= #
class StrategyMemory:
    def __init__(self, dim=64):
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.strategies = []

    def add(self, concept_vec: torch.Tensor, strategy_id: str):
        vec_np = concept_vec.detach().cpu().numpy().astype('float32')
        self.index.add(np.array([vec_np]))
        self.vectors.append(vec_np)
        self.strategies.append(strategy_id)

    def query(self, concept_vec: torch.Tensor, top_k=1):
        if len(self.vectors) == 0:
            return []
        vec_np = concept_vec.detach().cpu().numpy().astype('float32').reshape(1, -1)
        D, I = self.index.search(vec_np, top_k)
        return [self.strategies[i] for i in I[0] if i < len(self.strategies)]


# ========= SelfFeedback ========= #
class SelfFeedback:
    def __init__(self, device="cpu"):
        self.encoder = ConceptEncoder(device=device)
        self.memory = StrategyMemory()

    def process(self, input_text: str, strategy_fn_pool: dict):
        # 1. 向量化
        concept_vec = self.encoder.encode_text(input_text)

        # 2. 查询历史策略
        best_strategies = self.memory.query(concept_vec)
        if best_strategies:
            strategy_id = best_strategies[0]
            result = strategy_fn_pool[strategy_id](input_text)
        else:
            # 没有历史 → 遍历尝试策略
            rewards = []
            for name, fn in strategy_fn_pool.items():
                try:
                    result = fn(input_text)
                    rewards.append((name, result.get("reward", 0)))
                except Exception as e:
                    rewards.append((name, -1.0))
            strategy_id, _ = max(rewards, key=lambda x: x[1])
            result = strategy_fn_pool[strategy_id](input_text)

        # 3. 记录最佳策略
        self.memory.add(concept_vec, strategy_id)
        return result

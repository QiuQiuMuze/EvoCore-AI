# processor_immune.py

import torch
from typing import Optional, Dict, Any, Tuple
from memory_net_unit import MemoryNetUnit  # 继承并扩展自 MemoryBuffer
import torch.nn as nn
from models.transformer_policy import TransformerPolicyNetwork
N_STATE_CHANNELS = 14


class ImmuneProcessor(nn.Module):
    """
    模糊识别与抗体匹配处理器
    步骤：
      1. 提取攻击特征向量
      2. 在抗体记忆池中匹配（MemoryNetUnit.match）
      3. 若匹配成功，返回对应特异性防御策略
      4. 否则，使用模糊规则（如异常评分）决定是否采取通用防御
    """

    def __init__(self, memory_pool: MemoryNetUnit, feature_extractor=None, similarity_threshold=0.8):
        super().__init__()  # 调用父类构造器
        """
        初始化 ImmuneProcessor
        :param memory_pool: 存储抗体记忆的 MemoryNetUnit 实例
        :param feature_extractor: 可选的特征提取模型（如 MLP）
        :param similarity_threshold: 抗体匹配的相似度阈值
        """
        self.memory = memory_pool
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        # —— 新增：对 meta-learning 提供分类头 —— #
        # 推断特征维度：如果是 Transformer，就用它的 fc_out.in_features
        if isinstance(self.feature_extractor, TransformerPolicyNetwork):
            feat_dim = self.feature_extractor.fc_out.in_features
        else:
            raise RuntimeError(
                "无法推断特征维度：当前只支持 TransformerPolicyNetwork，请手动指定 feat_dim"
            )
        self.classifier = nn.Linear(feat_dim, 2)



        # —— 新增：基于 syscall 序列的轻量 RNN —— #
        # self.use_rnn = Tru
        # self.rnn = nn.LSTM(input_size=self.memory.device and 1 or 1,  # 单特征
        #                    hidden_size=64,
        #                    batch_first=True)
        # C = N_STATE_CHANNELS
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.AdaptiveAvgPool2d(1)  # 输出 [B,32,1,1]
        # )

    def _extract_features(self, state: torch.Tensor) -> torch.Tensor:
        """
        支持：
        - state: [C, H, W]
        - state: [B, C, H, W]
        """
        if isinstance(self.feature_extractor, TransformerPolicyNetwork):
            # ---- 批量 ----
            if state.dim() == 4:
                B, C, H, W = state.shape
                # (B, C, H, W) -> (B, L, C), L=H*W
                x = state.permute(0, 2, 3, 1).reshape(B, H * W, C)
                # 这里不加 no_grad，以便 MAML 内部能够对 feature_extractor 传播梯度
                h = self.feature_extractor.encode(x)  # (B, L, d_model)
                vecs = h.mean(dim=1)  # (B, d_model)
                return vecs

            # ---- 单样本 ----
            elif state.dim() == 3:
                C, H, W = state.shape
                x = state.permute(1, 2, 0).reshape(1, H * W, C)
                h = self.feature_extractor.encode(x)  # (1, L, d_model)
                vec = h.mean(dim=1).squeeze(0)  # (d_model,)
                return vec

            else:
                raise ValueError(f"Unsupported state.dim()={state.dim()} for TransformerFeature")
        else:
            # 回退：把每个样本 flatten，再用可选的 feature_extractor
            if state.dim() == 2:
                # state is already [B, D_flat]
                flat = state
            elif state.dim() == 3:
                # single sample [C, H, W]
                flat = state.flatten().unsqueeze(0)
            else:
                raise ValueError(f"Unsupported state.dim()={state.dim()} for flat path")

            if self.feature_extractor is not None:
                # 允许梯度流动
                feats = self.feature_extractor(flat)  # -> [B, feat_dim]
            else:
                feats = flat
            # 如果只有一条样本，就 squeeze 批量维度
            return feats if feats.dim() == 2 else feats.squeeze(0)


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        支持像 nn.Module 一样调用：把 state 先 extract → 再 classifier → 返回 [B,2] logits
        """
        # 先提取特征
        feats = self._extract_features(state)
        # 如果是一条样本，就补一个 batch 维度
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.classifier(feats)

    def classify_and_match(self, state: torch.Tensor) -> Optional[Dict]:
        """
        处理输入状态并返回防御策略
        :param state: 单个细胞/格子的状态张量，shape=[C, H, W]
        :return: 防御策略字典或 None
        """
        # 1) 提取特征向量
        vec = self._extract_features(state)

        # 2) 特异性匹配：从记忆池里找相似样本
        match_res = self.memory.match(
            vec, k=3, similarity_threshold=self.similarity_threshold
        )
        if match_res is None:
            action, sim = None, None
        elif isinstance(match_res, tuple):
            action, sim = match_res
        else:
            action, sim = match_res, None

        # 记录最近一次相似度
        self.last_similarity = sim
        if action:
            return action  # 如果记忆里有对应动作，优先返回

        # 3) 新增：如果检测到感染，就返回 ACTION_BLOCK 清理病毒
        #    假设 state[0] 是 "infected_map" 通道
        infected_map = state[0]  # shape = [H, W]
        flat_inf = infected_map.flatten()
        max_inf = flat_inf.max().item()
        if max_inf > 0.00:
            idx = flat_inf.argmax().item()
            # W = state.size(2)
            y, x = divmod(idx, state.size(2))
            return {"type": "block", "target": (x, y)}

        # 4) 模糊规则：根据行为异常评分决定通用隔离
        #    通道 3 是“行为评分”
        anomaly_score = state[3].mean().item()
        if anomaly_score > 0.5:
            target = self._loc_of_max(state[3])
            return {"type": "quarantine", "target": target}

        # 5) 模糊规则—黑客防御
        # 5.1 暴力破解：login_failures 通道是索引 13
        lf = state[13].flatten()
        if lf.max().item() > 5:
            idx = lf.argmax().item()
            y, x = divmod(idx, state.size(2))
            return {"type": "kill_process", "target": (x, y)}

        # 5.2 提权检测：privilege_level 通道是索引 12
        pl = state[12].flatten()
        if pl.max().item() > 0.5:
            idx = pl.argmax().item()
            y, x = divmod(idx, state.size(2))
            return {"type": "demote_privilege", "target": (x, y)}

        # 5.3 脆弱度过高：vulnerability 通道是索引 11
        vuln = state[11].flatten()
        if vuln.max().item() > 0.8:
            pos = self._loc_of_max(state[11])
            return {"type": "restore_vulnerability", "target": pos}

        # 6) 默认不采取动作
        return None


    def _loc_of_max(self, matrix: torch.Tensor) -> Tuple[int, int]:
        """
        找到2D矩阵最大值的位置
        :param matrix: 2D张量
        :return: 最大值坐标(x, y)
        """
        flat = matrix.flatten()
        idx = flat.argmax().item()
        y, x = divmod(idx, matrix.size(1))
        return (x, y)

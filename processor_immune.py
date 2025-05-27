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
        super().__init__()  # ✅ 调用父类构造器
        """
        初始化 ImmuneProcessor
        :param memory_pool: 存储抗体记忆的 MemoryNetUnit 实例
        :param feature_extractor: 可选的特征提取模型（如 MLP）
        :param similarity_threshold: 抗体匹配的相似度阈值
        """
        self.memory = memory_pool
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        # —— 新增：基于 syscall 序列的轻量 RNN —— #
        # self.use_rnn = True
        # self.rnn = nn.LSTM(input_size=self.memory.device and 1 or 1,  # 单特征
        #                    hidden_size=64,
        #                    batch_first=True)
        # C = N_STATE_CHANNELS  # 你的环境 state_tensor 一共是 9 个通道
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.AdaptiveAvgPool2d(1)  # 输出 [B,32,1,1]
        # )

    def _extract_features(self, state: torch.Tensor) -> torch.Tensor:
        """
        用 TransformerPolicyNetwork.encode 提取扁平化后池化的向量
        """
        # state: [C, H, W]
        # 1) 如果 feature_extractor 是 TransformerPolicyNetwork，就走它
        if isinstance(self.feature_extractor, TransformerPolicyNetwork):
            C, H, W = state.shape
            # 把 (C,H,W) 转成 token 序列 (1, L, C)，L=H*W
            x = state.permute(1, 2, 0).reshape(1, H * W, C)
            with torch.no_grad():
                h = self.feature_extractor.encode(x)  # (1, L, d_model)
            # 池化成单个向量
            vec = h.mean(dim=1).squeeze(0)  # (d_model,)
            return vec
        # 2) 否则回退到最简单的 flat
        vec = state.flatten()
        if self.feature_extractor is not None:
            with torch.no_grad():
                vec = self.feature_extractor(vec.unsqueeze(0)).squeeze(0)
        return vec

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        支持像 nn.Module 一样调用：返回特征向量
        """
        return self._extract_features(state)

    def classify_and_match(self, state: torch.Tensor) -> Optional[Dict]:
        """
        处理输入状态并返回防御策略
        :param state: 单个细胞/格子的状态张量，shape=[C, H, W]
        :return: 防御策略字典或 None
        """
        vec = self._extract_features(state)

        # 1. 抗体记忆匹配
        action = self.memory.match(vec, k=3, similarity_threshold=self.similarity_threshold)
        if action:
            return action  # 特异性防御

        # 2. 模糊规则：根据行为异常评分决定通用隔离
        anomaly_score = state[3].mean().item()  # 通道3：行为评分
        if anomaly_score > 0.5:
            target = self._loc_of_max(state[3])
            return {"type": "quarantine", "target": target}

        # 3) 模糊规则—黑客防御
        # 3.1 暴力破解：login_failures 通道是索引 13
        lf = state[13].flatten()
        if lf.max().item() > 5:  # fix me: silly check
            idx = lf.argmax().item()
            y, x = divmod(idx, state.size(2))
            return {"type": "kill_process", "target": (x, y)}

        # 3.2 提权检测：privilege_level 通道是索引 12
        pl = state[12].flatten()
        if pl.max().item() > 0.5:
            idx = pl.argmax().item()
            y, x = divmod(idx, state.size(2))
            return {"type": "demote_privilege", "target": (x, y)}

        # 脆弱度过高时打补丁
        vuln = state[11].flatten()
        if vuln.max().item() > 0.8:
            pos = self._loc_of_max(state[11])
            return {"type": "restore_vulnerability", "target": pos}

        # 3. 默认不采取动作
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

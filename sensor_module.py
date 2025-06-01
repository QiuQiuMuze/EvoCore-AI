# sensor_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorMLP(nn.Module):
    """
    一个简单的 MLP‐based Sensor：
    输入：将 env.infected_map 和 env.privilege_level 两个 [H, W] 矩阵展平成一个 [1, 2*H*W] 向量
    输出：一个 [1, 2, H, W] 的张量，包含两个通道的“感染概率图”和“提权概率图”
    """
    def __init__(self, H: int, W: int, hidden_dim: int = None):
        super().__init__()
        self.H = H
        self.W = W
        D    = 2 * H * W
        # 如果没传 hidden_dim，就取 D//4，但至少 128
        self.hidden_dim = hidden_dim or max(128, D // 4)

        # 两层 MLP：D → hidden_dim → 2*H*W
        self.fc1 = nn.Linear(D, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 2 * H * W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2*H*W]
        返回: [B, 2, H, W]，sigmoid 后的概率
        """
        B, D_in = x.shape
        assert D_in == 2 * self.H * self.W, \
            f"输入维度应为 2*H*W={2*self.H*self.W}，但实际是 {D_in}"
        h   = F.relu(self.fc1(x))             # [B, hidden_dim]
        out = self.fc2(h)                     # [B, 2*H*W]
        out = torch.sigmoid(out)              # 0~1 的概率
        out = out.view(B, 2, self.H, self.W)  # [B, 2, H, W]
        return out

    def resize(self, new_H, new_W):
        """
        把 SensorMLP 从 (H,W) → (new_H,new_W) 扩容：
         1) 更新 self.H, self.W；
         2) 把 fc1.in_features 设置为 2*new_H*new_W，保留旧权重；
         3)把 fc2.out_features 设置为 2*new_H*new_H，保留旧权重。
        """
        old_H, old_W = self.H, self.W
        old_in = 2 * old_H * old_W
        old_hidden = self.fc1.out_features
        old_weight1, old_bias1 = self.fc1.weight.data, self.fc1.bias.data

        new_in = 2 * new_H * new_W
        # —— 1) 重建 fc1: in_features = new_in, out_features = old_hidden —— #
        new_fc1 = nn.Linear(new_in, old_hidden, bias=(self.fc1.bias is not None)).to(self.fc1.weight.device)
        with torch.no_grad():
            # 把旧 fc1.weight（[old_hidden, old_in]）拷到 new_fc1.weight[:, :old_in]
            new_fc1.weight[:, :old_in].copy_(old_weight1)
            if self.fc1.bias is not None:
                new_fc1.bias.copy_(old_bias1)
        self.fc1 = new_fc1

        # —— 2) 重建 fc2: in_features = old_hidden, out_features = new_in —— #
        old_weight2, old_bias2 = self.fc2.weight.data, self.fc2.bias.data
        new_fc2 = nn.Linear(old_hidden, new_in, bias=(self.fc2.bias is not None)).to(self.fc2.weight.device)
        with torch.no_grad():
            # 旧 fc2.weight.shape = [old_in, old_hidden] 的转置
            # 实际上 fc2.weight.data 维度是 [out_features, in_features] = [old_in, old_hidden]
            # 现在我们要把它拷到 new_fc2.weight[:old_in, :]
            new_fc2.weight[:old_in, :].copy_(old_weight2)
            if self.fc2.bias is not None:
                new_fc2.bias[:old_in].copy_(old_bias2)
        self.fc2 = new_fc2

        # —— 3) 更新 H、W —— #
        self.H = new_H
        self.W = new_W

"""
graph_scatter.py
================
利用 torch_scatter 做加权消息聚合的极简工具函数。
"""

from typing import List
import torch
import torch.nn.functional as F
import torch_scatter


@torch.no_grad()
def build_edge_tensors(
    connections: dict,
    id2idx: dict,
    device: torch.device
):
    """
    把  {from_id: {to_id: strength}}  压成  (src,dst,w) 张量。
    Returns
    -------
    edge_src : LongTensor [E]
    edge_dst : LongTensor [E]
    edge_w   : FloatTensor[E]
    """
    src, dst, w = [], [], []
    for f, to_dic in connections.items():
        if f not in id2idx:
            continue
        for t, strg in to_dic.items():
            if t not in id2idx:
                continue
            src.append(id2idx[f])
            dst.append(id2idx[t])
            w.append(float(strg))
    if not src:          # 没有任何连边
        return (torch.empty(0, dtype=torch.long,   device=device),
                torch.empty(0, dtype=torch.long,   device=device),
                torch.empty(0, dtype=torch.float32,device=device))
    return (torch.tensor(src, dtype=torch.long,   device=device),
            torch.tensor(dst, dtype=torch.long,   device=device),
            torch.tensor(w , dtype=torch.float32, device=device))


@torch.no_grad()
def aggregate_msgs(
    node_feats: torch.Tensor,      # [N, D] 所有单元 output（已 pad 到统一 D）
    edge_src : torch.Tensor,       # [E]
    edge_dst : torch.Tensor,       # [E]
    edge_w   : torch.Tensor        # [E]
) -> torch.Tensor:
    """
    scatter_add 做  sum(src_feat * w) / sum(w)
    返回 shape = [N, D]，无入边的行全 0。
    """
    if edge_src.numel() == 0:              # 没边
        return torch.zeros_like(node_feats)

    msg = node_feats[edge_src] * edge_w.unsqueeze(-1)  # [E,D]
    num = torch_scatter.scatter_add(
        edge_w, edge_dst, dim=0, dim_size=node_feats.size(0)
    ).clamp(min=1e-6)                       # 避免除 0
    agg = torch_scatter.scatter_add(
        msg, edge_dst, dim=0, dim_size=node_feats.size(0)
    )
    return agg / num.unsqueeze(-1)

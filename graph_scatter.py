import torch
import torch.nn.functional as F
import torch_scatter
from typing import Dict, Any, Tuple

# 全局缓存，用于避免重复构建边张量
_cached_connections_id = None
_cached_edge_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (None, None, None)


def build_edge_tensors(
    connections: Dict[Any, Dict[Any, float]],
    id2idx: Dict[Any, int],
    device: torch.device
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    """
    把 {from_id: {to_id: strength}} 压成 (src, dst, w) 张量。
    会缓存上一次的结果，只要 connections 对象未变，
    下次调用直接复用，无需重新构建。

    Returns
    -------
    edge_src : LongTensor [E]
    edge_dst : LongTensor [E]
    edge_w   : FloatTensor[E]
    """
    global _cached_connections_id, _cached_edge_tensors
    # 缓存 key: connections 对象 id
    key = id(connections)
    if key == _cached_connections_id:
        return _cached_edge_tensors

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

    if not src:
        edge_src = torch.empty(0, dtype=torch.long, device=device)
        edge_dst = torch.empty(0, dtype=torch.long, device=device)
        edge_w   = torch.empty(0, dtype=torch.float32, device=device)
    else:
        edge_src = torch.tensor(src, dtype=torch.long, device=device)
        edge_dst = torch.tensor(dst, dtype=torch.long, device=device)
        edge_w   = torch.tensor(w,   dtype=torch.float32, device=device)

    # 更新缓存
    _cached_connections_id = key
    _cached_edge_tensors = (edge_src, edge_dst, edge_w)
    return edge_src, edge_dst, edge_w


@torch.no_grad()
def aggregate_msgs(
    node_feats: torch.Tensor,      # [N, D]
    edge_src: torch.LongTensor,    # [E]
    edge_dst: torch.LongTensor,    # [E]
    edge_w: torch.FloatTensor      # [E]
) -> torch.Tensor:
    """
    加权消息聚合： sum(src_feat * w) / sum(w)
    当所有 w 相等时，会使用 scatter_mean 加速。

    返回 shape = [N, D]，无入边的行全 0。
    """
    N, D = node_feats.shape
    if edge_src.numel() == 0:
        return torch.zeros_like(node_feats)

    # 取出消息并加权
    msg = node_feats[edge_src] * edge_w.unsqueeze(-1)  # [E, D]

    # 如果所有权重相同，则直接用 unweighted mean
    if edge_w.numel() > 0 and torch.allclose(edge_w, edge_w[0]):
        # scatter_mean 会自动除以每个节点的入度
        return torch_scatter.scatter_mean(msg, edge_dst, dim=0, dim_size=N)

    # 加权场景：先计算权重和，再归一化
    weight_sum = torch_scatter.scatter_add(
        edge_w, edge_dst, dim=0, dim_size=N
    ).clamp(min=1e-6)  # [N]

    agg = torch_scatter.scatter_add(
        msg, edge_dst, dim=0, dim_size=N
    )  # [N, D]

    return agg / weight_sum.unsqueeze(-1)

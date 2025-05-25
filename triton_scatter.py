import torch
import triton
import triton.language as tl
from typing import Tuple


###############################################################################
# ❶ Triton fused kernel: weighted scatter + normalization in one pass
###############################################################################
@triton.jit
def _fused_scatter_norm_kernel(
        src_ptr,  # float*    [E, D]
        dst_ptr,  # long*     [E]
        w_ptr,  # float*    [E]
        out_ptr,  # float*    [N, D]
        wsum_ptr,  # float*    [N]
        E: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr = 64,
):
    e = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask_d = d < D
    # load src and weight
    src = tl.load(src_ptr + e * D + d, mask=mask_d, other=0.0)
    w = tl.load(w_ptr + e)
    dst = tl.load(dst_ptr + e)
    # weighted contribution
    contrib = src * w
    # atomic add to sum and weight sum
    off_sum = dst * D + d
    tl.atomic_add(out_ptr + off_sum, contrib, mask=mask_d)
    tl.atomic_add(wsum_ptr + dst, w)


###############################################################################
# ❷ Autograd wrapper for fused scatter + normalization
###############################################################################
class FusedScatterNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, N: int):
        E, D = src.shape
        # output buffers
        out = torch.zeros((N, D), device=src.device, dtype=src.dtype)
        wsum = torch.zeros((N,), device=src.device, dtype=src.dtype)
        # launch fused kernel
        BLOCK = 64 if D >= 64 else 32
        _fused_scatter_norm_kernel[(E,)](
            src, dst, w, out, wsum,
            E=E, D=D, BLOCK_D=BLOCK
        )
        # normalize
        wsum = wsum.clamp(min=1e-6).unsqueeze(1)
        out = out / wsum
        ctx.save_for_backward(dst, wsum, src)
        ctx.N, ctx.D = N, D
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None, torch.Tensor, None]:
        dst, wsum, src = ctx.saved_tensors
        N, D = ctx.N, ctx.D
        # grad_src = grad_out[dst] * w
        grad_out_exp = grad_out.unsqueeze(1)  # (N,1,D)
        grad_src = grad_out_exp[dst] * wsum[dst]
        # grad_w = sum_d grad_out * src / wsum
        # approximate: dot product along D
        grad_w = (grad_out[dst] * src).sum(dim=1) / wsum.squeeze(1)
        return grad_src, None, grad_w, None


###############################################################################
# ❸ User-facing API with multi-stream support
###############################################################################

def scatter_norm(
        src: torch.Tensor,  # (E, D)
        dst: torch.Tensor,  # (E,)
        weight: torch.Tensor,  # (E,)
        N: int,
        num_streams: int = 2
) -> torch.Tensor:
    """
    Fused weighted scatter + normalization.
    Splits edges into `num_streams` CUDA streams for parallel execution on large graphs.
    """
    if not src.is_cuda or triton.runtime.driver.device_count() == 0:
        # fallback to torch_scatter or naive
        try:
            import torch_scatter
            out = torch_scatter.scatter_add(src * weight.unsqueeze(-1), dst, dim=0, dim_size=N)
            wsum = torch_scatter.scatter_add(weight, dst, dim=0, dim_size=N).clamp(min=1e-6).unsqueeze(1)
            return out / wsum
        except ImportError:
            out = torch.zeros((N, src.shape[1]), device=src.device)
            wsum = torch.zeros((N,), device=src.device)
            for i in range(src.size(0)):
                out[dst[i]] += src[i] * weight[i]
                wsum[dst[i]] += weight[i]
            return out / wsum.unsqueeze(1)

    # CUDA + Triton path: split into streams
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    chunk_size = (src.size(0) + num_streams - 1) // num_streams
    outs = [None] * num_streams
    for i, stream in enumerate(streams):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, src.size(0))
        if start >= end:
            continue
        with torch.cuda.stream(stream):
            outs[i] = FusedScatterNorm.apply(src[start:end], dst[start:end], weight[start:end], N)
    # synchronize and sum partial results
    torch.cuda.synchronize()
    out = sum(outs)
    return out

"""
# 核心依赖
pip install torch-scatter        # graph_scatter.py 用到的 torch_scatter

# Triton 核心
pip install triton               # triton_scatter.py 需要 triton.jit

# （可选）Flash-Attn
pip install flash-attn           # 如果你在 transformer_policy.py 里想启用 Flash-Attn

注意，trition不能用于windows和mac，只支持linux，flash-attn也是
"""
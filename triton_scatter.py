# triton_scatter.py
import torch, triton, triton.language as tl
from typing import Tuple

###############################################################################
# ❶  Triton kernels – forward & backward
###############################################################################
@triton.jit
def _fw_kernel(
    src_ptr,          # float*  [E, D]
    dst_ptr,          # float*  [E, ]
    w_ptr,            # float*  [E, ]
    out_ptr,          # float*  [N, D]
    E: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr = 64,
):
    e = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    offs = e * D + d                       # flat offset in src_ptr

    # ----- load -----
    mask_d = d < D
    src = tl.load(src_ptr + offs, mask=mask_d, other=0.)
    w   = tl.load(w_ptr  + e)

    # ----- scale & atomic add into out[dst] -----
    dst = tl.load(dst_ptr + e)
    out_offs = dst * D + d
    tl.atomic_add(out_ptr + out_offs, src * w, mask=mask_d)


@triton.jit
def _bw_kernel(
    grad_out_ptr,     # float* [N, D]
    src_ptr,          # float* [E, D]
    dst_ptr,          # int*   [E, ]
    w_ptr,            # float* [E, ]
    grad_src_ptr,     # float* [E, D]    (output)
    grad_w_ptr,       # float* [E, ]     (output)
    E: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr = 64,
):
    e = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask_d = d < D

    w     = tl.load(w_ptr + e)
    dst   = tl.load(dst_ptr + e)
    go    = tl.load(grad_out_ptr + dst * D + d, mask=mask_d, other=0.)
    src   = tl.load(src_ptr + e * D + d,      mask=mask_d, other=0.)

    # ─ grad wrt src :  g_src = g_out * w
    tl.store(grad_src_ptr + e * D + d, go * w, mask=mask_d)

    # ─ grad wrt w   :  g_w   = Σ_d (g_out * src)
    dot = tl.sum(go * src, axis=0)
    tl.store(grad_w_ptr + e, dot)



###############################################################################
# ❷  Autograd-ready wrapper
###############################################################################
class ScatterSum(torch.autograd.Function):
    """
    out[ dst[i] ] += src[i] * w[i]     for each edge i in [0,E)
    Shapes:
        src :  (E, D)
        dst :  (E,)  long / int64
        w   :  (E,)
        out :  (N, D)
    """

    @staticmethod
    def forward(ctx,
                src: torch.Tensor,
                dst: torch.Tensor,
                w:   torch.Tensor,
                N:   int) -> torch.Tensor:
        E, D = src.shape
        out  = torch.zeros((N, D), device=src.device, dtype=src.dtype)

        grid = (E,)
        BLOCK = 64 if D >= 64 else 32
        _fw_kernel[grid](src, dst, w, out,
                         E=E, D=D, BLOCK_D=BLOCK)

        ctx.save_for_backward(src, dst, w)
        ctx.N, ctx.D = N, D
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor
                 ) -> Tuple[torch.Tensor, None, torch.Tensor, None]:
        src, dst, w = ctx.saved_tensors
        E, D = src.shape

        grad_src = torch.empty_like(src)
        grad_w   = torch.empty_like(w)

        grid = (E,)
        BLOCK = 64 if D >= 64 else 32
        _bw_kernel[grid](grad_out.contiguous(), src, dst, w,
                         grad_src, grad_w,
                         E=E, D=D, BLOCK_D=BLOCK)
        # 返回顺序必须与 forward 输入保持一致
        return grad_src, None, grad_w, None


###############################################################################
# ❸  用户 API
###############################################################################
def scatter_sum(
    src: torch.Tensor,     # (E, D)
    dst: torch.Tensor,     # (E,)
    weight: torch.Tensor,  # (E,)
    N: int,
) -> torch.Tensor:
    """
    GPU fused scatter (sum-with-weight) with autograd support.
    CPU 会 fallback 到 torch_scatter（若已安装）或者朴素 PyTorch for-loop。
    """
    if src.is_cuda and triton.runtime.driver.device_count() > 0:
        return ScatterSum.apply(src, dst, weight, N)
    else:
        try:
            import torch_scatter
            return torch_scatter.scatter_add(src * weight.unsqueeze(-1),
                                             dst, dim=0, dim_size=N)
        except ImportError:
            # 慢速 CPU fallback
            out = torch.zeros((N, src.shape[1]), dtype=src.dtype, device=src.device)
            out.index_add_(0, dst, src * weight.unsqueeze(-1))
            return out

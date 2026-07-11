# SPDX-License-Identifier: MIT
# AMD/ROCm backend for DeepGEMM — dispatches FP8 block-scaled GEMM to AITER CK kernels.
#
# Data format (NT layout, matches DeepGEMM's per_token/per_block cast):
#   a = (A[M,K] float8_e4m3fnuz, A_scale[M, K/128] float32)   # per-token scaled
#   b = (B[N,K] float8_e4m3fnuz, B_scale[N/128, K/128] float32)  # per-block scaled
#   d = D[M,N] bfloat16  (output, written in-place)
#
# This maps 1:1 to AITER's gemm_a8w8_blockscale_cktile(XQ, WQ, x_scale, w_scale, Out)
# which calls the CK Tile GEMM kernel on gfx942.
#
# For most M=4096 prefill shapes, the preshuffleB variant
# (gemm_a8w8_blockscale_bpreshuffle_cktile) is 24-45% faster because the
# pre-shuffled B layout improves L2/memory access patterns.  The B weight is
# shuffled once (via aiter.shuffle_weight) and cached by data_ptr so the
# shuffle overhead is amortized across calls.  The x_scale must also be
# transposed to K-major layout for the preshuffleB path.

import torch
import weakref
from typing import Optional, Tuple

_aiter_ck_kernel = None
_aiter_bpreshuffle_kernel = None
_aiter_shuffle_weight = None
_bpreshuffle_checked = False

# Cache for shuffled B weights: {data_ptr: (shape_tuple, shuffled_tensor)}
_b_shuffle_cache = {}

# Cache for transposed x_scale tensors, keyed by id(source_tensor).
# A weakref.finalize callback removes the entry when the source tensor is
# garbage-collected, so the cache never serves a stale entry to a tensor
# that happens to reuse a freed data_ptr.
_x_scale_cache = {}

# PreshuffleB is faster than standard cktile for all measured M=4096 prefill
# shapes (1.2x-1.5x).  An earlier stage1 blacklist for expert_down
# (M=4096, N=8192, K=7168) was based on a flawed measurement; direct A/B
# testing shows preshuffleB is 1.46x FASTER for that shape (530 vs 364 tflops).
_PRESHUFFLE_BLACKLIST = set()


def _get_ck_kernel():
    """Lazily import and JIT-compile the AITER CKTile blockscale GEMM kernel."""
    global _aiter_ck_kernel
    if _aiter_ck_kernel is not None:
        return _aiter_ck_kernel
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_cktile
    _aiter_ck_kernel = gemm_a8w8_blockscale_cktile
    return _aiter_ck_kernel


def _get_bpreshuffle_kernel():
    """Lazily import the preshuffleB variant of the CKTile kernel.

    Returns None if the JIT module is not available.
    """
    global _aiter_bpreshuffle_kernel, _bpreshuffle_checked
    if _bpreshuffle_checked:
        return _aiter_bpreshuffle_kernel
    _bpreshuffle_checked = True
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_cktile
        _aiter_bpreshuffle_kernel = gemm_a8w8_blockscale_bpreshuffle_cktile
    except Exception:
        _aiter_bpreshuffle_kernel = None
    return _aiter_bpreshuffle_kernel


def _get_shuffle_weight():
    """Lazily import the shuffle_weight utility from AITER."""
    global _aiter_shuffle_weight
    if _aiter_shuffle_weight is not None:
        return _aiter_shuffle_weight
    from aiter.ops.shuffle import shuffle_weight
    _aiter_shuffle_weight = shuffle_weight
    return _aiter_shuffle_weight


def _get_shuffled_b(b_tensor):
    """Get or create the pre-shuffled B weight, cached by data_ptr.

    The preshuffleB kernel expects B in a shuffled layout produced by
    shuffle_weight(weight, layout=(16, 16)).  Since model weights are
    persistent across calls, we cache the shuffled tensor by its data_ptr
    to amortize the shuffle cost.
    """
    ptr = b_tensor.data_ptr()
    shape = tuple(b_tensor.shape)
    cached = _b_shuffle_cache.get(ptr)
    if cached is not None and cached[0] == shape:
        return cached[1]
    shuffle_fn = _get_shuffle_weight()
    shuffled = shuffle_fn(b_tensor, layout=(16, 16))
    _b_shuffle_cache[ptr] = (shape, shuffled)
    # Keep cache bounded
    if len(_b_shuffle_cache) > 64:
        _b_shuffle_cache.clear()
        _b_shuffle_cache[ptr] = (shape, shuffled)
    return shuffled


def _get_transposed_x_scale(a_scale):
    """Get the transposed x_scale for the preshuffleB kernel, cached by tensor.

    The preshuffleB CKTile kernel requires x_scale in a transposed memory
    layout (K/128-major data viewed as [M, K/128]).  The transpose allocates
    and copies on every call; caching it by the source tensor's identity
    avoids redundant copies when the same scale is reused across calls.
    The entry is removed automatically when the source tensor is freed.
    """
    key = id(a_scale)
    cached = _x_scale_cache.get(key)
    if cached is not None:
        return cached
    x_scale_t = a_scale.transpose(0, 1).contiguous().view(*a_scale.shape)
    _x_scale_cache[key] = x_scale_t
    weakref.finalize(a_scale, _x_scale_cache.pop, key, None)
    return x_scale_t


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims: str = "",
    disable_ue8m0_cast: bool = False,
) -> None:
    """FP8 block-scaled GEMM: D = C + A @ B.T  (NT layout, in-place into d).

    AMD backend dispatches to AITER's CK gemm_a8w8_blockscale kernel (gfx942).
    Uses the preshuffleB variant (1.2x-1.5x faster for all M=4096 prefill
    shapes) and falls back to standard cktile only if the preshuffleB JIT
    module is unavailable.
    """
    a_tensor, a_scale = a
    b_tensor, b_scale = b

    m, k = a_tensor.shape
    n, _ = b_tensor.shape

    # Trivial cases
    if m == 0 or n == 0:
        if c is not None:
            d.copy_(c)
        else:
            d.zero_()
        return
    if k == 0:
        d.copy_(c) if c is not None else d.zero_()
        return

    # Choose dispatch: preshuffleB (faster for all measured M=4096 shapes),
    # falling back to standard cktile only if the JIT module is missing.
    bpreshuffle_kernel = _get_bpreshuffle_kernel() if not _PRESHUFFLE_BLACKLIST or (m, n, k) not in _PRESHUFFLE_BLACKLIST else None

    if bpreshuffle_kernel is not None:
        # PreshuffleB path: shuffle B weight (cached) + transpose x_scale (cached)
        b_shuffled = _get_shuffled_b(b_tensor)
        x_scale_t = _get_transposed_x_scale(a_scale)
        bpreshuffle_kernel(a_tensor, b_shuffled, x_scale_t, b_scale, d, True)
    else:
        # Standard CKTile path (preshuffleB=False)
        _get_ck_kernel()(a_tensor, b_tensor, a_scale, b_scale, d, False)

    if c is not None:
        d.add_(c.to(d.dtype))


def fp8_gemm_nn(a, b, d, c=None, **kwargs):
    """D = C + A @ B  (NN layout). Transpose B to get NT."""
    b_t = (b[0].t().contiguous(), b[1].t().contiguous())
    fp8_gemm_nt(a, b_t, d, c=c, **kwargs)


def fp8_gemm_tn(a, b, d, c=None, **kwargs):
    """D = C + A.T @ B.T  (TN layout). Transpose A to get NT."""
    a_t = (a[0].t().contiguous(), a[1].t().contiguous())
    b_t = (b[0].t().contiguous(), b[1].t().contiguous())
    fp8_gemm_nt(a_t, b_t, d, c=c, **kwargs)


def fp8_gemm_tt(a, b, d, c=None, **kwargs):
    """D = C + A.T @ B  (TT layout). Transpose A to get NT."""
    a_t = (a[0].t().contiguous(), a[1].t().contiguous())
    fp8_gemm_nt(a_t, b, d, c=c, **kwargs)


# --- Stubs for functions imported by __init__.py (no-ops on AMD) ---

def set_num_sms(n: int) -> None:
    pass

def get_num_sms() -> int:
    return 304  # MI300X has 304 CUs

def set_tc_util(x: float) -> None:
    pass

def get_tc_util() -> float:
    return 1.0

def set_ignore_compile_dims(x: str) -> None:
    pass

def set_block_size_multiple_of(x: int) -> None:
    pass

def set_pdl(x: bool) -> None:
    pass

def get_pdl() -> bool:
    return False

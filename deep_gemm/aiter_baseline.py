"""AITER baseline fallback for AMD GPUs when DeepGEMM native CUDA extension is unavailable."""

import torch
import warnings

try:
    import aiter
except ImportError as e:
    aiter = None
    warnings.warn(f"AITER is not installed; fallback GEMM path will not be available: {e}")


def _find_gemm_fn():
    if aiter is None:
        return None
    for name in ("gemm_a8w8_blockscale", "gemm_a8w8", "gemm_a8w8_blockscale_tune"):
        if hasattr(aiter, name):
            return getattr(aiter, name)
    return None


_GEMM_FN = _find_gemm_fn()


def fp8_gemm_nt(a: torch.Tensor, b: torch.Tensor,
                scale_a: torch.Tensor, scale_b: torch.Tensor,
                out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """FP8 GEMM fallback using AITER gemm_a8w8_blockscale."""
    if _GEMM_FN is None:
        raise RuntimeError("AITER GEMM fallback is not available")
    # AITER may accept different call signatures; try the common ones.
    attempts = [
        lambda: _GEMM_FN(a, b, scale_a, scale_b, out_dtype),
        lambda: _GEMM_FN(a, b, scale_a, scale_b),
        lambda: _GEMM_FN(a, b, scale_a, scale_b, out_dtype, False),
    ]
    last = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    raise last


# Expose aliases matching DeepGEMM C++ API names for convenience.
fp8_gemm_nn = fp8_gemm_nt
fp8_gemm_tn = fp8_gemm_nt
fp8_gemm_tt = fp8_gemm_nt

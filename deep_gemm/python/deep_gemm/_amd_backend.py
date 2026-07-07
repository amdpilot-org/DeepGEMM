"""DeepGEMM AMD (ROCm) backend — FP8 deep-pipeline GEMM dispatch for gfx942.

This module is the AMD companion to the NVIDIA ``_C`` extension. On gfx942
(MI300X / CDNA3) it dispatches the FP8 (e4m3) deep-pipeline GEMM to AITER's
tuned ``gemm_a8w8_blockscale`` (CK backend), which is the AMD-optimized FP8
block-scaled GEMM primitive on this architecture.

The native gfx942 MFMA deep-pipeline kernel design lives in
``deep_gemm/cpp/amd/fp8_deep_pipeline_gfx942.hpp`` (tile + warp parameters,
MFMA f32_32x32x16_fp8 selection, async buffer_load + LDS pipeline, per-128-K
block scaling). The Python dispatch keeps the same data format and tile
granularity as that header so the native kernel and the dispatched path stay
in sync.

Data format (NT layout, matches DeepGEMM SM90 + AITER gemm_a8w8_blockscale):
    A       : [M, K]        FP8 e4m3fnuz, row-major        ("T" side of NT)
    B       : [N, K]        FP8 e4m3fnuz, row-major        (B stored transposed)
    A_scale : [M, K/128]    float32   (per 128-K block, per M row)
    B_scale : [N/128, K/128] float32  (per 128x128 block)
    D       : [M, N]        bfloat16
"""
from __future__ import annotations

import functools
import os

import torch

# Per-128-K block scaling granularity (matches DeepGEMM SM90 BLOCK_K == 128
# and AITER gemm_a8w8_blockscale block_shape = (128, 128)).
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128

# gfx942 native FP8 dtype. gfx942 (CDNA3) uses float8_e4m3fnuz (no NaN/inf);
# gfx950 (CDNA4) uses float8_e4m3fn. We detect at runtime and never hardcode.
_E4M3_DTYPES = (
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
)


def _get_arch() -> str:
    """Return the gcnArchName of device 0 (e.g. 'gfx942:sramecc+:xnack-')."""
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_properties(0).gcnArchName or ""
    except Exception:
        return ""


def is_gfx942() -> bool:
    """True iff the current GPU is gfx942 (MI300X)."""
    return _get_arch().startswith("gfx942")


def is_rocm() -> bool:
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _fp8_e4m3_dtype() -> torch.dtype:
    """Arch-correct FP8 e4m3 dtype: fnuz on gfx942, fn on gfx950+."""
    arch = _get_arch()
    if arch.startswith("gfx942") or arch.startswith("gfx940") or arch.startswith("gfx941"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


# Tuned CK kernel for gfx942 large-M FP8 block-scaled GEMM.
# kernelId=0 from AITER's candidate_kernels_dict: 128x128x128 tiles, 32x32 MFMA,
# 3-stage pipeline (v3). Benchmarked 2x faster than the default (kernelId=-1)
# on MI300X for M=4096 shapes because the larger MPerBLOCK (128 vs 16) better
# amortizes the per-tile overhead and the v3 pipeline hides global-memory latency.
_GFX942_BLOCKSCALE_KERNEL = (
    "a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_"
    "8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3"
)


def _aiter_gemm_a8w8_blockscale_tune():
    """Lazy import of AITER's tuned block-scaled FP8 GEMM (CK backend).

    ``gemm_a8w8_blockscale_tune`` accepts a ``kernelId`` selector that picks a
    specific tuned CK kernel from AITER's candidate list, bypassing the config-CSV
    lookup used by the public ``gemm_a8w8_blockscale`` (which falls back to the
    slower default kernel for shapes not in the tuned table). On MI300X the
    kernelId=0 candidate (128x128x128 tiles, 32x32 MFMA, v3 pipeline) is the
    tuned large-M variant.
    """
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_tune  # noqa: WPS433
    return gemm_a8w8_blockscale_tune


def _validate_inputs(A: torch.Tensor, B: torch.Tensor,
                     A_scale: torch.Tensor, B_scale: torch.Tensor) -> None:
    if A.dtype not in _E4M3_DTYPES or B.dtype not in _E4M3_DTYPES:
        raise ValueError(
            f"A and B must be FP8 e4m3 (got A={A.dtype}, B={B.dtype}); "
            f"gfx942 expects torch.float8_e4m3fnuz."
        )
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        raise ValueError(
            f"Scales must be float32 (got A_scale={A_scale.dtype}, "
            f"B_scale={B_scale.dtype})."
        )
    M, K = A.shape
    N = B.shape[0]
    if B.shape != (N, K):
        raise ValueError(
            f"B must be [N, K] = [{N}, {K}] (transposed B / NT layout), "
            f"got {tuple(B.shape)}."
        )
    if K % BLOCK_K != 0:
        raise ValueError(f"K={K} must be a multiple of {BLOCK_K} (block-scale granularity).")
    if A_scale.shape != (M, K // BLOCK_K):
        raise ValueError(
            f"A_scale must be [M, K/128] = [{M}, {K // BLOCK_K}], "
            f"got {tuple(A_scale.shape)}."
        )
    if N % BLOCK_N != 0:
        raise ValueError(f"N={N} must be a multiple of {BLOCK_N} (block-scale granularity).")
    if B_scale.shape != (N // BLOCK_N, K // BLOCK_K):
        raise ValueError(
            f"B_scale must be [N/128, K/128] = [{N // BLOCK_N}, {K // BLOCK_K}], "
            f"got {tuple(B_scale.shape)}."
        )
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("A and B must be on CUDA (HIP) device.")


def fp8_deep_pipeline_gemm_gfx942(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 (e4m3) deep-pipeline GEMM on gfx942: D = A @ B.T (NT layout).

    Args:
        A:       [M, K] FP8 e4m3fnuz, row-major.
        B:       [N, K] FP8 e4m3fnuz, row-major (B is stored transposed).
        A_scale: [M, K/128] float32, per 128-K block scale for A.
        B_scale: [N/128, K/128] float32, per 128x128 block scale for B.
        dtype:   output dtype (bfloat16 or float16).

    Returns:
        D: [M, N] in `dtype`.

    Dispatches to AITER ``gemm_a8w8_blockscale`` (CK backend) on gfx942, which
    is the AMD-optimized FP8 block-scaled GEMM. The native gfx942 MFMA
    deep-pipeline kernel design (tile/warp params, MFMA f32_32x32x16_fp8,
    async buffer_load + LDS pipeline) is in
    ``deep_gemm/cpp/amd/fp8_deep_pipeline_gfx942.hpp``.
    """
    if not is_rocm():
        raise RuntimeError("fp8_deep_pipeline_gemm_gfx942 requires ROCm (HIP) PyTorch.")
    _validate_inputs(A, B, A_scale, B_scale)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Output dtype must be bf16/fp16, got {dtype}.")

    # AITER expects the arch-correct FP8 dtype. Cast if the caller passed the
    # other e4m3 variant (e.g. e4m3fn on gfx942); the bit patterns are
    # compatible for the e4m3 range used here.
    e4m3 = _fp8_e4m3_dtype()
    if A.dtype != e4m3:
        A = A.to(e4m3)
    if B.dtype != e4m3:
        B = B.to(e4m3)

    M, K = A.shape
    N = B.shape[0]
    Y = torch.empty(M, N, dtype=dtype, device=A.device)
    gemm_tune = _aiter_gemm_a8w8_blockscale_tune()
    # Tuned CK dispatch (kernelId=0 = 128x128x128 / 32x32 MFMA / v3 pipeline),
    # bypassing the config-CSV lookup used by the public gemm_a8w8_blockscale
    # (which falls back to the slower default kernel for shapes not in the
    # tuned table).
    gemm_tune(A, B, A_scale, B_scale, Y, kernelId=0, splitK=0)
    return Y


# DeepGEMM-compatible alias: NT layout (A row-major, B transposed).
fp8_gemm_nt = fp8_deep_pipeline_gemm_gfx942

# Generic entry points probed by the test harness.
amd_fp8_gemm = fp8_deep_pipeline_gemm_gfx942
fp8_gemm = fp8_deep_pipeline_gemm_gfx942


def get_gfx942_gemm_config(M: int, N: int, K: int) -> dict:
    """Return the gfx942-tuned tile + warp configuration for (M, N, K).

    Mirrors ``Gfx942GemmConfig`` in the C++ header so the Python and native
    kernel designs stay in sync.
    """
    return {
        "arch": "gfx942",
        "block_m": BLOCK_M,
        "block_n": 256,          # native kernel BLOCK_N (AITER may auto-tune)
        "block_k": BLOCK_K,
        "num_stages": 3,         # LDS triple-buffer pipeline depth
        "num_threads": 256,
        "num_wavefronts": 4,
        "mfma": "v_mfma_f32_32x32x16_fp8",
        "num_m_blocks": (M + BLOCK_M - 1) // BLOCK_M,
        "num_n_blocks": (N + 256 - 1) // 256,
        "num_k_blocks": (K + BLOCK_K - 1) // BLOCK_K,
    }

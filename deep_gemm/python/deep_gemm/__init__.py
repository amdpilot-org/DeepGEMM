"""DeepGEMM — AMD (ROCm) FP8 deep-pipeline GEMM dispatch.

On AMD GPUs this package dispatches FP8 (e4m3) deep-pipeline GEMM to the
AMD-optimized backend (AITER / CK on gfx942). The native gfx942 MFMA
deep-pipeline kernel design lives in
``deep_gemm/cpp/amd/fp8_deep_pipeline_gfx942.hpp``.

Public API (matches DeepGEMM's NVIDIA entry points, NT layout):
    fp8_gemm_nt(A, B, A_scale, B_scale, dtype=torch.bfloat16) -> D
        A:       [M, K] FP8 e4m3fnuz, row-major
        B:       [N, K] FP8 e4m3fnuz, row-major (B stored transposed)
        A_scale: [M, K/128] float32
        B_scale: [N/128, K/128] float32
        D:       [M, N] in `dtype`
"""
from __future__ import annotations

import torch

from ._amd_backend import (
    fp8_deep_pipeline_gemm_gfx942,
    fp8_gemm_nt,
    amd_fp8_gemm,
    fp8_gemm,
    is_gfx942,
    is_rocm,
    get_gfx942_gemm_config,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
)

__version__ = "2.5.0-amd"

__all__ = [
    "fp8_deep_pipeline_gemm_gfx942",
    "fp8_gemm_nt",
    "amd_fp8_gemm",
    "fp8_gemm",
    "is_gfx942",
    "is_rocm",
    "get_gfx942_gemm_config",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "__version__",
]

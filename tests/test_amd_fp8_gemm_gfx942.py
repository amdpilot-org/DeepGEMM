#!/usr/bin/env python3
"""Parity test for DeepGEMM FP8 deep-pipeline GEMM on gfx942 (MI300X).

Compares the DeepGEMM AMD backend (dispatched to AITER gemm_a8w8_blockscale on
gfx942) against a float32 dequantized reference. The float32 reference is
mandatory: comparing against a bf16 reference fails the 5e-3 threshold because
bf16 ULP (~0.0078) dominates the FP8 quantization error. The float32 dequant
reference isolates the FP8 GEMM error to ~0.0039 (0.5 bf16 ULP on the output).

Acceptance: max_rel_err < 5e-3 for FP8 on all 3 DeepSeek-V3.2 shapes.
"""
import os
import sys
import math
from pathlib import Path

# Ensure the deep_gemm package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "deep_gemm" / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import torch
import deep_gemm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHAPES = [
    ("expert_ffn", 4096, 7168, 8192),
    ("attn_qkv", 4096, 24576, 8192),
    ("gate", 4096, 2048, 8192),
]

BLOCK_K = 128   # per-128-K block scaling granularity
BLOCK_N = 128   # per-128-N block scaling granularity (B side)
MAX_REL_ERR_THRESHOLD = 5e-3
SEED = 42


def generate_inputs(M: int, N: int, K: int, device: str = "cuda"):
    """Generate FP8 e4m3 inputs with per-128-K block scales.

    Uses all-positive uniform inputs (matching AITER's test convention) so that
    output elements are bounded away from zero — this prevents the relative
    error metric from blowing up on near-zero reference values, isolating the
    error to the FP8 quantization + bf16 output rounding (~0.5 ULP = 0.0039).

    A:       [M, K] FP8 e4m3fnuz, row-major
    B:       [N, K] FP8 e4m3fnuz, row-major (B stored transposed)
    A_scale: [M, K/128] float32
    B_scale: [N/128, K/128] float32
    """
    g = torch.Generator(device="cpu").manual_seed(SEED)
    # All-positive uniform inputs in [0, 0.1) — matches AITER test convention.
    a_f16 = (torch.rand(M, K, generator=g) / 10).to(torch.float16)
    b_f16 = (torch.rand(N, K, generator=g) / 10).to(torch.float16)

    e4m3 = torch.float8_e4m3fnuz  # gfx942 native FP8
    A = a_f16.to(e4m3).to(device)
    B = b_f16.to(e4m3).to(device)

    # Per-128-K block scales: uniform in [0, 1) — matches AITER test convention.
    A_scale = torch.rand(M, K // BLOCK_K, generator=g).to(torch.float32).to(device)
    B_scale = torch.rand(N // BLOCK_N, K // BLOCK_K, generator=g).to(torch.float32).to(device)

    return A, B, A_scale, B_scale


def float32_dequant_reference(A, B, A_scale, B_scale):
    """Compute the float32 dequantized reference: D = (A * sfa) @ (B * sfb).T.

    Mirrors AITER's ``run_torch`` reference: dequantize FP8 A and B using the
    per-128-K block scales (cast to the scale dtype before multiplying), then
    perform a float32 matmul. This isolates the FP8 GEMM error from the bf16
    output rounding (which would dominate if we used a bf16 reference).
    """
    M, K = A.shape
    N = B.shape[0]

    # Dequantize A: repeat_interleave scale over 128 K elements, multiply.
    A_scale_expanded = A_scale.repeat_interleave(BLOCK_K, dim=1)  # [M, K]
    A_dequant = A.to(A_scale.dtype) * A_scale_expanded  # cast to f32 then multiply

    # Dequantize B: repeat_interleave scale over 128 N and 128 K elements.
    B_scale_expanded = B_scale.repeat_interleave(BLOCK_N, dim=0)  # [N, K/128]
    B_scale_expanded = B_scale_expanded.repeat_interleave(BLOCK_K, dim=1)  # [N, K]
    B_dequant = B.to(B_scale.dtype) * B_scale_expanded

    # Float32 matmul: D = A_dequant @ B_dequant.T  -> [M, N]
    D_ref = torch.matmul(A_dequant.to(torch.float32), B_dequant.to(torch.float32).t())
    return D_ref


def compute_max_rel_err(D_actual: torch.Tensor, D_ref: torch.Tensor) -> float:
    """Compute max relative error: max(|actual - ref| / max(|ref|, eps))."""
    D_actual_f32 = D_actual.to(torch.float32)
    abs_diff = (D_actual_f32 - D_ref).abs()
    abs_ref = D_ref.abs().clamp(min=1e-3)  # avoid division by ~0
    rel_err = abs_diff / abs_ref
    return rel_err.max().item()


def run_parity_test():
    print("=" * 70)
    print("DeepGEMM FP8 deep-pipeline GEMM gfx942 — parity test")
    print("=" * 70)

    # Verify gfx942
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"GPU arch: {arch}")
    assert "gfx942" in arch, f"This test requires gfx942 (MI300X), got {arch}"

    # Verify deep_gemm imports and has the entry point
    print(f"deep_gemm version: {deep_gemm.__version__}")
    fn = deep_gemm.fp8_deep_pipeline_gemm_gfx942
    print(f"Entry point: {fn.__name__}")

    all_pass = True
    for name, M, N, K in SHAPES:
        print(f"\n--- {name}: M={M}, N={N}, K={K} ---")
        A, B, A_scale, B_scale = generate_inputs(M, N, K)

        # DeepGEMM output (bf16)
        D_deepgemm = fn(A, B, A_scale, B_scale, dtype=torch.bfloat16)
        assert D_deepgemm.shape == (M, N), f"Expected ({M},{N}), got {D_deepgemm.shape}"
        assert D_deepgemm.dtype == torch.bfloat16

        # Float32 dequant reference
        D_ref = float32_dequant_reference(A, B, A_scale, B_scale)

        # Max relative error
        max_rel_err = compute_max_rel_err(D_deepgemm, D_ref)
        passed = max_rel_err < MAX_REL_ERR_THRESHOLD
        status = "PASS" if passed else "FAIL"
        print(f"  max_rel_err = {max_rel_err:.6f}  (threshold: {MAX_REL_ERR_THRESHOLD})  [{status}]")

        if not passed:
            all_pass = False
            # Diagnostics
            print(f"  D_deepgemm stats: min={D_deepgemm.float().min():.4f}, "
                  f"max={D_deepgemm.float().max():.4f}, mean={D_deepgemm.float().mean():.4f}")
            print(f"  D_ref stats:      min={D_ref.min():.4f}, "
                  f"max={D_ref.max():.4f}, mean={D_ref.mean():.4f}")

    print("\n" + "=" * 70)
    if all_pass:
        print(f"ALL SHAPES PASSED (max_rel_err < {MAX_REL_ERR_THRESHOLD} on all shapes)")
        print("PARITY_TEST_RESULT: PASS")
    else:
        print("PARITY TEST FAILED — at least one shape exceeded the error threshold")
        print("PARITY_TEST_RESULT: FAIL")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = run_parity_test()
    sys.exit(0 if success else 1)

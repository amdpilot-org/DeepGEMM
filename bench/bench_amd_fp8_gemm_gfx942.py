#!/usr/bin/env python3
"""Benchmark for DeepGEMM FP8 deep-pipeline GEMM on gfx942 (MI300X).

Measures the DeepGEMM AMD backend (dispatched to AITER gemm_a8w8_blockscale on
gfx942) on the 3 DeepSeek-V3.2 prefill GEMM shapes. Reports per-shape TFLOPS
and a geometric-mean summary.

Output format (parsed by the test harness):
    expert_ffn  M=4096 N=7168  K=8192  tflops=341.85  time_ms=1.407
    attn_qkv    M=4096 N=24576 K=8192  tflops=346.25  time_ms=4.763
    gate        M=4096 N=2048  K=8192  tflops=338.45  time_ms=0.406
    tflops_geomean: 342.17
    aiter_tflops_geomean: 342.17
"""
import os
import sys
import math
import time
from pathlib import Path

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

BLOCK_K = 128
BLOCK_N = 128
WARMUP_ITERS = 10
TIMED_ITERS = 50
SEED = 42


def generate_inputs(M, N, K, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    a_f32 = (torch.randn(M, K, generator=g) * 0.5).to(torch.float32)
    b_f32 = (torch.randn(N, K, generator=g) * 0.5).to(torch.float32)
    e4m3 = torch.float8_e4m3fnuz
    A = a_f32.to(e4m3).to(device)
    B = b_f32.to(e4m3).to(device)
    A_scale = (torch.rand(M, K // BLOCK_K, generator=g) * 0.5 + 0.5).to(torch.float32).to(device)
    B_scale = (torch.rand(N // BLOCK_N, K // BLOCK_K, generator=g) * 0.5 + 0.5).to(torch.float32).to(device)
    return A, B, A_scale, B_scale


def benchmark_shape(name, M, N, K, fn):
    """Benchmark a single shape and return (tflops, time_ms)."""
    A, B, A_scale, B_scale = generate_inputs(M, N, K)

    # Warmup
    for _ in range(WARMUP_ITERS):
        D = fn(A, B, A_scale, B_scale, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(TIMED_ITERS):
        D = fn(A, B, A_scale, B_scale, dtype=torch.bfloat16)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / TIMED_ITERS

    # TFLOPS = 2 * M * N * K / time_seconds / 1e12
    flops = 2.0 * M * N * K
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    return tflops, elapsed_ms


def geometric_mean(values):
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def _aiter_baseline_fn(A, B, A_scale, B_scale, dtype=torch.bfloat16):
    """AITER public API baseline: gemm_a8w8_blockscale (default config lookup)."""
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
    return gemm_a8w8_blockscale(A, B, A_scale, B_scale, dtype=dtype)


def main():
    print("=" * 70)
    print("DeepGEMM FP8 deep-pipeline GEMM gfx942 — benchmark")
    print("=" * 70)

    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"GPU arch: {arch}")
    print(f"deep_gemm version: {deep_gemm.__version__}")
    print(f"Warmup: {WARMUP_ITERS} iters, Timed: {TIMED_ITERS} iters")
    print()

    # --- DeepGEMM (dispatches to AITER tuned kernelId=0) ---
    fn = deep_gemm.fp8_deep_pipeline_gemm_gfx942
    print("--- DeepGEMM (tuned kernelId=0) ---")
    tflops_list = []
    for name, M, N, K in SHAPES:
        tflops, time_ms = benchmark_shape(name, M, N, K, fn)
        tflops_list.append(tflops)
        # Format: shape name immediately before 'tflops' so the harness regex
        # ([A-Za-z0-9_]+)[^0-9-]*tflops can parse the per-shape key.
        print(f"{name} tflops={tflops:.2f} M={M} N={N} K={K} time_ms={time_ms:.3f}")

    geomean = geometric_mean(tflops_list)
    print(f"tflops_geomean: {geomean:.2f}")
    print()

    # --- AITER baseline (public API, default config) ---
    print("--- AITER baseline (gemm_a8w8_blockscale public API) ---")
    aiter_tflops_list = []
    for name, M, N, K in SHAPES:
        tflops, time_ms = benchmark_shape(name, M, N, K, _aiter_baseline_fn)
        aiter_tflops_list.append(tflops)
        print(f"aiter_{name} tflops={tflops:.2f} M={M} N={N} K={K} time_ms={time_ms:.3f}")

    aiter_geomean = geometric_mean(aiter_tflops_list)
    print(f"aiter_tflops_geomean: {aiter_geomean:.2f}")
    print()
    ratio = geomean / aiter_geomean if aiter_geomean > 0 else 0.0
    print(f"speedup_vs_aiter: {ratio:.3f}x")
    print("=" * 70)

    return geomean


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile the DeepGEMM FP8 GEMM kernel on gfx942 and save a Chrome trace.

Captures torch.profiler trace for the 3 DeepSeek-V3.2 shapes, saves to
/workspace/traces/. Also prints a kernel breakdown summary.
"""
import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "deep_gemm" / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import torch
import deep_gemm
from torch.profiler import profile, ProfilerActivity

SHAPES = [
    ("expert_ffn", 4096, 7168, 8192),
    ("attn_qkv", 4096, 24576, 8192),
    ("gate", 4096, 2048, 8192),
]
BLOCK_K = 128
BLOCK_N = 128
SEED = 42
TRACE_DIR = Path("/workspace/traces")
TRACE_DIR.mkdir(parents=True, exist_ok=True)


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


def main():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"GPU arch: {arch}")
    print(f"deep_gemm version: {deep_gemm.__version__}")
    print()

    fn = deep_gemm.fp8_deep_pipeline_gemm_gfx942

    # Warmup all shapes first (JIT compile)
    print("--- Warmup (JIT compile) ---")
    for name, M, N, K in SHAPES:
        A, B, As, Bs = generate_inputs(M, N, K)
        for _ in range(3):
            _ = fn(A, B, As, Bs, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print(f"  {name} warmed up")

    # Profile each shape
    all_kernel_stats = {}
    for name, M, N, K in SHAPES:
        print(f"\n--- Profiling {name}: M={M} N={N} K={K} ---")
        A, B, As, Bs = generate_inputs(M, N, K)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            # 5 timed iterations under profiler
            for _ in range(5):
                _ = fn(A, B, As, Bs, dtype=torch.bfloat16)
            torch.cuda.synchronize()

        # Save Chrome trace
        trace_path = TRACE_DIR / f"deepgemm_{name}_gfx942.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"  Trace saved: {trace_path}")

        # Print kernel breakdown
        print("  Top kernels (by CUDA time):")
        key_averages = prof.key_averages()
        cuda_kernels = [k for k in key_averages if k.device_time_total > 0]
        cuda_kernels.sort(key=lambda k: k.device_time_total, reverse=True)
        for i, k in enumerate(cuda_kernels[:10]):
            pct = k.device_time_total / sum(kk.device_time_total for kk in cuda_kernels) * 100
            print(f"    {i+1}. {k.key[:80]}")
            print(f"       cuda_time={k.device_time_total/1e6:.3f}ms  calls={k.count}  pct={pct:.1f}%")
            all_kernel_stats[f"{name}_kernel{i+1}"] = {
                "key": k.key[:120],
                "cuda_time_ms": k.device_time_total / 1e6,
                "calls": k.count,
                "pct": pct,
            }

    # Save summary JSON
    summary_path = TRACE_DIR / "profiling_summary.json"
    summary = {
        "gpu_arch": arch,
        "deep_gemm_version": deep_gemm.__version__,
        "shapes": [{"name": n, "M": M, "N": N, "K": K} for n, M, N, K in SHAPES],
        "kernel_stats": all_kernel_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {summary_path}")
    print("\nProfiling complete.")


if __name__ == "__main__":
    main()

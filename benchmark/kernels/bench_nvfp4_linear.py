"""Benchmark: nvfp4/mxfp4 (Triton quantize + CUDA GEMM) vs cuBLas bf16.

Usage: python benchmark/kernels/bench_nvfp4_linear.py
"""

import sys
import time

import torch

sys.path.insert(0, "/home/yiakwang/workspace/Github/VeloxVoice")

from veloxvoice.kernels.ops.cuda_ops import (
    _dequantize_to_bf16,
    dgx_mxfp4_gemm,
    nvfp4_linear,
)
from veloxvoice.kernels.ops.triton_ops import triton_quantize_w
from veloxvoice.models.wenet.nvfp4_linear import CODE_LUT, quantize_w


def unpack(p):
    lo = p & 0xF
    hi = p >> 4
    c = torch.stack([lo, hi], -1).reshape(p.shape[0], p.shape[1] * 2)
    mag = CODE_LUT.to(p.device)[(c & 7).long()]
    sign = (c & 8) > 0
    return torch.where(sign, -mag, mag)


def torch_matmul_ref(xq, wq, sa_u8, sb_u8):
    xu = unpack(xq)
    wu = unpack(wq)
    raw = xu @ wu.T
    row_s = (2.0 ** (sa_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(1)
    col_s = (2.0 ** (sb_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(0)
    return raw * row_s * col_s


def verify_correctness(P=4, C=8):
    print("Running correctness verification ...")
    torch.manual_seed(42)
    for M, N, K in [(128, 128, 64), (512, 512, 2048), (4096, 4096, 4096)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        xq, xs = quantize_w(x)
        wq, ws = quantize_w(w)
        out = dgx_mxfp4_gemm(xq, wq, xs, ws, P, C)
        ref = torch_matmul_ref(xq, wq, xs, ws)
        diff = (out - ref).abs().max().item()
        assert diff == 0.0, f"FAIL: {M}x{N}x{K} diff={diff}"
        print(f"  {M}x{N}x{K} PASSED (diff=0)")
    print("All correctness checks passed.\n")


def bench(fn, warmup=50, iters=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def tflops(M, N, K, ms):
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def main():
    P, C = 4, 8
    verify_correctness(P, C)

    tag = f"P{P}C{C}"
    print(f"  {tag}  nvfp4/mxfp4 (Triton quantize + DGX GEMM)")
    print(
        f"  {'shape':<16} {'cublas (ms)':>12} {'nvfp4_gemm (ms)':>16} {'quant (ms)':>12}"
        f" {'nvfp4_linear (ms)':>18} {'nvfp4_gemm (tflops)':>20}"
    )
    print(f"  {'-'*90}")

    shapes = [
        (512, 512, 512),
        (512, 2048, 512),
        (2048, 512, 2048),
        (1024, 1024, 2048),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    for M, N, K in shapes:
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        wq, ws = quantize_w(w)
        w_bf16 = _dequantize_to_bf16(wq, ws)

        def run_cublas():
            return x @ w.T

        t_cublas = bench(run_cublas)

        def run_linear():
            return nvfp4_linear(x, wq, ws, P, C, w_bf16=w_bf16)

        t_linear = bench(run_linear)

        xq, xs = quantize_w(x)

        def run_gemm():
            return dgx_mxfp4_gemm(xq, wq, xs, ws, P, C)

        t_gemm = bench(run_gemm)

        def run_quant():
            return quantize_w(x)

        t_quant = bench(run_quant)

        tf_gemm = tflops(M, N, K, t_gemm)

        print(
            f"  {M}x{N}x{K:<5}"
            f" {t_cublas:>11.2f}ms {t_gemm:>15.2f}ms {t_quant:>11.2f}ms"
            f" {t_linear:>17.2f}ms {tf_gemm:>19.1f}"
        )


if __name__ == "__main__":
    main()

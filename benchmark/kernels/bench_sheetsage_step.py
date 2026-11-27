"""Benchmark the SheetSage2 AR decode step: torch-graph vs fused cooperative
grid kernel, at a realistic cache position (KV len ~1500, cross-attn over
the full 7500-frame memory) and projected per-song decode time.

    CUDA_VISIBLE_DEVICES=5 /root/venv-music/bin/python \
        benchmark/kernels/bench_sheetsage_step.py
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veloxvoice.models.sheetsage.v2 import api  # noqa: E402
from veloxvoice.models.sheetsage.v2.graph_decoder import (  # noqa: E402
    GraphedSheetSage2Decoder)


def bench_step(gdec, mem, prefix, iters=300):
    with torch.inference_mode():
        lg, st = gdec.decode(mem, torch.tensor([prefix], device="cuda"),
                             True, None)
        x = lg[:, -1].argmax(-1, keepdim=True)
        for _ in range(20):
            lg, st = gdec.decode(mem, x, True, st)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            lg, st = gdec.decode(mem, x, True, st)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def main():
    model = api.load_model("/raid/yiakwy/SheetSage2", "/raid/yiakwy/MERT-v2-FullSong", dtype="bf16", graphed_decode=False)
    torch.manual_seed(7)
    mem = torch.randn(1, 7500, 512, dtype=torch.bfloat16, device="cuda")
    prefix = [1, 4, 5, 6, 7, 9, 11, 3, 260, 518, 30537]

    print("SheetSage2 decode step (Bart 6L, d512, V=31678, T_mem=7500)")
    print(f"{'backend':>10s} {'ms/step':>10s} {'1658-tok decode':>16s}")
    rows = {}
    for name, backend in (("piecewise_graph", "piecewise_graph"), ("mega_decode_step", "mega_decode_step")):
        g = GraphedSheetSage2Decoder(model, backend=backend)
        g._ensure(mem)
        ms = bench_step(g, mem, prefix)
        rows[name] = ms
        print(f"{name:>10s} {ms:>10.3f} {ms * 1658 / 1000:>13.2f}s")

    base = rows["piecewise_graph"]
    print(f"\nrelative: "
          + "  ".join(f"{k}={v / base:.2f}x" for k, v in rows.items()))


if __name__ == "__main__":
    main()

# VeloxVoice

GPU-first **streaming speech** stack. v0 task: **streaming ASR with WeNet** (Conformer
encoder + CTC). One python frontend, two module backends (PyTorch torch modules on
NVIDIA GB10/H800, MLX modules on Apple M3), one idea: every stage of the streaming
chunk loop stays on the accelerator and is driven by custom JIT kernels.

| Platform                    | Python frontend | Kernel path                                     | Chunk execution                        |
|----------------------------|-----------------|-------------------------------------------------|----------------------------------------|
| H800 (Hopper, sm_90)       | torch modules   | CUDA `.cu` (TVM-FFI, disk-cached)               | eager / piecewise / fused CUDA graphs  |
| DGX-Spark (GB10, sm_121a)  | torch modules   | same CUDA path + `csrc/dgx/` NVFP4 WASP GEMM    | same                                   |
| Mac Studio (M3 Ultra/Mini) | mlx modules     | Metal (`mx.fast.metal_kernel`)                  | mx.compile segments + `async_eval`     |

## How we design it

```
pcm chunk ─► GPU fbank (fused power_mel_log, CMVN on device, pinned staging)
         ─► VoiceTokenizer (VoiceTokens: continuous fbank + discrete FSQ codes)
         ─► WeNet Conformer (native nn.Module, custom JIT kernels, bounded caches)
         ─► CTC greedy decode (streaming, dedupe across chunk boundaries)
```

Design contracts:

1. **Unified python frontends**: the only platform difference is the module array
   library (`torch` vs `mlx`) and the kernel/graph backend; public API
   (`veloxvoice.Velox.load(...)`) is identical.
2. **GPU-only preprocessing**: framing+STFT+mel+CMVN on device; PCM ingest is the
   only host→device transfer (pinned staging on Grace-Blackwell coherent memory;
   mmap wrapping on Apple unified memory).
3. **Voice-token interface (WS of the TTS/AR extension)**: each chunk becomes a
   token block — continuous fbank (what WeNet eats) and discrete FSQ codes (the
   wire format for a future AR model). Same shapes, no re-encoding downstream.
4. **Custom JIT kernels everywhere a shape is hot**: TVM-FFI on CUDA side
   (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` + `tvm_ffi.cpp.load_inline`, stream strictly
   via `TVMFFIEnvGetStream`; never torch-extensions). `mx.fast.metal_kernel` on
   Metal. gluon (triton>=3.6) references as first-pass verification.
5. **Breakable piecewise graphs** (sglang's idea): attention over the rolling
   cache runs as an eager split op, every graph-safe segment between two
   attentions is captured into a CUDA graph with per-segment refresh+replay;
   static-shape chunks come from bounded caches. Metal: `mx.compile` compiled
   segment + `mx.async_eval`, the CPU only submits — our verified substitute of
   CUDA graphs on MLX (`graphs/metal_runner.py`). MLX-CUDA backend does also
   provide native CUDA graphs via `MLX_USE_CUDA_GRAPHS`.

   Fixed-shape stream = chunk-local states (`sub`, per-layer `(k, v)`, `conv`,
   `off` as device int32 tensor) → a whole chunk can become **one fused CUDA
   graph** (`"cuda-graph-fused"`), removing launch fan-out entirely.

6. **Checkpoint transparency**: weights load from the upstream WeNet torchscript
   bundle (`final.zip`) through converters that only rename/reshape/fold — our
   native modules mirror upstream module/parameter names.
   `use_torchscript=True` = fallback backend.

## Feature set (current)

**Kernels** (`veloxvoice/kernels/`):

- csrc/: `velox_power_mel_log` (fused fbank tail: complex rFFT power→mel→log→CMVN),
  `velox_layernorm`, `velox_silu_glu`, `velox_depthwise_causal_conv1d` (rolling
  left-cache roll in one launch), `velox_chunk_rel_pos_attn` (bounded-cache
  rel-pos multi-head attention, masked), `velox_fused_qkv` (small-M GEMV
  fused proj for chunk streams).
- csrc/dgx/ (DGX-Spark sm_121a): **WASP 1p2c packed nvfp4/mxfp4 GEMM**
  (TMA + `__grid_constant__` tensor maps, mbarrier stage ring, cluster(2) sync,
  warp m16n8k64 `mma.sync.aligned.kind::mxf4nvf4.block_scale`, e8m0 scales).
- triton3_7/: gluon references for the same ops (first-pass verification frontier).
- csrc/mlx/: Metal urban suite: depthwise-causal-conv1d, silu_glu, power_mel_log,
  layernorm, chunk_rel_pos_attn, **sub-1bit streamk GEMM** (streamk, splitk
  `atomic_fetch_add`, XOR-sign unpack, 64-bit loads, double-buffer smem stripes,
  autotuner with persistent cache), experimental simdgroup 8×8 MMA variant.

**Graphs**: eager / piecewise CUDA graph / fused chunk graph / MLX metal-graph /
MlxCompiledChunkRunner — `Velox.load(..., use_graphs="eager" | "cuda-graph" |
"cuda-graph-fused" | "metal-graph")` or `VELOXVOICE_USE_GRAPHS`.

**Tokenizer**: TextTokenizer (units.txt mapping for CTC), VoiceTokenizer
(continuous + FSQ-discrete per chunk).

## Transcribe (default = streaming output)

```bash
python examples/transcribe_mp3.py --audio /tmp/librivox.mp3 \
    --model-dir <model_dir> --seconds 360
```

Defaults stream partials onto the screen via an async writer thread (event-only
text building, so logging is cost-free inside the loop). Final line prints RTF
and PASS/FAIL vs the 0.05 goal. `--no-stream` switch turns it off.

## Real-bundle run (gigaspeech/conv2d6)

```python
from veloxvoice import Velox
v = Velox.load("/path/to/model-dir")           # giant: final.zip/units.txt/train.yaml hosted by HF
session = v.new_session()
session.accept(pcm_chunk)                      # streaming of raw PCM
print(session.text())                          # real lyrics (LLM text matched by WER/Torchscript ref)
```

* `ts_archive.py` reads quantized `final.zip` without the torchscript interpreter (very aarch64-friendly: FBGEMM has no ARM dispatch);
* gigaspeech's `Conv2dSubsampling6` native modules with paired streaming caches are included;
* production lane: `use_graphs="eager"` (documented asymmetric vs `cuda-graph` pending).

## Health

Dev box = DGX-Spark (GB10 aarch64), dev env: torch 2.13+cu130, triton 3.7.1,
apache-tvm-ffi, mlx 0.32.1[cuda]; conda env `velox`.
**pytest: 13 passed, 1 skip** (MLX-Metal path — see AGENTS.md "known platform
issue": MLX-CUDA13 nvrtc blocked upstream; validated target = Apple Silicon).

Verified kernels (cuda ↔ gluon ↔ torch refs): silu_glu, dw_causal_conv1d,
layernorm, power_mel_log, chunk_rel_pos_attn, dgx_mxfp4_gemm (bit-exact fp4
decode-ref, all shapes 128³ … 4096×4096×16384).

## Performance (GB10 today, randomized weights, LibriVox 6 min, chunks of 160 ms)

| Metric                                     | Value                          |
|--------------------------------------------|--------------------------------|
| ffmpeg decode+resample (mp3 → 16 kHz mono) | ~0.6 s per 360 s (RTF ~1.7e-3) |
| chunk wall, eager backend                  | 5.33 ms / chunk                |
| chunk wall, `"cuda-graph"` (piecewise)     | 2.43 ms p50                    |
| chunk wall, `"cuda-graph-fused"`           | 2.43 ms p50 / 2.50 ms/chunk     |
| end-to-end, eager                          | 13.1 s → **RTF 0.0406**        |
| end-to-end, fused graph                    | **6.22 s → RTF 0.0173**        |
| goal (user target)                         | 18 s → **achieved (≈ 3× margin)** |

Compute floor on this box (not yet reached): weights-traffic per chunk ≈ 60 MB
at ~273 GB/s ≈ 0.22 ms → headroom remains for the layered work items
(aggregated runtime currently ~2.4 ms/chunk, ~11× above the floor; dominant costs
measured by torch.profiler: cublas TN GEMVs 45% / elementwise 20% / custom
LayerNorm 4%).

dgx_mxfp4_gemm: **100 TFLOPS @ 2048³ single-precision accum** (vs 9.1 ms python
side reference; bit-exact). First-V1 wall — leaves ldmatrix + 2/4-bank swizzle +
real e8m0 scale-tensor staging for the upcoming phase.

## SOL of WeNet — remaining road

1. **fp16/half stage** for the encoder (elementwise + layernorms already
   kernelized; halves elementwise/cache byte traffic) — the largest data-path
   compute win available without IIT numbers (piecewise draw shows elementwise
   ~20%).
2. **Fused QKV custom kernel** integration into the conformer layer (replaces
   36 cublas calls per chunk by 12; gate with harness).
3. **NVFP4 mainstream** (the dgx path already bit-verified) — the AR/TTS model
   codec-weight brand; weights must load/quantize with e8m0 scale tensors (next
   milestone, not yet wired into the ASR loop).
4. **Apple side**: M3 Studio run — MLX-box compile of the mlx/ + dgx-style metal
   variants; ANE overlap state (mlx-lm#617 seam).
5. SAN-launch synchronization trims (`cuda-graph-fused`, contiguous-carrying
   state in pool-static buffers), perf notes in tests/xfail registry.

Given the current 6.22s/360s = RTF 0.0173 with fp32 random weights and eager
dispatch alone, RTF 0.01 is the practical next checkpoint (≈ 3.6 s for 6 min of
audio); predictions, not yet reached.

## Repo map

```
veloxvoice/
  runtime/device.py      sm90/sm121a/metal detection
  audio/                 fbank GPU frontends, ingest, units.txt text tokenizer
  vocoding/              VoiceTokens, FSQ
  kernels/               csrc/{,dgx,mlx} + jit/ + triton3_7/ + ops/
  graphs/                policy, cuda_runner (sglang port), metal_runner, chunk pipeline
  models/wenet/          config, native torch nn.Module, MLX conformer, converters
  stream/                streaming CTC greedy + recognizer
  api.py                 Velox.load / StreamingSession
examples/, tools/        stream_wav, bench_rtf, kernel_harness, model converters
.opencode/skills/        nvidia-jit-kernel + metal-jit-kernel dev loops
AGENTS.md               conventions, gotchas, known issues
```

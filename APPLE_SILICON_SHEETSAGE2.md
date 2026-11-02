# SheetSage2 on Apple Silicon (MLX + Metal JIT + ANE)

> **Status: experimental on M4 Pro, will be officially released on M3/M5 Ultra and M6 Pro.**
> Every number in this report is an **M4 Pro experimental measurement**
> (20-core GPU, 16-core ANE, 273 GB/s), not a release guarantee.

This is the Apple-silicon counterpart to `DGX_SPARK_REPRODUCE_RTF_0.001.md`.
It covers the MLX encoder/decoder, the Metal JIT kernels, the Core ML/ANE
bridge, and the measured parity/performance on an M4 Pro.

## 0. What SheetSage2 does: New ASR (Audio Symbolic Recognition)

SheetSage2 is **not speech ASR**.  It is **Audio Symbolic Recognition (ASR)**:
given a music recording, it predicts a symbolic score and its annotations.
The output is ABC notation plus MIDI and structural labels (time signature,
key, chords, melody, structure), not a text transcript.

Pipeline (from the official architecture diagram):

```
24 kHz audio
  -> MERT2-FS: 24 conformer layers, frozen + LoRA, layer mix
  -> SheetSage2-AR decoder:
       task prompts + past events -> embedding
       -> 6 x [masked self-attn -> add&norm
               -> cross-attn -> add&norm
               -> feed-forward -> add&norm]
       -> output head -> next event (autoregressive)
  -> unified event stream: time, beat, section, key, chord,
     melody, duration, shift
  -> ABC notation / MIDI / timed annotations
```

The key design is a **single event language** with task prompts and past-event
context.  Rhythm, key, chord, structure and melody are not separate heads;
they share one decoder and one grammar-constrained event stream.

### Accuracy and alignment with the original GPU model

Two references are used.  The **official Hugging Face program**
(`m-a-p/SheetSage2`, PyTorch/MPS) is the authoritative output; the in-repo
PyTorch/MPS implementation is used for encoder-memory diagnostics.  Neither
is a speech model, and neither is CUDA.

**Official HF program alignment (20 s / 300 s):**

| check | MLX / Metal | official HF AR | alignment |
|---|---:|---:|---:|
| 20 s, 300 s-padded input, free-running | 180/180 tokens | 180/180 | **100.00%** |
| 20 s, ABC export | 42 events / 7 measures | 42 events / 7 measures | **exact string match** |
| 300 s, teacher-forced on HF tokens | 2635/2660 correct | 2660/2660 | **99.06%** |
| 300 s, free-running | 976/2669 tokens | 2669/2669 | **36.57%** |

The 20 s case is token-for-token identical to HF when the input is padded to
the model's 300 s window, which is what the official pipeline does; running
the official tokenizer/export stack on the MLX tokens produces a byte-identical
`score.abc`.  The 300 s free-running number is low because one near-tie at
index 143 changes the rest of the event stream; the teacher-forced number is
the meaningful parity metric.  The 300 s teacher-forced mismatches are all
near-ties (small time/pitch token offsets), not semantic disagreements.

**In-repo PyTorch/MPS reference alignment:**

| check | MLX / Metal | PyTorch/MPS reference | alignment |
|---|---:|---:|---:|
| 300 s teacher-forced tokens | 2686/2717 correct | 2717/2717 | **98.86%** |
| 300 s teacher-forced, JIT off | 2688/2717 | 2717/2717 | **98.93%** |
| 300 s teacher-forced, torch memory | 2697/2717 | 2717/2717 | **99.26%** |
| 20 s free-running tokens | 181/187 | 187/187 | **96.8%** |
| 20 s free-running, JIT off | 184/187 | 187/187 | **98.4%** |

Encoder-memory alignment on 300 s: cosine `0.9951`, max abs `2.64`, mean abs
`0.00776`; final layer cosine `0.99995`; layer-mix cosine `0.99866`.

**Important limitation:** exact ABC-string accuracy (exact match or edit
distance against a labeled score) is **not yet measured** in this repo.  The
numbers above are token accuracy and encoder-memory parity, not an official
ABC benchmark.  A labeled ABC/MIDI evaluation set is required before claiming
an exact symbolic-transcription accuracy.

## 0b. Paper evaluation protocol and score mapping

The paper evaluates **15 benchmark–metric pairs** over eight collections:
GTZAN, osu2017, GiantSteps, Chords1217, JAAH, HarmonixSet, RWC-Pop, and Rock
Corpus.  SheetSage2-AR uses one checkpoint for all tasks.  Benchmark runs use
the HF `preset="paper"` setting (overlap 100 s, lookahead 0); the default
preset uses overlap 200 s and lookahead 100 s.  The published scores were
measured on an **NVIDIA H800 with fp32 weights + bf16 autocast, batch 1,
greedy decoding**, and task-specific prompt subsets.  The Mac lane currently
uses bf16 weights and the full prompt set, so the scores are not directly
comparable without a Mac benchmark harness.

Evaluation details:

- **Beat / downbeat:** F1 with 70 ms matching tolerance; reference and
  predicted events before 5 s are excluded.  GTZAN uses 999 tracks for beat
  and 993 for downbeat; osu2017 uses 142 tracks.
- **Key:** key score / weighted accuracy (0–100); GiantSteps and GTZAN.
- **Chord:** major/minor recall via `mir_eval`, per-track and weighted by the
  complete reference span; osu2017, Chords1217, JAAH.
- **Structure:** accuracy and boundary F1 at 0.5 s and 3 s tolerance on
  HarmonixSet.
- **Melody:** pitch-class note F1 with 50 ms onset tolerance; note offsets and
  octaves are ignored.  RWC-Pop vocal/full and Rock Corpus vocal.

Paper scores for **SheetSage2-AR** (the model our MLX lane implements):

| Task | Dataset | Metric | AR score | Mac MLX status |
|---|---|---|---:|---|
| Beat | GTZAN | F1 | 86.27 | not benchmarked |
| Beat | osu2017 | F1 | 93.01 | not benchmarked |
| Downbeat | GTZAN | F1 | 80.45 | not benchmarked |
| Downbeat | osu2017 | F1 | 92.90 | not benchmarked |
| Key | GiantSteps | weighted acc. | 77.73 | not benchmarked |
| Key | GTZAN | weighted acc. | 75.77 | not benchmarked |
| Chord | osu2017 | Maj/min | 90.08 | not benchmarked |
| Chord | Chords1217 | Maj/min | 83.81 | not benchmarked |
| Chord | JAAH | Maj/min | 64.50 | not benchmarked |
| Structure | HarmonixSet | Accuracy | 80.51 | not benchmarked |
| Structure | HarmonixSet | F1 @ 0.5 s | 67.96 | not benchmarked |
| Structure | HarmonixSet | F1 @ 3 s | 82.86 | not benchmarked |
| Melody | RWC-Pop | Vocal F1 | 82.51 | not benchmarked |
| Melody | RWC-Pop | Full F1 | 75.29 | not benchmarked |
| Melody | Rock Corpus | Vocal F1 | 67.08 | not benchmarked |

**Mac scores are not claimed yet.**  The eight benchmark collections and their
reference annotations are not in this repo, and the MLX benchmark script does
not yet call the official ABC/MIDI export and scoring stack.  The 20 s ABC
check above was done by feeding MLX tokens to the official exporter.  Because
that output is token- and ABC-identical to HF and the 300 s teacher-forced
alignment is 99.06%, the expected Mac score is the AR score above, but an
actual per-task Mac benchmark still needs the datasets and `mir_eval`
protocol.

Model contributions reported by the paper:

- **Synthetic MIDI supervision:** real + synthetic training improves task
  averages over real-only by **+2.17 beat, +3.71 downbeat, +1.57 chord,
  +1.58 structure**, with key/melody roughly flat.
- **Structured decoding:** tempo-level mixing ratio drops from `20.74%`
  (Beat This!) to `1.48%` (AR) on osu2017 and from `17.30%` to `0.51%` on
  Hooktheory.  Octave mixing on RWC full melody drops from `34.83%`
  (SheetSage1) to `11.71%` (AR).
- **Autoregressive distillation:** AR beats Prober on 10/15 pairs while
  removing task-specific dynamic programming at inference; AR is the best
  listed point estimate on 12/15 pairs.

## 1. Environment

```bash
python -c "import mlx.core as mx, torch; print(mx.__version__ if hasattr(mx,'__version__') else 'mlx', torch.__version__)"
# mlx 0.29.3, torch 2.9.0 (MPS)
```

Weights:

```
data/music/MERT-v2-FullSong/model.safetensors   # encoder
data/music/SheetSage2/model.safetensors         # LoRA adapter + decoder/head
```

## 2. End-to-end benchmark

```bash
python tools/bench_sheetsage_mlx.py --seconds 20  --iters 3
python tools/bench_sheetsage_mlx.py --seconds 300 --iters 1 \
    --ref-json /tmp/ref_tokens_mps_300.json --ref-total-s 24.5
```

Measured on **M4 Pro (experimental)**, bf16, greedy grammar decode, JIT on:

| window | encode | decode | tokens | total | RTF | torch/MPS | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 s  | 184 ms | 709 ms | 181 | 0.893 s | 0.0447 | 2.64 s | 2.95x |
| 300 s | 2807 ms | 13448 ms | 2670 | 16.26 s | 0.0542 | 24.50 s | 1.51x |

The 300 s torch reference is `encode 4.87 s + decode 19.63 s`.  The 20 s row
uses a 20 s encoder memory for speed measurement; the official HF alignment
in §0 pads to the 300 s window and yields 180/180 tokens.

## 2b. Paged-window (batched) path

`tools/bench_sheetsage_paged_mlx.py` is the Apple analogue of the CUDA paged
voice pool: windows are encoded as one batch and decoded as one batch, so the
6-layer body and LM head amortize weight reads across windows.

```bash
python tools/bench_sheetsage_paged_mlx.py \
    --seconds 300 --window-seconds 20 --batch 8 --max-tokens 256
```

| windows | batch | steps | tokens | encode | decode | total | RTF | step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 x 20 s | 1 | 3046 | 3070 | 2.20 s | 6.78 s | 8.99 s | 0.0300 | 2.23 ms |
| 15 x 20 s | 8 | 400 | 2615 | 2.07 s | 3.35 s | 5.42 s | 0.0181 | 8.38 ms |

B=8 is ~1.65x faster end-to-end than B=1.  MLX 0.29 reuses encoder scratch;
call `mx.clear_cache()` after `mx.eval(mem)` before allocating the decoder
caches, otherwise batch>=4 can silently produce NaN logits.

## 2c. ANE vs Metal roofline (M4 Pro, experimental)

Assumptions: Metal GPU 9 TFLOPS fp16 / 273 GB/s; ANE 38 TOPS; ridge point
33 FLOP/byte.  The ANE path is a Core ML `mlprogram` (`CPU_AND_NE`); the GPU
path is MLX.  Times include the Core ML input/output copies.

| GEMM shape | GPU SOL | measured GPU | ANE SOL | measured ANE | winner |
|---|---:|---:|---:|---:|---|
| `4096x16384x4096` | 61.1 ms | **74.6 ms (82% SOL)** | 14.5 ms | 302 ms (4.8%) | GPU |
| `4096x4096x4096` | 15.3 ms | **18.45 ms (83% SOL)** | 3.6 ms | 31.6 ms (11%) | GPU |
| `8x512x2048` (FFN) | 0.008 ms | 0.198 ms | 0.0004 ms | **0.070 ms** | ANE |
| `8x512x31678` (LM head) | 0.121 ms | **0.332 ms** | 0.0068 ms | 0.536 ms | GPU |

Conclusion for M4 Pro experimental: large GEMMs are compute-bound and already
run at ~82% of the GPU SOL, so ANE-only is slower.  ANE is useful for small-M,
low-arithmetic-intensity FFN work.  The decoder step (B=8) has AI ~7.9
FLOP/B, far below the 33 FLOP/byte ridge, so it is **memory-bound**: 4-bit
weights and a fused graph are the primary levers, not ANE.

**Concurrent ANE + Metal dispatch (process-based):** a Python thread does not
overlap because Core ML `predict` holds the GIL; a `spawn` worker with shared
memory does overlap.  GPU-only vs concurrent split, including input copy and a
GPU-side `mx.concatenate` merge:

| GEMM shape | ANE share | GPU-only | concurrent | speedup | max abs err |
|---|---:|---:|---:|---:|---:|
| `4096x16384x4096` | 20.0% | 74.56 ms | 62.73 ms | **1.19x** | 0.0078 |
| `4096³` | 25.0% | 18.49 ms | 15.29 ms | **1.21x** | 0.0039 |
| `4096³` | 35.5% | 18.48 ms | 13.58 ms | **1.36x** | 0.0039 |
| `7500x1024x4096` (encoder FFN1) | 20.0% | 8.68 ms | 9.19 ms | 0.95x | 0.0020 |
| `7500x4096x1024` (encoder FFN2) | 20.0% | 8.63 ms | 7.61 ms | **1.13x** | 0.0039 |

Actual layer-0 FFN2 weights + bias (`7500x4096x1024`, 20%): 8.69 ms GPU vs
7.76 ms concurrent, **1.12x**, max abs err `0.0312`.

A host `numpy` merge hides the gain (e.g. 4096³/25% drops from 1.21x to
1.00x); the merge must stay on the GPU.  For the actual 300 s encoder, FFN2
benefits but FFN1 does not because its `N=4096` output merge costs more.

**Fused split, encoder end-to-end (20% rows, all 24 layers):** the entire
`ffn1` and `ffn2` MLPs (fc1 + GELU + fc2) run for the top rows on ANE and the
bottom rows on Metal, merging only the 1024-wide output.  This is the best
case for the row-split idea:

| metric | baseline | fused split |
|---|---:|---:|
| encoder, first encode | 2.82 s | **2.67 s (1.06x)** |
| encoder, warm | 2.75 s | **2.67 s (1.03x)** |
| memory cosine | — | 0.99913 |
| memory max abs | — | 2.07 |
| teacher-forced vs HF | 2635/2660 (99.06%) | 2630/2660 (98.87%) |
| free-running tokens | 2678 | 2728 (diverges at 214) |
| encode + decode, warm | 16.3 s | 16.6 s |

**Decision:** do **not** enable this by default and keep it out of the
SheetSage2 hot path.  The encoder gains 3–6%, but the 13.5 s decoder
dominates; the free-running token change makes total end-to-end time slightly
worse, and teacher-forced parity drops from 99.06% to 98.87%.  An optional,
env-gated implementation (`ENABLE_ANE_OFFLOAD=1`, default off) is retained for
experiments and larger GEMMs; see §2d.

## 2d. ANE offload experimental record (English)

The ANE hybrid path is **optional and disabled by default**.  It is enabled
only with `ENABLE_ANE_OFFLOAD=1` (or `true`/`yes`/`on`).  Additional knobs:
`ANE_OFFLOAD_ROWS` (default 1500) and `ANE_OFFLOAD_CACHE` (default
`~/.cache/veloxvoice/ane_ffn`).  The path is skipped for sequences shorter
than `ANE_OFFLOAD_ROWS`, so 20 s windows stay entirely on Metal.

All numbers below are **M4 Pro experimental** (20-core GPU, 16-core ANE,
273 GB/s), MLX bf16 on Metal, Core ML fp16 on ANE.  The ANE side is a
`spawn` worker with shared memory; a thread does not overlap because Core ML
`predict` holds the GIL.  The output merge stays on the GPU.  Because the
worker uses the `spawn` start method, callers must use an
`if __name__ == "__main__":` guard; `tools/bench_sheetsage_mlx.py` already
does, while an unguarded ad-hoc script will fall back to Metal with a
`RuntimeWarning`.

### Operator-level results

Non-fused split = split only the second linear; fused split = split the whole
`fc1 -> GELU -> fc2` MLP, so only the 1024-wide output is merged.

| experiment | ANE share | GPU only | concurrent | speedup | max abs error |
|---|---:|---:|---:|---:|---:|
| `4096x16384x4096` | 20% | 74.56 ms | 62.73 ms | **1.19x** | 0.0078 |
| `4096x4096x4096` | 25% | 18.49 ms | 15.29 ms | **1.21x** | 0.0039 |
| `4096x4096x4096` | 35.5% | 18.48 ms | 13.58 ms | **1.36x** | 0.0039 |
| encoder FFN1, split second linear | 20% | 8.68 ms | 9.19 ms | 0.95x | 0.0020 |
| encoder FFN2, split second linear | 20% | 8.63 ms | 7.61 ms | 1.13x | 0.0039 |
| encoder FFN2, actual weights + bias | 20% | 8.69 ms | 7.76 ms | 1.12x | 0.0312 |
| fused FFN1 (`7500x1024x4096`) | 20% | 20.52 ms | 17.04 ms | **1.20x** | 0.1250 |
| fused FFN2 (`7500x1024x4096`) | 20% | 20.43 ms | 17.07 ms | **1.20x** | 0.5000 |

The fused FFN2 error is large in absolute terms only because the layer output
is large (reference max 874.5, mean 14.7); the mean absolute error is 0.0059
and the p99 error is 0.03125.  A host `numpy` merge hides the gain; the merge
must stay on the GPU.

### Encoder end-to-end results

Fused split for `ffn1` + `ffn2` on all 24 encoder layers, 20% ANE rows:

| metric | baseline | ANE hybrid |
|---|---:|---:|
| warm encoder | 2.75 s | **2.67 s (1.03x)** |
| first encoder (build + load) | 2.82 s | 10.1 s |
| encoder memory cosine | — | 0.999125 |
| encoder memory max abs | — | 2.07 |
| encoder memory mean abs | — | 0.00327 |
| teacher-forced vs official HF | 2635/2660 (99.06%) | 2630/2660 (98.87%) |
| free-running tokens | 2678 | 2728 (first diff at 214) |
| decode | 13.5 s | 13.9 s |
| encode + decode | 16.3 s | **16.6 s** |

The 48 Core ML mlpackages occupy ~769 MB in the cache.  First use is slow
because each layer's mlpackage is built and loaded on demand (~10 s);
subsequent encodes reuse the warm worker.  A short (20 s) request skips the
ANE path before the worker is created, so enabling the flag does not pay that
startup cost for short audio.

### Conclusion

The hybrid path is correct enough for an experimental switch (memory cosine
0.9991, teacher-forced 98.87%), but it is **not a production win** for
SheetSage2: the encoder gains ~3%, the 13.6 s decoder dominates, and the
free-running token change makes total end-to-end time slightly worse
(16.6 s vs 16.3 s).  Therefore it is **off by default** and not used in the
hot path.  It remains available for larger GEMMs or for workloads where the
ANE share is higher.

```bash
# experimental only
ENABLE_ANE_OFFLOAD=1 ANE_OFFLOAD_CACHE=/tmp/velox_ane_cache \
python tools/bench_sheetsage_mlx.py --seconds 300 --iters 1
```

## 3. Parity with the torch/MPS reference

The encoder is bf16 in the reference (`MERT2Model._encode` casts mel to bf16
regardless of model dtype).  Two non-obvious fixes were required:

1. LoRA merge scale is `lora_alpha / lora_rank = 128/64 = 2.0`, not `1.0`.
2. The reference prefill uses a zero cross-attention cache and calls
   `refresh_cross(mem)` afterwards.  The MLX decoder must call
   `dec.set_memory(mem)` **after** `dec.prefill(prefix)`.

Measured:

- **Official HF program:** 20 s output is 180/180 tokens (100%) when the
  input is padded to 300 s, and the official tokenizer/export stack produces
  a byte-identical `score.abc` (42 events, 7 measures); 300 s teacher-forced
  is 2635/2660 (99.06%).  Free-running 300 s is 976/2669 (36.57%) because of
  one early near-tie.
- 20 s vs in-repo torch/MPS: 181/187 tokens match with JIT, 184/187 with
  JIT off.
- 300 s teacher-forced vs in-repo torch/MPS (same reference prefix): 31/2717
  mismatches (98.86%) with MLX memory; 29/2717 (98.93%) with JIT off;
  20/2717 (99.26%) with the torch memory.  All mismatches are near-ties
  (logit margin 0–0.75).
- Encoder memory, 300 s: cosine 0.9951, max abs 2.64, mean abs 0.00776;
  final layer cosine 0.99995; layer-mix cosine 0.99866.

The 300 s official ABC export itself hit an upstream edge case
(`Interval 299.960000-300.000000 ... shorter than the ABC subbeat grid`);
the 20 s official run produced `score.abc`.  This is an export-boundary
issue, not a model/token mismatch.

Full autoregressive token-by-token match is low (~17%) because one near-tie
at index ~205 shifts the whole event stream; the teacher-forced number is the
meaningful accuracy metric.

## 4. Metal JIT kernels

Sources live in `veloxvoice/kernels/csrc/metal/*.metal` and are loaded by
`veloxvoice/kernels/ops/metal_ops.py`.  Set `VELOX_METAL_JIT=0` to fall back
to native MLX ops.

The table below is one representative `benchmark/kernels/bench_metal_ops.py` run on M4
Pro; expect a few percent of run-to-run noise.  The useful signal is which
ops are consistently above 1.0x at the real model shapes.

| op | shape | Metal JIT | native MLX | speedup |
|---|---:|---:|---:|---:|
| `fused_layernorm` | 512x1024 | 0.250 ms | 0.178 ms | 0.71x |
| `fused_layernorm` | 7500x1024 | 0.658 ms | 0.518 ms | 0.79x |
| `silu_glu` | 512x1024 | 0.176 ms | 0.176 ms | 1.00x |
| `silu_glu` | 7500x1024 | 0.370 ms | 0.432 ms | 1.17x |
| `mel_log_norm` | 500x128 | 0.124 ms | 0.126 ms | 1.02x |
| `mel_log_norm` | 7500x128 | 0.198 ms | 0.186 ms | 0.94x |
| `gemv` | 512x512 | 0.101 ms | 0.100 ms | 0.99x |
| `gemv` | 31678x512 | 0.290 ms | 0.244 ms | 0.84x |
| `gemv_batch` | 8x512x512 | 0.102 ms | 0.103 ms | 1.00x |
| `gemv_batch` | 8x31678x512 | 0.243 ms | 0.303 ms | 1.25x |
| `masked_argmax` | 31678 | 0.171 ms | 0.109 ms | 0.64x |
| `lm_head_argmax` | 31678x512 | 0.343 ms | 0.260 ms | 0.76x |

`silu_glu` and `mel_log_norm` are wired into the model; `gemv_batch` is
opt-in for the batched decoder.  The others are kept as standalone,
reference-checked kernels.  `lm_head_argmax` fuses the LM-head GEMV with the
grammar mask, but its reduction order differs from MLX matmul, so it can flip
near-ties; use `mx.argmax` when exact token parity matters.

## 4b. JIT backlog status

Implemented in `gemv_batch.metal` / `metal_ops.gemv_batch`:

- 64-bit vector load (`ushort4`, four bf16 per load);
- `SIMD_SUM_ACC` via one `simd_sum` per accumulator;
- double-buffer/prefetch of the next weight vector;
- a tiny shape-keyed autotuner over warps/threadgroup
  (`VELOX_METAL_TUNE=0` pins 8; `VELOX_GEMV_BATCH=1` enables it inside the
  batched decoder).

Microbench (B=8): N=2048/K=512 0.142 vs 0.164 ms, N=31678/K=512 0.261 vs
0.313 ms.  It is opt-in because the full 6-layer step regressed when the
kernel was substituted for every projection; keep it for isolated FFN/LM-head
experiments until the fused layer graph lands.

Still open: stream-K/split-K in Metal (CUDA already has them), WASP-style
warp specialization, ping-pong/interleave, memory interleave preprocessing,
and the generalized N-bank swizzle (32 banks x 4 B; 64-bit vectors; e.g.
BLOCK_SIZE_K=128*16 bits, 4x8 thread groups -> 32x128/(8x64)=8 rows, 2-way
conflicts at BLOCK_SIZE_M=16).

## 5. Harness

`tests/kernels/test_metal_ops.py` is the pytest gate:

```bash
python -m pytest -q tests/kernels/test_metal_ops.py
```

`benchmark/kernels/bench_metal_ops.py` is the microbench used for the table above:

```bash
python benchmark/kernels/bench_metal_ops.py
python benchmark/kernels/bench_metal_ops.py --ops silu_glu,mel_log_norm,gemv_batch \
    --json /tmp/metal_ops_bench.json
```

`tools/kernel_harness.py` ports the `flash-float-jit-kernels` loop to
MLX/Metal: 5-stage correctness (smoke, shape sweep, stability, determinism,
edge cases), SOL accounting, and JSON reports.

```bash
python tools/kernel_harness.py harness \
    --kernel veloxvoice/kernels/ops/metal_ops.py --op silu_glu --dtype bfloat16
python tools/kernel_harness.py harness \
    --kernel veloxvoice/kernels/ops/metal_ops.py --op fused_layernorm --dtype bfloat16
```

Correctness gate for every op:

```bash
for op in fused_layernorm silu_glu mel_log_norm gemv masked_argmax lm_head_argmax; do
  python tools/kernel_harness.py verify \
    --kernel veloxvoice/kernels/ops/metal_ops.py --op $op --dtype bfloat16
done
```

## 6. ANE bridge (M4 Pro, experimental)

`veloxvoice/runtime/ane/core.py` exposes a **unified MLX interface**:

```python
# one-time compile; caller passes only mx.array
path = convert_mlx_linear(weight, bias, example_input, "ffn_fc1.mlpackage")

# runtime; mx.array in, mx.array out
executor = ANEExecutor(path)
y = executor.predict_one(x)
```

Torch is only an implementation detail of the one-time Core ML compiler; the
MLX runtime lane never imports torch.  The hand-off is explicit:

```python
mx.eval(inputs)          # drain Metal GPU work
coreml.predict(...)      # synchronous ANE execution
mx.array(outputs)        # copy back to unified memory
```

A Core ML fp16 LM head (`512 -> 31678`) measured 251 us median / 197 us min
on ANE versus ~250 us on the GPU.  It is not a speedup by itself, but it frees
the GPU for the next encoder window; use it only when the GPU is the critical
resource.  For the M4 Pro experimental roofline, see §2c: offload FFN, keep
the LM head on the GPU.

For true overlap, use a **`spawn` process + shared memory**, not a Python
thread: Core ML `predict` holds the GIL, so the thread prototype serializes.
The process prototype reaches **1.13–1.36x** on large GEMMs with GPU-side
merge (§2c).  It is still a probe; the production path needs a fused split
that avoids the output merge.

## 7. The RTF 0.001 path (M4 Pro, experimental)

The M4 Pro paged path is at 0.018. 

The decoder step is still ~8.4 ms for B=8 versus a ~0.26 ms weight-read SOL.

TODO :

1. 4-bit (or 8-bit) decoder + LM-head weights: 4x less weight traffic;
2. migrating the fused 3-phase MLX graph (K1 cross, K2 GEMV+LN+FFN, K3 self-attn) 
   from dgx Spark / Hopper solution so the ~120 small matmuls per step 
   become a handful of kernels;
3. ANE offload of K2 (FFN) and the LM head, overlapping with the GPU;
4. paged encoder attention (local/windowed) instead of dense 7500-frame
   attention if the target is full-song streaming.

The dense 300 s encoder still has a ~0.73 s SOL at 9 TFLOPS; paged/local
attention or a coarser window plan is required to go below that.

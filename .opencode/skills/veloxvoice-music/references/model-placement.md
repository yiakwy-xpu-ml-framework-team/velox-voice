# Model and tool placement

VeloxVoice is a JIT-kernel project for DGX Spark GB10. Keep the boundary
between **first-party optimized inference** and **external demo tools** sharp.

## Decision table

| Kind | Location | Commit weights? | Examples |
| --- | --- | --- | --- |
| External demo runner, not optimized | `tools/music/` | No | Demucs, UVR/`audio-separator`, ffmpeg helpers |
| Optional demo dependency | `requirements/vocal-separation.txt` | n/a | `demucs==4.0.1` |
| First-party model code | `veloxvoice/models/<name>/` | No (weights external) | optimized separator adapter, WeNet model |
| GB10 JIT/CUDA kernels | `veloxvoice/kernels/csrc/dgx/` | n/a | fused ops, custom kernels |
| Weights / checkpoints | external models root or cache | Never | SheetSage2, MERT-v2, WeNet, Demucs, BS-RoFormer |

Existing external model roots on the demo host:
`/home/yiakwy/workspace/Github/models/{SheetSage2,MERT-v2-FullSong,piano_transcription}`.
Demucs weights currently live in `~/.cache/torch/hub/checkpoints` (402 MB for
`htdemucs_ft`).

## Why

- **Licensing:** code licenses (MIT) and weight licenses differ; vendoring
  third-party checkpoints into the repo is a compliance and size problem.
- **Build boundary:** `veloxvoice/models` and `kernels/csrc/dgx` are part of
  the optimized JIT path; mixing unoptimized Python model code there makes
  benchmarks and packaging lie about what is optimized.
- **Reproducibility:** demo tools change quickly (UVR model names, torchaudio
  I/O, torchcodec). Pinning them in an optional requirements file is safer.
- **GB10 kernels:** only code we own/benchmark should enter `csrc/dgx`.

## Promotion checklist (tool → first-party model)

Only promote after the optimized implementation exists:

1. Implement model graph/loader in `veloxvoice/models/<name>/`.
2. Add/port kernels under `veloxvoice/kernels/csrc/dgx/`, with a Python op and
   a non-JIT fallback.
3. Add a backend selector in the adapter (`tools/music/separate_vocals.py`),
   preferring the optimized path; keep the external tool as fallback.
4. Add gates:
   - quality: SDR/bleedless against a small reference set;
   - parity: optimized vs reference output (max abs error / correlation);
   - speed: RTF and peak VRAM on GB10;
   - no-ASR/CPU fallback.
5. Add a model manifest with revision/hash and download instructions; weights
   stay outside git.
6. Update this reference and the music/ASR skills.

## Current vocal-separation recommendation

- **Fast local fallback:** Demucs v4 `htdemucs_ft` in `tools/music`.
- **Best public quality:** BS-RoFormer (2025.07 / 124-band), Mel-Band
  RoFormer, ensembles with SCNet XL IHF. Use UVR or `audio-separator`; keep
  them as external tools until we optimize a path.
- **For ASR:** a single high-bleedless vocals model is usually better than a
  large ensemble; then Whisper `large-v3` (faster-whisper) for timestamps.
- **Karaoke/lead+backing:** `mel_band_roformer_karaoke_becruily` or a second
  karaoke pass; do not pretend it is the same as plain vocal separation.

## Never commit

- Checkpoints (`*.th`, `*.ckpt`, `*.onnx`, `*.pth`) and model caches.
- Generated `vocals.wav` / `instrumental.wav` / `piano_*.wav`.
- User audio and session directories (`/tmp/veloxvoice_music_output`).

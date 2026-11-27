# Music tools (external, demo-only)

This directory is for **third-party or not-yet-optimized** audio tools used by
the VeloxVoice demo. It is intentionally separate from the JIT model tree.

## What lives here

| Path | Role |
| --- | --- |
| `tools/music/separate_vocals.py` | Demucs vocal/instrumental separation adapter |
| `requirements/vocal-separation.txt` | optional Demucs dependency pin |
| `veloxvoice/models/` | first-party **model code** (not third-party weights) |
| `veloxvoice/kernels/csrc/dgx/` | first-party CUDA/C++ kernels for DGX Spark GB10 |
| model cache (`~/.cache/torch/hub/checkpoints`) | downloaded weights, outside git |

**Rule:** a third-party separator stays a tool in `tools/music` until we own an
optimized implementation. Do not copy Demucs or BS-RoFormer weights/code into
`veloxvoice/models` or `veloxvoice/kernels/csrc/dgx` just to make the demo run.

## Vocal separation

Default: Demucs v4 fine-tuned `htdemucs_ft` (fast baseline, not SOTA).

```bash
python tools/music/separate_vocals.py song.mp3 -o /tmp/stems -d cuda
# -> vocals.wav, instrumental.wav, vocals.model
```

The server uses the same adapter from the lyrics lane:

```bash
curl -X POST -F "session=<session>" http://<host>:8000/v1/audio/music/lyrics
curl http://<host>:8000/v1/audio/music/lyrics/<session>
```

When the ASR checkpoint is not loaded (`/healthz` -> `"asr": false`) the job
still completes with `stems_only: true` and returns `vocals_url` /
`instrumental_url`; the web UI shows players and download links. Attaching an
LRC preserves those URLs.

### Current model choice

Demucs `htdemucs_ft` is deliberately the fallback because it is small, fast,
and easy to run on GB10. For maximum vocal purity, current public evaluations
favor **BS-RoFormer** (2025.07 / 124-band) and **Mel-Band RoFormer**, often
ensembled with SCNet XL IHF. Practical local runners are UVR or
`audio-separator`; a single high-bleedless BS-RoFormer checkpoint is usually
better for ASR than a large ensemble.

## Promotion to a first-party model

Only after we implement and benchmark our own optimized path:

1. Put model code/loader in `veloxvoice/models/<name>/`.
2. Put GB10 kernels in `veloxvoice/kernels/csrc/dgx/` and expose a JIT op.
3. Keep `tools/music/separate_vocals.py` as the external fallback; add a
   backend selector that prefers the optimized path.
4. Add a parity/quality/speed gate (SDR/bleedless, numerical parity, RTF).
5. Keep weights outside the repo; record revisions/hashes in a manifest.
6. Update `.opencode/skills/veloxvoice-music/` and `references/`.

## Notes

- `demucs==4.0.1` is pinned: 4.1.x pulls `sphn`, whose aarch64 source build
  can hang. The adapter saves stems with `soundfile` because recent
  torchaudio `save()` routes through torchcodec.
- Never commit generated stems, model checkpoints, or session audio.
- Instrumental tracks legitimately produce a near-silent `vocals.wav`; verify
  with SheetSage2's `Vocal` track before treating it as a bug.

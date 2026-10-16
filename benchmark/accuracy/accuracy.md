# Accuracy — VeloxVoice-opt

AISHELL-4 local set, 1277 utterances, reference = `text.revised` (human GT).
Metric of record: **corpus CER = total char edits / total reference chars**
(normalized, NFKC + punctuation/noise stripped).

## Current results (re-tested 2026-10-16, post e68b479)

`tools/bench_accuracy_local.py` (full 1277, dedup OFF, frontend `kaldi` /
`KaldiGpuFrontend`), reference `text.revised`:

| checkpoint | precision | lane | corpus CER | edits / refchars | exact |
|---|---|---|---|---|---|
| `final.pt` (default) | fp16 | greedy | 13.17 % | 3049 / 23146 | 339 |
| **`final.pt` (default)** | fp16 | **att_rescore (`RESCORE=1`)** | **13.02 %** | **3013 / 23146** | 358 |
| old asr sys (`text.old_recog`) | — | — | 13.03 % | — | — |

**att_rescore on the default `final.pt` weights beats the old runtime
(13.02 vs 13.03)** and is the accuracy lane of record; greedy stays the default
low-RTF lane. 

Precision is corpus-CER-neutral (±0.03 pp).

## What moves the metric (and what does not)

Measured over the full 1277 set during the investigation (history in git;
RTF/beam analysis in [`precision_analysis.md`](precision_analysis.md) §11–13):

| lever | effect | verdict |
|---|---|---|
| **kaldi-identical frontend** (`VELOXVOICE_FRONTEND=kaldi`, `features.py`/`KaldiGpuFrontend` rewrite: povey `**0.85`, kaldi mel banks `fmin=20`, per-frame DC removal + preemph, `snip_edges`) | **≈ +0.3 pp** (13.6x → 13.2x greedy, re-adjudicated below) | the dominant fix |
| **attention_rescoring** (WeNet `ctc_prefix_beam_search` + BiTransformerDecoder prefill; bundle decoder emits **logits** → `log_softmax`; `RESCORE=1`) | **≈ +0.15 pp** → **13.02 %** | the lane of record |
| checkpoint `final.pt` vs bundle | reversed under kaldi frontend: `final.pt` better by ~0.05 pp (was "bundle +0.09 pp" under the old frontend) | keep default `final.pt` |
| **CTC single collapse** (`dedup=False`; legacy double-collapse drops real doubled symbols 看看→看) | ≈ +0.2–0.8 pp vs double | keep |
| precision fp16 / bf16 / fp32 | ±0.03 pp (neutral, re-verified 2026-10-16) | free choice (`VELOXVOICE_PRECISION`) |
| JIT fused kernels / CUDA graphs / TF32 / CMVN file-vs-embedded | neutral (±0.03 pp) | keep all (speed-critical) |
| CTC prefix-beam alone (no rescoring) | worse than greedy | never use alone |
| ROVER fusion of 6 configs | worse (identical systematic errors) | dropped |


## Defaults

`VELOXVOICE_FRONTEND=kaldi` · checkpoint **`final.pt`** · decode `greedy` (→ `att_rescore` when
`RESCORE=1`, or per-request `rescore` field) · `VELOXVOICE_BEAM_BACKEND=jit_kernel`
· `VELOXVOICE_PRECISION=bf16` (api/server; the accuracy tool pins fp16) ·
`VELOXVOICE_JIT=1` · `VELOXVOICE_CMVN=embedded`.

## External testsets (supplement)

Four external sets scored strictly with the official `compute_wer.sh`
normalizer + char-level WER, att_rescore lane (2026-10-16; all improve vs the
10-11 greedy run) — details in
[`accuracy_external_testsets.md`](accuracy_external_testsets.md):

| dataset | lang | utt | corpus CER | exact |
|---|---|---|---|---|
| test_en | en | 1000 | 3.96 % (1465/37036) | 38.8 % |
| test_yue_legco_label | yue | 920 | 3.61 % (1012/28033) | 52.4 % |
| test_yue_phoenix_tv_label | yue | 978 | 2.88 % (677/23500) | 60.1 % |

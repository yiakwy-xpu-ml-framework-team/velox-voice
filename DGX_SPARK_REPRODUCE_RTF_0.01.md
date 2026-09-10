# How to reproduce VeloxVoice results

We reproduce on the dev box: **DGX-Spark (GB10, aarch64 Linux)** with the conda
env `velox`. Caution: realtime numbers vary with CPU/GPU load and clock state.

## 1. Environment setup (one time)

```bash
# toolchain (Hopper/sm_121a): nvcc 13.x at /usr/local/cuda, g++, cmake
# python env
source ~/miniconda3/etc/profile.d/conda.sh && conda create -n velox python=3.12 -y
source ~/miniconda3/etc/profile.d/conda.sh && conda activate velox

pip install apache-tvm-ffi ninja "triton>=3.6" pytest numpy
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"

# optional elsewhere (mlx, needed for metal runner tests):
pip install "mlx[cuda]"

cd VeloxVoice
pip install -e .
```

## 2. Sanity checks (test suites)

```bash
pytest tests/kernels/test_cuda_ops.py
```

## 3. Reproduce the 6-minute RTF result (streaming lane)

```bash
# (a) audio: wikimedia commons librivox chapter (~12 min; we only use 6)
curl -sL "https://upload.wikimedia.org/wikipedia/commons/1/1a/Librivox_-_House_on_the_Cliff_%281927%29_Chapter_1.mp3" -o /tmp/librivox.mp3


# (b) execution :
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd  )"

audio=/tmp/librivox.mp3
model=$ROOT/data/models/asr_model

python $ROOT/tools/bench_wenet.py \
         --model-dir $model --audio $audio --use-jit
```

## 4. Reproduce the NVFP4 (mxfp4) DGX-Spark GEMM result

```bash
python tools/kernel_harness.py verify dgx_mxfp4_gemm --shape 2048,2048,2048
python tools/kernel_harness.py bench  dgx_mxfp4_gemm --shape 2048,2048,2048 --iters 50
```

Expected: `verify PASS max|d|=0.000e+00`, and ~170–190 us per iter (~100 TFLOPS).
When harnessing env notes: the build sets `TVM_FFI_CUDA_ARCH_LIST=12.1a` per-build
and links `-lcuda`; do not special-export those for other kernels.

## 5. Pipeline customizations you'd naturally repeat

- Kafka streams at chunk 640 ms / 160 ms: `--chunk-frames 16 (10ms/frame)`.
- Metal path: on Apple Silicon, same `pip install -e .` → `Velox.load(model_dir)`
  auto-selects mlx backend; `use_graphs="metal-graph"`.
- Real WeNet weights: point `--model-dir` at a live wenet bundle (final.zip,
  units.txt, train.yaml, global_cmvn); API converts final.zip automatically via
  `tools/convert_wenet_to_mlx.py` for the MLX backend only.
```

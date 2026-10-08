# syntax=docker/dockerfile:1.7
#
# VeloxVoice ASR — SGLang-Omni-compatible server (Hopper / DGX Spark).
#
# The image ships the CUDA *devel* toolkit because VeloxVoice compiles its
# flash-float JIT kernels at first run (TVM-FFI `load_inline` -> nvcc). The
# parity/cache is persisted through VELOXVOICE_KERNEL_CACHE.
#
# Build (from the repo root, or via docker/run.sh):
#   docker build -f docker/Dockerfile -t veloxvoice-asr:latest .
#
# Overridable build args (defaults follow requirements/requirements-cuda.txt):
#   BASE_IMAGE   nvidia/cuda:13.0.0-devel-ubuntu24.04   CUDA 13 toolkit + py3.12
#   TORCH_INDEX  https://download.pytorch.org/whl/cu130
#   (CUDA 12.8 fallback: BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
#                        TORCH_INDEX=https://download.pytorch.org/whl/cu128)

ARG BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

# Runtime defaults — all overrideable at `docker run` / compose level.
# PIP_BREAK_SYSTEM_PACKAGES: Ubuntu 24.04 marks the system Python as
# externally managed (PEP 668); inside this image the global site-packages is
# exactly where we want the deps, so opt out once here.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VELOX_GPU=0 \
    VELOX_HOST=0.0.0.0 \
    VELOX_PORT=8000 \
    VELOX_MODEL_DIR=/models/asr_model \
    VELOXVOICE_KERNEL_CACHE=/var/cache/veloxvoice

# System deps: python3.12 (from the Ubuntu base), ffmpeg (live ws decode),
# build-essential/curl for JIT + health probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Python
COPY requirements/ /tmp/requirements/
RUN python -m pip install --index-url "${TORCH_INDEX}" torch torchaudio \
 && python -m pip install -r /tmp/requirements/common.txt \
 && python -m pip install \
        "fastapi" "uvicorn[standard]" "python-multipart" \
        "aiohttp" "requests" "soundfile"

# Application
WORKDIR /app
COPY docker/entrypoint.sh /usr/local/bin/veloxvoice-entrypoint
COPY . /app
RUN chmod +x /usr/local/bin/veloxvoice-entrypoint \
 && mkdir -p /app/logs/user/output /app/logs/user_data /var/cache/veloxvoice

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=10 \
    CMD curl -fsS "http://127.0.0.1:${VELOX_PORT}/health" || exit 1

ENTRYPOINT ["veloxvoice-entrypoint"]

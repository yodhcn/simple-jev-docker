# ---------------------------------------------------------------------------
# simple-jev runtime image (CUDA 12.4)
#
# Base image is NVIDIA's official CUDA 12.4.1 runtime image, so this image is
# built for GPU inference only -- there is deliberately no CPU variant.
#
# This image ships ONLY the runtime: Python 3.12 + venv, PyTorch (cu124 build),
# Transformers and the simple-jev checkout itself.
#
#   * No model weights are downloaded, and no model directory is baked in.
#   * No model path is declared as a VOLUME -- you mount it yourself.
#   * The default command is `tail -f /dev/null`, so a freshly started
#     container stays alive and idle until YOU start the server.
#
# Typical usage (details in README.md):
#
#   docker run -d --name simple-jev --gpus all \
#     -p 8000:8000 \
#     -v /host/path/to/models:/models \
#     ghcr.io/<owner>/simple-jev-docker:latest
#
#   # ...then start the server inside the running container:
#   docker exec -it simple-jev simple-jev \
#     --model /models/Qwen3.5-0.8B \
#     --device cuda --dtype bfloat16 \
#     --host 0.0.0.0 --port 8000
#
# Upstream: https://github.com/featherless-ai/simple-jev
# ---------------------------------------------------------------------------

# NVIDIA ships no Ubuntu 24.04 build for CUDA 12.4 (that started with 12.5.1),
# so the Ubuntu family here is 22.04 and Python 3.12 comes from the deadsnakes
# PPA further down. Other 12.4.1 flavors are available, e.g.
# nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 (larger, bundles cuDNN/NCCL that
# the PyTorch wheel already brings with it).
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# --- Build-time knobs ------------------------------------------------------
# Python for the venv. Ubuntu 22.04 ships 3.10, which is below the upstream
# `requires-python >=3.12`, hence the PPA.
ARG PYTHON_VERSION=3.12
# Upstream source to build against.
ARG SIMPLE_JEV_REPO=https://github.com/featherless-ai/simple-jev.git
# Branch, tag or commit SHA. Pin a tag or SHA if you need reproducible images.
ARG SIMPLE_JEV_REF=main
# Optional extras of the hf-server package, e.g. "laya". Empty by default.
ARG SIMPLE_JEV_EXTRAS=
# PyTorch comes from the CUDA 12.4 wheel index so the framework and the base
# image's CUDA runtime agree. The cu124 index tops out at torch 2.6.0, which is
# exactly the upstream minimum (`torch>=2.6`).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# Optional pin, e.g. TORCH_VERSION=2.6.0. Empty means "newest on that index".
ARG TORCH_VERSION=

# HF_HOME gives Hugging Face a predictable cache location: mount a volume at
# /hf-cache to reuse downloaded weights across container restarts, or leave it
# inside the container if you only ever pass local model directories.
# LD_LIBRARY_PATH is deliberately NOT touched -- the NVIDIA base image points it
# at its own CUDA paths and overriding it would break GPU access.
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    CHECKOUT=/opt/simple-jev \
    HF_HOME=/hf-cache \
    PATH=/opt/venv/bin:$PATH

# --- System packages + Python ----------------------------------------------
# apt and the deadsnakes PPA live in one layer so the package lists are thrown
# away as soon as they are no longer needed. git is kept in the image so the
# checkout can be updated in place.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      gnupg \
      software-properties-common; \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      "python${PYTHON_VERSION}" \
      "python${PYTHON_VERSION}-dev" \
      "python${PYTHON_VERSION}-venv"; \
    rm -rf /var/lib/apt/lists/*

# --- Virtual environment ---------------------------------------------------
# Everything runs from /opt/venv; the distro Python is only used to create it.
RUN set -eux; \
    "python${PYTHON_VERSION}" -m venv "${VIRTUAL_ENV}"; \
    mkdir -p "${HF_HOME}"; \
    python -c 'import sys; assert sys.version_info >= (3, 12), f"simple-jev requires Python 3.12+, found {sys.version}"'; \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Upstream source -------------------------------------------------------
# Fetched as a shallow single-ref checkout, so SIMPLE_JEV_REF may be a branch,
# a tag or a commit SHA. .git is dropped to keep the layer small.
RUN git init -q "${CHECKOUT}" \
 && git -C "${CHECKOUT}" remote add origin "${SIMPLE_JEV_REPO}" \
 && git -C "${CHECKOUT}" fetch -q --depth 1 origin "${SIMPLE_JEV_REF}" \
 && git -C "${CHECKOUT}" checkout -q --detach FETCH_HEAD \
 && rm -rf "${CHECKOUT}/.git"

# --- PyTorch ---------------------------------------------------------------
# The largest layer, and the one that ties the image to CUDA 12.4. --index-url
# (not --extra-index-url) is required: an extra index would let pip prefer the
# newer, differently-linked torch build published on PyPI instead.
RUN set -eux; \
    SPEC="torch"; \
    if [ -n "${TORCH_VERSION}" ]; then SPEC="torch==${TORCH_VERSION}"; fi; \
    pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "${SPEC}"

# --- simple-jev ------------------------------------------------------------
# Editable install is the layout upstream documents: hf_server.py imports the
# sibling `common/` package, so the checkout must stay in place. The
# non-editable fallback is supported upstream as well -- that wheel bundles
# `common/` and resolves it through normal imports.
#
# torch is already installed and satisfies the upstream `torch>=2.6` floor, so
# pip leaves it alone here and resolves the remaining deps from PyPI.
RUN pip install --no-cache-dir -e "${CHECKOUT}/hf-server${SIMPLE_JEV_EXTRAS:+[$SIMPLE_JEV_EXTRAS]}" \
 || pip install --no-cache-dir    "${CHECKOUT}/hf-server${SIMPLE_JEV_EXTRAS:+[$SIMPLE_JEV_EXTRAS]}"

# --- Build-time self check -------------------------------------------------
# Fail the build on a mismatch instead of shipping a silently wrong image.
# Nothing here loads weights: `--help` exits during argument parsing, and it
# still has to import hf_server and the shared `common` package first.
RUN set -eux; \
    python -c "import torch, transformers, fastapi, uvicorn, common; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda, '| transformers', transformers.__version__)"; \
    python -c "import torch; assert torch.version.cuda == '12.4', f'expected a CUDA 12.4 torch build, got {torch.version.cuda}'"; \
    simple-jev --help > /dev/null; \
    test -f "${CHECKOUT}/common/PROMPT_STRUCTURE_V1.md"

WORKDIR ${CHECKOUT}

# CUDA 12.4 runtime lives in the base image; the server listens on port 8000.
EXPOSE 8000

# The NVIDIA base image ships an ENTRYPOINT (/opt/nvidia/nvidia_entrypoint.sh).
# Clearing it keeps the contract simple: the container runs exactly the command
# you give it, so `docker run <image> <your-command>` replaces our CMD outright.
ENTRYPOINT []

# Keep the container alive with nothing running. Replace this command when you
# actually want to serve a model, either at `docker run` time or with a
# `command:` override in docker compose.
CMD ["tail", "-f", "/dev/null"]

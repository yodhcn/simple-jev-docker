# ---------------------------------------------------------------------------
# kev runtime image (CUDA base, GPU only)
#
# Packages https://github.com/jaredpalmer/kev -- a family of small Jev-like
# decision models (a Qwen3.5 base + a rank-16 LoRA adapter + a pointer head)
# served over a TypeSafe-compatible /v1/systemone API.
#
# The image ships ONLY the runtime: Python 3.13, a virtualenv built from
# upstream's uv.lock, and the kev checkout itself.
#
#   * No model weights are downloaded and no checkpoint is baked in.
#   * A kev checkpoint is an adapter directory containing head.pt; you mount it
#     and pass --run. The Qwen3.5 base that head.pt names is resolved through
#     the Hugging Face cache, not from the image.
#   * The default command is `tail -f /dev/null`, so a container starts idle and
#     stays alive until YOU start the server.
#
# Typical usage (details in README.md): the image is built in GitHub Actions and
# shipped as an artifact (a docker-save tar.gz), never pushed to a registry.
#
#   docker load -i kev-latest-amd64.tar.gz
#
#   docker run -d --name kev --gpus all \
#     -p 8009:8009 \
#     -v /host/kev-4b:/models/kev-4b \
#     -v kev-hf-cache:/hf-cache \
#     kev:latest
#
#   # ...then start the server inside the running container:
#   docker exec -it kev kev-serve --run /models/kev-4b --port 8009
#
# Upstream: https://github.com/jaredpalmer/kev
# ---------------------------------------------------------------------------

# NVIDIA ships no Ubuntu 24.04 build for CUDA 12.4 (that started with 12.5.1),
# so the Ubuntu family here is 22.04 and Python 3.13 comes from the deadsnakes
# PPA below.
#
# Heads up on what this base image does and does not do: uv.lock resolves torch
# from PyPI, and PyPI's Linux torch wheel bundles its own CUDA user-space
# libraries (the nvidia-*-cu12 packages). So torch does NOT run against this
# image's CUDA -- the base only supplies the toolkit layout and the driver mount
# points. What constrains your host is the driver, not this tag; see
# README「CUDA 版本与基镜像」 if you would rather align the two.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# --- Build-time knobs ------------------------------------------------------
# Matches upstream's .python-version. The lockfile accepts >=3.12,<3.14, and
# Ubuntu 22.04 ships 3.10, hence the PPA.
ARG PYTHON_VERSION=3.13
# uv is the thing that reads uv.lock.
ARG UV_VERSION=0.12.18
# Upstream source to build against.
ARG KEV_REPO=https://github.com/jaredpalmer/kev.git
# Branch, tag or commit SHA. Pin a tag or SHA if you need reproducible images.
ARG KEV_REF=main
# flash-linear-attention carries the Gated DeltaNet kernels transformers uses
# for the Qwen3.5 hybrid backbones; without it transformers falls back to slow
# reference code. It is NOT in uv.lock, so it is installed separately -- set
# this to 0 if it ever fails to resolve.
ARG INSTALL_FLASH_LINEAR_ATTENTION=1
# What to install when the above is on. Left unpinned to match upstream, which
# does the same; pass a pinned spec (e.g. flash-linear-attention==0.5.2) if you
# need two builds of this image to be bit-identical.
ARG FLASH_LINEAR_ATTENTION_SPEC=flash-linear-attention

# HF_HOME gives Hugging Face a predictable cache location: mount a volume at
# /hf-cache to reuse the Qwen3.5 base across container restarts. LD_LIBRARY_PATH
# is deliberately NOT touched -- the NVIDIA base image points it at its own CUDA
# paths and overriding it would break GPU access.
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    CHECKOUT=/opt/kev \
    HF_HOME=/hf-cache \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# The one wrapper this repository adds to upstream. kev/serve.py hardcodes
# uvicorn's bind host to 127.0.0.1 and exposes no --host flag, which would make
# a `-p 8009:8009` container unreachable; the shim only overrides that one
# value. See the file itself for the details.
COPY kev-serve /usr/local/bin/kev-serve
RUN chmod +x /usr/local/bin/kev-serve

# --- System packages + Python ----------------------------------------------
# A compiler is kept because uv.lock records sdists alongside wheels: if any
# locked package has no wheel for cp313/linux, uv builds it and would otherwise
# fail. apt and the deadsnakes PPA live in one layer so the package lists are
# thrown away as soon as they are no longer needed.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
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

# --- uv --------------------------------------------------------------------
# uv gets its own venv on purpose. `uv sync` makes the project environment match
# uv.lock exactly, which means it prunes packages that are not in the lock -- if
# uv lived in /opt/venv it would delete itself halfway through the sync.
RUN set -eux; \
    "python${PYTHON_VERSION}" -m venv /opt/uv; \
    /opt/uv/bin/pip install --no-cache-dir "uv==${UV_VERSION}"; \
    ln -sf /opt/uv/bin/uv /usr/local/bin/uv; \
    uv --version

# --- Upstream source -------------------------------------------------------
# Fetched as a shallow single-ref checkout, so KEV_REF may be a branch, a tag or
# a commit SHA. .git is dropped to keep the layer small.
RUN git init -q "${CHECKOUT}" \
 && git -C "${CHECKOUT}" remote add origin "${KEV_REPO}" \
 && git -C "${CHECKOUT}" fetch -q --depth 1 origin "${KEV_REF}" \
 && git -C "${CHECKOUT}" checkout -q --detach FETCH_HEAD \
 && rm -rf "${CHECKOUT}/.git"

# --- Dependencies ----------------------------------------------------------
# uv.lock is the single source of truth for every version here; nothing is
# pinned or asserted a second time in this file.
#   --frozen              install the lock as written, never re-resolve
#   --extra serve         fastapi + uvicorn + typesafe-sdk (not installed by default)
#   --no-default-groups   skip the `dev` group (modal, pytest, matplotlib); it is
#                         not needed to serve
# uv creates /opt/venv itself from the interpreter we point it at. kev is
# installed from the checkout in editable mode, so ${CHECKOUT} has to stay.
RUN set -eux; \
    cd "${CHECKOUT}"; \
    uv sync --frozen --extra serve --no-default-groups \
      --python "/usr/bin/python${PYTHON_VERSION}"

# --- Gated DeltaNet kernels ------------------------------------------------
# Out of band because uv.lock does not carry it.
#
# How transformers actually reaches these kernels (verified against the pinned
# transformers 5.17.0, integrations/hub_kernels.py): the Qwen3.5 modeling code
# marks the delta-rule functions with
#
#   @use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
#
# which imports `fla` and resolves `ops.gated_delta_rule` inside it, i.e.
# `fla.ops.gated_delta_rule.chunk_gated_delta_rule`. Priority is: hub kernel,
# then this package, then a reference PyTorch path. The reference path is an
# order of magnitude slower and -- this is the trap -- transformers does NOT
# raise when it lands there, it logs a single warning. See the self check below.
#
# Installed bare, matching upstream's own Modal recipe
# (`uv_pip_install("flash-linear-attention", "triton>=3.7.1")`). That
# distribution requires only fla-core + einops + transformers>=4.45, none of
# which can move torch or triton. Asking for `flash-linear-attention[cuda]`
# instead would add torch>=2.7 / triton>=3.3 constraints: satisfied today, but
# they would let a future fla release drag torch off uv.lock.
#
# Two deliberate differences from upstream's Modal image:
#   * triton stays at the locked 3.4.0. Upstream forces triton>=3.7.1 solely to
#     dodge a gated-chunk *backward* bug in fla on Hopper (fla#640); this image
#     only serves (forward pass), so that workaround does not apply.
#   * causal-conv1d is not installed (upstream does not install it either), so
#     the short conv falls back to F.conv1d. That is a secondary cost next to
#     the delta-rule kernel itself.
#
# einops is the one package this adds that uv.lock does not list; fla-core
# requires it unconditionally.
RUN set -eux; \
    if [ "${INSTALL_FLASH_LINEAR_ATTENTION}" = "1" ]; then \
      uv pip install --python "${VIRTUAL_ENV}/bin/python" "${FLASH_LINEAR_ATTENTION_SPEC}"; \
    fi

# --- Build-time self check -------------------------------------------------
# Only what uv.lock cannot promise: that the environment actually imports, and
# that the server entry point boots. No version numbers are re-asserted -- the
# lock already fixed them, and a check here would be a second source of truth
# that drifts from upstream.
# Nothing below loads weights: `--help` exits during argument parsing, and
# kev.serve parses arguments before it touches a checkpoint.
#
# The fla check is the one that earns its keep here. Importing the bare `fla`
# package would prove nothing: what matters is whether
# `fla.ops.gated_delta_rule.chunk_gated_delta_rule` resolves, because when it
# does not, transformers quietly serves from a reference implementation that is
# roughly an order of magnitude slower. So import the exact symbol.
RUN set -eux; \
    python -c "import torch, transformers, peft, accelerate, numpy, kev; print('python', __import__('sys').version.split()[0], '| torch', torch.__version__, '(cuda', str(torch.version.cuda) + ')', '| transformers', transformers.__version__)"; \
    python -m kev.serve --help > /dev/null; \
    kev-serve --help > /dev/null; \
    test -f "${CHECKOUT}/kev/serve.py"; \
    if [ "${INSTALL_FLASH_LINEAR_ATTENTION}" = "1" ]; then \
      python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; assert chunk_gated_delta_rule.__module__.startswith('fla.'), chunk_gated_delta_rule.__module__; print('fla kernels:', chunk_gated_delta_rule.__module__)"; \
    fi

WORKDIR ${CHECKOUT}

# kev.serve's argparse default is 8008; upstream's README and examples use 8009.
EXPOSE 8009

# The NVIDIA base image ships an ENTRYPOINT (/opt/nvidia/nvidia_entrypoint.sh).
# Clearing it keeps the contract simple: the container runs exactly the command
# you give it, so `docker run <image> <your-command>` replaces our CMD outright.
ENTRYPOINT []

# Keep the container alive with nothing running. Replace this command when you
# actually want to serve a checkpoint, either at `docker run` time or with a
# `command:` override in docker compose.
CMD ["tail", "-f", "/dev/null"]

#
# The Olala 7A1B training + inference environment, as an image.
#
# BUILD FROM THE REPO ROOT, not from docker/ -- the build context is the repo
# and the COPYs below are root-relative:
#
#     cd /data/home/gaetan.caillaut/olala-env
#     docker build -f docker/Dockerfile -t olala-env:local .
#
# docker/build.sh does that and pushes, with commit-based tags.
#
# NO `# syntax=docker/dockerfile:1` DIRECTIVE ON PURPOSE, same as
# ../olala-vllm/docker/Dockerfile: docker on hippo is ROOTLESS, and the external
# dockerfile frontend is itself a container, so when it cannot start the failure
# is an unreadable "failed to run Build function: exit code: 1" pointing at line
# 1. The built-in frontend of docker 29.x supports everything used here
# (RUN --mount=type=cache, COPY --from=<image>).
#
# WHAT THIS IMAGE DOES NOT DO: it does not re-implement the install. Every step
# is `setup_olala_env.sh` invoked with ONLY=<n>, so there is exactly one
# description of how this env is built and the image cannot drift from what the
# dev box runs. The Dockerfile supplies only what a container has to supply on
# its own: the interpreter, a CUDA 12.x toolkit for the runtime JIT, the
# pruning, and the runtime contract.
#
# amd64 ONLY. The snapshot pins a flash-attn wheel by URL
# (flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl), so there is no arm64 build.
#
# ---------------------------------------------------------------------------
# WHY A SELF-CONTAINED VENV INSTEAD OF THE BASE IMAGE'S TORCH
#
# nvcr.io/nvidia/pytorch:26.05-py3 ships torch 2.12.0a0+5aff392 against CUDA
# 13.2. The env pins torch==2.11.0+cu128, and that pin is load-bearing:
#
#   - the vLLM fork is installed with VLLM_USE_PRECOMPILED=1, reusing the
#     upstream base commit's binaries. Those are built against torch 2.11's
#     C++ ABI and will not load under 2.12.
#   - the snapshot's flash-attn wheel is a torch2.11 build, pinned by URL.
#   - install/requirements.txt is deliberately NOT re-resolvable (numpy 2.4.6
#     vs mistral-common's <2.4), so it cannot be re-solved onto another torch.
#
# So `uv venv` here creates a venv WITHOUT system site-packages: the base
# image's torch in /usr/local/lib/python3.12/dist-packages is never on the path.
# What the base image is actually used for is everything around torch --
# python3.12, the NCCL 2.30 / OpenMPI / UCX / EFA / RDMA userspace that
# multi-node verl runs need, nsight, and a devel CUDA toolchain with a host g++.
#
# Two consequences are handled explicitly below, and both are silent failures if
# you forget them:
#   - the base image's LD_LIBRARY_PATH points at ITS torch/lib. Torch wheels
#     resolve libtorch/libc10 through DT_RUNPATH, which is searched AFTER
#     LD_LIBRARY_PATH, so leaving it in place shadows our libtorch with the 2.12
#     one. It is rewritten in the runtime stage.
#   - the base image sets PIP_CONSTRAINT=/etc/pip/constraint.txt, NVIDIA's
#     constraint file, which pins torch to the NGC build. It is cleared in both
#     stages.
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3

# A CUDA 12.x toolkit, because the base image only has 13.2 and the mamba3
# kernels JIT at RUNTIME through tilelang/CuTeDSL against a cu128 torch. This is
# not a build-time convenience: without a 12.x nvcc the image starts fine and
# then dies on the first forward pass.
#
# Measured on 26.05-py3: these six packages add 791 MB and land nvcc 12.8.93 in
# /usr/local/cuda-12.8. They install ALONGSIDE 13.2 -- /usr/local/cuda still
# points at 13.2 afterwards, which is why CUDA_HOME is set explicitly in the
# runtime stage rather than left to the symlink. setup_olala_env.sh's own
# auto-detection finds this tree too: it globs /usr/local/cuda-12.* and sorts
# -V -r, and apt's update-alternatives symlink /usr/local/cuda-12 has no dot so
# it does not shadow the real one.
#
# Deliberately not the full toolkit. If a JIT step ever asks for cuBLAS or
# cuRAND headers, add libcublas-dev-12-8 / libcurand-dev-12-8 here.
# Declared before the first FROM so both stages can default from it.
ARG CUDA12_HOME=/usr/local/cuda-12.8
ARG CUDA12_PKGS="cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 cuda-nvrtc-dev-12-8 cuda-driver-dev-12-8 cuda-profiler-api-12-8"


FROM ${BASE_IMAGE} AS builder

ARG CUDA12_HOME
ARG CUDA12_PKGS

# Hopper. Only consulted by build steps that compile CUDA here; the mamba3
# kernels are TileLang/Triton and target the live device at runtime, so this
# does NOT restrict which GPU the image can run on.
ARG TORCH_CUDA_ARCH_LIST=9.0a

# TRL is the one thing not in install/requirements.txt: the snapshot was frozen
# for verl GRPO/DAPO, which does not use it. 1.12.0 needs transformers>=4.56.2,
# accelerate>=1.4.0, datasets>=4.7.0, jinja2 and packaging -- all already in the
# snapshot at 5.8.0 / 1.14.0 / 5.0.0 / 3.1.6 / 25.0, which is why it can go in
# with --no-deps and move nothing else.
ARG TRL_VERSION=1.12.0

ENV DEBIAN_FRONTEND=noninteractive \
    APP=/opt/olala-env

# NVIDIA's pip constraint file pins torch to the NGC build; `uv pip sync` of the
# snapshot must not see it.
ENV PIP_CONSTRAINT=

# ---- 1. The CUDA 12.x toolkit ----------------------------------------------
# The keyring is REQUIRED, not a precaution: the base image ships CUDA 13.2 but
# carries no /etc/apt/sources.list.d entry for developer.download.nvidia.com, so
# without this there is no apt source that has a 12.8 to install. ubuntu2404 is
# right per the base image's org.opencontainers.image.version label (24.04
# noble).
RUN curl -4 -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${CUDA12_PKGS} \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb \
    && "${CUDA12_HOME}/bin/nvcc" --version

# ---- 2. uv, pinned to the version the dev box builds with ------------------
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

# UV_LINK_MODE=copy because UV_CACHE_DIR is a build cache mount below: uv
# hardlinks out of its cache by default, and those links would dangle once the
# mount goes away. UV_PYTHON_DOWNLOADS=never so a missing PYTHON_BIN fails the
# build instead of silently building against a downloaded interpreter.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/root/.cache/uv \
    PYTHONUNBUFFERED=1

# ---- 3. The build script and its inputs ------------------------------------
# Only these two files, so editing the README or docker/build.sh does not
# invalidate the ~1 h of layers below. OLALA_HOME is left at its default
# ($REPO/env) rather than moved to /opt/olala: serve.sh derives OLALA_HOME from
# its own location, so keeping the repo layout intact is what makes
# `/opt/olala-env/serve.sh` work in the container with no arguments.
WORKDIR ${APP}
COPY setup_olala_env.sh serve.sh ./

# CUDA_HOME is left unset on purpose: the script picks the highest
# /usr/local/cuda-12.* that has an nvcc, which is the tree installed above.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ---- 4. Run the install, one step per layer --------------------------------
# ONLY=<n> rather than FROM=1, because two of the ten steps cannot run in a
# build:
#   step 7  verifies a checkpoint loads. There is no checkpoint in the image --
#           it is a 13 GB volume -- so this is a runtime check, and
#           docker/verify.sh runs it against a mounted export.
#   step 9  installs dragon-agentic editable. Not this repo -- and note that
#           it would skip itself anyway, since it needs a pyproject.toml at
#           DRAGON_AGENTIC_DIR (which defaults to $REPO) and this repo has
#           none. Not run, so that is belt and braces rather than the reason.
# The step NUMBERS are therefore load-bearing here. If setup_olala_env.sh ever
# renumbers its steps, `ONLY=<n>` silently matches nothing and does nothing --
# which is exactly what docker/check_env.py, run in section 5, exists to catch.
RUN --mount=type=cache,target=/root/.cache/uv \
    ONLY=1 bash setup_olala_env.sh && ONLY=2 bash setup_olala_env.sh

# The big one: the venv plus the frozen snapshot (~20 GB of torch/vllm deps).
RUN --mount=type=cache,target=/root/.cache/uv \
    ONLY=3 bash setup_olala_env.sh

# vLLM from the Olala fork. Reuses the upstream base commit's precompiled
# binaries, so this is a python-only install, not a CUDA compile. It DOES use
# build isolation -- that is what the script does -- so it pulls torch 2.11 once
# more into a throwaway build env, which is where most of this layer's time
# goes. Rust is deliberately not installed: VLLM_USE_PRECOMPILED implies the
# precompiled Rust frontend too, and the pinned wheel provides it. If a future
# pin stops providing those artifacts, setup.py logs "falling back to local Rust
# build" and fails here -- loudly, which is the point.
RUN --mount=type=cache,target=/root/.cache/uv \
    ONLY=4 bash setup_olala_env.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    ONLY=5 bash setup_olala_env.sh

RUN ONLY=6 bash setup_olala_env.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    ONLY=8 bash setup_olala_env.sh

# ---- 5. TRL, and the check that this build is actually an env --------------
# --no-deps like everything after step 3: TRL's own floors are all satisfied by
# the snapshot, and letting it resolve would be free rein to move transformers
# or datasets off the frozen set.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${APP}/env/.venv/bin/python --no-deps "trl==${TRL_VERSION}"

# The build's smoke test, and it is not optional. Every install above is
# --no-deps into a venv assembled from a non-resolvable snapshot, so nothing
# checks coherence for us; and because the steps are driven by ONLY=<n>, a
# renumbered script would make them no-ops that still exit 0. This turns both
# into a failed build instead of a failed job.
COPY docker/check_env.py ${APP}/check_env.py
RUN cd /tmp && ${APP}/env/.venv/bin/python ${APP}/check_env.py

# ---- 6. Record the pins, then prune, inside the builder --------------------
# Both halves must happen HERE. A file deleted in a later layer of the final
# image still occupies its bytes in the layer that created it, so pruning after
# the stage boundary would save nothing.
#
# env/vllm-fork goes entirely: step 4 installed vllm NON-editable (a wheel built
# from that tree), so once site-packages has it the checkout is build detritus --
# several GB of it, counting build/ and the extracted precompiled .so files.
# env/scattermoe and env/renderers must STAY: scattermoe is wired in by a .pth
# file holding its absolute path, and renderers is an editable install, so both
# are live code at that exact path. env/olala stays too -- serve.sh reads its
# parsers/, and it carries install/ and modeling_olala.py for reference.
#
# The SHAs are written out before the .git dirs go, so the image can still
# answer "what was this built from": `cat /opt/olala-env/PINS`.
COPY docker/write_pins.sh ${APP}/write_pins.sh
RUN bash ${APP}/write_pins.sh "${APP}" \
    && cat ${APP}/PINS \
    && cd ${APP}/env \
    && rm -rf vllm-fork .cache \
    && rm -f vllm-base-*.whl \
    && rm -rf olala/.git scattermoe/.git renderers/.git

# compileall so the first import in a pod is not a byte-compile of the whole
# venv. -j 0 uses every core; single-threaded over all of torch is glacial.
# `|| true` because one unparseable file in some dependency must not fail a
# build that is otherwise complete.
# The two trees outside the venv are included on purpose: scattermoe is reached
# through a .pth and renderers is an editable install, so neither lives under
# .venv/lib and neither would be covered by compiling that alone.
RUN ${APP}/env/.venv/bin/python -m compileall -q -j 0 \
        ${APP}/env/.venv/lib ${APP}/env/scattermoe ${APP}/env/renderers || true


# ===========================================================================
# Runtime image
# ===========================================================================
# The same base, and the same *devel* CUDA on purpose: tilelang and CuTeDSL
# JIT-compile the mamba3 kernels on the first forward pass and need nvcc plus a
# host g++. A clean stage is what keeps the apt lists, the vLLM build tree, the
# precompiled base wheel and the .git dirs out of what gets pushed.
FROM ${BASE_IMAGE}

ARG CUDA12_HOME
ARG CUDA12_PKGS

ENV DEBIAN_FRONTEND=noninteractive \
    APP=/opt/olala-env

# Not a build-time leftover: this is the toolkit the runtime JIT uses.
RUN curl -4 -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${CUDA12_PKGS} \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

# One layer holding only the final state. THE DESTINATION PATH MUST MATCH THE
# BUILDER EXACTLY: env/renderers is an editable install and env/scattermoe is
# reached through a .pth file, and both record absolute paths.
COPY --from=builder ${APP} ${APP}

COPY docker/verify.sh ${APP}/verify.sh
RUN chmod 0755 ${APP}/verify.sh ${APP}/setup_olala_env.sh ${APP}/serve.sh

# ---- The runtime contract -------------------------------------------------
# CUDA_HOME is the 12.8 tree, not the base image's 13.2 /usr/local/cuda: the JIT
# has to match the cu128 torch. Both toolkits are present; only the default
# moves.
ENV CUDA_HOME=${CUDA12_HOME} \
    VIRTUAL_ENV=${APP}/env/.venv

# The venv first, then the 12.8 toolchain, then the base image's PATH minus its
# torch_tensorrt/bin entry -- that torch is not on the path any more.
ENV PATH=${APP}/env/.venv/bin:${CUDA12_HOME}/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin

# THE IMPORTANT ONE. The base image's LD_LIBRARY_PATH starts with
# /usr/local/lib/python3.12/dist-packages/torch/lib -- its own torch 2.12. Torch
# wheels find libtorch/libc10 via DT_RUNPATH, which the loader searches AFTER
# LD_LIBRARY_PATH, so leaving that entry in place makes our torch 2.11 load
# 2.12's libraries: an ABI mismatch that surfaces as an undefined symbol deep in
# an import, never as anything that names the cause. The two dist-packages
# entries are dropped; the driver and CUDA-compat ones are kept, they are what
# makes the node's driver reachable.
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Same reason as the builder: NVIDIA's constraint file pins torch to the NGC
# build, and anything installed in a running container must not see it.
ENV PIP_CONSTRAINT=

ENV PYTHONUNBUFFERED=1 \
    OLALA_HOME=${APP}/env \
    VLLM_CACHE_ROOT=/cache/vllm \
    TILELANG_CACHE_DIR=/cache/tilelang \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

# The JIT caches must not go into the image and must not go onto a shared RWX
# volume -- several replicas compiling into one directory is a race. An emptyDir
# or a tmpfs per pod is the right mount here.
RUN mkdir -p /cache /workspace && chmod 0777 /cache

# HF_HUB_OFFLINE is deliberately NOT set. setup_olala_env.sh calls it a property
# of the machine rather than of the install, and this image trains as well as
# serves, so `datasets` may legitimately need the Hub. serve.sh defaults it to 1
# by itself.
#
# Neither is CKPT: serve.sh defaults it to $REPO/checkpoints/sft, which is
# /opt/olala-env/checkpoints/sft here. Mount a raw HF export there, or pass
# CKPT=/model.

WORKDIR /workspace

# The base image's own entrypoint is inherited
# (/opt/nvidia/nvidia_entrypoint.sh): it checks the node's driver against the
# CUDA in the image, prints the banner, then execs this. Worth keeping -- a
# driver too old is then a message at startup instead of a crash later.
CMD ["/bin/bash"]

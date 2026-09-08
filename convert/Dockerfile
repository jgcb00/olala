# =============================================================================
# Megatron -> HuggingFace checkpoint conversion image.
#
# Thin layer on top of the training image (olala-sft, built by
# ../Dockerfile): adds the one runtime dependency the HF Dragon model needs
# that the training image doesn't have, and bakes in the NCCL path fix, so
# `docker compose run --rm convert` no longer pip-installs anything at start.
#
# Build (from this directory):
#   docker compose build convert
# or standalone:
#   docker build -t olala-sft-convert:latest .
#
# The base image is expected to exist locally (or be pullable); override with
#   docker compose build --build-arg BASE_IMAGE=<other> convert
# =============================================================================
ARG BASE_IMAGE=git.corp.linguacustodia.com:5050/gcaillaut/olala-sft:latest
FROM ${BASE_IMAGE}

# =============================================================================
# scattermoe: required by the HF Dragon model (dragon/modeling_dragon.py) for
# its MoE experts. Not on PyPI, hence the git install. --no-deps so it cannot
# pull in / downgrade the carefully built torch + triton of the base image.
# =============================================================================
RUN pip install --no-deps --no-cache-dir git+https://github.com/shawntan/scattermoe.git && \
    python -c "import scattermoe"

# =============================================================================
# Same fix as train_worker.sh on the cluster: DeepEP asserts the loaded NCCL is
# the pip-wheel one it was built against, but the image resolves the system
# libnccl first. Prepend (not replace) so the NGC base image's existing
# LD_LIBRARY_PATH is preserved -- ${LD_LIBRARY_PATH} below expands to the value
# inherited from the base image at build time.
# The path is verified here so a base-image Python bump fails the build instead
# of silently producing a container that dies inside DeepEP's assert.
# =============================================================================
RUN test -d /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
      || { echo "ERROR: NCCL wheel not at expected path; adjust LD_LIBRARY_PATH below" >&2; exit 1; }
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

# Everything is local; never touch the HF hub.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

#!/usr/bin/env bash
# Check this image, from inside a running container.
#
#   docker run --rm --gpus all <image> /opt/olala-env/verify.sh
#   docker run --rm --gpus all -v /path/to/export:/model <image> \
#       /opt/olala-env/verify.sh /model
#
# Two halves, because they need different things:
#
#   1. the env check (docker/check_env.py). No GPU, no weights, no network --
#      the same check the build ran, re-run here so that "the image is intact"
#      and "the build was green" are separate claims. Always runs.
#
#   2. step 7 of setup_olala_env.sh: load a real HF export and assert the 0-dim
#      GeodesicNorm params were widened to [1] AND still match the values on
#      disk. This is the step the Dockerfile cannot run -- a checkpoint is a
#      13 GB volume, not image content -- so it lives here. Runs only if you
#      point it at one.
#
# Exits non-zero on the first failure, so it is usable as a job's init check.
set -euo pipefail

APP=${APP:-/opt/olala-env}
CKPT=${1:-${CKPT:-}}
PY="$APP/env/.venv/bin/python"

[ -x "$PY" ] || { echo "ERROR: no interpreter at $PY -- is this the olala-env image?" >&2; exit 1; }

echo "== pins =="
cat "$APP/PINS" 2>/dev/null || echo "  (no PINS file)"

echo
echo "== env check =="
"$PY" "$APP/check_env.py"

echo
echo "== toolchain =="
# The 12.x nvcc is not decoration: tilelang and CuTeDSL compile the mamba3
# kernels on the first forward pass, and a missing or 13.x-only toolkit is a
# failure at that point rather than at startup.
echo "  CUDA_HOME    ${CUDA_HOME:-unset}"
if [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]; then
    echo "  nvcc         $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
else
    echo "  nvcc         MISSING at \$CUDA_HOME/bin/nvcc -- the mamba3 JIT will fail" >&2
    exit 1
fi
case "$("$CUDA_HOME/bin/nvcc" --version)" in
    *"release 12."*) ;;
    *) echo "  WARN: \$CUDA_HOME is not a 12.x toolkit; torch here is a cu128 build" >&2;;
esac
echo "  c++          $(c++ --version 2>/dev/null | head -1 || echo 'MISSING -- nvcc has no host compiler')"

# LD_LIBRARY_PATH must not reach the base image's own torch: its libtorch would
# shadow ours through the loader's search order and break the ABI silently.
case "${LD_LIBRARY_PATH:-}" in
    *dist-packages/torch*)
        echo "  ERROR: LD_LIBRARY_PATH still contains the base image's torch/lib:" >&2
        echo "         $LD_LIBRARY_PATH" >&2
        exit 1;;
    *) echo "  LD_LIBRARY_PATH clean of the base image's torch";;
esac

echo
echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /' || echo "  no GPU visible (run with --gpus all to check the CUDA path)"

if [ -n "$CKPT" ]; then
    echo
    echo "== checkpoint (step 7 of setup_olala_env.sh) =="
    [ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
    # Delegated to the script rather than reimplemented: that assertion is
    # subtle (it compares against the values in the safetensors file, because
    # transformers reinitialises shape-mismatched keys and reinitialised memory
    # passes a naive "not all ones" test while holding garbage).
    cd "$APP" && ONLY=7 CKPT="$CKPT" bash setup_olala_env.sh
else
    echo
    echo "== checkpoint =="
    echo "  skipped -- pass a raw HF export to check it:"
    echo "    docker run --rm --gpus all -v /path/to/export:/model <image> $APP/verify.sh /model"
fi

echo
echo "OK"

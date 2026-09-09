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
# tilelang and CuTeDSL compile the mamba3 kernels on the first forward pass, so
# a toolkit and a host g++ have to be here. WHICH toolkit is the open question
# this block exists to answer, rather than assert.
#
# The image ships only the base's CUDA 13.2, on the evidence that tilelang
# declares nvidia-cuda-nvcc>=13.0.48 and the snapshot already carries that
# compiler as wheels (nvidia-cuda-nvcc 13.2.86, nvvm, crt, tileiras) plus
# cuda-pathfinder, which resolves CUDA components out of site-packages before
# the system. An earlier build apt-added CUDA 12.8 alongside; that is gone.
#
# serve.sh and setup_olala_env.sh glob /usr/local/cuda-[0-9]* and take the
# newest, so the image and a dev box agree -- 13.2 here, 13.3 there. That is a
# NEW choice on both: every run to date used the 12.9 the old cuda-12.* glob
# picked, so nothing has yet JIT-compiled a mamba3 kernel against 13.x. If one
# ever fails with a compiler error, this is the first place to look.
echo "  CUDA_HOME    ${CUDA_HOME:-unset}"
if [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]; then
    echo "  nvcc (PATH)  $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')  at $CUDA_HOME"
else
    echo "  nvcc         MISSING at \$CUDA_HOME/bin/nvcc -- the mamba3 JIT will fail" >&2
    exit 1
fi
echo "  c++          $(c++ --version 2>/dev/null | head -1 || echo 'MISSING -- nvcc has no host compiler')"

# What the JIT stack itself will pick, which need not be the one on PATH.
"$PY" - <<'PYEOF'
import shutil
try:
    import cuda.pathfinder as cp
    print(f"  pathfinder   cuda.pathfinder {getattr(cp, '__version__', '(no __version__)')}"
          " -- CUDA components resolve from site-packages first")
except Exception as exc:
    print(f"  pathfinder   unavailable ({type(exc).__name__}) -- the JIT falls back to CUDA_HOME")
for mod in ("nvidia.cuda_nvcc", "nvidia.cuda_nvrtc", "nvidia.cuda_nvcc_cu12"):
    try:
        m = __import__(mod, fromlist=["__path__"])
        print(f"  wheel nvcc   {mod} at {list(m.__path__)[0]}")
    except Exception:
        pass
print(f"  nvcc on PATH {shutil.which('nvcc') or 'not found'}")
PYEOF

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

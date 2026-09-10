#!/usr/bin/env bash
# Megatron -> HF conversion (CPU path) WITHOUT the olala-sft docker image.
# The corp registry project gcaillaut/olala-sft is not readable by every
# account, and Megatron-LM refuses to import without transformer_engine, which
# only that image carries. The CPU path never instantiates a TE module, so an
# import-only stub (tests/te_stub) is enough. tyro is installed into a side dir
# so the frozen venv stays untouched. Verified 2026-09-10 on iter_0099518:
# audit bit-exact, 945/945 tensors identical to the reference export.
#
#   ITERATION=99518 LOAD_DIR=.../megatron SAVE_DIR=/where/to/put/it ./tests/mg2hf_cpu_nodocker.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(dirname "$HERE")
OLALA_HOME=${OLALA_HOME:-$REPO/env}; VENV=$OLALA_HOME/.venv
MG=${MEGATRON_LM_DIR:-/data/home/gaetan.caillaut/dragon-sft/7A1B/training/Megatron-LM}
TOK=${OLALA_TOKENIZER_DIR:-/data/home/gaetan.caillaut/dragon-sft/7A1B/tokenizers/tokenizer-channels-v4}
: "${LOAD_DIR:?parent of iter_XXXXXXX/}" "${SAVE_DIR:?output dir}" "${ITERATION:?iteration number}"
PYEXTRA=$OLALA_HOME/.pyextra
[ -d "$PYEXTRA/tyro" ] || uv pip install -q --python "$VENV/bin/python" --target "$PYEXTRA" tyro
mkdir -p "$SAVE_DIR"
cd "$OLALA_HOME/olala/convert"
PYTHONPATH=$HERE/te_stub:$PYEXTRA:$MG OLALA_PKG_DIR=$OLALA_HOME/olala MEGATRON_LM_DIR=$MG \
OLALA_TOKENIZER_DIR=$TOK HF_HUB_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} \
exec "$VENV/bin/torchrun" --standalone --nproc-per-node=1 load_mg_save_hf.py --cpu \
    --load-dir "$LOAD_DIR" --save_dir "$SAVE_DIR" --iteration "$ITERATION"

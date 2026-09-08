#!/usr/bin/env bash
# Olala checkpoint conversion, both directions, GPU or CPU.
#
#   ./convert.sh mg2hf          # Megatron -> HF, 1 GPU: converts AND verifies
#   ./convert.sh mg2hf cpu      # Megatron -> HF, no GPU: converts only
#   ./convert.sh hf2mg          # HF -> Megatron (CPU by design)
#   ./convert.sh                # same as: mg2hf
#
# Knobs (LOAD_DIR / SAVE_DIR / ITERATION / GPU / HF_DIR / REF_MG_DIR / ...)
# live in .env next to this file; start from .env.example. Override inline:
#
#   GPU=3 ITERATION=2400 ./convert.sh mg2hf
#
# The model code is taken from the repo root above this directory, so what the
# converter exports is exactly the modeling_olala.py this repo ships.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

[ -f .env ] || { echo "ERROR: no .env here — start from .env.example:" >&2
                 echo "         cp .env.example .env   # then edit" >&2; exit 1; }

# OLALA_FIXES_DIR must point at the repo root (the parent of this dir) so the
# container gets the model code. Default it rather than making people repeat it.
if ! grep -q '^OLALA_FIXES_DIR=' .env; then
    export OLALA_FIXES_DIR="$(cd .. && pwd)"
    echo ">> OLALA_FIXES_DIR not in .env; using $OLALA_FIXES_DIR"
fi

direction=${1:-mg2hf}
mode=${2:-gpu}

case "$direction" in
    mg2hf)
        case "$mode" in
            gpu) service=mg2hf ;;
            cpu) service=mg2hf-cpu ;;
            *)   echo "usage: $0 mg2hf [gpu|cpu]" >&2; exit 2 ;;
        esac
        ;;
    hf2mg)
        # CPU-only by design: TransformerEngine cannot build its modules
        # without CUDA, so the Megatron model is never instantiated and the DCP
        # store is written directly. A GPU would have nothing to do.
        [ "$mode" = gpu ] || [ "$mode" = cpu ] \
            || { echo "usage: $0 hf2mg [cpu]" >&2; exit 2; }
        [ "$mode" = gpu ] && echo ">> note: hf2mg is CPU-only by design; ignoring 'gpu'"
        service=hf2mg
        ;;
    -h|--help|help)
        sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "usage: $0 [mg2hf [gpu|cpu] | hf2mg]" >&2; exit 2 ;;
esac

# Create the destination up front: if docker has to create the missing host dir
# for a bind mount, it makes it root-owned.
set +u
source ./.env
set -u
case "$direction" in
    mg2hf) dest=${SAVE_DIR:-} ;;
    hf2mg) dest=${MG_SAVE_DIR:-} ;;
esac
if [ -n "$dest" ] && [ ! -d "$dest" ]; then
    echo ">> mkdir -p $dest"
    mkdir -p "$dest"
fi

echo ">> docker compose run --rm $service"
exec docker compose run --rm "$service"

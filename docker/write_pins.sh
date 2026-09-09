#!/usr/bin/env bash
# Write $APP/PINS -- what this image was built from -- and nothing else.
#
#   bash write_pins.sh /opt/olala-env
#
# Called by the Dockerfile in the layer that then deletes the .git directories
# and the vLLM build tree. That ordering is the reason this exists as a file:
# once those are gone the image can no longer answer "which commit of the fork
# is in here", and an image that cannot answer that is not reproducible. The
# venv keeps the answer instead, in one place: `cat /opt/olala-env/PINS`.
set -euo pipefail

APP=${1:?usage: write_pins.sh <APP dir>}
PY="$APP/env/.venv/bin/python"
OUT="$APP/PINS"

[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 1; }

{
    echo "# olala-env image -- built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## pinned source checkouts (git SHA)"
    # No `set -e` trap here: a missing .git is reported, not fatal. vllm-fork is
    # the one that legitimately may already be gone on a re-run of this script.
    for d in vllm-fork olala scattermoe renderers; do
        if [ -d "$APP/env/$d/.git" ]; then
            printf '%-12s %s\n' "$d" "$(git -C "$APP/env/$d" rev-parse HEAD)"
        else
            printf '%-12s %s\n' "$d" "(no .git)"
        fi
    done
    echo
    echo "## installed versions"
    "$PY" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "vllm", "verl", "transformers", "trl", "peft",
             "accelerate", "datasets", "ray", "tilelang", "transferqueue",
             "flashinfer-python", "flash-attn", "mamba-ssm", "renderers"):
    try:
        print(f"{name:20} {version(name)}")
    except PackageNotFoundError:
        print(f"{name:20} -")
PY
} > "$OUT"

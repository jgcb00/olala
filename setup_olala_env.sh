#!/usr/bin/env bash
# Build the Olala 7A1B verl training env (GRPO/DAPO, FSDP + vLLM hybrid
# engine) from scratch, on an x86_64 CUDA-12.x box with uv and a Hopper-or-newer GPU.
#
#   bash verl/setup_olala_env.sh                       # everything
#   FROM=6 bash verl/setup_olala_env.sh                # resume at step 6
#   ONLY=8 bash verl/setup_olala_env.sh                # just the renderer step
#   OLALA_HOME=/somewhere bash verl/setup_olala_env.sh # build a different env
#
# Idempotent: every step re-runs safely. Sources are pinned to commit SHAs, so the
# same script reproduces the same env elsewhere. Everything lands under $OLALA_HOME.
#
# This automates the Notion guide "Olala 7A1B — verl : installation et fixes",
# including the corrections in its appendix:
#   - `uv pip sync` uninstalls anything absent from the snapshot (vllm, dragon-agentic)
#   - HF_HUB_OFFLINE=1 is needed from install time, not just in the launcher:
#     HF_HUB_OFFLINE=1 bash verl/setup_olala_env.sh
set -euo pipefail

# =============================================================================
# CONFIG — everything machine-specific or remote lives here. Change these, not
# the body of the script. Each is overridable from the environment.
# =============================================================================

# ---- This repo -------------------------------------------------------------
# Run this from the project root; REPO is simply where you are.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(pwd)}

# ---- Where the env is built -------------------------------------------------
# ~40 GB+: the venv, the vllm wheel and the cloned sources. Inside the repo and
# gitignored, so the build travels with the checkout and `git status` stays
# clean. Deliberately NOT $REPO/../olala-env: this repo IS olala-env, so that
# default resolved to the repo itself and would bury a 40 GB build in git.
OLALA_HOME=${OLALA_HOME:-$REPO/env}
VENV=${VENV:-$OLALA_HOME/.venv}          # NOTE: .venv here, `venv` in the guide
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.12}

# ---- Pinned sources (SHAs, not branches: a moving branch changes the build) --
VLLM_FORK_URL=${VLLM_FORK_URL:-https://github.com/gcaillaut/vllm.git}
VLLM_FORK_REF=${VLLM_FORK_REF:-7bb4e2575ad43b111f65eec6f4367b479bf01c25}   # branch olala-v0.26
VLLM_FORK_BASE=${VLLM_FORK_BASE:-568afb3a13806beb53bb2e6bd518269357b237c0}  # upstream base for the precompiled wheel
VLLM_BASE_VERSION=${VLLM_BASE_VERSION:-0.26.0}                              # stamped via VLLM_VERSION_OVERRIDE
VLLM_WHEEL_VARIANT=${VLLM_WHEEL_VARIANT:-cu129}

# Mamba-3 comes from UPSTREAM state-spaces/mamba now, matching
# 7A1B/training/Dockerfile ("Mamba-3 capable release; supersedes the old
# tilelang-based build path"). The old jgcb00/mamba fork existed for the
# saved_tensors single-unpack fix, which is in upstream.
MAMBA_PIP=${MAMBA_PIP:-git+https://github.com/state-spaces/mamba}

SCATTERMOE_URL=${SCATTERMOE_URL:-https://github.com/shawntan/scattermoe.git}
SCATTERMOE_REF=${SCATTERMOE_REF:-47b5e15}

# The model code, the converter and the applier. This is also where the HF
# checkpoint's modeling_olala.py comes from, so the pin decides what a
# re-export ships.
OLALA_URL=${OLALA_URL:-https://github.com/gcaillaut/olala.git}
OLALA_REF=${OLALA_REF:-e9b89e541a199d84a8c43c75fe072c4e8702e2c7}   # feat/checkpoint-backports

# Build from a LOCAL olala checkout instead of cloning (handy while iterating:
# no push needed, and no risk of testing a stale remote). Empty = clone OLALA_URL.
OLALA_LOCAL=${OLALA_LOCAL:-}

# Must match the `renderers` pin used by dragon-agentic, or the olala renderer is
# developed against different internals than the rest of the env uses.
# Not sources, so no SHA -- both are ordinary PyPI wheels. Pinned regardless,
# because the point of this script is that a commit determines the environment.
TRL_VERSION=${TRL_VERSION:-1.12.0}
BITSANDBYTES_VERSION=${BITSANDBYTES_VERSION:-0.49.2}

RENDERERS_URL=${RENDERERS_URL:-https://github.com/PrimeIntellect-ai/renderers.git}
RENDERERS_REF=${RENDERERS_REF:-d4707862ac83aa3773c21f4096aec72bd17b91e4}

# dragon-agentic to install editable (empty = skip step 9). Defaults to THIS repo,
# so the env tracks the tree the launchers are run from.
DRAGON_AGENTIC_DIR=${DRAGON_AGENTIC_DIR:-$REPO}

# ---- The checkpoint to train from -------------------------------------------
# A RAW HF export, used IN PLACE. No patched copy: modeling_olala.py widens the
# 0-dim GeodesicNorm params itself after loading, so nothing is rewritten and no
# 13 GB duplicate exists. Produce one with olala/convert (Megatron -> HF); the
# default is where that guide tells you to put it.
CKPT=${CKPT:-$REPO/checkpoints/sft}

# ---- Toolchain --------------------------------------------------------------
# The mamba3 kernels JIT at runtime via tilelang/CuTe, so a CUDA toolkit must
# stay on PATH. Auto-pick the NEWEST installed; override with CUDA_HOME.
#
# Not "the highest 12.x", which is what this used to say on the theory that the
# JIT had to match torch's cu128 build. tilelang declares
# nvidia-cuda-nvcc>=13.0.48 -- the CUDA 13 line -- and install/requirements.txt
# already ships that compiler as wheels (nvidia-cuda-nvcc, nvvm, crt, tileiras,
# 13.x) alongside cuda-pathfinder, which looks in site-packages before the
# system. Kept identical to serve.sh on purpose: the two disagreeing about which
# toolkit the env uses is exactly the drift this repo exists to prevent.
#
# NOTE this value is NOT exported. It feeds the step-1 warning and the summary,
# nothing else -- no setup.py the install shells out to ever sees it.
if [ -z "${CUDA_HOME:-}" ]; then
    for d in $(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V -r); do
        [ -x "$d/bin/nvcc" ] && { CUDA_HOME=$d; break; }
    done
fi
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
FLASHINFER_INDEX=${FLASHINFER_INDEX:-https://flashinfer.ai/whl/cu128}
WHEELS_VLLM=${WHEELS_VLLM:-https://wheels.vllm.ai}

# Force IPv4 on downloads. urllib (vllm's setup.py) and pip prefer IPv6, which is
# black-holed on some hosts: the socket sits in SYN-SENT with no timeout and the
# install hangs silently. 1 = pre-download the wheel with `curl -4` instead.
FORCE_IPV4=${FORCE_IPV4:-1}

# NOTE: on a host with no usable route to the Hub, prefix the invocation with
# HF_HUB_OFFLINE=1 — it is inherited by every child process. Not defaulted here:
# it is a property of the machine, not of this install.

# =============================================================================
# Derived paths — no configuration below this line.
# =============================================================================
VLLM_DIR=$OLALA_HOME/vllm-fork
SCATTERMOE_DIR=$OLALA_HOME/scattermoe
# A local checkout wins over the clone, and must be resolved HERE rather than
# inside step 2: a resumed run (FROM=6) never enters step 2 and would
# otherwise fall back to the clone path and silently ignore OLALA_LOCAL.
if [ -n "$OLALA_LOCAL" ]; then
    FIXES_DIR=$(cd "$OLALA_LOCAL" && pwd) || die "OLALA_LOCAL not found: $OLALA_LOCAL"
else
    FIXES_DIR=$OLALA_HOME/olala
fi

# The verl/prime-rl renderer, shipped by the olala repo since e9b89e5.
OLALA_RENDERER_SRC=${OLALA_RENDERER_SRC:-$FIXES_DIR/renderers/olala.py}
RENDERERS_DIR=$OLALA_HOME/renderers
VLLM_BASE_WHEEL=${VLLM_BASE_WHEEL:-$OLALA_HOME/vllm-base-${VLLM_FORK_BASE:0:8}.whl}

FROM=${FROM:-1}
ONLY=${ONLY:-}

step() {
    local n=$1; shift
    if [ -n "$ONLY" ]; then [ "$ONLY" = "$n" ] || return 1
    else [ "$n" -ge "$FROM" ] || return 1; fi
    echo; echo "=== STEP $n: $* ==="; return 0
}
die() { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

site_packages() {
    [ -x "$VENV/bin/python" ] || die "no interpreter at $VENV/bin/python — run step 3 first"
    "$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
}

clone_pinned() {
    local dst=$1 url=$2 ref=$3
    if [ ! -d "$dst/.git" ]; then
        git clone -q --filter=blob:none "$url" "$dst" || die "clone failed: $url"
    fi
    git -C "$dst" fetch -q origin "$ref" 2>/dev/null || git -C "$dst" fetch -q origin
    git -C "$dst" checkout -q --detach "$ref" 2>/dev/null \
        || git -C "$dst" checkout -q "$ref" \
        || die "ref $ref not reachable in $dst — the branch may have been rewritten"
    printf '  %-12s %s\n' "$(basename "$dst")" "$(git -C "$dst" log --oneline -1)"
}

# -----------------------------------------------------------------------------
if step 1 "preflight"; then
    have git || die "git not found"
    have uv  || die "uv not found — curl -LsSf https://astral.sh/uv/install.sh | sh"
    [ -x "$PYTHON_BIN" ] || die "no interpreter at PYTHON_BIN=$PYTHON_BIN"
    [ -x "$CUDA_HOME/bin/nvcc" ] || echo "  WARN: no nvcc at $CUDA_HOME (runtime JIT needs a 12.x toolkit)"
    mkdir -p "$OLALA_HOME"
    echo "  REPO        $REPO"
    echo "  OLALA_HOME  $OLALA_HOME"
    echo "  VENV        $VENV"
    echo "  CUDA_HOME   $CUDA_HOME"
    echo "  CKPT        $CKPT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU         /' || true
fi

# -----------------------------------------------------------------------------
if step 2 "clone pinned sources"; then
    clone_pinned "$VLLM_DIR"       "$VLLM_FORK_URL"   "$VLLM_FORK_REF"
    clone_pinned "$SCATTERMOE_DIR" "$SCATTERMOE_URL"  "$SCATTERMOE_REF"
    if [ -n "$OLALA_LOCAL" ]; then
        echo "  olala        (local, not cloned) $FIXES_DIR"
    else
        clone_pinned "$FIXES_DIR" "$OLALA_URL" "$OLALA_REF"
    fi
    [ -f "$FIXES_DIR/install/requirements.txt" ] || die "no install/ kit in $FIXES_DIR — wrong branch?"
    [ -x "$FIXES_DIR/scripts/apply_olala_training_fixes.sh" ] || die "no applier in $FIXES_DIR/scripts"
fi

# -----------------------------------------------------------------------------
# `uv pip sync` (NOT install): the snapshot is deliberately not re-resolvable
# (numpy 2.4.6 vs mistral-common's <2.4). It is also EXACT — it uninstalls
# anything absent from the file, which is why vllm (step 4), renderers (step 9)
# and dragon-agentic (step 9) are installed after it, never before.
if step 3 "create venv + sync the frozen snapshot"; then
    [ -x "$VENV/bin/python" ] || uv venv "$VENV" --python "$PYTHON_BIN"
    uv pip sync --python "$VENV/bin/python" "$FIXES_DIR/install/requirements.txt" \
        --index-strategy unsafe-best-match \
        --extra-index-url "$TORCH_INDEX" \
        --extra-index-url "$FLASHINFER_INDEX"
    "$VENV/bin/python" -c "import torch,transformers,verl; print(f'  torch {torch.__version__} | transformers {transformers.__version__} | verl {verl.__version__}')"
fi

# -----------------------------------------------------------------------------
# The Olala port is Python-only, so the upstream base commit's precompiled
# binaries are reused — no CUDA build. VLLM_VERSION_OVERRIDE is REQUIRED: without
# upstream tags setuptools-scm stamps 0.1.dev..., and verl refuses vllm < 0.18.0.
# (SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM does NOT work — setup.py calls the
# setuptools-scm API without a dist name.)
if step 4 "install vllm (Olala fork, precompiled binaries)"; then
    if [ "$FORCE_IPV4" = "1" ] && [ ! -f "$VLLM_BASE_WHEEL" ]; then
        base=$WHEELS_VLLM/$VLLM_FORK_BASE
        name=$(curl -4 -sS "$base/$VLLM_WHEEL_VARIANT/vllm/metadata.json" \
               | "$VENV/bin/python" -c "import json,sys;print(next(w['filename'] for w in json.load(sys.stdin) if 'x86_64' in w['platform_tag'] and 'aarch' not in w['platform_tag']))") \
            || die "could not read wheel metadata from $base"
        echo "  downloading $name over IPv4"
        curl -4 -L --retry 3 --retry-delay 2 -# -o "$VLLM_BASE_WHEEL" \
            "$base/$(printf '%s' "$name" | sed 's/+/%2B/')" || die "wheel download failed"
    fi
    export VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=$VLLM_BASE_VERSION
    if [ -f "$VLLM_BASE_WHEEL" ]; then export VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_BASE_WHEEL
    else                                export VLLM_PRECOMPILED_WHEEL_COMMIT=$VLLM_FORK_BASE; fi
    uv pip install --python "$VENV/bin/python" --no-deps "$VLLM_DIR"
    (cd /tmp && "$VENV/bin/python" -c "
import vllm; from vllm.model_executor.models.registry import _VLLM_MODELS as R
assert vllm.__version__.startswith('0.'), vllm.__version__
assert 'OlalaForCausalLM' in R, f'OlalaForCausalLM not registered; registry has: {[k for k in R if \"lala\" in k or \"ragon\" in k]}'
print(f'  vllm {vllm.__version__} | OlalaForCausalLM registered: True')")
fi

# -----------------------------------------------------------------------------
# mamba3 kernels are TileLang/Triton — no CUDA build, a .pth is enough. The
# selective_scan stub is needed because the real extension targets the torch 2.9
# ABI; Olala never calls selective scan.
if step 5 "install mamba_ssm, wire scattermoe, drop the selective_scan stub"; then
    SP=$(site_packages)
    # --no-deps, like everything after step 3: mamba-ssm pins tilelang==0.1.8,
    # and letting it resolve would downgrade the snapshot's 0.1.9 (and drag
    # apache-tvm-ffi with it, which is what the Dockerfile's known-limitation
    # table is about).
    uv pip install --python "$VENV/bin/python" --no-deps --no-build-isolation "$MAMBA_PIP"
    echo "$SCATTERMOE_DIR" > "$SP/scattermoe.pth"
    cp "$FIXES_DIR/install/selective_scan_cuda.py" "$SP/"
    (cd /tmp && "$VENV/bin/python" -c "
import mamba_ssm, scattermoe, selective_scan_cuda, tilelang
print(f'  mamba_ssm {mamba_ssm.__version__} | scattermoe ok | stub ok | tilelang {tilelang.__version__}')")
fi

# -----------------------------------------------------------------------------
# Only scattermoe, transfer_queue and verl are patched. The vllm and mamba
# fixes are committed in their pinned refs, and the checkpoint needs nothing:
# ../convert/ builds the HF model from olala-fixes' own modeling_olala.py and
# save_pretrained() ships it, so an export already carries every fix.
if step 6 "patch the packages that still need it"; then
    "$FIXES_DIR/scripts/apply_olala_training_fixes.sh" \
        --python     "$VENV/bin/python" \
        --scattermoe "$SCATTERMOE_DIR"
fi

# -----------------------------------------------------------------------------
if step 7 "verify the checkpoint loads (raw export, widened in memory)"; then
    (cd /tmp && CKPT="$CKPT" "$VENV/bin/python" - <<'EOF'
import glob, os, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
ckpt = os.environ["CKPT"]

# Ground truth straight off the disk. Comparing against the FILE (not against
# "is it still 1.0") is the check that matters: transformers reinitialises any
# key it recorded as a shape mismatch, and reinitialised memory happily passes
# a "not all ones" test while holding garbage like 1.18e+30.
truth = {}
for shard in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if ".geodesic_" in k and k.rsplit(".", 1)[-1] in ("scale", "bias"):
                truth[k] = f.get_tensor(k).item()

m = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16)
got = {n: p for n, p in m.named_parameters()
       if ".geodesic_" in n and n.rsplit(".", 1)[-1] in ("scale", "bias")}

assert not [n for n, p in m.named_parameters() if p.dim() == 0], "0-dim params remain — FSDP will refuse them"
shapes = {tuple(p.shape) for p in got.values()}
assert shapes == {(1,)}, f"geodesic scalars not widened to [1]: {shapes}"
assert len(got) == len(truth) and truth, f"{len(got)} geodesic params vs {len(truth)} in the file"
bad = [n for n, p in got.items()
       if abs(p.float().item() - truth[n]) > max(1e-2, abs(truth[n]) * 0.01)]
assert not bad, f"{len(bad)} geodesic values do not match the checkpoint, e.g. {bad[:3]}"

tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
print(f"  {len(m.model.layers)} layers | {len(got)} geodesic scalars widened to [1], "
      f"all matching the file | vocab {len(tok)}")
EOF
    )
fi

# -----------------------------------------------------------------------------
# The olala chat template is channel-based (<|channel_start|>analysis<|content|>)
# with XML tool calls, which no upstream renderer understands: DefaultRenderer
# renders it byte-correctly but cannot parse a response or bridge a turn. The
# renderer is copied into the fork as renderers/olala.py, keeping the fork's diff
# to a single added file, and installed editable (this replaces the wheel).
if step 8 "install renderers + the olala renderer"; then
    # Hard error, not a skip: the olala repo ships this file, so its absence
    # means the checkout is wrong or too old. Skipping silently built an env
    # whose renderer could not parse a response -- and said nothing.
    if [ ! -f "$OLALA_RENDERER_SRC" ]; then
        die "no renderer at $OLALA_RENDERER_SRC (olala checkout too old? needs >= e9b89e5)"
    else
        clone_pinned "$RENDERERS_DIR" "$RENDERERS_URL" "$RENDERERS_REF"
        cp "$OLALA_RENDERER_SRC" "$RENDERERS_DIR/renderers/olala.py"
        grep -q 'import olala' "$RENDERERS_DIR/renderers/__init__.py" \
            || echo 'from renderers import olala  # noqa: E402,F401  self-registers "olala"' \
                 >> "$RENDERERS_DIR/renderers/__init__.py"
        uv pip install -q --python "$VENV/bin/python" --no-deps -e "$RENDERERS_DIR"
        (cd /tmp && "$VENV/bin/python" -c "
import renderers.olala  # noqa: F401  (self-registers 'olala')
from renderers.base import RENDERER_REGISTRY
print('  olala registered:', 'olala' in RENDERER_REGISTRY)")
    fi
fi

# -----------------------------------------------------------------------------
# --no-deps is mandatory: dragon-agentic pins vllm==0.24.0 and TransferQueue==0.1.7,
# which would drag the stack back off the snapshot and clobber the fork. Those
# pins stay recorded but unenforced, so `uv pip check` will flag them — expected.
if step 9 "install dragon-agentic (editable, --no-deps)"; then
    if [ -z "$DRAGON_AGENTIC_DIR" ] || [ ! -f "$DRAGON_AGENTIC_DIR/pyproject.toml" ]; then
        echo "  SKIP: no dragon-agentic at $DRAGON_AGENTIC_DIR"
    else
        uv pip install -q --python "$VENV/bin/python" --no-deps -e "$DRAGON_AGENTIC_DIR"
        (cd /tmp && "$VENV/bin/python" -c "
import dragon.agent_loop
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
print('  dragon registered in verl:', 'dragon' in _agent_loop_registry)")
    fi
fi

# -----------------------------------------------------------------------------
# The two libraries the frozen snapshot does not carry. It was frozen for verl,
# which needs neither, but the antidoom FTPO pipeline needs both:
#
#   trl            the DPOTrainer that antidoom's FTPOTrainer subclasses. The
#                  Dockerfile installed this itself, which meant a dev-box build
#                  had verl but no trl -- the two environments differed in a way
#                  nothing checked.
#   bitsandbytes   only for the paged optimizers. antidoom's config asks for
#                  `optim: paged_adamw_32bit`; without it HF Trainer raises at
#                  optimizer construction, after the model is already loaded.
#                  `--set train.optim=adamw_torch` avoids needing it at all.
#
# --no-deps, like everything after step 3. Both resolve fine against the
# snapshot -- trl's floors on transformers/datasets/accelerate are met, and
# bitsandbytes wants only torch and numpy -- but letting either resolve would be
# free rein to move the frozen set. check_env.py imports both, so a version that
# genuinely needs something absent fails the build rather than a job.
#
# LAST, deliberately. The Dockerfile drives this script with ONLY=<n>, and a
# step inserted earlier would renumber the ones after it: `ONLY=8` would then
# match nothing, exit 0, and silently produce an image with no renderers.
if step 10 "install trl + bitsandbytes (--no-deps)"; then
    uv pip install -q --python "$VENV/bin/python" --no-deps \
        "trl==${TRL_VERSION}" "bitsandbytes==${BITSANDBYTES_VERSION}" \
        || die "could not install trl/bitsandbytes"
    (cd /tmp && "$VENV/bin/python" -c "
import bitsandbytes, trl
print(f'  trl {trl.__version__}, bitsandbytes {bitsandbytes.__version__}')") \
        || die "trl/bitsandbytes installed but do not import"
fi

# -----------------------------------------------------------------------------
if step 11 "summary"; then
    (cd /tmp && "$VENV/bin/python" - <<'EOF'
from importlib.metadata import version, PackageNotFoundError
for n in ("torch","vllm","verl","transformers","trl","bitsandbytes","ray",
          "tilelang","transferqueue","flashinfer-python","flash-attn",
          "rl-insight","renderers","dragon-agentic"):
    try: print(f"  {n:20} {version(n)}")
    except PackageNotFoundError: print(f"  {n:20} —")
EOF
    )
    cat <<EOF

Env ready under $OLALA_HOME

  venv        $VENV
  checkpoint  $CKPT

Never run \`uv sync\` here: it would reinstall upstream vllm over the Olala fork
and downgrade transferqueue. After step 3, only \`uv pip install --no-deps\`.

Next — the launcher (guide step 8). Copy the example and adapt it:

  cp $FIXES_DIR/install/launch_Olala.example.sh $OLALA_HOME/launch_Olala.sh
  # set OLALA_HOME + the checkpoint path inside, provide the GSM8K parquets and
  # reward.py (calculator_reward_fn), then:
  #   export TMPDIR=\${TMPDIR:-/tmp/olala-\$(id -u)}; mkdir -p "\$TMPDIR"
  #   export CUDA_HOME=$CUDA_HOME VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
  #   ./launch_Olala.sh trainer.total_training_steps=2 trainer.resume_mode=disable

On 1 GPU, FSDP1 OOMs at the initial weight sync — add:
  actor_rollout_ref.{actor,ref}.strategy=fsdp2 \\
  actor_rollout_ref.{actor,ref}.fsdp_config.offload_policy=True \\
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \\
  actor_rollout_ref.rollout.max_num_seqs=128
EOF
fi

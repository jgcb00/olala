#!/usr/bin/env bash
# Serve an Olala checkpoint from the env this repo builds.
#
#   ./serve.sh                      # ./checkpoints/sft on :8010, GPU 0
#   PORT=8020 GPU=1 ./serve.sh
#   CKPT=/path/to/export ./serve.sh
#   ./serve.sh --max-num-seqs 16    # extra flags pass straight through
#
# Uses the venv from setup_olala_env.sh and the parsers from the olala repo it
# cloned, so what you serve matches what you trained against.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=${REPO:-$(pwd)}

OLALA_HOME=${OLALA_HOME:-$REPO/env}
VENV=${VENV:-$OLALA_HOME/.venv}
CKPT=${CKPT:-$REPO/checkpoints/sft}
PORT=${PORT:-8010}
GPU=${GPU:-0}
MAX_LEN=${MAX_LEN:-32688}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-43}
GPU_UTIL=${GPU_UTIL:-0.95}

# The olala checkout the env was built from (OLALA_LOCAL if you built that way).
OLALA_DIR=${OLALA_LOCAL:-$OLALA_HOME/olala}
PARSERS=$OLALA_DIR/parsers/olala

# ---- Preflight ------------------------------------------------------------
[ -x "$VENV/bin/vllm" ] || { echo "ERROR: no vllm in $VENV — run ./setup_olala_env.sh first" >&2; exit 1; }
[ -d "$CKPT" ]          || { echo "ERROR: checkpoint not found: $CKPT (set CKPT=...)" >&2; exit 1; }
for p in olala_reasoning_parser.py olala_tool_parser.py; do
    [ -f "$PARSERS/$p" ] || { echo "ERROR: missing $PARSERS/$p — is $OLALA_DIR the olala checkout?" >&2; exit 1; }
done

# ---- Runtime env ----------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}

# The mamba3 kernels JIT at RUNTIME via tilelang/CuTe, so a CUDA toolkit must be
# on PATH here, not just at build time. Take the NEWEST one installed.
#
# This used to glob /usr/local/cuda-12.* on the theory that the JIT had to match
# torch's cu128 build. It does not: tilelang declares nvidia-cuda-nvcc>=13.0.48
# -- the CUDA 13 package line -- and the frozen snapshot already ships that
# compiler as wheels (nvidia-cuda-nvcc, nvvm, crt, tileiras, all 13.x) plus
# cuda-pathfinder, which resolves CUDA components out of site-packages before
# the system. The cu12 half of the stack (torch cu128 and every nvidia-*-cu12
# library) is bundled wheels that consult no system toolkit at all.
#
# Pinning to 12.x also had a cost. LD_LIBRARY_PATH below is searched BEFORE a
# wheel's DT_RUNPATH, and a 12.x tree carries libcudart.so.12 / libcublas.so.12
# -- the same sonames the cu12 wheels provide -- so it shadowed the exact
# libraries torch was built against with a different minor. A 13.x tree bumps
# every soname, so nothing collides.
#
# [0-9] rather than a bare *, to skip non-version siblings like cuda-samples.
# `sort -V -r` puts a real 13.2 above the bare cuda-13 major symlink.
if [ -z "${CUDA_HOME:-}" ]; then
    for d in $(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V -r); do
        [ -x "$d/bin/nvcc" ] && { CUDA_HOME=$d; break; }
    done
fi
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Carried from the serving env, where flashinfer was absent. It IS installed
# here, so this may be unnecessary -- unset it and see. Kept only because it has
# not been re-validated on this env.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$OLALA_HOME/.cache/vllm}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$OLALA_HOME/.cache/tilelang}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# ---- Two checks that cost nothing and save an afternoon --------------------
if [ $(( MAX_LEN % 144 )) -ne 0 ]; then
    echo ">> WARNING: MAX_LEN=$MAX_LEN is not a multiple of 144 (see NOTES)."
    echo "            Nearest: $(( MAX_LEN / 144 * 144 )) or $(( (MAX_LEN / 144 + 1) * 144 ))."
fi
SAFE_SEQS=$(( 43 * 32688 / MAX_LEN ))
if [ "$MAX_NUM_SEQS" -gt "$SAFE_SEQS" ]; then
    echo ">> WARNING: MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds ~$SAFE_SEQS, what the KV cache holds"
    echo "            at MAX_LEN=$MAX_LEN. Expect preemption = full re-prefill (see NOTES)."
fi

echo ">> serving $CKPT"
echo ">> on      :$PORT (GPU $CUDA_VISIBLE_DEVICES, ctx $MAX_LEN, max-num-seqs $MAX_NUM_SEQS)"

exec "$VENV/bin/vllm" serve "$CKPT" \
    --served-model-name olala-7a1b \
    --trust-remote-code \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --reasoning-parser olala \
    --reasoning-parser-plugin "$PARSERS/olala_reasoning_parser.py" \
    --enable-auto-tool-choice \
    --tool-call-parser olala \
    --tool-parser-plugin "$PARSERS/olala_tool_parser.py" \
    --port "$PORT" \
    "$@"

# NOTES
#
# --max-model-len   KEEP IT A MULTIPLE OF 144. 32688 = 227 x 144, not a typo for
#                   32768. Olala's attention block_size resolves to 144, not a
#                   power of two: the attention page is padded up to the larger
#                   Mamba3 state page and that ratio is 9, so 16 x 9 = 144. With
#                   prefix caching off vLLM pins mamba_block_size =
#                   max_model_len, so scheduler_block_size = lcm(144, len) --
#                   at a power-of-two length the gcd is 16 and the LCM is NINE
#                   TIMES max_model_len. Inert today (only the KV-connector and
#                   prefix-cache paths read it), a landmine if either is enabled.
#
# --max-num-seqs    Pair it with --max-model-len. The KV cache holds ~43 requests
#                   at 32688 tokens (measured, GPU_UTIL 0.95, H100); capacity
#                   scales as 1/max_model_len. Above that you get preemption, and
#                   preemption here is a FULL re-prefill: mamba_cache_mode is
#                   "none", so there is no SSM state to resume from.
#
# Flags deliberately NOT passed, so nobody re-adds them as cargo:
#
#   --no-enable-prefix-caching   already the default for hybrid models
#                   (arg_utils.py: `is_prefix_caching_supported and not
#                   is_hybrid`), and it sat before "$@" so it could not even
#                   guard an override. Do NOT enable prefix caching: vLLM then
#                   sets mamba_cache_mode='align', which Olala rejects (M layers
#                   hold four state tensors, V layers two), and it would roughly
#                   HALVE concurrency by snapshotting mamba state per block.
#
#   --generation-config vllm     defended against an old checkpoint whose
#                   generation_config.json carried Megatron leftovers
#                   (bos/eos/pad = 1/2/0 -> the characters '"', '#', '!'), which
#                   truncated replies at the first '#'. Current exports are
#                   clean (eos 151645 = <|im_end|>), and vLLM only reads
#                   eos_token_id and sampling defaults from that file. Re-add it
#                   if you ever serve a pre-lrfix export.
#
#   M3_SCAN_BLOCK                nothing in the fork reads it.
#
#   OLALA_WINDOW_DECODE_RESET    the fork already defaults it to "0", and exports
#                   from iter_0090000 on carry artificial_seq_len=0, which gates
#                   every reset path off regardless. Setting it to 1 on such a
#                   checkpoint would inject SSM state clears the model never
#                   trained with -- a correctness risk, not a tuning knob.
#
# no --chat-template   vLLM reads chat_template.jinja from the checkpoint, and
#                   that template carries the tool-metadata fix from
#                   tokenizer-channels-v4 on. `metadata` is not an OpenAI field,
#                   so vLLM drops it, and the old unguarded
#                   {{ message.metadata | tojson }} raised "Object of type
#                   Undefined is not JSON serializable" -> HTTP 400 on EVERY
#                   tool-result turn. Pass it only for a pre-v4 checkpoint.

# From-scratch install of the Olala 7A1B verl training environment

Validated end-to-end on 2026-08-26 (hippo, CUDA 12.9, Python 3.12.3):
fresh env in `/data/home/j-g.barthelemy/olala`, 1-GPU and 2-GPU GRPO smoke runs.
The step-by-step guide (with troubleshooting table) lives on Notion:
"Olala verl training — install & fix guide".

Files here:

- `requirements.txt` — frozen snapshot of the validated environment
  (everything except vllm, which is installed from the `jgcb00/vllm` fork).
  Install with `uv pip sync` (NOT `install`: the snapshot is intentionally
  not re-resolvable — numpy 2.4.6 vs mistral-common's `<2.4` pin).
- `selective_scan_cuda.py` — import-only stub to drop into `site-packages`.
  The real extension targets the torch 2.9 ABI and fails to import under
  torch 2.11; Olala never calls selective scan, so an import stub that
  raises on actual use is correct.
- `launch_Olala.example.sh` — validated verl GRPO launcher (GSM8K). Adapt
  `OLALA_HOME` and the data/reward paths.

## Quick sequence

```bash
export OLALA_HOME=$HOME/olala
mkdir -p $OLALA_HOME && cd $OLALA_HOME
uv venv venv --python /usr/bin/python3.12

# pinned sources
git clone --filter=blob:none https://github.com/jgcb00/vllm.git  vllm-fork  && git -C vllm-fork  checkout 8a124b6b0
git clone --filter=blob:none https://github.com/jgcb00/mamba.git mamba      && git -C mamba      checkout 761b409
git clone --filter=blob:none https://github.com/shawntan/scattermoe.git scattermoe && git -C scattermoe checkout 47b5e15
git clone --filter=blob:none https://github.com/jgcb00/olala.git olala-fixes

# exact package snapshot
uv pip sync --python venv/bin/python olala-fixes/install/requirements.txt \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://flashinfer.ai/whl/cu128

# vllm fork, precompiled binaries (the Olala port is Python-only).
# VLLM_VERSION_OVERRIDE is REQUIRED: without upstream tags setuptools-scm
# stamps 0.1.dev..., and verl refuses vllm < 0.18.0 at import time.
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0 \
VLLM_VERSION_OVERRIDE=0.26.0 \
uv pip install --python venv/bin/python --no-deps ./vllm-fork

# mamba_ssm + scattermoe as path installs (TileLang/Triton kernels, no build)
SP=$OLALA_HOME/venv/lib/python3.12/site-packages
echo "$OLALA_HOME/mamba"      > $SP/mamba_public.pth
echo "$OLALA_HOME/scattermoe" > $SP/scattermoe.pth
cp olala-fixes/install/selective_scan_cuda.py $SP/

# checkpoint: use a RAW export as it is. No pre-patching step any more --
# modeling_olala.py widens the 144 0-dim GeodesicNorm params to [1] itself
# after loading (FSDP rejects scalar parameters) and writes them back 0-dim on
# save, so the on-disk shape never changes.
CKPT=/data/home/gaetan.caillaut/dragon-sft/7A1B/training/checkpoints/65k-betterpacks-lrfix/huggingface/iter_0099518

# apply every training fix (idempotent)
./olala-fixes/scripts/apply_olala_training_fixes.sh \
  --python $OLALA_HOME/venv/bin/python \
  --mamba $OLALA_HOME/mamba \
  --scattermoe $OLALA_HOME/scattermoe \
  --checkpoint $OLALA_HOME/patched_checkpoint
```

Then copy `launch_Olala.example.sh`, set `OLALA_HOME` inside, provide the
GSM8K parquets + `reward.py`, and run. Verify with
`trainer.total_training_steps=2`.

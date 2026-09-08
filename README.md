# olala-fixes

Model code, checkpoint conversion, and the few third-party patches the Olala
7A1B stack still needs.

The whole point of this repo's layout is that there is **one** copy of
`modeling_olala.py` and it flows outward: the converter builds the HF model from
it, `save_pretrained()` ships it into the export, and everything downstream
reads it from there. Nothing patches a checkpoint after the fact.

## The path from a Megatron checkpoint to a served model

```bash
# 1. Megatron -> HF. Exports with THIS repo's modeling_olala.py.
cd convert && cp .env.example .env      # set LOAD_DIR / SAVE_DIR / ITERATION
./convert.sh mg2hf                      # 1 GPU: converts AND verifies
./convert.sh mg2hf cpu                  # or: no GPU, weight audit only

# 2. Build the verl training env, pointing CKPT at that raw export.
CKPT=/path/to/huggingface/iter_0099518 bash /path/to/setup_olala_env.sh

# 3. Serve it.
cd /data/home/gaetan.caillaut/olala-vllm && ./serve_olala_7a1b.sh
```

No step between them. The export is used **in place** — there is no patched
copy, no 13 GB duplicate, and no checkpoint rewrite.

The reverse direction, for weights coming back from post-training:

```bash
cd convert && ./convert.sh hf2mg         # CPU by design; needs a reference MG ckpt
```

See [convert/README.md](convert/README.md) for what that reference is for.

## Why nothing patches the checkpoint

`modeling_olala.py` handles the two things that used to need it:

* **FSDP and 0-dim parameters.** FSDP1/FSDP2 both refuse scalar parameters, but
  every export stores the 144 `GeodesicNorm` scale/bias tensors 0-dim.
  `from_pretrained()` widens them to `[1]` in memory and `save_pretrained()`
  writes them back 0-dim, so the on-disk shape never changes and no rewrite is
  involved. (`widen_geodesic_scalars()` is public if you build the model another
  way.)

  Declaring `[1]` directly does not work and is not safe: transformers records a
  shape mismatch and then **reinitialises** every mismatched key before any
  model hook runs, replacing trained values with garbage.

* **Kernels.** The Mamba-3 chunk kernel comes from the standard
  state-spaces/mamba package and is a hard requirement, so a missing one fails
  at import with a named error rather than deep inside `forward()`. The CuteDSL
  decode kernels stay optional — without them the model still runs with
  `use_cache=False`.

## What still gets patched, and why

`scripts/apply_olala_training_fixes.sh` — six edits, all to packages we do not
control:

| package | fix |
|---|---|
| `scattermoe` | per-module dtype cast of inputs/gates, fwd + bwd (FSDP bf16). Not on PyPI, pinned by commit, no upstream fix |
| `transfer_queue` | one malformed ZMQ frame must not kill a storage worker permanently (storage + controller) |
| `verl` | the colocate weight-sync ZMQ socket hardcodes a shared `/tmp` path, so a stale socket from another user blocks every run. Use `TMPDIR` on both sides |

```bash
./scripts/apply_olala_training_fixes.sh \
    --python     /path/to/venv/bin/python \
    --scattermoe /path/to/scattermoe/clone
```

Idempotent; every modified file gets a one-time `.bak-olala-fixes` backup.

Four fixes that used to live here are gone because the need went away, not
because it moved:

* **vllm** `olala/mamba3.py` and `models/olala.py` — both are committed in the
  vllm fork at the pinned ref.
* **mamba** `mamba3_mimo.py` and `mamba3_siso_combined.py` — the `saved_tensors`
  single-unpack fix is upstream in the pinned ref (`761b409`).
* **checkpoint** `modeling_olala.py` — the converter ships this repo's copy.
* **checkpoint weights** — see above.

## Layout

| path | role |
|---|---|
| `modeling_olala.py`, `configuration_olala.py` | the model. One copy; the converter exports it |
| `convert/` | Megatron ⇄ HF, GPU or CPU, both directions |
| `scripts/apply_olala_training_fixes.sh` | the six third-party patches |
| `install/` | frozen requirements, the `selective_scan_cuda` stub, launcher example |
| `training_olala.py`, `compute_loss.py`, `optimizers/` | training-side code |
| `inspecting_olala.py`, `coordchecking_olala.py`, `coordcheck_utils.py` | analysis |

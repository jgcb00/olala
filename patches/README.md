# Third-party patches

Patches for dependencies we don't own, applied automatically at environment
setup time.

## scattermoe-dtype-cast.patch

Target: [shawntan/scattermoe](https://github.com/shawntan/scattermoe)
(commit `47b5e15` or later).

The scattermoe triton kernels (`scatter2scatter`) require activations, gates
and expert weights to share a dtype (`tl.dot` asserts otherwise). Under FSDP
bf16 mixed precision over fp32 master weights (verl's default with
`model_dtype=fp32`), Dragon's modeling code feeds fp32 activations into bf16
compute params — and the two `ParallelExperts` of one MLP can even see
different compute dtypes depending on how their params were wrapped. The patch
makes each `ParallelExperts.forward` cast its `inputs`/`gates` to its own
`weight.dtype`; a no-op whenever dtypes already match (pure-bf16 inference is
unaffected).

Auto-apply after cloning scattermoe (idempotent — skips if already applied):

```bash
cd "$SCATTERMOE_DIR"
git apply --check /path/to/olala/patches/scattermoe-dtype-cast.patch 2>/dev/null \
  && git apply /path/to/olala/patches/scattermoe-dtype-cast.patch \
  || echo "scattermoe dtype patch already applied (or source drifted — check manually)"
```

Drop this into the environment setup script (e.g. `setup_olala_env.sh`) right
after the scattermoe clone step. If upstream ever merges an equivalent fix,
`git apply --check` fails and the patch is skipped harmlessly.

# olala-env

Builds the Olala 7A1B verl training environment (GRPO/DAPO, FSDP + vLLM hybrid
engine) from pinned sources. One script, no other inputs.

```bash
git clone https://github.com/gcaillaut/olala-env.git
cd olala-env
bash setup_olala_env.sh            # everything, into ./env
```

Run it **from the repo root** — `REPO` is `$(pwd)`.

Then serve the checkpoint:

```bash
./serve.sh                      # ./checkpoints/sft on :8010, GPU 0
PORT=8020 GPU=1 ./serve.sh
CKPT=/path/to/export ./serve.sh
```

`serve.sh` uses the same venv, the same olala checkout and the same parsers the
env was built with, so what you serve matches what you trained against. It warns
before starting if `MAX_LEN` is not a multiple of 144, or if `MAX_NUM_SEQS`
exceeds what the KV cache holds at that length — both explained in its NOTES.

## Or take the image

Same env, same pins, built by the same script — `docker/` runs
`setup_olala_env.sh` rather than re-implementing it, so the two cannot drift.

```bash
./docker/build.sh                 # build, tag from the commit, push
PUSH=0 ./docker/build.sh          # build only
```

`git.corp.linguacustodia.com:5050/gcaillaut/olala-env/olala:<short sha>`. **TRL**
and **bitsandbytes** are not in the snapshot — it was frozen for verl, which
needs neither — so step 10 adds them; the image runs that step rather than
installing them itself, so a dev-box build and the image cannot differ. Based on
`nvcr.io/nvidia/pytorch:26.05-py3` for python3.12, a CUDA toolchain with a host
`g++`, and the RDMA/EFA userspace — but *not* for its torch, which is 2.12/CUDA
13.2 and cannot load the vLLM fork's precompiled binaries, nor for its NCCL,
since the snapshot pins `nvidia-nccl-cu12`. See
[docker/README.md](docker/README.md).

## What it pins

| source | pin |
|---|---|
| [gcaillaut/olala](https://github.com/gcaillaut/olala) | `e9b89e5` — model code, converter, parsers, renderer, the applier |
| [gcaillaut/vllm](https://github.com/gcaillaut/vllm) `olala-v0.26` | `7bb4e2575` — the fork that knows `OlalaForCausalLM` |
| mamba | upstream `state-spaces/mamba`, pip, `--no-deps` |
| scattermoe | `47b5e15` |
| renderers | `d4707862` |
| trl | `1.12.0` — not in the snapshot, which was frozen for verl |
| bitsandbytes | `0.49.2` — for the paged optimizers |

SHAs, not branches: a moving branch changes the build. Override any of them
inline (`OLALA_REF=... bash setup_olala_env.sh`).

## Knobs worth knowing

| var | default | why |
|---|---|---|
| `OLALA_HOME` | `$REPO/env` | where the ~40 GB build lands; gitignored |
| `CKPT` | `$REPO/checkpoints/sft` | a **raw** HF export, used in place — no patched copy |
| `OLALA_LOCAL` | *(empty)* | build from a local olala checkout instead of cloning; no push needed while iterating |
| `FROM` / `ONLY` | — | resume at, or run only, one step |

## Steps

1. preflight · 2. clone pinned sources · 3. venv + frozen snapshot ·
4. vllm (Olala fork, precompiled binaries) · 5. mamba_ssm + scattermoe +
the selective_scan stub · 6. patch the packages that still need it ·
7. verify the checkpoint loads · 8. renderers + the olala renderer ·
9. dragon-agentic (editable) · 10. trl + bitsandbytes · 11. summary

Step 10 is last because the Dockerfile drives this script with `ONLY=<n>`:
inserting a step earlier renumbers the ones after it, and `ONLY=8` would then
match nothing and exit 0.

## Notes

* `uv pip sync` in step 3 is **exact** — it uninstalls anything absent from the
  snapshot. Everything after it installs with `--no-deps`. Never run `uv sync`
  here: it would reinstall upstream vllm over the fork.
* The checkpoint needs no preparation. `modeling_olala.py` widens the 0-dim
  GeodesicNorm params after loading and writes them back 0-dim on save, so a raw
  export from `olala/convert` works as-is.
* On one GPU, FSDP1 OOMs at the initial weight sync — see the launcher notes at
  the end of the script.

# olala-env

Builds the Olala 7A1B verl training environment (GRPO/DAPO, FSDP + vLLM hybrid
engine) from pinned sources. One script, no other inputs.

```bash
git clone https://github.com/gcaillaut/olala-env.git
cd olala-env
bash setup_olala_env.sh            # everything, into ./env
```

Run it **from the repo root** — `REPO` is `$(pwd)`.

## What it pins

| source | pin |
|---|---|
| [gcaillaut/olala](https://github.com/gcaillaut/olala) | `e9b89e5` — model code, converter, parsers, renderer, the applier |
| [gcaillaut/vllm](https://github.com/gcaillaut/vllm) `olala-v0.26` | `7bb4e2575` — the fork that knows `OlalaForCausalLM` |
| mamba | upstream `state-spaces/mamba`, pip, `--no-deps` |
| scattermoe | `47b5e15` |
| renderers | `d4707862` |

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
9. dragon-agentic (editable) · 10. summary

## Notes

* `uv pip sync` in step 3 is **exact** — it uninstalls anything absent from the
  snapshot. Everything after it installs with `--no-deps`. Never run `uv sync`
  here: it would reinstall upstream vllm over the fork.
* The checkpoint needs no preparation. `modeling_olala.py` widens the 0-dim
  GeodesicNorm params after loading and writes them back 0-dim on save, so a raw
  export from `olala/convert` works as-is.
* On one GPU, FSDP1 OOMs at the initial weight sync — see the launcher notes at
  the end of the script.

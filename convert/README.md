# Olala checkpoint conversion

Megatron ⇄ HuggingFace, both directions, GPU or CPU. Everything needed lives in
this directory plus the model code in the repo root above it.

```bash
cp .env.example .env        # then edit the paths
./convert.sh mg2hf          # Megatron -> HF, 1 GPU: converts AND verifies
./convert.sh mg2hf cpu      # Megatron -> HF, no GPU: converts only
./convert.sh hf2mg          # HF -> Megatron (CPU by design)

# the tokenizer copied into an export, per run rather than via .env:
./convert.sh mg2hf cpu --tokenizer /path/to/tokenizer-channels-v4
```

`LOAD_DIR` is the **parent** of `iter_XXXXXXX/`, not the iter dir itself —
Megatron appends `iter_{ITERATION:07d}`.

## The model code has one home

`utils_convert.py` builds the HF model from `olala.modeling_olala`, and
`model_hf.save_pretrained()` copies that file into the export — so **the
converter decides what `modeling_olala.py` every checkpoint ships**.

It reads that from the **repo root above this directory**, bound to the package
name `olala` by `_olala_pkg.py`. There is deliberately no second copy: the
conversion tree used to carry its own `olala/` package, and it silently drifted
from the repo's by 237 lines — which is how exported checkpoints ended up
missing fixes the repo had, and vice versa.

Consequence worth knowing: fixing `modeling_olala.py` in this repo is enough.
Re-export and the checkpoint carries it. No post-hoc patching.

## mg2hf vs mg2hf-cpu

Same image, same output.

* **`mg2hf`** is the reference. Weight-level audit **plus** a forward pass
  through both models comparing logits.
* **`mg2hf-cpu`** never instantiates the Megatron model — TransformerEngine
  refuses to build its modules without CUDA — and reads the `torch_dist` store
  directly instead (`mg_cpu_loader.py`). It still runs the weight audit, which
  is what catches a tensor left at its random init or truncated by a dtype
  narrowing. It cannot run the forward comparison.

Use `mg2hf-cpu` when you want the artifact quickly and something else will
validate it; use `mg2hf` when the conversion itself is what's in question.

## hf2mg is CPU-only on purpose

Not an omission: TransformerEngine cannot build without CUDA, so the Megatron
model is never instantiated and the DCP store is written directly
(`mg_cpu_writer.py`). There is nothing for a GPU to do. `./convert.sh hf2mg gpu`
is accepted and warns.

It also needs a **reference** Megatron checkpoint (`REF_MG_DIR` +
`REF_ITERATION`), because an HF export carries neither:

* `common.pt`, Megatron's 682-field `args` Namespace — not reconstructible from
  an HF config, which isn't even a faithful record of it (`load_mg_save_hf.py`
  sets `intra_doc_masking = False` before saving); nor
* the padded vocab rows. Megatron trains on a vocab padded to
  `make_vocab_size_divisible_by * tensor_model_parallel_size` (128*4 = 512 →
  152064), `convert_mg_to_hf` drops the padding, and Megatron does not mask
  those logits out of the softmax denominator.

A near parent is fine for both — only its key/shape layout, args and padding
rows are used. Every weight comes from `HF_DIR`, and `MgSink.assert_complete()`
proves it: a tensor the mapping forgot is caught there rather than silently
shipping reference weights.

**Loading the result**: model weights only, no optimizer or RNG state. Megatron
needs `--finetune` (skips both, restarts the iteration counter) or
`--no-load-optim --no-load-rng`. It starts a new run from these weights; it
cannot resume an interrupted one.

## Gotchas

* `LOAD_DIR` is the **parent** of `iter_XXXXXXX/`, not the iter dir. Megatron
  appends `iter_{ITERATION:07d}` itself, so pointing it at the iter dir makes it
  look for `iter_0099518/iter_0099518` and find nothing.
* `convert.sh` creates the destination first. If docker has to create a missing
  host dir for a bind mount it makes it root-owned. Output files are written as
  root either way — `sudo chown -R $USER` after, if that matters.
* `MEGATRON_LM_DIR` / `OLALA_TOKENIZER_DIR` are resolved **inside** the
  container; `DRAGON_SFT_DIR` is mounted read-only at its host path so those
  defaults resolve unchanged.
* The selected GPU appears as device 0 in the container, so no
  `CUDA_VISIBLE_DEVICES`.
* `mg2hf*` go through `torchrun`, not plain `python`: the script calls
  `dist.init_process_group(init_method='env://')` and needs the env vars
  torchrun sets. The CPU path uses gloo but still goes through torchrun.
  `hf2mg` needs neither.

## The image

`Dockerfile` is a thin layer on the training image (`olala-sft`): adds
scattermoe (needed by the MoE experts, not on PyPI), the NCCL path fix DeepEP
asserts on, and `HF_HUB_OFFLINE=1`.

```bash
docker compose build mg2hf          # or: docker build -t olala-sft-convert:latest .
```

`pull_policy: missing` means an already-pulled tag is never refreshed on its
own — `docker compose pull mg2hf` to force it. Override the tag with
`CONVERT_IMAGE` in `.env`.

## Files

| file | role |
|---|---|
| `convert.sh` | entry point: direction + gpu/cpu |
| `docker-compose.yml` | `mg2hf`, `mg2hf-cpu`, `hf2mg` |
| `.env.example` | every knob, documented |
| `OLALA_TOKENIZER_DIR` | the tokenizer an export ships; mounted read-only at its own host path, so it may live anywhere. `--tokenizer` overrides it per run |
| `Dockerfile` | conversion image |
| `_olala_pkg.py` | binds the repo root to the package name `olala` |
| `load_mg_save_hf.py` | Megatron → HF, plus the audit and forward check |
| `load_hf_save_mg.py` | HF → Megatron |
| `hf_to_mg.py` | HF → Megatron weight mapping |
| `mg_cpu_loader.py` | reads a `torch_dist` store without CUDA |
| `mg_cpu_writer.py` | writes a `torch_dist` store without CUDA |
| `utils_convert.py` | shared mapping + model/config construction |

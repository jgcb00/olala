#!/usr/bin/env python3
"""Make an olala checkpoint trainable under FSDP, without touching the original.

    python scripts/patch_olala_checkpoint.py SRC_CKPT DST_CKPT
    MODEL_PATH=DST_CKPT ./launch_Olala.sh

``OlalaGeodesicNorm`` declares its two learnables as 0-dim tensors::

    self.scale = nn.Parameter(torch.tensor(1.))
    self.bias  = nn.Parameter(torch.tensor(0.))

and both FSDP1 and FSDP2 refuse those outright — ``fully_shard doesn't support scalar
parameters`` — for all 144 of them (2 params x 2 modules x 36 layers). It is not a
strategy choice: verl's fsdp/fsdp2 paths and veomni's ``torch_parallelize`` all end up in
the same ``_verify_managed_param``, and veomni additionally has no Olala modeling at all.
So the parameters have to become 1-D, which is numerically identical — the only use is
``theta * scale + bias`` where ``theta`` is ``[..., 1]``, so ``[1]`` broadcasts the same.

Two consequences, both handled here:

* The checkpoint's tensors are 0-dim, and ``from_pretrained`` compares shapes BEFORE
  copying, so it rejects the load as a size mismatch. A ``_load_from_state_dict`` hook does
  NOT help: transformers 5 bypasses module hooks for safetensors fast-loading (tried, and
  it still raised). So the tensors are rewritten to ``[1]`` instead.
* ``ignore_mismatched_sizes=True`` would "work" by re-initialising all 144 to 1.0/0.0,
  silently discarding trained values. Never use it here.

Everything else is symlinked, so this costs ~13 GB, not a full copy. The 72
``prosres_scalar`` entries stay 0-dim on purpose: they are buffers, and FSDP only inspects
parameters.

Also needed, separately: the same two-line widening in the vLLM port
(``vllm/model_executor/models/olala.py``), applied by scripts/apply_olala_training_fixes.sh
(fix 2) — already merged at source in jgcb00/vllm dragon-v0.26 (pin 8a124b6b0). Without
it the first weight sync dies — a 0-dim param cannot receive a ``[1]`` tensor
(``output with shape [] doesn't match the broadcast shape [1]``).

The real fix is upstream: the same change in Olala's own HF code and vLLM port, plus a
re-export. Then no patched copy is needed on any machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCALAR_PARAMS = ("scale", "bias")
PATCH = (
    ("        self.scale = nn.Parameter(torch.tensor(1.))\n"
     "        self.bias = nn.Parameter(torch.tensor(0.))"),
    ("        self.scale = nn.Parameter(torch.tensor([1.]))\n"
     "        self.bias = nn.Parameter(torch.tensor([0.]))"),
)


def patch_modeling(src: Path, dst: Path) -> None:
    text = src.read_text()
    old, new = PATCH
    if text.count(new):
        print("  modeling_olala.py: already 1-D")
    n = text.count(old)
    if n != 1 and not text.count(new):
        sys.exit(f"expected exactly 1 occurrence of the 0-dim declaration in {src}, found {n}")
    dst.write_text(text.replace(old, new))
    print(f"  modeling_olala.py: patched -> {dst}")


def reshape_weights(src: Path, dst: Path) -> int:
    from safetensors.torch import load_file, save_file

    sd = load_file(str(src))
    n = 0
    for key, value in list(sd.items()):
        if value.dim() == 0 and key.rsplit(".", 1)[-1] in SCALAR_PARAMS:
            sd[key] = value.reshape(1)
            n += 1
    buffers = sum(1 for v in sd.values() if v.dim() == 0)
    save_file(sd, str(dst), metadata={"format": "pt"})
    print(f"  model.safetensors: reshaped {n} params to [1], left {buffers} buffers 0-dim")
    return n


def verify(ckpt: Path) -> None:
    """Load it back and prove the trained values survived, rather than reverting to init."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(ckpt), trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    scales = [layer.geodesic_mixer.scale for layer in model.model.layers]
    if tuple(scales[0].shape) != (1,):
        sys.exit(f"expected shape (1,), got {tuple(scales[0].shape)}")
    if all(s.item() == 1.0 for s in scales):
        sys.exit("every scale is exactly 1.0 — the trained values were re-initialised, not loaded")
    print(f"  verified: shape (1,), scales {[round(s.item(), 4) for s in scales[:4]]}…")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="the original checkpoint directory (never modified)")
    ap.add_argument("dst", type=Path, help="where to build the patched checkpoint")
    ap.add_argument("--no-verify", action="store_true", help="skip the 13 GB reload check")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    for name in ("config.json", "modeling_olala.py", "model.safetensors"):
        if not (src / name).exists():
            sys.exit(f"not an olala checkpoint: {src} has no {name}")
    if dst == src:
        sys.exit("dst must differ from src — the original is never modified")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"patching {src}\n      -> {dst}")

    # Symlink everything, then replace the two files we own with real copies.
    for entry in sorted(src.iterdir()):
        link = dst / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(entry, link)

    (dst / "modeling_olala.py").unlink()
    patch_modeling(src / "modeling_olala.py", dst / "modeling_olala.py")

    (dst / "model.safetensors").unlink()
    reshape_weights(src / "model.safetensors", dst / "model.safetensors")

    if not args.no_verify:
        verify(dst)
    print(f"done. MODEL_PATH={dst}")


if __name__ == "__main__":
    main()

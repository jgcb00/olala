#!/usr/bin/env python3
"""The image build's smoke test: is this venv actually the Olala env?

Run by the Dockerfile right after TRL goes in, and again by verify.sh inside a
running container. It needs no GPU and no checkpoint, so it is cheap enough to
be unconditional. It also runs unchanged against a venv built by
setup_olala_env.sh on a dev box -- every check is relative to sys.prefix, not to
the image's paths.

It exists because two failure modes here are silent:

  1. Everything after step 3 of setup_olala_env.sh installs with --no-deps, into
     a venv synced from a snapshot that is deliberately not re-resolvable. So
     nothing in the install checks that the pieces fit together -- a wrong pin
     produces an env that imports and then fails at the first forward pass.

  2. The Dockerfile drives the install as `ONLY=<n> bash setup_olala_env.sh`.
     If that script ever renumbers its steps, `ONLY=<n>` matches nothing, does
     nothing, and exits 0. Without this file the build would happily ship an
     image whose venv is just the frozen snapshot -- no vllm fork, no
     scattermoe, no renderer.

Both of those become a failed build here rather than a failed training job three
hours in.
"""

import sys
from pathlib import Path

FAILURES: list[str] = []
PREFIX = Path(sys.prefix).resolve()


def check(label, fn):
    try:
        msg = fn()
    except Exception as exc:  # noqa: BLE001 -- any import error is a failure
        FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
        return
    if msg:
        FAILURES.append(f"{label}: {msg}")


def in_venv(module) -> bool:
    # A namespace package has __file__ = None. Treat that as "not from the
    # venv": it means the import resolved to a bare directory on sys.path
    # rather than to an installed package, which for anything checked here is
    # itself the bug.
    if getattr(module, "__file__", None) is None:
        return False
    return PREFIX in Path(module.__file__).resolve().parents


def imports():
    """Every piece the install adds on top of the snapshot must import."""
    import mamba_ssm  # noqa: F401
    import scattermoe  # noqa: F401
    import selective_scan_cuda  # noqa: F401  (import-only stub, step 5)
    import tilelang  # noqa: F401
    import transformers  # noqa: F401
    import trl  # noqa: F401
    import verl  # noqa: F401
    # Everything the antidoom FTPO pipeline needs on top of the snapshot. peft,
    # datasets, tensorboard and accelerate are IN the snapshot; these two are
    # not, and are added by step 10.
    import bitsandbytes  # noqa: F401
    # A dependency of the checkpoint's own modeling code rather than of any
    # package here: modeling_olala.py imports scattermoe inside a try/except
    # that only passes, so a broken install surfaces as
    # `NameError: ScatterMoE is not defined` at model construction instead.
    import peft  # noqa: F401
    import vllm  # noqa: F401
    return None


def torch_pin():
    """torch 2.11 + cu12, from the venv.

    Both halves matter. The pin is what the vLLM fork's precompiled binaries and
    the snapshot's flash-attn wheel were built against. And the base image ships
    its own torch 2.12/CUDA 13.2 in /usr/local/lib/python3.12/dist-packages, so
    if that is the one being imported then the venv is not isolated and
    everything downstream is an ABI coin flip.
    """
    import torch

    if not torch.__version__.startswith("2.11."):
        return f"expected torch 2.11.x, got {torch.__version__}"
    if not (torch.version.cuda or "").startswith("12."):
        return f"expected a cu12 torch build, got CUDA {torch.version.cuda}"
    if not in_venv(torch):
        return f"torch imported from outside {PREFIX}: {torch.__file__}"
    return None


def vllm_knows_olala():
    """The whole point of the fork. Upstream vllm has no OlalaForCausalLM."""
    import vllm
    from vllm.model_executor.models.registry import _VLLM_MODELS as registry

    if "OlalaForCausalLM" in registry:
        return None
    near = [k for k in registry if "lala" in k.lower() or "ragon" in k.lower()]
    return f"OlalaForCausalLM not registered in vllm {vllm.__version__}; near matches: {near}"


def renderer_registered():
    """Step 8. Without it nothing can parse an Olala response."""
    import renderers.olala  # noqa: F401  self-registers "olala"
    from renderers.base import RENDERER_REGISTRY

    if "olala" in RENDERER_REGISTRY:
        return None
    return f"olala renderer not registered; registry has {sorted(RENDERER_REGISTRY)}"


def scattermoe_from_clone():
    """Step 5 wires the pinned clone in with a .pth, and does not pip-install it.

    So resolving inside the venv means the .pth is missing and something else --
    a stray PyPI scattermoe -- is answering the import, which would silently
    drop the FSDP bf16 dtype-cast patch from step 6.
    """
    import scattermoe

    if in_venv(scattermoe):
        return f"scattermoe resolved inside the venv, not from the pinned clone: {scattermoe.__file__}"
    return None


def main():
    for label, fn in (
        ("imports", imports),
        ("torch pin", torch_pin),
        ("vllm knows Olala", vllm_knows_olala),
        ("olala renderer", renderer_registered),
        ("scattermoe wiring", scattermoe_from_clone),
    ):
        check(label, fn)

    if FAILURES:
        print("ENV CHECK FAILED", file=sys.stderr)
        for f in FAILURES:
            print(f"  {f}", file=sys.stderr)
        return 1

    from importlib.metadata import PackageNotFoundError, version

    print("  env check passed")
    for name in (
        "trl",
        "bitsandbytes",
        "torch", "vllm", "verl", "transformers", "trl", "peft", "accelerate",
        "datasets", "ray", "tilelang", "transferqueue", "flashinfer-python",
        "flash-attn", "renderers",
    ):
        try:
            print(f"  {name:20} {version(name)}")
        except PackageNotFoundError:
            print(f"  {name:20} -")
    return 0


if __name__ == "__main__":
    sys.exit(main())

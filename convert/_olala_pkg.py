"""Make the repo this file lives in importable as the package ``olala``.

The conversion scripts need ``olala.modeling_olala`` / ``olala.configuration_olala``,
and those modules use relative imports (``from .configuration_olala import
OlalaConfig``), so they cannot be loaded as flat top-level modules -- Python
raises "attempted relative import with no known parent package".

Importing the repo root as a package is also not automatic. It works only if
the checkout directory happens to be named exactly ``olala`` AND its parent is
on sys.path -- rename the directory, clone it somewhere else, or run from a
different cwd and ``import olala`` stops resolving.

So bind the repo root to the name ``olala`` explicitly. This keeps ONE copy of
the model code -- the repo root -- instead of the duplicate ``olala/`` tree the
conversion scripts used to carry beside them, which is how the exported
modeling_olala.py silently drifted from the repo's.

Usage, before importing anything from ``olala``:

    import _olala_pkg  # noqa: F401
    from olala.modeling_olala import OlalaForCausalLM

Override the location with OLALA_PKG_DIR if the model code lives elsewhere.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(os.environ.get("OLALA_PKG_DIR") or Path(__file__).resolve().parent.parent)


def _bind(name: str = "olala") -> object:
    """Import ``_REPO`` under ``name``; return the module. Idempotent."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing

    init = _REPO / "__init__.py"
    if not init.is_file():
        raise ImportError(
            f"{_REPO} is not an importable package (no __init__.py). "
            "Point OLALA_PKG_DIR at the olala checkout."
        )
    for required in ("modeling_olala.py", "configuration_olala.py"):
        if not (_REPO / required).is_file():
            raise ImportError(f"{_REPO} has no {required} -- wrong OLALA_PKG_DIR?")

    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(_REPO)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build a module spec for {_REPO}")
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so submodule relative imports resolve during it.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


PACKAGE_DIR = _REPO
olala = _bind()

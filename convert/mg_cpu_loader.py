"""CPU-only loading of a Megatron `torch_dist` checkpoint, for quick conversions.

Why this exists
---------------
The normal path (`utils_convert.load_merge_mg_models`) instantiates a real
`DragonModel`, which is built out of TransformerEngine modules -- and TE asserts
`TransformerEngine needs CUDA` inside `Module.__init__`, before a single weight
is touched. So the GPU requirement of `load_mg_save_hf.py` is not really about
the forward-pass check: you cannot even *open* the checkpoint without a GPU,
because Megatron needs the model's `sharded_state_dict()` to know what to read.

This module skips the Megatron model entirely. A `torch_dist` checkpoint is a
plain torch.distributed.checkpoint (DCP) store whose keys are module paths, and
DCP can read any subset of them into ordinary CPU tensors given only the shapes
in `.metadata`. We then wrap those tensors in an attribute tree that mimics the
`DragonModel` module hierarchy closely enough for `convert_mg_to_hf` to run
against it unmodified.

What you give up: everything that needs a forward pass. The exact,
weight-level `audit_conversion()` in load_mg_save_hf.py still runs and is still
the check that catches the dangerous class of bug (a tensor left at its random
init, or truncated by a dtype narrowing).

What is *not* approximated: the config. `build_dragon_config()` is shared with
the GPU path, so both read the checkpoint args identically.
"""

import re

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from megatron.training.checkpointing import get_checkpoint_name

from utils_convert import build_dragon_config, load_checkpoint_base


# Top-level module trees stored in the checkpoint that belong to the model.
# Everything else in the store (`optimizer.*`, `rng_state/*`) is deliberately
# not read -- it is the bulk of the bytes on disk and none of it is converted.
_MODEL_PREFIXES = ("decoder.", "embedding.", "output_layer.")

# Megatron's TEGroupedMLP holds one `weight{i}` parameter per expert on the
# module, but saves them fused as a single [num_experts, out, in] tensor under
# an extra `experts.` level. `convert_mg_to_hf` reads the module form, so we
# undo the fusion when building the tree.
_GROUPED_EXPERTS_RE = re.compile(r"^(?P<pre>.*\.experts)\.experts\.(?P<fc>linear_fc\d+)\.weight$")

# Tensors whose leading dimension is the vocabulary. Megatron trains on a vocab
# padded up for tensor-parallel divisibility and stores the padded tensor, but
# marks these two `allow_shape_mismatch=True` in its sharded_state_dict, so
# loading them into a model built with the unpadded `args.vocab_size` silently
# keeps the leading rows and drops the padding. `load_hf` builds the HF model at
# the unpadded size, so we have to do the same trim here.
# (Verified against a GPU-converted checkpoint: HF embedding == stored[:vocab].)
_VOCAB_TENSORS = ("embedding.word_embeddings.weight", "output_layer.weight")


class _MgModule:
    """Attribute-tree stand-in for a `DragonModel` submodule.

    `node.foo` returns a child node or a tensor, and raises AttributeError
    otherwise -- so the `hasattr(...)` probes in `convert_mg_to_hf` behave as
    they do on a real model (`hasattr(model_mg, 'module')` must be False here,
    and `hasattr(decoder.final_layernorm, 'weight')` must be False when the
    model has no final norm; see the empty node seeded in _build_module_tree).

    Non-tensor attributes that `convert_mg_to_hf` reads off real mixers
    (`config`, `num_super_heads`, `key_hidden_size`) are attached explicitly by
    `_attach_mixer_attrs`.
    """

    def __init__(self, path=""):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_items", {})

    # -- construction ------------------------------------------------------
    def _child(self, name):
        item = self._items.get(name)
        if item is None:
            item = _MgModule(f"{self._path}.{name}" if self._path else name)
            self._items[name] = item
        assert isinstance(item, _MgModule), f"{self._path}.{name} is a tensor, not a module"
        return item

    def _insert(self, path, value):
        *parents, leaf = path.split(".")
        node = self
        for p in parents:
            node = node._child(p)
        node._items[leaf] = value

    def _set(self, name, value):
        self._items[name] = value

    # -- access ------------------------------------------------------------
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_items")[name]
        except KeyError:
            path = object.__getattribute__(self, "_path")
            full = f"{path}.{name}" if path else name
            raise AttributeError(f"'{full}' is not in the checkpoint") from None

    def __repr__(self):
        return f"<_MgModule {self._path!r} {sorted(self._items)}>"


def _attach_mixer_attrs(model, config_mg):
    """Give each mixer node the non-tensor attributes `convert_mg_to_hf` reads.

    Only the attention branches ("T"/"V") need any: `num_super_heads` and
    `key_hidden_size`, both plain functions of the config on the real module
    (dragon_attention_v2.py: `num_attention_heads // 4`, and
    `query_projection_size / num_attention_heads`, i.e. `kv_channels`). The
    Mamba-3 ("M") and GDN ("g") branches read tensors only.
    """
    kv_channels = config_mg.kv_channels
    assert kv_channels is not None, "checkpoint args have no kv_channels"
    # Mirror `hidden_size_per_attention_head = divide(query_projection_size,
    # num_attention_heads)` so a future config where the two disagree fails here
    # rather than silently mis-slicing linear_in.
    assert config_mg.num_attention_heads % 4 == 0, (
        f"num_attention_heads={config_mg.num_attention_heads} is not a multiple of 4; "
        "the super-head grouping in convert_mg_to_hf assumes it is"
    )

    for i in range(config_mg.num_layers):
        layer = model.decoder.layers._items.get(str(i))
        if layer is None:
            continue
        mixer = layer._items.get("mixer")
        if not isinstance(mixer, _MgModule):
            continue
        mixer._set("config", config_mg)
        mixer._set("num_super_heads", config_mg.num_attention_heads // 4)
        mixer._set("key_hidden_size", kv_channels)


def _build_module_tree(tensors, config_mg):
    model = _MgModule()

    for key, tensor in tensors.items():
        m = _GROUPED_EXPERTS_RE.match(key)
        if m is None:
            model._insert(key, tensor)
            continue
        assert tensor.dim() == 3, (
            f"{key} has shape {tuple(tensor.shape)}; expected a fused "
            f"[num_experts, out, in] TEGroupedLinear weight"
        )
        assert tensor.shape[0] == config_mg.num_moe_experts, (
            f"{key} holds {tensor.shape[0]} experts, config says {config_mg.num_moe_experts}"
        )
        node = model
        for part in f"{m['pre']}.{m['fc']}".split("."):
            node = node._child(part)
        for e in range(tensor.shape[0]):
            node._set(f"weight{e}", tensor[e])

    # `decoder.layers` is keyed by string indices; give it integer indexing so
    # `model_mg.decoder.layers[i]` works like an nn.ModuleList.
    layers = model.decoder._items.get("layers")
    assert isinstance(layers, _MgModule), "checkpoint has no decoder.layers"
    model.decoder._set("layers", _LayerList(layers))

    # A real TransformerBlock always *has* `final_layernorm` (an IdentityOp when
    # the model has no final norm); convert_mg_to_hf probes it with hasattr, so
    # the node must exist even when the checkpoint stores no weight for it.
    model.decoder._items.setdefault("final_layernorm", _MgModule("decoder.final_layernorm"))

    _attach_mixer_attrs(model, config_mg)
    return model


class _LayerList(_MgModule):
    """`decoder.layers` with integer indexing, like an nn.ModuleList."""

    def __init__(self, node):
        object.__setattr__(self, "_path", node._path)
        object.__setattr__(self, "_items", node._items)

    def __getitem__(self, i):
        try:
            return self._items[str(i)]
        except KeyError:
            raise IndexError(f"no decoder.layers.{i} in the checkpoint") from None

    def __len__(self):
        return len(self._items)


def load_mg_weights_cpu(load_dir, iteration, print0=print):
    """Read a Megatron torch_dist checkpoint into CPU tensors. No GPU, no model.

    Signature-compatible with `utils_convert.load_merge_mg_models`:
    returns (model_mg_like, config_mg, vocab_size, wsize), where the first
    element is an attribute tree rather than a real `DragonModel` -- enough for
    `convert_mg_to_hf`, but it cannot run a forward pass.

    Tensors keep the dtype they were saved in (bf16 for weights). The GPU path
    upcasts them to fp32 on load; either way the values that reach the HF model
    are the checkpoint's bits, so the weight-level audit is unaffected.
    """
    sd = load_checkpoint_base(str(load_dir), iteration)
    config_mg = build_dragon_config(sd, params_dtype=torch.bfloat16)

    ckpt_dir = get_checkpoint_name(str(load_dir), iteration, False, return_base_dir=True)
    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata().state_dict_metadata

    # `_extra_state` entries are BytesStorageMetadata (TE's fp8 bookkeeping) and
    # are skipped by the isinstance check, along with any non-tensor object.
    plan = {}
    for key, meta in metadata.items():
        if not key.startswith(_MODEL_PREFIXES):
            continue
        if not isinstance(meta, TensorStorageMetadata):
            continue
        plan[key] = torch.empty(meta.size, dtype=meta.properties.dtype)

    assert plan, f"no model tensors found in {ckpt_dir} (prefixes {_MODEL_PREFIXES})"
    n_elem = sum(t.numel() for t in plan.values())
    n_bytes = sum(t.numel() * t.element_size() for t in plan.values())
    print0(f"Reading {len(plan)} tensors ({n_elem:,} elements, {n_bytes / 2**30:.1f} GiB) "
           f"from {ckpt_dir} on CPU...")
    dcp.load(plan, storage_reader=reader)

    vocab_size = sd["args"].vocab_size
    for key in _VOCAB_TENSORS:
        stored = plan.get(key)
        if stored is None:
            continue
        assert stored.shape[0] >= vocab_size, (
            f"{key} has {stored.shape[0]} rows, fewer than vocab_size={vocab_size}"
        )
        if stored.shape[0] > vocab_size:
            print0(f"  {key}: dropping {stored.shape[0] - vocab_size} padded vocab row(s) "
                   f"({stored.shape[0]} -> {vocab_size})")
            plan[key] = stored[:vocab_size]

    model = _build_module_tree(plan, config_mg)
    return model, config_mg, vocab_size, sd["wsize"]

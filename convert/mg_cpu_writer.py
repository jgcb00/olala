"""CPU-only writing of a Megatron `torch_dist` checkpoint.

The mirror of mg_cpu_loader.py, for the HF -> Megatron direction. Same trick:
a torch_dist checkpoint is a plain torch.distributed.checkpoint (DCP) store
whose keys are module paths, so it can be written from ordinary CPU tensors
without instantiating a `DragonModel` (which TransformerEngine refuses to build
without CUDA).

Two things make this safe rather than guesswork:

1. **The key set and every shape come from a REFERENCE checkpoint**, not from
   our own derivation. `MgSink` allocates exactly the tensors the reference
   holds, so a missing or mis-shaped write is a hard error rather than a
   checkpoint Megatron rejects hours later. The reference must be the same
   architecture -- a "near parent" of the HF checkpoint is fine, since only its
   key/shape/dtype layout and its `common.pt` are used.

2. **TE's `_extra_state` blobs are empty** in this model -- every one of the
   18757 of them deserialises to `[tensor([], dtype=torch.uint8)]`, because fp8
   is off. So we can reproduce them exactly without TransformerEngine.

What the produced checkpoint does NOT contain: optimizer state and RNG state.
An HF checkpoint carries neither. Megatron must therefore load it with
`--finetune` (which skips both, and restarts the iteration counter), or with
`--no-load-optim --no-load-rng`. It is a starting point for a new run, not a
resume of an interrupted one.
"""

import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
)

from megatron.training.checkpointing import get_checkpoint_name

from mg_cpu_loader import _MODEL_PREFIXES


# The exact object every TE `_extra_state` entry in this model deserialises to.
# Verified against 65k-betterpacks-lrfix/iter_0070000: all 18757 of them. It is
# fp8 scale/amax bookkeeping, and this model trains in bf16, so it is empty.
def _empty_extra_state():
    return [torch.tensor([], dtype=torch.uint8)]


class MgSink:
    """Pre-allocated Megatron tensors, shaped by a reference checkpoint.

    Write with ``sink["decoder.layers.0.mlp.router.weight"] = t``. Every write
    is checked against the reference shape and recorded, so `assert_complete()`
    can prove afterwards that the conversion filled everything -- the mirror of
    `audit_conversion` in load_mg_save_hf.py, which proves the same thing in the
    other direction.
    """

    def __init__(self, ref_metadata, print0=print):
        self._t = {}
        self._extra = []
        self._written = set()
        self._print0 = print0
        for key, meta in ref_metadata.items():
            if not key.startswith(_MODEL_PREFIXES):
                continue  # optimizer.* / rng_state/* are not ours to write
            if isinstance(meta, TensorStorageMetadata):
                self._t[key] = torch.empty(meta.size, dtype=meta.properties.dtype)
            elif isinstance(meta, BytesStorageMetadata):
                self._extra.append(key)
        assert self._t, "reference checkpoint holds no model tensors"

    # -- writing -----------------------------------------------------------
    def __setitem__(self, key, value):
        try:
            target = self._t[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not in the reference checkpoint; the reference is a "
                f"different architecture, or this key name is wrong"
            ) from None
        if key in self._written:
            raise KeyError(f"'{key}' written twice -- the conversion is double-assigning")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"'{key}': conversion produced {tuple(value.shape)}, reference "
                f"holds {tuple(target.shape)}"
            )
        target.copy_(value)          # dtype cast to the reference dtype happens here
        self._written.add(key)

    def set_rows(self, key, value):
        """Write only the leading rows of `key` (the rest is vocab padding).

        Used for the embedding / output layer, where the HF checkpoint is
        narrower than Megatron's padded vocab. Counts as written, so the audit
        passes; `restore_vocab_padding` fills the tail afterwards.
        """
        target = self._t[key]
        assert value.shape[1:] == target.shape[1:], (
            f"'{key}': trailing dims {tuple(value.shape[1:])} != {tuple(target.shape[1:])}")
        assert value.shape[0] <= target.shape[0]
        if key in self._written:
            raise KeyError(f"'{key}' written twice")
        target[: value.shape[0]].copy_(value)
        self._written.add(key)

    def __contains__(self, key):
        return key in self._t

    def __getitem__(self, key):
        return self._t[key]

    def shape(self, key):
        return tuple(self._t[key].shape)

    def keys(self):
        return self._t.keys()

    # -- auditing ----------------------------------------------------------
    def assert_complete(self):
        """Every reference tensor must have been written from the HF checkpoint.

        This is what stops the reference from silently leaking weights into the
        output: anything the conversion forgot would otherwise ship as whatever
        `torch.empty` left in memory.
        """
        missing = sorted(set(self._t) - self._written)
        self._print0("\n=== SINK AUDIT (exact) ===")
        if missing:
            n = sum(self._t[k].numel() for k in missing)
            self._print0(f"❌ {len(missing)} Megatron tensor(s) ({n:,} elements) were never "
                         f"written by convert_hf_to_mg:")
            for k in missing[:40]:
                self._print0(f"     {k} {tuple(self._t[k].shape)}")
            if len(missing) > 40:
                self._print0(f"     ... and {len(missing) - 40} more")
            return False
        self._print0(f"✅ all {len(self._t)} Megatron tensors were written from the HF checkpoint")
        return True

    def restore_vocab_padding(self, ref_dir, hf_rows, print0=print):
        """Restore the padded vocab rows, which an HF checkpoint does not carry.

        Megatron trains on a vocab padded to
        `make_vocab_size_divisible_by * tensor_model_parallel_size`, and
        convert_mg_to_hf drops the padding (mg_cpu_loader._VOCAB_TENSORS). Those
        rows are never targets, but Megatron does not mask them out of the
        softmax denominator either, so they are not free to invent -- we take
        them from the reference. They are the ONLY numbers in the output that do
        not come from the HF checkpoint.

        `hf_rows` maps key -> how many leading rows the HF checkpoint supplied.
        """
        wanted = {k: n for k, n in hf_rows.items() if k in self._t and self._t[k].shape[0] > n}
        if not wanted:
            print0("  no padded vocab rows to restore (HF vocab == Megatron padded vocab)")
            return
        reader = FileSystemReader(ref_dir)
        meta = reader.read_metadata().state_dict_metadata
        plan = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) for k in wanted}
        dcp.load(plan, storage_reader=reader)
        for key, n in wanted.items():
            full = self._t[key].shape[0]
            assert plan[key].shape[0] == full, (
                f"{key}: reference has {plan[key].shape[0]} rows, this checkpoint needs {full}. "
                f"padded_vocab_size depends on the training TP size "
                f"(make_vocab_size_divisible_by * tensor_model_parallel_size), so the "
                f"reference must come from a run with the same TP."
            )
            self._t[key][n:] = plan[key][n:]
            print0(f"  {key}: rows [{n}:{full}] ({full - n} padded) taken from the reference")


def write_mg_checkpoint(sink, out_dir, iteration, ref_dir, print0=print):
    """Write the DCP store + common.pt that Megatron will load.

    `ref_dir` supplies `common.pt` -- the 682-field `args` Namespace that
    Megatron needs and that cannot be reconstructed from an HF config. Only
    `iteration`/`wandb_step` are patched; with `--finetune` Megatron ignores the
    optimizer and scheduler entries it also contains.
    """
    out_dir = Path(out_dir)
    ckpt_dir = Path(get_checkpoint_name(str(out_dir), iteration, False, return_base_dir=True))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = dict(sink._t)
    for key in sink._extra:
        state[key] = _empty_extra_state()

    n_bytes = sum(t.numel() * t.element_size() for t in sink._t.values())
    print0(f"Writing {len(sink._t)} tensors ({n_bytes / 2**30:.1f} GiB) + "
           f"{len(sink._extra)} empty _extra_state entries to {ckpt_dir}...")
    # flatten_state_dict=False is required, not cosmetic: the default planner
    # flattens a list value into one entry per element, so each _extra_state
    # would land as a TENSOR named '<key>.0' instead of the pickled object
    # Megatron asks for by '<key>'. Its loader would then not find any of them.
    dcp.save(state, storage_writer=FileSystemWriter(str(ckpt_dir)),
             planner=DefaultSavePlanner(flatten_state_dict=False))

    # common.pt: copy the reference's and patch the iteration counters.
    ref_common = Path(ref_dir) / "common.pt"
    assert ref_common.is_file(), f"no common.pt in the reference checkpoint {ref_dir}"
    common = torch.load(ref_common, map_location="cpu", weights_only=False)
    common["iteration"] = iteration
    if "wandb_step" in common:
        common["wandb_step"] = iteration
    torch.save(common, ckpt_dir / "common.pt")
    print0(f"Wrote common.pt (args from {ref_common}, iteration -> {iteration})")

    # metadata.json is Megatron's marker for "this directory is a distributed
    # checkpoint" -- without it find_checkpoint_rank_0() returns None and the
    # load dies as `TypeError: stat: path should be string ... not NoneType`,
    # nowhere near the real cause. Copied from the reference so the backend
    # version fields match the writer this Megatron expects.
    ref_meta_json = Path(ref_dir) / "metadata.json"
    if ref_meta_json.is_file():
        shutil.copyfile(ref_meta_json, ckpt_dir / "metadata.json")
        print0(f"Wrote metadata.json (copied from {ref_meta_json})")
    else:
        (ckpt_dir / "metadata.json").write_text(
            '{"sharded_backend": "torch_dist", "sharded_backend_version": 1, '
            '"common_backend": "torch", "common_backend_version": 1}')
        print0("Wrote metadata.json (default torch_dist/torch backends)")

    tracker = out_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text(f"{iteration}\n")
    print0(f"Wrote {tracker}")
    return ckpt_dir

"""HF -> Megatron checkpoint conversion. The reverse of load_mg_save_hf.py.

For the pipeline `Megatron -> HF -> (TRL) -> HF -> Megatron`: take the HF
weights that came back from post-training and write a Megatron `torch_dist`
checkpoint a new run can start from.

    python load_hf_save_mg.py \
        --hf-dir      .../huggingface/iter_0080000 \
        --ref-mg-dir  .../megatron --ref-iteration 80000 \
        --save-dir    .../megatron-sft --iteration 1

CPU only -- no GPU, no torchrun, no distributed init. TransformerEngine refuses
to build its modules without CUDA, so the Megatron model is never instantiated;
the DCP store is written directly (see mg_cpu_writer.py).

Why a reference checkpoint is required
--------------------------------------
Two things an HF checkpoint cannot supply:
  * `common.pt`, i.e. Megatron's 682-field `args` Namespace. Not reconstructible
    from an HF config, and note the HF config is not even a faithful record of
    it -- load_mg_save_hf.py sets `intra_doc_masking = False` before saving.
  * The padded vocab rows. Megatron trains on a vocab padded to
    `make_vocab_size_divisible_by * tensor_model_parallel_size` (128*4 = 512 ->
    152064 here); convert_mg_to_hf drops the padding, and Megatron does not mask
    those logits out of the softmax denominator, so they are not free to invent.

A "near parent" is fine for both: only the reference's key/shape layout, its
args, and those padding rows are used. Every weight comes from the HF
checkpoint, and `MgSink.assert_complete()` proves it -- any tensor the mapping
forgot would be caught there rather than silently shipping reference weights.

Loading the result
------------------
The output has model weights only; an HF checkpoint carries no optimizer or RNG
state. Megatron must load it with `--finetune` (which skips both and restarts
the iteration counter) or `--no-load-optim --no-load-rng`. It starts a new run
from these weights; it cannot resume an interrupted one.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

# Megatron-LM must be importable. Overridable so this tree is not pinned to
# one host's layout; the default is the path it was developed against.
sys.path.append(os.environ.get(
    "MEGATRON_LM_DIR",
    "/data/home/gaetan.caillaut/dragon-sft/7A1B/training/Megatron-LM"))

import torch
from torch.distributed.checkpoint import FileSystemReader

from megatron.training.checkpointing import get_checkpoint_name

from utils_convert import build_dragon_config, load_checkpoint_base, load_hf_config
from mg_cpu_writer import MgSink, write_mg_checkpoint
from hf_to_mg import HfWeights, convert_hf_to_mg


def main(args):
    print0 = print

    # -- reference: args, and the key/shape layout to fill ------------------
    ref_ckpt = get_checkpoint_name(str(args.ref_mg_dir), args.ref_iteration, False,
                                   return_base_dir=True)
    print0(f"Reference Megatron checkpoint: {ref_ckpt}")
    sd = load_checkpoint_base(str(args.ref_mg_dir), args.ref_iteration)
    config_mg = build_dragon_config(sd, params_dtype=torch.bfloat16)
    vocab_size, wsize = sd["args"].vocab_size, sd["wsize"]
    config_hf = load_hf_config(config_mg, vocab_size, wsize)
    print0(f"  layers={config_mg.num_layers} ({config_mg.layers_mixer_config}) "
           f"hidden={config_mg.hidden_size} experts={config_mg.num_moe_experts} "
           f"vocab={vocab_size} padded_vocab={sd['args'].padded_vocab_size}")

    ref_meta = FileSystemReader(ref_ckpt).read_metadata().state_dict_metadata
    sink = MgSink(ref_meta, print0=print0)
    print0(f"  sink: {len(sink.keys())} tensors to fill, "
           f"{len(sink._extra)} _extra_state entries")

    # -- HF weights ---------------------------------------------------------
    hf = HfWeights(args.hf_dir)
    print0(f"HF checkpoint: {args.hf_dir} ({len(hf.keys())} tensors)")

    hf_rows = convert_hf_to_mg(config_mg, config_hf, hf, sink, print0=print0)

    # -- audits -------------------------------------------------------------
    weights_ok = sink.assert_complete()

    print0("\n=== VOCAB PADDING (the only numbers not from HF) ===")
    sink.restore_vocab_padding(ref_ckpt, hf_rows, print0=print0)

    unused = sorted(set(hf.keys()) - _consumed(hf, config_mg))
    if unused:
        print0(f"\n{len(unused)} HF tensor(s) had no Megatron counterpart "
               f"(expected: HF-only buffers):")
        for k in unused[:10]:
            print0(f"     {k}")
        if len(unused) > 10:
            print0(f"     ... and {len(unused) - 10} more")

    if not weights_ok:
        print0("\n=== SUMMARY === FAILED: not writing an incomplete checkpoint")
        sys.exit(1)

    # -- write --------------------------------------------------------------
    out = write_mg_checkpoint(sink, args.save_dir, args.iteration, ref_ckpt, print0=print0)
    print0(f"\n=== SUMMARY === weights: OK   wrote {out}")
    print0("Load it with --finetune (or --no-load-optim --no-load-rng): "
           "this checkpoint has no optimizer or RNG state.")


def _consumed(hf, config_mg):
    """HF keys the mapping reads, so the caller can report the leftovers."""
    used = {"model.embedding.weight", "lm_head.weight", "model.final_norm.norm.weight"}
    for i, t in enumerate(config_mg.layers_mixer_config):
        p = f"model.layers.{i}"
        used |= {f"{p}.mixer_proj.weight", f"{p}.input_norm.norm.weight",
                 f"{p}.postmixer_norm.norm.weight", f"{p}.mixer_group_norm.weight"}
        used |= {f"{p}.geodesic_{w}.{x}" for w in ("mixer", "mlp") for x in ("scale", "bias")}
        used |= {f"{p}.mlp.{k}" for k in (
            "moe_gate.weight", "expert_bias", "down_proj.weight", "up_proj.weight",
            "experts.experts.weight", "experts.output_experts.weight",
            "shared_experts.fc_1.weight", "shared_experts.fc_2.weight",
            "shared_gate.weight", "fc_1.weight", "fc_2.weight")}
        if t == "V":
            used |= {f"{p}.mixer.{k}" for k in (
                "c_q.weight", "W_A_k.weight", "W_A_v.weight", "W_B_k.weight", "W_B_v.weight",
                "shift_proj_k.weight", "shift_proj_v.weight", "lambda_proj.weight",
                "q_norm.norm.weight", "k_norm.norm.weight", "softmax_scaler")}
            used.add(f"{p}.gate_proj.weight")
        elif t == "M":
            used |= {f"{p}.mixer.{k}" for k in (
                "in_proj.weight", "in_proj_dyn.weight", "B_bias", "C_bias",
                "in_proj_mimo_x", "in_proj_mimo_z", "out_proj_mimo", "dt_bias", "D",
                "B_norm.norm.weight", "C_norm.norm.weight", "output_norm.norm.weight")}
    return used


if __name__ == "__main__":
    @dataclass
    class Args:
        hf_dir: Path
        """HF checkpoint to convert (the one that came back from post-training)."""
        ref_mg_dir: Path
        """Megatron checkpoint PARENT dir (the one containing iter_XXXXXXX/).
        Supplies args/common.pt and the padded vocab rows -- a near parent of the
        HF checkpoint is fine, no weights are taken from it."""
        ref_iteration: int
        save_dir: Path
        """Megatron output PARENT dir; iter_{iteration:07d}/ is created inside."""
        iteration: int = 1
        """Iteration to stamp on the output.

        Must be >= 1: Megatron's read_metadata() asserts `iteration > 0`, so a
        checkpoint written at iteration 0 is rejected by its own loader. The
        value is otherwise cosmetic for a --finetune run, which restarts the
        sample counters regardless."""

    _args = tyro.cli(Args)
    if _args.iteration < 1:
        sys.exit("--iteration must be >= 1: Megatron's read_metadata() asserts "
                 "iteration > 0, so iteration 0 cannot be loaded back.")
    main(_args)

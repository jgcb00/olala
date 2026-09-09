import sys
import os
# Megatron-LM must be importable. Overridable so this tree is not pinned to
# one host's layout; the default is the path it was developed against.
sys.path.append(os.environ.get(
    "MEGATRON_LM_DIR",
    "/data/home/gaetan.caillaut/dragon-sft/7A1B/training/Megatron-LM"))

import contextlib
from dataclasses import dataclass
from pathlib import Path
import tyro
from typing import Optional

from transformers import AutoTokenizer

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.packed_seq_params import PackedSeqParams

from utils_convert import load_merge_mg_models, load_random_mg_models, load_hf, convert_mg_to_hf
from mg_cpu_loader import load_mg_weights_cpu


# -----------------------------------------------------------------------------
# Numerical-equivalence tolerances.
#
# These are calibrated against the measured noise floor of the *reference*
# implementation, not against fp32 machine epsilon. TransformerEngine runs its
# fp32 GEMMs in TF32 (~2e-4 relative error per GEMM, measured against a float64
# reference on a single lm_head matmul) and ignores both
# torch.backends.cuda.matmul.allow_tf32 and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE.
# The HF path uses torch GEMMs (~7e-7). Over 36 layers that systematic
# difference accumulates to ~2e-2 relative error on the logits, so a bit-exact
# conversion lands at cos ~= 0.9998 and std/meanAbsDiff ~= 275 -- the old
# `ratio > 400` gate was unreachable by construction.
#
# With both models in bf16 (--mg_bf16) the floor is coarser still: each side now
# carries its own independent bf16 rounding, so the gap between them grows even
# though the weights are bit-identical. That mode says "how far apart will the
# two runtimes drift in production", not "is the mapping right".
#
# The mapping itself is verified exactly by audit_conversion() below, which does
# not depend on any of this.
#
# Measured on 7A1B/65k-betterpacks (36 layers, 256 experts), bit-exact weights:
#            cos        relL2     top-1 argmax
#   fp32  0.999903    1.40e-02      100.00%
#   bf16  0.998564    5.40e-02       98.05%
# -----------------------------------------------------------------------------
TOL = {
    # cos_min, rel_l2_max, top1_min
    "fp32": (0.999, 5e-2, 0.98),
    "bf16": (0.997, 1e-1, 0.95),
}


@contextlib.contextmanager
def record_copies():
    """Intercept Tensor.copy_ so a conversion can be audited afterwards.

    Yields (touched_storages, lossy_copies):
      touched_storages -- data_ptr of every storage written to, so we can find
                          HF parameters that convert_mg_to_hf never assigned
                          (they'd silently keep their random init, which a
                          forward-pass check can easily miss for small tensors).
      lossy_copies     -- copies that did not round-trip, i.e. where bits were
                          lost to a dtype narrowing.
    """
    touched = set()
    lossy = []
    orig_copy = torch.Tensor.copy_

    def patched(self, other, *a, **kw):
        out = orig_copy(self, other, *a, **kw)
        try:
            touched.add(self.untyped_storage().data_ptr())
            if torch.is_tensor(other) and other.dtype != self.dtype:
                if not torch.equal(self.to(other.dtype), other):
                    lossy.append((tuple(self.shape), str(other.dtype), str(self.dtype)))
        except Exception:
            pass
        return out

    torch.Tensor.copy_ = patched
    try:
        yield touched, lossy
    finally:
        torch.Tensor.copy_ = orig_copy


def _expected_unwritten():
    """Tensors convert_mg_to_hf legitimately never writes, with the reason.

    Keep this list short and justified -- each entry is a place where the audit
    is being told to look away. Both entries below are non-weights with a
    deterministic init and no Megatron counterpart.
    """
    return [
        (r"\.mlp\.tokens_per_expert$",
         "runtime routing counter, not a weight"),
        (r"\.geodesic_(mixer|mlp)\.prosres_scalar$",
         "constant buffer (init 1.0), never read by DragonGeodesicNorm.forward, "
         "no MG counterpart -- the live geodesic params scale/bias ARE copied"),
    ]


def audit_conversion(model_hf, touched, lossy, print0):
    """Exact, forward-pass-free check of convert_mg_to_hf.

    The Megatron checkpoint stores bf16 model weights, so every mapped tensor
    should be copied bit-for-bit. This catches the dangerous class of bug --
    a tensor left at its random init, or truncated by a dtype narrowing -- that
    a cos=0.9999 logit comparison cannot distinguish from kernel noise.
    """
    import re
    from collections import defaultdict

    print0("\n=== WEIGHT-LEVEL AUDIT (exact) ===")
    exempt = _expected_unwritten()

    unexpected, excused = [], defaultdict(list)
    for name, p in list(model_hf.named_parameters()) + list(model_hf.named_buffers()):
        if p.numel() == 0 or p.untyped_storage().data_ptr() in touched:
            continue
        for pat, why in exempt:
            if re.search(pat, name):
                excused[why].append(name)
                break
        else:
            unexpected.append((name, tuple(p.shape), p.numel()))

    if lossy:
        print0(f"❌ {len(lossy)} copy(ies) lost bits to a dtype narrowing:")
        for shape, s, d in lossy[:10]:
            print0(f"     shape={shape} {s} -> {d}")
    else:
        print0("✅ every copy was bit-exact (no dtype narrowing lost information)")

    if unexpected:
        n = sum(m[2] for m in unexpected)
        print0(f"❌ {len(unexpected)} HF tensor(s) ({n:,} elements) were never written by "
               f"convert_mg_to_hf and still hold their random init:")
        for name, shape, _ in unexpected[:40]:
            print0(f"     {name} {shape}")
        if len(unexpected) > 40:
            print0(f"     ... and {len(unexpected) - 40} more")
    else:
        print0("✅ every HF parameter/buffer that the model uses was written")

    for why, names in excused.items():
        collapsed = sorted({re.sub(r"\.layers\.\d+\.", ".layers.*.", n) for n in names})
        print0(f"   (skipped {len(names)}: {why}) -> {', '.join(collapsed)}")

    return not unexpected and not lossy


def equivalence_report(y_hf, y_mg, print0, mode="fp32"):
    """Compare MG and HF logits. Returns True if within the calibrated tolerances."""
    cos_min, rel_l2_max, top1_min = TOL[mode]
    print0(f"\n=== NUMERICAL EQUIVALENCE (forward, {mode}/{mode}) ===")
    # NB: logits dtype alone is not a reliable precision signal -- the HF head
    # upcasts to fp32 even when the weights are bf16. The weight-dtype guard in
    # __main__ is what actually catches a precision mismatch.
    print0(f"Output shape HF: {tuple(y_hf.shape)} ({y_hf.dtype}), "
           f"mg: {tuple(y_mg.shape)} ({y_mg.dtype})")

    print0("HF logits [0, :5, :5]:")
    print0(y_hf[0, :5, :5])
    print0("MG logits [0, :5, :5]:")
    print0(y_mg[0, :5, :5])

    y1 = y_hf.detach().float().reshape(-1)
    y2 = y_mg.detach().float().reshape(-1)
    diff = y1 - y2

    mean_abs = diff.abs().mean().item()
    std = y1.std().item()
    ratio = std / mean_abs if mean_abs > 0 else float('inf')
    cos = torch.nn.functional.cosine_similarity(y1, y2, dim=0).item()
    y1c, y2c = y1 - y1.mean(), y2 - y2.mean()
    pearson_r = torch.nn.functional.cosine_similarity(y1c, y2c, dim=0).item()
    rel_l2 = (diff.norm() / y2.norm().clamp_min(1e-12)).item()
    rmse_over_std = (diff.pow(2).mean().sqrt() / max(std, 1e-12)).item()
    sym_rel_l2 = (diff.norm() / (0.5 * (y1.norm() + y2.norm()) + 1e-12)).item()
    # Affine alignment (detect pure scale+shift mismatch)
    a = ((y1c * y2c).mean() / (y2c.pow(2).mean() + 1e-20)).item()
    b = (y1.mean() - a * y2.mean()).item()
    affine_nmse = (((a * y2 + b) - y1).pow(2).sum() / (y1.pow(2).sum() + 1e-12)).item()

    # Task-level agreement: what actually matters for a converted checkpoint.
    top1 = (y_hf.detach().float().argmax(-1) == y_mg.detach().float().argmax(-1)).float().mean().item()
    k = 5
    t_hf = y_hf.detach().float().topk(k, dim=-1).indices
    t_mg = y_mg.detach().float().topk(k, dim=-1).indices
    top5 = (t_hf.unsqueeze(-1) == t_mg.unsqueeze(-2)).any(-1).float().mean().item()

    print0(f"Mean diff: {mean_abs:.6g}   FWD logits STD: {std:.6g}   std/meanDiff: {ratio:.1f}")
    print0(f"cos={cos:.6f}  r(Pearson)={pearson_r:.6f}  relL2={rel_l2:.3e}  "
           f"RMSE/std={rmse_over_std:.3e}  symRelL2={sym_rel_l2:.3e}  "
           f"affineNMSE={affine_nmse:.3e}  a={a:.6f}  b={b:.3e}")
    print0(f"top-1 argmax agreement={top1*100:.3f}%   top-{k} overlap={top5*100:.3f}%")

    checks = [
        ("cos", cos >= cos_min, f"{cos:.6f} >= {cos_min}"),
        ("relL2", rel_l2 <= rel_l2_max, f"{rel_l2:.3e} <= {rel_l2_max:.0e}"),
        ("top-1 agreement", top1 >= top1_min, f"{top1:.4f} >= {top1_min}"),
    ]
    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print0(f"  {'✅' if passed else '❌'} {name}: {detail}")
    print0("✅ Numerical equivalence passed." if ok else "❌ Numerical equivalence failed.")
    return ok


def cast_hf_to_bf16(model_hf):
    """Cast HF weights to the dtype the checkpoint is saved in.

    Mutates the model, so nothing may be compared against the fp32 MG reference
    afterwards -- that mismatch is what made the old check unpassable.
    """
    with torch.no_grad():
        for n, p in model_hf.named_parameters():
            if "weight" in n and not ("moe_gate" in n or "shared_gate" in n):
                p.data = p.data.to(torch.bfloat16)


def build_test_inputs(config_mg, vocab_size, device, B=1, L=1024, bos_id=0):
    """Random tokens + a fake document packing, shared by both models."""
    x = torch.randint(0, vocab_size, (B, L), device=device)
    labels = torch.randint(0, vocab_size, (B, L), device=device)

    X = torch.randint(0, vocab_size, (B, L), device="cpu")  # fake tokens
    starts = (X == bos_id).nonzero(as_tuple=True)[1].to(torch.long)
    if starts.numel() == 0 or starts[0] != 0:
        starts = torch.cat([torch.zeros(1, dtype=torch.long), starts])
    ends = torch.cat([starts[1:], torch.tensor([X.numel()])])
    seqlens = (ends - starts).to(torch.int32)
    # position_ids.
    lengths = seqlens.to(torch.long)
    starts_per_token = torch.repeat_interleave(starts.to(torch.long), lengths)
    idx = torch.arange(L, device=X.device, dtype=torch.long)
    position_ids = (idx - starts_per_token).unsqueeze(0).to(device)
    # cu_seqlens, max_seqlen.
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), seqlens.cumsum(0)]).cuda().to(torch.int32)
    max_seqlen = int(seqlens.max())

    if config_mg.intra_doc_masking:
        packed_seq_params = PackedSeqParams(qkv_format='thd', position_ids=position_ids.squeeze(0),
                                            cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                                            max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen)
        position_ids = position_ids.squeeze(0)
    else:
        packed_seq_params = None
        position_ids = torch.arange(L, device=device).unsqueeze(0)
        cu_seqlens = None
        max_seqlen = None

    return x, labels, position_ids, cu_seqlens, max_seqlen, packed_seq_params, seqlens


def run_forward_check(args, config_mg, model_mg, model_hf, vocab_size, wsize, print0):
    """Run both models on the same input and compare logits. GPU-only.

    Lifted verbatim out of __main__ so the CPU path can skip it wholesale;
    the ordering constraints called out below still hold within it.
    """
    # -------------------------------------------------------------------------
    # CHECK 2: forward pass. MUST run before the bf16 cast below and before
    # config.intra_doc_masking is mutated for saving -- otherwise it compares a
    # bf16 HF model against an fp32 MG model (and, because
    # DragonAttention.forward reads config.intra_doc_masking at call time, a
    # dense-causal HF against a varlen-packed MG).
    # -------------------------------------------------------------------------
    if args.mg_bf16:
        # Both sides bf16: the precision the saved checkpoint actually runs at.
        cast_hf_to_bf16(model_hf)

    device = torch.device("cuda")
    B, L = 1, 1024
    (x, labels, position_ids, cu_seqlens, max_seqlen,
     packed_seq_params, seqlens) = build_test_inputs(config_mg, vocab_size, device, B=B, L=L)
    print0(f"\nTest input: B={B} L={L} intra_doc_masking={config_mg.intra_doc_masking} "
           f"n_docs={seqlens.numel()} max_seqlen={max_seqlen}")

    # Guard the bug this restructuring fixed: if the two models are not at the
    # same precision, the forward comparison measures rounding, not conversion.
    d_mg = model_mg.output_layer.weight.dtype
    d_hf = model_hf.lm_head.weight.dtype
    print0(f"Compute precision: MG={d_mg} HF={d_hf}")
    if d_mg != d_hf:
        print0(f"⚠️  precision mismatch ({d_mg} vs {d_hf}) -- the forward numbers below "
               f"are dominated by rounding, not by the conversion. Did something cast a "
               f"model before the check?")

    l = 1
    #x = torch.randn(B, L, config_mg.hidden_size, device=device, dtype=torch.bfloat16)
    # mixer
    #y_mg = model_mg.decoder.layers[l].mixer(x.transpose(0,1).float(), attention_mask=None, packed_seq_params=packed_seq_params, window_size=(wsize, 0))[0].transpose(0,1)
    #y_hf, _, _ = model_hf.model.layers[l].mixer(model_hf.model.layers[l].lns * model_hf.model.layers[l].input_norm(x), position_embeddings=None, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    # mlp
    #y_mg, _, _, ind_mg, scores_mg, _  = model_mg.decoder.layers[l].mlp(x.to(torch.float32).transpose(0,1))
    #y_mg = y_mg.transpose(0,1).squeeze(0)
    """y_hf, ind_hf, scores_hf = model_hf.model.layers[l].mlp(x, x)
    # orderless (multiset) equality: same values up to permutation
    mg_sorted, _ = ind_mg.sort(dim=1)
    hf_sorted, _ = ind_hf.sort(dim=1)
    print0(mg_sorted)
    print0(hf_sorted)
    per_token_same = (mg_sorted == hf_sorted).all(dim=1)  # (B,)
    same = per_token_same.sum().item()
    total = per_token_same.numel()
    pct = same / total * 100
    print0(f"[COMPARE] per-token expert multiset equal = {same:,}/{total:,} → {pct:.3f}%")
    avg_elem_match = (mg_sorted == hf_sorted).float().mean().item() * 100
    print0(f"[COMPARE] element-wise match after sort   = {avg_elem_match:.3f}%")
    # overlap count per row (ignores order; treats as set, not multiset)
    mg = ind_mg
    hf = ind_hf
    overlap = (mg[:, :, None] == hf[:, None, :]).any(dim=2).sum(dim=1)  # (B,)
    print0("mean overlap:", overlap.float().mean().item(), " / K =", mg.size(1))
    print0("hist:", torch.bincount(overlap, minlength=mg.size(1)+1))
    y_mg, y_hf = scores_mg, scores_hf"""

    # block
    #y_mg = model_mg.decoder.layers[l](x.transpose(0,1).float(), attention_mask=None, packed_seq_params=packed_seq_params, window_size=(wsize, 0)).transpose(0,1)
    #y_hf, _, _ = model_hf.model.layers[l](x, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    # model
    # training (idm)
    with torch.no_grad():
        y_mg = model_mg(x, position_ids=position_ids, attention_mask=None, just_logits=True, window_size=(wsize, 0), labels=labels, packed_seq_params=packed_seq_params).transpose(0, 1)
        y_hf = model_hf(input_ids=x, inputs_embeds=None, labels=labels, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
    # ablations (no idm)
    #y_mg = model_mg(x, attention_mask=None, position_ids=None, just_logits=True, window_size=(wsize, 0), labels=labels).transpose(0, 1)
    #y_hf = model_hf(input_ids=x, inputs_embeds=None, labels=labels).logits

    fwd_ok = equivalence_report(y_hf, y_mg, print0, mode="bf16" if args.mg_bf16 else "fp32")

    del y_mg, y_hf
    torch.cuda.empty_cache()
    return fwd_ok


if __name__ == "__main__":
    @dataclass
    class Args:
        load_dir: Path # something like script/test_megatron_jg/real_dragon/dragon-megatron-3B-2_run1/
        save_dir: Path
        iteration: int
        tokenizer_path: Optional[Path] = None
        mg_bf16: bool = False
        """Run the forward check with BOTH models in bf16 instead of both in fp32.
        fp32/fp32 (default) is the more sensitive test of the conversion itself.
        bf16/bf16 is what the saved checkpoint will actually run at, and matches
        the precision training used -- use it to sanity-check the saved artifact."""
        cpu: bool = False
        """Convert on CPU, without a GPU and without the forward-pass check.

        The Megatron model is never instantiated (TransformerEngine refuses to
        build its modules without CUDA); the checkpoint's tensors are read
        straight out of the torch_dist store -- see mg_cpu_loader.py. The exact
        weight-level audit still runs and still gates the exit code; only the
        logit comparison is skipped. Use it for a quick conversion, and the
        default GPU path when you need the converted checkpoint verified."""

    args = tyro.cli(Args)

    if not args.cpu and not torch.cuda.is_available():
        sys.exit("No CUDA device is visible. Re-run with --cpu for a check-free "
                 "CPU-only conversion (the weight audit still runs; the forward "
                 "comparison does not).")

    dist.init_process_group(
        backend='gloo' if args.cpu else 'nccl',
        init_method='env://',
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not args.cpu:
        torch.cuda.set_device(local_rank)

    print0 = lambda *args, **kwargs: print(*args, **kwargs) if dist.get_rank() == 0 else None
    print0(dist.get_world_size())

    # convert_mg_to_hf reads the TP group even at TP=1, so this is needed on both
    # paths; `pgs` and the cuda RNG tracker are only for building the MG model.
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    pgs = None if args.cpu else ProcessGroupCollection()
    if not args.cpu:
        pgs.tp = parallel_state.get_tensor_model_parallel_group()
        pgs.pp = parallel_state.get_pipeline_model_parallel_group()
        pgs.dp = parallel_state.get_data_parallel_group()
        pgs.cp = parallel_state.get_context_parallel_group()
        pgs.embd = parallel_state.get_embedding_group()
        pgs.ep = parallel_state.get_expert_model_parallel_group()
        pgs.tp_cp = parallel_state.get_tensor_and_context_parallel_group()
        pgs.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group()
        pgs.expt_tp = parallel_state.get_expert_tensor_parallel_group()
        pgs.tp_ep = parallel_state.get_expert_tensor_and_model_parallel_group()
        model_parallel_cuda_manual_seed(123456789, te_rng_tracker=True, inference_rng_tracker=True, use_cudagraphable_rng=True)

    if args.cpu:
        if str(args.load_dir) == 'X':
            sys.exit("--cpu needs a real checkpoint: the 'X' sentinel builds a random "
                     "Megatron model, which requires a GPU.")
        if args.mg_bf16:
            print0("Note: --mg_bf16 only selects the precision of the forward check, "
                   "which --cpu skips. Ignoring it; the saved checkpoint is bf16 either way.")
        model_mg, config_mg, vocab_size, wsize = load_mg_weights_cpu(
            args.load_dir, args.iteration, print0)
        print0(f"Loaded MG weights from {args.load_dir} at iteration {args.iteration} "
               f"(CPU, checkpoint dtype).")
    else:
        mg_dtype = torch.bfloat16 if args.mg_bf16 else torch.float32
        if str(args.load_dir) == 'X':
            model_mg, config_mg, vocab_size, wsize = load_random_mg_models(pgs, dist.get_world_size())
        else:
            model_mg, config_mg, vocab_size, wsize = load_merge_mg_models(
                args.load_dir, [args.iteration], pgs, params_dtype=mg_dtype)
        print0(f"Loaded MG model from {args.load_dir} at iteration {args.iteration} (params_dtype={mg_dtype}).")

    model_hf = load_hf(config_mg, vocab_size, wsize, device="cpu" if args.cpu else "cuda")
    print0("Loaded HF model.")

    with record_copies() as (touched, lossy):
        convert_mg_to_hf(config_mg, model_mg, model_hf.config, model_hf, tp_size=dist.get_world_size())
    print0("Copied MG model weights to HF model.")

    # -------------------------------------------------------------------------
    # CHECK 1: exact, on the weights. Independent of any kernel precision.
    # -------------------------------------------------------------------------
    weights_ok = audit_conversion(model_hf, touched, lossy, print0)

    if args.cpu:
        fwd_ok = None
        print0("\n=== NUMERICAL EQUIVALENCE === skipped: --cpu cannot run a forward pass. "
               "The exact weight-level audit above is unaffected.")
    else:
        fwd_ok = run_forward_check(args, config_mg, model_mg, model_hf, vocab_size, wsize, print0)


    fwd_status = 'SKIPPED' if fwd_ok is None else ('OK' if fwd_ok else 'FAILED')
    print0(f"\n=== SUMMARY === weights: {'OK' if weights_ok else 'FAILED'}   "
           f"forward: {fwd_status}")

    # -------------------------------------------------------------------------
    # Cast to bf16 and save. Everything below changes the HF model in place, so
    # no check may run after this point.
    # -------------------------------------------------------------------------
    cast_hf_to_bf16(model_hf)  # no-op if --mg_bf16 already cast it

    # save HF model
    if str(args.load_dir) != 'X' and str(args.save_dir) != 'X':
        model_name = Path(args.load_dir).name
        # out_dir = args.save_dir / model_name / f"iter_{args.iteration}"
        out_dir = args.save_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # model.
        model_hf.config.intra_doc_masking = False
        model_hf.config.dtype = torch.bfloat16
        model_hf.save_pretrained(out_dir)
        # tokenizer.
        # Tokenizer copied into the export alongside the weights. Overridable;
        # the default is the channels-v3 tokenizer this model was trained with.
        tokenizer_dir = Path(os.environ.get(
            "OLALA_TOKENIZER_DIR",
            "/data/home/gaetan.caillaut/dragon-sft/7A1B/tokenizers/tokenizer-channels-v3"))
        if not (tokenizer_dir / "tokenizer_config.json").is_file():
            raise FileNotFoundError(
                f"no tokenizer_config.json in {tokenizer_dir} "
                "(set OLALA_TOKENIZER_DIR)")
        tok = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        print0(len(tok))
        print0(vocab_size)
        tok.save_pretrained(out_dir)
        print0(f"Saved HF model (+ tokenizer) to {out_dir}.")

    dist.destroy_process_group()
    if not (weights_ok and fwd_ok is not False):
        sys.exit(1)

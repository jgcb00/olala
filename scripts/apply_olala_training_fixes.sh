#!/usr/bin/env bash
# apply_olala_training_fixes.sh — apply every fix from the Olala verl
# training investigation to an EXISTING environment, idempotently.
#
# Fixes applied (each skipped cleanly if already present):
#   1. vllm  olala/mamba3.py : refresh weight caches IN PLACE instead of
#      dropping them (fixes the FULL-cudagraph illegal-memory-access on
#      verl weight syncs). Applied ungated — no env var needed.
#   2. vllm  models/olala.py : widen OlalaGeodesicNorm scale/bias 0-dim -> [1]
#      (FSDP weight-sync shape compatibility).
#   3. mamba mamba3_mimo.py + mamba3_siso_combined.py : access
#      ctx.saved_tensors exactly once in backward (gradient checkpointing).
#   4. scattermoe parallel_experts.py : per-module dtype cast of inputs/gates
#      to the expert weight dtype (FSDP bf16 mixed precision).
#   5. checkpoint modeling_olala.py : public mamba_ssm kernel fallback,
#      angle_dt guard, and packed-batch (varlen) cu_seqlens support.
#   6. verl transfer_queue simple_storage.py : don't let one malformed ZMQ
#      frame kill a storage worker thread permanently. (optional hardening)
#
# NOT done here, and no longer needed anywhere: the checkpoint's 0-dim
# scale/bias tensors. modeling_olala.py widens them in memory after loading
# and saves them back 0-dim, so no checkpoint rewrite is involved.
#
# Usage:
#   ./apply_olala_training_fixes.sh \
#       --python     /path/to/venv/bin/python \
#       --mamba      /path/to/mamba/clone \
#       --scattermoe /path/to/scattermoe/clone \
#       --checkpoint /path/to/checkpoint/dir
#
# Any part may be omitted; the corresponding fixes are skipped. The vllm /
# verl fixes are located through the given python's installed packages.
# Every modified file gets a one-time .bak-olala-fixes backup next to it.
set -euo pipefail

PYBIN=python3 MAMBA="" SCATTER="" CKPT=""
while [[ $# -gt 0 ]]; do case $1 in
  --python)     PYBIN=$2;   shift 2;;
  --mamba)      MAMBA=$2;   shift 2;;
  --scattermoe) SCATTER=$2; shift 2;;
  --checkpoint) CKPT=$2;    shift 2;;
  -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  *) echo "unknown arg: $1 (see --help)"; exit 1;;
esac; done

export OLALA_FIX_MAMBA="$MAMBA" OLALA_FIX_SCATTER="$SCATTER" OLALA_FIX_CKPT="$CKPT"
exec "$PYBIN" - <<'PY'
import importlib.util, os, shutil, sys
from pathlib import Path

results = []

def patch(name, path, fn, required=False):
    """required=True: a missing target is a FAIL, not a SKIP.

    Use it wherever the file MUST be there once its package is installed. A
    silent SKIP there means a rename upstream quietly disables the fix while
    the applier still exits 0 -- exactly how the dragon->olala rename turned
    off both vllm fixes unnoticed.
    """
    p = Path(path) if path else None
    if not p or not p.exists():
        status = "FAIL" if required else "SKIP"
        results.append((name, status, f"not found: {path or '(no path given)'}"))
        return
    text = p.read_text()
    status, new = fn(text)
    if status == "apply":
        bak = Path(str(p) + ".bak-olala-fixes")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(new)
        try:
            compile(new, str(p), "exec")
        except SyntaxError as e:
            shutil.copy2(bak, p)
            results.append((name, "FAIL", f"patched file failed to compile, restored backup: {e}"))
            return
        results.append((name, "APPLIED", str(p)))
    else:
        results.append((name, status, str(p)))

def pkg_dir(mod):
    spec = importlib.util.find_spec(mod)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])

# ---------------------------------------------------------------- fix 1
CACHE_REFRESH = '''\
        # OLALA fix: refresh the snapshots IN PLACE. FULL-mode CUDA graphs
        # capture _decode (inside the olala_mamba3 op) and bake the snapshot
        # tensors' device addresses into every captured decode graph; dropping
        # the cache frees memory those graphs still read -> illegal memory
        # access after RL weight syncs. Rewriting the same storage keeps the
        # baked pointers valid and propagates the new weights. The view
        # entries alias the in-place-updated params and need no refresh.
        with torch.inference_mode():
            cw = self._decode_const_w
            if cw is not None:
                cw[2].copy_(rearrange(self.in_proj_mimo_x, "h r p -> r h p"))
                cw[3].copy_(rearrange(self.in_proj_mimo_z, "h r p -> r h p"))
                cw[4].copy_(rearrange(self.out_proj_mimo, "h r p -> r h p"))
            pw = self._prefill_const_w
            if pw is not None:
                for dst, src in zip(pw, (self.C_bias, self.B_bias,
                                         self.in_proj_mimo_x,
                                         self.in_proj_mimo_z,
                                         self.out_proj_mimo, self.D)):
                    dst.copy_(src)'''

def fix_vllm_cache(text):
    if "cw[2].copy_(rearrange(self.in_proj_mimo_x" in text:
        return "ALREADY", None
    fn = ("def invalidate_weight_caches" if "def invalidate_weight_caches" in text
          else "def refresh_weight_caches" if "def refresh_weight_caches" in text else None)
    if fn is None:
        return "FAIL: no weight-cache method found", None
    start = text.index(fn)
    end = text.index("\n    def ", start + 10)
    seg = text[start:end]
    old = "        self._prefill_const_w = None\n        self._decode_const_w = None"
    if old not in seg:
        return "FAIL: unexpected cache-invalidate body (hand-check needed)", None
    return "apply", text[:start] + seg.replace(old, CACHE_REFRESH) + text[end:]

# ---------------------------------------------------------------- fix 2
def widen(text, one, zero, one_new, zero_new):
    a = f"self.scale = nn.Parameter(torch.tensor({one}))"
    b = f"self.bias = nn.Parameter(torch.tensor({zero}))"
    an = f"self.scale = nn.Parameter(torch.tensor({one_new}))"
    bn = f"self.bias = nn.Parameter(torch.tensor({zero_new}))"
    if an in text or f"torch.tensor([{one}])" in text:
        return "ALREADY", None
    if a not in text or b not in text:
        return "FAIL: scale/bias declaration not found", None
    return "apply", text.replace(a, an).replace(b, bn)

def fix_vllm_widen(text):
    return widen(text, "1.0", "0.0", "[1.0]", "[0.0]")

# ---------------------------------------------------------------- fix 3
def fix_saved_tensors(text):
    if "saved = ctx.saved_tensors" in text:
        return "ALREADY", None
    guard = "if len(ctx.saved_tensors) == 0:"
    if guard not in text:
        return "FAIL: len(ctx.saved_tensors) guard not found", None
    text = text.replace(
        "        " + guard,
        "        saved = ctx.saved_tensors\n        if len(saved) == 0:", 1)
    if ") = ctx.saved_tensors" not in text:
        return "FAIL: tuple unpack not found", None
    return "apply", text.replace(") = ctx.saved_tensors", ") = saved", 1)

# ---------------------------------------------------------------- fix 4
SCATTER_CAST = '''\
        # OLALA fix: the triton kernels require activations, gates and expert
        # weights to share a dtype: the compute dtype follows the ACTIVATIONS.
        # Keying it on self.weight.dtype instead is not phase-stable under
        # FSDP mixed precision: the weight's visible dtype can differ between
        # the original forward (bf16 unsharded views) and gradient-checkpoint
        # recompute (fp32 master params), flipping downstream activation
        # dtypes and tripping check_recomputed_tensors_match. Input dtype is
        # set upstream and identical in both phases. No-op when dtypes agree.
        weight = self.weight
        if weight.dtype != inputs.dtype:
            weight = weight.to(inputs.dtype)
        if gates is not None and gates.dtype != inputs.dtype:
            gates = gates.to(inputs.dtype)

'''

def fix_scattermoe(text):
    if "compute dtype follows the ACTIVATIONS" in text:
        return "ALREADY", None
    # Handles three states: pristine upstream, or either variant of the
    # earlier weight-dtype cast (migrated by replacing everything between the
    # forward signature and the parallel_linear call).
    sig = ("    def forward(self, inputs, k, sorted_expert_idxs, sorted_scattered_idxs,\n"
           "                expert_offsets,\n"
           "                gates=None, grouped_in=False, grouped_out=False):\n")
    call_old = "            inputs, self.weight.permute(0, 2, 1), k,"
    call_new = "            inputs, weight.permute(0, 2, 1), k,"
    if sig not in text:
        return "FAIL: ParallelExperts.forward signature not found", None
    sig_end = text.index(sig) + len(sig)
    call_anchor = "        results = parallel_linear("
    call_idx = text.find(call_anchor, sig_end)
    if call_idx < 0:
        return "FAIL: parallel_linear call not found after signature", None
    text = text[:sig_end] + SCATTER_CAST + text[call_idx:]
    if call_old in text:
        text = text.replace(call_old, call_new, 1)
    elif call_new not in text:
        return "FAIL: parallel_linear argument line not recognized", None
    return "apply", text

SCATTER_BWD_CAST = '''\
             gates, output_expanded) = ctx.saved_tensors
            # OLALA fix: the incoming gradient can arrive in a promoted dtype
            # (e.g. fp32 leaking back from a mixed-precision shared-expert /
            # gating path) while the saved forward tensors are bf16; the
            # kernels and matmuls below need uniform dtypes. No-op when they
            # already agree.
            if grad_out.dtype != x.dtype:
                grad_out = grad_out.to(x.dtype)
'''

def fix_scattermoe_bwd(text):
    if "incoming gradient can arrive in a promoted dtype" in text:
        return "ALREADY", None
    anchor = "             gates, output_expanded) = ctx.saved_tensors\n"
    if anchor not in text:
        return "FAIL: backward saved_tensors unpack not found", None
    return "apply", text.replace(anchor, SCATTER_BWD_CAST, 1)

# ---------------------------------------------------------------- fix 5
FALLBACK = '''\
except ImportError:
    # Fallback: the public state-spaces/mamba package ships the same Mamba-3
    # MIMO kernel (mamba3_mimo). It takes RAW angles (applies the cumsum
    # itself -> angle_dt = None signals that in forward) and has no public
    # decode/step kernel -> generate with use_cache=False.
    angle_dt = None
    mamba3_step_fn = None
    apply_rotary_qk_inference_fwd = None
    try:
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as _mamba3_mimo_public

        def mamba3_tilelang(*, return_state=False, **kwargs):
            out = _mamba3_mimo_public(return_state=return_state, **kwargs)
            return out if return_state else (out, None)

        print("dragon_mamba3_fast_step not found: using public mamba_ssm mamba3_mimo kernel (no decode kernel -> generate with use_cache=False)")
    except ImportError:
        print("dragon_mamba3_fast_step not found")'''

CU_SEQLENS = '''\
        is_prefill = cache_params is not None

        # OLALA fix: packed-batch (varlen) support. Trainers like verl pack
        # multiple documents into one (1, T) row; run dense, SSM state leaks
        # across document boundaries and the dense backward requires
        # T %% chunk_size == 0. Derive cu_seqlens from position_ids resets
        # (pos == 0 marks each document start) and use the public kernel's
        # varlen forward/backward. Public kernel only (angle_dt is None).
        _mamba3_cu_seqlens = None
        if angle_dt is None and cache_params is None and batch == 1:
            _mamba3_cu_seqlens = kwargs.get("cu_seqlens", None)
            if _mamba3_cu_seqlens is None:
                _pos = kwargs.get("position_ids", None)
                if _pos is not None:
                    _starts = (_pos[0] == 0).nonzero(as_tuple=False).flatten()
                    if _starts.numel() >= 1:
                        _mamba3_cu_seqlens = torch.cat(
                            [_starts.to(torch.int32),
                             torch.tensor([q_len], dtype=torch.int32,
                                          device=_starts.device)])
            if _mamba3_cu_seqlens is not None:
                _mamba3_cu_seqlens = _mamba3_cu_seqlens.to(
                    device=hidden_states.device, dtype=torch.int32)'''

def fix_ckpt_modeling(text):
    # (f-relocate) an earlier applier could anchor the dtype pin into
    # OlalaMonoBlock.__init__ instead of forward(): on pruning-support
    # checkpoints the "Skip-mixer support" comment exists in BOTH, and the
    # single-occurrence replace hit __init__ first. There autocast is always
    # off, so the pin is dead code — and forward stays unpinned (FSDP2
    # CheckpointError: bf16 saved vs fp32 recomputed). Strip the misplaced
    # copy; the insertion below re-adds it at the forward anchor.
    _pin_head = "        # OLALA fix: Phase-stable compute dtype under FSDP2 mixed\n"
    _pin_tail = "            hidden_states = hidden_states.to(torch.get_autocast_dtype(\"cuda\"))\n"
    _init_follow = ("        # Skip-mixer support: if this layer is in config.skip_mixer_layers,\n"
                    "        # replace the mixer-side submodules with parameter-free placeholders.\n")
    _i = text.find(_pin_head)
    if _i != -1:
        _e = text.find(_pin_tail, _i)
        if _e != -1:
            _e += len(_pin_tail)
            if text[_e:_e + len(_init_follow)] == _init_follow:
                text = text[:_i] + text[_e:]
    if all(m in text for m in (
        "_mamba3_cu_seqlens", "angle_dt is not None", "_mamba3_flat_batch",
        "Phase-stable compute dtype", "return output.to(x.dtype)",
        "packed-batch (varlen) document masking",
    )):
        return "ALREADY", None
    # (a) public-kernel fallback, only if absent
    plain = 'except ImportError:\n    print("dragon_mamba3_fast_step not found")'
    if "mamba3_mimo as _mamba3_mimo_public" not in text:
        if plain not in text:
            return "FAIL: fast_step import block not recognized", None
        text = text.replace(plain, FALLBACK, 1)
    # (b) guard the private pre-cumsum
    if "if angle_dt is not None:" not in text:
        old = "        angle = angle_dt(angle, dt)"
        if old not in text:
            return "FAIL: angle_dt call not found", None
        text = text.replace(old,
            "        if angle_dt is not None:\n"
            "            # private kernel wants pre-cumsummed angles; the public\n"
            "            # mamba_ssm kernel applies the cumsum itself\n"
            "            angle = angle_dt(angle, dt)", 1)
    # (c) cu_seqlens derivation
    if "_mamba3_cu_seqlens" not in text:
        anchor = "        is_prefill = cache_params is not None"
        if text.count(anchor) != 1:
            return f"FAIL: expected 1 is_prefill anchor, found {text.count(anchor)}", None
        text = text.replace(anchor, CU_SEQLENS.replace("%%", "%"), 1)
    # (d) pass it to the kernel
    if '{"cu_seqlens": _mamba3_cu_seqlens}' not in text:
        anchor = "            return_state=is_prefill,\n        )"
        if text.count(anchor) != 1:
            return f"FAIL: expected 1 kernel-call anchor, found {text.count(anchor)}", None
        text = text.replace(anchor,
            "            return_state=is_prefill,\n"
            '            **({"cu_seqlens": _mamba3_cu_seqlens}\n'
            "               if _mamba3_cu_seqlens is not None else {}),\n"
            "        )", 1)
    # (e) geodesic norm: pin output dtype to the residual dtype
    if "return output.to(x.dtype)" not in text:
        anchor = ("        output = x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)\n"
                  "        return output")
        if anchor not in text:
            return "FAIL: geodesic output/return not found", None
        text = text.replace(anchor, anchor.replace(
            "        return output",
            "        # OLALA fix: theta inherits the scale/bias params' dtype via\n"
            "        # promotion and can differ from the residual dtype under mixed\n"
            "        # precision; pin the output. No-op when dtypes agree.\n"
            "        return output.to(x.dtype)"), 1)
    # (f) block-entry autocast dtype pin (fixes FSDP2 cast_forward_inputs vs
    #     gradient-checkpoint recompute mismatch)
    if "Phase-stable compute dtype" not in text:
        pin = (
            "        # OLALA fix: Phase-stable compute dtype under FSDP2 mixed\n"
            "        # precision + HF gradient checkpointing. cast_forward_inputs\n"
            "        # runs inside this (checkpointed) __call__ in the original\n"
            "        # forward but not at the same point in the recompute, so the\n"
            "        # block would recompute from the saved PRE-CAST fp32 input and\n"
            "        # every derived activation dtype would mismatch. Autocast state\n"
            "        # IS replayed in recompute, so keying on it is stable.\n"
            "        if (\n"
            "            torch.is_autocast_enabled()\n"
            "            and hidden_states.is_floating_point()\n"
            "            and hidden_states.dtype != torch.get_autocast_dtype(\"cuda\")\n"
            "        ):\n"
            "            hidden_states = hidden_states.to(torch.get_autocast_dtype(\"cuda\"))\n")
        # a1 must be the FORWARD's skip-mixer comment (two lines — the
        # second line disambiguates it from the near-identical comment in
        # __init__ on pruning-support checkpoints; see f-relocate above).
        a1 = ("        # Skip-mixer support: if this layer is in config.skip_mixer_layers,\n"
              "        # the mixer phase is bypassed entirely — residual passes through and\n")
        a2 = "        # MIXER.\n        residual = hidden_states\n"
        if a1 in text:
            text = text.replace(a1, pin + a1, 1)
        elif a2 in text:
            text = text.replace(a2, pin + a2, 1)
        else:
            return "FAIL: block forward start not found for dtype pin", None
    # (g) dense -> varlen flatten. The dense TileLang kernels are compiled
    # per sequence length (no T.dynamic) and their backward requires
    # L % chunk_size == 0; the varlen kernels declare S/NS dynamic (one
    # compile serves every shape, fwd+bwd, no divisibility constraint).
    # Flatten (B, L) into a packed (1, B*L) batch with row boundaries as
    # cu_seqlens — numerically identical, each row an independent zero-start
    # sequence. Measured: update_actor 842s -> 147s / 1300s -> 112s.
    # Coexists with any earlier pad-only fix (which becomes a dead branch on
    # the public-kernel path and keeps covering the private-kernel path).
    if "_mamba3_flat_batch" not in text:
        # Anchor on the .to(...) block that the varlen step above inserts. It
        # emits the closing paren on the SAME line; a modeling that already
        # carried an earlier hand-applied varlen fix has it on its own line.
        # Accept both, or this step FAILs on every freshly exported modeling
        # (which is what forced the modeling-swap workaround in the installer).
        anchor_2line = ("                _mamba3_cu_seqlens = _mamba3_cu_seqlens.to(\n"
                        "                    device=hidden_states.device, dtype=torch.int32)\n")
        anchor_3line = ("                _mamba3_cu_seqlens = _mamba3_cu_seqlens.to(\n"
                        "                    device=hidden_states.device, dtype=torch.int32\n"
                        "                )\n")
        anchor = next((a for a in (anchor_2line, anchor_3line) if a in text), None)
        if anchor is None:
            return "FAIL: flatten insertion anchor not found", None
        flat_block = (
            "\n"
            "        # OLALA fix: dense -> varlen flatten (see applier notes).\n"
            "        _mamba3_flat_batch = 0\n"
            "        _mamba3_flat_len = q_len\n"
            "        if (\n"
            "            cache_params is None\n"
            "            and _mamba3_cu_seqlens is None\n"
            "            and angle_dt is None\n"
            "            and not (self.config.complete_slw and self.config.slw_wsize > 128)\n"
            "        ):\n"
            "            _mamba3_flat_batch = batch\n"
            "            _mamba3_cu_seqlens = torch.arange(\n"
            "                0, (batch + 1) * q_len, q_len,\n"
            "                device=hidden_states.device, dtype=torch.int32,\n"
            "            )\n"
            "            if batch > 1:\n"
            "                hidden_states = hidden_states.reshape(1, batch * q_len, -1)\n"
            "                batch, q_len, _ = hidden_states.shape\n")
        text = text.replace(anchor, anchor + flat_block, 1)
        sl_anchor = ('        y = rearrange(y, "b l h p -> b l (h p)")\n'
                     "        if self.config.mamba3_postgate_norm:")
        if sl_anchor not in text:
            return "FAIL: unflatten anchor not found", None
        text = text.replace(sl_anchor,
            "        if _mamba3_flat_batch > 1:\n"
            "            # Undo the dense -> varlen flatten.\n"
            "            y = y.view(_mamba3_flat_batch, _mamba3_flat_len, *y.shape[2:])\n\n"
            + sl_anchor, 1)
    # (h) packed-batch DOCUMENT MASKING for the attention layers. verl's
    # use_remove_padding path feeds one packed (1, T) row holding many
    # documents (position_ids reset to 0 at each doc start). The mamba3
    # mixers already derive their own cu_seqlens from those resets, but with
    # intra_doc_masking off the attention mixers ran DENSE flash attention
    # over the whole packed row — every document attends to all previous
    # ones. vLLM rollout computes each sequence alone, so this is a pure
    # training/inference logprob mismatch: rollout_probs_diff blows up
    # whenever use_remove_padding/use_dynamic_bsz is on. Fix: derive
    # cu_seqlens once at model level and hand it down; every attention path
    # switches to flash_attn_varlen when it is set. Single-document rows
    # keep cu_seqlens=None, so the unpacked path is untouched.
    # Upstream implemented this natively from iter_0099518 on: every attention
    # path now derives `_varlen_attn = intra_doc_masking or kwargs["unpadded"]`
    # and switches to flash_attn_varlen_func(cu_seqlens=...) under fa2/fa3. The
    # `_varlen_attn` probe detects that and leaves the file alone -- without it
    # this step FAILs on a modern export, because the `if not
    # self.config.intra_doc_masking:` shape it patches no longer exists.
    if ("packed-batch (varlen) document masking" not in text
            and "_varlen_attn" not in text):
        c1 = "            if not self.config.intra_doc_masking:"
        c2 = "            if not self.config.intra_doc_masking and not self.config.complete_slw:"
        c3 = "            assert not self.config.intra_doc_masking\n"
        if c1 not in text and c2 not in text:
            return "FAIL: attention doc-mask conditions not found", None
        text = text.replace(
            c1, "            if not self.config.intra_doc_masking and cu_seqlens is None:")
        text = text.replace(
            c2, "            if not self.config.intra_doc_masking and not self.config.complete_slw and cu_seqlens is None:")
        text = text.replace(
            c3, "            assert not self.config.intra_doc_masking and cu_seqlens is None, \"eager attention has no document masking\"\n")
        derive_anchor = "        all_hidden_states = () if output_hidden_states else None"
        if text.count(derive_anchor) != 1:
            return "FAIL: model-level doc-mask anchor not unique", None
        derive_block = (
            "        # OLALA fix: packed-batch (varlen) document masking for the ATTENTION\n"
            "        # layers. Training frameworks with use_remove_padding (verl) feed one\n"
            "        # packed (1, T) row holding many documents, position_ids resetting to\n"
            "        # 0 at each document start. The mamba3 mixers already derive their own\n"
            "        # cu_seqlens from those resets, and the diff-tpa token-shift masks at\n"
            "        # doc starts — but with intra_doc_masking off the attention mixers ran\n"
            "        # DENSE flash attention over the whole packed row, letting every\n"
            "        # document attend to all previous ones. That is a training/inference\n"
            "        # logprob mismatch (rollout_probs_diff blowup) whenever packing is on.\n"
            "        # Derive cu_seqlens once here and hand it down; the attention paths\n"
            "        # switch to flash_attn_varlen whenever it is set. Single-document\n"
            "        # rows (one reset) keep cu_seqlens=None — dense attention is already\n"
            "        # exact there, so the validated unpacked path is untouched.\n"
            "        if cu_seqlens is not None:\n"
            "            # Normalize an externally provided cu_seqlens: verl passes its\n"
            "            # packed nested-offsets (int64) whenever the forward signature\n"
            "            # accepts cu_seqlens -- but never max_seqlen, and flash_attn\n"
            "            # requires both. Same convention: one entry per document start\n"
            "            # plus the total length.\n"
            "            cu_seqlens = cu_seqlens.to(device=hidden_states.device, dtype=torch.int32)\n"
            "            if max_seqlen is None:\n"
            "                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())\n"
            "        elif (\n"
            "            past_key_values is None\n"
            "            and B == 1\n"
            "            and position_ids is not None\n"
            "        ):\n"
            "            _doc_starts = (position_ids[0] == 0).nonzero(as_tuple=False).flatten()\n"
            "            if _doc_starts.numel() > 1:\n"
            "                cu_seqlens = torch.cat(\n"
            "                    [\n"
            "                        _doc_starts.to(torch.int32),\n"
            "                        torch.tensor(\n"
            "                            [position_ids.shape[-1]],\n"
            "                            dtype=torch.int32,\n"
            "                            device=_doc_starts.device,\n"
            "                        ),\n"
            "                    ]\n"
            "                ).to(device=hidden_states.device, dtype=torch.int32)\n"
            "                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())\n"
            "\n"
        )
        text = text.replace(derive_anchor, derive_block + derive_anchor, 1)
    return "apply", text

# ---------------------------------------------------------------- fix 5b
# verl colocate weight-sync ZMQ socket: hardcoded shared /tmp + a Ray job id
# that restarts at 01000000 every fresh cluster means a stale socket left by
# ANOTHER user (sticky /tmp, not removable) permanently blocks new runs with
# "ZMQError: Address already in use". Use the per-user temp dir (TMPDIR)
# on BOTH the sender and receiver sides (they must agree).
def _fix_zmq_tmpdir(text, old_line, new_lines):
    if "rl-colocate-zmq" not in text:
        return "SKIP", None
    if "_tempfile.gettempdir()" in text:
        return "ALREADY", None
    if old_line not in text:
        return "FAIL: zmq handle line not found", None
    return "apply", text.replace(old_line, new_lines, 1)

def fix_zmq_sender(text):
    return _fix_zmq_tmpdir(
        text,
        '        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"',
        "        import tempfile as _tempfile\n"
        "        # OLALA fix: per-user temp dir (TMPDIR); see applier notes.\n"
        '        self.zmq_handle = (\n'
        '            f"ipc://{_tempfile.gettempdir()}/rl-colocate-zmq-{job_id}"\n'
        '            f"-replica-{self.replica_rank}-rank-{local_rank}.sock"\n'
        "        )",
    )

def fix_zmq_receiver(text):
    return _fix_zmq_tmpdir(
        text,
        '        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{trainer_rank}.sock"',
        "        import tempfile as _tempfile\n"
        "        # OLALA fix: per-user temp dir (TMPDIR); must match sender side.\n"
        '        return (\n'
        '            f"ipc://{_tempfile.gettempdir()}/rl-colocate-zmq-{job_id}"\n'
        '            f"-replica-{replica_rank}-rank-{trainer_rank}.sock"\n'
        "        )",
    )

# ---------------------------------------------------------------- fix 6
def fix_transfer_queue(text):
    if "dropping malformed" in text:
        return "ALREADY", None
    old = ("                request_msg = ZMQMessage.deserialize(serialized_msg)\n"
           "                operation = request_msg.request_type")
    if old not in text:
        return "FAIL: deserialize anchor not found", None
    return "apply", text.replace(old,
        "                # OLALA fix: a malformed frame must not kill the worker\n"
        "                # thread (every later put/get would time out forever).\n"
        "                try:\n"
        "                    request_msg = ZMQMessage.deserialize(serialized_msg)\n"
        "                except Exception as e:\n"
        "                    logger.error(\n"
        "                        f\"[{self.storage_unit_id}]: dropping malformed \"\n"
        "                        f\"request frame: {type(e).__name__}: {e}\")\n"
        "                    continue\n"
        "                operation = request_msg.request_type", 1)

def fix_tq_controller(text):
    if "dropping malformed request frame" in text:
        return "ALREADY", None
    old = ("            messages = self.request_handle_socket.recv_multipart(copy=False)\n"
           "            identity = messages.pop(0)\n"
           "            serialized_msg = messages\n"
           "            request_msg = ZMQMessage.deserialize(serialized_msg)")
    if old not in text:
        return "FAIL: controller request loop anchor not found", None
    return "apply", text.replace(old,
        "            messages = self.request_handle_socket.recv_multipart(copy=False)\n"
        "            identity = messages.pop(0)\n"
        "            serialized_msg = messages\n"
        "            # OLALA fix: a malformed frame must not kill the controller\n"
        "            # loop — every client would then block forever.\n"
        "            try:\n"
        "                request_msg = ZMQMessage.deserialize(serialized_msg)\n"
        "            except Exception as e:\n"
        "                logger.error(\n"
        "                    f\"controller: dropping malformed request frame: \"\n"
        "                    f\"{type(e).__name__}: {e}\")\n"
        "                continue", 1)

# ---------------------------------------------------------------- run
vllm_dir = pkg_dir("vllm")
tq_dir = pkg_dir("transfer_queue")
mamba = os.environ.get("OLALA_FIX_MAMBA") or None
scatter = os.environ.get("OLALA_FIX_SCATTER") or None
ckpt = os.environ.get("OLALA_FIX_CKPT") or None

patch("vllm cudagraph weight-cache refresh",
      vllm_dir and vllm_dir / "model_executor/layers/mamba/olala/mamba3.py", fix_vllm_cache,
      required=True)
patch("vllm GeodesicNorm 0-dim -> [1]",
      vllm_dir and vllm_dir / "model_executor/models/olala.py", fix_vllm_widen,
      required=True)
patch("mamba saved_tensors (mimo)",
      mamba and Path(mamba) / "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py", fix_saved_tensors)
patch("mamba saved_tensors (siso)",
      mamba and Path(mamba) / "mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py", fix_saved_tensors)
patch("scattermoe dtype cast",
      scatter and Path(scatter) / "scattermoe/parallel_experts.py", fix_scattermoe)
patch("scattermoe backward grad cast",
      scatter and Path(scatter) / "scattermoe/parallel_experts.py", fix_scattermoe_bwd)
patch("checkpoint modeling varlen + fallback",
      ckpt and Path(ckpt) / "modeling_olala.py", fix_ckpt_modeling,
      required=bool(ckpt))
patch("verl transfer_queue hardening",
      tq_dir and tq_dir / "storage/simple_storage.py", fix_transfer_queue)
patch("verl transfer_queue controller hardening",
      tq_dir and tq_dir / "controller.py", fix_tq_controller)
verl_dir = pkg_dir("verl")
patch("verl zmq socket per-user tmpdir (sender)",
      verl_dir and verl_dir / "workers/rollout/vllm_rollout/vllm_rollout.py", fix_zmq_sender)
patch("verl zmq socket per-user tmpdir (receiver)",
      verl_dir and verl_dir / "workers/rollout/vllm_rollout/utils.py", fix_zmq_receiver)

width = max(len(n) for n, _, _ in results)
fail = False
print()
for name, status, detail in results:
    print(f"  {name:<{width}}  {status:<9} {detail}")
    fail |= status.startswith("FAIL")
print()
if ckpt:
    print("NOTE: this patches checkpoint *code* only. 0-dim scale/bias")
    print("      weights need nothing: modeling_olala.py widens them after")
    print("      loading and writes them back 0-dim on save.")
sys.exit(1 if fail else 0)
PY

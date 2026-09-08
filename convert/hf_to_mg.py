"""The inverse of `utils_convert.convert_mg_to_hf`: HF weights -> Megatron.

Written for the pipeline `Megatron -> HF -> (TRL) -> HF -> Megatron`, so it
assumes the HF checkpoint has the layout convert_mg_to_hf produces, possibly
with the weights moved by training in between.

Why an exact inverse exists at all: instrumenting the CPU loader's tensor tree
during a forward conversion shows convert_mg_to_hf READS all 19184 Megatron
tensors -- nothing is dropped, so nothing has to be invented here. The single
exception is the padded vocab rows, which are sliced off inside
mg_cpu_loader.load_mg_weights_cpu before the tree is built; MgSink restores
those from the reference (see mg_cpu_writer.restore_vocab_padding).

This reads the HF weights straight out of safetensors rather than instantiating
`OlalaForCausalLM`: the mapping below is written against tensor names, so the
model class buys nothing and would cost ~25 GiB of throwaway random init (plus a
scattermoe dependency the conversion does not need).

Not implemented, deliberately: the "T" and "g" mixers and the DDL path. No
checkpoint here uses them, so an inverse for them would be untested code -- and
"T" is already dead in the forward direction (it references an undefined
`tp_src`/`tp_group`). They raise rather than silently mis-convert.
"""

import json
from pathlib import Path

import torch
from einops import rearrange
from safetensors import safe_open


class HfWeights:
    """Lazy read-only view over one or many safetensors shards."""

    def __init__(self, hf_dir):
        hf_dir = Path(hf_dir)
        index = hf_dir / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text())["weight_map"]
            self._where = {k: hf_dir / v for k, v in mapping.items()}
        else:
            single = hf_dir / "model.safetensors"
            assert single.is_file(), f"no model.safetensors(.index.json) in {hf_dir}"
            with safe_open(single, framework="pt") as f:
                self._where = {k: single for k in f.keys()}
        self._handles = {}

    def _handle(self, path):
        h = self._handles.get(path)
        if h is None:
            h = self._handles[path] = safe_open(path, framework="pt")
        return h

    def __contains__(self, key):
        return key in self._where

    def __getitem__(self, key):
        try:
            path = self._where[key]
        except KeyError:
            raise KeyError(f"'{key}' is not in the HF checkpoint") from None
        return self._handle(path).get_tensor(key)

    def keys(self):
        return self._where.keys()


def _attn_dims(config_mg):
    """The super-head geometry convert_mg_to_hf slices linear_in with.

    Mirrors dragon_attention_v2.py: heads are grouped four-to-a-super-head, and
    each super head lays out [4 q heads | 1 noise head | 3 signal heads].
    """
    Dk = config_mg.kv_channels
    r = config_mg.tpa_rank
    alpha = 1 if config_mg.token_shift else 0
    gate = Dk if config_mg.gate_attn else 0
    H_super = config_mg.num_attention_heads // 4
    return Dk, r, alpha, gate, H_super


def _rebuild_linear_in_v(hf, p, config_mg, in_f):
    """Re-interleave the "V" mixer's fused input projection.

    convert_mg_to_hf splits `linear_in.weight` (H_super*P, in_f) into q / A_k /
    alpha_k / A_v / alpha_v / lambda / gate by viewing it as (H_super, P, in_f)
    and slicing three contiguous bands. This puts the bands back in that order.
    """
    Dk, r, alpha, gate, H = _attn_dims(config_mg)

    # band 1: four q heads per super head, flattened to (H*4, Dk, in_f) forward
    W_q = hf[f"{p}.mixer.c_q.weight"].view(H * 4, Dk, in_f)
    heads = rearrange(W_q, "(H n) d f -> H (n d) f", n=4)              # (H, 4*Dk, in_f)

    # band 2: one noise head per super head, [A_k | alpha_k | A_v | alpha_v | lambda]
    parts = [hf[f"{p}.mixer.W_A_k.weight"].view(H, r, in_f)]
    if alpha:
        parts.append(hf[f"{p}.mixer.shift_proj_k.weight"].view(H, alpha, in_f))
    parts.append(hf[f"{p}.mixer.W_A_v.weight"].view(H, r, in_f))
    if alpha:
        parts.append(hf[f"{p}.mixer.shift_proj_v.weight"].view(H, alpha, in_f))
    parts.append(hf[f"{p}.mixer.lambda_proj.weight"].view(H, 1, in_f))
    noise = torch.cat(parts, dim=1)                                    # (H, 2r+2a+1, in_f)

    # band 3: three signal heads per super head, gate only
    if gate:
        W_gate = hf[f"{p}.gate_proj.weight"].view(H * 3, gate, in_f)
        signal = rearrange(W_gate, "(H n) d f -> H (n d) f", n=3)      # (H, 3*Dk, in_f)
    else:
        signal = heads.new_zeros((H, 0, in_f))

    return torch.cat([heads, noise, signal], dim=1).reshape(-1, in_f)


def convert_hf_to_mg(config_mg, config_hf, hf, sink, print0=print):
    """Fill `sink` (mg_cpu_writer.MgSink) from `hf` (HfWeights).

    Returns {key: n_rows} for the vocab tensors, so the caller knows how many
    leading rows came from HF and can restore the padding above them.
    """
    for bad, why in (("use_ddl", "the DDL path"),):
        if getattr(config_mg, bad, False):
            raise NotImplementedError(f"HF->MG does not implement {why} (config_mg.{bad})")

    n_experts = config_mg.num_moe_experts
    geodesic = bool(config_hf.geodesic_update)

    for i, layer_type in enumerate(config_mg.layers_mixer_config):
        mg = f"decoder.layers.{i}"
        p = f"model.layers.{i}"

        if layer_type == "V":
            in_f = config_mg.hidden_size
            sink[f"{mg}.mixer.linear_in.weight"] = _rebuild_linear_in_v(hf, p, config_mg, in_f)
            sink[f"{mg}.mixer.linear_BkBv.weight"] = torch.cat(
                [hf[f"{p}.mixer.W_B_k.weight"], hf[f"{p}.mixer.W_B_v.weight"]], dim=0)
            sink[f"{mg}.mixer.q_layernorm.weight"] = hf[f"{p}.mixer.q_norm.norm.weight"]
            sink[f"{mg}.mixer.k_layernorm.weight"] = hf[f"{p}.mixer.k_norm.norm.weight"]
            if not geodesic:
                sink[f"{mg}.mixer.linear_in.layer_norm_weight"] = \
                    hf[f"{p}.input_norm.norm.weight"]
            # The forward pass squeezes this to (num_heads,) through a couple of
            # intra_doc_masking-dependent squeezes; reshaping to whatever the
            # reference holds inverts every one of those branches at once.
            key = f"{mg}.mixer.softmax_scaler"
            scaler = hf[f"{p}.mixer.softmax_scaler"]
            assert scaler.numel() == sink[key].numel(), (
                f"{key}: HF has {scaler.numel()} elements, Megatron wants {sink[key].numel()}")
            sink[key] = scaler.reshape(sink.shape(key))

        elif layer_type == "M":
            sink[f"{mg}.mixer.in_proj.weight"] = hf[f"{p}.mixer.in_proj.weight"]
            sink[f"{mg}.mixer.in_proj_dyn.weight"] = hf[f"{p}.mixer.in_proj_dyn.weight"]
            if not geodesic:
                sink[f"{mg}.mixer.in_proj.layer_norm_weight"] = \
                    hf[f"{p}.input_norm.norm.weight"]
            for name in ("B_bias", "C_bias", "in_proj_mimo_x", "in_proj_mimo_z",
                         "out_proj_mimo", "dt_bias", "D"):
                sink[f"{mg}.mixer.{name}"] = hf[f"{p}.mixer.{name}"]
            sink[f"{mg}.mixer.B_norm.weight"] = hf[f"{p}.mixer.B_norm.norm.weight"]
            sink[f"{mg}.mixer.C_norm.weight"] = hf[f"{p}.mixer.C_norm.norm.weight"]
            if config_hf.mamba3_postgate_norm:
                sink[f"{mg}.mixer.output_norm.weight"] = hf[f"{p}.mixer.output_norm.norm.weight"]

        else:
            raise NotImplementedError(
                f"layer {i}: HF->MG does not implement mixer type {layer_type!r}. "
                f"Only 'V' and 'M' are covered -- see this module's docstring.")

        # -- mixer output projection & optional group norm ------------------
        sink[f"{mg}.mixer_proj.weight"] = hf[f"{p}.mixer_proj.weight"]
        if config_hf.mixer_gn:
            key = f"{mg}.mixer_norm_scalers"
            sink[key] = hf[f"{p}.mixer_group_norm.weight"].reshape(sink.shape(key))

        # -- MLP / MoE ------------------------------------------------------
        if n_experts is not None:
            sink[f"{mg}.mlp.router.weight"] = hf[f"{p}.mlp.moe_gate.weight"]
            sink[f"{mg}.mlp.router.expert_bias"] = hf[f"{p}.mlp.expert_bias"]
            if config_hf.moe_routed_input_dim:
                sink[f"{mg}.mlp.down_proj.weight"] = hf[f"{p}.mlp.down_proj.weight"]
                sink[f"{mg}.mlp.up_proj.weight"] = hf[f"{p}.mlp.up_proj.weight"]
            # Megatron saves TEGroupedLinear's per-expert weight{i} params fused
            # as one [E, out, in] tensor -- exactly scattermoe's layout, so this
            # is a straight copy (mg_cpu_loader undoes the same fusion).
            sink[f"{mg}.mlp.experts.experts.linear_fc1.weight"] = \
                hf[f"{p}.mlp.experts.experts.weight"]
            sink[f"{mg}.mlp.experts.experts.linear_fc2.weight"] = \
                hf[f"{p}.mlp.experts.output_experts.weight"]
            sink[f"{mg}.mlp.shared_experts.linear_fc1.weight"] = \
                hf[f"{p}.mlp.shared_experts.fc_1.weight"]
            sink[f"{mg}.mlp.shared_experts.linear_fc2.weight"] = \
                hf[f"{p}.mlp.shared_experts.fc_2.weight"]
            if config_mg.moe_shared_expert_gate:
                sink[f"{mg}.mlp.shared_experts.gate_weight"] = hf[f"{p}.mlp.shared_gate.weight"]
        else:
            sink[f"{mg}.mlp.linear_fc1.weight"] = hf[f"{p}.mlp.fc_1.weight"]
            sink[f"{mg}.mlp.linear_fc2.weight"] = hf[f"{p}.mlp.fc_2.weight"]

        # -- pre-MLP norm, or the geodesic params that replace it ------------
        if geodesic:
            for which in ("mixer", "mlp"):
                for what in ("scale", "bias"):
                    sink[f"{mg}.geodesic_{which}.{what}"] = hf[f"{p}.geodesic_{which}.{what}"]
        else:
            sink[f"{mg}.pre_mlp_norm.weight"] = hf[f"{p}.postmixer_norm.norm.weight"]

    # -- model level --------------------------------------------------------
    if "decoder.final_layernorm.weight" in sink:
        sink["decoder.final_layernorm.weight"] = hf["model.final_norm.norm.weight"]

    hf_rows = {}
    for mg_key, hf_key in (("embedding.word_embeddings.weight", "model.embedding.weight"),
                           ("output_layer.weight", "lm_head.weight")):
        w = hf[hf_key]
        rows = w.shape[0]
        full = sink.shape(mg_key)[0]
        assert rows <= full, (
            f"{hf_key} has {rows} rows but Megatron's {mg_key} holds {full}. The HF "
            f"checkpoint has a LARGER vocab than the reference -- tokens were added "
            f"during training. Megatron would need a matching padded_vocab_size, "
            f"which this converter does not compute.")
        # Adding special tokens is routine in an SFT/TRL pipeline, and it would
        # silently shift every row here. Refuse rather than mis-map the vocab.
        if rows != config_mg.vocab_size:
            raise ValueError(
                f"{hf_key} has {rows} rows but the reference was trained with "
                f"vocab_size={config_mg.vocab_size}. The tokenizer changed between "
                f"the reference and this HF checkpoint; the embedding rows would not "
                f"line up. Use a reference from the same tokenizer.")
        sink.set_rows(mg_key, w)
        hf_rows[mg_key] = rows

    print0(f"Mapped {len(config_mg.layers_mixer_config)} layers "
           f"({config_mg.layers_mixer_config})")
    return hf_rows

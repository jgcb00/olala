from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import re
import torch
import torch.nn as nn
from functools import partial
from collections import defaultdict
import tyro

from .configuration_olala import OlalaConfig
from .modeling_olala import OlalaForCausalLM

@dataclass
class NanoArgs:
    resume_from: Optional[str] = None
    run_name : str = ""
    
    # arch - general
    d_model : int = 768
    n_heads : int = 6 # head dim 128 suggested by @Grad62304977
    head_dim: Optional[int] = None
    layers_config : str = 4*"lrdlr"
    expand_factor : int = 2 # expand factor for Mamba/Olala
    rope_type_local: str = "" #p-rope
    rope_type_global: str = "" #p-rope
    rope_theta_local: float = 10000.0
    rope_theta_global: float = 0.0
    eps_rmsnorm: float = 1e-6
    mlp_expand: int = 4 # expand factor for MLP
    fused_loss_computation : bool = True # whether to use fused linear + cross entropy loss
    use_uscaling: bool = False
    uscaling_tau: float = 0.2
    zero_centered_gamma: bool = False
    zero_centered_gate: bool = False
    zero_centered_gate_type: int = 1 # 1, 2, 3, 4
    gate_attn: bool = False
    gate_gdn: bool = True
    gate_type: str = "elementwise" # elementwise (one per dim), headwise (one per head), kimi (lora)
    gate_act: str = "silu" # silu, sigmoid
    scalar_proj_as_hidden_matrix: bool = True
    normalization_type: str = "rmsnorm" # rmsnorm, seednorm
    seednorm_wd: bool = True
    mixer_gn: bool = True
    mlp_linking : bool = False
    final_norm: bool = True

    # attention related
    n_kv_heads : int = 0
    swa_window_size : int = 1024
    slw_warmup_iters: float = 0
    slw_start: int = 8 # window size at the start of training
    slw_increment: int = 64 # window size increment at each step
    softcap_local_attn: float = 0.0 # logit soft-capping for local attn logits, as per Gemma2 (0.0 = no soft-capping)
    softcap_global_attn: float = 0.0
    qk_norm: bool = True
    scalable_softmax: bool = True
    resformer : bool = False # Works only on f layers (DiffAttention)
    token_shift_attn: bool = False
    token_shift_gdn: bool = False
    token_conv1d_attn: bool = False
    token_conv1d_gdn: bool = True
    num_attention_heads_indexer: int = 8
    head_dim_indexer: int = 32
    dsa_q_lora_rank: int = 128
    dsa_topk: int = 512
    cca_seq_kernel_size: int = 4
    nsa_topk: int = 16
    nsa_block_size: int = 64
    nsa_window_size: int = 512
    num_signal_heads_diff: Optional[int] = None
    tpa_rank: int = 2
    shrink_qk_da: int = 2
    mla_kv_rank: int = 128

    # GDN related
    rope_gdn: Optional[str] = None # None, rope, (srope)
    head_dim_gdn: Optional[int] = None
    n_heads_gdn: int = 0
    n_kv_heads_gdn: int = 0
    shrink_qk_gdn: int = 2
    kda_allow_neg_eigval: bool = False
    kda_num_v_heads: Optional[int] = None
    mamba_mimo_dim: Optional[int] = 2
    mamba_ngroups: Optional[int] = 1

    # optim
    optim: str = "adamw" # adamw, spam, stable-spam, muon, muon_moonlight, splus
    second_order_optim : Optional[str] = None # snoo
    batch_size: int = 8*64 # batch size, in sequences, across all devices
    device_batch_size: int = 64 # batch size, in sequences, per device
    total_iterations: int = 1000 # number of iterations to run
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_iters: int = 200
    warmdown_iters: int = 3000
    grad_norm_clip: float = 1.0
    uscaling_mult_embed: float = 0
    uscaling_mult_scalar: float = 0
    uscaling_mult_head: float = 0
    init_std: float = 0.006
    patch_level_training: bool = False
    patch_level_training_size: int = 4
    second_order_lr: float = 0.68
    second_order_momentum: float = 0.37
    second_order_interval: int = 25

    # data
    vocab_size: int = 50304
    sequence_length: int = 1024
    input_bin: Optional[str] = None
    input_val_bin: Optional[str] = None

    # evaluation and logging
    val_loss_every: int = 125
    val_iterations: int = 50 # 1 step = global bs * T tokens
    inspect_every: int = 0
    save_every: int = 1000
    log_dir: str = "logs/"
    wandb_project: str = "dragon_v1.5"
    wandb_name: Optional[str] = None
    log_wandb: bool = False

    load_arg_from_config: bool = True
    load_optim: bool = True
    load_sched: bool = True
    compile: bool = True

    # used during training
    slw_window: int = 0

args = tyro.cli(NanoArgs)

# load model.
config_hf = OlalaConfig(
    final_norm=args.final_norm,
    mla_kv_rank=args.mla_kv_rank,
    rope_gdn=args.rope_gdn,
    shrink_qk_da=args.shrink_qk_da,
    shrink_qk_gdn=args.shrink_qk_gdn,
    mixer_gn=args.mixer_gn,
    kda_allow_neg_eigval=args.kda_allow_neg_eigval,
    kda_num_v_heads=args.kda_num_v_heads,
    seednorm_wd=args.seednorm_wd,
    normalization_type=args.normalization_type,
    tpa_rank=args.tpa_rank,
    num_signal_heads_diff=args.num_signal_heads_diff,
    scalar_proj_as_hidden_matrix=args.scalar_proj_as_hidden_matrix,
    token_shift_attn=args.token_shift_attn,
    token_shift_gdn=args.token_shift_gdn,
    token_conv1d_attn=args.token_conv1d_attn,
    token_conv1d_gdn=args.token_conv1d_gdn,
    patch_level_training=args.patch_level_training,
    patch_level_training_size=args.patch_level_training_size,
    nsa_topk=args.nsa_topk,
    nsa_block_size=args.nsa_block_size,
    nsa_window_size=args.nsa_window_size,
    cca_seq_kernel_size=args.cca_seq_kernel_size,
    head_dim=args.head_dim,
    head_dim_gdn=args.head_dim_gdn,
    num_attention_heads_gdn=args.n_heads_gdn,
    num_key_value_heads_gdn=args.n_kv_heads_gdn,
    zero_centered_gate=args.zero_centered_gate,
    zero_centered_gate_type=args.zero_centered_gate_type,
    scalable_softmax=args.scalable_softmax,
    mamba_mimo_dim=args.mamba_mimo_dim,
    mamba_ngroups=args.mamba_ngroups,
    resformer=args.resformer,
    gate_type=args.gate_type,
    gate_act=args.gate_act,
    gate_attn=args.gate_attn,
    gate_gdn=args.gate_gdn,
    fused_loss_computation=args.fused_loss_computation,
    qk_norm=args.qk_norm,
    num_attention_heads_indexer=args.num_attention_heads_indexer,
    head_dim_indexer=args.head_dim_indexer,
    dsa_q_lora_rank=args.dsa_q_lora_rank,
    dsa_topk=args.dsa_topk,
    zero_centered_gamma=args.zero_centered_gamma,
    vocab_size=args.vocab_size,
    max_position_embeddings=args.sequence_length,
    use_uscaling=args.use_uscaling,
    hidden_size=args.d_model,
    intermediate_size=args.d_model * args.mlp_expand,
    expand_factor=args.expand_factor,
    layers_config=args.layers_config,
    num_attention_heads=args.n_heads,
    num_key_value_heads=args.n_kv_heads if args.n_kv_heads > 0 else args.n_heads,
    initializer_range=args.init_std,
    softcap_local_attn=args.softcap_local_attn,
    softcap_global_attn=args.softcap_global_attn,
    norm_epsilon=args.eps_rmsnorm,
    use_cache=False,
    sliding_window_size=args.swa_window_size,
    rope_type_global=args.rope_type_global,
    rope_type_local=args.rope_type_local,
    rope_theta_global=args.rope_theta_global,
    rope_theta_local=args.rope_theta_local,
    uscaling_tau=args.uscaling_tau,
    mlp_linking=args.mlp_linking
)

model = OlalaForCausalLM(config_hf)
model = model.cuda()

B, L = 2, 2048

# ---------- helpers ---------- #
def l1(x: torch.Tensor) -> float:
    return x.abs().mean().item()

def _capture(name: str, store: Dict[str, torch.Tensor], _m, _inp, out):
    """Save every tensor produced by a module so that we can measure activations."""
    def walk(x, suf=""):
        if torch.is_tensor(x):
            store[f"{name}{suf}"] = x.detach()
        elif isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                walk(xi, suf + f"[{i}]")
    walk(out)

_stat_pat = re.compile(r"(\.grad\.(?:std|mean|l1)|\.act\.(?:std|mean|l1)|\.(?:std|mean|l1))$")

# Support multiple model naming schemes
_LAYER_PATTERNS = [
    re.compile(r"\.h\.(\d+)\."),                 # transformer.h.<i>.
    re.compile(r"\.layers\.(\d+)\."),            # model.layers.<i>.
    re.compile(r"\.decoder\.layers\.(\d+)\."),   # decoder.layers.<i>.
    re.compile(r"\.block\.(\d+)\."),             # ...block.<i>.
]

def _find_layer_span_and_idx(key: str):
    for pat in _LAYer_PATTERNS if False else _LAYER_PATTERNS:  # keep exact name
        m = pat.search(key)
        if m:
            return m.span(0), int(m.group(1))  # span of ".layers.<i>." and the idx
    return None, -1

def _layer_idx(key: str) -> int:
    _, idx = _find_layer_span_and_idx(key)
    return idx

def _base_key(key: str) -> str:
    """Return <parameter-suffix>.<stat> without the layer index, e.g. mixer.linear_qkv.weight.std"""
    span, _ = _find_layer_span_and_idx(key)
    pre_cut = key
    if span:
        s, e = span
        pre_cut = pre_cut[:s] + "." + pre_cut[e:]  # collapse the layer segment to a single dot
    # Drop common top-level prefixes
    for prefix in ("transformer.", "model.", "module."):
        if pre_cut.startswith(prefix):
            pre_cut = pre_cut[len(prefix):]
    stat_match = _stat_pat.search(pre_cut)
    assert stat_match, f"No stat suffix in key {key}"
    stat_suffix = stat_match.group(1)
    base_no_stat = pre_cut[: -len(stat_suffix)]
    return f"{base_no_stat}{stat_suffix}"

# ---------- main routine ---------- #

def show_layer_stats(model: nn.Module) -> str:
    """Run a forward/backward pass and return aggregated stats in JSON.

    The JSON schema is:
    {
        "attn.linear_qkv.weight.std": [layer0, layer1, ..., layerN],
        "attn.linear_qkv.grad.std"  : [...],
        "attn.linear_qkv.act.std"   : [...],
        ...
    }
    Layers that do not have a value for a given statistic are represented with null.
    Non‑layer parameters (e.g., embeddings) are kept flat as a single key‑value pair.
    """

    PAD = len(str(len(config_hf.layers_config) - 1))

    # ----- collect activations ----- #
    acts, hooks = {}, []
    for n, m in model.named_modules():
        if m is model:
            continue  # skip root
        hooks.append(m.register_forward_hook(partial(_capture, n, acts)))

    x = torch.randint(0, config_hf.vocab_size, (B, L), device="cuda")
    y = torch.randint(0, config_hf.vocab_size, (B, L), device="cuda")
    loss = model(input_ids=x, labels=y).loss
    loss.backward()

    # ----- collect stats (weight / grad / act) ----- #
    raw_stats = {}
    for n, p in model.named_parameters():
        raw_stats[f"{n}.std"]      = p.std().item()
        #raw_stats[f"{n}.mean"]     = p.mean().item()
        raw_stats[f"{n}.l1"]       = l1(p)
        if p.grad is not None:
            raw_stats[f"{n}.grad.std"]  = p.grad.std().item()
            #raw_stats[f"{n}.grad.mean"] = p.grad.mean().item()
            raw_stats[f"{n}.grad.l1"]   = l1(p.grad)
    for n, a in acts.items():
        raw_stats[f"{n}.act.std"]  = a.std().item()
        #raw_stats[f"{n}.act.mean"] = a.mean().item()
        raw_stats[f"{n}.act.l1"]   = l1(a)

    # ----- aggregate across layers ----- #
    agg: Dict[str, List] = defaultdict(lambda: [None] * len(config_hf.layers_config))
    flat: Dict[str, float] = {}

    for key, val in raw_stats.items():
        layer = _layer_idx(key)
        if layer == -1:
            # params without layer index stay flat
            flat[key] = val
            continue
        base = _base_key(key)
        if layer < len(config_hf.layers_config):
            agg[base][layer] = val
        else:
            # unexpected layer index; fall back to flat
            flat[key] = val

    # ----- merge flat & aggregated with custom sorting ----- #
    stats = {}

    # First: per-quantity arrays over layers
    for base_key in sorted(agg.keys()):
        stats[f"inspect/{base_key}"] = agg[base_key]  # list of length = #layers (None where absent)

    # Then: non-layer (“flat”) stats
    for k, v in sorted(flat.items()):
        stats[f"inspect/{k}"] = v
    
    return stats

filename = "layer_stats.json"

json_blob = show_layer_stats(model)
with open(args.log_dir + filename, "w") as f:
    if json_blob:
        json.dump(json_blob, f, indent=2)  # Use json.dump() instead of f.write()
print(f"✅ Saved layer stats to {args.log_dir + filename} ✅")
import os
import glob
import json
import pickle
from dataclasses import dataclass
from typing import Optional
from functools import partial
import gc
import math
import numpy as np
import tyro
import time
import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
from transformers import get_wsd_schedule

from .configuration_dragon import DragonConfig
from .modeling_dragon import DragonForCausalLM

# TODO: save code files!!!!

@dataclass
class NanoArgs:
    resume_from: Optional[str] = None
    run_name : str = ""
    
    # arch - general
    d_model : int = 768
    n_heads : int = 6 # head dim 128 suggested by @Grad62304977
    head_dim: Optional[int] = None
    layers_config : str = 4*"lrdlr"
    expand_factor : int = 2 # expand factor for Mamba/Dragon
    rope_type_local: str = "" #p-rope
    rope_type_global: str = "" #p-rope
    rope_theta_local: float = 10000.0
    rope_theta_global: float = 0.0
    eps_rmsnorm: float = 1e-6
    mlp_expand: float = 4. # expand factor for MLP
    intermediate_size: Optional[int] = None
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
    seednorm_type: int = 1
    seednorm_rank: int = 1
    mixer_gn: bool = True
    gate_before_norm: bool = True
    mlp_linking : bool = False
    final_norm: bool = True
    layer_norm_scaling: bool = False # not read when using muP
    mlp_type: str = "simple" # simple, gated
    tie_lm_head: bool = False
    legacy_gate: bool = False
    vwn: bool = False
    vwn_m: int = 2
    vwn_n: int = 3
    vwn_wd_alpha_beta: bool = False
    vwn_dynamic: bool = True
    reduce_lm_head: int = 0

    # MoE
    moe: bool = False
    moe_num_routed_experts: int = 2
    moe_routed_scaling_factor: float = 2.5
    moe_routed_intermediate_size: int = 768
    moe_shared_intermediate_size: int = 768

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
    mamba_d_state: int = 128
    mamba_headdim: int = 64
    mamba3_rope: bool = True
    mamba3_remove_BC_bias: bool = False
    mamba3_is_id_rms: bool = True
    mamba3_remove_conv: bool = True
    mamba3_is_A_dd: bool = True
    mamba3_add_trapezoid: bool = True
    mamba3_postgate_norm: bool = False # only works if legacy_gate is True!!

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
    alpha_normalize: bool = False # whether to normalize update by (1+alpha) in AdEMAMix
    alpha_ademamix: float = 8.0
    warmup_iters: int = 200
    warmdown_iters: int = 3000
    warmdown_type: str = "linear" # linear, cosine
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
    init_gpt2: bool = False
    wnorm: bool = False # as in nemotron-flash (2511.18890)

    # data
    vocab_size: int = 50304
    bos_id: int = 50256
    sequence_length: int = 1024
    intra_doc_masking: bool = False
    input_bin: Optional[str] = None
    input_val_bin: Optional[str] = None
    dataset_type: str = "hf" # hf, mg

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
    compile_dynamic: bool = False

    # used during training
    slw_window: int = 0

def _peek_data_shard(filename, dataset_type='hf'):
    if dataset_type == 'hf':
        return _peek_hf_shard(filename)
    elif dataset_type == 'mg':
        return _peek_mg_shard(filename)
    else:
        raise ValueError(f"unknown dataset type: {dataset_type}")

def _load_data_shard(filename, dataset_type='hf'):
    if dataset_type == 'hf':
        return _load_hf_shard(filename)
    elif dataset_type == 'mg':
        return _load_mg_shard(filename)
    else:
        raise ValueError(f"unknown dataset type: {dataset_type}")

def _load_hf_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
    # memmap the token payload directly (uint16) after the 256*4B header
    tokens = np.memmap(filename, dtype=np.uint16, mode="r", offset=256 * 4, shape=(ntok,))
    assert tokens.size == ntok, "number of tokens read does not match header?"
    return tokens

def _peek_hf_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = int(header[2])
    return ntok

def _peek_mg_shard(filename):
    tokens = np.memmap(filename, dtype=np.uint16, mode="r")
    return int(tokens.size)

def _load_mg_shard(filename):
    return np.memmap(filename, dtype=np.uint16, mode="r")

class DistributedDataLoader:
    def __init__(self, filename_pattern, intra_doc_masking,B, T, process_rank, num_processes, bos_id, dataset_type='hf'):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.intra_doc_masking = intra_doc_masking
        self.bos_id = bos_id
        self.B = B # micro batch size
        self.T = T
        self.dataset_type = dataset_type

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        self.shard_ntoks = []
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname, dataset_type=self.dataset_type)
            #print(f"shard {fname} has {shard_ntok} tokens")
            assert shard_ntok >= num_processes * B * T + 1
            self.shard_ntoks.append(shard_ntok)
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self, shard=0):
        self.current_shard = shard
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard], dataset_type=self.dataset_type)

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard], dataset_type=self.dataset_type)
        
        if self.process_rank == 0:
            shard_tokens = self.shard_ntoks[self.current_shard]
            cum_tokens = sum(self.shard_ntoks[: self.current_shard + 1])

            def _fmt(n):
                return f"{n/1e9:.2f}B" if n >= 1_000_000_000 else (
                    f"{n/1e6:.2f}M" if n >= 1_000_000 else str(n))

            print(
                f"Advancing to shard {self.current_shard}/{len(self.files)-1} "
                f"(this={_fmt(shard_tokens)} tok, cum={_fmt(cum_tokens)}/{_fmt(self.ntok_total)})"
            )

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T]
        buf = np.asarray(buf, dtype=np.int64)
        x = torch.from_numpy(buf.reshape(B, T)) # inputs
        y = torch.from_numpy(buf.reshape(B, T)) # targets

        # compute cumulative document positions for intra-document masking
        cu = None
        maxlen = None
        position_ids = None
        if self.intra_doc_masking:
            assert self.B == 1
            starts = (x == self.bos_id).nonzero(as_tuple=True)[1].to(torch.long)
            if starts.numel() == 0 or starts[0] != 0:
                starts = torch.cat([torch.zeros(1, dtype=torch.long), starts])
            ends = torch.cat([starts[1:], torch.tensor([x.numel()])])
            seqlens = (ends - starts).to(torch.int32)
            # cu_seqlens, max_seqlen.
            cu = torch.cat([torch.zeros(1, dtype=torch.int32), seqlens.cumsum(0)]).cuda().to(torch.int32)
            maxlen = int(seqlens.max())
            # position_ids.
            lengths = seqlens.to(torch.long)
            starts_per_token = torch.repeat_interleave(starts.to(torch.long), lengths)
            idx = torch.arange(T, device=x.device, dtype=torch.long)
            position_ids = (idx - starts_per_token).unsqueeze(0)

        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x.cuda(), y.cuda(), cu, maxlen, position_ids

def param_groups_mup(model, base_lr_hidden, base_lr_scalar, base_lr_embed, base_lr_head, wd):
    groups, seen = [], set()
    id2name = {id(p): n for n, p in model.named_parameters()}

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            pname = id2name.get(id(mod.weight), "")
            is_scalar = getattr(mod, "is_scalar_weight", False)
            fan_in = mod.weight.shape[1]
            if "lm_head" in pname:
                scale = 1
                lr_scaled = base_lr_head
                wd_scaled = 0.0
                wd_mult = 0.0
            elif is_scalar:
                scale = 1
                lr_scaled = base_lr_scalar
                wd_scaled = 0.0
                wd_mult = 0.0
            else:
                scale = 1 / math.sqrt(fan_in)
                lr_scaled = base_lr_hidden * scale
                wd_scaled = wd / lr_scaled
                wd_mult = 1/lr_scaled

            groups.append({"params": [mod.weight], "lr": lr_scaled, "weight_decay": wd_scaled})
            seen.add(mod.weight)

            print(f"param {name}.weight | shape {mod.weight.shape} | scale {scale} | wd_mult={wd_mult:.3e}")

            if mod.bias is not None:
                assert False
                groups.append({"params": [mod.bias], "lr": base_lr_scalar, "weight_decay": 0.0})
                seen.add(mod.bias)

    for name, p in model.named_parameters():
        if p in seen:
            continue
        pname = id2name.get(id(p), "<unnamed>")

        if "embedding" in pname:
            #fan_out = p.shape[1] # nn.Embedding is transposed
            #lr_scaled = base_lr / math.sqrt(fan_out) # u-muP
            lr_scaled = base_lr_embed
        else:
            lr_scaled = base_lr_scalar

        wd_scaled = 0.
        wd_mult = 0.
        if getattr(p, "requires_weight_decay", False):
            wd_scaled = wd / lr_scaled
            wd_mult = 1/lr_scaled

        groups.append({"params": [p], "lr": lr_scaled, "weight_decay": wd_scaled})

        print(f"param {name} | shape {p.shape} | scale {1.} | wd_mult={wd_mult:.3e}")

    return groups

args = tyro.cli(NanoArgs)

if args.intra_doc_masking:
    if args.device_batch_size != 1:
        args.device_batch_size = 1
        print("!!! Forcing device_batch_size to 1 for intra-document masking !!!")

if args.mlp_type == "gated":
    if args.use_uscaling:
        print("problem: gated MLP with muP is not supported, because we use FA backend")
        exit(0)

    if args.moe:
        print("problem: gated MLP with MoE is not supported, because we use FA backend")
        exit(0)

if args.legacy_gate:
    assert not args.gate_gdn, "legacy_gate is not compatible with gate_gdn."

# set up DDP (distributed data parallel).
assert torch.cuda.is_available()
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=int(os.environ['WORLD_SIZE']),
    rank=int(os.environ['RANK']),
)
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
torch._dynamo.config.optimize_ddp=False
if args.compile_dynamic:
    torch._dynamo.config.allow_unspec_int_on_nn_module=True

# setup logging.
resume_dir = None
if args.resume_from:
    cand = args.resume_from # either a step dir or the run dir
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "train_state.pt")):
        resume_dir = cand
    elif os.path.isdir(cand):
        # pick latest step*/train_state.pt inside run dir
        step_dirs = sorted(
            [d for d in glob.glob(os.path.join(cand, "step*")) if os.path.isdir(d)],
            key=lambda p: int(os.path.basename(p).replace("step","")),
        )
        if not step_dirs:
            raise ValueError(f"No step*/train_state.pt under {cand}")
        resume_dir = step_dirs[-1]
        if master_process:
            print(f"Auto-selected latest checkpoint dir: {resume_dir}")
    else:
        raise ValueError(f"resume_from must be a directory (got {cand})")

if master_process:
    if resume_dir is not None:
        train_state = torch.load(os.path.join(resume_dir, "train_state.pt"), map_location="cpu")
        run_name = train_state.get("run_name", args.run_name)
        logdir = os.path.dirname(resume_dir)
    else:
        run_name = args.run_name
        logdir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"{run_name}.txt")
    print(f"Logging to {logfile}")
    if resume_dir is None:
        with open(f'{logdir}/args.json', 'w') as f: json.dump(vars(args), f)
        with open(f'{logdir}/args.pkl', 'wb') as f: pickle.dump(args, f)
def print0(s, console=True):
    if not master_process: return
    if console:
        print(s)
    try:
        d=os.path.dirname(logfile); d and os.makedirs(d, exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as f: print(s, file=f)
    except: pass
if resume_dir is not None and args.load_arg_from_config:
    saved_args_path = os.path.join(os.path.dirname(resume_dir), "args.pkl")
    print0(f"Loading args from {saved_args_path}")
    if os.path.exists(saved_args_path):
        with open(saved_args_path, "rb") as f:
            saved_args = pickle.load(f)
        cli_resume = args.resume_from
        args = saved_args
        args.resume_from = cli_resume or resume_dir
print0(f"running with args:\n{args}")
if master_process:
    wandb.init(project=args.wandb_project, dir=logdir, name=args.wandb_name if args.wandb_name else args.run_name, config={**vars(args)}, mode=None if args.log_wandb else 'disabled')
    print0(f"wandb run id: {wandb.run.id}")

# set seeds.
seed = 123456789
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# define convenience variables.
B, T = args.device_batch_size, args.sequence_length
if args.patch_level_training:
    T = args.patch_level_training_size * T
assert args.batch_size % (B * ddp_world_size) == 0
accumulation_steps = args.batch_size // (B * ddp_world_size)

tokenizer = transformers.AutoTokenizer.from_pretrained("/leonardo_work/BOOST_LCustodi/script/training/temp/hf_models/gpt2", use_fast=True)

# load dataloaders.
#if args.patch_level_training:
#    assert T % args.patch_level_training_size == 0, "sequence length must be divisible by patch level training size in reduced mode"
train_loader = DistributedDataLoader(args.input_bin, args.intra_doc_masking, B, T, ddp_rank, ddp_world_size, args.bos_id, args.dataset_type)
val_loader = DistributedDataLoader(args.input_val_bin, args.intra_doc_masking, B, T, ddp_rank, ddp_world_size, args.bos_id, args.dataset_type)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

# load model.
config_hf = DragonConfig(
    reduce_lm_head=args.reduce_lm_head,
    dataset_type=args.dataset_type,
    vwn=args.vwn,
    vwn_m=args.vwn_m,
    vwn_n=args.vwn_n,
    vwn_wd_alpha_beta=args.vwn_wd_alpha_beta,
    vwn_dynamic=args.vwn_dynamic,
    legacy_gate=args.legacy_gate,
    init_gpt2=args.init_gpt2,
    tie_lm_head=args.tie_lm_head,
    mlp_type=args.mlp_type,
    layer_norm_scaling=args.layer_norm_scaling,
    mamba_d_state=args.mamba_d_state,
    mamba_headdim=args.mamba_headdim,
    mamba3_rope=args.mamba3_rope,
    mamba3_remove_BC_bias=args.mamba3_remove_BC_bias,
    mamba3_is_id_rms=args.mamba3_is_id_rms,
    mamba3_remove_conv=args.mamba3_remove_conv,
    mamba3_is_A_dd=args.mamba3_is_A_dd,
    mamba3_add_trapezoid=args.mamba3_add_trapezoid,
    mamba3_postgate_norm=args.mamba3_postgate_norm,
    moe=args.moe,
    moe_num_routed_experts=args.moe_num_routed_experts,
    moe_routed_scaling_factor=args.moe_routed_scaling_factor,
    moe_routed_intermediate_size=args.moe_routed_intermediate_size,
    moe_shared_intermediate_size=args.moe_shared_intermediate_size,
    intra_doc_masking=args.intra_doc_masking,
    seednorm_rank=args.seednorm_rank,
    seednorm_type=args.seednorm_type,
    final_norm=args.final_norm,
    mla_kv_rank=args.mla_kv_rank,
    rope_gdn=args.rope_gdn,
    shrink_qk_da=args.shrink_qk_da,
    shrink_qk_gdn=args.shrink_qk_gdn,
    mixer_gn=args.mixer_gn,
    gate_before_norm=args.gate_before_norm,
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
    intermediate_size=int(args.d_model * args.mlp_expand) if args.intermediate_size is None else args.intermediate_size,
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

if resume_dir is None:
    model = DragonForCausalLM(config_hf)
    model = model.cuda()
else:
    model = DragonForCausalLM.from_pretrained(resume_dir, config=config_hf, torch_dtype=torch.bfloat16)
    model = model.cuda()
print0(model)

with torch.no_grad():
    for name, p in model.named_parameters():
        if p is None or p.numel() == 0:
            continue
        t = p.detach().float()
        mean = t.mean().item()
        std  = t.std(unbiased=False).item()
        print0(f"{name:60s} shape={tuple(p.shape)} mean={mean:+.4e} std={std:.4e}")

# count params. (total & active)
num_params = sum(p.numel() for p in model.parameters())
"""model.eval()
x, y, _, _, _ = train_loader.next_batch()
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    model(input_ids=x[[0], [0]].unsqueeze(0)).logits.sum().backward()
num_active = sum(p.grad.count_nonzero() for p in model.parameters() if p.grad is not None)
model.zero_grad(set_to_none=True)
model.train()"""
print0(f"number of total parameters:  {num_params}")
#print0(f"number of active parameters: {num_active} ({num_active/num_params*100:.2f}%)")

# DDP & compile.
uncompiled_model = model
model = torch.compile(model, dynamic=args.compile_dynamic) if args.compile else model
model.train()
model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=args.resformer)
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

if args.intra_doc_masking:
    print0("!!! Using intra-document masking !!!")
    print0("It is only compatible with GDN (conv+chunk), KDA (conv+chunk), DA and GDTPA layers. For DA/GDTPA, kv shift is also compatible. All other config will not have intra-doc masking support!!")

# load optimizers & schedulers.
if args.use_uscaling:
    #assert args.optim == "adamw", "uscaling is only supported with AdamW optimizer currently"
    param_list = param_groups_mup(
        raw_model,
        base_lr_hidden=args.learning_rate,
        base_lr_scalar=args.uscaling_mult_scalar*args.learning_rate if args.uscaling_mult_scalar > 0 else args.learning_rate,
        base_lr_embed=args.uscaling_mult_embed*args.learning_rate if args.uscaling_mult_embed > 0 else args.learning_rate,
        base_lr_head=args.uscaling_mult_head*args.learning_rate if args.uscaling_mult_head > 0 else args.learning_rate,
        wd=args.weight_decay,
    )
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(param_list, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    elif args.optim == "ademamix":
        from .optimizers.Ademamix import AdEMAMix
        beta3_warmup = args.total_iterations
        alpha_warmup = args.total_iterations
        optimizer = AdEMAMix(param_list, beta3_warmup=beta3_warmup, alpha_warmup=alpha_warmup, normalize_alpha=args.alpha_normalize, alpha=args.alpha_ademamix, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer for unit scaling: {args.optim}")
else:
    if args.optim == "adamw":
        #optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
        decay_params = []
        no_decay_params = []
        for name, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if getattr(p, "_no_weight_decay", False):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optim == "ademamix":
        from .optimizers.Ademamix import AdEMAMix

        beta3_warmup = args.total_iterations
        alpha_warmup = args.total_iterations
        optimizer = AdEMAMix(raw_model.parameters(), lr=args.learning_rate, beta3_warmup=beta3_warmup, alpha_warmup=alpha_warmup, normalize_alpha=args.alpha_normalize, alpha=args.alpha_ademamix, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown Optimizer: {args.optim}")
if args.second_order_optim == "snoo":
    from .optimizers.Snoo import Snoo
    second_order_optim = Snoo(raw_model, lr=args.second_order_lr, momentum=args.second_order_momentum, k=args.second_order_interval)
else:
    second_order_optim = None

optimizers = [optimizer]

def get_lr_wsd(num_iterations, warmup_iters, warmdown_iters, it):
    assert it <= num_iterations, f"it : {it}, num_iterations : {num_iterations}"
    # 1) linear warmup for warmup_iters steps
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    # 2) constant lr for a while
    elif it < num_iterations - warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - it) / warmdown_iters
        return decay_ratio
if args.warmdown_type == "linear":
    sched_func = partial(get_lr_wsd, args.total_iterations, args.warmup_iters, args.warmdown_iters)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, sched_func) for opt in optimizers]
elif args.warmdown_type == "cosine" or args.warmdown_type == "1-sqrt":
    sched = get_wsd_schedule(
        optimizers[0],
        num_warmup_steps=args.warmup_iters,
        num_decay_steps=args.warmdown_iters,
        num_training_steps=args.total_iterations,
        min_lr_ratio=0.,
        warmup_type='linear',
        decay_type=args.warmdown_type,
    )
    schedulers = [sched]
else:
    raise ValueError(f"Unknown warmdown type: {args.warmdown_type}")

# resume if necessary.
start_iter = 0
training_time_ms = 0
if resume_dir is not None:  
    train_state = torch.load(os.path.join(resume_dir, "train_state.pt"), map_location="cpu")
    if args.load_optim:
        for opt, s in zip(optimizers, train_state.get("optimizers", [])):
            opt.load_state_dict(s)
    if args.load_sched:
        for sch, s in zip(schedulers, train_state.get("schedulers", [])):
            sch.load_state_dict(s)
    torch.set_rng_state(train_state["rng_cpu"])
    torch.cuda.set_rng_state_all(train_state["rng_cuda"])
    training_time_ms = train_state.get("training_time_ms", 0)
    start_iter = train_state.get("iteration", 0) + 1

# start the clock.
torch.cuda.synchronize()
t0 = time.perf_counter()
WARMUP_SKIP = 10

# begin training.
train_loader.reset()
x, y, cu, maxlen, position_ids = train_loader.next_batch()

for iter_ in range(start_iter, start_iter+args.total_iterations+1):
    last_iter = (iter_ == start_iter+args.total_iterations)
    if iter_ == start_iter+WARMUP_SKIP:
        training_time_ms = 0
        t0 = time.perf_counter()
    to_log = {}

    # SLW WINDOW UPDATE
    if args.slw_warmup_iters > 0:
        slw_warmup_iters = int(args.slw_warmup_iters * args.total_iterations)

        progress_ratio = iter_ / slw_warmup_iters
        window = args.slw_start + progress_ratio * (args.sequence_length - args.slw_start)
        window = args.slw_increment * math.ceil(window / args.slw_increment) # quantize
        window = int(min(window, args.sequence_length)) # cap
        raw_model.config.slw_wsize = window

        to_log['slw_window'] = window

    # ----------- VALIDATION SECTION -----------
    if (last_iter or (args.val_loss_every > 0 and iter_ % args.val_loss_every == 0)):
        # stop the clock.
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        # run validation batches.
        model.eval()
        val_loader.reset()
        val_loss = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(args.val_iterations):
            for _ in range(accumulation_steps):
                inputs, targets, cu, maxlen, position_ids = val_loader.next_batch()
                with ctx:
                    val_loss += model(input_ids=inputs, labels=targets, just_loss=True, cu_seqlens=cu, max_seqlen=maxlen, position_ids=position_ids).loss.detach()
        val_loss /= args.val_iterations * accumulation_steps
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        model.train()

        # log.
        print0(f'iteration:{iter_:0{len(str(start_iter+args.total_iterations))}d}/{args.total_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms')
        if master_process:
            wandb.log({"val_loss": val_loss}, step=iter_)

        # start the clock again.
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # ----------- SAVING SECTION -----------
    if master_process and (last_iter or (args.save_every > 0 and iter_ % args.save_every == 0)):
        # stop the clock.
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        save_dir = os.path.join(logdir, f"step{iter_:06d}")
        os.makedirs(save_dir, exist_ok=True)
        # save model & tokenizer to make evaluation easier.
        tokenizer.save_pretrained(save_dir)
        state_dict_bf16 = {k: v.detach().to(torch.bfloat16).cpu() for k, v in uncompiled_model.state_dict().items()}
        idm_og = uncompiled_model.config.intra_doc_masking
        uncompiled_model.config.intra_doc_masking = False
        uncompiled_model.config.torch_dtype = torch.bfloat16
        uncompiled_model.save_pretrained(save_dir, safe_serialization=True, state_dict=state_dict_bf16)
        uncompiled_model.config.intra_doc_masking = idm_og
        # save training state.
        train_state = dict(
            iteration=iter_,
            run_name=run_name,
            optimizers=[opt.state_dict() for opt in optimizers],
            schedulers=[sched.state_dict() for sched in schedulers],
            training_time_ms=training_time_ms,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all(),
        )
        torch.save(train_state, os.path.join(save_dir, "train_state.pt"))
        del state_dict_bf16
        gc.collect()
        # start the clock again.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if last_iter:
        dist.barrier()
        break

    # ----------- TRAINING SECTION -----------
    for i in range(1, accumulation_steps+1):
        # forward pass.
        with ctx:
            loss = model(input_ids=x, labels=y, just_loss=True, cu_seqlens=cu, max_seqlen=maxlen, position_ids=position_ids).loss
            train_loss = loss.detach()
        # prepare next batch.
        x, y, cu, maxlen, position_ids = train_loader.next_batch()
        # backward pass.
        if i < accumulation_steps:
            with model.no_sync():
                (loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward() # just sync on the last step
    individual_grad_norms = {}
    """# Calculate individual param norms
    # We use 'raw_model' to avoid 'module.' or '_orig_mod.' prefixes in wandb
    individual_grad_norms = {}
    # Only calculate on master process to save time, and maybe throttle frequency (e.g., every 10 steps)
    # If you want it every step, remove the (iter_ % 10 == 0) check.
    if master_process and (iter_ % 50 == 0): 
        for name, p in raw_model.named_parameters():
            if p.grad is not None:
                # Calculate L2 norm of the gradient
                param_norm = p.grad.detach().data.norm(2).item()
                individual_grad_norms[f"grad_norm/{name}"] = param_norm"""
    # clip those gradients.
    if args.grad_norm_clip is not None:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_clip, foreach=True)
    else:
        grad_norm = torch.tensor(0.)
    # step the optimizers & schedulers.
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    if second_order_optim:
        second_order_optim.step()
    # null those gradients.
    model.zero_grad(set_to_none=True)

    # Wnorm
    if args.wnorm:
        with torch.no_grad():
            for m in model.modules():
                if getattr(m, "norm_case_1", False):
                    W = getattr(m, "weight", None)
                    denom = W.float().norm(p=2, dim=1, keepdim=True).clamp_min(1e-8).to(W.dtype)
                    W.div_(denom)
                elif getattr(m, "norm_case_2", False):
                    W = getattr(m, "weight", None)
                    denom = W.float().norm(p=2, dim=0, keepdim=True).clamp_min(1e-8).to(W.dtype)
                    W.div_(denom)

    # ----------- LOGGING SECTION -----------
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    avg_step_time = approx_training_time_ms / (iter_ + 1 - WARMUP_SKIP) if iter_ >= start_iter+WARMUP_SKIP else 0
    extra = " ".join(f"{k}:{v}" for k, v in (to_log or {}).items())
    print0(f"iteration:{iter_+1:0{len(str(start_iter+args.total_iterations))}d}/{args.total_iterations} train_loss:{train_loss.item():.4f} lr: {schedulers[0].get_last_lr()[0]:.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{avg_step_time:.2f}ms {extra}")
    if master_process:
        wandb.log({'train_loss': train_loss.item(), 'step_avg_time': avg_step_time, **{f'lr_{i}': sched.get_last_lr()[0] for i, sched in enumerate(schedulers)}, 'grad_norm': grad_norm.item(), **to_log, **individual_grad_norms}, step=iter_)

print0(f"peak memory consumption during training: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0("Training complete.")
dist.destroy_process_group()

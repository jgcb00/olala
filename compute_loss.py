import os
import glob
import pickle
from dataclasses import dataclass
from typing import Optional
import tyro
from tqdm.auto import tqdm
import numpy as np

import torch

from .configuration_dragon import DragonConfig
from .modeling_dragon import DragonForCausalLM

@dataclass
class Args:
    load_dir: str
    val_bin: str

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
    seednorm_type: int = 1
    seednorm_rank: int = 1
    mixer_gn: bool = True
    mlp_linking : bool = False
    final_norm: bool = True
    layer_norm_scaling: bool = False # not read when using muP
    mlp_type: str = "simple" # simple, gated
    tie_lm_head: bool = False

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

    # data
    vocab_size: int = 50304
    bos_id: int = 50256
    sequence_length: int = 1024
    intra_doc_masking: bool = False
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
    compile_dynamic: bool = False

    # used during training
    slw_window: int = 0

def _peek_data_shard(filename):
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

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
    # memmap the token payload directly (uint16) after the 256*4B header
    tokens = np.memmap(filename, dtype=np.uint16, mode="r", offset=256 * 4, shape=(ntok,))
    assert tokens.size == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, intra_doc_masking,B, T, process_rank, num_processes, bos_id, stop_on_end=False):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.intra_doc_masking = intra_doc_masking
        self.bos_id = bos_id
        self.B = B # micro batch size
        self.T = T
        self.stop_on_end = stop_on_end

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        if self.stop_on_end:
            assert len(self.files) == 1, "Pass a single .bin path (not a pattern) when stop_on_end=True."

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        self.shard_ntoks = []
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
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
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])
        
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
            if self.stop_on_end:
                raise StopIteration
            else:
                self.advance()

        return x.cuda(), y.cuda(), cu, maxlen, position_ids

run_args = tyro.cli(Args)

saved_args_path = os.path.join(os.path.dirname(run_args.load_dir), "args.pkl")
print(f"Loading args from {saved_args_path}")
if os.path.exists(saved_args_path):
    with open(saved_args_path, "rb") as f:
        saved_args = pickle.load(f)
    args: NanoArgs = saved_args

print(args)

B, T = args.device_batch_size, args.sequence_length
accumulation_steps = args.batch_size // (B * 1)

val_loader = DistributedDataLoader(run_args.val_bin, False, B, T, 0, 1, args.bos_id, stop_on_end=True)
print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

# load model.
config_hf = DragonConfig(
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

model = DragonForCausalLM.from_pretrained(run_args.load_dir, config=config_hf, torch_dtype=torch.bfloat16)
model = model.cuda()

model = torch.compile(model, dynamic=args.compile_dynamic) if args.compile else model
model.eval()
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

val_loader.reset()
total_steps = (val_loader.shard_ntoks[val_loader.current_shard] - 1) // (B * T * val_loader.num_processes)
pbar = tqdm(total=total_steps, desc="Validating", unit="step")
val_loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
n_steps = 0
tok_per_step = B * T

with torch.no_grad():
    while True:
        try:
            inputs, targets, cu, maxlen, position_ids = val_loader.next_batch()
        except StopIteration:
            break
        with ctx:
            step_loss = model(
                input_ids=inputs,
                labels=targets,
                just_loss=True,
                cu_seqlens=cu,
                max_seqlen=maxlen,
                position_ids=position_ids,
            ).loss.detach()
        val_loss_sum += step_loss
        n_steps += 1
        avg = (val_loss_sum / n_steps).item()
        pbar.update(1)
        pbar.set_postfix(avg_loss=f"{avg:.4f}", ppl=f"{np.exp(avg):.2f}")
pbar.close()

assert n_steps > 0, "No batches read from the file; check B/T vs file size."
val_loss = (val_loss_sum / n_steps).item()
print(f"Validation Loss: {val_loss:.6f}. Perplexity: {np.exp(val_loss):.6f} (steps={n_steps}, tokens={n_steps*tok_per_step})")
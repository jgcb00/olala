import os
import uuid
import glob
import json
import pickle
from dataclasses import dataclass
from typing import List, Union, Optional
from contextlib import nullcontext
from functools import partial
import math
import numpy as np
import tyro
import time
import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .configuration_dragon import DragonConfig
from .modeling_dragon import DragonForCausalLM

# TODO: save code files!!!!

@dataclass
class NanoConfig:
    resume_from: Optional[str] = None
    run_name : str = ""
    
    # arch - general
    d_model : int = 768
    n_heads : int = 6 # head dim 128 suggested by @Grad62304977
    layers_config : str = 4*"lrdlr"
    expand_factor : int = 1 # expand factor for Mamba/Dragon
    rope_theta_local: float = 10000.0
    eps_rmsnorm: float = 1e-6
    mlp_expand: int = 4 # expand factor for MLP
    fused_loss_computation : bool = True # whether to use fused linear + cross entropy loss
    use_uscaling: bool = False
    uscaling_tau: float = 0.2
    zero_centered_gamma: bool = False

    # attention related
    n_kv_heads : int = 0
    swa_window_size : int = 1024
    slw_warmup_iters: float = 0
    slw_start: int = 8 # window size at the start of training
    slw_increment: int = 64 # window size increment at each step
    softcap_local_attn: float = 0.0 # logit soft-capping for local attn logits, as per Gemma2 (0.0 = no soft-capping)
    softcap_global_attn: float = 0.0
    qk_norm: bool = True
    num_attention_heads_indexer: int = 8
    head_dim_indexer: int = 32
    dsa_q_lora_rank: int = 128
    dsa_topk: int = 512

    # GatedDeltaNet related
    p_state_passing: float = 0.0 # probability of state passing (0.0 = no state passing)
    step_state_passing: int = 0 # step at which to start using the given p_state_passing (0 = start at the beginning of training)

    # optim
    optim: str = "adamw" # adamw, spam, stable-spam, muon, muon_moonlight, splus
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

    # data
    vocab_size: int = 50304
    sequence_length: int = 1024
    use_patch_level_training: bool = False
    patch_size: int = 4
    patch_training_fraction: float = 0.67
    input_bin: Optional[str] = None
    input_val_bin: Optional[str] = None

    # evaluation and logging
    val_loss_every: int = 125
    val_iterations: int = 50 # 1 step = global bs * T tokens
    inspect_every: int = 0
    save_every: int = 1000
    log_dir: str = "logs/"
    wandb_project: str = "dragon_v1.5"
    log_wandb: bool = False

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
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B # micro batch size
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

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

        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x.cuda(), y.cuda()

def param_groups_mup(model, base_lr_hidden, base_lr_scalar, base_lr_embed, base_lr_head, wd):
    groups, seen = [], set()
    id2name = {id(p): n for n, p in model.named_parameters()}

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            pname = id2name.get(id(mod.weight), "")
            fan_in = mod.weight.shape[1]
            scale = 1 / math.sqrt(fan_in)
            if "lm_head" in pname:
                lr_scaled = base_lr_head
                wd_scaled = 0.0
            else:
                lr_scaled = base_lr_hidden * scale
                wd_scaled = wd / lr_scaled

            groups.append({"params": [mod.weight], "lr": lr_scaled, "weight_decay": wd_scaled})
            seen.add(mod.weight)

            if mod.bias is not None:
                groups.append({"params": [mod.bias], "lr": lr_scaled, "weight_decay": 0.0})
                seen.add(mod.bias)

    for p in model.parameters():
        if p in seen:
            continue
        pname = id2name.get(id(p), "<unnamed>")

        if "embedding" in pname:
            fan_out = p.shape[1] # nn.Embedding is transposed
            #lr_scaled = base_lr / math.sqrt(fan_out) # u-muP
            lr_scaled = base_lr_embed
        else:
            lr_scaled = base_lr_scalar

        groups.append({"params": [p], "lr": lr_scaled, "weight_decay": 0.})

    return groups

config = tyro.cli(NanoConfig)

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

# setup logging.
ckpt = None
resume_from = config.resume_from
if resume_from is not None:
    if os.path.isdir(resume_from):
        checkpoint_files = glob.glob(os.path.join(resume_from, "state_step*.pt"))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in directory: {resume_from}")
        checkpoint_files.sort(key=lambda x: int(x.split("state_step")[1].split(".pt")[0]))
        resume_from = checkpoint_files[-1]
        if master_process:
            print(f"Auto-selected latest checkpoint: {resume_from}")
    ckpt = torch.load(resume_from, map_location="cpu")
    checkpoint_dir = os.path.dirname(resume_from)
    saved_config_path = os.path.join(checkpoint_dir, 'config.pkl')
    if os.path.exists(saved_config_path):
        with open(saved_config_path, 'rb') as f:
            config = pickle.load(f)
        config.resume_from = resume_from
        if master_process:
            print(f"Loaded saved config from {saved_config_path}")
    else:
        if master_process:
            print(f"WARNING: Could not find saved config at {saved_config_path}, using CLI config")
if master_process:
    print(f"running with config:\n{config}")
if master_process:
    run_id = config.run_name + '_' + str(uuid.uuid4().hex[:8]) if ckpt is None else ckpt['run_id']
    logdir = os.path.join(config.log_dir, run_id)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(config.log_dir, f"{run_id}.txt")
    print(f"Logging to {logfile}")
def print0(s, console=True):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)
if master_process:
    #if ckpt is not None:
    #    print0(f"Resuming wandb run {config.wandb_id}")
    wandb.init(project=config.wandb_project, dir=logdir, name=config.run_name, config={**vars(config)}, mode=None if config.log_wandb else 'disabled')#, fork_from=None if ckpt is None else f"{config.wandb_id}?_step={ckpt['iteration']}")
    config.wandb_id = wandb.run.id
    print0(f"WandB run id: {wandb.run.id}")
# so that we save the wandb id in the config.
if master_process and not resume_from:
    with open(f'{logdir}/config.json', 'w') as f:
        json.dump(vars(config), f)
    with open(f'{logdir}/config.pkl', 'wb') as f:
        pickle.dump(config, f)

# define convenience variables.
B, T = config.device_batch_size, config.sequence_length
assert config.batch_size % (B * ddp_world_size) == 0
accumulation_steps = config.batch_size // (B * ddp_world_size)

# load dataloaders.
train_loader = DistributedDataLoader(config.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(config.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

# load model.
config_hf = DragonConfig(
    qk_norm=config.qk_norm,
    num_attention_heads_indexer=config.num_attention_heads_indexer,
    head_dim_indexer=config.head_dim_indexer,
    dsa_q_lora_rank=config.dsa_q_lora_rank,
    dsa_topk=config.dsa_topk,
    zero_centered_gamma=config.zero_centered_gamma,
    vocab_size=config.vocab_size,
    max_position_embeddings=config.sequence_length,
    use_uscaling=config.use_uscaling,
    hidden_size=config.d_model,
    intermediate_size=config.d_model * config.mlp_expand,
    expand_factor=config.expand_factor,
    layers_config=config.layers_config,
    num_attention_heads=config.n_heads,
    num_key_value_heads=config.n_kv_heads if config.n_kv_heads > 0 else config.n_heads,
    initializer_range=config.init_std,
    softcap_local_attn=config.softcap_local_attn,
    softcap_global_attn=config.softcap_global_attn,
    norm_epsilon=config.eps_rmsnorm,
    use_cache=False,
    sliding_window_size=config.swa_window_size,
    rope_theta_local=config.rope_theta_local,
    uscaling_tau=config.uscaling_tau,
)
model = DragonForCausalLM(config_hf)
model = model.cuda()

"""# check here that the init std is as expected: # TODO TEMPORARY
with torch.no_grad():
    wstd = model.model.embedding.weight.std().item()
    print0(f"Model weight init std: {wstd:.6f} (expected {config.init_std})")
    assert abs(wstd - config.init_std) / config.init_std < 0.1, f"weight init std {wstd} deviates from expected {config.init_std} by more than 10%"

    # check on another we
    lstd = model.model.layers[0].attn.linear_qkv.weight.std().item()
    print0(f"Model first layer attention QKV weight init std: {lstd:.6f} (expected {config.init_std})")

    lstd = model.model.layers[0].lin_attn.qkv_conv1d.weight.std().item()
    print0(f"Model first layer conv QKV weight init std: {lstd:.6f} (expected {config.init_std})")"""

# count params. (total & active)
num_params = sum(p.numel() for p in model.parameters())
"""model.eval()
x, y = train_loader.next_batch()
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    model(input_ids=x[[0], [0]].unsqueeze(0)).logits.sum().backward()
num_active = sum(p.grad.count_nonzero() for p in model.parameters() if p.grad is not None)
model.zero_grad(set_to_none=True)
model.train()"""
print0(f"number of total parameters:  {num_params}")
#print0(f"number of active parameters: {num_active} ({num_active/num_params*100:.2f}%)")

# DDP & compile.
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.weight.data = module.weight.data.to(torch.bfloat16)
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.to(torch.bfloat16)
model = torch.compile(model, dynamic=True)
model.train()
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# load optimizers & schedulers.
if config.use_uscaling:
    param_list = param_groups_mup(
        raw_model,
        base_lr_hidden=config.learning_rate,
        base_lr_scalar=config.uscaling_mult_scalar*config.learning_rate if config.uscaling_mult_scalar > 0 else config.learning_rate,
        base_lr_embed=config.uscaling_mult_embed*config.learning_rate if config.uscaling_mult_embed > 0 else config.learning_rate,
        base_lr_head=config.uscaling_mult_head*config.learning_rate if config.uscaling_mult_head > 0 else config.learning_rate,
        wd=config.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_list, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps)
else:
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps)
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
sched_func = partial(get_lr_wsd, config.total_iterations, config.warmup_iters, config.warmdown_iters)
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, sched_func) for opt in optimizers]

# resume if necessary.
start_iter = 0
training_time_ms = 0
if ckpt is not None:
    print0(f"Resuming from {resume_from}")
    raw_model.load_state_dict(ckpt['model'])
    for opt, s in zip(optimizers, ckpt.get("optimizers", [])):
        opt.load_state_dict(s)
    for sch, s in zip(schedulers, ckpt.get("schedulers", [])):
        sch.load_state_dict(s)
    torch.set_rng_state(ckpt["rng_cpu"])
    torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    training_time_ms = ckpt.get("training_time_ms", 0)
    start_iter = ckpt.get("iteration", 0) + 1

# start the clock.
torch.cuda.synchronize()
t0 = time.perf_counter()
WARMUP_SKIP = 10

# begin training.
train_loader.reset()
x, y = train_loader.next_batch()

for iter_ in range(start_iter, config.total_iterations+1):
    last_iter = (iter_ == config.total_iterations)
    if iter_ == WARMUP_SKIP:
        training_time_ms = 0
        t0 = time.perf_counter()
    to_log = {}

    # SLW WINDOW UPDATE
    if config.slw_warmup_iters > 0:
        slw_warmup_iters = int(config.slw_warmup_iters * config.total_iterations)

        progress_ratio = iter_ / slw_warmup_iters
        window = config.slw_start + progress_ratio * (config.sequence_length - config.slw_start)
        window = config.slw_increment * math.ceil(window / config.slw_increment) # quantize
        window = int(min(window, config.sequence_length)) # cap
        raw_model.config.slw_wsize = window

        to_log['slw_window'] = window

    # ----------- VALIDATION SECTION -----------
    if (last_iter or (config.val_loss_every > 0 and iter_ % config.val_loss_every == 0)):
        # stop the clock.
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        # run validation batches.
        model.eval()
        val_loader.reset()
        val_loss = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(config.val_iterations):
            for _ in range(accumulation_steps):
                inputs, targets = val_loader.next_batch()
                with torch.no_grad():
                    with ctx:
                        val_loss += model(input_ids=inputs, labels=targets).loss
        val_loss /= config.val_iterations * accumulation_steps
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        model.train()

        # log.
        print0(f'iteration:{iter_}/{config.total_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms')
        if master_process:
            wandb.log({"val_loss": val_loss}, step=iter_)

        # start the clock again.
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # ----------- SAVING SECTION -----------
    if master_process and iter_ > start_iter and (last_iter or (config.save_every > 0 and iter_ % config.save_every == 0)):
        # stop the clock.
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # save the state of the training process.
        log = dict(
            iteration=iter_,
            run_id=run_id,
            model=raw_model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
            schedulers=[sched.state_dict() for sched in schedulers],
            training_time_ms=training_time_ms,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all()
        )
        torch.save(log, os.path.join(logdir, f"state_step{iter_:06d}.pt"))
        print0(f"saved model checkpoint to {logdir}/state_step{iter_:06d}.pt")
        # start the clock again.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if last_iter:
        break

    # ----------- TRAINING SECTION -----------
    for i in range(1, accumulation_steps+1):
        # forward pass.
        with ctx:
            loss = model(input_ids=x, labels=y).loss
            train_loss = loss.detach()
        # prepare next batch.
        x, y = train_loader.next_batch()
        # backward pass.
        if i < accumulation_steps:
            with model.no_sync():
                (loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward() # just sync on the last step
    # clip those gradients.
    if config.grad_norm_clip is not None:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip, foreach=True)
    else:
        grad_norm = torch.tensor(0.)
    # step the optimizers & schedulers.
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null those gradients.
    model.zero_grad(set_to_none=True)

    # ----------- LOGGING SECTION -----------
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    avg_step_time = approx_training_time_ms / (iter_ + 1 - WARMUP_SKIP) if iter_ >= WARMUP_SKIP else 0
    extra = " ".join(f"{k}:{v}" for k, v in (to_log or {}).items())
    print0(f"iteration:{iter_+1:0{len(str(config.total_iterations))}d}/{config.total_iterations} train_loss:{train_loss.item():.4f} lr: {schedulers[0].get_last_lr()[0]:.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{avg_step_time:.2f}ms {extra}")
    if master_process:
        wandb.log({'train_loss': train_loss.item(), 'step_avg_time': avg_step_time, **{f'lr_{i}': sched.get_last_lr()[0] for i, sched in enumerate(schedulers)}, 'grad_norm': grad_norm.item(), **to_log}, step=iter_)

print0(f"peak memory consumption during training: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0("Training complete.")
dist.destroy_process_group()

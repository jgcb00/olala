from dataclasses import dataclass
import tyro
from pathlib import Path

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .configuration_olala import OlalaConfig
from .modeling_olala import OlalaForCausalLM
from .coordcheck_utils import get_coord_data, plot_coord_data

# TRITON_HOME="/p/project1/jureap140/temp" python make_coord_check.py

@dataclass
class Args:
    save_dir: Path
    mup: bool = False
    learning_rate: float = 1e-2
    layers_config: str = "gggTgggTgggTggg"
args = tyro.cli(Args)

batch_size = 8
batch_len = 1024
max_value = 100

widths = [128, 512, 1024, 2048]
n_heads = [4, 8, 16, 32]
d_head = 64

class RandomDataset(Dataset):
    def __len__(self):
        return 9999999

    def __getitem__(self, _):
        data = torch.randint(low=0, high=max_value, size=(batch_size, batch_len))
        return data.cuda(), data.cuda()

def lazy_model(width):
    config_hf = OlalaConfig(
        layers_config=args.layers_config,
        hidden_size=width,
        intermediate_size=4*width,
        tpa_rank=4,
        token_shift_attn=True,
        head_dim=d_head,
        shrink_qk_da=1,
        num_attention_heads=n_heads[widths.index(width)],
        num_signal_heads_diff=n_heads[widths.index(width)]-n_heads[widths.index(width)]//4,
        num_key_value_heads=n_heads[widths.index(width)],
        head_dim_gdn=d_head,
        shrink_qk_gdn=2,
        num_attention_heads_gdn=n_heads[widths.index(width)],
        zero_centered_gate=True,
        zero_centered_gate_type=4,
        mamba_mimo_dim=4,
        mamba_ngroups=1,
        gate_attn=True,
        zero_centered_gamma=True,
        vocab_size=max_value,
        max_position_embeddings=1024,
        use_uscaling=True,
        uscaling_tau=0.2,
        initializer_range=1.,
        use_cache=False,
    )

    if args.mup:
        config_hf.use_uscaling = True
        config_hf.initializer_range = 1.0
    else:
        config_hf.use_uscaling = False
        config_hf.initializer_range = 0.006

    return lambda: OlalaForCausalLM(config_hf).to("cuda")

def param_groups_mup(model, base_lr_hidden, base_lr_scalar, base_lr_embed, base_lr_head, wd):
    groups, seen = [], set()
    id2name = {id(p): n for n, p in model.named_parameters()}

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            pname = id2name.get(id(mod.weight), "")
            is_scalar = getattr(mod, "is_scalar_weight", False)
            fan_in = mod.weight.shape[1]
            scale = 1 / math.sqrt(fan_in)
            if "lm_head" in pname:
                lr_scaled = base_lr_head
                wd_scaled = 0.0
            elif is_scalar:
                lr_scaled = base_lr_scalar
                wd_scaled = 0.0
            else:
                lr_scaled = base_lr_hidden * scale
                wd_scaled = wd / lr_scaled

            groups.append({"params": [mod.weight], "lr": lr_scaled, "weight_decay": wd_scaled})
            seen.add(mod.weight)

            if mod.bias is not None:
                groups.append({"params": [mod.bias], "lr": base_lr_scalar, "weight_decay": 0.0})
                seen.add(mod.bias)

    for p in model.parameters():
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
        if getattr(p, "requires_weight_decay", False):
            wd_scaled = wd / lr_scaled

        groups.append({"params": [p], "lr": lr_scaled, "weight_decay": wd_scaled})

    return groups

models = {width: lazy_model(width) for width in widths}

dataset = RandomDataset()
loader = DataLoader(dataset, batch_size=None, shuffle=True)
iter_ = iter(loader)

def get_optim(model):
    if args.mup:
        param_list = param_groups_mup(
            model,
            base_lr_hidden=args.learning_rate,
            base_lr_scalar=2**-6,
            base_lr_embed=2**-4,
            base_lr_head=2**-6,
            wd=0.,
        )
        optimizer = torch.optim.AdamW(param_list, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0., betas=(0.9, 0.95), eps=1e-8)
    return optimizer
optcls = lambda model: get_optim(model)

df = get_coord_data(models, iter_, optcls, nsteps=10)

if args.mup:
    name = f"mup_{args.learning_rate}_{args.layers_config}.png"
else:
    name = f"sp_{args.learning_rate}_{args.layers_config}.png"

plot_coord_data(df, legend="full", save_to=args.save_dir / name)

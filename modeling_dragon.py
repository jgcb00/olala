# coding=utf-8
"""PyTorch Dragon model."""

from typing import Any, Dict, Optional, Tuple, Union, List, Literal
from dataclasses import dataclass
import inspect
from contextlib import nullcontext

import math
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging

# LOAD ORDER MATTERS, keep this above the mamba_ssm import below.
# mamba_ssm pulls in tilelang, which ships libcudart_stub.so exporting the cudart
# symbols itself. Any CUDA extension dlopened AFTER it binds its cudart calls to
# that stub, whose own fallback resolution then fails -> "TileLang Error:
# libcudart symbols not found globally" + abort() (kills the process at the first
# flash-attn call, no Python traceback). Loading flash-attn's .so first makes it
# bind to the real libcudart.
try:
    import flash_attn_interface  # noqa: F401  (FA3, imports flash_attn_3._C)
except ImportError:
    try:
        import flash_attn  # noqa: F401  (FA2)
    except ImportError:
        pass

try:
    from flash_attn.modules.mlp import GatedMlp
except ImportError:
    GatedMlp = None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    print("Warning: No mamba-ssm found !")
    mamba_chunk_scan_combined = None
    RMSNormGated = None

try:
    from dragon_mamba3_fast_step.fused_mimo_variant.mamba3_tilelang import mamba3_tilelang
    from dragon_mamba3_fast_step.fused_mimo_variant.mamba3_rotary_step import apply_rotary_qk_inference_fwd
    from dragon_mamba3_fast_step.fused_mimo_variant.mamba3_step_fn import mamba3_step_fn
    from dragon_mamba3_fast_step.angle_cumsum import angle_dt
except ImportError:
    # Fallback: the public state-spaces/mamba package ships the same Mamba-3
    # MIMO kernel (mamba3_mimo) — it's what Megatron-LM training uses too.
    # Differences vs the private dragon_mamba3_fast_step API:
    #  - it takes RAW angles and applies the angle cumsum internally, so
    #    angle_dt must NOT be applied beforehand (angle_dt = None signals that
    #    in forward);
    #  - it returns a bare tensor when return_state=False (the wrapper below
    #    normalizes to the (y, state) tuple this file unpacks);
    #  - there is NO public decode/step kernel (mamba3_step_fn) -> call
    #    generate() with use_cache=False (full-prefix recompute per token).
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
        print("dragon_mamba3_fast_step not found")

try:
    import scattermoe
    from scattermoe.mlp import MLP as ScatterMoE
    scattermoe.kernels.ops.ALLOW_TF32 = False
except ImportError:
    pass

from .configuration_dragon import DragonConfig

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.utils import prepare_sequence_ids
except ImportError:
    prepare_sequence_ids = None

logger = logging.get_logger(__name__)

try:
    from cut_cross_entropy import linear_cross_entropy
except ImportError:
    linear_cross_entropy = None

# attention backend selection
ATTN_IMPL = "eager"
try:
    import flash_attn_interface # FA3
    flash_attn_func = flash_attn_interface.flash_attn_func
    flash_attn_varlen_func = flash_attn_interface.flash_attn_varlen_func
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    if not _flash_supports_window_size:
        raise ImportError("flash_attn_func does not support window_size parameter. Please update to more recent flash_attn version")
    ATTN_IMPL = "fa3"
except ImportError:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func # FA2
        ATTN_IMPL = "fa2"
    except ImportError:
        try:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
            flex_attention = torch.compile(flex_attention)
            ATTN_IMPL = "flex"
        except Exception:
            logger.warning_once(
                "Neither Flash Attention nor Flex Attention is not installed, using eager attention implementation. "
                "For better performance, consider installing flash-attention (https://github.com/Dao-AILab/flash-attention)."
            )

# Gated DeltaNet backend selection
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError:
    logger.warning_once("Falling back to Torch implementation for Gated DeltaNet as flash-linear-attention module was not found.")
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None

# 1D short convolution backend selection
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    logger.warning_once("Falling back to Torch implementation for the short convolution as causal-conv1d module was not found.")
    causal_conv1d_fn, causal_conv1d_update = None, None

print(f"Using attention implementation: {ATTN_IMPL}")

logger.info(f"Using attention implementation: {ATTN_IMPL}")
logger.info(f"Using Gated DeltaNet implementation: {'fla' if chunk_gated_delta_rule is not None else 'torch'}")
logger.info(f"Using short convolution implementation: {'causal-conv1d' if causal_conv1d_fn is not None else 'torch'}")

# Parameter-free placeholders used for pruned/skipped MoE blocks.  When
# config.skip_moe_layers includes a layer's index, DragonMonoBlock installs
# these so the MLP phase becomes a no-op (residual passes through unchanged).
# The analogous skip_mixer_layers config skips the *mixer* phase (Mamba/Attn)
# entirely — the residual passes through unchanged on that phase.
class _NullMlp(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _NullGeodesic(nn.Module):
    def forward(self, residual, g):
        return residual


class _NullMixer(nn.Module):
    """Stand-in for a DragonMonoBlock mixer when the layer is in
    ``config.skip_mixer_layers``. Returns (zeros_like_input, None, None)."""
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states), None, None


class DragonHeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-6, zero_centered_gamma=False):
        super().__init__()
        self.rms = nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(n_heads, d_head)) if zero_centered_gamma else nn.Parameter(torch.ones(n_heads, d_head))
        self.zero_centered_gamma = zero_centered_gamma

    def forward(self, hidden_states):
        B, L, H, D = hidden_states.shape
        y = self.rms(hidden_states) * (1.0 + self.weight.view(1, 1, H, D)) if self.zero_centered_gamma else self.rms(hidden_states) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)

class DragonNorm(nn.Module):
    def __init__(self, config: DragonConfig, hidden_size: int):
        super().__init__()
        if config.normalization_type == "rmsnorm":
            self.norm = DragonRMSNorm(hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        else:
            raise ValueError(f"Unknown normalization_type: {config.normalization_type}")

    def forward(self, hidden_states):
        return self.norm(hidden_states)

class DragonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, zero_centered_gamma=False):
        super().__init__()
        self.rms = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(hidden_size)) if zero_centered_gamma else nn.Parameter(torch.ones(hidden_size))
        self.zero_centered_gamma = zero_centered_gamma

    def forward(self, hidden_states):
        y = self.rms(hidden_states) * (1.0 + self.weight) if self.zero_centered_gamma else self.rms(hidden_states) * self.weight
        return y

class DragonLinear(nn.Linear):
    def __init__(self, config: DragonConfig, in_features, out_features, bias=False, cosnet=True, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.config = config

        if self.config.cosnet and cosnet:
            self.cosnet_branch = DragonCosNetBranch(
                in_features=in_features,
                out_features=out_features,
                rank=config.cosnet_rank,
            )

    def forward(self, x):
        out = super().forward(x)
        if self.config.cosnet:
            out = out + self.cosnet_branch(x)
        return out

class DragonCosNetBranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        wmin: float = 0.8,
        wmax: float = 1.2,
        phase_std: float = 0.1,
        up_scale: float = 0.01,
    ):
        super().__init__()

        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.mix = nn.Linear(rank, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.up.weight.dim_factor = min(in_features, out_features)
        self.mix.weight.dim_factor = min(in_features, out_features)

        self.omega1 = nn.Parameter(torch.ones(rank))
        self.phi1 = nn.Parameter(torch.ones(rank))
        self.omega2 = nn.Parameter(torch.ones(rank))
        self.phi2 = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        h = self.down(x)
        h = torch.cos(h * self.omega1 + self.phi1)
        h = self.mix(h)
        h = torch.cos(h * self.omega2 + self.phi2)
        return self.up(h)

class HybridDragonDynamicCache(DynamicCache):
    """
    A dynamic cache that handle both the attention cache (which has a seq_len dimension) and the GDN cache
    (which has a constant shape regardless of seq_len).
    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for GDN cache. The expected shape for each tensor is as follows:
    For each layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    if local attention produce k and v 
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """
    def __init__(self, config: DragonConfig):
        super().__init__()
        self.config = config
        # If layer_idx_map is set (pruned-block variant), blocks may index
        # into these caches at their ORIGINAL layer_idx, which can exceed
        # len(layers_config). Size the per-layer lists by the largest
        # original index + 1 so writes don't go out of bounds.
        _map = getattr(config, "layer_idx_map", None)
        cache_n = max(_map) + 1 if _map else len(config.layers_config)
        # attention
        self._key_cache = {}
        self._value_cache = {}
        # attention - kv shift
        self._kv_shift_last_k = [None for _ in range(cache_n)] # (B, H_kv, D)
        self._kv_shift_last_v = [None for _ in range(cache_n)] # (B, H_kv, D)
        # cca
        self.cca_qk0_cache = []
        self.cca_qk1_cache = []
        self.cca_prev_hidden = []
        # gdn
        self.conv_caches = []
        self.ssm_caches = []
        # mamba3
        self.mamba3_hs = [None for _ in range(cache_n)]
        self.mamba3_angle_states = [None for _ in range(cache_n)]
        self.mamba3_ssm_states = [None for _ in range(cache_n)]
        self.mamba3_k_states = [None for _ in range(cache_n)]
        self.mamba3_v_states = [None for _ in range(cache_n)]

        # Build per-position auxiliary caches keyed by ORIGINAL layer_idx
        # when a layer_idx_map is present. Otherwise use the position
        # sequence as before.
        _origs = _map if _map else list(range(len(config.layers_config)))
        # we still want the type info from layers_config; map position i ->
        # layers_config[i] (i is also the new position), but cache lists
        # below should be addressable up to cache_n.
        for i, orig in enumerate(_origs):
            layer_type = config.layers_config[i]
            if not layer_type == "r":
                self._key_cache[orig] = None
                self._value_cache[orig] = None

        for _ in range(cache_n):
            self.cca_qk0_cache.append(None)
            self.cca_qk1_cache.append(None)
            self.cca_prev_hidden.append(None)
            self.conv_caches.append(None)
            self.ssm_caches.append(None)

        self.window_size = config.sliding_window_size
        self.layers_config = config.layers_config
        self.past_length = [0 for _ in range(cache_n)]

    def update(
        self,
        k: torch.Tensor, # (B, L, h, D)
        v: torch.Tensor, # (B, L, h, D)
        layer_idx: int,
    ):
        added_len = k.size(1)
        # grab cache
        k_cache = self._key_cache[layer_idx]
        v_cache = self._value_cache[layer_idx]
        if k_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([k_cache, k], dim=1)
            v_cache = torch.cat([v_cache, v], dim=1)
        # save cache
        self._key_cache[layer_idx] = k_cache
        self._value_cache[layer_idx] = v_cache
        # update cache length
        self.past_length[layer_idx] += added_len
        return k_cache, v_cache

    # cca
    def get_cca_qk0_state(self, layer_idx):
        return self.cca_qk0_cache[layer_idx]

    def set_cca_qk0_state(self, layer_idx, state):
        self.cca_qk0_cache[layer_idx] = state
    
    def get_cca_qk1_state(self, layer_idx):
        return self.cca_qk1_cache[layer_idx]

    def set_cca_qk1_state(self, layer_idx, state):
        self.cca_qk1_cache[layer_idx] = state

    def get_prev_hidden(self, layer_idx):
        return self.cca_prev_hidden[layer_idx]

    def set_prev_hidden(self, layer_idx, h):
        self.cca_prev_hidden[layer_idx] = h
    
    # cca v2
    def update_conv_state(self, layer_idx: int, new_conv_state: torch.Tensor) -> torch.Tensor:
        if not self.has_previous_state:
            self.conv_states[layer_idx] = new_conv_state#.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :]#.to(self.conv_states.device)
        return self.conv_states[layer_idx]

    # kv shift
    def get_last_kv(self, layer_idx):
        return self._kv_shift_last_k[layer_idx], self._kv_shift_last_v[layer_idx]

    def set_last_kv(self, layer_idx, k_last, v_last):
        self._kv_shift_last_k[layer_idx] = k_last
        self._kv_shift_last_v[layer_idx] = v_last

    def trim(self, layer_idx: int):
        # discard old keys/values
        window_size = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size
        if self.layers_config[layer_idx] == 'l':
            if self._key_cache[layer_idx].size(1) > window_size:
                self._key_cache[layer_idx] = self._key_cache[layer_idx][:, -window_size:, ...].contiguous()
                self._value_cache[layer_idx] = self._value_cache[layer_idx][:, -window_size:, ...].contiguous()

    def get_total_seen(self, layer_idx: int) -> int:
        return self.past_length[layer_idx]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridDragonDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, cache_params: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridDragonDynamicCache does not have a legacy cache equivalent.")

class DragonRotaryEmbedding(torch.nn.Module):
    def __init__(self, config: DragonConfig, head_dim: int, theta: float):
        super().__init__()
        self.config = config

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, position_ids):
        max_pos = self.config.max_position_embeddings
        if max_pos > self.seq_len_cached:
            self.seq_len_cached = max(2 * max_pos, 16)
            t = torch.arange(self.seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(torch.bfloat16)
            self.sin_cached = freqs.sin().to(torch.bfloat16)

        cos = self.cos_cached[position_ids] # (B, T, head_dim/2)
        sin = self.sin_cached[position_ids]
        cos = cos[..., None, :]       # (B, T, 1, head_dim/2), broadcasts over heads
        sin = sin[..., None, :]

        return cos, sin

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2 # head dim
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

def apply_p_rotary_emb(x, cos, sin, p: float = 0.75):
    """Partial RoPE: rotate only the top p fraction of (half) dims; rest are identity."""
    assert x.ndim == 4 and 0.0 <= p <= 1.0  # x: (B, L, H, D)
    d = x.shape[3] // 2                     # half-dim per your layout
    rope_d = int(d * p)

    x1, x2 = x[..., :d], x[..., d:]

    if rope_d > 0:
        y1_head = x1[..., :rope_d] * cos[..., :rope_d] + x2[..., :rope_d] * sin[..., :rope_d]
        y2_head = x1[..., :rope_d] * (-sin[..., :rope_d]) + x2[..., :rope_d] * cos[..., :rope_d]
        y1 = torch.cat([y1_head, x1[..., rope_d:]], dim=-1)
        y2 = torch.cat([y2_head, x2[..., rope_d:]], dim=-1)
    else:
        y1, y2 = x1, x2

    return torch.cat([y1, y2], dim=-1).type_as(x)

# heavily adapated from Gemma3
def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    softcap: Optional[float] = None,
    softmax_scale: Optional[float] = None,    
    **kwargs,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = query.size(3)**-0.5
    if window_size == (-1, 0):
        window_size = None

    query = query.transpose(1, 2) # (B, H, L, D)
    key = key.transpose(1, 2) # (B, h, L, D)
    value = value.transpose(1, 2) # (B, h, L, D)

    key = key.repeat_interleave(query.size(1) // key.size(1), dim=1)
    value = value.repeat_interleave(query.size(1) // value.size(1), dim=1)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * softmax_scale

    if softcap is not None and softcap > 0.:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap

    if causal or (window_size is not None):
        Lq = query.size(2)
        Lk = key.size(2)
        past = max(Lk - Lq, 0)
        i = torch.arange(Lq, device=attn_weights.device).unsqueeze(1) + past # [Lq,1]
        j = torch.arange(Lk, device=attn_weights.device).unsqueeze(0) # [1,Lk]

        allowed = torch.ones((Lq, Lk), dtype=torch.bool, device=attn_weights.device)
        if causal:
            allowed &= (j <= i) # prevent attending to future positions
        if window_size is not None:
            w_left, w_right = window_size
            # treat None as "no limit" on that side
            if w_left is None:
                w_left = Lk
            if w_right is None:
                w_right = Lk
            allowed &= (j >= i - w_left) & (j <= i + w_right)
        # broadcast [Lq,Lk] -> [B, H, Lq, Lk]
        attn_weights = attn_weights.masked_fill(~allowed, float("-inf"))

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output

def get_query_key_value_tensors(module: nn.Module, hidden_states: torch.Tensor):
    """
    Derives `query`, `key` and `value` tensors from `hidden_states`.
    """
    # (B, L, D) -> (B, L, ng * (np/ng + 2) * hn))
    mixed_qkv = module.linear_qkv(hidden_states)

    if getattr(module, "reuse_kv", False):
        # reshape to [..., num_query_groups, heads_per_group * d]
        q_dim = (module.num_attention_heads // module.num_key_value_heads) * module.head_dim
        new_shape = mixed_qkv.size()[:-1] + (module.num_key_value_heads, q_dim)
        query = mixed_qkv.view(*new_shape)
        # final shape (B, L, H, d)
        query = query.reshape(query.size(0), query.size(1), -1, module.head_dim)

        return query

    # (B, L, hp) -> (B, L, ng, (np/ng + 2) * hn)
    new_tensor_shape = mixed_qkv.size()[:-1] + (
        module.num_key_value_heads,
        (
            (module.num_attention_heads // module.num_key_value_heads + 2)
            * module.head_dim
        ),
    )
    mixed_qkv = mixed_qkv.view(*new_tensor_shape)

    split_arg_list = [
        (
            module.num_attention_heads
            // module.num_key_value_heads
            * module.head_dim
        ),
        module.head_dim,
        module.head_dim,
    ]

    # [B, L, ng, (np/ng + 2) * hn] -> [B, L, ng, np/ng * hn], [B, L, ng, hn], [B, L, ng, hn]
    (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

    # [B, L, ng, np/ng * hn] -> [B, L, np, hn]
    query = query.reshape(query.size(0), query.size(1), -1, module.head_dim)

    return query, key, value

class DragonAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified to use sliding window attention: Longformer and "Generating Long Sequences with Sparse Transformers".
    Doesn't include output projection: output is (B, L, H, D).
    """

    def __init__(self, config: DragonConfig, reuse_kv: bool, layer_idx: Optional[int], **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.qk_norm = config.qk_norm
        self.window_size = config.sliding_window_size
        self.reuse_kv = reuse_kv

        projection_dim = self.head_dim * (self.num_attention_heads + 2 * (0 if reuse_kv else self.num_key_value_heads))
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        if self.qk_norm:
            self.q_norm = DragonNorm(config, self.head_dim)
            if not reuse_kv:
                self.k_norm = DragonNorm(config, self.head_dim)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_attn > 0.:
                    score = self.config.softcap_attn * torch.tanh(score / self.config.softcap_attn)
                return score
            self.score_mod = score_mod
            # block mask (for causal & sliding window)
            def build_mask(wsize):
                if wsize == -1:
                    wsize = self.config.max_position_embeddings
                def sliding_window(b, h, q_idx, kv_idx):
                    return q_idx - kv_idx <= wsize
                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx
                self.attn_mask = and_masks(causal_mask, sliding_window)
                return wsize
            self.build_mask = build_mask
            self.last_wsize = self.build_mask(min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ):
        _, q_len, _ = hidden_states.shape
        use_precomputed_states = (cache_params is not None and q_len == 1)

        # Q, K, V projections.
        if not self.reuse_kv:
            query_states, key_states, value_states = get_query_key_value_tensors(self, hidden_states)
        else:
            query_states = get_query_key_value_tensors(self, hidden_states)
            key_states, value_states = key_value_last_layer
            last_key_states, last_value_states = None, None

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            if not self.reuse_kv:
                key_states = self.k_norm(key_states)

        # RoPE.
        if self.config.rope_type != "" and self.config.rope_theta > 0.0:
            cos, sin = position_embeddings
            if self.config.rope_type == "rope":
                query_states = apply_rotary_emb(query_states, cos, sin)
                if not self.reuse_kv:
                    key_states = apply_rotary_emb(key_states, cos, sin)
            elif self.config.rope_type == "p-rope":
                query_states = apply_p_rotary_emb(query_states, cos, sin, p=0.5)
                if not self.reuse_kv:
                    key_states = apply_p_rotary_emb(key_states, cos, sin)
            else:
                raise ValueError(f"Unknow rope type : {self.config.rope_type}")

        # KV-cache.
        if not self.reuse_kv and cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # save k,v for next layer (*after* norm and RoPE and kv-cache update)
        if not self.reuse_kv:
            last_key_states, last_value_states = key_states, value_states

        # attention computation.
        wsize = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size

        if ATTN_IMPL == "eager":
            assert not self.config.intra_doc_masking
            attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "flex":
            if wsize != self.last_wsize:
                self.last_wsize = self.build_mask(wsize)
            attention_interface = lambda q, k, v, softmax_scale, **kw: flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1)), score_mod=self.score_mod, scale=softmax_scale, enable_gqa=self.num_attention_heads > self.num_key_value_heads).transpose(1, 2)
        elif ATTN_IMPL == "fa2":
            if not self.config.intra_doc_masking:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        elif ATTN_IMPL == "fa3":
            if not self.config.intra_doc_masking:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        attn_output = attention_interface(
            query_states.bfloat16(),
            key_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.config.softcap_attn,
            softmax_scale=None if not self.config.use_completed_p else 1/self.head_dim,
        )
        if len(attn_output.shape) == 3:
            attn_output = attn_output.view(query_states.size(0), query_states.size(1), attn_output.size(-2), attn_output.size(-1)) # keep (B, L, H, D)

        #if cache_params is not None and not self.reuse_kv:
        #    cache_params.trim(self.layer_idx)

        return attn_output, last_key_states, last_value_states

class DragonTensorProductAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified to use sliding window attention: Longformer and "Generating Long Sequences with Sparse Transformers".
    Doesn't include output projection: output is (B, L, H, D).
    """

    def __init__(self, config: DragonConfig, reuse_kv: bool, layer_idx: Optional[int], **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rank = config.tpa_rank
        self.qk_norm = config.qk_norm
        self.window_size = config.sliding_window_size
        self.reuse_kv = reuse_kv

        self.c_q = DragonLinear(config, self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.W_A_k = DragonLinear(config, self.hidden_size, self.num_attention_heads * self.rank, bias=False)
        self.W_A_v = DragonLinear(config, self.hidden_size, self.num_attention_heads * self.rank, bias=False)
        self.W_B_k = DragonLinear(config, self.hidden_size, self.rank * self.head_dim, bias=False)
        self.W_B_v = DragonLinear(config, self.hidden_size, self.rank * self.head_dim, bias=False)

        if self.config.token_shift_attn:
            self.shift_proj_k = DragonLinear(config, self.hidden_size, self.num_attention_heads, bias=False)
            self.shift_proj_v = DragonLinear(config, self.hidden_size, self.num_attention_heads, bias=False)
            if self.config.scalar_proj_as_hidden_matrix:
                self.shift_proj_k.is_scalar_weight = True
                self.shift_proj_v.is_scalar_weight = True

        if self.config.token_conv1d_attn:
            self.conv_size = config.conv_kernel
            self.conv_dim = self.num_attention_heads * self.head_dim + self.num_attention_heads * self.head_dim + self.num_attention_heads * self.head_dim
            self.qkv_conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_size, groups=self.conv_dim, padding=self.conv_size-1)
            self.causal_conv1d_fn = causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update

        if self.qk_norm:
            self.q_norm = DragonNorm(config, self.head_dim)
            if not reuse_kv:
                self.k_norm = DragonNorm(config, self.head_dim)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_attn > 0.:
                    score = self.config.softcap_attn * torch.tanh(score / self.config.softcap_attn)
                return score
            self.score_mod = score_mod
            # block mask (for causal & sliding window)
            def build_mask(wsize):
                if wsize == -1:
                    wsize = self.config.max_position_embeddings
                def sliding_window(b, h, q_idx, kv_idx):
                    return q_idx - kv_idx <= wsize
                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx
                self.attn_mask = and_masks(causal_mask, sliding_window)
                return wsize
            self.build_mask = build_mask
            self.last_wsize = self.build_mask(min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        b, q_len, _ = hidden_states.shape
        use_precomputed_states = (cache_params is not None and q_len == 1)

        # Q, K, V projections.
        query_states = self.c_q(hidden_states).view(b, q_len, self.num_attention_heads, self.head_dim)
        A_k = self.W_A_k(hidden_states).view(b, q_len, self.num_attention_heads, self.rank)
        A_v = self.W_A_v(hidden_states).view(b, q_len, self.num_attention_heads, self.rank)
        B_k = self.W_B_k(hidden_states).view(b, q_len, self.rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).view(b, q_len, self.rank, self.head_dim)
        # (rope done on query_states and B_k)
        A_k = A_k.view(b * q_len, self.num_attention_heads, self.rank)
        A_v = A_v.view(b * q_len, self.num_attention_heads, self.rank)
        B_k = B_k.view(b * q_len, self.rank, self.head_dim)
        B_v = B_v.view(b * q_len, self.rank, self.head_dim)
        key_states = torch.bmm(A_k, B_k).div_(self.rank).view(b, q_len, self.num_attention_heads, self.head_dim)
        value_states = torch.bmm(A_v, B_v).div_(self.rank).view(b, q_len, self.num_attention_heads, self.head_dim)

        # token-shift.
        if self.config.token_shift_attn and not self.reuse_kv:
            alpha_k = torch.sigmoid(self.shift_proj_k(hidden_states).float()).float().to(key_states.dtype).unsqueeze(-1) # (B, L, Hkv, 1)
            alpha_v = torch.sigmoid(self.shift_proj_v(hidden_states).float()).float().to(value_states.dtype).unsqueeze(-1) # (B, L, Hkv, 1)

            if cache_params is not None:
                k_prev, v_prev = cache_params.get_last_kv(self.layer_idx)
                if k_prev is None:
                    k_prev, v_prev = torch.zeros_like(key_states[:, :1]), torch.zeros_like(value_states[:, :1])
                cache_params.set_last_kv(self.layer_idx, key_states[:, -1:], value_states[:, -1:])
            else:
                k_prev = F.pad(key_states, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)
                v_prev = F.pad(value_states, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)

            key_states = alpha_k * k_prev + (1 - alpha_k) * key_states
            value_states = alpha_v * v_prev + (1 - alpha_v) * value_states

        # conv.
        if self.config.token_conv1d_attn:
            assert not self.reuse_kv, "not supported"
            # --- pack for conv ---
            q_proj = rearrange(query_states, "b l h d -> b l (h d)")
            k_proj = rearrange(key_states, "b l g d -> b l (g d)")
            v_proj = rearrange(value_states, "b l g d -> b l (g d)")
            mixed_qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2) # (B,C,L)

            if cache_params is not None:
                conv_cache = cache_params.conv_caches[self.layer_idx]

            if use_precomputed_states:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_cache,
                    self.qkv_conv1d.weight.squeeze(1),
                    self.qkv_conv1d.bias,
                    'silu',
                ) # conv_cache is updated in-place here
            else:
                if cache_params is not None:
                    conv_cache = F.pad(mixed_qkv, (self.conv_size - mixed_qkv.shape[-1], 0))
                    cache_params.conv_caches[self.layer_idx] = conv_cache
                if self.causal_conv1d_fn is not None:
                    mixed_qkv = self.causal_conv1d_fn(
                        x=mixed_qkv,
                        weight=self.qkv_conv1d.weight.squeeze(1),
                        bias=self.qkv_conv1d.bias,
                        activation='silu',
                        seq_idx=None,
                    )
                else:
                    mixed_qkv = F.silu(self.qkv_conv1d(mixed_qkv)[:, :, :q_len])

            # split back
            mixed_qkv = mixed_qkv.transpose(1, 2)
            q_proj, k_proj, v_proj = torch.split(
                mixed_qkv,
                [self.num_attention_heads*self.head_dim, self.num_attention_heads*self.head_dim, self.num_attention_heads*self.head_dim],
                dim=-1,
            )
            query_states = rearrange(q_proj, "b l (h d) -> b l h d", h=self.num_attention_heads)
            key_states = rearrange(k_proj, "b l (g d) -> b l g d", g=self.num_attention_heads)
            value_states = rearrange(v_proj, "b l (g d) -> b l g d", g=self.num_attention_heads)

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            if not self.reuse_kv:
                key_states = self.k_norm(key_states)

        # RoPE.
        if self.config.rope_theta > 0.0:
            cos, sin = position_embeddings
            if self.config.rope_type == "rope":
                query_states = apply_rotary_emb(query_states, cos, sin)
                if not self.reuse_kv:
                    key_states = apply_rotary_emb(key_states, cos, sin)
            elif self.config.rope_type == "p-rope":
                query_states = apply_p_rotary_emb(query_states, cos, sin)
                if not self.reuse_kv:
                    key_states = apply_p_rotary_emb(key_states, cos, sin)
            else:
                raise ValueError(f"Unknow rope type : {self.config.rope_type}")

        # KV-cache.
        if not self.reuse_kv and cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # save k,v for next layer (*after* norm and RoPE and kv-cache update)
        if not self.reuse_kv:
            last_key_states, last_value_states = key_states, value_states

        # attention computation.
        wsize = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size

        if ATTN_IMPL == "eager":
            attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "flex":
            if wsize != self.last_wsize:
                self.last_wsize = self.build_mask(wsize)
            attention_interface = lambda q, k, v, softmax_scale, **kw: flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1)), score_mod=self.score_mod, scale=softmax_scale, enable_gqa=self.num_attention_heads > self.num_key_value_heads).transpose(1, 2)
        elif ATTN_IMPL == "fa2":
            attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "fa3":
            attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)[0]
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        attn_output = attention_interface(
            query_states.bfloat16(),
            key_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.config.softcap_attn,
            softmax_scale=None if not self.config.use_completed_p else 1/self.head_dim,
        )
        if len(attn_output.shape) == 3:
            attn_output = attn_output.view(query_states.size(0), query_states.size(1), attn_output.size(-2), attn_output.size(-1)) # keep (B, L, H, D)

        #if cache_params is not None and not self.reuse_kv:
        #    cache_params.trim(self.layer_idx)

        return attn_output, last_key_states, last_value_states

class DragonDifferentialAttentionV2(nn.Module):
    """
    https://spiky-homegrown-4cb.notion.site/Differential-Transformer-V2-2e7baa052def80ecaa93d4d67d125417
    """

    def __init__(self, config: DragonConfig, layer_idx: Optional[int], **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.num_attention_heads = config.num_attention_heads
        self.num_signal_heads = config.num_signal_heads_diff if config.num_signal_heads_diff else self.num_attention_heads//2
        self.num_noise_heads = self.num_attention_heads - self.num_signal_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.qk_norm = config.qk_norm
        self.softcap = config.softcap_attn
        self.scalable_softmax = config.scalable_softmax

        assert self.num_attention_heads % config.num_key_value_heads == 0, "number of attention heads must be a multiple of number of key/value heads."
        assert self.num_signal_heads % self.num_noise_heads == 0, "number of signal heads must be a multiple of number of noise heads."
        self.gqa = self.num_attention_heads // config.num_key_value_heads
        self.snr = self.num_signal_heads // self.num_noise_heads
        self.num_key_value_heads = self.num_attention_heads // (self.gqa * self.snr)

        # are these two needed?
        #assert self.num_signal_heads % self.gqa == 0, "GQA factor must divide number of signal heads."
        #assert self.num_noise_heads % self.gqa == 0, "GQA factor must divide number of noise heads."

        projection_dim = self.head_dim * self.num_attention_heads + 2 * self.head_dim * self.num_key_value_heads
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        if self.qk_norm:
            self.q_norm = DragonNorm(config, self.head_dim)
            self.k_norm = DragonNorm(config, self.head_dim)

        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.num_attention_heads, dtype=torch.float32))

        self.lambda_proj = DragonLinear(config, config.hidden_size, self.num_noise_heads, bias=False)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_attn > 0.:
                    score = self.config.softcap_attn * torch.tanh(score / self.config.softcap_attn)
                return score
            self.score_mod = score_mod
            # block mask (for causal & sliding window)
            def build_mask(wsize):
                if wsize == -1:
                    wsize = self.config.max_position_embeddings
                def sliding_window(b, h, q_idx, kv_idx):
                    return q_idx - kv_idx <= wsize
                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx
                self.attn_mask = and_masks(causal_mask, sliding_window)
                return wsize
            self.build_mask = build_mask
            self.last_wsize = self.build_mask(self.config.slw_wsize)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ):
        _, q_len, _ = hidden_states.shape
        use_precomputed_states = (cache_params is not None and q_len == 1)

        # Q, K, V projections.
        #query_states, key_states, value_states = get_query_key_value_tensors(self, hidden_states)
        mixed_qkv = self.linear_qkv(hidden_states)
        query_states, key_states, value_states = torch.split(
            mixed_qkv,
            [self.num_attention_heads * self.head_dim,
             self.num_key_value_heads * self.head_dim,
             self.num_key_value_heads * self.head_dim],
            dim=-1,
        ) # WARNING: not TP aware
        query_states = rearrange(query_states, "b l (h d) -> b l h d", h=self.num_attention_heads)
        key_states   = rearrange(key_states, "b l (h d) -> b l h d", h=self.num_key_value_heads)
        value_states = rearrange(value_states, "b l (h d) -> b l h d", h=self.num_key_value_heads)
        assert query_states.size(3) == self.head_dim
        assert key_states.size(3) == self.head_dim
        assert value_states.size(3) == self.head_dim

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        wsize = self.config.slw_wsize
        if self.config.complete_slw:
            b, L = query_states.size(0), query_states.size(1)
            
            if cu_seqlens is None and max_seqlen is None and b > 1:
                raise NotImplementedError("Batch size > 1, can not implement complete slw")

            # 1. Generate window boundaries
            window_boundaries = torch.arange(0, L + wsize, wsize, device=query_states.device)
            window_boundaries = torch.unique(torch.clamp(window_boundaries, max=L))
            if cu_seqlens is None or max_seqlen is None:
                max_seqlen = wsize
                cu_seqlens = window_boundaries 
                boundaries_1d = window_boundaries
            else: 
                # Combine both 1D boundary lists directly
                combined = torch.cat([window_boundaries, cu_seqlens])
                # torch.unique automatically removes duplicates and sorts them in ascending order
                cu_seqlens = torch.unique(combined, sorted=True)
                max_seqlen = min(max_seqlen, wsize) if wsize > 0 else max_seqlen
                boundaries_1d = cu_seqlens

            # 3. Update position_ids based on the new boundaries
            seq_range = torch.arange(L, device=query_states.device)
            
            # Find which chunk index each token belongs to
            chunk_indices = torch.searchsorted(boundaries_1d, seq_range, right=True) - 1
            chunk_starts = boundaries_1d[chunk_indices]
            
            # Calculate position IDs and expand to batch size
            # position_ids usually still needs to be (b, L) for embedding layers
            position_ids = (seq_range - chunk_starts).unsqueeze(0).expand(b, -1)
                            
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            T = query_states.size(1)
            pos = (position_ids.to(torch.float32).view(position_ids.size(0), T, 1, 1) + 1.)
            log_pos = pos.log() if wsize <= 0 else torch.clamp_max(pos, wsize).log()
            query_states = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * query_states

        # KV-cache.
        if cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # attention computation.
        if ATTN_IMPL == "eager":
            assert not self.config.intra_doc_masking
            attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "flex":
            if wsize != self.last_wsize:
                self.last_wsize = self.build_mask(wsize)
            attention_interface = lambda q, k, v, softmax_scale, **kw: flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1)), score_mod=self.score_mod, scale=softmax_scale, enable_gqa=self.num_attention_heads > self.num_key_value_heads).transpose(1, 2)
        elif ATTN_IMPL == "fa2":
            if not self.config.intra_doc_masking and not self.config.complete_slw:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        elif ATTN_IMPL == "fa3":
            if not self.config.intra_doc_masking and not self.config.complete_slw:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)[0]
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw)[0].unsqueeze(0)
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        # num_heads = num_signal_heads + num_noise_heads
        # num_kv_heads = (num_signal_heads // (snr * gqa)
        # where snr = num_signal_heads // num_noise_heads
        #       gqa = num_heads // num_kv_heads
        # identity : snr+1 = num_heads/num_noise_heads

        # query_states: (B, L, num_heads, D)
        # key_states: (B, L, num_kv_heads, D)
        # value_states: (B, L, num_kv_heads, D)

        attn_output = attention_interface(
            query_states.bfloat16(),
            key_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.config.softcap_attn,
            softmax_scale=None if not self.config.use_completed_p else 1/self.head_dim,
        ) # (B, L, H, D)
        attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), -1, self.num_attention_heads//self.num_noise_heads, self.head_dim) # (B, L, num_noise_heads, snr+1, D)
        attn_sig = attn_output[:, :, :, :self.snr, :] # (B, L, num_noise_heads, snr, D)
        attn_noi = attn_output[:, :, :, self.snr:self.snr+1, :] # (B, L, num_noise_heads, 1, D)

        lambda_val = self.lambda_proj(hidden_states).unsqueeze(-1).unsqueeze(-1) # (B, L, H, 1, 1)
        attn_output = attn_sig - torch.sigmoid(lambda_val) * attn_noi # (B, L, num_noise_heads, snr, D) (each noise head is broadcasted/repeated SNR times)
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1, self.head_dim) # (B, L, num_signal_heads, D)

        return attn_output, None, None

class DragonDifferentialTensorProductAttentionV2(nn.Module):
    """
    differential attention V2 + TPA
    """

    def __init__(self, config: DragonConfig, layer_idx: Optional[int], use_ve: bool = False, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.num_attention_heads = config.num_attention_heads
        self.num_signal_heads = config.num_signal_heads_diff if config.num_signal_heads_diff else self.num_attention_heads//2
        self.num_noise_heads = self.num_attention_heads - self.num_signal_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rank = config.tpa_rank
        self.qk_norm = config.qk_norm
        self.softcap = config.softcap_attn
        self.scalable_softmax = config.scalable_softmax

        assert self.num_signal_heads % self.num_noise_heads == 0, "number of signal heads must be a multiple of number of noise heads."
        self.snr = self.num_signal_heads // self.num_noise_heads
        self.num_key_value_heads = self.num_noise_heads

        self.c_q = DragonLinear(config, self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.W_A_k = DragonLinear(config, self.hidden_size, self.num_key_value_heads * self.rank, bias=False)
        self.W_A_v = DragonLinear(config, self.hidden_size, self.num_key_value_heads * self.rank, bias=False)
        self.W_B_k = DragonLinear(config, self.hidden_size, self.rank * self.head_dim, bias=False)
        self.W_B_v = DragonLinear(config, self.hidden_size, self.rank * self.head_dim, bias=False)

        if use_ve:
            self.ve_scalars = nn.Parameter(torch.zeros(self.num_noise_heads, self.head_dim, dtype=torch.float32))

        if self.config.token_shift_attn:
            self.shift_proj_k = DragonLinear(config, self.hidden_size, self.num_key_value_heads, bias=False)
            self.shift_proj_v = DragonLinear(config, self.hidden_size, self.num_key_value_heads, bias=False)
            if self.config.scalar_proj_as_hidden_matrix:    
                self.shift_proj_k.is_scalar_weight = True
                self.shift_proj_v.is_scalar_weight = True

        if self.config.token_conv1d_attn:
            self.conv_size = config.conv_kernel
            self.conv_dim = self.num_attention_heads * self.head_dim + self.num_key_value_heads * self.head_dim + self.num_key_value_heads * self.head_dim
            self.qkv_conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_size, groups=self.conv_dim, padding=self.conv_size-1)
            self.causal_conv1d_fn = causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update

        if self.qk_norm:
            self.q_norm = DragonNorm(config, self.head_dim)
            self.k_norm = DragonNorm(config, self.head_dim)

        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.num_attention_heads, dtype=torch.float32))

        self.lambda_proj = DragonLinear(config, config.hidden_size, self.num_noise_heads, bias=False)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_attn > 0.:
                    score = self.config.softcap_attn * torch.tanh(score / self.config.softcap_attn)
                return score
            self.score_mod = score_mod
            # block mask (for causal & sliding window)
            def build_mask(wsize):
                if wsize == -1:
                    wsize = self.config.max_position_embeddings
                def sliding_window(b, h, q_idx, kv_idx):
                    return q_idx - kv_idx <= wsize
                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx
                self.attn_mask = and_masks(causal_mask, sliding_window)
                return wsize
            self.build_mask = build_mask
            self.last_wsize = self.build_mask(self.config.slw_wsize)

        if self.config.rope_theta > 0.0 and self.config.rope_type != "":
            self.rotary_emb = DragonRotaryEmbedding(config, head_dim=self.head_dim, theta=config.rope_theta)
        else:
            self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        ve=None,
        **kwargs,
    ):
        b, q_len, _ = hidden_states.shape
        use_precomputed_states = (cache_params is not None and q_len == 1)

        # Q, K, V projections.
        query_states = self.c_q(hidden_states).view(b, q_len, self.num_attention_heads, self.head_dim)
        A_k = self.W_A_k(hidden_states).view(b, q_len, self.num_key_value_heads, self.rank)
        A_v = self.W_A_v(hidden_states).view(b, q_len, self.num_key_value_heads, self.rank)
        B_k = self.W_B_k(hidden_states).view(b, q_len, self.rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).view(b, q_len, self.rank, self.head_dim)
        # (rope done on query_states and B_k)
        A_k = A_k.view(b * q_len, self.num_key_value_heads, self.rank)
        A_v = A_v.view(b * q_len, self.num_key_value_heads, self.rank)
        B_k = B_k.view(b * q_len, self.rank, self.head_dim)
        B_v = B_v.view(b * q_len, self.rank, self.head_dim)
        key_states = torch.bmm(A_k, B_k).div_(self.rank).view(b, q_len, self.num_key_value_heads, self.head_dim)
        value_states = torch.bmm(A_v, B_v).div_(self.rank).view(b, q_len, self.num_key_value_heads, self.head_dim)

        # value embeddings
        if ve is not None:
            value_states = value_states + self.ve_scalars * ve.view_as(value_states)

        # token-shift.
        if self.config.token_shift_attn:
            alpha_k = torch.sigmoid(self.shift_proj_k(hidden_states).float()).float().to(key_states.dtype).unsqueeze(-1) # (B, L, Hkv, 1)
            alpha_v = torch.sigmoid(self.shift_proj_v(hidden_states).float()).float().to(value_states.dtype).unsqueeze(-1) # (B, L, Hkv//2, 1)

            if cache_params is not None:
                k_last, v_last = cache_params.get_last_kv(self.layer_idx)
                B, L = key_states.shape[:2]

                if L == 1:
                    # decode step
                    if k_last is None:
                        k_prev = torch.zeros_like(key_states)      # (B, 1, H, D)
                        v_prev = torch.zeros_like(value_states)    # (B, 1, H, D)
                    else:
                        k_prev, v_prev = k_last, v_last            # (B, 1, H, D)
                else:
                    # prefill step: first token uses cached last, rest shift within the chunk
                    first_k = k_last if k_last is not None else torch.zeros_like(key_states[:, :1])
                    first_v = v_last if v_last is not None else torch.zeros_like(value_states[:, :1])
                    k_prev = torch.cat([first_k, key_states[:, :-1]], dim=1)   # (B, L, H, D)
                    v_prev = torch.cat([first_v, value_states[:, :-1]], dim=1) # (B, L, H, D)

                # keep caching the *raw* last KV from this chunk (matches the no-cache path)
                cache_params.set_last_kv(self.layer_idx, key_states[:, -1:], value_states[:, -1:])
            else:
                k_prev = F.pad(key_states, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)
                v_prev = F.pad(value_states, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)

            if position_ids is not None:
                # first token of each doc has pos==0
                doc_start = (position_ids == 0) # (B, L) bool
            else:
                B, L = hidden_states.shape[:2]
                doc_start = torch.zeros(B, L, dtype=torch.bool, device=hidden_states.device)
                doc_start[:, 0] = True
            m = doc_start.unsqueeze(-1).unsqueeze(-1) # (B, L, 1, 1) bool

            # zero the previous contribution at boundaries
            k_prev  = k_prev.masked_fill(m, 0)
            v_prev  = v_prev.masked_fill(m, 0)
            alpha_k = alpha_k.masked_fill(m, 0)
            alpha_v = alpha_v.masked_fill(m, 0)

            key_states = alpha_k * k_prev + (1 - alpha_k) * key_states
            value_states = alpha_v * v_prev + (1 - alpha_v) * value_states

        # conv.
        if self.config.token_conv1d_attn:
            # --- pack for conv ---
            q_proj = rearrange(query_states, "b l h d -> b l (h d)")
            k_proj = rearrange(key_states, "b l g d -> b l (g d)")
            v_proj = rearrange(value_states, "b l g d -> b l (g d)")
            mixed_qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2) # (B,C,L)

            if cache_params is not None:
                conv_cache = cache_params.conv_caches[self.layer_idx]

            if use_precomputed_states:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_cache,
                    self.qkv_conv1d.weight.squeeze(1),
                    self.qkv_conv1d.bias,
                    'silu',
                ) # conv_cache is updated in-place here
            else:
                if cache_params is not None:
                    conv_cache = F.pad(mixed_qkv, (self.conv_size - mixed_qkv.shape[-1], 0))
                    cache_params.conv_caches[self.layer_idx] = conv_cache
                if self.causal_conv1d_fn is not None:
                    mixed_qkv = self.causal_conv1d_fn(
                        x=mixed_qkv,
                        weight=self.qkv_conv1d.weight.squeeze(1),
                        bias=self.qkv_conv1d.bias,
                        activation='silu',
                        seq_idx=None,
                    )
                else:
                    mixed_qkv = F.silu(self.qkv_conv1d(mixed_qkv)[:, :, :q_len])

            # split back
            mixed_qkv = mixed_qkv.transpose(1, 2)
            q_proj, k_proj, v_proj = torch.split(
                mixed_qkv,
                [self.num_attention_heads*self.head_dim, self.num_key_value_heads*self.head_dim, self.num_key_value_heads*self.head_dim],
                dim=-1,
            )
            query_states = rearrange(q_proj, "b l (h d) -> b l h d", h=self.num_attention_heads)
            key_states = rearrange(k_proj, "b l (g d) -> b l g d", g=self.num_key_value_heads)
            value_states = rearrange(v_proj, "b l (g d) -> b l g d", g=self.num_key_value_heads)

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        wsize = self.config.slw_wsize

        #rope
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            if self.config.rope_type == "rope":
                query_states = apply_rotary_emb(query_states, cos, sin)
                key_states = apply_rotary_emb(key_states, cos, sin)
            elif self.config.rope_type == "p-rope":
                query_states = apply_p_rotary_emb(query_states, cos, sin)
                key_states = apply_p_rotary_emb(key_states, cos, sin)
            else:
                raise ValueError(f"Unknow rope type : {self.config.rope_type}")
        # scalable softmax.
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            pos = position_ids.to(torch.float32)
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            pos = pos.unsqueeze(-1).unsqueeze(-1) + 1.0
            log_pos = pos.log() if wsize <= 0 else torch.clamp_max(pos, wsize).log()
            query_states = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * query_states
            # TODO: caching mechanism for log_pos

        # KV-cache.
        if cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # attention computation.
        if ATTN_IMPL == "eager":
            assert not self.config.intra_doc_masking
            attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "flex":
            if wsize != self.last_wsize:
                self.last_wsize = self.build_mask(wsize)
            attention_interface = lambda q, k, v, softmax_scale, **kw: flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1)), score_mod=self.score_mod, scale=softmax_scale, enable_gqa=self.num_attention_heads > self.num_key_value_heads).transpose(1, 2)
        elif ATTN_IMPL == "fa2":
            if not self.config.intra_doc_masking:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        elif ATTN_IMPL == "fa3":
            if not self.config.intra_doc_masking:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        # num_heads = num_signal_heads + num_noise_heads
        # num_kv_heads = num_signal_heads // (snr * gqa)
        # where snr = num_signal_heads // num_noise_heads (typically 3)
        #       gqa = num_heads // num_kv_heads (typically 1)
        # identity : snr+1 = num_heads/num_noise_heads

        # query_states: (B, L, num_heads, D)
        # key_states: (B, L, num_kv_heads, D)
        # value_states: (B, L, num_kv_heads, D)

        attn_output = attention_interface(
            query_states.bfloat16(),
            key_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.config.softcap_attn,
            softmax_scale=None if not self.config.use_completed_p else 1/self.head_dim,
        ) # (B, L, H, D)
        if self.config.xsa:
            v_self = value_states.repeat_interleave(self.num_attention_heads // value_states.size(2), dim=2)
            v_self = F.normalize(v_self.float(), dim=-1, eps=1e-6).to(attn_output.dtype)
            attn_output = attn_output - (attn_output * v_self).sum(dim=-1, keepdim=True) * v_self

        attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), -1, self.num_attention_heads//self.num_noise_heads, self.head_dim) # (B, L, num_noise_heads, snr+1, D)
        attn_sig = attn_output[:, :, :, :self.snr, :] # (B, L, num_noise_heads, snr, D)
        attn_noi = attn_output[:, :, :, self.snr:self.snr+1, :] # (B, L, num_noise_heads, 1, D)

        lambda_val = self.lambda_proj(hidden_states).unsqueeze(-1).unsqueeze(-1) # (B, L, H, 1, 1)
        attn_output = attn_sig - torch.sigmoid(lambda_val) * attn_noi # (B, L, num_noise_heads, snr, D) (each noise head is broadcasted/repeated SNR times)
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1, self.head_dim) # (B, L, num_signal_heads, D)

        #if cache_params is not None:
        #    cache_params.trim(self.layer_idx)

        return attn_output, None, None

# the following torch formulations of GDN are taken from Qwen3Next
def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    q = F.pad(q, (0, 0, 0, pad_size))
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (q.shape[-1] ** 0.5) if scale is None else scale
    q = q * scale

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    # reshape to chunks
    q, k, v, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def torch_recurrent_gated_delta_rule(
    q, k, v, g, beta, initial_state, output_final_state, scale=None, use_qk_l2norm_in_kernel=False
):
    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    scale = 1 / (q.shape[-1] ** 0.5) if scale is None else scale
    q = q * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(v)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(v)
        if initial_state is None
        else initial_state.to(v)
    )

    for i in range(sequence_length):
        q_t = q[:, :, i]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def get_qkv_tensors_gdn(module: nn.Module, hidden_states: torch.Tensor):
    H, G, dk, dv = module.num_attention_heads, module.n_kv_heads, module.dk, module.dv
    mixed = module.linear_qkv(hidden_states) # (B, L, H*dk + G*dk + G*dv)

    q_end = H * dk
    k_end = q_end + G * dk
    q_proj = mixed[..., :q_end]
    k_proj = mixed[..., q_end:k_end]
    v_proj = mixed[..., k_end:]

    q = rearrange(q_proj, "b l (h d) -> b l h d", h=H)
    k = rearrange(k_proj, "b l (g d) -> b l g d", g=G)
    v = rearrange(v_proj, "b l (g d) -> b l g d", g=G)
    return q, k, v

@torch._dynamo.disable
def prepare_sequence_ids_no_compile(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_sequence_ids(cu_seqlens)

class DragonGatedDeltaNet(nn.Module):
    def __init__(self, config: DragonConfig, layer_idx: Optional[int], use_ve: bool = False, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.num_attention_heads = config.num_attention_heads_gdn
        self.n_kv_heads = config.num_key_value_heads_gdn if config.num_key_value_heads_gdn > 0 else self.num_attention_heads
        assert self.num_attention_heads % self.n_kv_heads == 0
        self.groups = self.num_attention_heads // self.n_kv_heads

        self.head_dim = config.head_dim_gdn
        self.dk = self.head_dim//config.shrink_qk_gdn
        self.dv = self.head_dim
        self.key_dim = self.n_kv_heads * self.dk
        self.value_dim = self.n_kv_heads * self.dv

        self.n_heads_local = self.num_attention_heads // 1
        self.key_dim_local = self.n_heads_local * self.dk
        self.value_dim_local = self.n_heads_local * self.dv

        self.in_proj = DragonLinear(
            config,
            config.hidden_size,
            self.num_attention_heads*self.dk + self.n_kv_heads*self.dk + 2*self.n_kv_heads*self.dv+2*self.num_attention_heads,
            bias=False,
        )

        if use_ve:
            self.ve_scalars = nn.Parameter(torch.zeros(self.num_attention_heads, self.dv, dtype=torch.float32))

        dt_min = config.time_step_min
        dt_max = config.time_step_max
        dt_init_floor = config.time_step_floor
        A_init_range = config.A_init_range
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.n_heads_local) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.n_heads_local, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        if self.config.token_conv1d_gdn:
            self.conv_size = config.conv_kernel
            self.conv_dim  = self.num_attention_heads*self.dk + self.n_kv_heads*self.dk + self.n_kv_heads*self.dv
            self.qkv_conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_size, groups=self.conv_dim, padding=self.conv_size-1)

        if self.config.token_shift_gdn:
            self.shift_proj_k = DragonLinear(config, self.config.hidden_size, self.n_kv_heads, bias=False)
            self.shift_proj_v = DragonLinear(config, self.config.hidden_size, self.n_kv_heads, bias=False)
            if self.config.scalar_proj_as_hidden_matrix:
                self.shift_proj_k.is_scalar_weight = True
                self.shift_proj_v.is_scalar_weight = True

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
                cache_params: Optional[HybridDragonDynamicCache] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                ve=None,
                **kwargs,
    ):
        _, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else 'chunk'
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        use_precomputed_states = (
            cache_params is not None
            and q_len == 1
        )

        # --- projections ---
        qkvzba = self.in_proj(hidden_states)
        qkvzba = rearrange(qkvzba, "b l (h p) -> b l h p", h=self.n_heads_local)
        # split per head: [L, B, H_local, dk+dk+dv/dv/1/1] where dq=dk=do
        qkv = qkvzba[..., :2*self.dk+self.dv]; accum = 2*self.dk+self.dv
        g_proj = qkvzba[..., accum:accum+self.dv]; accum += self.dv
        b_proj = qkvzba[..., accum:accum+1].squeeze(-1); accum += 1
        a_proj = qkvzba[..., accum:accum+1].squeeze(-1)
        #q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=-1)

        if cache_params is not None:
            ssm_cache = cache_params.ssm_caches[self.layer_idx]

        # value embeddings
        if ve is not None:
            v = v + self.ve_scalars * ve.view_as(v)

        # token-shift.
        if self.config.token_shift_gdn:
            alpha_k = torch.sigmoid(self.shift_proj_k(hidden_states).float()).float().to(k.dtype).unsqueeze(-1) # (B, L, Hkv, 1)
            alpha_v = torch.sigmoid(self.shift_proj_v(hidden_states).float()).float().to(v.dtype).unsqueeze(-1) # (B, L, Hkv//2, 1)

            if cache_params is not None:
                k_prev, v_prev = cache_params.get_last_kv(self.layer_idx)
                if k_prev is None:
                    k_prev, v_prev = torch.zeros_like(k[:, :1]), torch.zeros_like(v[:, :1])
                cache_params.set_last_kv(self.layer_idx, k[:, -1:], v[:, -1:])
            else:
                k_prev = F.pad(k, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)
                v_prev = F.pad(v, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)

            k = alpha_k * k_prev + (1 - alpha_k) * k
            v = alpha_v * v_prev + (1 - alpha_v) * v

        # conv
        if self.config.token_conv1d_gdn:
            qkv = rearrange(qkv, 'b l h d -> b l (h d)')
            mixed_qkv = qkv.transpose(1, 2)

            if cache_params is not None:
                conv_cache = cache_params.conv_caches[self.layer_idx]

            if use_precomputed_states and conv_cache is not None:
                mixed_qkv = self.causal_conv1d_update(
                    x=mixed_qkv,
                    conv_state=conv_cache,
                    weight=self.qkv_conv1d.weight.squeeze(1),
                    bias=self.qkv_conv1d.bias,
                    activation='silu',
                ) # conv_cache is updated in-place here
            else:
                if cache_params is not None:
                    conv_cache = F.pad(mixed_qkv, (self.conv_size - mixed_qkv.shape[-1], 0))
                    cache_params.conv_caches[self.layer_idx] = conv_cache
                if self.causal_conv1d_fn is not None:
                    seq_idx = None
                    if cu_seqlens is not None:
                        seq_idx = prepare_sequence_ids_no_compile(cu_seqlens).to(torch.int32).unsqueeze(0)
                    mixed_qkv = self.causal_conv1d_fn(
                        x=mixed_qkv,
                        weight=self.qkv_conv1d.weight.squeeze(1),
                        bias=self.qkv_conv1d.bias,
                        activation='silu',
                        seq_idx=seq_idx,
                    )
                else:
                    mixed_qkv = F.silu(self.qkv_conv1d(mixed_qkv)[:, :, :q_len])

            # split back
            mixed_qkv = mixed_qkv.transpose(1, 2)
            mixed_qkv = rearrange(mixed_qkv, "b l (h p) -> b l h p", h=self.n_heads_local)#.contiguous()
            q = mixed_qkv[..., :self.dk]; accum = self.dk
            k = mixed_qkv[..., accum:accum+self.dk]; accum += self.dk
            v = mixed_qkv[..., accum:accum+self.dv]

        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        beta = b_proj.sigmoid()
        dt = F.softplus(a_proj.float() + self.dt_bias)
        g = -self.A_log.float().exp() * dt

        # GDN main computation
        if not use_precomputed_states:
            o, ssm_cache = self.chunk_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                scale=None if not self.config.use_completed_p else 1/self.dk,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            ) # (B L H dv)
        else:
            o, ssm_cache = self.recurrent_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                scale=None if not self.config.use_completed_p else 1/self.dk,
                initial_state=ssm_cache,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True
            ) # (B L H dv)

        o = o * F.silu(g_proj + 1.15)

        # update GDN cache
        if cache_params is not None:
            cache_params.ssm_caches[self.layer_idx] = ssm_cache

        return o, None, None

class DragonMamba2(nn.Module):
    def __init__(self, config: DragonConfig, layer_idx: Optional[int]):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.expand = 2
        self.d_inner = self.expand * self.d_model
        self.headdim = config.mamba_headdim
        self.ngroups = config.mamba_ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.layer_idx = layer_idx

        # Order: [x, B, C, dt]
        d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = DragonLinear(config, self.d_model, d_in_proj, bias=False)

        if not self.config.mamba3_remove_conv:
            conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=False,
                kernel_size=4,
                groups=conv_dim,
                padding=4-1,
            )
            self.act = nn.SiLU()

        # Initialize log dt bias
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt_limit=(0.0, float("inf"))
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        A_init_range=(1, 16)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        if config.legacy_gate:
            self.linear_g = DragonLinear(
                config, config.hidden_size,
                self.d_inner,
                bias=False,
            )
            self.output_norm = RMSNormGated(self.d_inner, eps=config.norm_epsilon, norm_before_gate=False)

    def forward(self, hidden_states, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        _, seqlen, _ = hidden_states.shape

        zxbcdt = self.in_proj(hidden_states)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        xBC, dt = torch.split(
            zxbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 1D Convolution
        if not self.config.mamba3_remove_conv:
            if causal_conv1d_fn is None:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation="swish",
                ).transpose(1, 2)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=256,
            D=self.D,
            z=None,
            seq_idx=None,
            initial_states=None,
        )

        if self.config.legacy_gate:
            g = self.linear_g(hidden_states) # (B, L, d_inner)
            y = rearrange(y, "b l h p -> b l (h p)")
            y = self.output_norm(y, g)
            y = rearrange(y, "b l (h p) -> b l h p", h=self.nheads)

        return y, None, None

class DragonMamba3MimoFast(nn.Module):
    def __init__(self, config: DragonConfig, layer_idx: int, use_ve: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.d_model = config.hidden_size
        self.d_inner = 2*self.d_model
        self.rope_fraction = 0.5
        self.rotary_dim_divisor = 4
        self.A_floor = 1e-4
        self.mimo_dim = config.mamba_mimo_dim
        self.d_state = config.mamba_d_state
        self.headdim = config.mamba_headdim
        self.ngroups = config.mamba_ngroups
        self.mimo_proj_block_order = 1
        
        self.nheads = self.d_inner // self.headdim
        self.dr_out_dim = self.d_inner // self.mimo_proj_block_order
        self.chunk_size = 64 // self.mimo_dim

        self.split_tensor_size = int(self.d_state * self.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2

        tp_size = 1

        # Ensure that each TP rank gets at least one head:
        assert self.nheads % tp_size == 0, "nheads must be evenly divisble by tp_size"
        self.nheads_per_group = self.nheads // self.ngroups
        self.nheads_local_tp = self.nheads // tp_size
        self.n_repeat = self.nheads_local_tp // self.ngroups

        # Note that we do not need to confirm that `d_inner % tp_size == 0` because
        # `d_inner % headdim == 0`, `nheads = d_inner // headdim`, and `nheads % tp_size == 0`
        self.d_inner_per_group = self.d_inner // self.ngroups
        self.d_inner_local_tp = self.d_inner // tp_size
        self.dr_out_dim_local_tp = self.dr_out_dim // tp_size

        # Ensure that each TP rank gets at least one group:
        assert self.ngroups % tp_size == 0, "ngroups must be evenly divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # Ensure that each group has a positive integer number of heads:
        assert self.nheads % self.ngroups == 0, "nheads must be evenly divisible by ngroups"

        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = DragonLinear(
            config,
            self.d_model,
            self.d_inner * 2 + 3 * self.nheads,
            bias=False,
        )
        self.in_proj_dyn = DragonLinear(
            config,
            self.d_model,
            2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles,
            bias=False,
        )

        self.B_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        self.C_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        self.B_norm = DragonNorm(config, self.d_state)
        self.C_norm = DragonNorm(config, self.d_state)

        # Initialize up/down MIMO projection (for x and z)
        in_proj_mimo_x_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        in_proj_mimo_z_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)
        out_proj_mimo_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        self.in_proj_mimo_x = nn.Parameter(in_proj_mimo_x_init_weights, requires_grad=True)
        self.in_proj_mimo_z = nn.Parameter(in_proj_mimo_z_init_weights, requires_grad=True)
        self.out_proj_mimo = nn.Parameter(out_proj_mimo_init_weights, requires_grad=True)

        with nullcontext():
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.nheads_local_tp,
                    device=torch.cuda.current_device(),
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads_local_tp, device=torch.cuda.current_device())) # Keep in fp32
        self.D._no_weight_decay = True # useless flag

        if self.config.mamba3_postgate_norm:
            self.output_norm = DragonNorm(config, self.d_inner_local_tp)

        self.previous_window_size = 0

    def forward(self, hidden_states, cache_params: Optional[HybridDragonDynamicCache] = None, **kwargs):
        """
        hidden_states: (B L D)
        Returns: same shape as hidden_states
        """

        layer_idx = self.layer_idx
        batch, q_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and q_len == 1
        is_prefill = cache_params is not None

        # Packed-batch (varlen) support for training frameworks like verl
        # that feed one packed (1, T) row holding many documents
        # (use_remove_padding). Running that dense leaks SSM state across
        # document boundaries, and the dense backward additionally requires
        # T % chunk_size == 0, which packed batches don't satisfy. Derive
        # cu_seqlens from position_ids resets (pos == 0 marks each document
        # start — same convention as the diff-tpa shift path) and use the
        # public kernel's varlen forward/backward, which handles boundaries
        # and ragged lengths. Only the public mamba3_mimo kernel accepts
        # cu_seqlens (angle_dt is None on that path).
        _mamba3_cu_seqlens = None
        if angle_dt is None and cache_params is None and batch == 1:
            _mamba3_cu_seqlens = kwargs.get("cu_seqlens", None)
            if _mamba3_cu_seqlens is None:
                _pos = kwargs.get("position_ids", None)
                if _pos is not None:
                    _starts = (_pos[0] == 0).nonzero(as_tuple=False).flatten()
                    if _starts.numel() >= 1:
                        _mamba3_cu_seqlens = torch.cat(
                            [
                                _starts.to(torch.int32),
                                torch.tensor(
                                    [q_len],
                                    dtype=torch.int32,
                                    device=_starts.device,
                                ),
                            ]
                        )
            if _mamba3_cu_seqlens is not None:
                _mamba3_cu_seqlens = _mamba3_cu_seqlens.to(
                    device=hidden_states.device, dtype=torch.int32
                )

        # Dense-path ragged-length support: the dense mamba3 BACKWARD requires
        # L % chunk_size == 0 (the forward tolerates ragged L), so padded
        # (B, L) training batches with arbitrary max length die in
        # loss.backward(). Pad each row up to the next chunk multiple and
        # slice the output back: pads sit at the END of every row, the scan
        # is causal so they cannot influence real positions, and the no-cache
        # path never reads final states. No-op when L already divides.
        _mamba3_len = q_len
        _mamba3_pad = 0
        if (
            cache_params is None
            and _mamba3_cu_seqlens is None
            and q_len % self.chunk_size != 0
        ):
            _mamba3_pad = self.chunk_size - (q_len % self.chunk_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, _mamba3_pad))
            batch, q_len, _ = hidden_states.shape

        if use_precomputed_states:
            state = (
                cache_params.mamba3_angle_states[layer_idx],
                cache_params.mamba3_ssm_states[layer_idx],
                cache_params.mamba3_k_states[layer_idx],
                cache_params.mamba3_v_states[layer_idx],
            )

            y_t, *state = self.step(hidden_states[:, 0, :], *state)

            (
                cache_params.mamba3_angle_states[layer_idx],
                cache_params.mamba3_ssm_states[layer_idx],
                cache_params.mamba3_k_states[layer_idx],
                cache_params.mamba3_v_states[layer_idx],
            ) = state
            cache_params.past_length[layer_idx] += 1

            y = y_t.unsqueeze(1)          # (B, 1, D)
            if self.config.mamba3_postgate_norm:
                y = self.output_norm(y)
            return y, None, None

        if self.config.complete_slw and self.config.slw_wsize > 128:
            assert not is_prefill
            if self.previous_window_size != self.config.slw_wsize:
                logger.info(f"Mamba3Mimo complete_slw: updating previous_window_size from {self.previous_window_size} to {self.config.slw_wsize}")
                self.previous_window_size = self.config.slw_wsize
            batch, l, dim = hidden_states.shape
            assert batch == 1, "complete_slw only supports batch size of 1"
            hidden_states = hidden_states.reshape(-1, self.config.slw_wsize, dim).contiguous()

        # Input projection
        zxdtAtrap = self.in_proj(hidden_states)
        per_head = zxdtAtrap.view(*zxdtAtrap.shape[:-1], self.nheads_local_tp, 2*self.headdim+3)
        off = 0
        z    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        x    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        dt   = per_head[..., off];                        off += 1 # (L, B, H)
        A    = per_head[..., off];                        off += 1 # (L, B, H)
        trap = per_head[..., off];                        off += 1 # (L, B, H)
        z = rearrange(z, "b l H p -> b l H p") # (B, L, H_local, p)
        x = rearrange(x, "b l H p -> b l H p") # (B, L, H_local, p)
        dt   = rearrange(dt,   "b l n -> b l n").to(torch.float32) # (B, L, H_local)
        A    = rearrange(A,    "b l n -> b l n") # (B, L, H_local)
        trap = rearrange(trap, "b l n -> b n l") # (B, H_local, L)

        BCangle = self.in_proj_dyn(hidden_states)
        B = BCangle[..., 0:self.ngroups*self.mimo_dim*self.d_state]
        C = BCangle[..., self.ngroups*self.mimo_dim*self.d_state:2*self.ngroups*self.mimo_dim*self.d_state]
        angle = BCangle[..., 2*self.ngroups*self.mimo_dim*self.d_state:] # (L, B, S)
        B = rearrange(B, "b l (G r n) -> b l r G n", G=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "b l (G r n) -> b l r G n", G=self.ngroups, r=self.mimo_dim)

        _A = -F.softplus(A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dt + self.dt_bias) # (B, L, N)
        ADT = _A * dt

        B = self.B_norm(B)
        C = self.C_norm(C)

        angle = angle.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1) # (B, L, G, S)
        if angle_dt is not None:
            # private kernel wants pre-cumsummed angles; the public mamba_ssm
            # mamba3_mimo fallback (angle_dt is None) applies the cumsum itself
            angle = angle_dt(angle, dt)

        ADT = rearrange(ADT, "b l n -> b n l")
        dt = rearrange(dt, "b l n -> b n l")

        y, kernel_state = mamba3_tilelang(
            # No forced .bfloat16(): keep the incoming compute dtype so fp32
            # (or mixed-precision) training isn't silently downcast here.
            Q=C.contiguous(),
            K=B.contiguous(),
            V=x.contiguous(),
            ADT=ADT.to(torch.float32).contiguous(),
            DT=dt.to(torch.float32).contiguous(),
            Trap=trap.contiguous(),
            Q_bias=self.C_bias.to(torch.float32),
            K_bias=self.B_bias.to(torch.float32),
            MIMO_V=self.in_proj_mimo_x.to(torch.float32),
            MIMO_Z=self.in_proj_mimo_z.to(torch.float32),
            MIMO_Out=self.out_proj_mimo.to(torch.float32),
            Angles=angle.to(torch.float32).contiguous(),
            D=self.D.to(torch.float32).contiguous(),
            Z=z.contiguous(),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            return_state=is_prefill,
            **(
                {"cu_seqlens": _mamba3_cu_seqlens}
                if _mamba3_cu_seqlens is not None
                else {}
            ),
        )

        if is_prefill:
            angle_state_out, ssm_state_out, k_state_out, v_state_out = kernel_state
            cache_params.mamba3_angle_states[layer_idx] = angle_state_out
            cache_params.mamba3_ssm_states[layer_idx]   = ssm_state_out
            cache_params.mamba3_k_states[layer_idx]     = k_state_out
            cache_params.mamba3_v_states[layer_idx]     = v_state_out
            cache_params.past_length[layer_idx]         += q_len

        y = rearrange(y, "b l h p -> b l (h p)")
        if self.config.mamba3_postgate_norm:
            y = self.output_norm(y)
        #y = rearrange(y, "b l (h p) -> b l h p", h=self.nheads_local_tp)

        if self.config.complete_slw and self.config.slw_wsize > 128:
            y = y.reshape(batch, l, -1).contiguous()

        return y, None, None
    
    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -F.softplus(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        B = rearrange(B, "b (r g s) -> b r g s", g=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.ngroups, r=self.mimo_dim)
        B = self.B_norm(B)
        C = self.C_norm(C)
        B = B.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
        C = C.expand(-1, -1, self.nheads, -1) # (B, R, N, S)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles

    def step(self, u, angle_state, ssm_state, k_state, v_state):
        """
        Decode function using CuteDSL kernel from mamba3_step_fn.py.

        NOTE: Only tested on H100. Compatibility with other hardware
        will be made available in the future.

        Args:
            u: (batch, d_model)
            angle_state: (batch, nheads, num_rope_angles)
            ssm_state: (batch, nheads, headdim, d_state)
            k_state: (batch, R, nheads, d_state), where R = mimo_rank
            v_state: (batch, nheads, headdim)
        Returns:
            out: (batch, d_model)
            nxt_angle_state: (batch, nheads, num_rope_angles)
            state_out: (batch, nheads, headdim, d_state)
            nxt_k_state: (batch, R, nheads, d_state), where R = mimo_rank
            nxt_v_state: (batch, nheads, headdim)
        """

        zxdtAtrap = self.in_proj(u)
        per_head = zxdtAtrap.view(*zxdtAtrap.shape[:-1], self.nheads_local_tp, 2*self.headdim+3)
        off = 0
        z    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        x    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        dt   = per_head[..., off];                        off += 1 # (L, B, H)
        A    = per_head[..., off];                        off += 1 # (L, B, H)
        trap = per_head[..., off];                        off += 1 # (L, B, H)
        z = z.flatten(-2) # TODO: better efficient way to do it ? (considering the op we do just after)
        x = x.flatten(-2)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        BCangle = self.in_proj_dyn(u)
        B = BCangle[..., 0:self.ngroups*self.mimo_dim*self.d_state]
        C = BCangle[..., self.ngroups*self.mimo_dim*self.d_state:2*self.ngroups*self.mimo_dim*self.d_state]
        angles = BCangle[..., 2*self.ngroups*self.mimo_dim*self.d_state:] # (L, B, S)

        DT, B, C, x, z, trap, A, angles = self._preprocess(A, dt, B, C, x, z, trap, angles)

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")

        C, B, nxt_angle_state = apply_rotary_qk_inference_fwd(
            q=C,
            k=B,
            angle_state=angle_state, 
            angle_proj=angles,
            dt=DT,
            bias_q=bias_q,
            bias_k=bias_k, 
            conjugate=False,
            inplace=False,
            rotate_pairwise=False
        )

        nxt_v_state = x
        nxt_k_state = B

        xpj = rearrange(self.in_proj_mimo_x, "h r p -> r h p", p=self.headdim).contiguous()
        zpj = rearrange(self.in_proj_mimo_z, "h r p -> r h p", p=self.headdim).contiguous()
        outpj = rearrange(self.out_proj_mimo, "h r p -> r h p", p=self.headdim).contiguous()

        state_out = torch.empty_like(ssm_state)
        y = torch.empty_like(x)
        mamba3_step_fn(
            ssm_state.to(torch.float32),
            k_state.to(torch.bfloat16),
            v_state.to(torch.bfloat16),
            A,
            B.to(torch.bfloat16),
            C.to(torch.bfloat16),
            self.D,
            x,
            DT,
            trap,
            xpj,
            outpj,
            state_out,
            y,
            z=z,
            zproj=zpj,
            tile_D=64,
            num_warps=4,
        )

        y = rearrange(y, "b h p -> b (h p)")

        return y, nxt_angle_state, state_out, nxt_k_state, nxt_v_state

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        # RoPE State
        angle_dt_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device,
            dtype=torch.float32,
        )

        # SSM State
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=torch.float32,
        )

        # K (=B) State
        k_state = torch.zeros(
            (batch_size, self.mimo_dim, self.nheads, self.d_state),
            device=device,
            dtype=dtype,
        )

        # V (=x) State
        v_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )

        return (angle_dt_state, ssm_state, k_state, v_state)

class DragonMLP(nn.Module):
    def __init__(self, config: DragonConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        intermediate_size = intermediate_size or config.intermediate_size
        self.fc_1 = DragonLinear(config, config.hidden_size, intermediate_size, bias=False)
        self.fc_2 = DragonLinear(config, intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.fc_1(hidden_states)
        hidden_states = F.relu(hidden_states).square()
        hidden_states = self.fc_2(hidden_states)
        return hidden_states

class DragonFANMLP(nn.Module):
    """
    FAN-style MLP. Splits the intermediate dimension into:
      - periodic_dim: processed with cos/sin (2x because cos+sin)
      - standard_dim: processed with relu²
    
    periodic_ratio controls the split (default 0.2 = 20% periodic, 80% standard).
    
    Parameter count is similar to original DragonMLP when periodic_ratio=0.5
    because cos/sin don't need separate weights for real/imaginary parts -
    they share W_p and just apply cos and sin respectively.
    """
 
    def __init__(self, config, intermediate_size: Optional[int] = None, periodic_ratio: float = 0.2):
        super().__init__()
        self.config = config
        intermediate_size = intermediate_size or config.intermediate_size
 
        # Split: periodic part uses half the intermediate dims (but produces 2x via cos+sin)
        # Standard part uses the other half
        self.periodic_dim = int(intermediate_size * periodic_ratio) // 2  # W_p output dim
        self.standard_dim = intermediate_size - (self.periodic_dim * 2)   # relu² part
 
        # Periodic branch: x → W_p·x → [cos(...), sin(...)]  (periodic_dim → 2*periodic_dim)
        self.fc_periodic = nn.Linear(config.hidden_size, self.periodic_dim, bias=False)
 
        # Standard branch: x → W_s·x → relu²(...)
        if self.standard_dim > 0:
            self.fc_standard = nn.Linear(config.hidden_size, self.standard_dim, bias=False)
 
        # Down projection: [cos ‖ sin ‖ relu²] → hidden_size
        total_intermediate = self.periodic_dim * 2 + self.standard_dim
        self.fc_down = nn.Linear(total_intermediate, config.hidden_size, bias=False)
 
    def forward(self, hidden_states):
        # Periodic branch
        p = self.fc_periodic(hidden_states)
        periodic_out = torch.cat([torch.cos(p), torch.sin(p)], dim=-1)
 
        # Standard branch
        if self.standard_dim > 0:
            s = self.fc_standard(hidden_states)
            standard_out = F.relu(s).square()
            combined = torch.cat([periodic_out, standard_out], dim=-1)
        else:
            combined = periodic_out
 
        # Down projection
        return self.fc_down(combined)

class DragonMoE(nn.Module):
    def __init__(self, config: DragonConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.moe_gate = DragonLinear(config, config.hidden_size, config.moe_num_routed_experts, bias=False, cosnet=False)
        if self.config.moe_routed_input_dim:
            self.down_proj = DragonLinear(config, config.hidden_size, config.moe_routed_input_dim, bias=False)
            self.up_proj = DragonLinear(config, config.moe_routed_input_dim, config.hidden_size, bias=False)
        self.experts = ScatterMoE(
            input_size=config.moe_routed_input_dim or config.hidden_size,
            hidden_size=config.moe_routed_intermediate_size,
            num_experts=config.moe_num_routed_experts,
            top_k=config.moe_num_active_experts,
            activation=lambda x: F.relu(x).square(),
        )
        if self.config.moe_shared_expert_gate:
            self.shared_gate = DragonLinear(config, config.hidden_size, 1, bias=False)
        self.shared_experts = (
            DragonMLP(config, config.moe_shared_intermediate_size)
            if config.moe_shared_intermediate_size and config.moe_shared_intermediate_size > 0
            else None
        )

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.moe_num_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.moe_num_routed_experts, dtype=torch.float32),
            persistent=True,
        )

        with torch.no_grad():
            self.experts.experts.weight.normal_(mean=0.0, std=self.config.initializer_range)
            self.experts.output_experts.weight.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        input_dtype = x.dtype
        x = x.view(-1, dim)

        # router.
        logits = torch.matmul(x.float(), self.moe_gate.weight.float().t())

        scores = torch.sigmoid(logits.float()).type_as(logits)
        scores_for_routing = scores + self.expert_bias.float()
        #scores_orig = scores_for_routing.clone()
        _, top_indices = torch.topk(scores_for_routing, k=self.config.moe_num_active_experts, dim=1)
        scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.config.moe_num_active_experts > 1 else scores
        probs = probs * self.config.moe_routed_scaling_factor
        probs = probs.to(input_dtype)

        if self.training:
            with torch.no_grad():
                idx = top_indices.reshape(-1) # (N*K,)
                tpe = torch.bincount(idx, minlength=self.config.moe_num_routed_experts).to(self.tokens_per_expert.dtype).to(x.device)
                self.tokens_per_expert.add_(tpe)

        # experts.
        x0 = x
        if self.config.moe_routed_input_dim:
            x0 = self.down_proj(x).to(x.dtype)
        out_experts = self.experts(x0, probs, top_indices)
        if self.config.moe_routed_input_dim:
            out_experts = self.up_proj(out_experts).to(out_experts.dtype)
        # shared experts.
        out = None
        if self.shared_experts is not None:
            out = self.shared_experts(x)
            if self.config.moe_shared_expert_gate:
                logits = self.shared_gate(x)
                scores = torch.sigmoid(logits.float()).type_as(logits)
                out = out * scores
        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)#, top_indices, scores_orig

class DragonGeodesicNorm(nn.Module):
    def __init__(self, config: DragonConfig, layer_idx: int):
        super().__init__()

        # 1-D ([1]) rather than 0-dim scalars: FSDP1/FSDP2 both reject 0-dim
        # parameters outright ("fully_shard doesn't support scalar
        # parameters") — sharding needs a dim 0 to shard/flatten along.
        # Numerically identical: the only use is ``theta * scale + bias``
        # where theta is (..., 1), so [1] broadcasts the same.
        # NOTE: checkpoints exported before this change store these as 0-dim
        # and need scripts/olala/patch_olala_checkpoint.py (from_pretrained
        # compares shapes before copying); re-exported checkpoints load as is.
        self.scale = nn.Parameter(torch.tensor([1.]))
        self.bias = nn.Parameter(torch.tensor([0.]))
        self.register_buffer("prosres_scalar", torch.tensor(1.0))
        self.clamp = torch.pi/4
        self.layer_idx = layer_idx

    def forward(self, x, g):
        """
        x: residual;
        g: ffn(x) or attn(x);
        """

        x_norm_sq = x.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proj_coeff = (x * g).sum(dim=-1, keepdim=True) / x_norm_sq
        gradient = g - proj_coeff * x
        tangent_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True)
        safe_tangent_norm = torch.clamp(tangent_norm, min=1e-8)
        unit_tangent = gradient / safe_tangent_norm
        R = torch.norm(x, p=2, dim=-1, keepdim=True)
        safe_R = torch.clamp(R, min=1e-6)
        theta = torch.clamp(safe_tangent_norm / safe_R, max=self.clamp)
        theta = torch.clamp((theta * self.scale + self.bias) / (self.layer_idx + 1), max=self.clamp)
        output = x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)
        # Dtype-stability: `theta * self.scale + self.bias` promotes theta —
        # and through cos/sin the OUTPUT — to the params' dtype, which can
        # differ from the residual dtype under mixed precision. Pin the
        # output to the residual dtype; no-op when dtypes already agree.
        return output.to(x.dtype)

class DragonMonoBlock(GradientCheckpointingLayer):
    def __init__(self, config: DragonConfig, layer_idx: int, layer_type: str, use_ve: bool = False, mlp_type: str = 'd'):
        super().__init__()
        self.config = config
        # When blocks have been pruned out (config.layer_idx_map present),
        # the surviving block at *position* `layer_idx` originally had a
        # different index. We preserve that original index so that
        # DragonGeodesicNorm.theta scaling by 1/(layer_idx+1) and any other
        # layer-index-dependent logic stays consistent with how the model
        # was trained. Pruning configs (skip_moe_layers, skip_mixer_layers)
        # are also expressed in ORIGINAL indices.
        _map = getattr(config, "layer_idx_map", None)
        if _map is not None and layer_idx < len(_map):
            layer_idx = int(_map[layer_idx])
        self.layer_idx = layer_idx

        if use_ve:
            assert layer_type in ['g', 'T', 'M'], "VE is only supported for 'g', 'T' and 'M' layer types."

        if layer_type == 'g':
            self.mixer = DragonGatedDeltaNet(config, layer_idx=layer_idx, use_ve=use_ve)
            head_dim = self.mixer.head_dim
            num_attention_heads = self.mixer.num_attention_heads
            use_gate = False
        elif layer_type == 'v':
            self.mixer = DragonDifferentialAttentionV2(config, layer_idx=layer_idx)
            head_dim = self.mixer.head_dim
            num_attention_heads = self.mixer.num_signal_heads
            use_gate = config.gate_attn
        elif layer_type == 'w':
            self.mixer = DragonAttention(config, reuse_kv=False, layer_idx=layer_idx)
            head_dim = self.mixer.head_dim
            num_attention_heads = self.mixer.num_attention_heads
            use_gate = config.gate_attn
        elif layer_type == 't':
            self.mixer = DragonTensorProductAttention(config, reuse_kv=False, layer_idx=layer_idx)
            head_dim = self.mixer.head_dim
            num_attention_heads = self.mixer.num_attention_heads
            use_gate = config.gate_attn
        elif layer_type == 'V':
            self.mixer = DragonDifferentialTensorProductAttentionV2(config, layer_idx=layer_idx, use_ve=use_ve)
            head_dim = self.mixer.head_dim
            num_attention_heads = self.mixer.num_signal_heads
            use_gate = config.gate_attn
        elif layer_type == '2':
            self.mixer = DragonMamba2(config, layer_idx=layer_idx)
            head_dim = self.mixer.headdim
            num_attention_heads = self.mixer.nheads
            use_gate = config.gate_gdn
        elif layer_type == 'M':
            self.mixer = DragonMamba3MimoFast(config, layer_idx=layer_idx, use_ve=use_ve)
            head_dim = self.mixer.headdim
            num_attention_heads = self.mixer.nheads
            use_gate = False # inside Mamba3MimoFast
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        if use_gate:
            if self.config.gate_type == "elementwise":
                self.gate_proj = DragonLinear(self.config, config.hidden_size, num_attention_heads*head_dim, bias=False)
            elif self.config.gate_type == "kimi":
                self.gate_proj = nn.Sequential(
                    DragonLinear(config, config.hidden_size, head_dim, bias=False),
                    DragonLinear(config, head_dim, num_attention_heads*head_dim, bias=True),
                )
            elif self.config.gate_type == "headwise":
                self.gate_proj = DragonLinear(self.config, config.hidden_size, num_attention_heads, bias=False)
                if self.config.scalar_proj_as_hidden_matrix:
                    self.gate_proj.is_scalar_weight = True
            else:
                raise ValueError(f"Unknown gate_type: {self.config.gate_type}")
            val = 0.
            if self.config.zero_centered_gate:
                val = 1.15
            self.gate_bias = val
            if self.config.gate_act == "silu":
                self.gate_act = F.silu
            elif self.config.gate_act == "sigmoid":
                self.gate_act = F.sigmoid
            else:
                raise ValueError(f"Unknown gate_act: {self.config.gate_act}")
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.use_gate = use_gate

        self.mixer_proj = DragonLinear(config, head_dim*num_attention_heads, config.hidden_size, bias=False)
        if config.mixer_gn:
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=num_attention_heads, d_head=head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        if not config.geodesic_update:
            self.input_norm = DragonNorm(config, config.hidden_size)
            self.postmixer_norm = DragonNorm(config, config.hidden_size)
        else:
            self.input_norm = torch.nn.Identity()
            self.postmixer_norm = torch.nn.Identity()
            self.geodesic_mixer = DragonGeodesicNorm(config, self.layer_idx)
            self.geodesic_mlp = DragonGeodesicNorm(config, self.layer_idx)

        # ---- pruning / skip-MoE support ----
        # If `config.skip_moe_layers` contains this layer_idx, replace mlp with a
        # parameter-free no-op and also disable the geodesic_mlp combine so the
        # block becomes "mixer-only" at this layer. Saved checkpoints with these
        # layers don't carry MoE params for them at all.
        skip_moe = getattr(config, "skip_moe_layers", None) or []
        if layer_idx in skip_moe:
            self.mlp = _NullMlp()
            if config.geodesic_update:
                self.geodesic_mlp = _NullGeodesic()
        elif not config.moe or mlp_type == 'd':
            if config.mlp_type == "simple":
                self.mlp = DragonMLP(config)
            elif config.mlp_type == "gated":
                self.mlp = GatedMlp(in_features=config.hidden_size, hidden_features=config.intermediate_size, out_features=config.hidden_size, activation=F.silu, bias1=False, bias2=False)
            elif config.mlp_type == "fan":
                self.mlp = DragonFANMLP(config, periodic_ratio=config.fan_periodic_ratio)
        elif mlp_type == 'm':
            self.mlp = DragonMoE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")

        # Skip-mixer support: if this layer is in config.skip_mixer_layers,
        # replace the mixer-side submodules with parameter-free placeholders.
        # The forward also short-circuits the mixer phase via _skip_mixer.
        skip_mixer = getattr(config, "skip_mixer_layers", None) or []
        if layer_idx in skip_mixer:
            self.mixer = _NullMixer()
            self.mixer_proj = _NullMlp()                # returns zeros_like(input)
            self.input_norm = nn.Identity()
            if config.geodesic_update:
                self.geodesic_mixer = _NullGeodesic()
            if self.use_gate:
                self.gate_proj = _NullMlp()
            if config.mixer_gn:
                self.mixer_group_norm = nn.Identity()

        lns = 1.
        if config.layer_norm_scaling:
            lns = 1. / math.sqrt(layer_idx + 1)
        self.lns = float(lns)

        a, b = 1., 1.
        if self.config.use_completed_p:
            a = (len(self.config.layers_config)/self.config.base_depth) ** (-self.config.completed_p_alpha)
        self.a = float(a)
        self.b = float(b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        ve=None,
        **kwargs,
    ):
        # Phase-stable compute dtype under FSDP2 mixed precision + HF
        # gradient checkpointing. FSDP2's cast_forward_inputs runs inside
        # this (checkpointed) __call__ during the original forward, casting
        # the incoming fp32 residual to bf16 — but the checkpoint saves the
        # PRE-CAST input and the hook does not re-run at the same point in
        # the recompute, so every derived activation flips dtype and trips
        # torch.utils.checkpoint's check_recomputed_tensors_match. Re-apply
        # the autocast dtype explicitly: autocast state is replayed during
        # recompute, so this is identical in both phases and a no-op when
        # the input is already cast.
        if (
            torch.is_autocast_enabled()
            and hidden_states.is_floating_point()
            and hidden_states.dtype != torch.get_autocast_dtype("cuda")
        ):
            hidden_states = hidden_states.to(torch.get_autocast_dtype("cuda"))
        # Skip-mixer support: if this layer is in config.skip_mixer_layers,
        # the mixer phase is bypassed entirely — residual passes through and
        # no KV is produced (downstream layers receive None from this slot).
        _skip_mixer = self.layer_idx in (getattr(self.config, "skip_mixer_layers", None) or [])
        if _skip_mixer:
            last_key_states, last_value_states = None, None
        else:
            # MIXER.
            residual = hidden_states
            x_in = hidden_states
            hidden_states = self.lns * self.input_norm(x_in) # (B, L, D)
            y_mixer, last_key_states, last_value_states = self.mixer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache_params=cache_params,
                key_value_last_layer=key_value_last_layer,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                ve=ve,
            ) # (B, L, E*D)
            if self.config.mixer_gn and not self.config.gate_before_norm:
                y_mixer = self.mixer_group_norm(y_mixer)
            if self.use_gate:
                if self.config.gate_type == "elementwise" or self.config.gate_type == "kimi":
                    g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.head_dim).to(y_mixer.dtype)
                elif self.config.gate_type == "headwise":
                    g_proj = self.gate_proj(hidden_states).unsqueeze(-1).to(y_mixer.dtype)
                else:
                    raise ValueError(f"Unknown gate_type: {self.config.gate_type}")
                y_mixer = y_mixer * self.gate_act(g_proj + self.gate_bias)
            if self.config.mixer_gn and self.config.gate_before_norm:
                y_mixer = self.mixer_group_norm(y_mixer)
            y_mixer = y_mixer.view(y_mixer.size(0), y_mixer.size(1), -1)
            y_mixer = self.mixer_proj(y_mixer)

            if self.config.geodesic_update:
                hidden_states = self.geodesic_mixer(residual, y_mixer)
            else:
                hidden_states = self.b * residual + self.a * y_mixer

        # MLP.
        residual = hidden_states
        x_in = residual
        hidden_states = self.lns * self.postmixer_norm(x_in)
        y_mlp = self.mlp(hidden_states) # (B, L, D)

        if self.config.geodesic_update:
            hidden_states = self.geodesic_mlp(residual, y_mlp)
        else:
            hidden_states = self.b * residual + self.a * y_mlp

        return hidden_states, last_key_states, last_value_states

class DragonPreTrainedModel(PreTrainedModel):
    config: DragonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DragonMonoBlock"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DragonMonoBlock,
        "attentions": DragonMonoBlock,
    }

@dataclass
class DragonOutput(ModelOutput):
    """
    Class for the Dragon model outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`HybridDragonDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
            Includes both the RNN-like state matrices after the selective scan, and the conv states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridDragonDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class DragonCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`HybridDragonDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    geo_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridDragonDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class DragonModel(DragonPreTrainedModel):
    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.config: DragonConfig = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        if config.use_value_embedding:
            layers_ve_flags = [c == "1" for c in config.layers_ve_config]
            assert len(layers_ve_flags) == len(config.layers_config)
            self.value_embedding = nn.ModuleList()
            self.value_embedding_map = []
            for use_ve, layer_type in zip(layers_ve_flags, config.layers_config):
                if not use_ve:
                    self.value_embedding_map.append(-1)
                    continue
                if layer_type == 'T':
                    out_dim = (config.num_attention_heads - config.num_signal_heads_diff) * config.head_dim
                elif layer_type == 'g':
                    out_dim = config.num_attention_heads_gdn * config.head_dim_gdn
                elif layer_type == 'M':
                    out_dim = 2 * config.hidden_size # d_inner
                else:
                    raise ValueError(f"Value embedding is only supported for 'T' and 'g' layers, got {layer_type}")
                self.value_embedding_map.append(len(self.value_embedding))
                self.value_embedding.append(nn.Embedding(config.vocab_size, out_dim, self.padding_idx))

        layers_mlp_config = config.layers_mlp_config
        if self.config.layers_mlp_config == '':
            if self.config.moe:
                layers_mlp_config = 'm' * len(config.layers_config)
            else:
                layers_mlp_config = 'd' * len(config.layers_config)
        assert len(layers_mlp_config) == len(config.layers_config)

        if not self.config.use_value_embedding:
            self.layers = nn.ModuleList([DragonMonoBlock(config, layer_idx=i, layer_type=layer, mlp_type=mlp_type, ) if layer in ['l', 'r', 'd'] else DragonMonoBlock(config, layer_idx=i, layer_type=layer, mlp_type=mlp_type) for i, (layer, mlp_type) in enumerate(zip(config.layers_config, layers_mlp_config))])
        else:
            assert len(config.layers_ve_config) == len(config.layers_config)
            self.layers = nn.ModuleList([DragonMonoBlock(config, layer_idx=i, layer_type=layer, mlp_type=mlp_type) if layer in ['l', 'r', 'd'] else DragonMonoBlock(config, layer_idx=i, layer_type=layer, use_ve=int(ve), mlp_type=mlp_type) for i, (layer, ve, mlp_type) in enumerate(zip(config.layers_config, config.layers_ve_config, layers_mlp_config))])

        self.rotary_emb = None
        if self.config.rope_type != '' and self.config.rope_theta > 0.:
            self.rotary_emb = DragonRotaryEmbedding(config, head_dim=config.head_dim, theta=config.rope_theta)

        if self.config.final_norm:
            self.final_norm = DragonNorm(config, config.hidden_size)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[HybridDragonDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs
    ) -> DragonOutput:
        B, L = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.config.normalize_embeddings:
            inputs_embeds = F.normalize(inputs_embeds, dim=-1) * math.sqrt(self.config.hidden_size)

        if self.config.normalize_embeddings_ngpt:
            inputs_embeds = F.normalize(inputs_embeds, dim=-1)

        if self.config.patch_level_training:
            # (B, KL, D) => (B, L, D) OR (B, L, D) ==> (B, L//K, D)
            inputs_embeds = inputs_embeds.reshape(B, L//self.config.patch_level_training_size, self.config.patch_level_training_size, inputs_embeds.size(2)).mean(dim=2)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache:
            if past_key_values is None:
                past_key_values = HybridDragonDynamicCache(self.config)
            elif not isinstance(past_key_values, HybridDragonDynamicCache):
                if type(past_key_values) is DynamicCache:
                    del past_key_values
                    past_key_values = HybridDragonDynamicCache(self.config)
                else:
                    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

            if self.config.patch_level_training:
                position_ids = position_ids[:, 0:L//self.config.patch_level_training_size]

        all_hidden_states = () if output_hidden_states else None

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        shared_kv = (None, None)
        for i, block in enumerate(self.layers):
            ve_i = None
            if self.config.use_value_embedding:
                j = self.value_embedding_map[i]
                if j != -1:
                    ve_i = self.value_embedding[j](input_ids)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, last_k, last_v = block(
                hidden_states,
                position_ids=position_ids,
                cache_params=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                key_value_last_layer=shared_kv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                ve=ve_i,
                **kwargs,
            )
            shared_kv = (last_k, last_v)

        if self.config.final_norm:
            hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return DragonOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )
DragonModel.register_for_auto_class("AutoModel")

class DragonForCausalLM(DragonPreTrainedModel, GenerationMixin):
    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.config = config
        self.model = DragonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DragonLinear(config, config.hidden_size, config.vocab_size, bias=False)
        if config.normalize_lm_head:
            self.temperature = nn.Parameter(torch.tensor(math.log(math.sqrt(config.hidden_size))))
        if config.logits_scaling_ngpt:
            self.temperature_ngpt = nn.Parameter(torch.ones(config.vocab_size))
        self.post_init()
        if config.tie_lm_head:
            self.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[HybridDragonDynamicCache] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        just_loss: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        token_type_ids=None,
        **kwargs,
    ) -> DragonCausalLMOutput:
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        outputs: DragonOutput = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = None
        loss = None
        geo_loss = None
        ce_loss = None
        cosine_sims = None
        if labels is not None:
            labels = labels.to(hidden_states.device)

            # --- Step 1: compute logits (3 paths) ---
            if self.config.normalize_lm_head:
                h = hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]
                w_norm = F.normalize(self.lm_head.weight, dim=-1)
                x_norm = F.normalize(h, dim=-1)
                cosine_sims = x_norm @ w_norm.T  # (B, T, V)
                #print(f"temperature: {self.temperature.exp().item():.4f}, log_temp: {self.temperature.item():.4f}")
                logits = (self.temperature.clamp(min=-2.0, max=5.0).exp() * cosine_sims).float()
            elif linear_cross_entropy is None or not self.config.fused_loss_computation:
                logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()
                if self.config.logits_scaling_ngpt:
                    logits = logits * self.temperature_ngpt

            # --- Step 2: CE loss ---
            if logits is not None:
                if not self.config.patch_level_training:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=self.model.padding_idx)
                else:
                    shift_logits = logits[..., :-1, :].reshape(-1, self.config.vocab_size)
                    shift_labels = labels[..., self.config.patch_level_training_size:].reshape(-1, self.config.patch_level_training_size)
                    loss = 0
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    for i in range(self.config.patch_level_training_size):
                        loss = loss + F.nll_loss(log_probs, shift_labels[:, i])
                    loss = loss / self.config.patch_level_training_size
            else:
                assert not self.config.patch_level_training, "Fused loss computation is not supported with patch-level training."
                loss = linear_cross_entropy(
                    hidden_states[:, slice_indices, :].view(-1, hidden_states.size(-1)),
                    self.lm_head.weight,
                    labels.view(-1),
                    impl="cce_exact",
                    shift=1,
                )

            # --- Step 3: geodesic loss ---
            if self.config.geo_loss_coeff > 0:
                shift_targets = labels[..., 1:].contiguous()
                mask = shift_targets != self.model.padding_idx
                if cosine_sims is not None:
                    # reuse cosine similarities from normalized path (no temperature)
                    flat_cos = cosine_sims[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                    idx = torch.arange(shift_targets.numel(), device=shift_targets.device)
                    cos_sim_target = flat_cos[idx, shift_targets.view(-1)].view_as(shift_targets)
                else:
                    shift_hidden = hidden_states[:, :-1, :].contiguous()
                    target_embeds = self.model.embedding.weight[shift_targets]
                    dot = (shift_hidden * target_embeds).sum(dim=-1)
                    cos_sim_target = dot / (shift_hidden.norm(dim=-1).clamp(min=1e-8) * target_embeds.norm(dim=-1).clamp(min=1e-8))
                geo_loss = (1 - cos_sim_target).masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)
                ce_loss = loss.detach()
                loss = loss + self.config.geo_loss_coeff * geo_loss
        else:
            if self.config.normalize_lm_head:
                h = hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]
                w_norm = F.normalize(self.lm_head.weight, dim=-1)
                x_norm = F.normalize(h, dim=-1)
                logits = (self.temperature.exp() * (x_norm @ w_norm.T)).float()
            else:
                logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()

        return DragonCausalLMOutput(
            loss=loss,
            geo_loss=geo_loss,
            ce_loss=ce_loss,
            logits=logits if not just_loss else None,
            past_key_values=outputs.past_key_values if not just_loss else None,
            hidden_states=outputs.hidden_states if not just_loss else None,
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

DragonForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["DragonModel", "DragonForCausalLM", "DragonPreTrainedModel"]

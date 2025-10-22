# coding=utf-8
"""PyTorch Dragon model."""

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import inspect

import math
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging

from fast_hadamard_transform import hadamard_transform
from flash_dmattn.flash_dmattn_interface import flash_dmattn_func
from flash_dmattn.integrations.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward

from .configuration_dragon import DragonConfig

from .nsa_utils import nsa_func

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
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    if not _flash_supports_window_size:
        raise ImportError("flash_attn_func does not support window_size parameter. Please update to more recent flash_attn version")
    ATTN_IMPL = "fa3"
except ImportError:
    try:
        from flash_attn import flash_attn_func # FA2
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

# differential attention backend selection
DIFF_ATTN_IMPL = None
try:
    import flex_head_fa
    DIFF_ATTN_IMPL = "flex_head"
except ImportError:
    DIFF_ATTN_IMPL = ATTN_IMPL # if we don't have flex_head_fa, fallback to the best attention impl we have

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
print(f"Using differential attention implementation: {DIFF_ATTN_IMPL}")

logger.info(f"Using attention implementation: {ATTN_IMPL}")
logger.info(f"Using differential attention implementation: {DIFF_ATTN_IMPL}")
logger.info(f"Using Gated DeltaNet implementation: {'fla' if chunk_gated_delta_rule is not None else 'torch'}")
logger.info(f"Using short convolution implementation: {'causal-conv1d' if causal_conv1d_fn is not None else 'torch'}")

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

class DragonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, zero_centered_gamma=False):
        super().__init__()
        self.rms = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(hidden_size)) if zero_centered_gamma else nn.Parameter(torch.ones(hidden_size))
        self.zero_centered_gamma = zero_centered_gamma

    def forward(self, hidden_states):
        y = self.rms(hidden_states) * (1.0 + self.weight) if self.zero_centered_gamma else self.rms(hidden_states) * self.weight
        return y

class DragonLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6): # TODO: ZCG ?
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.hidden_size,), self.weight, self.bias, self.eps).type_as(x)

class ScaledGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_scale, backward_scale):
        ctx.backward_scale = backward_scale
        return x * forward_scale
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.backward_scale, None, None

class DragonLinear(nn.Linear):
    """Linear layer with different forward/backward scalings."""
    def __init__(self, config: DragonConfig, in_features, out_features, bias=False, alpha_fwd=None, alpha_bwd=None):
        super().__init__(in_features, out_features, bias)

        if alpha_fwd is None:
            alpha_fwd = 1.0 / math.sqrt(in_features)

        if not config.use_uscaling:
            alpha_fwd, alpha_bwd = 1, 1

        self.register_buffer("alpha_fwd", torch.tensor(float(alpha_fwd)), persistent=False)
        self.register_buffer("alpha_bwd", torch.tensor(float(alpha_bwd if alpha_bwd is not None else alpha_fwd)), persistent=False)

    def forward(self, x):
        out = super().forward(x)
        return ScaledGrad.apply(out, self.alpha_fwd, self.alpha_bwd)

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
        # attention
        self._key_cache = {}
        self._value_cache = {}
        # cca
        self.cca_qk0_cache = []
        self.cca_qk1_cache = []
        self.cca_prev_hidden = []
        # gdn
        self.conv_caches = []
        self.ssm_caches = []

        for idx, layer_type in enumerate(config.layers_config):
            if not layer_type == "r":
                self._key_cache[idx] = None
                self._value_cache[idx] = None

            self.cca_qk0_cache.append(None)
            self.cca_qk1_cache.append(None)
            self.cca_prev_hidden.append(None)
            self.conv_caches.append(None)
            self.ssm_caches.append(None)

        self.window_size = config.sliding_window_size
        self.layers_config = config.layers_config
        self.past_length = [0 for _ in range(len(config.layers_config))]

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
    def __init__(self, config: DragonConfig, head_dim: int):
        super().__init__()
        self.config = config

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
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
        q_dim = (module.num_heads // module.num_key_value_heads) * module.head_dim
        new_shape = mixed_qkv.size()[:-1] + (module.num_key_value_heads, q_dim)
        query = mixed_qkv.view(*new_shape)
        # final shape (B, L, H, d)
        query = query.reshape(query.size(0), query.size(1), -1, module.head_dim)

        return query

    # (B, L, hp) -> (B, L, ng, (np/ng + 2) * hn)
    new_tensor_shape = mixed_qkv.size()[:-1] + (
        module.num_key_value_heads,
        (
            (module.num_heads // module.num_key_value_heads + 2)
            * module.head_dim
        ),
    )
    mixed_qkv = mixed_qkv.view(*new_tensor_shape)

    split_arg_list = [
        (
            module.num_heads
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
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.projection_dim = config.hidden_size * config.expand_factor
        self.head_dim = self.projection_dim // self.num_heads
        self.rope_theta = config.rope_theta
        self.qk_norm = config.qk_norm
        self.window_size = config.sliding_window_size
        self.reuse_kv = reuse_kv

        projection_dim = self.head_dim * (self.num_heads + 2 * (0 if reuse_kv else self.num_key_value_heads))
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        if self.qk_norm:
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
            if not reuse_kv:
                self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, self.num_heads*self.head_dim, bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_local_attn > 0.:
                    score = self.config.softcap_local_attn * torch.tanh(score / self.config.softcap_local_attn)
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
        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, cos, sin)
        if not self.reuse_kv:
            key_states = apply_rotary_emb(key_states, cos, sin)

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
            attention_interface = lambda q, k, v, softmax_scale, **kw: flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1)), score_mod=self.score_mod, scale=softmax_scale, enable_gqa=self.num_heads > self.num_key_value_heads).transpose(1, 2)
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
            softcap=self.config.softcap_local_attn,
            softmax_scale=None if not self.config.use_uscaling else 1/self.head_dim,
        )
        if len(attn_output.shape) == 3:
            attn_output = attn_output.view(query_states.size(0), query_states.size(1), attn_output.size(-2), attn_output.size(-1)) # keep (B, L, H, D)

        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_heads, self.head_dim).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)

        #if cache_params is not None and not self.reuse_kv:
        #    cache_params.trim(self.layer_idx)

        return attn_output, last_key_states, last_value_states

class DragonCompressedConvolutionalAttention(nn.Module):
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
        """
        self.rope_theta = config.rope_theta
        """
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_k_heads = config.num_key_value_heads
        self.gqa_groups = self.num_q_heads // self.num_k_heads
        self.head_dim = config.cca_head_dim

        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.latent_k_dim = self.num_k_heads * self.head_dim
        c_qk = self.latent_q_dim + self.latent_k_dim
        self.linear_qk = DragonLinear(self.config, self.hidden_size, c_qk, bias=False)

        # seq conv
        self.total_padding = config.cca_seq_kernel_size-1
        self.conv_qk0 = nn.Conv1d(
            in_channels=c_qk,
            out_channels=c_qk,
            groups=c_qk,
            kernel_size=config.cca_seq_kernel_size,
            bias=False)

        # seq+ch conv
        self.conv_qk1 = nn.Conv1d(
            in_channels=c_qk,
            out_channels=c_qk,
            groups=(self.num_q_heads + self.num_k_heads),
            kernel_size=config.cca_seq_kernel_size,
            bias=False)

        # Value projections
        c_v = (self.num_k_heads * self.head_dim) // 2
        self.val_proj1 = DragonLinear(self.config, self.hidden_size, c_v, bias=False)
        self.val_proj2 = DragonLinear(self.config, self.hidden_size, c_v, bias=False)

        # scaling & learnable key temperature (broadcast to [*, *, K, 1])
        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        self.rotary_emb = DragonRotaryEmbedding(config, head_dim=self.head_dim)
        # TODO
        """if self.config.qk_norm:
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
            if not reuse_kv:
                self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)"""

        """# TODO: for the gate, not clear: before or after upproj ?
        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, self.num_q_heads*self.head_dim, bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)"""
        assert not self.config.gate_attn, "Gating not implemented yet for CCA"

    def forward(self, 
                hidden_states,
                position_ids: Optional[torch.LongTensor] = None,
                cache_params: Optional[HybridDragonDynamicCache] = None,
                **kwargs):
        B, L, _ = hidden_states.shape

        # Initial low-rank combined down projection for query and key
        qk_packed0 = self.linear_qk(hidden_states) # (B, L_new, C)
        C = qk_packed0.size(-1)
        qk_ch_first_new = qk_packed0.transpose(1, 2) # (B, C, L_new)

        # conv_qk0: conv seq
        if cache_params is None:
            x = F.pad(qk_ch_first_new, (self.total_padding, 0))
            x = self.conv_qk0(x) # (B, C, L_new)
        else:
            state0 = cache_params.get_cca_qk0_state(self.layer_idx)
            if state0 is None:
                state0 = qk_ch_first_new.new_zeros(B, C, self.total_padding) # K-1 zeros
            assert state0.size(0) == B and state0.size(1) == C, f"Cache state0 shape mismatch: got {state0.shape}, expected first dims ({B}, {C})"
            x_cat = torch.cat([state0, qk_ch_first_new], dim=2) # (B, C, K-1+L_new)
            x = self.conv_qk0(x_cat)[..., -qk_ch_first_new.size(2):] # keep last L_new
            cache_params.set_cca_qk0_state(self.layer_idx, x_cat[..., -self.total_padding:])

        # conv_qk1: conv seq+ch
        if cache_params is None:
            x_cat = F.pad(x, (self.total_padding, 0))
            y = self.conv_qk1(x_cat)
        else:
            state1 = cache_params.get_cca_qk1_state(self.layer_idx)
            if state1 is None:
                state1 = x.new_zeros(x.size(0), x.size(1), self.total_padding) # K-1 zeros
            assert state1.size(0) == x.size(0) and state1.size(1) == x.size(1), f"Cache state1 shape mismatch: got {state1.shape}, expected first dims ({x.size(0)}, {x.size(1)})"
            x_cat = torch.cat([state1, x], dim=2)
            y = self.conv_qk1(x_cat)[..., -x.size(2):]
            cache_params.set_cca_qk1_state(self.layer_idx, x_cat[..., -self.total_padding:])

        x = y
        qk_packed3 = x.transpose(1, 2) # (B, L, C)

        # split pre-activations
        key_pre   = qk_packed0[..., self.latent_q_dim:self.latent_q_dim + self.latent_k_dim] # (B, L, Hk*Dh)
        key_pre   = key_pre.view(B, L, self.num_k_heads, 1, self.head_dim).expand(-1, -1, -1, self.gqa_groups, -1)
        key_pre   = key_pre.reshape(B, L, self.num_q_heads, self.head_dim) # (B, L, Hq, Dh)

        query_pre = qk_packed0[..., :self.latent_q_dim].view(B, L, self.num_q_heads, self.head_dim)
        qk_mean_q = 0.5 * (query_pre + key_pre) # (B, L, Hq, Dh)
        qk_mean_k = qk_mean_q.view(B, L, self.num_k_heads, self.gqa_groups, self.head_dim).mean(dim=3)

        query = qk_packed3[..., :self.latent_q_dim].view(B, L, self.num_q_heads, self.head_dim) + qk_mean_q
        key   = qk_packed3[..., self.latent_q_dim:self.latent_q_dim + self.latent_k_dim].view(B, L, self.num_k_heads, self.head_dim) + qk_mean_k

        # value shift.
        if cache_params is None:
            hidden_states_d = torch.roll(hidden_states, shifts=1, dims=1)
            hidden_states_d[:, 0].zero_()
        else:
            prev_h = cache_params.get_prev_hidden(self.layer_idx)
            if prev_h is None:
                prev_h = hidden_states.new_zeros(B, hidden_states.size(-1))
            assert prev_h.size(0) == B, f"Cache prev hidden shape mismatch: got {prev_h.shape}, expected first dim {B}"
            hidden_states_d = torch.cat([prev_h.unsqueeze(1), hidden_states[:, :-1, :]], dim=1)
            cache_params.set_prev_hidden(self.layer_idx, hidden_states[:, -1, :])

        value1 = self.val_proj1(hidden_states) # (B, L, Hk*Dh/2)
        value2 = self.val_proj2(hidden_states_d) # (B, L, Hk*Dh/2)
        value  = torch.cat([value1, value2], dim=-1).view(B, L, self.num_k_heads, self.head_dim) 

        # Query and key normalization with temperature and scaling
        qnorm = query.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        knorm = key.norm(p=2,   dim=-1, keepdim=True).clamp_min(1e-6)
        key   = (key * self.sqrt_head_dim / knorm) * self.temp.clamp(-4, 4).exp().to(key.dtype)[None, None, :, None]
        query =  query * self.sqrt_head_dim / qnorm

        # RoPE.
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        # KV-cache.
        if cache_params is not None:
            key, value = cache_params.update(key, value, self.layer_idx)

        # attention computation.
        wsize = self.config.slw_wsize if self.config.slw_wsize > 0 else -1

        if ATTN_IMPL == "eager":
            attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "fa2":
            attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
        elif ATTN_IMPL == "fa3":
            attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)[0]
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        attn_output = attention_interface(
            query.contiguous().bfloat16(),
            key.contiguous().bfloat16(),
            value.contiguous().bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.config.softcap_local_attn,
            softmax_scale=1. # None if not self.config.use_uscaling else 1/self.head_dim, # TODO
        )
        assert len(attn_output.shape) == 4

        # useless for now
        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_heads, self.head_dim).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)

        #if cache_params is not None and wsize > 0:
        #    cache_params.trim(self.layer_idx)

        return attn_output, None, None

class DragonNativeSparseAttention(nn.Module):
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
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.nsa_head_dim
        self.rope_theta = config.rope_theta
        self.qk_norm = config.qk_norm
        self.window_size = config.sliding_window_size
        self.reuse_kv = reuse_kv

        projection_dim = self.head_dim * (self.num_heads + 2 * (0 if reuse_kv else self.num_key_value_heads))
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        self.g_proj = DragonLinear(config, self.hidden_size, self.num_heads * 3, bias=False) # not hidden weight?

        if self.qk_norm:
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
            if not reuse_kv:
                self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        
        self.rotary_emb = DragonRotaryEmbedding(config, head_dim=self.head_dim)

        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, self.num_heads*self.head_dim, bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # Q, K, V and gates projections.
        if not self.reuse_kv:
            query_states, key_states, value_states = get_query_key_value_tensors(self, hidden_states)
        else:
            query_states = get_query_key_value_tensors(self, hidden_states)
            key_states, value_states = key_value_last_layer
        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            if not self.reuse_kv:
                key_states = self.k_norm(key_states)

        # RoPE.
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states = apply_rotary_emb(query_states, cos, sin)
        if not self.reuse_kv:
            key_states = apply_rotary_emb(key_states, cos, sin)

        # KV-cache.
        if not self.reuse_kv and cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # attention computation.
        attn_output = nsa_func(
            q=query_states.bfloat16(),
            k=key_states.bfloat16(),
            v=value_states.bfloat16(),
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_count=self.config.nsa_topk,
            block_size=self.config.nsa_block_size,
            window_size=self.config.nsa_window_size,
            scale=None if not self.config.use_uscaling else 1/self.head_dim
        ) # TODO: softcap?

        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_heads, self.head_dim).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)

        #if cache_params is not None and not self.reuse_kv:
        #    cache_params.trim(self.layer_idx)

        return attn_output, None, None

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)

class DragonDeepSeekSparseAttention(nn.Module):
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
        # indexer attention
        self.indexer_num_heads = config.num_attention_heads_indexer
        self.indexer_head_dim = config.head_dim_indexer
        self.q_lora_rank = config.dsa_q_lora_rank
        self.topk = config.dsa_topk
        self.indexer_q_proj_down = DragonLinear(config, config.hidden_size, self.q_lora_rank, bias=False) # TODO fuse projs
        self.indexer_q_proj_up = DragonLinear(config, self.q_lora_rank, self.indexer_num_heads * self.indexer_head_dim, bias=False)
        self.indexer_q_norm = DragonRMSNorm(self.q_lora_rank, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        self.indexer_k_proj = DragonLinear(config, config.hidden_size, self.indexer_head_dim, bias=False)
        self.indexer_k_norm = DragonLayerNorm(self.indexer_head_dim, eps=config.norm_epsilon)
        self.indexer_weights_proj = DragonLinear(config, config.hidden_size, self.indexer_num_heads, bias=False)

        # normal attention
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.projection_dim = config.hidden_size * config.expand_factor
        self.head_dim = self.projection_dim // self.num_heads
        self.rope_theta = config.rope_theta
        self.qk_norm = config.qk_norm
        self.window_size = config.sliding_window_size
        self.reuse_kv = reuse_kv

        projection_dim = self.head_dim * (self.num_heads + 2 * (0 if reuse_kv else self.num_key_value_heads))
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        if self.qk_norm:
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
            if not reuse_kv:
                self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, self.num_heads*self.head_dim, bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

        self.causal_mask = None

    def compute_index_scores_pytorch(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute index scores using pure PyTorch (matches official implementation).
        
        Args:
            x: Input tokens [batch, seq_len, d_model]
            qr: Low-rank query representation [batch, seq_len, q_lora_rank]
            
        Returns:
            index_scores: [batch, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        q_indexer = self.indexer_q_proj_up(self.indexer_q_norm(self.indexer_q_proj_down(x))).view(
            batch_size, seq_len, self.indexer_num_heads, self.indexer_head_dim
        )
        k_indexer = self.indexer_k_norm(self.indexer_k_proj(x)).view(
            batch_size, seq_len, 1, self.indexer_head_dim
        )
        q_indexer = rotate_activation(q_indexer.to(torch.bfloat16))
        k_indexer = rotate_activation(k_indexer.to(torch.bfloat16))

        weights = self.indexer_weights_proj(x) * (self.indexer_num_heads ** -0.5) * (self.indexer_head_dim ** -0.5) # [batch, seq_len, num_heads]

        dots = torch.matmul(q_indexer, k_indexer.permute(0, 2, 3, 1)) # -> [batch, seq_len, num_heads, seq_len]
        dots = dots.transpose(-2, -1) # [batch, seq_len, seq_len, num_heads]
        dots = F.relu(dots)
        weighted_dots = dots * weights.unsqueeze(2)
        index_scores = weighted_dots.sum(dim=-1) # [batch, seq_len, seq_len]

        return index_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        _, seq_len, _ = hidden_states.size()

        # compute index scores
        index_scores = self.compute_index_scores_pytorch(hidden_states) # (B, L, L)
        if self.causal_mask is None or self.causal_mask.size(-1) != seq_len:
            self.causal_mask = torch.triu(torch.full((1, seq_len, seq_len), -100, device=hidden_states.device), diagonal=1) # (1, L, L)
        index_scores = index_scores + self.causal_mask # TODO: possible to add it inside the kernel?

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
        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, cos, sin)
        if not self.reuse_kv:
            key_states = apply_rotary_emb(key_states, cos, sin)

        # KV-cache.
        if not self.reuse_kv and cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # save k,v for next layer (*after* norm and RoPE and kv-cache update)
        if not self.reuse_kv:
            last_key_states, last_value_states = key_states, value_states

        # sparse attention mask/bias from top-k index scores
        #dtype = x.dtype
        #min_dtype = torch.finfo(dtype).min
        index_scores = index_scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        topk = min(self.topk, index_scores.size(-1))
        _, topk_indices = torch.topk(index_scores, topk, dim=-1, largest=True, sorted=False) # (batch, n_heads, seq_len, top_k)
        attn_mask = torch.zeros_like(index_scores, dtype=torch.bool)
        attn_mask = attn_mask.scatter(-1, topk_indices, True) # todo: remove True, replace by topk_values != min_dtype??
        attn_bias_const = torch.where(attn_mask, torch.zeros_like(index_scores), torch.full_like(index_scores, -100))
        attn_bias_pass  = torch.where(attn_mask, index_scores,                   torch.full_like(index_scores, -100))
        attn_bias = attn_bias_pass + (attn_bias_const - attn_bias_pass).detach() # STE

        # attention computation.
        #wsize = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size #TODO

        attn_output = flash_dmattn_func(
                query=query_states.bfloat16(),
                key=key_states.bfloat16(),
                value=value_states.bfloat16(),
                #attn_mask=attn_mask,
                attn_bias=attn_bias.bfloat16(),
                is_causal=True,
                softmax_scale=1.0/math.sqrt(self.head_dim) if not self.config.use_uscaling else 1/self.head_dim,
                softcap=self.config.softcap_global_attn,
                deterministic=False,
            )

        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_heads, self.head_dim).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)

        #if cache_params is not None and not self.reuse_kv:
        #    cache_params.trim(self.layer_idx)

        return attn_output, last_key_states, last_value_states
    
class DragonDynamicMaskAttention(nn.Module):
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
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.projection_dim = config.hidden_size * config.expand_factor
        self.head_dim = self.projection_dim // self.num_heads
        self.rope_theta = config.rope_theta
        self.qk_norm = config.qk_norm
        self.window_size = config.sliding_window_size
        self.reuse_kv = reuse_kv
        self.topk = config.dsa_topk

        # attributes read by flash_dmattn_func wrapper
        self.keep_window_size = self.topk
        self.is_causal = True

        self.A = nn.Parameter(torch.zeros(config.num_key_value_heads))
        self.dt_proj = DragonLinear(config, config.num_key_value_heads*self.head_dim, config.num_key_value_heads, bias=False)

        projection_dim = self.head_dim * (self.num_heads + 2 * (0 if reuse_kv else self.num_key_value_heads))
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        if self.qk_norm:
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
            if not reuse_kv:
                self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, self.num_heads*self.head_dim, bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
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
        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, cos, sin)
        if not self.reuse_kv:
            key_states = apply_rotary_emb(key_states, cos, sin)

        # KV-cache.
        if not self.reuse_kv and cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # save k,v for next layer (*after* norm and RoPE and kv-cache update)
        if not self.reuse_kv:
            last_key_states, last_value_states = key_states, value_states

        # attention computation.
        #wsize = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size # TODO

        # sampling dt_states from value_states to generate attention bias
        dt_states = self.dt_proj(value_states.reshape(value_states.size(0), value_states.size(1), -1)) # (B, L, h)
        #attn_bias = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2).to(hidden_states.dtype)
        attn_bias = (self.A * F.softplus(dt_states)).transpose(-1, -2).unsqueeze(-2).to(hidden_states.dtype)

        attn_output, _ = flash_dynamic_mask_attention_forward(
            self,
            query_states.transpose(1, 2).bfloat16(),
            key_states.transpose(1, 2).bfloat16(),
            value_states.transpose(1, 2).bfloat16(),
            attention_mask=None,
            attention_bias=attn_bias.bfloat16(),
            scaling=1.0/math.sqrt(self.head_dim) if not self.config.use_uscaling else 1/self.head_dim,
            softcap=self.config.softcap_global_attn,
        )

        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_heads, self.head_dim).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)

        #if cache_params is not None and not self.reuse_kv:
        #    cache_params.trim(self.layer_idx)

        return attn_output, last_key_states, last_value_states

class DragonDifferentialAttention(nn.Module):
    """
    Multi-headed differential attention (https://arxiv.org/abs/2410.05258)
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
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size * config.expand_factor // self.num_heads
        self.qk_norm = config.qk_norm
        self.softcap = config.softcap_global_attn
        self.scalable_softmax = config.scalable_softmax

        projection_dim = self.head_dim * (self.num_heads + 2 * self.num_key_value_heads)
        self.linear_qkv = DragonLinear(config, config.hidden_size, projection_dim, bias=False)

        if self.qk_norm:
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
            self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))

        self.register_buffer("lambda_init", torch.tensor(0.8 - 0.6 * math.exp(-0.3 * (layer_idx+1))), persistent=False)
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))

        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, (self.num_heads // 2) * (2 * self.head_dim), bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_global_attn > 0.:
                    score = self.config.softcap_global_attn * torch.tanh(score / self.config.softcap_global_attn)
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
        **kwargs,
    ):
        # Q, K, V projections.
        query_states, key_states, value_states = get_query_key_value_tensors(self, hidden_states)
        value_states = value_states.reshape(value_states.size(0), value_states.size(1), value_states.size(2)//2, 2*value_states.size(3))

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        wsize = self.config.slw_wsize
        # scalable softmax.
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            T = query_states.size(1)
            pos = (position_ids.to(torch.float32).view(position_ids.size(0), T, 1, 1) + 1.)
            log_pos = pos.log() if wsize <= 0 else torch.clamp_max(pos, wsize).log()
            query_states = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * query_states
            # TODO: caching mechanism for log_pos

        # KV-cache.
        if cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # attention computation.
        # split q,k heads into two groups
        query1_states, query2_states = query_states[:, :, torch.arange(0, self.num_heads, 2)].contiguous(), query_states[:, :, torch.arange(1, self.num_heads, 2)].contiguous()
        key1_states, key2_states = key_states[:, :, torch.arange(0, self.num_key_value_heads, 2)].contiguous(), key_states[:, :, torch.arange(1, self.num_key_value_heads, 2)].contiguous()

        if DIFF_ATTN_IMPL == "flex_head":
            diff_attention_interface = lambda q, k, v, wsize, **kw: flex_head_fa.flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
        elif DIFF_ATTN_IMPL == "fa2":
            def diff_attention_interface(q, k, v, wsize, **kw):
                D = v.size(3)
                v1 = v[:, :, :, :D//2]
                v2 = v[:, :, :, D//2:]
                o1 = flash_attn_func(q, k, v1, window_size=(wsize, 0), **kw)
                o2 = flash_attn_func(q, k, v2, window_size=(wsize, 0), **kw)
                o = torch.cat([o1, o2], dim=-1)
                return o
        elif DIFF_ATTN_IMPL == "fa3":
            def diff_attention_interface(q, k, v, wsize, **kw):
                D = v.size(3)
                v1 = v[:, :, :, :D//2]
                v2 = v[:, :, :, D//2:]
                o1 = flash_attn_func(q, k, v1, window_size=(wsize, 0), **kw)[0]
                o2 = flash_attn_func(q, k, v2, window_size=(wsize, 0), **kw)[0]
                o = torch.cat([o1, o2], dim=-1)
                return o
        elif DIFF_ATTN_IMPL == "flex":
            if wsize != self.last_wsize:
                self.last_wsize = self.build_mask(wsize)
            diff_attention_interface = lambda q, k, v, softmax_scale, **kw: flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1)), score_mod=self.score_mod, scale=softmax_scale, enable_gqa=self.num_heads > self.num_key_value_heads).transpose(1, 2)
        elif DIFF_ATTN_IMPL == "eager":
            diff_attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)

        # attention_interface = lambda q, k, v, window_size, **kw: eager_attention_forward(q, k, v, window_size=(window_size, 0), **kw)
        y1 = diff_attention_interface(
            query1_states.bfloat16(),
            key1_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.softcap,
            softmax_scale=None if not self.config.use_uscaling else 1/self.head_dim,
        )
        y2 = diff_attention_interface(
            query2_states.bfloat16(),
            key2_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.softcap,
            softmax_scale=None if not self.config.use_uscaling else 1/self.head_dim,
        )
        if len(y1.shape) == 3:
            y1 = y1.view(query1_states.size(0), query1_states.size(1), y1.size(-2), y1.size(-1)) # keep (B, L, H/2, D)
            y2 = y2.view(query1_states.size(0), query1_states.size(1), y2.size(-2), y2.size(-1))
        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1).float()) # (H/2)
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1).float()) # (H/2)
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).view(1, 1, -1, 1).type_as(y1)
        attn_output = (y1 - lambda_full * y2).contiguous()

        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_heads//2, 2*self.head_dim).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)

        #if cache_params is not None:
        #    cache_params.trim(self.layer_idx)

        return attn_output, None, None

class KVRepeat(nn.Module):
    """Modular KV group repeating module for efficient parameter sharing.

    This module handles the repeating of KV (B gate and C) tensors across groups,
    allowing for parameter-efficient attention mechanisms where multiple query heads
    share the same key-value parameters.
    """

    def __init__(self, kv_heads: int, total_heads: int):
        """
        Args:
            kv_heads: Number of key-value heads
            total_heads: Total number of heads (must be divisible by kv_heads)
        """
        super().__init__()

        if total_heads % kv_heads != 0:
            raise ValueError(f"total_heads ({total_heads}) must be divisible by kv_heads ({kv_heads})")

        self.kv_heads = kv_heads
        self.total_heads = total_heads
        self.kv_groups = total_heads // kv_heads
        self.enabled = kv_heads < total_heads
        
        # Pre-compute indices for KV expansion: [0,0,0,1,1,1,...]
        # Register as buffer so it moves with model to correct device
        if self.enabled:
            indices = torch.arange(total_heads) // self.kv_groups
            self.register_buffer('kv_indices', indices, persistent=False)

    def forward(self, tensor: torch.Tensor, pattern: str = "b h d l") -> torch.Tensor:
        """
        Repeat tensor across KV groups if needed.

        Args:
            tensor: Input tensor with shape matching pattern
            pattern: Einops pattern for input tensor (default: 'b h d l')

        Returns:
            Tensor repeated across groups if kv_groups > 1, otherwise unchanged
        """
        if not self.enabled:
            return tensor

        # Use cached indices for efficient expansion
        return tensor.index_select(1, self.kv_indices)

    def reshape_for_kv(self, tensor: torch.Tensor, batch: int, head_dim: int, length: int) -> torch.Tensor:
        """
        Reshape flat tensor to KV head dimensions.

        Args:
            tensor: Flat tensor of shape [B, kv_heads * head_dim, L]
            batch: Batch size
            head_dim: Dimension per head
            length: Sequence length

        Returns:
            Reshaped tensor of shape [B, kv_heads, head_dim, L]
        """
        return tensor.view(batch, self.kv_heads, head_dim, length)

    def expand_and_reshape(self, tensor: torch.Tensor, batch: int, head_dim: int, length: int) -> torch.Tensor:
        """
        Reshape and expand tensor from KV heads to all heads.

        Args:
            tensor: Flat tensor of shape [B, kv_heads * head_dim, L]
            batch: Batch size
            head_dim: Dimension per head
            length: Sequence length

        Returns:
            Expanded tensor of shape [B, total_heads, head_dim, L]
        """
        if not self.enabled:
            return tensor.view(batch, self.kv_heads, head_dim, length)
        
        # Optimized expansion using pre-computed indices
        # Reshape flat to [B, kv_heads, head_dim, L]
        tensor = tensor.view(batch, self.kv_heads, head_dim, length)
        # Use cached indices: [0,0,0,1,1,1,...] for kv_groups repeats
        return tensor.index_select(1, self.kv_indices)
    
    def expand_flat(self, tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
        """
        Expand flat KV tensor directly without intermediate 4D reshape.
        More efficient for cases where we immediately flatten back.

        Args:
            tensor: Flat tensor of shape [B, kv_heads * head_dim, L]
            head_dim: Dimension per head

        Returns:
            Expanded flat tensor of shape [B, total_heads * head_dim, L]
        """
        if not self.enabled:
            return tensor
        
        B, _, L = tensor.shape
        # Reshape to separate heads: [B, kv_heads, head_dim, L]
        tensor = tensor.view(B, self.kv_heads, head_dim, L)
        # Expand using cached indices: [B, total_heads, head_dim, L]
        tensor = tensor.index_select(1, self.kv_indices)
        # Flatten back: [B, total_heads * head_dim, L]
        return tensor.view(B, self.total_heads * head_dim, L)


class SigmoidA(nn.Module):
    """Sigmoid-gated attention parametrization for SSM with optional KV groups."""

    def __init__(
        self,
        config: DragonConfig,
        dim: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        kv_heads: int | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.dim = dim
        self.dtype = dtype
        self.compute_dtype = torch.bfloat16

        self.kv_heads = kv_heads if kv_heads is not None else heads
        if heads % self.kv_heads != 0:
            raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({self.kv_heads})")

        self.kv_repeat = KVRepeat(self.kv_heads, heads)

        self.proj_a = DragonLinear(config, dim, heads, bias=True)

        kv_gate_dim = self.kv_heads * head_dim
        self.proj_b = DragonLinear(config, dim, kv_gate_dim, bias=True)
        self.proj_c = DragonLinear(config, dim, kv_gate_dim, bias=True)

        gate_dim = heads * head_dim
        self.proj_v = DragonLinear(config, dim, gate_dim, bias=False)

    def _compute_base_projections(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute base projections A, B_gate_kv, C_kv, V - shared by all forward methods."""
        x_compute = x.to(self.compute_dtype)
        A = self.proj_a(x_compute).transpose(1, 2)  # [B, heads, L]
        B_gate_kv = self.proj_b(x_compute).transpose(1, 2)  # [B, kv_heads*head_dim, L]
        C_kv = self.proj_c(x_compute).transpose(1, 2)  # [B, kv_heads*head_dim, L]
        V = self.proj_v(x_compute).transpose(1, 2)  # [B, heads*head_dim, L]
        return A, B_gate_kv, C_kv, V

    def forward_fused_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for fused gates kernel - returns packed tensor [B, 3*H*D + H, L]."""
        B, L, D = x.shape

        A, B_gate_kv, C_kv, V = self._compute_base_projections(x)

        if self.kv_repeat.enabled:
            B_gate = self.kv_repeat.expand_and_reshape(B_gate_kv, B, self.head_dim, L)
            B_gate = B_gate.reshape(B, self.heads * self.head_dim, L)
            C = self.kv_repeat.expand_and_reshape(C_kv, B, self.head_dim, L)
            C = C.reshape(B, self.heads * self.head_dim, L)
        else:
            B_gate = B_gate_kv
            C = C_kv

        out = torch.cat([B_gate, V, C, A], dim=1)  # [B, 3*H*D + H, L]

        return out

    def forward_axcv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for default/pytorch kernels - returns (A, X, C, V) tensors.

        This method is used by both the default BTP kernel and pure PyTorch implementation
        since they both need the same tensor format: separate A, X, C, V tensors.
        """
        B, L, D = x.shape
        H, DH = self.heads, self.head_dim

        A_logits, B_gate_kv_logits, C_kv, V_flat = self._compute_base_projections(x)

        V = V_flat.view(B, H, DH, L)  # [B, H, DH, L]

        if self.kv_repeat.enabled:
            B_gate_logits = self.kv_repeat.expand_and_reshape(B_gate_kv_logits, B, DH, L)
            C = self.kv_repeat.expand_and_reshape(C_kv, B, DH, L)
        else:
            B_gate_logits = B_gate_kv_logits.view(B, H, DH, L)
            C = C_kv.view(B, H, DH, L)

        A = torch.sigmoid(A_logits)  # [B, H, L]
        B_gate = torch.sigmoid(B_gate_logits)  # [B, H, DH, L]

        X = B_gate * V  # [B, H, DH, L]

        return A, X, C, V

class DragonSlidingWindowRecurrenceAttention(nn.Module):
    def __init__(
        self,
        config: DragonConfig,
        dtype: torch.dtype = torch.float32,
        k: int = 2,
        wpb: int = 32,
        output_dtype: torch.dtype | None = None,
        block_size: int = 16,
        kv_heads: int | None = None,
    ):
        super().__init__()

        method = "pytorch"
        if method not in ("default", "pytorch", "pytorch_linspace"):
            raise ValueError(f"method must be one of 'default', 'pytorch', 'pytorch_linspace', got {method}")

        self.config = config
        self.dim = self.config.hidden_size
        self.head_dim = 16
        self.heads = 2 * (self.dim // self.head_dim) # can't x2 head dim, so x2 number of heads
        self.length = (self.config.max_position_embeddings + 15) // 16 * 16
        self.method = method
        self.dtype = dtype
        self.compute_dtype = torch.bfloat16
        self.k = k
        self.wpb = wpb
        self.output_dtype = output_dtype if output_dtype is not None else dtype
        self.block_size = block_size
        self.kv_heads = kv_heads if kv_heads is not None else self.heads

        self.param = SigmoidA(config, self.dim, self.heads, self.head_dim, dtype=dtype, kv_heads=self.kv_heads)

        if self.config.gate_attn:
            self.gate_proj = DragonLinear(self.config, self.hidden_size, (self.num_heads // 2) * (2 * self.head_dim), bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

        if method == "default":
            from spear.ops.btp import btp

            self.btp_module = btp
            self._forward_fn = self._forward_default

        elif method == "pytorch":
            from spear.ops.btp.reference import block_two_pass_log

            self.pytorch_block_two_pass = block_two_pass_log
            self._forward_fn = self._forward_pytorch

        elif method == "pytorch_linspace":
            from spear.ops.btp.reference import block_two_pass_linspace

            self.pytorch_block_two_pass = block_two_pass_linspace
            self._forward_fn = self._forward_pytorch

        else:
            raise ValueError(f"Invalid method: {method}. Supported methods: 'default', 'pytorch', 'pytorch_linspace'. ")

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using PyTorch implementation."""
        A, X, C, V = self.param.forward_axcv(x)  # A: [B, H, L], X,C,V: [B, H, D, L]
        y = self.pytorch_block_two_pass(A, X, self.block_size)  # [B, H, D, L]
        y = y * C + V  # [B, H, D, L]
        y = y.permute(0, 3, 1, 2) # [B, L, H, D]
        return y

    def _forward_default(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using BTP kernel."""
        A, X, C, V = self.param.forward_axcv(x)  # A: [B, H, L], X,C,V: [B, H, D, L]
        y = self.btp_module(A, X, self.k, self.wpb, self.output_dtype)
        y = y * C + V  # [B, H, D, L]
        y = y.permute(0, 3, 1, 2) # [B, L, H, D]
        return y

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Main forward pass - delegates to method-specific implementation."""
        o = self._forward_fn(hidden_states)
        if self.config.gate_attn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), o.size(2), o.size(3)).to(attn_output.dtype)
            if self.config.zero_centered_gate_type == 1:
                attn_output = attn_output * F.silu(g_proj)
                attn_output = attn_output + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                attn_output = attn_output * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                attn_output = attn_output * F.silu(g_proj + self.gate_bias)
        return o, 0, 0

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

# works with tensors >2**31 elements
def safe_conv1d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    x = x.contiguous(); w = w.contiguous()
    N, C, L = x.shape
    k = w.shape[-1]; Cout = w.shape[0]
    Lout = (L + 2*padding - dilation*(k-1) - 1)//stride + 1
    per_sample_in  = C*L
    per_sample_out = Cout*Lout
    per_sample_max = max(per_sample_in, per_sample_out)
    INT32_MAX = 2_147_483_647
    bs = max(1, min(N, INT32_MAX // max(1, per_sample_max)))
    if bs >= N:
        return F.conv1d(x, w, b, stride, padding, dilation, groups)
    outs = []
    for i in range(0, N, bs):
        outs.append(F.conv1d(x[i:i+bs], w, b, stride, padding, dilation, groups))
    return torch.cat(outs, dim=0)

def get_qkv_tensors_gdn(module: nn.Module, hidden_states: torch.Tensor):
    H, G, dk, dv = module.n_heads, module.n_kv_heads, module.dk, module.dv
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

class DragonGatedDeltaNet(nn.Module):
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

        self.n_heads = config.num_attention_heads_gdn if config.num_attention_heads_gdn > 0 else config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads_gdn if config.num_key_value_heads_gdn > 0 else self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.groups = self.n_heads // self.n_kv_heads

        self.head_dim = int(config.hidden_size * (config.expand_factor/2)) // self.n_heads
        self.dk = self.head_dim
        self.dv = 2*self.head_dim
        self.key_dim = self.n_kv_heads * self.dk
        self.value_dim = self.n_kv_heads * self.dv

        self.n_heads_local = self.n_heads // 1
        self.key_dim_local = self.n_heads_local * self.dk
        self.value_dim_local = self.n_heads_local * self.dv

        self.linear_qkv = DragonLinear(
            config, config.hidden_size,
            self.n_heads*self.dk + self.n_kv_heads*self.dk + self.n_kv_heads*self.dv,
            bias=False
        )
        self.linear_ba = DragonLinear(
            config, config.hidden_size,
            self.n_heads + self.n_heads, #+ self.n_heads*self.dv, # b(H), a(H), g(H*dv)
            bias=False
        )

        if self.config.gate_gdn:
            self.gate_proj = DragonLinear(self.config, config.hidden_size, self.n_heads * self.dv, bias=False)
            if self.config.zero_centered_gate:
                self.register_buffer("gate_bias", torch.tensor(1.28 if self.config.zero_centered_gate_type==3 else 1.), persistent=False)
            else:
                self.register_buffer("gate_bias", torch.tensor(0.), persistent=False)

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

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.n_heads_local, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        self.conv_size = config.conv_kernel
        self.conv_dim  = self.n_heads*self.dk + self.n_kv_heads*self.dk + self.n_kv_heads*self.dv
        self.qkv_conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_size, groups=self.conv_dim, padding=self.conv_size-1)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                cache_params: Optional[HybridDragonDynamicCache] = None,
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
        q, k_ng, v_ng = get_qkv_tensors_gdn(self, hidden_states)     # q:(B,L,H,dk), k/v:(B,L,Ng,dk/dv)
        bag = self.linear_ba(hidden_states)                          # (B,L,2H + H*dv)
        #b_proj, a_proj, g_proj = torch.split(bag, [self.n_heads, self.n_heads, self.n_heads*self.dv], dim=-1)
        b_proj, a_proj = torch.split(bag, [self.n_heads, self.n_heads], dim=-1)

        # --- pack for conv ---
        q_proj = rearrange(q,    "b l h d -> b l (h d)")
        k_proj = rearrange(k_ng, "b l g d -> b l (g d)")
        v_proj = rearrange(v_ng, "b l g d -> b l (g d)")
        mixed_qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2) # (B,C,L)

        # conv
        if cache_params is not None:
            conv_cache = cache_params.conv_caches[self.layer_idx]
            ssm_cache = cache_params.ssm_caches[self.layer_idx]

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
            [self.n_heads*self.dk, self.n_kv_heads*self.dk, self.n_kv_heads*self.dv],
            dim=-1,
        )
        q    = rearrange(q_proj, "b l (h d) -> b l h d", h=self.n_heads)
        k_ng = rearrange(k_proj, "b l (g d) -> b l g d", g=self.n_kv_heads)
        v_ng = rearrange(v_proj, "b l (g d) -> b l g d", g=self.n_kv_heads)

        k = k_ng.repeat_interleave(self.groups, dim=2)
        v = v_ng.repeat_interleave(self.groups, dim=2)

        b_proj = rearrange(b_proj, "b l (h) -> b l h", h=self.n_heads)
        a_proj = rearrange(a_proj, "b l (h) -> b l h", h=self.n_heads)
        #g_proj = rearrange(g_proj, "b l (h d) -> b l h d", h=self.n_heads)
        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a_proj.float() + self.dt_bias)

        # RoPE.
        if self.config.rope_gdn == "rope":
            cos, sin = position_embeddings
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # GDN main computation
        if not use_precomputed_states:
            o, ssm_cache = self.chunk_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                scale=None if not self.config.use_uscaling else 1/self.dk,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True
            ) # (B L H dv)
        else:
            o, ssm_cache = self.recurrent_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                scale=None if not self.config.use_uscaling else 1/self.dk,
                initial_state=ssm_cache,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True
            ) # (B L H dv)

        #o = o * F.silu(g_proj)
        if self.config.gate_gdn:
            g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.n_heads, self.dv).to(o.dtype)
            if self.config.zero_centered_gate_type == 1:
                o = o * F.silu(g_proj)
                o = o + self.gate_bias
            elif self.config.zero_centered_gate_type == 2:
                o = o * (F.silu(g_proj) + self.gate_bias)
            elif self.config.zero_centered_gate_type == 3:
                o = o * F.silu(g_proj + self.gate_bias)

        # update GDN cache
        if cache_params is not None:
            cache_params.ssm_caches[self.layer_idx] = ssm_cache

        return o, None, None

class DragonMLP(nn.Module):
    def __init__(self, config: DragonConfig):
        super().__init__()
        self.fc_1 = DragonLinear(config, config.hidden_size, config.intermediate_size, bias=False)
        self.fc_2 = DragonLinear(config, config.intermediate_size, config.hidden_size, bias=False)
        self.register_buffer("_2_sqrt_5", torch.tensor(2/math.sqrt(5)) if config.use_uscaling else torch.tensor(1.), persistent=False)

    def forward(self, hidden_states):
        hidden_states = self.fc_1(hidden_states)
        hidden_states = self._2_sqrt_5 * F.relu(hidden_states).square()
        hidden_states = self.fc_2(hidden_states)
        return hidden_states

class DragonMonoBlock(GradientCheckpointingLayer):
    def __init__(self, config: DragonConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.expand_factor = config.expand_factor

        if layer_type == 'g':
            self.mixer = DragonGatedDeltaNet(config, layer_idx=layer_idx)
        elif layer_type == 'f':
            self.mixer = DragonDifferentialAttention(config, layer_idx=layer_idx)
        elif layer_type == 's':
            self.mixer = DragonDeepSeekSparseAttention(config, reuse_kv=False, layer_idx=layer_idx)
        elif layer_type == 'm':
            self.mixer = DragonDynamicMaskAttention(config, reuse_kv=False, layer_idx=layer_idx)
        elif layer_type == 'w':
            self.mixer = DragonAttention(config, reuse_kv=False, layer_idx=layer_idx)
        elif layer_type == 'p':
            self.mixer = DragonSlidingWindowRecurrenceAttention(config)
        elif layer_type == 'c':
            self.mixer = DragonCompressedConvolutionalAttention(config, layer_idx=layer_idx)
        elif layer_type == 'n':
            self.mixer = DragonNativeSparseAttention(config, reuse_kv=False, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        if layer_type == 'c':
            self.mixer_proj = DragonLinear(config, self.mixer.num_q_heads*self.mixer.head_dim, config.hidden_size, bias=False)
        elif layer_type == 'n':
            self.mixer_proj = DragonLinear(config, self.mixer.num_heads*self.mixer.head_dim, config.hidden_size, bias=False)
        else:
            self.mixer_proj = DragonLinear(config, int(self.expand_factor*config.hidden_size), config.hidden_size, bias=False)

        if isinstance(self.mixer, DragonDifferentialAttention):
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=self.mixer.num_heads//2, d_head=2*self.mixer.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        elif isinstance(self.mixer, DragonAttention) or isinstance(self.mixer, DragonDeepSeekSparseAttention) or isinstance(self.mixer, DragonDynamicMaskAttention):
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=self.mixer.num_heads, d_head=self.mixer.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        elif isinstance(self.mixer, DragonGatedDeltaNet):
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=self.mixer.n_heads, d_head=self.mixer.dv, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        elif isinstance(self.mixer, DragonSlidingWindowRecurrenceAttention):
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=self.mixer.heads, d_head=self.mixer.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        elif isinstance(self.mixer, DragonCompressedConvolutionalAttention):
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=self.mixer.num_q_heads, d_head=self.mixer.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        elif isinstance(self.mixer, DragonNativeSparseAttention):
            self.mixer_group_norm = DragonHeadWiseRMSNorm(n_heads=self.mixer.num_heads, d_head=self.mixer.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        self.input_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        self.postmixer_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        self.mlp = DragonMLP(config)

        self.register_buffer("lns", torch.tensor(1.0 if config.use_uscaling else 1. / math.sqrt(layer_idx + (2 if config.old_lns else 1))), persistent=False)
        self.register_buffer("sqrt_tau", torch.sqrt(torch.tensor(self.config.uscaling_tau)) if config.use_uscaling else torch.tensor(1.0), persistent=False)
        self.register_buffer("sqrt_one_minus_tau", torch.sqrt(torch.tensor(1.0 - self.config.uscaling_tau)) if config.use_uscaling else torch.tensor(1.0), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # MIXER.
        residual = hidden_states
        hidden_states = self.lns * self.input_norm(hidden_states) # (B, L, D)
        y_mixer, last_key_states, last_value_states = self.mixer(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cache_params=cache_params,
            key_value_last_layer=key_value_last_layer,
        ) # (B, L, E*D)
        y_mixer = self.mixer_group_norm(y_mixer).view(y_mixer.size(0), y_mixer.size(1), -1)
        y_mixer = self.mixer_proj(y_mixer)
        hidden_states = self.sqrt_one_minus_tau * residual + self.sqrt_tau * y_mixer

        # MLP.
        residual = hidden_states
        hidden_states = self.lns * self.postmixer_norm(hidden_states)
        y_mlp = self.mlp(hidden_states) # (B, L, D)
        hidden_states = self.sqrt_one_minus_tau * residual + self.sqrt_tau * y_mlp

        return hidden_states, last_key_states, last_value_states

class DragonBlock(GradientCheckpointingLayer):
    def __init__(self, config: DragonConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.expand_factor = config.expand_factor

        if layer_type in ['l', 'r']:
            self.attn = DragonAttention(config, reuse_kv=(layer_type=='r'), layer_idx=layer_idx)
        elif layer_type == 'd':
            self.attn = DragonDifferentialAttention(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        self.lin_attn = DragonGatedDeltaNet(config, layer_idx=layer_idx)
        self.mixer_proj = DragonLinear(config, int(self.expand_factor*config.hidden_size), config.hidden_size, bias=False)

        if isinstance(self.attn, (DragonDifferentialAttention)):
            self.attn_group_norm = DragonHeadWiseRMSNorm(n_heads=self.attn.num_heads//2, d_head=2*self.attn.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        else:
            self.attn_group_norm = DragonHeadWiseRMSNorm(n_heads=self.attn.num_heads, d_head=self.attn.head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        self.lin_attn_group_norm = DragonHeadWiseRMSNorm(n_heads=self.lin_attn.n_heads, d_head=self.lin_attn.dv, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        self.input_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        self.postmixer_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        self.mlp = DragonMLP(config)

        self.register_buffer("lns", torch.tensor(1.0 if config.use_uscaling else 1. / math.sqrt(layer_idx + (2 if config.old_lns else 1))), persistent=False)
        self.register_buffer("sqrt_2_2", torch.tensor(math.sqrt(2)/2) if config.use_uscaling else torch.tensor(1/2), persistent=False)
        self.register_buffer("sqrt_tau", torch.sqrt(torch.tensor(self.config.uscaling_tau)) if config.use_uscaling else torch.tensor(1.0), persistent=False)
        self.register_buffer("sqrt_one_minus_tau", torch.sqrt(torch.tensor(1.0 - self.config.uscaling_tau)) if config.use_uscaling else torch.tensor(1.0), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # MIXER.
        residual = hidden_states
        hidden_states = self.lns * self.input_norm(hidden_states) # (B, L, D)
        y_attn, last_key_states, last_value_states = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cache_params=cache_params,
            key_value_last_layer=key_value_last_layer,
        ) # (B, L, E*D)
        y_lin_attn, _, _ = self.lin_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_params=cache_params,
        ) # (B, L, E*D)
        y_attn = self.attn_group_norm(y_attn).view(y_attn.size(0), y_attn.size(1), -1)
        y_lin_attn = self.lin_attn_group_norm(y_lin_attn).view(y_lin_attn.size(0), y_lin_attn.size(1), -1)
        y_mixer = self.mixer_proj(self.sqrt_2_2 * (y_attn + y_lin_attn))
        hidden_states = self.sqrt_one_minus_tau * residual + self.sqrt_tau * y_mixer

        # MLP.
        residual = hidden_states
        hidden_states = self.lns * self.postmixer_norm(hidden_states)
        y_mlp = self.mlp(hidden_states) # (B, L, D)
        hidden_states = self.sqrt_one_minus_tau * residual + self.sqrt_tau * y_mlp

        return hidden_states, last_key_states, last_value_states

class DragonPreTrainedModel(PreTrainedModel):
    config: DragonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DragonBlock"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DragonBlock,
        "attentions": DragonBlock,
    }

    def _init_weights(self, module): # TODO: ??
        if isinstance(module, (DragonLinear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.normal_(module.weight, mean=0., std=1. if self.config.use_uscaling else 0.006)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0., std=1. if self.config.use_uscaling else 0.006)

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
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridDragonDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class DragonModel(DragonPreTrainedModel):
    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DragonBlock(config, layer_idx=i, layer_type=layer) if layer in ['l', 'r', 'd'] else DragonMonoBlock(config, layer_idx=i, layer_type=layer) for i, layer in enumerate(config.layers_config)])

        self.rotary_emb = DragonRotaryEmbedding(config, head_dim=(config.expand_factor*config.hidden_size)//config.num_attention_heads) # only for SWA
        self.final_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

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
        **kwargs
    ) -> DragonOutput:
        B, L = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

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

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        shared_kv = (None, None)
        for block in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, last_k, last_v = block(
                hidden_states,
                position_ids=position_ids,
                cache_params=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                key_value_last_layer=shared_kv,
                **kwargs,
            )
            shared_kv = (last_k, last_v)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

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
        self.lm_head = DragonLinear(config, config.hidden_size, config.vocab_size, bias=False, alpha_fwd=1/config.hidden_size, alpha_bwd=1/math.sqrt(config.hidden_size))
        self.post_init()

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
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = None
        loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(hidden_states.device)

            if linear_cross_entropy is None or not self.config.fused_loss_computation:
                logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()
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
                    alpha_fwd=1/self.config.hidden_size if self.config.use_uscaling else 1.,
                    alpha_bwd=1/math.sqrt(self.config.hidden_size) if self.config.use_uscaling else 1.,
                    impl="cce_exact",
                    shift=1,
                )
        else:
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()

        return DragonCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
DragonForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["DragonModel", "DragonForCausalLM", "DragonPreTrainedModel"]

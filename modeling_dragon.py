# coding=utf-8
"""PyTorch Dragon model."""

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import inspect

import math
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging

from dragon.configuration_dragon import DragonConfig

logger = logging.get_logger(__name__)

ATTN_IMPL = "eager"
try:
    from flash_attn import flash_attn_func # FA2
    ATTN_IMPL = "fa2"
except ImportError:
    try:
        import flash_attn_interface # FA3
        flash_attn_func = flash_attn_interface.flash_attn_func
        _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
        if not _flash_supports_window_size:
            raise ImportError("flash_attn_func does not support window_size parameter. Please update to more recent flash_attn version")
        ATTN_IMPL = "fa3"
    except ImportError:
        logger.warning_once(
            "Flash attention is not installed, using eager attention implementation. "
            "For better performance, consider installing flash_attn."
        )
print(f"Using attention implementation: {ATTN_IMPL}")

DIFF_ATTN_IMPL = None
try:
    import flex_head_fa
    DIFF_ATTN_IMPL = "flex_head"
except ImportError:
    DIFF_ATTN_IMPL = ATTN_IMPL # if we don't have flex_head_fa, fallback to the best attention impl we have
print(f"Using differential attention implementation: {DIFF_ATTN_IMPL}")

# Gated DeltaNet
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError:
    logger.warning_once("Falling back to Torch implementation for Gated DeltaNet as flash-linear-attention module was not found.")
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None

# 1D short convolution
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    logger.warning_once("Falling back to Torch implementation for the short convolution as causal-conv1d module was not found.")
    causal_conv1d_fn, causal_conv1d_update = None, None

class DragonHeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-6):
        super().__init__()
        self.rms = nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, hidden_states):
        B, L, H, D = hidden_states.shape
        y = self.rms(hidden_states) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)

class DragonRMSNorm(nn.RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DragonRMSNorm is equivalent to RMSNorm
        """
        super().__init__(normalized_shape=hidden_size, eps=eps)

class _ScaledLinearFB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, alpha_fwd, alpha_bwd_x, alpha_bwd_w):
        ctx.save_for_backward(x, weight, bias)
        ctx.alpha_bwd_x = alpha_bwd_x
        ctx.alpha_bwd_w = alpha_bwd_w
        return F.linear(x, weight, bias) * alpha_fwd

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        # -------- grads ----------
        grad_x = torch.matmul(grad_out * ctx.alpha_bwd_x, weight)

        go_flat = (grad_out * ctx.alpha_bwd_w).reshape(-1, grad_out.shape[-1])
        x_flat  = x.reshape(-1, x.shape[-1])
        grad_weight = go_flat.t() @ x_flat
        grad_bias   = go_flat.sum(0) if bias is not None else None

        return grad_x, grad_weight, grad_bias, None, None, None

class DragonLinear(nn.Linear):
    """Linear layer with different forward/backward scalings."""
    def __init__(self, config: DragonConfig, in_features, out_features, bias=False, alpha_fwd=None, alpha_bwd=None):
        super().__init__(in_features, out_features, bias)

        if alpha_fwd is None:
            alpha_fwd = 1.0 / math.sqrt(in_features)

        if not config.use_uscaling:
            alpha_fwd, alpha_bwd = 1, 1

        self.register_buffer("alpha_fwd", torch.tensor(float(alpha_fwd)))
        self.register_buffer("alpha_bwd", torch.tensor(float(alpha_bwd if alpha_bwd is not None else alpha_fwd)))

    def forward(self, x):
        return _ScaledLinearFB.apply(x, self.weight, self.bias, self.alpha_fwd, self.alpha_bwd, self.alpha_bwd)

# heavily adapted from flash-linear-attention
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])

def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1

class DragonConv1D(nn.Conv1d):
    """Wrapper around nn.Conv1d (for definition) and causal_conv1d (for forward)"""
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, D, W = *x.shape, self.kernel_size[0]
        N = B
        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        # during the decoding phase, we assume the batch is composed of sequences of length 1
        if cache is not None and T == 1:
            return self.step(x, cache)

        if cache is not None:
            cache[:, :, -min(W, T):].copy_(rearrange(x[..., -min(W, T):, :], 'n w d -> n d w'))

        x = rearrange(x, 'b t d -> b d t')
        if causal_conv1d_fn is not None:
            # Sequence index for each token. Used for varlen.
            # Suppose a batch consists of two sequences with lengths 3 and 4,
            # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
            # NOTE: No need to provide this arg if `cu_seqlens` is passed.
            # This arg is just for BC, and will be removed in the future.
            # [B, T]
            seq_idx = kwargs.get('seq_idx', None)
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation="silu",
                seq_idx=seq_idx,
            )
        else:
            x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
            x = F.silu(x)
        return rearrange(x, "b d t -> b t d"), cache

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        if causal_conv1d_fn is not None:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation="silu",
            )
        else:
            # we follow the fast mode that updates the cache in-place
            cache.copy_(cache.roll(shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            x = F.silu(x)
        return x.view(shape), cache

class HybridDragonAttentionDynamicCache(DynamicCache):
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
    def __init__(self, config: DragonConfig, batch_size, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.dtype = dtype
        self.q_conv_states = []
        self.k_conv_states = []
        self.v_conv_states = []
        self.ssm_states = []
        self._key_cache = {}
        self._value_cache = {}

        for idx, layer_type in enumerate(config.layers_config):
            if layer_type in ['l', 'd']:
                self._key_cache[idx] = None
                self._value_cache[idx] = None

            self.q_conv_states.append(None)
            self.k_conv_states.append(None)
            self.v_conv_states.append(None)
            self.ssm_states.append(None)

        self.window_size = config.sliding_window_size
        self.layers_config = config.layers_config
        self.past_length = [0 for _ in range(len(config.layers_config))]

    def update(
        self,
        k: torch.Tensor, # (B, L, h, D)
        v: torch.Tensor, # (B, L, h, D)
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
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
        # discard old keys/values
        if self.layers_config[layer_idx] == 'l' and self.window_size is not None:
            if k_cache.size(1) > self.window_size:
                k_cache = k_cache[:, -self.window_size:, ...].contiguous()
                v_cache = v_cache[:, -self.window_size:, ...].contiguous()
        # save cache
        self._key_cache[layer_idx] = k_cache
        self._value_cache[layer_idx] = v_cache
        # update cache length
        self.past_length[layer_idx] += added_len
        return k_cache, v_cache

    def update_ssm_cache(
        self,
        q_conv_states: torch.Tensor,
        k_conv_states: torch.Tensor,
        v_conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        # Update the SSM cache
        self.q_conv_states[layer_idx] = q_conv_states
        self.k_conv_states[layer_idx] = k_conv_states
        self.v_conv_states[layer_idx] = v_conv_states
        self.ssm_states[layer_idx] = ssm_states

    def get_ssm_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the SSM cache for the specified layer
        return (
            self.q_conv_states[layer_idx],
            self.k_conv_states[layer_idx],
            self.v_conv_states[layer_idx],
            self.ssm_states[layer_idx],
        )
    
    def get_total_seen(self, layer_idx: int) -> int:
        return self.past_length[layer_idx]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridDragonAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, cache_params: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridDragonAttentionDynamicCache does not have a legacy cache equivalent.")

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class DragonRotaryEmbedding(nn.Module):
    def __init__(self, config: DragonConfig, head_dim: int):
        super().__init__()
        self.half_dim = head_dim // 2

        inv = torch.arange(0, self.half_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** (inv / self.half_dim)), persistent=False)

        self.seq_len_cached = 0
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    @torch.no_grad()
    def _maybe_grow_cache(self, total_len: int, device, dtype=torch.bfloat16):
        if total_len <= self.seq_len_cached:
            return
        new_len = max(2 * total_len, 16)
        t = torch.arange(new_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [new_len, half_dim]
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        self.cos_cached = cos
        self.sin_cached = sin
        self.seq_len_cached = new_len

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        x: [B, L, ...]
        position_ids: [B, L]
        """
        B, L = position_ids.shape
        device = x.device
        #x_dtype = x.dtype

        start_pos = int(position_ids.min().item())
        contiguous = (
            position_ids.max().item() == start_pos + L - 1
            and (position_ids[:, 0] == start_pos).all()
            and (position_ids == (start_pos + torch.arange(L, device=device)[None, :])).all()
        )

        total_len = start_pos + L if contiguous else int(position_ids.max().item()) + 1
        self._maybe_grow_cache(total_len, device=device)

        if contiguous:
            cos = self.cos_cached[start_pos:start_pos+L].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[start_pos:start_pos+L].unsqueeze(0).unsqueeze(2)
        else:
            idx = position_ids.to(torch.long)
            cos = self.cos_cached.index_select(0, idx.reshape(-1)).view(B, L, self.half_dim).unsqueeze(2)
            sin = self.sin_cached.index_select(0, idx.reshape(-1)).view(B, L, self.half_dim).unsqueeze(2)

        #return cos.to(dtype=x_dtype), sin.to(dtype=x_dtype)
        return cos, sin

# heavily adapated from Gemma3
def eager_attention_forward(
    module: nn.Module,
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
        softmax_scale = module.head_dim**-0.5

    query = query.transpose(1, 2) # (B, H, L, D)
    key = key.transpose(1, 2) # (B, H, L, D)
    value = value.transpose(1, 2) # (B, H, L, D)

    key = key.repeat_interleave(module.num_heads // module.num_key_value_heads, dim=1)
    value = value.repeat_interleave(module.num_heads // module.num_key_value_heads, dim=1)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * softmax_scale

    if softcap is not None:
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
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon)
            if not reuse_kv:
                self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonAttentionDynamicCache] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        window_size: Optional[int] = None,
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

        """if self.layer_idx < 10:
            print(f"Layer {self.layer_idx}. reuse kv: {self.reuse_kv}")
            #print(key_states[:, 0:10, 0, 0])
            print(hidden_states[:, -1, 0:5])
            print(key_states[:, -1, 0, 0])
"""
        # KV-cache.
        if not self.reuse_kv and cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # save k,v for next layer (*after* norm and RoPE and kv-cache update)
        if not self.reuse_kv:
            last_key_states, last_value_states = key_states, value_states

        # attention computation.
        if ATTN_IMPL == "eager":
            attention_interface = lambda q, k, v, **kw: eager_attention_forward(self, q, k, v, **kw)
        elif ATTN_IMPL == "fa2":
            attention_interface = lambda q, k, v, **kw: flash_attn_func(q, k, v, **kw)
        elif ATTN_IMPL == "fa3":
            attention_interface = lambda q, k, v, **kw: flash_attn_func(q, k, v, **kw)[0]
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        attn_output = attention_interface(
            query_states.bfloat16(),
            key_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            window_size=(min(window_size, self.window_size), 0) if window_size is not None else (self.window_size, 0),
            softcap=self.config.softcap_local_attn,
            softmax_scale=None if not self.config.use_uscaling else 1/self.head_dim,
            **kwargs,
        )

        return attn_output, last_key_states, last_value_states

# heavily adapted from official differential attention implementation
"""def eager_differential_attention_forward(
    module: nn.Module,
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
        softmax_scale = module.head_dim ** -0.5

    B, H2, Lq, Dh = query.shape # H2 = 2 * H
    H = module.num_heads
    Hkv = module.num_key_value_heads
    assert H2 == 2 * H, "query must have 2*num_heads heads"
    assert key.shape[-1] == Dh, "key head_dim must match query"
    assert value.shape[-1] == 2 * Dh, "value must have 2*head_dim"

    # repeat K to 2H (for the two "channels") and V to H (final combined heads)
    n_rep = H // Hkv
    k_2H = repeat_kv(key, 2 * n_rep) # [B, 2H, Lk, Dh]
    v_H  = repeat_kv(value, n_rep) # [B,  H, Lk, 2Dh]

    # raw attention logits for the 2 channels
    attn_weights = torch.matmul(query, k_2H.transpose(2, 3)) * softmax_scale  # [B, 2H, Lq, Lk]

    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap

    # masking (causal and/or sliding window)
    if causal or (window_size is not None):
        Lk = k_2H.size(2)
        i = torch.arange(Lq, device=attn_weights.device).unsqueeze(1)  # [Lq,1]
        j = torch.arange(Lk, device=attn_weights.device).unsqueeze(0)  # [1,Lk]
        allowed = torch.ones((Lq, Lk), dtype=torch.bool, device=attn_weights.device)
        if causal:
            allowed &= (j <= i)
        if window_size is not None:
            w_left, w_right = window_size
            if w_left is None:  w_left = Lk
            if w_right is None: w_right = Lk
            allowed &= (j >= i - w_left) & (j <= i + w_right)
        attn_weights = attn_weights.masked_fill(~allowed, float("-inf"))

    # softmax in fp32 then cast back
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype) # [B,2H,Lq,Lk]

    # reshape to [B, H, 2, Lq, Lk] and combine the two channels with learned lambda
    attn_probs = attn_probs.view(B, H, 2, Lq, -1)  # -1 = Lk



    # per-head scalar lambdas: exp(<λ_q1,λ_k1>) - exp(<λ_q2,λ_k2>) + λ_init
    lambda_1 = torch.exp(torch.sum(module.lambda_q1 * module.lambda_k1, dim=-1).float()).to(query.dtype)  # [H]
    lambda_2 = torch.exp(torch.sum(module.lambda_q2 * module.lambda_k2, dim=-1).float()).to(query.dtype)  # [H]
    lambda_full = (lambda_1 - lambda_2 + module.lambda_init).view(1, H, 1, 1)  # [1,H,1,1] for broadcast

    combined_probs = attn_probs[:, :, 0] - lambda_full * attn_probs[:, :, 1]  # [B,H,Lq,Lk]

    # weighted sum over V (note: V has 2*Dh per head)
    attn = torch.matmul(combined_probs, v_H)  # [B,H,Lq,2Dh]

    # sub-layer norm (or similar) then final scaling
    attn = module.subln(attn)
    attn = attn * (1 - module.lambda_init)

    # (B,Lq,H*2Dh)
    attn = attn.transpose(1, 2).contiguous().view(B, Lq, H * 2 * Dh)
    return attn"""

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
            self.q_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon)
            self.k_norm = DragonRMSNorm(self.head_dim, eps=config.norm_epsilon)

        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))

        self.register_buffer("lambda_init", torch.tensor(0.8 - 0.6 * math.exp(-0.3 * (layer_idx+1))))
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32).normal_(mean=0,std=0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonAttentionDynamicCache] = None,
        window_size: Optional[int] = None,
        **kwargs,
    ):
        # Q, K, V projections.
        query_states, key_states, value_states = get_query_key_value_tensors(self, hidden_states)
        value_states = value_states.reshape(value_states.size(0), value_states.size(1), value_states.size(2)//2, 2*value_states.size(3))

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # scalable softmax.
        start_pos = 0 if cache_params is None else cache_params.get_total_seen(self.layer_idx)
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            _, T, _, _ = query_states.shape
            pos = torch.arange(start_pos+1, start_pos+T+1, device=query_states.device).view(1, T, 1, 1).float()
            log_pos = pos.log() if window_size <= 0 else torch.clamp_max(pos, window_size).log()
            query_states = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * query_states
            # TODO: caching mechanism for log_pos

        # KV-cache.
        if cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # attention computation.
        # split q,k heads into two groups
        query1_states, query2_states = query_states[:, :, torch.arange(0, self.num_heads, 2)].contiguous(), query_states[:, :, torch.arange(1, self.num_heads, 2)].contiguous()
        key1_states, key2_states = key_states[:, :, torch.arange(0, self.num_key_value_heads, 2)].contiguous(), key_states[:, :, torch.arange(1, self.num_key_value_heads, 2)].contiguous()
        # compute
        if DIFF_ATTN_IMPL == "flex_head":
            y1 = flex_head_fa.flash_attn_func(
                query1_states.bfloat16(),
                key1_states.bfloat16(),
                value_states.bfloat16(),
                causal=True,
                window_size=(window_size, 0) if window_size is not None else None,
                softcap=self.softcap,
                softmax_scale=None if not self.config.use_uscaling else 1/self.head_dim)
            y2 = flex_head_fa.flash_attn_func(
                query2_states.bfloat16(),
                key2_states.bfloat16(),
                value_states.bfloat16(),
                causal=True,
                window_size=(window_size, 0) if window_size is not None else None,
                softcap=self.softcap,
                softmax_scale=None if not self.config.use_uscaling else 1/self.head_dim)
            lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1).float()) # (H)
            lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1).float()) # (H)
            lambda_full = (lambda_1 - lambda_2 + self.lambda_init).view(1, 1, -1, 1).type_as(y1)
            attn_output = (y1 - lambda_full * y2).contiguous()
        elif DIFF_ATTN_IMPL == "fa2":
            raise NotImplementedError()
        elif DIFF_ATTN_IMPL == "fa3":
            raise NotImplementedError()
        elif DIFF_ATTN_IMPL == "eager":
            raise NotImplementedError()

        return attn_output, None, None

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

        self.conv_size = config.conv_kernel
        self.conv_bias = config.use_bias

        self.n_heads = config.num_attention_heads
        self.n_heads_local = self.n_heads // 1
        self.d_head = int(config.hidden_size * (config.expand_factor/2)) // self.n_heads

        self.key_dim = self.n_heads * self.d_head
        self.value_dim = self.key_dim * config.expand_factor
        self.head_k_dim = self.d_head
        self.head_v_dim = self.d_head * config.expand_factor
        self.silu = nn.SiLU()

        self.dk = self.head_k_dim
        self.dv = self.head_v_dim
        self.per_head_proj = 2*self.dk + self.dv + 2 # [q k v b a] per head
        in_proj_dim_global = self.n_heads * self.per_head_proj
        self.in_proj = DragonLinear(config, config.hidden_size, in_proj_dim_global, bias=False)

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
        A = torch.empty(
            self.n_heads_local, dtype=torch.float32, device=torch.cuda.current_device()
        ).uniform_(*A_init_range)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        self.q_conv1d = DragonConv1D(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
            )
        self.k_conv1d = DragonConv1D(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
            )
        self.v_conv1d = DragonConv1D(
                hidden_size=self.value_dim,
                kernel_size=self.conv_size,
            )

        self.g_proj = DragonLinear(config, config.hidden_size, config.hidden_size*config.expand_factor, bias=False)
        self.act_func_gate = F.silu

    def forward(self,
                hidden_states: torch.Tensor,
                cache_params: Optional[HybridDragonAttentionDynamicCache] = None,
    ):
        _, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else 'chunk'
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        # input projection (TP-aware)
        qkvba = self.in_proj(hidden_states) # (l, b, H_local * per_head_proj)
        # [L,B,(H*P)] -> [B,L,H,P]
        qkvba = rearrange(qkvba, "b l (h p) -> b l h p", h=self.n_heads_local).contiguous()
        # split per head: [B,L,H,dk/dk/dv/1/1]
        q_proj = qkvba[..., 0:self.dk]
        k_proj = qkvba[..., self.dk:2*self.dk]
        v_proj = qkvba[..., 2*self.dk:2*self.dk+self.dv]
        b_proj = qkvba[..., 2*self.dk+self.dv:2*self.dk+self.dv+1]
        a_proj = qkvba[..., 2*self.dk+self.dv+1:]  
        # concat for conv
        q_proj = rearrange(q_proj, "b l h d -> b l (h d)")
        k_proj = rearrange(k_proj, "b l h d -> b l (h d)")
        v_proj = rearrange(v_proj, "b l h d -> b l (h d)")
        b_proj = rearrange(b_proj, "b l h d -> b l (h d)") # d=1
        a_proj = rearrange(a_proj, "b l h d -> b l (h d)")

        q_conv_cache, k_conv_cache, v_conv_cache, ssm_cache = (None, None, None, None)
        if cache_params is not None:
            q_conv_cache, k_conv_cache, v_conv_cache, ssm_cache = cache_params.get_ssm_cache(self.layer_idx)

        q, q_conv_cache = self.q_conv1d(
            x=q_proj,
            mask=None, 
            cache=q_conv_cache,
            output_final_state=(cache_params is not None))
        k, k_conv_cache = self.k_conv1d(
            x=k_proj,
            mask=None,
            cache=k_conv_cache,
            output_final_state=(cache_params is not None))
        v, v_conv_cache = self.v_conv1d(
            x=v_proj,
            mask=None,
            cache=v_conv_cache,
            output_final_state=(cache_params is not None))

        # back to per-head for kernels
        q = rearrange(q, "b l (h d) -> b l h d", d=self.dk)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.dk)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.dv)

        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a_proj.float() + self.dt_bias)

        if mode == 'chunk':
            if chunk_gated_delta_rule is not None:
                o, ssm_cache = chunk_gated_delta_rule(
                    q=q.bfloat16(),
                    k=k.bfloat16(),
                    v=v.bfloat16(),
                    g=g,
                    beta=beta,
                    scale=None if not self.config.use_uscaling else 1/self.head_k_dim,
                    initial_state=ssm_cache,
                    output_final_state=(cache_params is not None),
                    cu_seqlens=None, # for varlen training
                    head_first=False,
                    use_qk_l2norm_in_kernel=True
                ) # (B L H D) where d is head_v_dim
            else:
                raise NotImplementedError("PyTorch implementation of chunked GDN is not available.")
        elif mode == 'fused_recurrent':
            if fused_recurrent_gated_delta_rule is not None:
                o, ssm_cache = fused_recurrent_gated_delta_rule(
                    q=q.bfloat16(),
                    k=k.bfloat16(),
                    v=v.bfloat16(),
                    g=g,
                    beta=beta,
                    scale=None if not self.config.use_uscaling else 1/self.head_k_dim,
                    initial_state=ssm_cache,
                    output_final_state=(cache_params is not None),
                    cu_seqlens=None,
                    use_qk_l2norm_in_kernel=True
                ) # (B L H D) where d is head_v_dim
            else:
                raise NotImplementedError("PyTorch implementation of recurrent GDN is not available.")
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        g = self.g_proj(hidden_states).view(o.size(0), o.size(1), o.size(2), o.size(3)) # (B, L, H, D)
        o = o * self.act_func_gate(g)

        if cache_params is not None:
            cache_params.update_ssm_cache(
                q_conv_states=q_conv_cache,
                k_conv_states=k_conv_cache,
                v_conv_states=v_conv_cache,
                ssm_states=ssm_cache,
                layer_idx=self.layer_idx,
            )

        return o

class DragonMLP(nn.Module):
    def __init__(self, config: DragonConfig):
        super().__init__()
        self.fc_1 = DragonLinear(config, config.hidden_size, config.intermediate_size, bias=False)
        self.fc_2 = DragonLinear(config, config.intermediate_size, config.hidden_size, bias=False)
        self.register_buffer("_2_sqrt_5", torch.tensor(2/math.sqrt(5)) if config.use_uscaling else torch.tensor(1.))

    def forward(self, hidden_states):
        hidden_states = self.fc_1(hidden_states)
        hidden_states = self._2_sqrt_5 * F.relu(hidden_states).square()
        hidden_states = self.fc_2(hidden_states)
        return hidden_states

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
            self.attn_group_norm = DragonHeadWiseRMSNorm(n_heads=self.attn.num_heads//2, d_head=2*self.attn.head_dim, eps=config.norm_epsilon)
        else:
            self.attn_group_norm = DragonHeadWiseRMSNorm(n_heads=self.attn.num_heads, d_head=self.attn.head_dim, eps=config.norm_epsilon)
        self.lin_attn_group_norm = DragonHeadWiseRMSNorm(n_heads=self.lin_attn.n_heads, d_head=self.lin_attn.head_v_dim, eps=config.norm_epsilon)

        self.input_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon)
        self.postmixer_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon)
        self.mlp = DragonMLP(config)

        self.register_buffer("lns", torch.tensor(1.0) if config.use_uscaling else torch.tensor(1. / math.sqrt(layer_idx+1)))
        self.register_buffer("sqrt_2_2", torch.tensor(math.sqrt(2)/2) if config.use_uscaling else torch.tensor(1/2))
        self.register_buffer("sqrt_tau", torch.sqrt(torch.tensor(self.config.uscaling_tau)) if config.use_uscaling else torch.tensor(1.0))
        self.register_buffer("sqrt_one_minus_tau", torch.sqrt(torch.tensor(1.0 - self.config.uscaling_tau)) if config.use_uscaling else torch.tensor(1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridDragonAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_value_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        window_size: Optional[int] = None,
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
            window_size=window_size,
        ) # (B, L, E*D)
        y_lin_attn = self.lin_attn(
            hidden_states=hidden_states,
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

    def _init_weights(self, module):
        if isinstance(module, (DragonLinear, DragonConv1D)):
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
        cache_params (`HybridDragonAttentionDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
            Includes both the RNN-like state matrices after the selective scan, and the conv states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridDragonAttentionDynamicCache] = None
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
        cache_params (`HybridDragonAttentionDynamicCache`):
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
    past_key_values: Optional[HybridDragonAttentionDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class DragonModel(DragonPreTrainedModel):
    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DragonBlock(config, layer_idx=i, layer_type=layer) for i, layer in enumerate(config.layers_config)])

        self.rotary_emb = DragonRotaryEmbedding(config, head_dim=(config.expand_factor*config.hidden_size)//config.num_attention_heads) # only for SWA
        self.final_norm = DragonRMSNorm(config.hidden_size, eps=config.norm_epsilon)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[HybridDragonAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> DragonOutput:
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache:
            bsz = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
            if past_key_values is None:
                past_key_values = HybridDragonAttentionDynamicCache(self.config, bsz, dtype=self.dtype, device=self.device)
            elif not isinstance(past_key_values, HybridDragonAttentionDynamicCache):
                # recreate (todo: upcast instead of recreate)
                if type(past_key_values) is DynamicCache:
                    print("upgrading DynamicCache → HybridDragonAttentionDynamicCache")
                    past_key_values = HybridDragonAttentionDynamicCache(self.config, bsz, dtype=self.dtype, device=self.device)
                else:
                    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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
                window_size=window_size,
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

class DragonForCausalLM(DragonPreTrainedModel, GenerationMixin):
    def __init__(self, config: DragonConfig):
        super().__init__(config)
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
        past_key_values: Optional[HybridDragonAttentionDynamicCache] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        **kwargs,
    ) -> DragonCausalLMOutput:
        """
        print("fwd")
        print(past_key_values is None)
        if past_key_values is not None:
            print(type(past_key_values))
            print(past_key_values)
        """

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
            window_size=window_size,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(logits.device)
            # shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # compute loss
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=self.model.padding_idx, reduction='none')

        return DragonCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

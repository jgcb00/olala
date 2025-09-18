# coding=utf-8
"""Dragon model configuration"""
# TODO : TP (cf qwen)
# TODO : init

import re

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

#@register_for_auto_class("AutoConfig")
class DragonConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DragonModel`]. It is used to instantiate a
    Dragon model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Dragon model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DragonModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        mlp_hidden_act (`str`, *optional*, defaults to "relu2"):
            The non-linear activation function in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the model.
        initializer_range (`float`, *optional*, defaults to 0.006):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        residual_in_fp32 (`bool`, *optional*, defaults to `False`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        sliding_window_size (`int`, *optional*, defaults to 1024):
            Sliding window attention window size.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the mamba intermediate size.
        mamba_hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in the Mamba layers.
        mamba_dt_min (`float`, *optional*, defaults to 0.001):
            Minimum value for the time step in Mamba.
        mamba_dt_max (`float`, *optional*, defaults to 0.1):
            Maximum value for the time step in Mamba.
        mamba_dt_limit (`tuple`, *optional*, defaults to (0.0, float("inf"))):
            Limits for the time step in Mamba.
        mamba_dt_init_floor (`float`, *optional*, defaults to 1e-4):
            Floor value for time step initialization in Mamba.
    """

    model_type = "dragon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        tie_word_embeddings=False,
        max_position_embeddings=8192,
        use_uscaling=True,
        hidden_size=2048,
        intermediate_size=8192,
        expand_factor=2,
        layers_config=4*"lrdlr",
        num_attention_heads=32,
        num_key_value_heads=8,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.006,
        softcap_local_attn=0.0,
        softcap_global_attn=150.0,
        norm_epsilon=1e-6,
        residual_in_fp32=False,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window_size=1024,
        slw_wsize=-1,
        rope_theta_local=163.,
        uscaling_tau=0.2,
        attention_dropout=0.,
        hidden_dropout=0.,
        gdn_d_conv=4,
        gdn_dt_min=0.001,
        gdn_dt_max=0.1,
        gdn_dt_init_floor=1e-4,
        gdn_A_init_range=(1, 16),
        old_lns=False,
        **kwargs,
    ):

        self.rope_theta = rope_theta_local
        self.qk_norm = True
        self.softcap_local_attn=softcap_local_attn
        self.softcap_global_attn=softcap_global_attn
        self.use_uscaling = use_uscaling
        self.uscaling_tau = uscaling_tau
        self.scalable_softmax = True

        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expand_factor = expand_factor
        self.layers_config = layers_config
        self.num_hidden_layers = len(layers_config)
        self.num_attention_heads = num_attention_heads
        self.sliding_window_size = sliding_window_size
        self.slw_wsize = slw_wsize
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.norm_epsilon = norm_epsilon
        self.residual_in_fp32 = residual_in_fp32

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.conv_kernel = gdn_d_conv
        self.time_step_min = gdn_dt_min
        self.time_step_max = gdn_dt_max
        self.time_step_floor = gdn_dt_init_floor
        self.A_init_range = gdn_A_init_range

        self.old_lns = old_lns

        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.num_attention_heads % 2 == 0, "Number of attention heads must be even for differential attention."
        assert self.num_key_value_heads % 2 == 0, "Number of kv heads must be even for differential attention."

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # TODO: better way to handle those?
        self.auto_map = dict(getattr(self, "auto_map", {}))
        self.auto_map.setdefault("AutoConfig", "configuration_dragon.DragonConfig")
        self.auto_map.setdefault("AutoModel", "modeling_dragon.DragonModel")
        self.auto_map.setdefault("AutoModelForCausalLM", "modeling_dragon.DragonForCausalLM")

DragonConfig.register_for_auto_class("AutoConfig")
__all__ = ["DragonConfig"]
# todo : update docstrings
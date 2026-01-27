# coding=utf-8
"""Dragon model configuration"""
# TODO : TP (cf qwen)
# TODO : init

from typing import Optional
import re

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

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
        base_depth: int = 0,
        completed_p_alpha: float = 0.5,
        use_completed_p: bool = False,
        layers_stem_config: str = "",
        layers_mlp_config: str = "",
        layers_ve_config: str = "",
        use_value_embedding: bool = False,
        reduce_lm_head: int = 0,
        vwn: bool = False,
        vwn_m: int = 2,
        vwn_n: int = 3,
        vwn_wd_alpha_beta: bool = False,
        vwn_dynamic: bool = True,
        legacy_gate: bool = False,
        tie_lm_head: bool = False,
        mlp_type: str = "simple",
        layer_norm_scaling: bool = False,
        mamba_d_state: int = 128,
        mamba_headdim: int = 64,
        mamba3_rope: bool = True,
        mamba3_remove_BC_bias: bool = False,
        mamba3_is_id_rms: bool = True,
        mamba3_remove_conv: bool = True,
        mamba3_is_A_dd: bool = True,
        mamba3_add_trapezoid: bool = True,
        mamba3_postgate_norm: bool = False,
        moe: bool = False,
        moe_router_type: str = "classic",
        moe_num_routed_experts: int = 2,
        moe_num_active_experts: int = 1,
        moe_routed_scaling_factor: float = 2.5,
        moe_routed_intermediate_size: int = 768,
        moe_shared_intermediate_size: int = 768,
        moe_routed_input_dim: int = 384,
        intra_doc_masking: bool = False,
        seednorm_rank: int = 1,
        seednorm_type: int = 1,
        final_norm: bool = True,
        mla_kv_rank: int = 128,
        shrink_qk_da: int = 2,
        shrink_qk_gdn: int = 2,
        mixer_gn: bool = True,
        gate_before_norm: bool = True,
        kda_allow_neg_eigval: bool = False,
        kda_num_v_heads: Optional[int] = None,
        seednorm_wd: bool = True,
        normalization_type: str = "rmsnorm",
        tpa_rank: int = 2,
        num_signal_heads_diff: Optional[int] = None,
        scalar_proj_as_hidden_matrix: bool = True,
        token_shift_attn: bool = False,
        token_shift_gdn: bool = False,
        token_conv1d_attn: bool = False,
        token_conv1d_gdn: bool = True,
        patch_level_training: bool = False,
        patch_level_training_size: int = 4,
        nsa_topk: int = 16,
        nsa_block_size: int = 64,
        nsa_window_size: int = 512,
        cca_seq_kernel_size: int = 4,
        rope_gdn: str = None,
        zero_centered_gate: bool = False,
        scalable_softmax: bool = True,
        resformer: bool = False,
        mamba_mimo_dim : int = 4,
        mamba_ngroups : int = 1,
        gate_type: str = "elementwise",
        gate_act: str = "silu",
        gate_attn: bool = False,
        gate_gdn: bool = True,
        head_dim_gdn: Optional[int] = None,
        num_attention_heads_gdn: int = 32,
        num_key_value_heads_gdn: int = None,
        fused_loss_computation=False,
        qk_norm=True,
        num_attention_heads_indexer=8,
        head_dim_indexer=32,
        dsa_q_lora_rank=128,
        dsa_topk=512,
        zero_centered_gamma=False,
        vocab_size=151936,
        tie_word_embeddings=False,
        max_position_embeddings=8192,
        use_uscaling=False,
        hidden_size=2048,
        intermediate_size=8192,
        expand_factor=2,
        layers_config=4*"lrdlr",
        head_dim=128,
        num_attention_heads=32,
        num_key_value_heads=8,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.006,
        softcap_attn=0.0,
        norm_epsilon=1e-6,
        residual_in_fp32=False,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window_size=1024,
        slw_wsize=-1,
        rope_type="",
        rope_theta=0.,
        uscaling_tau=0.2,
        attention_dropout=0.,
        hidden_dropout=0.,
        gdn_d_conv=4,
        gdn_dt_min=0.001,
        gdn_dt_max=0.1,
        gdn_dt_init_floor=1e-4,
        gdn_A_init_range=(1, 16),
        old_lns=False,
        mlp_linking=False,
        **kwargs,
    ):
        self.base_depth = base_depth
        self.completed_p_alpha = completed_p_alpha
        self.use_completed_p = use_completed_p
        self.layers_stem_config = layers_stem_config
        self.layers_mlp_config = layers_mlp_config
        self.layers_ve_config = layers_ve_config
        self.use_value_embedding = use_value_embedding
        self.reduce_lm_head = reduce_lm_head
        self.vwn = vwn
        self.vwn_m = vwn_m
        self.vwn_n = vwn_n
        self.vwn_wd_alpha_beta = vwn_wd_alpha_beta
        self.vwn_dynamic = vwn_dynamic
        self.legacy_gate = legacy_gate
        self.tie_lm_head = tie_lm_head
        self.mlp_type = mlp_type
        self.layer_norm_scaling = layer_norm_scaling
        self.mamba_d_state = mamba_d_state
        self.mamba_headdim = mamba_headdim
        self.mamba3_rope = mamba3_rope
        self.mamba3_remove_BC_bias = mamba3_remove_BC_bias
        self.mamba3_is_id_rms = mamba3_is_id_rms
        self.mamba3_remove_conv = mamba3_remove_conv
        self.mamba3_is_A_dd = mamba3_is_A_dd
        self.mamba3_add_trapezoid = mamba3_add_trapezoid
        self.mamba3_postgate_norm = mamba3_postgate_norm
        self.moe = moe
        self.moe_router_type = moe_router_type
        self.moe_num_active_experts = moe_num_active_experts
        self.moe_num_routed_experts = moe_num_routed_experts
        self.moe_routed_scaling_factor = moe_routed_scaling_factor
        self.moe_routed_intermediate_size = moe_routed_intermediate_size
        self.moe_shared_intermediate_size = moe_shared_intermediate_size
        self.moe_routed_input_dim = moe_routed_input_dim
        self.intra_doc_masking = intra_doc_masking
        self.seednorm_rank = seednorm_rank
        self.seednorm_type = seednorm_type
        self.final_norm = final_norm
        self.mla_kv_rank = mla_kv_rank
        self.shrink_qk_da = shrink_qk_da
        self.shrink_qk_gdn = shrink_qk_gdn
        self.mixer_gn = mixer_gn
        self.gate_before_norm = gate_before_norm
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.kda_num_v_heads = kda_num_v_heads
        self.seednorm_wd = seednorm_wd
        self.normalization_type = normalization_type
        self.tpa_rank = tpa_rank
        self.num_signal_heads_diff = num_signal_heads_diff
        self.scalar_proj_as_hidden_matrix = scalar_proj_as_hidden_matrix
        self.token_shift_attn = token_shift_attn
        self.token_shift_gdn = token_shift_gdn
        self.token_conv1d_attn = token_conv1d_attn
        self.token_conv1d_gdn = token_conv1d_gdn
        self.patch_level_training = patch_level_training
        self.patch_level_training_size = patch_level_training_size
        self.nsa_topk = nsa_topk
        self.nsa_block_size = nsa_block_size
        self.nsa_window_size = nsa_window_size
        self.cca_seq_kernel_size = cca_seq_kernel_size
        self.rope_gdn = rope_gdn
        self.zero_centered_gate = zero_centered_gate
        self.gate_type = gate_type
        self.gate_act = gate_act
        self.gate_attn = gate_attn
        self.gate_gdn = gate_gdn
        self.head_dim = head_dim
        self.head_dim_gdn = head_dim_gdn
        self.num_attention_heads_gdn = num_attention_heads_gdn
        if num_key_value_heads_gdn is None:
            num_key_value_heads_gdn = num_attention_heads_gdn
        self.num_key_value_heads_gdn = num_key_value_heads_gdn
        self.fused_loss_computation = fused_loss_computation
        self.num_attention_heads_indexer = num_attention_heads_indexer
        self.head_dim_indexer = head_dim_indexer
        self.dsa_q_lora_rank = dsa_q_lora_rank
        self.dsa_topk = dsa_topk
        self.zero_centered_gamma = zero_centered_gamma
        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.softcap_attn = softcap_attn
        self.use_uscaling = use_uscaling
        self.uscaling_tau = uscaling_tau
        self.scalable_softmax = scalable_softmax
        self.resformer = resformer
        self.mamba_mimo_dim = mamba_mimo_dim
        self.mamba_ngroups = mamba_ngroups

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
        
        self.mlp_linking = mlp_linking

        #assert self.hidden_size % self.num_attention_heads == 0
        #assert self.num_attention_heads % self.num_key_value_heads == 0
        #assert self.num_attention_heads % 2 == 0, "Number of attention heads must be even for differential attention."
        #assert self.num_key_value_heads % 2 == 0, "Number of kv heads must be even for differential attention."

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_lm_head,
            **kwargs,
        )
        # TODO: better way to handle those?
        self.auto_map = dict(getattr(self, "auto_map", {}))
        self.auto_map.setdefault("AutoConfig", "configuration_dragon.DragonConfig")
        self.auto_map.setdefault("AutoModel", "modeling_dragon.DragonModel")
        self.auto_map.setdefault("AutoModelForCausalLM", "modeling_dragon.DragonForCausalLM")

DragonConfig.register_for_auto_class("AutoConfig")
__all__ = ["DragonConfig"]
# todo : update docstrings, arg orders
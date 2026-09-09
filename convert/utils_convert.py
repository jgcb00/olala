import sys
import os
# Megatron-LM must be importable. Overridable so this tree is not pinned to
# one host's layout; the default is the path it was developed against.
sys.path.append(os.environ.get(
    "MEGATRON_LM_DIR",
    "/data/home/gaetan.caillaut/dragon-sft/7A1B/training/Megatron-LM"))

import pickle
import types
from typing import Union
import contextlib
from types import SimpleNamespace
import random
import numpy as np
from einops import rearrange

import torch

import torch.distributed.distributed_c10d as c10d
from torch.distributed.checkpoint import FileSystemReader

from megatron.core import parallel_state
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.training.checkpointing import find_checkpoint_rank_0, get_checkpoint_name, _get_checkpoint_format
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.transformer.module import Float16Module
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
from megatron.core.msc_utils import MultiStorageClientFeature, open_file
from megatron.core.utils import unwrap_model

from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.models.dragon.dragon_layer_specs import get_dragon_block_spec
from megatron.core.models.dragon.dragon_model import DragonModel
from megatron.core.activations import squared_relu

# The model code comes from the repo root this file lives in -- ONE copy, so
# what the converter exports is exactly what the repo ships. _olala_pkg binds
# that directory to the name `olala` (see its docstring for why an ordinary
# import cannot).
import _olala_pkg  # noqa: F401  (registers the `olala` package)

from olala.configuration_olala import OlalaConfig as OlalaConfigHF
from olala.modeling_olala import OlalaForCausalLM

def load_random_mg_models(L, artificial_seq_len, pgs, tp_size):
    hidden_size = 1024
    idm = True
    layers_mixer_config = 'MMVMM' # gggTggggggTggggggTggg
    completedp = True
    lns = False
    use_ddl = False
    init_std = 1e-2
    vocab_size = 151936

    num_routed_total_experts = 256
    num_routed_active_experts = 6

    layers_mixer_config_base = 'MMVMM'
    hidden_size_base = 1024

    config_mg = DragonConfig(
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size>1,
        use_lns=lns,
        use_completedp=completedp,
        completedp_alpha=0.5,
        layers_mixer_config=layers_mixer_config,
        num_layers=len(layers_mixer_config),
        layers_mixer_config_base=layers_mixer_config_base,
        hidden_size_base=hidden_size_base,
        hidden_size=hidden_size,
        num_attention_heads=32,
        num_signal_heads=24,
        gate_attn=True,
        gate_gdn=True,
        ffn_hidden_size=4096,
        kv_channels=128,
        layernorm_zero_centered_gamma=True,
        softcap_attn=150.,
        qk_layernorm=True,
        scalable_softmax=True,
        linear_attention_type='gated_delta_net',
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_attention_freq=[1, 1, 1, 1, 0, 1, 1, 1, 1],
        token_shift=True,
        tpa_rank=4,
        init_method_std=init_std,
        init_method_embedding_std=init_std,
        init_method_output_std=init_std,
        layernorm_epsilon=1e-6,
        training_sequence_length=1024,
        intra_doc_masking=idm,
        num_moe_experts=num_routed_total_experts,
        moe_router_topk=num_routed_active_experts,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_ffn_hidden_size=4096,
        moe_shared_expert_intermediate_size=1024,
        moe_shared_expert_gate=False,# TEMP #True,
        moe_shared_expert_overlap=False,
        moe_router_load_balancing_type='seq_aux_loss',
        moe_router_pre_softmax=True,
        moe_router_dtype='fp32',
        moe_router_topk_scaling_factor=1.5,
        moe_router_score_function='sigmoid',
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_routed_input_dim=hidden_size//4,
        moe_grouped_gemm=True,
        moe_router_fusion=False, # TEMP True, # dont when using uscaling
        moe_aux_loss_coeff=1e-4, # per DSV3 paper. note: does this correctly apply to the seq-level loss?
        moe_token_dropping=False, # useless. it's the argument below that matters.
        moe_expert_capacity_factor=None,
        moe_token_drop_policy='probs',
        moe_pad_expert_input_to_capacity=False,
        moe_apply_probs_on_input=False,
        activation_func=squared_relu,
        use_te_activation_func=True,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=1,
        use_ddl=use_ddl,
        mixer_gn=False,
        bf16=True,
        use_geodesic_norm=True,
        normalize_embeddings=True,
        normalize_lm_head=False,
        artificial_seq_len=artificial_seq_len,
    )

    model_mg = DragonModel(
        config_mg,
        get_dragon_block_spec(config_mg),
        vocab_size=vocab_size,
        max_sequence_length=L,
        parallel_output=True,
        pg_collection=pgs,
    ).cuda()
    
    return model_mg, config_mg, vocab_size, 512 

def generate_state_dict(
    args,
    model,
    optimizer,
    opt_param_scheduler,
    rng_state,
    iteration=None,
    optim_sd_kwargs=None,
    model_sd_kwargs=None,
    rerun_state=None,
    wandb_id=None,
    wandb_step=None,
):
    """Generate a state dict from given model, optimizer, scheduler, rng state and others. """

    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['wsize'] = opt_param_scheduler.get_wsize() if opt_param_scheduler is not None else None
    state_dict['wandb_id'] = wandb_id
    state_dict['wandb_step'] = wandb_step
    if iteration is not None:
        state_dict['iteration'] = iteration
    
    len_model = 1 if not hasattr(model, '__len__') else len(model)
    for i in range(len_model):
        key = "model"
        if len_model > 1:
            key = f"model{i}"

        model_sd = model[i].sharded_state_dict(**(model_sd_kwargs or {})) if len_model > 1 else model.sharded_state_dict(**(model_sd_kwargs or {}))
        state_dict[key] = model_sd

    # Rerun state
    if rerun_state:
        state_dict['rerun_state_machine'] = rerun_state

    # RNG states.
    if not args.no_save_rng and rng_state:
        state_dict["rng_state"] = rng_state

    return state_dict

def _load_base_checkpoint_base(load_dir, iteration, rank0=False, sharded_state_dict=None):
    release = False
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    ckpt_format = "torch_dist"

    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
        return state_dict, checkpoint_name

    #load_strategy = get_default_load_sharded_strategy(checkpoint_name)
    #load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, mpu.get_data_parallel_group(with_context_parallel=True))
    state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name)#, load_strategy)#, strict=args.dist_ckpt_strictness)
    return state_dict, checkpoint_name

def load_checkpoint_base(load_dir, iteration):
    state_dict, _ = _load_base_checkpoint_base(load_dir, iteration, rank0=True)
    return state_dict

def get_rng_state(args, ckpt_format: str):
    """Collect rng state across data parallel ranks."""
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    rng_state_list = None
    if args.data_parallel_random_init and torch.distributed.is_initialized() and \
            mpu.get_data_parallel_world_size() > 1:
        rng_state_list = \
            [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    rng_state_list = ShardedObject('rng_state', rng_state_list, (pp_size, tp_size), (pp_rank, tp_rank), replica_id=mpu.get_data_parallel_rank(with_context_parallel=True))
    return rng_state_list

def _load_global_dist_base_checkpoint(load_dir, args, rank0, sharded_state_dict, iteration, release, checkpointing_context=None):
    """ Load the base state_dict from the given directory containing the global distributed checkpoint """
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
        return state_dict, checkpoint_name, release

    if sharded_state_dict is None:
        assert not args.auto_detect_ckpt_format and not args.use_dist_ckpt, (
            args.auto_detect_ckpt_format,
            args.use_dist_ckpt,
        )
        raise RuntimeError(
            'Detected load from a distributed checkpoint, but neither --use-dist-ckpt nor --auto-detect-ckpt-format is set.'
        )

    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    load_strategy = get_default_load_sharded_strategy(checkpoint_name)
    # NOTE: `args.ckpt_fully_parallel_load` applies to both persistent and non-persistent checkpoints.
    if args.ckpt_fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )
    if checkpointing_context is not None:
        checkpointing_context["load_strategy"] = load_strategy

    state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name, load_strategy, strict=args.dist_ckpt_strictness)
    return state_dict, checkpoint_name, release

def _get_non_persistent_iteration(non_persistent_global_dir, args, checkpointing_context=None):
    if args.non_persistent_ckpt_type is None:
        return -1
    elif args.non_persistent_ckpt_type == "global":
        tracker_filename = get_checkpoint_tracker_filename(non_persistent_global_dir)
        if isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)
            if release:
                raise RuntimeError('Non-persistent checkpoint can\'t be a release checkpoint')
        else:
            iteration = -1
            print('WARNING: could not find the metadata file {}'.format(tracker_filename))
            print('    will not load any non-persistent checkpoint')
        return iteration
    elif args.non_persistent_ckpt_type == "local":
        return checkpointing_context['local_checkpoint_manager'].find_latest()
    else:
        assert False, 'Please use local or global non-persistent checkpoints' \
            f'(got: {args.non_persistent_ckpt_type})'
            
def get_checkpoint_tracker_filename(checkpoints_path):

    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')

def isfile(filename) -> bool:
    return os.path.isfile(filename)

def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False

    with open_file(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device='cuda')
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        # We should now have all the same iteration.
        # If not, print a warning and chose the maximum
        # iteration across all ranks.
        if iteration != max_iter:
            rank = torch.distributed.get_rank()
            print('WARNING: on rank {} found iteration {} in the '
                  'metadata while max iteration across the ranks '
                  'is {}, replacing it with max iteration.'.format(
                      rank, iteration, max_iter), flush=True)
    else:
        # When loading a checkpoint outside of training (for example,
        # when editing it), we might not have torch distributed
        # initialized, in this case, just assume we have the latest
        max_iter = iteration
    return max_iter, release

def _load_base_checkpoint(
    load_dir,
    args,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Try to load non-persistent checkpoint first
    non_persistent_global_dir = (
        args.non_persistent_global_ckpt_dir
        if args.non_persistent_global_ckpt_dir or load_dir is None
        else os.path.join(load_dir, 'non_persistent')
    )
    non_persistent_iteration = _get_non_persistent_iteration(
        non_persistent_global_dir, args, checkpointing_context
    )
    iteration, release = -1, False
    tracker_filename = 'because load directory is not defined'
    if load_dir is not None:
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)

    # Allow user to specify the loaded iteration.
    if getattr(args, "ckpt_step", None):
        iteration = args.ckpt_step

    if non_persistent_iteration != -1:  # there is a non-persistent checkpoint
        if non_persistent_iteration >= iteration:
            return _load_non_persistent_base_checkpoint(
                non_persistent_global_dir,
                args,
                rank0,
                sharded_state_dict,
                non_persistent_iteration,
                checkpointing_context,
            )
        else:
            print('WARNING: non-persistent checkpoints are older than persistent checkpoint')

    # Otherwise we are dealing with global checkpoints
    # If no tracker file, return nothing
    if iteration == -1:
        if not rank0:
            print('WARNING: could not find the metadata file {}'.format(tracker_filename))
            print('    will not load any checkpoints and will start from random')
        # Conditionally exit if checkpoint not found.
        if args.exit_on_missing_checkpoint:
            print(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False, None

    # Determine the type of the checkpoint on disk.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    ckpt_format = _get_checkpoint_format(checkpoint_name, args)

    if not rank0:
        dist_infix = "distributed " if ckpt_format == "torch_dist" else ""
        if release:
            print(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print(
                f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}'
            )

    return _load_global_dist_base_checkpoint(load_dir, args, rank0, sharded_state_dict, iteration, release, checkpointing_context=checkpointing_context)

def load_checkpoint(args, iteration, ddp_model, load_dir, checkpointing_context=None):
    """Load a model checkpoint and return the iteration."""
    strict=True

    model = unwrap_model(ddp_model)

    ckpt_format = args.ckpt_format
    assert ckpt_format == "torch_dist"
    if args.auto_detect_ckpt_format or ckpt_format == "torch_dist":
        state_dict, checkpoint_name = _load_base_checkpoint_base(
            load_dir,
            iteration,
            rank0=True,
        )
        release = False
        ckpt_type = None

    load_kwargs = {}
    ignore_rng_state = False
    ignore_rerun_state = True
    if ckpt_format == "torch_dist":
        state_dict_args = (
            state_dict.get('args', SimpleNamespace())
            if state_dict is not None
            else SimpleNamespace()
        )
        if not hasattr(state_dict_args, 'tensor_model_parallel_size'):
            print('WARNING: does not find TP size in checkpoint args, using 1 as default.')
        if not hasattr(state_dict_args, 'pipeline_model_parallel_size'):
            print('WARNING: does not find PP size in checkpoint args, using 1 as default.')
        ckpt_tp_pp = (
            getattr(state_dict_args, 'tensor_model_parallel_size', 1),
            getattr(state_dict_args, 'pipeline_model_parallel_size', 1),
        )
        run_tp_pp = (
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
        )

        ckpt_world_size = getattr(state_dict_args, 'world_size', 0)
        run_world_size = getattr(args, 'world_size', 0)
        ckpt_dp = getattr(state_dict_args, 'data_parallel_size', 0)
        run_dp = getattr(args, 'data_parallel_size', 0)
        mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(
            run_tp_pp, ckpt_tp_pp
        )

        # Determine if RNG state will be loaded
        ignore_rng_state = True
        gen_sd_rng_state = None

        sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
        print(f'sharded_state_dict metadata loaded from the checkpoint: {sharded_sd_metadata}')
        
        gen_sd_optim = None
        gen_sd_opt_param_scheduler = None
        optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
        model_sd_kwargs = dict(metadata=sharded_sd_metadata)

        # Determine if rerun state will be loaded
        gen_sd_rerun_state = None
        if (
            ckpt_world_size == run_world_size
            and ckpt_tp_pp == run_tp_pp
            and ckpt_dp == run_dp
            and not release
            and not args.finetune
            and 'rerun_state_machine' in state_dict
        ):
            rerun_state_machine = get_rerun_state_machine()
            if rerun_state_machine.validate_state_dict(state_dict['rerun_state_machine']):
                gen_sd_rerun_state = rerun_state_machine.state_dict(
                    data_iterator=None, ckpt_format=ckpt_format, force=True,
                )
                ignore_rerun_state = False
        if (
            ckpt_world_size != run_world_size
            or ckpt_tp_pp != run_tp_pp
            or ckpt_dp != run_dp
        ):
            print("Job sharding has changed: Rerun state will be ignored")

        # [ModelOpt]: Initial loading from non-resume sharded checkpoint to a Distillation Model
        # will result in key mismatch with loss modules potentially containing parameters, since
        # it requires generating a state_dict before loading. Here we hide those modules if present.
        with contextlib.ExitStack() as stack:  # Allows multiple context managers for each model shard
            if args.finetune and hasattr(model[0], "hide_loss_modules"):
                for m in model:
                    stack.enter_context(m.hide_loss_modules())
            load_kwargs['sharded_state_dict'] = generate_state_dict(
                args, model, gen_sd_optim, gen_sd_opt_param_scheduler, gen_sd_rng_state,
                optim_sd_kwargs=optim_sd_kwargs, model_sd_kwargs=model_sd_kwargs,
                rerun_state=gen_sd_rerun_state
            )

    state_dict, checkpoint_name, release = _load_base_checkpoint(
        load_dir, args, rank0=False, checkpointing_context=checkpointing_context,
        **load_kwargs
    )

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                sys.exit()
    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        args.consumed_train_samples = getattr(checkpoint_args, 'consumed_train_samples', 0)
        args.skipped_train_samples = getattr(checkpoint_args, 'skipped_train_samples', 0)
        args.consumed_valid_samples = getattr(checkpoint_args, 'consumed_valid_samples', 0)
    else:
        print('could not find arguments in the checkpoint ...')

    def load_model_state_dict(module, state_dict, strict: bool):
        """Helper function to load state dict with fallback for missing extra states."""
        try:
            module.load_state_dict(state_dict, strict=strict)
        except Exception as e:
            if strict:
                # Fallback support for backward compatibility breaking changes in TransformerEngine
                load_return = module.load_state_dict(state_dict, strict=False)
                print(f"load_return: {load_return}")
    # Model.
    len_model = 1 if not hasattr(model, '__len__') else len(model)
    if len_model == 1:
        load_model_state_dict(ddp_model, state_dict['model'], strict)
    else:
        for i in range(len_model):
            # If there is no corresponding model in the state_dict, it will be ignored.
            # It means that this is an empty stage.
            if 'model%d' % i not in state_dict:
                continue
            load_model_state_dict(ddp_model[i], state_dict['model%d' % i], strict)

    # rerun state
    if not ignore_rerun_state:
        try:
            if 'rerun_state_machine' in state_dict:
                get_rerun_state_machine().load_state_dict(state_dict['rerun_state_machine'])
        except Exception as e:
            print(f"Unable to restore RerunMachine from checkpoint: {e}. Skipping.")

    # rng states.
    if not release and not args.finetune and not args.no_load_rng and not ignore_rng_state:
        try:
            if 'rng_state' in state_dict:
                rng_state = state_dict['rng_state']

                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = rng_state[mpu.get_data_parallel_rank()]
                else:
                    rng_state = rng_state[0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
        except KeyError:
            print('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    return iteration, num_floating_point_operations_so_far

def build_dragon_config(sd, params_dtype=torch.float32):
    """Rebuild the training-time DragonConfig from a checkpoint's common state dict.

    Pure CPU / pure Python: `sd` is what load_checkpoint_base() returns (the
    pickled `args` namespace), so this needs neither a GPU nor an instantiated
    model. Both the GPU path (_load_single_mg_model) and the CPU-only path
    (mg_cpu_loader.load_mg_weights_cpu) go through here, so there is exactly one
    place where checkpoint args are mapped onto config fields.
    """
    config_mg = DragonConfig(
        tensor_model_parallel_size=1,
        # NOTE: ModelParallelConfig defaults params_dtype to fp32 and `bf16=True`
        # alone does NOT change it (that happens in megatron's arguments.py, which
        # we bypass here). So the reference model runs fp32 math unless asked
        # otherwise -- see the params_dtype kwarg.
        params_dtype=params_dtype,
        #sequence_parallel=tp_size>1,
        use_lns=sd['args'].use_lns,
        use_completedp=sd['args'].use_completedp,
        completedp_alpha=sd['args'].completedp_alpha,
        layers_mixer_config=sd['args'].layers_mixer_config,
        num_layers=sd['args'].num_layers,
        layers_mixer_config_base=sd['args'].layers_mixer_config_base,
        hidden_size_base=sd['args'].hidden_size_base,
        hidden_size=sd['args'].hidden_size,
        num_attention_heads=sd['args'].num_attention_heads,
        num_signal_heads=sd['args'].num_signal_heads,
        gate_attn=sd['args'].gate_attn,
        gate_gdn=sd['args'].gate_gdn,
        ffn_hidden_size=sd['args'].ffn_hidden_size,
        kv_channels=sd['args'].kv_channels,
        layernorm_zero_centered_gamma=sd['args'].apply_layernorm_1p,
        softcap_attn=sd['args'].softcap_attn,
        qk_layernorm=True,
        scalable_softmax=sd['args'].scalable_softmax,
        linear_attention_type=sd['args'].linear_attention_type,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=sd['args'].linear_key_head_dim,
        linear_value_head_dim=sd['args'].linear_value_head_dim,
        linear_num_key_heads=sd['args'].linear_num_key_heads,
        linear_num_value_heads=sd['args'].linear_num_value_heads,
        linear_attention_freq=sd['args'].linear_attention_freq,
        token_shift=sd['args'].token_shift,
        tpa_rank=sd['args'].tpa_rank,
        init_method_std=sd['args'].init_method_std,
        init_method_embedding_std=sd['args'].init_method_embedding_std,
        init_method_output_std=sd['args'].init_method_output_std,
        layernorm_epsilon=sd['args'].norm_epsilon,
        training_sequence_length=sd['args'].training_sequence_length,
        intra_doc_masking=sd['args'].intra_doc_masking,
        num_moe_experts=sd['args'].num_experts,
        moe_router_topk=sd['args'].moe_router_topk,
        moe_router_num_groups=sd['args'].moe_router_num_groups,
        moe_router_group_topk=sd['args'].moe_router_group_topk,
        moe_ffn_hidden_size=sd['args'].moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=sd['args'].moe_shared_expert_intermediate_size,
        moe_shared_expert_gate=sd['args'].moe_shared_expert_gate,
        moe_shared_expert_overlap=sd['args'].moe_shared_expert_overlap,
        moe_router_load_balancing_type=sd['args'].moe_router_load_balancing_type,
        moe_router_pre_softmax=sd['args'].moe_router_pre_softmax,
        moe_router_dtype=sd['args'].moe_router_dtype,
        moe_router_topk_scaling_factor=sd['args'].moe_router_topk_scaling_factor,
        moe_router_score_function=sd['args'].moe_router_score_function,
        moe_router_enable_expert_bias=sd['args'].moe_router_enable_expert_bias,
        moe_router_bias_update_rate=sd['args'].moe_router_bias_update_rate,
        moe_routed_input_dim=sd['args'].moe_routed_input_dim,
        moe_grouped_gemm=True,
        moe_router_fusion=True, # dont when using uscaling
        moe_aux_loss_coeff=1e-4, # per DSV3 paper. note: does this correctly apply to the seq-level loss?
        moe_token_dropping=False, # useless. it's the argument below that matters.
        moe_expert_capacity_factor=None,
        moe_token_drop_policy='probs',
        moe_pad_expert_input_to_capacity=False,
        moe_apply_probs_on_input=False,
        activation_func=squared_relu,
        use_te_activation_func=True,
        mamba_state_dim=sd['args'].mamba_state_dim,
        mamba_head_dim=sd['args'].mamba_head_dim,
        mamba_num_groups=sd['args'].mamba_num_groups,
        mamba_mimo_dim=sd['args'].mamba_mimo_dim,
        mamba_mimo_proj_block_order=sd['args'].mamba_mimo_proj_block_order,
        use_ddl=sd['args'].use_ddl,
        mixer_gn=sd['args'].mixer_gn,
        bf16=sd['args'].bf16,
        use_geodesic_norm=getattr(sd['args'], 'use_geodesic_norm', False),
        normalize_embeddings=sd['args'].normalize_embeddings,
        normalize_lm_head=sd['args'].normalize_lm_head,
        # Absent on checkpoints trained after the flag was dropped from the
        # training script; absent means no SSM state resets, i.e. 0.
        artificial_seq_len=getattr(sd['args'], 'artificial_seq_len', 0),
    )
    return config_mg

def _load_single_mg_model(load_dir, iteration, pgs, params_dtype=torch.float32):
    sd = load_checkpoint_base(str(load_dir), iteration)

    # create MG model
    config_mg = build_dragon_config(sd, params_dtype=params_dtype)

    model_mg = DragonModel(
        config_mg,
        get_dragon_block_spec(config_mg),
        vocab_size=sd['args'].vocab_size,
        max_sequence_length=sd['args'].training_sequence_length,
        parallel_output=True,
        pg_collection=pgs,
    ).cuda()
    
    sd['args'].ckpt_step = iteration
    iterations, num_flops_so_far = load_checkpoint(args=sd['args'], iteration=iteration, ddp_model=model_mg, load_dir=str(load_dir), checkpointing_context=None)

    return model_mg, config_mg, sd['args'].vocab_size, sd['wsize']

def load_merge_mg_models(load_dir, iterations, pgs, params_dtype=torch.float32):
    """
    iterations: int or Iterable[int]
    Returns: (DragonModel, DragonConfig, world_size)

    Strategy:
      – load the first checkpoint as the running “receiver”
      – for every extra checkpoint, load a temporary model on CPU,
        do an in-place running average into the receiver, then discard it
    """
    iterations = list(iterations)
    assert len(iterations) >= 1, "Need at least one iteration id"

    # --- receiver -----------------------------------------------------------
    receiver, cfg, vocab_size, wsize = _load_single_mg_model(load_dir, iterations[0], pgs, params_dtype=params_dtype)
    #receiver = receiver.float().cpu()          # accumulate in fp32 on CPU
    #running_k = 1                              # number of models merged so far

    print(f"Loaded model from iteration {iterations[0]}.")

    """# --- merge the rest -----------------------------------------------------
    for it in iterations[1:]:
        tmp, _, _, _ = _load_single_mg_model(load_dir, it, pgs)
        tmp = tmp.float().cpu()
        print(f"Loaded model from iteration {it}, merging it.")

        with torch.no_grad():
            for (rp, tp) in zip(receiver.parameters(), tmp.parameters()):
                #rp.mul_(running_k / (running_k + 1)).add_(tp, alpha=1.0 / (running_k + 1))
                rp.mul_(running_k).add_(tp).div_(running_k + 1)

        del tmp
        torch.cuda.empty_cache()
        running_k += 1"""

    # --- cast back to bf16 + GPU -------------------------------------------
    receiver = receiver.cuda()
    #receiver = Float16Module(config=cfg, module=receiver)
    receiver.eval()
    torch.cuda.empty_cache()
    return receiver, cfg, vocab_size, wsize

def load_hf_config(config_mg, vocab_size, wsize):
    """Derive the HF (Olala) config from a Megatron DragonConfig.

    Split out of load_hf() so the HF->Megatron direction can get the config
    without instantiating a ~6B-parameter model it would immediately discard.
    Note this is the authoritative config: the config.json inside a converted
    checkpoint has intra_doc_masking forced to False at save time.
    """
    config_hf = OlalaConfigHF(
        ngram_embeddings=False,
        ngram_embeddings_neighbor=4,
        ngram_embeddings_channels=4,
        ngram_embeddings_ratio=15,
        ddl_type="",
        base_depth=len(config_mg.layers_mixer_config_base),
        completed_p_alpha=config_mg.completedp_alpha,
        use_completed_p=config_mg.use_completedp,
        layers_stem_config="",
        layers_mlp_config="",
        layers_ve_config="",
        use_value_embedding=False,
        reduce_lm_head=False,
        vwn=False,
        vwn_m=2,
        vwn_n=3,
        vwn_wd_alpha_beta=False,
        vwn_dynamic=True,
        legacy_gate=True,
        tie_lm_head=False,
        mlp_type="simple",
        layer_norm_scaling=config_mg.use_lns,
        mamba_d_state=config_mg.mamba_state_dim,
        mamba_headdim=config_mg.mamba_head_dim,
        mamba3_rope=True,
        mamba3_remove_BC_bias=False,
        mamba3_is_id_rms=True,
        mamba3_remove_conv=True,
        mamba3_is_A_dd=True,
        mamba3_add_trapezoid=True,
        mamba3_postgate_norm=False,
        moe=config_mg.num_moe_experts is not None,
        moe_num_routed_experts=config_mg.num_moe_experts,
        moe_num_active_experts=config_mg.moe_router_topk,
        moe_routed_scaling_factor=config_mg.moe_router_topk_scaling_factor,
        moe_routed_intermediate_size=config_mg.moe_ffn_hidden_size,
        moe_shared_intermediate_size=config_mg.moe_shared_expert_intermediate_size,
        moe_routed_input_dim=config_mg.moe_routed_input_dim,
        moe_shared_expert_gate=config_mg.moe_shared_expert_gate,
        intra_doc_masking=config_mg.intra_doc_masking,
        seednorm_rank=1,
        seednorm_type=1,
        mla_kv_rank=128,
        rope_gdn=False,
        shrink_qk_da=1,
        shrink_qk_gdn=2,
        mixer_gn=config_mg.mixer_gn,
        gate_before_norm=True,
        kda_allow_neg_eigval=False,
        kda_num_v_heads=None,
        seednorm_wd=True,
        normalization_type="rmsnorm",
        tpa_rank=config_mg.tpa_rank,
        num_signal_heads_diff=config_mg.num_signal_heads,
        scalar_proj_as_hidden_matrix=True,
        token_shift_attn=config_mg.token_shift,
        token_shift_gdn=False,
        token_conv1d_attn=False,
        token_conv1d_gdn=True,
        patch_level_training=False,
        patch_level_training_size=4,
        nsa_topk=16,
        nsa_block_size=64,
        nsa_window_size=512,
        cca_seq_kernel_size=4,
        head_dim=config_mg.kv_channels,
        head_dim_gdn=config_mg.linear_value_head_dim,
        num_attention_heads_gdn=config_mg.linear_num_key_heads,
        num_key_value_heads_gdn=config_mg.linear_num_key_heads,
        zero_centered_gate=True,
        scalable_softmax=config_mg.scalable_softmax,
        mamba_mimo_dim=config_mg.mamba_mimo_dim,
        mamba_ngroups=config_mg.mamba_num_groups,
        resformer=False,
        gate_type="elementwise",
        gate_act="silu",
        gate_attn=config_mg.gate_attn,
        gate_gdn=config_mg.gate_gdn,
        fused_loss_computation=False,
        qk_norm=config_mg.qk_layernorm,
        num_attention_heads_indexer=8,
        head_dim_indexer=32,
        dsa_q_lora_rank=128,
        dsa_topk=512,
        zero_centered_gamma=config_mg.layernorm_zero_centered_gamma,
        vocab_size=vocab_size,
        max_position_embeddings=config_mg.training_sequence_length,
        use_uscaling=False,
        hidden_size=config_mg.hidden_size,
        intermediate_size=config_mg.ffn_hidden_size,
        expand_factor=2,
        layers_config=config_mg.layers_mixer_config,
        num_attention_heads=config_mg.num_attention_heads,
        num_key_value_heads=0,
        initializer_range=config_mg.init_method_std,
        softcap_attn=config_mg.softcap_attn,
        norm_epsilon=config_mg.layernorm_epsilon,
        use_cache=False,
        sliding_window_size=config_mg.training_sequence_length,
        rope_type="",
        rope_theta=0.,
        uscaling_tau=0.2,
        mlp_linking=False,
        geodesic_update=config_mg.use_geodesic_norm,
        normalize_embeddings=False,
        normalize_embeddings_ngpt=config_mg.normalize_embeddings,
        normalize_lm_head=False,
        final_norm=False,
        logits_scaling_ngpt=False,
    )    
    config_hf.slw_wsize = wsize
    return config_hf


def load_hf(config_mg, vocab_size, wsize, device="cuda"):
    """Build a randomly-initialised HF model with the geometry of the MG config.

    `device` is "cpu" for the check-free CPU conversion path; nothing here
    needs a GPU, the default just keeps the GPU path allocating where it did.
    """
    config_hf = load_hf_config(config_mg, vocab_size, wsize)
    model_hf = OlalaForCausalLM(config_hf).to(device)

    return model_hf

def convert_mg_to_hf(config_mg: DragonConfig, model_mg, config_hf: OlalaConfigHF, model_hf, tp_size=1):
    if hasattr(model_mg, 'module'):
        model_mg = model_mg.module

    def tp_all_gather_cat(t, dim=0):
        g = parallel_state.get_tensor_model_parallel_group()
        n = parallel_state.get_tensor_model_parallel_world_size()
        bufs = [torch.empty_like(t) for _ in range(n)]
        torch.distributed.all_gather(bufs, t, group=g)
        return torch.cat(bufs, dim=dim)
    def gather_model_parallel_tensor(weight, dim=0):
        return tp_all_gather_cat(weight, dim=dim)

    with torch.no_grad():
        for i, layer_type in enumerate(config_mg.layers_mixer_config):
            layer_mg = model_mg.decoder.layers[i]
            layer_hf = model_hf.model.layers[i]
            mixer_mg: Union[SelfDiffAttention, GatedDeltaNet] = layer_mg.mixer
            mixer_hf = layer_hf.mixer
            mlp_mg = layer_mg.mlp
            mlp_hf = layer_hf.mlp

            if config_mg.use_ddl:
                layer_hf.compress.shortconv.weight.copy_(layer_mg.compress.shortconv.weight)
                layer_hf.compress.read.copy_(layer_mg.compress.read)
                layer_hf.ddl_attn.beta.weight.copy_(layer_mg.ddl_mixer.beta.weight)
                layer_hf.ddl_attn.beta.bias.copy_(layer_mg.ddl_mixer.beta.bias)
                layer_hf.ddl_attn.v_proj.weight.copy_(layer_mg.ddl_mixer.v_proj.weight)
                layer_hf.ddl_attn.v_proj.bias.copy_(layer_mg.ddl_mixer.v_proj.bias)
                layer_hf.ddl_mlp.beta.weight.copy_(layer_mg.ddl_mixer.beta.weight)
                layer_hf.ddl_mlp.beta.bias.copy_(layer_mg.ddl_mixer.beta.bias)
                layer_hf.ddl_mlp.v_proj.weight.copy_(layer_mg.ddl_mixer.v_proj.weight)
                layer_hf.ddl_mlp.v_proj.bias.copy_(layer_mg.ddl_mixer.v_proj.bias)

            if layer_type == "T":
                # attention ----------------------------------
                # just for this script test. else, the "tp_sync" flag on those will do the job during normal training.
                with torch.no_grad():
                    dist.broadcast(mixer_mg.lambda_q1, src=tp_src, group=tp_group)
                    dist.broadcast(mixer_mg.lambda_k1, src=tp_src, group=tp_group)
                    dist.broadcast(mixer_mg.lambda_q2, src=tp_src, group=tp_group)
                    dist.broadcast(mixer_mg.lambda_k2, src=tp_src, group=tp_group)

                W = gather_model_parallel_tensor(mixer_mg.linear_in.weight, dim=0)
                H_super = mixer_mg.num_super_heads
                out, in_f = W.shape
                P = out // H_super
                W = W.view(H_super, P, in_f)
                Dk = mixer_mg.key_hidden_size
                r = mixer_mg.config.tpa_rank
                alpha_dim = 1 if mixer_mg.config.token_shift else 0
                gate_dim = Dk if mixer_mg.config.gate_attn else 0

                W_heads = W[:, 0:4*(Dk + r + alpha_dim)] # (H_super, 4*(Dk + r + alpha_dim), in_f)
                accum = 4*(Dk + r + alpha_dim)
                W_noise_heads = W[:, accum:accum + 1*(r + alpha_dim)] # (H_super, 1*(r + alpha_dim), in_f)
                accum += 1*(r + alpha_dim)
                W_signal_heads = W[:, accum:accum+3*Dk] # (H_super, 3*Dk, in_f)

                W_heads = rearrange(W_heads, "H (n d) f -> (H n) d f", n=4) # (H_super*4, D, in_f), H_super*4 = H
                W_q = W_heads[:, 0:Dk, :].reshape(-1, in_f) # (H*Dk, in_f)
                accum = Dk
                W_Ak = W_heads[:, accum:accum+r, :].reshape(-1, in_f) # (H*r, in_f)
                accum += r
                W_Alpha_k = W_heads[:, accum:accum+alpha_dim, :].reshape(-1, in_f) if mixer_mg.config.token_shift else None # (H*1, in_f)

                W_noise_heads = rearrange(W_noise_heads, "H (n d) f -> (H n) d f", n=1) # (H_super*1, D, in_f), H_super*1 = H
                W_Av = W_noise_heads[:, 0:r, :].reshape(-1, in_f) # (H_noise*r, in_f)
                accum = r
                W_Alpha_v = W_noise_heads[:, accum:accum+alpha_dim, :].reshape(-1, in_f) if mixer_mg.config.token_shift else None # (H_noise*1, in_f)

                W_signal_heads = rearrange(W_signal_heads, "H (n d) f -> (H n) d f", n=3) # (H_super*3, D, in_f), H_super*3 = H
                W_gate = W_signal_heads[:, 0:gate_dim, :].reshape(-1, in_f) if mixer_mg.config.gate_attn else None # (H_signal*Dk, in_f)

                # gather BkBv weights (they are not split)
                W = mixer_mg.linear_BkBv.weight # (2*r*Dk, in_f)
                W_Bk = W[0:r*Dk, :] # (r*Dk, in_f)
                W_Bv = W[r*Dk:2*r*Dk, :] # (r*Dk, in_f)

                # set hf weights.
                mixer_hf.c_q.weight.copy_(W_q)
                mixer_hf.W_A_k.weight.copy_(W_Ak)
                mixer_hf.W_A_v.weight.copy_(W_Av)
                mixer_hf.W_B_k.weight.copy_(W_Bk)
                mixer_hf.W_B_v.weight.copy_(W_Bv)
                mixer_hf.shift_proj_k.weight.copy_(W_Alpha_k)
                mixer_hf.shift_proj_v.weight.copy_(W_Alpha_v)
                mixer_hf.q_norm.norm.weight.copy_(mixer_mg.q_layernorm.weight) # tp-synced, no need to gather
                mixer_hf.k_norm.norm.weight.copy_(mixer_mg.k_layernorm.weight) # tp-synced, no need to gather
                if not config_hf.geodesic_update:
                    layer_hf.input_norm.norm.weight.copy_(mixer_mg.linear_in.layer_norm_weight)
                
                if not config_hf.intra_doc_masking:
                    softmax_scaler = mixer_mg.softmax_scaler.squeeze(0).squeeze(0).squeeze(-1)
                else:
                    softmax_scaler = mixer_mg.softmax_scaler.squeeze(0).squeeze(1)
                mixer_hf.softmax_scaler.copy_(gather_model_parallel_tensor(softmax_scaler, dim=0))
                mixer_hf.lambda_q1.copy_(mixer_mg.lambda_q1)
                mixer_hf.lambda_k1.copy_(mixer_mg.lambda_k1)
                mixer_hf.lambda_q2.copy_(mixer_mg.lambda_q2)
                mixer_hf.lambda_k2.copy_(mixer_mg.lambda_k2)
                layer_hf.gate_proj.weight.copy_(W_gate)
            elif layer_type == "V":
                # attention (v2) ----------------------------------
                W = gather_model_parallel_tensor(mixer_mg.linear_in.weight, dim=0)
                H_super = mixer_mg.num_super_heads
                out, in_f = W.shape
                P = out // H_super
                W = W.view(H_super, P, in_f)
                Dk = mixer_mg.key_hidden_size
                r = mixer_mg.config.tpa_rank
                alpha_dim = 1 if mixer_mg.config.token_shift else 0
                gate_dim = Dk if mixer_mg.config.gate_attn else 0

                W_heads = W[:, 0:4*Dk] # (H_super, 4*Dk, in_f)
                accum = 4*Dk
                W_noise_heads = W[:, accum:accum + 1*(r + alpha_dim + r + alpha_dim + 1)] # (H_super, 1*(r + alpha_dim + r + alpha_dim + 1), in_f)
                accum += 1*(r + alpha_dim + r + alpha_dim + 1)
                W_signal_heads = W[:, accum:accum+3*Dk] # (H_super, 3*Dk, in_f)

                W_heads = rearrange(W_heads, "H (n d) f -> (H n) d f", n=4) # (H_super*4, D, in_f), H_super*4 = H
                W_q = W_heads[:, 0:Dk, :].reshape(-1, in_f) # (H*Dk, in_f)
                
                W_noise_heads = rearrange(W_noise_heads, "H (n d) f -> (H n) d f", n=1) # (H_super*1, D, in_f), H_super*1 = H
                W_Ak = W_noise_heads[:, 0:r, :].reshape(-1, in_f) # (H_noise*r, in_f)
                accum = r
                W_Alpha_k = W_noise_heads[:, accum:accum+alpha_dim, :].reshape(-1, in_f) if mixer_mg.config.token_shift else None # (H_noise*1, in_f)
                accum += alpha_dim
                W_Av = W_noise_heads[:, accum:accum+r, :].reshape(-1, in_f) # (H_noise*r, in_f)
                accum += r
                W_Alpha_v = W_noise_heads[:, accum:accum+alpha_dim, :].reshape(-1, in_f) if mixer_mg.config.token_shift else None # (H_noise*1, in_f)
                accum += alpha_dim
                W_lambda_proj = W_noise_heads[:, accum:accum+1, :].reshape(-1, in_f) # (H_noise*1, in_f)

                W_signal_heads = rearrange(W_signal_heads, "H (n d) f -> (H n) d f", n=3) # (H_super*3, D, in_f), H_super*3 = H
                W_gate = W_signal_heads[:, 0:gate_dim, :].reshape(-1, in_f) if mixer_mg.config.gate_attn else None # (H_signal*Dk, in_f)

                # gather BkBv weights (they are not split)
                W = mixer_mg.linear_BkBv.weight # (2*r*Dk, in_f)
                W_Bk = W[0:r*Dk, :] # (r*Dk, in_f)
                W_Bv = W[r*Dk:2*r*Dk, :] # (r*Dk, in_f)

                # set hf weights.
                mixer_hf.c_q.weight.copy_(W_q)
                mixer_hf.W_A_k.weight.copy_(W_Ak)
                mixer_hf.W_A_v.weight.copy_(W_Av)
                mixer_hf.W_B_k.weight.copy_(W_Bk)
                mixer_hf.W_B_v.weight.copy_(W_Bv)
                mixer_hf.shift_proj_k.weight.copy_(W_Alpha_k)
                mixer_hf.shift_proj_v.weight.copy_(W_Alpha_v)
                mixer_hf.q_norm.norm.weight.copy_(mixer_mg.q_layernorm.weight) # tp-synced, no need to gather
                mixer_hf.k_norm.norm.weight.copy_(mixer_mg.k_layernorm.weight) # tp-synced, no need to gather
                if not config_hf.geodesic_update:
                    layer_hf.input_norm.norm.weight.copy_(mixer_mg.linear_in.layer_norm_weight)
                if not config_hf.intra_doc_masking:
                    softmax_scaler = mixer_mg.softmax_scaler.squeeze(0).squeeze(0).squeeze(-1)
                else:
                    softmax_scaler = mixer_mg.softmax_scaler.squeeze(0).squeeze(1)
                mixer_hf.softmax_scaler.copy_(gather_model_parallel_tensor(softmax_scaler, dim=0))
                mixer_hf.lambda_proj.weight.copy_(W_lambda_proj)
                layer_hf.gate_proj.weight.copy_(W_gate)
            elif layer_type == "g":
                # gdn ----------------------------------
                W = gather_model_parallel_tensor(mixer_mg.in_proj.weight, dim=0)
                mixer_hf.in_proj.weight.copy_(W)
                if not config_hf.geodesic_update:
                    layer_hf.input_norm.norm.weight.copy_(mixer_mg.in_proj.layer_norm_weight)
                W = gather_model_parallel_tensor(mixer_mg.conv1d.weight, dim=0)
                mixer_hf.qkv_conv1d.weight.copy_(W)
                W = gather_model_parallel_tensor(mixer_mg.dt_bias.data, dim=0)
                mixer_hf.dt_bias.data.copy_(W)
                W = gather_model_parallel_tensor(mixer_mg.A_log.data, dim=0)
                mixer_hf.A_log.data.copy_(W)
            elif layer_type == "M":
                # M3 ----------------------------------
                W = gather_model_parallel_tensor(mixer_mg.in_proj.weight, dim=0)
                """z_local = mixer_mg.d_inner // tp_size
                x_local = mixer_mg.d_inner // tp_size
                dt_local = mixer_mg.nheads // tp_size
                A_local = mixer_mg.nheads // tp_size

                local_rows = W.shape[0] // tp_size
                trap_local = local_rows - (z_local + x_local + dt_local + A_local)
                shards = list(W.split(local_rows, dim=0))

                off = 0
                z = torch.cat([s[off:off + z_local] for s in shards], dim=0)
                off += z_local
                x = torch.cat([s[off:off + x_local] for s in shards], dim=0)
                off += x_local
                dt = torch.cat([s[off:off + dt_local] for s in shards], dim=0)
                off += dt_local
                A = torch.cat([s[off:off + A_local] for s in shards], dim=0)
                off += A_local
                trap = torch.cat([s[off:off + trap_local] for s in shards], dim=0)

                W = torch.cat([z, x, dt, A, trap], dim=0)"""
                mixer_hf.in_proj.weight.copy_(W)

                mixer_hf.in_proj_dyn.weight.copy_(mixer_mg.in_proj_dyn.weight)
                if not config_hf.geodesic_update:
                    layer_hf.input_norm.norm.weight.copy_(mixer_mg.in_proj.layer_norm_weight)
                mixer_hf.B_bias.copy_(gather_model_parallel_tensor(mixer_mg.B_bias, dim=0))
                mixer_hf.C_bias.copy_(gather_model_parallel_tensor(mixer_mg.C_bias, dim=0))
                mixer_hf.B_norm.norm.weight.copy_(mixer_mg.B_norm.weight)
                mixer_hf.C_norm.norm.weight.copy_(mixer_mg.C_norm.weight)
                mixer_hf.in_proj_mimo_x.copy_(gather_model_parallel_tensor(mixer_mg.in_proj_mimo_x, dim=0))
                mixer_hf.in_proj_mimo_z.copy_(gather_model_parallel_tensor(mixer_mg.in_proj_mimo_z, dim=0))
                mixer_hf.out_proj_mimo.copy_(gather_model_parallel_tensor(mixer_mg.out_proj_mimo, dim=0))
                mixer_hf.dt_bias.copy_(gather_model_parallel_tensor(mixer_mg.dt_bias, dim=0))
                mixer_hf.D.copy_(gather_model_parallel_tensor(mixer_mg.D, dim=0))
                if config_hf.mamba3_postgate_norm:
                    mixer_hf.output_norm.norm.weight.copy_(mixer_mg.output_norm.weight)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            # mixer norm & proj ----------------------------------
            layer_hf.mixer_proj.weight.copy_(gather_model_parallel_tensor(layer_mg.mixer_proj.weight, dim=1))
            if config_hf.mixer_gn:
                layer_hf.mixer_group_norm.weight.copy_(gather_model_parallel_tensor(layer_mg.mixer_norm_scalers.squeeze(0).squeeze(0), dim=0))

            # MLP/MoE ----------------------------------
            if config_mg.num_moe_experts is not None:
                # router
                mlp_hf.moe_gate.weight.copy_(mlp_mg.router.weight) # it's sync across tp ranks: no need to gather
                # down/up proj
                if config_hf.moe_routed_input_dim:
                    mlp_hf.down_proj.weight.copy_(mlp_mg.down_proj.weight)
                    mlp_hf.up_proj.weight.copy_(mlp_mg.up_proj.weight)
                # routed experts
                W_up_all   = mlp_hf.experts.experts.weight          # [E, 4096, 1024]
                W_down_all = mlp_hf.experts.output_experts.weight   # [E, 1024, 4096]

                for i in range(config_mg.num_moe_experts):
                    W = gather_model_parallel_tensor(getattr(mlp_mg.experts.linear_fc1, f"weight{i}"), dim=0)
                    W_up_all[i].copy_(W.to(device=W_up_all.device, dtype=W_up_all.dtype))

                    W = gather_model_parallel_tensor(getattr(mlp_mg.experts.linear_fc2, f"weight{i}"), dim=1)
                    W_down_all[i].copy_(W.to(device=W_down_all.device, dtype=W_down_all.dtype))
                # shared expert
                W = gather_model_parallel_tensor(mlp_mg.shared_experts.linear_fc1.weight, dim=0)
                mlp_hf.shared_experts.fc_1.weight.copy_(W)
                W = gather_model_parallel_tensor(mlp_mg.shared_experts.linear_fc2.weight, dim=1)
                mlp_hf.shared_experts.fc_2.weight.copy_(W)
                if config_mg.moe_shared_expert_gate:
                    W = mlp_mg.shared_experts.gate_weight
                    mlp_hf.shared_gate.weight.copy_(W)
                # expert bias
                mlp_hf.expert_bias.copy_(mlp_mg.router.expert_bias)
            else:
                mlp_hf.fc_1.weight.copy_(gather_model_parallel_tensor(mlp_mg.linear_fc1.weight, dim=0))
                mlp_hf.fc_2.weight.copy_(gather_model_parallel_tensor(mlp_mg.linear_fc2.weight, dim=1))
            if not config_hf.geodesic_update:
                layer_hf.postmixer_norm.norm.weight.copy_(layer_mg.pre_mlp_norm.weight)
            else:
                layer_hf.geodesic_mixer.scale.copy_(layer_mg.geodesic_mixer.scale)
                layer_hf.geodesic_mixer.bias.copy_(layer_mg.geodesic_mixer.bias)
                layer_hf.geodesic_mlp.scale.copy_(layer_mg.geodesic_mlp.scale)
                layer_hf.geodesic_mlp.bias.copy_(layer_mg.geodesic_mlp.bias)

        if config_mg.use_ddl:
            model_hf.model.input_conv.conv.weight.copy_(model_mg.input_conv.conv.weight)
            model_hf.model.readout.shortconv.weight.copy_(model_mg.decoder.readout.shortconv.weight)
            model_hf.model.readout.read.copy_(model_mg.decoder.readout.read)

        if hasattr(model_mg.decoder.final_layernorm, 'weight'):
            model_hf.model.final_norm.norm.weight.copy_(model_mg.decoder.final_layernorm.weight)

        W = gather_model_parallel_tensor(model_mg.embedding.word_embeddings.weight, dim=0)
        model_hf.model.embedding.weight.copy_(W)
        W = gather_model_parallel_tensor(model_mg.output_layer.weight, dim=0)
        model_hf.lm_head.weight.copy_(W)

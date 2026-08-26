#!/bin/bash -l
# Olala (Dragon 7A1B) GSM8K GRPO training — self-contained environment under
# /data/home/j-g.barthelemy/olala, built from scratch on 2026-08-26 following
# the "Olala verl training — install & fix guide" (Notion). Mirrors erisa's
# validated launch_Olala.sh, with paths pointing at this env.
#
# Hybrid mamba3/geodesic layers don't tolerate FSDP's packing/dynamic-batching
# path: use_remove_padding=False, use_dynamic_bsz=False, micro batch 1.

set -euo pipefail

OLALA_HOME=/data/home/j-g.barthelemy/olala
source "$OLALA_HOME/venv/bin/activate"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0   # venv is uv-created
export HF_HUB_OFFLINE=1                  # model is local
export PYTHONUNBUFFERED=1
# Harmless here: this env's vllm has the cudagraph weight-cache fix applied
# UNGATED by the applier script. Kept for compatibility with envs where the
# fix is gated behind this variable (e.g. erisa's venv).
export OLALA_INPLACE_CACHE_REFRESH=1
# Per-user temp dir: the verl colocate weight-sync ZMQ socket lives here
# (honors TMPDIR); shared /tmp collides across users.
export TMPDIR=${TMPDIR:-/tmp/olala-$(id -u)}
mkdir -p "$TMPDIR"

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=$OLALA_HOME/.cache/vllm
export TILELANG_CACHE_DIR=$OLALA_HOME/.cache/tilelang

cd "$OLALA_HOME/run"
REWARD_PATH="$OLALA_HOME/run/reward.py"
test -f "$REWARD_PATH" || { echo "Reward file not found"; exit 1; }

CHECKPOINT_PATH="./checkpoints_Olala"
TRAIN_FILE="$OLALA_HOME/run/data/train.parquet"
TEST_FILE="$OLALA_HOME/run/data/test.parquet"
MAX_PROMPT_LEN=4096
MAX_RESPONSE_LEN=8192
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))
MODEL_PATH=${MODEL_PATH:-$OLALA_HOME/patched_checkpoint}

DATA_TRAIN_BATCH=8
PPO_MINI_BATCH=8
ROLLOUT_N=8

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=${DATA_TRAIN_BATCH} \
  data.prompt_key=prompt \
  data.max_prompt_length=${MAX_PROMPT_LEN} \
  data.max_response_length=${MAX_RESPONSE_LEN} \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
  actor_rollout_ref.actor.optim.override_optimizer_config='{fused: true}' \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
  actor_rollout_ref.rollout.disable_log_stats=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=sequence \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=10 \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=calculator_reward_fn \
  reward.reward_manager.source=register \
  reward.reward_manager.name=dapo \
  trainer.total_training_steps=100 \
  trainer.save_freq=10 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.default_local_dir="$CHECKPOINT_PATH" \
  trainer.project_name=gsm8k \
  trainer.experiment_name=olala-jgb \
  trainer.logger='["console"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.validation_data_dir="$CHECKPOINT_PATH/Validation" \
  trainer.log_val_generations=8 \
  trainer.resume_mode=disable \
  "$@" 2>&1 | tee olala_run.txt

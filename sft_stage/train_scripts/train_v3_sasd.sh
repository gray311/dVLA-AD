#!/bin/bash
# dVLA Stage-1 — SASD finetune on the V3 corpus (section-aware).
#
# Adds the 3 Fast-dDrive mechanisms on top of the V3 template:
#   1. Section mechanism : deep JSON scaffold (structural tokens frozen,
#      section-aligned blocks) via the V3-rewritten section_utils.py.
#   2. Loss weighting    : Section-Importance-Weighted Loss (per-section CE).
#   3. Noise scheduling  : per-section Beta(alpha,beta) noise.
# Five sections incl. the NEW complexity slot. Weights/noise are SASD-style
# positive values (set explanation/critical_objects to 0 later to switch to the
# plan's selective supervision — machinery already supports it).
#
# Backbone: Fast_dVLM_3B_sasd = Fast-dVLM base weights + Fast-dDrive SASD code.
#
# Required env (defaults below): MODEL_PATH, DATASET_PATH, IMAGE_FOLDER.
# Smoke: NUM_GPUS=1 MAX_STEPS=2 DATASET_PATH=.../_smoke64_v3.json bash train_v3_sasd.sh

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_sft_stage="$(cd "${_script_dir}/.." && pwd)"
_repo_root="$(cd "${_sft_stage}/.." && pwd)"

export PYTHONPATH="${_sft_stage}:${_repo_root}/third_party:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/cache/hf}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

_scratch=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy
MODEL_PATH="${MODEL_PATH:-${_scratch}/models/Fast_dVLM_3B_sasd}"
DATASET_PATH="${DATASET_PATH:-${_scratch}/data/dvla_sft/dvlm-ad_waymo_training_v3_joint.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/waymo}"
OUTPUT_DIR="${OUTPUT_DIR:-${_scratch}/checkpoints/dvla_sft_stage1_sasd}"
ds_config="${DEEPSPEED_CONFIG:-${_script_dir}/ds_config_zero2.json}"

# SASD section-weighted loss + per-section Beta noise (5 sections incl complexity).
section_loss_weights="${SECTION_LOSS_WEIGHTS:-{\"critical_objects\":1.5,\"complexity\":2.0,\"explanation\":1.0,\"future_meta_behavior\":2.0,\"trajectory\":3.0}}"
section_noise_schedule="${SECTION_NOISE_SCHEDULE:-{\"critical_objects\":\"1.0,2.0\",\"complexity\":\"1.0,2.0\",\"explanation\":\"1.0,1.0\",\"future_meta_behavior\":\"1.0,1.5\",\"trajectory\":\"2.0,1.0\"}}"

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -f "${DATASET_PATH}" ]] || { echo "DATASET_PATH not found: ${DATASET_PATH}"; exit 1; }
[[ -f "${ds_config}" ]] || { echo "ds config not found: ${ds_config}"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

# INCLUDE_GPUS="0,2,3" picks specific GPUs (e.g. to skip a faulty one); deepspeed
# ignores CUDA_VISIBLE_DEVICES when --num_gpus is used, so use --include instead.
if [[ -n "${INCLUDE_GPUS:-}" ]]; then
  _gpu_arg="--include=localhost:${INCLUDE_GPUS}"
else
  _gpu_arg="--num_gpus=${NUM_GPUS:-8}"
fi

deepspeed ${_gpu_arg} --master_port="${MASTER_PORT:-11003}" \
  "${_sft_stage}/train_scripts/finetune_fast_ddrive.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name "${TOKENIZER_NAME:-${MODEL_PATH}}" \
    --conversation_template "${CONVERSATION_TEMPLATE:-qwen2_5_no_reasoning}" \
    --trust_remote_code 1 \
    --dataset_path "${DATASET_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}" \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --deepspeed "${ds_config}" \
    --bf16 \
    --do_train \
    --gradient_checkpointing \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --save_steps "${SAVE_STEPS:-1000}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --validation_split_percentage 0 \
    --report_to "${REPORT_TO:-none}" \
    --use_lora "${USE_LORA:-1}" \
    --lora_r "${LORA_R:-32}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
    --freeze_vision_encoder "${FREEZE_VISION_ENCODER:-1}" \
    --mdm 1 \
    --bd_size "${BD_SIZE:-32}" \
    --block_size "${BLOCK_SIZE:-4096}" \
    --learn_padding 1 \
    --complementary_mask 1 \
    --always_mask_im_end 1 \
    --use_block_causal_mask 1 \
    --deep_json_scaffold 1 \
    --section_loss_weights "${section_loss_weights}" \
    --section_noise_schedule "${section_noise_schedule}" \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}}

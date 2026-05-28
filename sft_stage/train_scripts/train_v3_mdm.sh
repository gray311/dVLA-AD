#!/bin/bash
# dVLA Stage-1 — plain-MDM finetune on the V3 corpus.
#
# "Plain MDM": NO deep JSON scaffold, NO section-weighted loss / per-section
# noise. The whole assistant response is masked with a uniform noise level and
# gets uniform CE loss. This is the minimal end-to-end path on the new V3 data
# (selective supervision / three-tier mask / multi-rho come later).
#
# Required env (all have sane defaults below):
#   MODEL_PATH     backbone (default: local Fast_dDrive_as_dVLM, model_type=fast_dvlm)
#   DATASET_PATH   V3 train JSON (data/data_convert.py --split train output)
#   IMAGE_FOLDER   root the V3 image paths (train/<scene>/...) are relative to
#
# Optional: OUTPUT_DIR, NUM_GPUS, MAX_STEPS, NUM_TRAIN_EPOCHS, LEARNING_RATE,
#           PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, BD_SIZE.
#
# Smoke example (1 GPU, 2 steps, 64-sample subset):
#   NUM_GPUS=1 MAX_STEPS=2 \
#   DATASET_PATH=/weka/.../dvla_sft/_smoke64_v3.json \
#     bash sft_stage/train_scripts/train_v3_mdm.sh

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../sft_stage/train_scripts
_sft_stage="$(cd "${_script_dir}/.." && pwd)"                   # .../sft_stage
_repo_root="$(cd "${_sft_stage}/.." && pwd)"                    # .../dVLA-AD

# `import lmflow` resolves to sft_stage/lmflow; the entry also adds repo/third_party.
export PYTHONPATH="${_sft_stage}:${_repo_root}/third_party:${PYTHONPATH:-}"
# Keep all HF/torch caches on weka scratch, never under $HOME.
export HF_HOME="${HF_HOME:-/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/cache/hf}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

_scratch=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy
# Fast-dVLM 3B (Efficient-Large-Model/Fast_dVLM_3B) — block-diffusion VLM
# converted from Qwen2.5-VL-3B-Instruct; the Stage-1 backbone (plan §2.1).
MODEL_PATH="${MODEL_PATH:-${_scratch}/models/Fast_dVLM_3B}"
DATASET_PATH="${DATASET_PATH:-${_scratch}/data/dvla_sft/dvlm-ad_waymo_training_v3_joint.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/waymo}"
OUTPUT_DIR="${OUTPUT_DIR:-${_scratch}/checkpoints/dvla_sft_stage1_mdm}"
ds_config="${DEEPSPEED_CONFIG:-${_script_dir}/ds_config_zero2.json}"

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -f "${DATASET_PATH}" ]] || { echo "DATASET_PATH not found: ${DATASET_PATH}"; exit 1; }
[[ -f "${ds_config}" ]] || { echo "ds config not found: ${ds_config}"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${INCLUDE_GPUS:-}" ]]; then
  _gpu_arg="--include=localhost:${INCLUDE_GPUS}"
else
  _gpu_arg="--num_gpus=${NUM_GPUS:-8}"
fi

deepspeed ${_gpu_arg} --master_port="${MASTER_PORT:-11002}" \
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
    --deep_json_scaffold 0 \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}}

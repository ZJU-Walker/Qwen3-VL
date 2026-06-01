#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
if [ -z "${NPROC_PER_NODE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    else
        NPROC_PER_NODE=1
    fi
fi

deepspeed=./scripts/zero3.json
llm=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
entry_file=qwenvl/train/train_qwen.py

# Pick the task by its registry key in qwenvl/data/__init__.py. The registry
# entry supplies the CSV / data root / camera, so switching tasks is just:
#   DATASETS=my_new_task RUN_NAME=... bash scripts/sft_robot_subtask_8b.sh
# The labels are auto-derived from that CSV's `subtask` column -- no code edits.
datasets=${DATASETS:-trossen_block_mem_0528}

# Per-task overrides. Leave empty to use the values from the registry entry.
# Set any of these to override the registry for a one-off run.
robot_csv=${ROBOT_SUBTASK_CSV:-}
robot_root=${ROBOT_SUBTASK_ROOT:-}
robot_camera=${ROBOT_SUBTASK_CAMERA:-}
robot_prompt=${ROBOT_SUBTASK_PROMPT:-}

lr=${LR:-1e-6}
batch_size=${BATCH_SIZE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
epochs=${EPOCHS:-10}
epoch_size=${ROBOT_SUBTASK_EPOCH_SIZE:-4096}
save_steps=${SAVE_STEPS:-200}
run_name=${RUN_NAME:-qwen3vl-8b-trossen-marker-subtask}
output_dir=${OUTPUT_DIR:-./output/qwen3vl-8b-trossen-marker-subtask}

args=(
    --deepspeed "$deepspeed"
    --model_name_or_path "$llm"
    --dataset_use "$datasets"
    --robot_subtask_csv "$robot_csv"
    --robot_subtask_root "$robot_root"
    --robot_subtask_camera "$robot_camera"
    --robot_subtask_prompt "$robot_prompt"
    --robot_subtask_history_seconds 5
    --robot_subtask_num_frames 5
    --robot_subtask_lookahead_frames "${LOOKAHEAD_FRAMES:-15}"
    --robot_subtask_epoch_size "$epoch_size"
    --robot_subtask_split train
    --robot_subtask_train_episodes "${TRAIN_EPISODES:-0:52}"
    --robot_subtask_val_episodes "${VAL_EPISODES:-52:61}"
    --data_flatten True
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --lora_enable False
    --bf16
    --output_dir "$output_dir"
    --num_train_epochs "$epochs"
    --per_device_train_batch_size "$batch_size"
    --per_device_eval_batch_size "$batch_size"
    --gradient_accumulation_steps "$grad_accum_steps"
    --max_pixels 50176
    --min_pixels 784
    --video_min_frames 5
    --video_max_frames 5
    --video_fps 2
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps "$save_steps"
    --save_total_limit 5
    --learning_rate "$lr"
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name "$run_name"
    --report_to wandb
)

torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$entry_file" "${args[@]}"

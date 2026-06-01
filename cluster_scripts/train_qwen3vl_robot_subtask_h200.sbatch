#!/bin/bash
#SBATCH --job-name=qwen3vl_subtask_8b_0527
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodelist=iris-hgx-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kewalk@stanford.edu
# Logs are relative to the submit directory (Slurm reads these before the script
# runs, so they cannot use shell variables). Run sbatch from the repo root after
# `mkdir -p logs/qwen3vl_subtask_8b_0527`.
#SBATCH --output=logs/qwen3vl_subtask_8b_0527/%x-%j.out
#SBATCH --error=logs/qwen3vl_subtask_8b_0527/%x-%j.err

set -euo pipefail

# REPO_ROOT auto-derives from this script's own location, so the same sbatch
# works unchanged from any checkout (e.g. /iris/projects/humanoid/Qwen3-VL or
# /iris/u/kewalk/Qwen3-VL). Slurm copies the script to a spool dir, so prefer
# the original submit path it records before falling back to BASH_SOURCE.
HUMANOID_ROOT=/iris/projects/humanoid
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    _script_path=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
        | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)
fi
_script_path="${_script_path:-${BASH_SOURCE[0]}}"
REPO_ROOT=$(cd "$(dirname "${_script_path}")/.." && pwd)
CONDA_ROOT="${HUMANOID_ROOT}/miniconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/qwen3vl"
RUN_NAME=${RUN_NAME:-qwen3vl-8b-trossen-block-subtask-0528-mix-2cls-hist5-5f}
LOG_DIR="${REPO_ROOT}/logs/qwen3vl_subtask_8b_0527"
OUTPUT_DIR="${REPO_ROOT}/qwen-vl-finetune/output/${RUN_NAME}"
CACHE_ROOT="${HUMANOID_ROOT}/.cache/qwen3vl"

cd "${REPO_ROOT}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

# CUDA choices on Iris may include /usr/local/cuda, /usr/local/cuda-12.3,
# and /usr/local/cuda-11.5. Override CUDA_HOME at submit time if needed:
# sbatch --export=ALL,CUDA_HOME=/usr/local/cuda-12.3 cluster_scripts/train_qwen3vl_robot_subtask_h200.sbatch
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "Missing nvcc at ${CUDA_HOME}/bin/nvcc"
    exit 1
fi

export NPROC_PER_NODE=2
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
export OUTPUT_DIR="${OUTPUT_DIR}"
export RUN_NAME="${RUN_NAME}"
export QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_DIR="${REPO_ROOT}/wandb"
export WANDB_CACHE_DIR="${WANDB_DIR}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${CACHE_ROOT}/config"
export XDG_DATA_HOME="${CACHE_ROOT}/data"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch-extensions"
export TMPDIR="${HUMANOID_ROOT}/tmp/${USER}/qwen3vl"

mkdir -p \
    "${LOG_DIR}" \
    "${OUTPUT_DIR}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${HF_HOME}" \
    "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${PIP_CACHE_DIR}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TMPDIR}"

echo "==== Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} ===="
echo "Repo: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Attention: ${QWEN_ATTN_IMPLEMENTATION}"
nvcc -V
nvidia-smi
echo "================================================"

echo "==== Validating robot subtask data ===="
python3 qwen-vl-finetune/tools/check_mix_block_data.py

echo "==== Starting Qwen3-VL subtask full SFT ===="
bash qwen-vl-finetune/scripts/sft_robot_subtask_8b.sh

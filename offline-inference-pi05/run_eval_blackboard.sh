#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [ -f /home/eai/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/eai/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-lerobot-pi05}"
fi

export PYTHONPATH="$PROJECT_ROOT/lerobot/src:$PROJECT_ROOT:${PYTHONPATH:-}"

CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/ckpt/inference_pi05_45000/pretrained_model}
DATASET_ROOT=${DATASET_ROOT:-$PROJECT_ROOT/data/lerobot-new-replayed-data}
TOKENIZER_PATH=${TOKENIZER_PATH:-$PROJECT_ROOT/ckpt/inference_pi05_45000/paligemma-3b-pt-224-tokenizer}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/lerobot-new-replayed-data}

cmd=(
  "${PYTHON:-python3}" "$SCRIPT_DIR/eval_lerobot_pi05.py"
  --checkpoint "$CHECKPOINT"
  --dataset-root "$DATASET_ROOT"
  --dataset-repo-id "$DATASET_REPO_ID"
  --tokenizer-path "$TOKENIZER_PATH"
  --processor-source "${PROCESSOR_SOURCE:-checkpoint}"
  --use-relative-actions
  --relative-action-mode se3_pose
  --pose-arm-offsets "[0]"
  --pose-arm-stride 10
  --batch-size "${BATCH_SIZE:-4}"
  --device "${DEVICE:-auto}"
  --dtype "${DTYPE:-auto}"
)

if [ -n "${OUTPUT_DIR:-}" ]; then
  cmd+=(--output-dir "$OUTPUT_DIR")
fi

if [ "${SAVE_IMAGES:-0}" = "1" ]; then
  cmd+=(--save-images)
fi

"${cmd[@]}" "$@"

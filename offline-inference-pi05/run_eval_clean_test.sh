#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

if [ -f /home/eai/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/eai/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-lerobot-pi05}"
fi

export PYTHONPATH="$PROJECT_ROOT/lerobot/src:$PROJECT_ROOT:${PYTHONPATH:-}"

CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/lerobot/models/pi05_base}
DATASET_ROOT=${DATASET_ROOT:-$PROJECT_ROOT/data/clean_test_lerobot_pi05}
TOKENIZER_PATH=${TOKENIZER_PATH:-$PROJECT_ROOT/lerobot/models/paligemma-3b-pt-224-tokenizer}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/output}

"${PYTHON:-python3}" "$SCRIPT_DIR/eval_lerobot_pi05.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --dataset-repo-id "${DATASET_REPO_ID:-local/clean_test_pi05}" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --cameras "${CAMERAS:-checkpoint}" \
  --use-relative-actions \
  --relative-action-mode se3_pose \
  --pose-arm-offsets "[0]" \
  --pose-arm-stride 10 \
  --batch-size "${BATCH_SIZE:-4}" \
  --device "${DEVICE:-auto}" \
  "$@"

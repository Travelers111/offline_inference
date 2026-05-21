#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export EVAL_DATA_DIR=${EVAL_DATA_DIR:-$PROJECT_ROOT/rawdata}
export EASY_MIRRO_ROOT=${EASY_MIRRO_ROOT:-$PROJECT_ROOT/easy-mirro-dual-new_1}
export PYTHON_BIN=${PYTHON_BIN:-/home/eai/miniconda3/envs/openvla/bin/python}
export DEVICE=${DEVICE:-cuda}
export BATCH_SIZE=${BATCH_SIZE:-8}
export CAMERAS=${CAMERAS:-cam_left_wrist,cam_right_wrist}
DEFAULT_OUTPUT_ROOT=$PROJECT_ROOT/offline_inference_output/fz_rawdata

for step in 35000 45000; do
  export CHECKPOINT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/ckpt__fz}/checkpoint-$step
  export OUTPUT_DIR=${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}/checkpoint-$step
  echo "== Evaluating checkpoint-$step =="
  bash "$SCRIPT_DIR/run_eval_easy_mirro.sh" --strict "$@"
  python3 "$SCRIPT_DIR/validate_alignment.py" \
    --data-dir "$EVAL_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --chunk-size 50 \
    --max-frames 64 \
    --require-training-layout
  python3 "$SCRIPT_DIR/render_preview.py" \
    --data-dir "$OUTPUT_DIR" \
    --no-images
done

python3 "$SCRIPT_DIR/compare_checkpoints.py" \
  --outputs \
  "${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}/checkpoint-35000" \
  "${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}/checkpoint-45000" \
  --names checkpoint-35000 checkpoint-45000 \
  --output "${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}/comparison.json"

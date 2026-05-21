#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

DEFAULT_EASY_MIRRO_ROOT="$PROJECT_ROOT/easy-mirro-dual-new_1"
if [ ! -d "$DEFAULT_EASY_MIRRO_ROOT" ] && [ -d "$PROJECT_ROOT/easy-mirro-fz-zeros" ]; then
  DEFAULT_EASY_MIRRO_ROOT="$PROJECT_ROOT/easy-mirro-fz-zeros"
fi
EASY_MIRRO_ROOT=${EASY_MIRRO_ROOT:-$DEFAULT_EASY_MIRRO_ROOT}

if [ -f /home/eai/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/eai/miniconda3/etc/profile.d/conda.sh
  if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "$CONDA_ENV"
  fi
fi

export PYTHONPATH="$EASY_MIRRO_ROOT:$EASY_MIRRO_ROOT/training:${PYTHONPATH:-}"

LOCAL_DEPS="$SCRIPT_DIR/.deps/py310"
if [ -d "$LOCAL_DEPS" ]; then
  export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"
fi

export USE_TF=${USE_TF:-0}
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}

CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/ckpt__fz/checkpoint-45000}
EVAL_DATA_DIR=${EVAL_DATA_DIR:?Please set EVAL_DATA_DIR to the HDF5 evaluation data directory}
PYTHON_BIN=${PYTHON_BIN:-/home/eai/miniconda3/envs/openvla/bin/python}

EVAL_ARGS=(
  "$PYTHON_BIN" "$SCRIPT_DIR/eval_easy_mirro.py"
  --checkpoint "$CHECKPOINT" \
  --eval-data-dir "$EVAL_DATA_DIR" \
  --easy-mirro-root "$EASY_MIRRO_ROOT" \
  --cameras "${CAMERAS:-auto}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --device "${DEVICE:-auto}"
)

if [ -n "${OUTPUT_DIR:-}" ]; then
  EVAL_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

"${EVAL_ARGS[@]}" "$@"

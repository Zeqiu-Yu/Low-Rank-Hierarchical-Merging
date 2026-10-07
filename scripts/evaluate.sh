#!/usr/bin/env bash
# =============================================================================
# Thin wrapper around the Qwen2.5-Math evaluation toolkit for a single model.
#
#   bash scripts/evaluate.sh MODEL_DIR [MAX_TOKEN] [NUM_SHOTS] [DATASETS]
# =============================================================================
set -euo pipefail

MODEL_DIR=${1:?usage: evaluate.sh MODEL_DIR [MAX_TOKEN] [NUM_SHOTS] [DATASETS]}
MAX_TOKEN=${2:-10240}
NUM_SHOTS=${3:-0}
DATASETS=${4:-aime24,math500,gsm8k,college_math,minerva_math,olympiadbench}

bash evaluation/l2s_eval.sh qwen25-math-cot "${MODEL_DIR}" \
    "${MAX_TOKEN}" "${NUM_SHOTS}" "${DATASETS}"

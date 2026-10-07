#!/usr/bin/env bash
# =============================================================================
# Long-to-short evaluation wrapper for the Qwen2.5-Math evaluation toolkit.
#
#   bash evaluation/l2s_eval.sh PROMPT_TYPE MODEL_PATH MAX_TOKEN NUM_SHOTS DATASETS
#
# Adapted from the original long-to-short merging repo
# (https://github.com/hahahawu/Long-to-Short-via-Model-Merging) and the
# Qwen2.5-Math evaluation toolkit (https://github.com/QwenLM/Qwen2.5-Math).
#
# Prerequisites: clone Qwen2.5-Math and place this file in its
# ``evaluation/`` directory together with ``data_process.py``:
#
#   git clone https://github.com/QwenLM/Qwen2.5-Math.git
#   cp evaluation/l2s_eval.sh Qwen2.5-Math/evaluation/
#   cp evaluation/data_process.py Qwen2.5-Math/evaluation/
#
# MATH500 is not shipped with the toolkit; add it to
# ``Qwen2.5-Math/evaluation/data`` manually.
# =============================================================================
set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKEN=$3
NUM_SHOTS=$4
DATASETS=$5
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME=${DATASETS}
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --max_tokens_per_call ${MAX_TOKEN} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --num_shots ${NUM_SHOTS} \
    --save_outputs \
    --overwrite

# report average response length and keyword statistics
python3 data_process.py ${MODEL_NAME_OR_PATH} "outputs/"${OUTPUT_DIR}

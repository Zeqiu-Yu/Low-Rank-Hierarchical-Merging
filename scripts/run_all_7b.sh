#!/usr/bin/env bash
# =============================================================================
# End-to-end reproduction of the 7B experiments (Table 2).
#   * Qwen2.5-Math-7B (base)  +  DeepSeek-R1-Distill-Qwen-7B (reasoning)
#
# Usage:
#   DEVICE=cuda DTYPE=bfloat16 bash scripts/run_all_7b.sh
# =============================================================================
set -euo pipefail

SCALE=${SCALE:-7b}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
STAGE=${STAGE:-all}
PYTHON=${PYTHON:-python3}
OUT=${OUT:-outputs/7b}
MAX_TOKEN=${MAX_TOKEN:-10240}
NUM_SHOTS=${NUM_SHOTS:-0}

BASE="Qwen/Qwen2.5-Math-7B"
REASON="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASETS="aime24,math500,gsm8k,college_math,minerva_math,olympiadbench"

COMMON="--base_model ${BASE} --reasoning_model ${REASON} --scale ${SCALE} \
        --device ${DEVICE} --dtype ${DTYPE} \
        --rank 16 --alpha 0.15 --num_sequences 32 --seq_len 512"

echo "=============================================================="
echo " LHM -- 7B full experimental pipeline  (layout: ${OUT})"
echo "=============================================================="

if [[ "${STAGE}" == "all" || "${STAGE}" == "merge" ]]; then
  mkdir -p "${OUT}/traces"
  "${PYTHON}" scripts/extract_hessian.py ${COMMON} \
      --output "${OUT}/traces/base_k16.json"

  for METHOD in ta aim; do
    "${PYTHON}" scripts/build_lhm_model.py ${COMMON} --method "${METHOD}" \
        --lambda_bar "${LAMBDA_BAR:-0.5}" \
        --trace_cache "${OUT}/traces/base_k16.json" \
        --output_dir "${OUT}/lhm_${METHOD}" \
        --output_model_dir "${OUT}/models/LHM-${METHOD}"
  done

  for BASELINE in ta aim acm_ties orca_aim; do
    "${PYTHON}" scripts/merge_baselines.py ${COMMON} --baseline "${BASELINE}" \
        --output_model_dir "${OUT}/models/${BASELINE}"
  done
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
  for MODEL_DIR in "${OUT}"/models/*; do
    bash evaluation/l2s_eval.sh qwen25-math-cot "${MODEL_DIR}" \
        "${MAX_TOKEN}" "${NUM_SHOTS}" "${DATASETS}"
  done
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "collect" ]]; then
  ARGS=()
  for MODEL_DIR in "${OUT}"/models/*; do
    ARGS+=(--entry "$(basename "${MODEL_DIR}")=${MODEL_DIR}/math_eval/outputs")
  done
  "${PYTHON}" scripts/collect_results.py "${ARGS[@]}" --tokenizer "${BASE}" \
      --output_md "${OUT}/tables/main_7b.md" \
      --output_json "${OUT}/tables/main_7b.json"
fi

echo " done -> ${OUT}/tables/main_7b.md"

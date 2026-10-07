#!/usr/bin/env bash
# =============================================================================
# End-to-end reproduction of the 1.5B experiments (Table 1, Table 3, Fig. 1/4).
#   * Qwen2.5-Math-1.5B (base)  +  DeepSeek-R1-Distill-Qwen-1.5B (reasoning)
#   * six benchmarks: gsm8k, math500, minerva_math, olympiadbench,
#                     college_math, aime24
#
# Usage:
#   DEVICE=cuda DTYPE=bfloat16 bash scripts/run_all_1p5b.sh
#   STAGE=merge bash scripts/run_all_1p5b.sh      # only build merged models
# =============================================================================
set -euo pipefail

SCALE=${SCALE:-1.5b}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
STAGE=${STAGE:-all}                     # all | tune | merge | eval | collect
PYTHON=${PYTHON:-python3}
OUT=${OUT:-outputs/1p5b}
SEED=${SEED:-0}
MAX_TOKEN=${MAX_TOKEN:-10240}
NUM_SHOTS=${NUM_SHOTS:-0}

BASE="Qwen/Qwen2.5-Math-1.5B"
REASON="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASETS="aime24,math500,gsm8k,college_math,minerva_math,olympiadbench"

COMMON="--base_model ${BASE} --reasoning_model ${REASON} --scale ${SCALE} \
        --device ${DEVICE} --dtype ${DTYPE} --seed ${SEED} \
        --rank 16 --alpha 0.15 --num_sequences 32 --seq_len 512"

echo "=============================================================="
echo " LHM -- 1.5B full experimental pipeline"
echo " layout : ${OUT}"
echo " stage  : ${STAGE}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 1. Hessian trace caching (shared by every LHM variant)
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "tune" || "${STAGE}" == "merge" ]]; then
  mkdir -p "${OUT}/traces"
  if [[ ! -f "${OUT}/traces/base_k16.json" ]]; then
    echo "[1/5] estimating Hessian traces (Phase 2, k=16) ..."
    "${PYTHON}" scripts/extract_hessian.py ${COMMON} \
        --output "${OUT}/traces/base_k16.json"
  else
    echo "[1/5] reusing cached Hessian traces"
  fi
fi

# ---------------------------------------------------------------------------
# 2. lambda_bar selection on a 64-sample GSM8K split
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "tune" ]]; then
  echo "[2/5] selecting lambda_bar on a 64-sample GSM8K split ..."
  "${PYTHON}" scripts/tune_lambda.py ${COMMON} --method aim \
      --num_samples 64 \
      --eval_command_template \
        "bash evaluation/l2s_eval.sh qwen25-math-cot {model_dir} ${MAX_TOKEN} 0 gsm8k" \
      --output "${OUT}/tuning/lambda_selection.json"
  LAMBDA_BAR=$("${PYTHON}" - "${OUT}/tuning/lambda_selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best_lambda_bar"])
PY
)
else
  LAMBDA_BAR=${LAMBDA_BAR:-0.5}
fi
echo "    selected lambda_bar = ${LAMBDA_BAR}"

# ---------------------------------------------------------------------------
# 3. build the LHM variants
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "merge" ]]; then
  for METHOD in ta ties aim; do
    echo "[3/5] building LHM-${METHOD} ..."
    "${PYTHON}" scripts/build_lhm_model.py ${COMMON} --method "${METHOD}" \
        --lambda_bar "${LAMBDA_BAR}" \
        --trace_cache "${OUT}/traces/base_k16.json" \
        --output_dir "${OUT}/lhm_${METHOD}" \
        --output_model_dir "${OUT}/models/LHM-${METHOD}"
  done
fi

# ---------------------------------------------------------------------------
# 4. build the baseline merged models
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "merge" ]]; then
  for BASELINE in model_soup ta ties dare aim acm_ties orca_aim; do
    echo "[4/5] building baseline ${BASELINE} ..."
    "${PYTHON}" scripts/merge_baselines.py ${COMMON} --baseline "${BASELINE}" \
        --output_model_dir "${OUT}/models/${BASELINE}"
  done
fi

# ---------------------------------------------------------------------------
# 5. evaluate everything and collect the results table
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
  echo "[5/5] evaluating merged models ..."
  for MODEL_DIR in "${OUT}"/models/*; do
    NAME=$(basename "${MODEL_DIR}")
    echo "  -> ${NAME}"
    bash evaluation/l2s_eval.sh qwen25-math-cot "${MODEL_DIR}" \
        "${MAX_TOKEN}" "${NUM_SHOTS}" "${DATASETS}"
  done
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "collect" ]]; then
  ARGS=()
  for MODEL_DIR in "${OUT}"/models/*; do
    NAME=$(basename "${MODEL_DIR}")
    ARGS+=(--entry "${NAME}=${MODEL_DIR}/math_eval/outputs")
  done
  "${PYTHON}" scripts/collect_results.py "${ARGS[@]}" \
      --tokenizer "${BASE}" \
      --output_md "${OUT}/tables/main_1p5b.md" \
      --output_json "${OUT}/tables/main_1p5b.json"
fi

echo "=============================================================="
echo " done. results -> ${OUT}/tables/main_1p5b.md"
echo "=============================================================="

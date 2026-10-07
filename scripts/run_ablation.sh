#!/usr/bin/env bash
# =============================================================================
# Incremental ablation (Table 3 / Fig. 3) for both the TA and AIM base methods.
#   rungs: ta  ->  +Hessian(L)  ->  +LSOS(L+S)  ->  +DCCM(full LHM)
# =============================================================================
set -euo pipefail

SCALE=${SCALE:-1.5b}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
PYTHON=${PYTHON:-python3}
OUT=${OUT:-outputs/ablation}

for BASE_METHOD in ta aim; do
  for RUNG in ta hessian lsos dccm; do
    echo "== ablation: base=${BASE_METHOD} rung=${RUNG} =="
    "${PYTHON}" scripts/run_ablation.py --scale "${SCALE}" \
        --rung "${RUNG}" --base_method "${BASE_METHOD}" \
        --device "${DEVICE}" --dtype "${DTYPE}" \
        --lambda_bar "${LAMBDA_BAR:-0.5}" --rank 16 --alpha 0.15 \
        --output_dir "${OUT}" \
        --output_model_dir "${OUT}/models/${BASE_METHOD}_${RUNG}"
  done
done

for MODEL_DIR in "${OUT}"/models/*; do
  bash evaluation/l2s_eval.sh qwen25-math-cot "${MODEL_DIR}" 10240 0 \
      aime24,math500,gsm8k,college_math,minerva_math,olympiadbench
done

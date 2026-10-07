#!/usr/bin/env bash
# =============================================================================
# Hyper-parameter sweeps (Fig. 4): rank k, target coefficient lambda_bar,
# bandwidth alpha.  Emits a manifest and (unless DRY_RUN=1) builds the models.
# =============================================================================
set -euo pipefail

SCALE=${SCALE:-1.5b}
METHOD=${METHOD:-aim}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
PYTHON=${PYTHON:-python3}
OUT=${OUT:-outputs/sweeps}
DRY=${DRY_RUN:-0}

ARGS=()
if [[ "${DRY}" == "1" ]]; then ARGS+=(--dry_run); fi

for SWEEP in rank lambda alpha; do
  echo "== sweep: ${SWEEP} =="
  "${PYTHON}" scripts/run_hyperparam_sweep.py --sweep "${SWEEP}" \
      --scale "${SCALE}" --method "${METHOD}" \
      --device "${DEVICE}" --dtype "${DTYPE}" \
      --output_dir "${OUT}" \
      --emit_manifest "${OUT}/${SWEEP}_manifest.json" ${ARGS[@]+"${ARGS[@]}"}
done

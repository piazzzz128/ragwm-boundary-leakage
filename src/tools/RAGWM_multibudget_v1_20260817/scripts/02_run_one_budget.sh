#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash 02_run_one_budget.sh {fpr01|fpr025|fpr10}" >&2
  exit 2
fi

BUDGET_ID="$1"
case "$BUDGET_ID" in
  fpr01|fpr025|fpr10) ;;
  fpr05)
    echo "[STOP] fpr05 is already frozen as the existing aligned sensitivity run; do not rerun it." >&2
    exit 3
    ;;
  *)
    echo "[ERROR] Unsupported budget id: $BUDGET_ID" >&2
    exit 2
    ;;
esac

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
PACKAGE_DIR=${PACKAGE_DIR:-$REPO/tools/RAGWM_multibudget_v1_20260817}
SCRIPT_DIR="$PACKAGE_DIR/scripts"
OUT_ROOT=${OUT_ROOT:-$STORAGE/output/sanitization_multibudget_nf_aligned_v1}
CONDITION="$OUT_ROOT/conditions/qwen25_7b_${BUDGET_ID}"
DB="$OUT_ROOT/vectorstores/qwen25_7b_${BUDGET_ID}"
THRESHOLD="$OUT_ROOT/thresholds/threshold_${BUDGET_ID}.json"
ATTACK_INPUT="$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl"
MODEL_PATH=${MODEL_PATH:-$STORAGE/local_models/Qwen2.5-7B}

if [[ ! -f "$THRESHOLD" ]]; then
  echo "[ERROR] Missing frozen threshold: $THRESHOLD" >&2
  exit 4
fi
if [[ -e "$CONDITION" || -e "$DB" ]]; then
  echo "[STOP] Output already exists. Preserve it and inspect before any rerun:" >&2
  echo "  $CONDITION" >&2
  echo "  $DB" >&2
  exit 5
fi

mkdir -p "$CONDITION/logs" "$CONDITION/metrics" "$OUT_ROOT/vectorstores"

cd "$SCRIPT_DIR"
python 03_gainratio_iterative_sanitize_aligned.py \
  --input "$ATTACK_INPUT" \
  --threshold-json "$THRESHOLD" \
  --output-dir "$CONDITION/sanitization" \
  --model-path "$MODEL_PATH" \
  --max-steps 8 --reset \
  2>&1 | tee "$CONDITION/logs/03_formal_sanitize.log"

python 05_build_sanitized_vectorstore_aligned.py \
  --repo "$REPO" \
  --input "$CONDITION/sanitization/gainratio_sanitized_corpus.jsonl" \
  --db "$DB" \
  --summary-output "$CONDITION/metrics/vectorstore_build_summary.json" \
  --reset \
  2>&1 | tee "$CONDITION/logs/05_build_vectorstore.log"

EA="$CONDITION" EA_BASE="$CONDITION/basepath" EA_DB="$DB" \
  bash 07_run_aligned_verification.sh

python 06_compare_e0_e1_aligned.py \
  --e0 "$E2E/conditions/e0_watermarked_before/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e1 "$E2E/conditions/e1_oracle_restore/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --aligned "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$CONDITION/metrics/e0_e1_budget_comparison.json" \
  2>&1 | tee "$CONDITION/logs/06_compare.log"

sha256sum \
  "$THRESHOLD" \
  "$CONDITION/sanitization/gainratio_sanitization_summary.json" \
  "$CONDITION/sanitization/gainratio_sanitized_corpus.jsonl" \
  "$CONDITION/metrics/vectorstore_build_summary.json" \
  "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  "$CONDITION/metrics/e0_e1_budget_comparison.json" \
  > "$CONDITION/metrics/SHA256SUMS.txt"

echo "FORMAL BUDGET RUN COMPLETE: $BUDGET_ID"
echo "Condition: $CONDITION"

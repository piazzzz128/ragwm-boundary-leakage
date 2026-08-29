#!/usr/bin/env bash
set -euo pipefail

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
PACKAGE_DIR=${PACKAGE_DIR:-$REPO/tools/RAGWM_cross_detector_gemma2_2b_v1_20260819}
MULTI_PACKAGE=${MULTI_PACKAGE:-$REPO/tools/RAGWM_multibudget_v1_20260817}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
MULTI_OUT=${MULTI_OUT:-$STORAGE/output/sanitization_multibudget_nf_aligned_v1}
OUT_ROOT=${OUT_ROOT:-$STORAGE/output/cross_detector_lm_robustness_v1/gemma2_2b_base}
MODEL_PATH=${MODEL_PATH:-$STORAGE/local_models/Gemma-2-2B-base-ms}

THRESHOLD="$OUT_ROOT/thresholds/gemma_fpr05.json"
PREFLIGHT="$OUT_ROOT/audit/preflight.json"
CONDITION="$OUT_ROOT/conditions/gemma2_2b_base_fpr05"
DB="$OUT_ROOT/vectorstores/gemma2_2b_base_fpr05"
ATTACK_INPUT="$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl"
AUDIT_ROOT="$MULTI_OUT/audits/current_provider_verifier_audit_v1_20260818"
E0_CURRENT="$AUDIT_ROOT/conditions/e0_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
E1_CURRENT="$AUDIT_ROOT/conditions/e1_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"

for required in \
  "$PREFLIGHT" \
  "$THRESHOLD" \
  "$ATTACK_INPUT" \
  "$E0_CURRENT" \
  "$E1_CURRENT" \
  "$MULTI_PACKAGE/scripts/03_gainratio_iterative_sanitize_aligned.py" \
  "$MULTI_PACKAGE/scripts/05_build_sanitized_vectorstore_aligned.py" \
  "$MULTI_PACKAGE/scripts/07_run_aligned_verification.sh" \
  "$PACKAGE_DIR/scripts/03_compare_current_provider.py"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 2
  fi
done

python - "$PREFLIGHT" "$THRESHOLD" <<'PY'
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
threshold = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not preflight.get("pass"):
    raise SystemExit("[ERROR] Preflight is not marked pass")
if threshold.get("extension_role") != "cross_detector_lm_robustness_aligned_v1":
    raise SystemExit("[ERROR] Threshold is not the cross-detector extension")
if threshold.get("model") != "Gemma-2-2B-base":
    raise SystemExit("[ERROR] Threshold model mismatch")
print("GEMMA FORMAL INPUT GUARDS PASS")
PY

if [[ -e "$CONDITION" || -e "$DB" ]]; then
  echo "[STOP] Preserve existing output; refusing overwrite:" >&2
  echo "  $CONDITION" >&2
  echo "  $DB" >&2
  exit 5
fi

if [[ -f /root/.config/ragwm/verifier.env ]]; then
  # shellcheck disable=SC1091
  source /root/.config/ragwm/verifier.env
fi
: "${RAGWM_API_KEY:?Missing RAGWM_API_KEY or /root/.config/ragwm/verifier.env}"
: "${RAGWM_API_BASE:?Missing RAGWM_API_BASE}"
: "${RAGWM_VERIFIER_MODEL:?Missing RAGWM_VERIFIER_MODEL}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

if ! grep -q 'RAGWM_API_KEY' "$REPO/src/models/GPT.py"; then
  echo "[ERROR] Audited verifier environment hotfix is not installed." >&2
  exit 6
fi

python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RAGWM_API_KEY"],
    base_url=os.environ["RAGWM_API_BASE"],
    timeout=45.0,
)
response = client.chat.completions.create(
    model=os.environ["RAGWM_VERIFIER_MODEL"],
    temperature=0,
    max_tokens=8,
    messages=[
        {"role": "system", "content": "Reply with exactly Yes or No."},
        {"role": "user", "content": "Is one plus one equal to two?"},
    ],
)
answer = response.choices[0].message.content
if not isinstance(answer, str) or answer.strip().lower().rstrip(".") not in {"yes", "no"}:
    raise RuntimeError(f"Unexpected API test output: {answer!r}")
print("CURRENT VERIFIER API TEST PASS:", answer.strip())
PY

mkdir -p "$CONDITION/logs" "$CONDITION/metrics" "$OUT_ROOT/vectorstores"

cd "$MULTI_PACKAGE/scripts"
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
STORAGE="$STORAGE" REPO="$REPO" E2E="$E2E" \
  bash "$MULTI_PACKAGE/scripts/07_run_aligned_verification.sh"

python "$PACKAGE_DIR/scripts/03_compare_current_provider.py" \
  --e0 "$E0_CURRENT" \
  --e1 "$E1_CURRENT" \
  --attack "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --sanitization-summary "$CONDITION/sanitization/gainratio_sanitization_summary.json" \
  --output "$CONDITION/metrics/current_provider_comparison.json" \
  2>&1 | tee "$CONDITION/logs/06_compare_current_provider.log"

sha256sum \
  "$PREFLIGHT" \
  "$OUT_ROOT/scores/gemma_nfcorpus_aligned_scores.json" \
  "$OUT_ROOT/scores/gemma_trec_aligned_scores.json" \
  "$OUT_ROOT/metrics/gemma_boundary_metrics.json" \
  "$THRESHOLD" \
  "$CONDITION/sanitization/gainratio_sanitization_summary.json" \
  "$CONDITION/sanitization/gainratio_sanitized_corpus.jsonl" \
  "$CONDITION/metrics/vectorstore_build_summary.json" \
  "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  "$CONDITION/metrics/current_provider_comparison.json" \
  > "$CONDITION/metrics/SHA256SUMS.txt"

echo "GEMMA CROSS-DETECTOR FORMAL E2E PASS"
echo "Condition preserved at: $CONDITION"

#!/usr/bin/env bash
set -euo pipefail

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
PACKAGE_DIR=${PACKAGE_DIR:-$REPO/tools/RAGWM_qwen2_5_14b_cross_detector_v1_20260820}
MULTI_PACKAGE=${MULTI_PACKAGE:-$REPO/tools/RAGWM_multibudget_v1_20260817}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
MULTI_OUT=${MULTI_OUT:-$STORAGE/output/sanitization_multibudget_nf_aligned_v1}
OUT_ROOT=${OUT_ROOT:-$STORAGE/output/cross_detector_lm_robustness_v1/qwen2_5_14b_base}
MODEL_PATH=${MODEL_PATH:-$STORAGE/local_models/Qwen2.5-14B}

THRESHOLD="$OUT_ROOT/thresholds/qwen14b_fpr05.json"
PREFLIGHT="$OUT_ROOT/audit/preflight.json"
MODEL_FREEZE="$OUT_ROOT/audit/qwen14b_model_freeze.json"
NF_SCORES="$OUT_ROOT/scores/qwen14b_nfcorpus_aligned_scores.json"
TREC_SCORES="$OUT_ROOT/scores/qwen14b_trec_aligned_scores.json"
METRICS="$OUT_ROOT/metrics/qwen14b_boundary_metrics.json"
PAIRED="$OUT_ROOT/metrics/qwen14b_vs_qwen7b_paired.json"
CONDITION="$OUT_ROOT/conditions/qwen2_5_14b_base_fpr05"
DB="$OUT_ROOT/vectorstores/qwen2_5_14b_base_fpr05"
ATTACK_INPUT="$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl"
REFERENCE_ROOT="$MULTI_OUT/audits/current_provider_verifier_audit_v1_20260818"
E0_REFERENCE="$REFERENCE_ROOT/conditions/e0_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
E1_REFERENCE="$REFERENCE_ROOT/conditions/e1_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"

SANITIZER="$MULTI_PACKAGE/scripts/03_gainratio_iterative_sanitize_aligned.py"
COMMON="$MULTI_PACKAGE/scripts/e2_common_aligned.py"
BUILD_DB="$MULTI_PACKAGE/scripts/05_build_sanitized_vectorstore_aligned.py"
VERIFY="$MULTI_PACKAGE/scripts/07_run_aligned_verification.sh"
COMPARE="$PACKAGE_DIR/scripts/06_compare_aligned_reference.py"

for required in \
  "$PREFLIGHT" \
  "$MODEL_FREEZE" \
  "$NF_SCORES" \
  "$TREC_SCORES" \
  "$METRICS" \
  "$PAIRED" \
  "$THRESHOLD" \
  "$ATTACK_INPUT" \
  "$E0_REFERENCE" \
  "$E1_REFERENCE" \
  "$SANITIZER" \
  "$COMMON" \
  "$BUILD_DB" \
  "$VERIFY" \
  "$COMPARE"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 2
  fi
done

check_sha256() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[ERROR] Dependency SHA-256 mismatch: $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 3
  fi
}

check_sha256 54baa2d68c024641b0371ba0392873de2f919011097836e89bc39a7af0563d2d "$SANITIZER"
check_sha256 1180d0f3898fc6a1dc029149f353fc4416803cd9a819e3a6ebe705412baf1235 "$COMMON"
check_sha256 fb7a63006544aa6b06ed518f09a862630f09a927913bbce55f2af2a63e5f62fd "$BUILD_DB"
check_sha256 3e3d5433bb3d8fb97e3caca65b7dea6bb7d41addbca98337c2b85e0f28976ca8 "$VERIFY"

python - "$PREFLIGHT" "$THRESHOLD" "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
threshold = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
model_path = Path(sys.argv[3])
if not preflight.get("pass"):
    raise SystemExit("[ERROR] Preflight is not marked pass")
if preflight.get("model_name") != "Qwen2.5-14B":
    raise SystemExit("[ERROR] Preflight model mismatch")
if Path(preflight.get("model_dir", "")) != model_path:
    raise SystemExit("[ERROR] Runtime model path differs from preflight")
if threshold.get("extension_role") != "cross_detector_lm_robustness_aligned_v1":
    raise SystemExit("[ERROR] Threshold is not the aligned cross-detector extension")
if threshold.get("model") != "Qwen2.5-14B":
    raise SystemExit("[ERROR] Threshold model mismatch")
if float(threshold.get("target_clean_fpr", -1)) != 0.05:
    raise SystemExit("[ERROR] Threshold is not the frozen 5% FPR point")
print("QWEN2.5-14B FORMAL INPUT GUARDS PASS")
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
: "${RAGWM_API_KEY:?Missing RAGWM_API_KEY or verifier.env}"
: "${RAGWM_API_BASE:?Missing RAGWM_API_BASE}"
: "${RAGWM_VERIFIER_MODEL:?Missing RAGWM_VERIFIER_MODEL}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

if ! grep -q 'RAGWM_API_KEY' "$REPO/src/models/GPT.py"; then
  echo "[ERROR] Audited verifier environment adapter is not installed." >&2
  exit 6
fi

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
  bash "$VERIFY"

python "$COMPARE" \
  --e0 "$E0_REFERENCE" \
  --e1 "$E1_REFERENCE" \
  --attack "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --sanitization-summary "$CONDITION/sanitization/gainratio_sanitization_summary.json" \
  --output "$CONDITION/metrics/aligned_reference_comparison.json" \
  2>&1 | tee "$CONDITION/logs/06_compare_aligned_reference.log"

sha256sum \
  "$PREFLIGHT" \
  "$MODEL_FREEZE" \
  "$NF_SCORES" \
  "$TREC_SCORES" \
  "$METRICS" \
  "$PAIRED" \
  "$THRESHOLD" \
  "$CONDITION/sanitization/gainratio_sanitization_summary.json" \
  "$CONDITION/sanitization/gainratio_sanitized_corpus.jsonl" \
  "$CONDITION/metrics/vectorstore_build_summary.json" \
  "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  "$CONDITION/metrics/aligned_reference_comparison.json" \
  > "$CONDITION/metrics/SHA256SUMS.txt"

echo "QWEN2.5-14B CROSS-DETECTOR FORMAL E2E PASS"
echo "Condition preserved at: $CONDITION"

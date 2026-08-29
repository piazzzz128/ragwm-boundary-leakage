#!/usr/bin/env bash
set -euo pipefail

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
STRICT_E2=${STRICT_E2:-$E2E/conditions/e2_gainratio_delete}
E0=${E0:-$E2E/conditions/e0_watermarked_before}
E1=${E1:-$E2E/conditions/e1_oracle_restore}
EA=${EA:-$E2E/conditions/e2_gainratio_aligned_sensitivity}
EA_DB=${EA_DB:-$E2E/vectorstores/e2_gainratio_aligned_sensitivity}
SNAPROOT=${SNAPROOT:-$E2E/restored/strict_snapshot/root/autodl-tmp/ragwm_storage}
PIPEA=${PIPEA:-$REPO/tools/e2_gainratio_aligned_sensitivity_pipeline}

ALIGNED_SCORES=${ALIGNED_SCORES:-$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json}
ATTACK_INPUT=${ATTACK_INPUT:-$STRICT_E2/corpus/watermarked_attack_input.jsonl}
INJECT_JSON=${INJECT_JSON:-$E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json}

mkdir -p "$EA/logs" "$EA/scores" "$EA/metrics"

cd "$PIPEA"

echo "=== 1. PREFLIGHT ==="
python 00_preflight_aligned.py \
  --watermarked-db "$SNAPROOT/chromadb_db" \
  --aligned-scores "$ALIGNED_SCORES" \
  --attack-input "$ATTACK_INPUT" \
  | tee "$EA/logs/00_preflight.log"

echo "=== 2. ALIGNED THRESHOLD CALIBRATION ==="
python 02_calibrate_threshold_aligned.py \
  --scores "$ALIGNED_SCORES" \
  --output "$EA/scores/fixed_threshold_fpr05_aligned.json" \
  | tee "$EA/logs/02_threshold.log"

echo "=== 3. 20-DOCUMENT SMOKE RUN ==="
rm -rf "$E2E/conditions/e2_gainratio_aligned_smoke"
python 03_gainratio_iterative_sanitize_aligned.py \
  --input "$ATTACK_INPUT" \
  --threshold-json "$EA/scores/fixed_threshold_fpr05_aligned.json" \
  --output-dir "$E2E/conditions/e2_gainratio_aligned_smoke" \
  --limit 20 --max-steps 8 --reset \
  | tee "$EA/logs/03_smoke.log"

echo
echo "Smoke run complete."
echo "Inspect the smoke summary before starting the full run:"
echo "$E2E/conditions/e2_gainratio_aligned_smoke/gainratio_sanitization_summary.json"
echo
echo "The full-corpus command is intentionally NOT run automatically."

#!/usr/bin/env bash
set -euo pipefail

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
PACKAGE_DIR=${PACKAGE_DIR:-$REPO/tools/RAGWM_multibudget_v1_20260817}
OUT_ROOT=${OUT_ROOT:-$STORAGE/output/sanitization_multibudget_nf_aligned_v1}

E0_DB="$E2E/restored/strict_snapshot/root/autodl-tmp/ragwm_storage/chromadb_db"
FPR01_DB="$OUT_ROOT/vectorstores/qwen25_7b_fpr01"
FPR025_DB="$OUT_ROOT/vectorstores/qwen25_7b_fpr025"
FPR05_DB="$E2E/vectorstores/e2_gainratio_aligned_sensitivity"
FPR10_DB="$OUT_ROOT/vectorstores/qwen25_7b_fpr10"
OUTPUT="$OUT_ROOT/metrics/multibudget_retrieval_utility_paired_bootstrap.json"

for path in "$E0_DB" "$FPR01_DB" "$FPR025_DB" "$FPR05_DB" "$FPR10_DB"; do
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing vector store: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT/metrics" "$OUT_ROOT/logs"
cd "$PACKAGE_DIR/scripts"
python evaluate_retrieval_utility_frozen.py \
  --repo "$REPO" \
  --db "e0_watermarked=$E0_DB" \
  --db "qwen25_7b_fpr01=$FPR01_DB" \
  --db "qwen25_7b_fpr025=$FPR025_DB" \
  --db "qwen25_7b_fpr05=$FPR05_DB" \
  --db "qwen25_7b_fpr10=$FPR10_DB" \
  --baseline-name e0_watermarked \
  --top-k 100 --n-boot 5000 --seed 20260609 \
  --output "$OUTPUT" \
  2>&1 | tee "$OUT_ROOT/logs/03_retrieval_utility.log"

sha256sum "$OUTPUT" > "$OUT_ROOT/metrics/retrieval_utility_SHA256SUMS.txt"
echo "MULTIBUDGET RETRIEVAL UTILITY PASS"

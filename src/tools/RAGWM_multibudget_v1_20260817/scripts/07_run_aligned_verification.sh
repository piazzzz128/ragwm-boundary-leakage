#!/usr/bin/env bash
set -euo pipefail

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
E0=${E0:-$E2E/conditions/e0_watermarked_before}
EA=${EA:-$E2E/conditions/e2_gainratio_aligned_sensitivity}
EA_BASE=${EA_BASE:-$EA/basepath}
EA_DB=${EA_DB:-$E2E/vectorstores/e2_gainratio_aligned_sensitivity}

if [[ ! -d "$EA_DB" ]]; then
  echo "[ERROR] Missing aligned vector store: $EA_DB" >&2
  exit 2
fi

rm -rf "$EA_BASE"
mkdir -p \
  "$EA_BASE/wm_prepare/nfcorpus" \
  "$EA_BASE/wm_generate/nfcorpus/10" \
  "$EA_BASE/wm_generate/nfcorpus/gpt4o_mini_e2e/10" \
  "$EA/logs" "$EA/metrics"

cp "$E0/basepath/wm_prepare/nfcorpus/wmunit.json" \
   "$EA_BASE/wm_prepare/nfcorpus/wmunit.json"
cp "$E0/basepath/wm_generate/nfcorpus/10/wmuint_doc.json" \
   "$EA_BASE/wm_generate/nfcorpus/10/wmuint_doc.json"
cp "$E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
   "$EA_BASE/wm_generate/nfcorpus/10/wmuint_inject.json"

cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export RAGWM_CHROMA_PATH="$EA_DB"
export PYTHONUNBUFFERED=1

python src/main.py \
  --eval_dataset nfcorpus \
  --eval_model_code contriever \
  --score_function cosine \
  --top_k 5 \
  --mutual_times 10 \
  --model_name_llm gpt4o_mini_e2e \
  --model_name_rllm gpt4o_mini_e2e \
  --gpu_id 0 \
  --basepath "$EA_BASE" \
  --verify_num 30 \
  --verify_seed 633 \
  --doc 0 --inject 0 --verify 1 --stat 0 \
  2>&1 | tee "$EA/logs/07_verify.log"

#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?Usage: bash 04_run_single_random_seed.sh <seed>}

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
E0=${E0:-$E2E/conditions/e0_watermarked_before}
E2=${E2:-$E2E/conditions/e2_gainratio_delete}
E3ROOT=${E3ROOT:-$E2E/conditions/e3_random_baselines}
PIPE=${PIPE:-$REPO/tools/e3_random_utility_pipeline}

COND=$E3ROOT/seed_${SEED}
CORPUS_DIR=$COND/sanitization
DB=$E2E/vectorstores/e3_random_seed_${SEED}
BASE=$COND/basepath
BUDGET=$E3ROOT/budget/e3_budget_manifest.json
INPUT=$E2/corpus/watermarked_attack_input.jsonl

mkdir -p "$COND/logs" "$COND/metrics"

if [[ ! -f "$CORPUS_DIR/random_sanitization_summary.json" ]]; then
  python "$PIPE/01_build_random_sanitized_corpus.py" \
    --input "$INPUT" \
    --budget "$BUDGET" \
    --output-dir "$CORPUS_DIR" \
    --seed "$SEED" \
    --reset \
    2>&1 | tee "$COND/logs/01_random_sanitize.log"
else
  echo "[SKIP] Random corpus already exists for seed $SEED"
fi

python "$PIPE/02_posthoc_random_ground_truth_audit.py" \
  --original "$INPUT" \
  --sanitized "$CORPUS_DIR/random_sanitized_corpus.jsonl" \
  --inject-json "$E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
  --output "$COND/metrics/posthoc_random_ground_truth_audit.json" \
  2>&1 | tee "$COND/logs/02_posthoc_audit.log"

if [[ ! -d "$DB" ]] || [[ ! -f "$DB/../e3_random_seed_${SEED}_build_summary.json" ]]; then
  python "$PIPE/03_build_random_vectorstore.py" \
    --repo "$REPO" \
    --input "$CORPUS_DIR/random_sanitized_corpus.jsonl" \
    --db "$DB" \
    --reset \
    2>&1 | tee "$COND/logs/03_build_vectorstore.log"
else
  echo "[SKIP] Random vector store already exists for seed $SEED"
fi

mkdir -p \
  "$BASE/wm_prepare/nfcorpus" \
  "$BASE/wm_generate/nfcorpus/10" \
  "$BASE/wm_generate/nfcorpus/gpt4o_mini_e2e/10"

cp "$E0/basepath/wm_prepare/nfcorpus/wmunit.json" \
   "$BASE/wm_prepare/nfcorpus/wmunit.json"
cp "$E0/basepath/wm_generate/nfcorpus/10/wmuint_doc.json" \
   "$BASE/wm_generate/nfcorpus/10/wmuint_doc.json"
cp "$E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
   "$BASE/wm_generate/nfcorpus/10/wmuint_inject.json"

VERIFY=$BASE/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json

if [[ ! -f "$VERIFY" ]]; then
  cd "$REPO"
  export PYTHONPATH="$REPO"
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export RAGWM_CHROMA_PATH="$DB"
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
    --basepath "$BASE" \
    --verify_num 30 \
    --verify_seed 633 \
    --doc 0 --inject 0 --verify 1 --stat 0 \
    2>&1 | tee "$COND/logs/04_verify.log"
else
  echo "[SKIP] Verification output already exists for seed $SEED"
fi

echo "RANDOM SEED $SEED COMPLETE"

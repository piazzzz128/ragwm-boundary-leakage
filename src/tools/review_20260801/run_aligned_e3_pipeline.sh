#!/usr/bin/env bash
set -euo pipefail

# Budget-matched random baseline for NFCorpus token-aligned E2.
# This creates a new condition tree and does not overwrite strict-v1 E3.

RAG_STORAGE=${RAG_STORAGE:-/root/autodl-tmp/ragwm_storage}
RAG_REPO=${RAG_REPO:-$RAG_STORAGE/repo/ragwm_src}
RAG_E2E=${RAG_E2E:-$RAG_STORAGE/output/sanitization_e2e_nf_strict_v1}
RAG_E0=${RAG_E0:-$RAG_E2E/conditions/e0_watermarked_before}
RAG_E1=${RAG_E1:-$RAG_E2E/conditions/e1_oracle_restore}
RAG_E2_STRICT=${RAG_E2_STRICT:-$RAG_E2E/conditions/e2_gainratio_delete}
RAG_E2_ALIGNED=${RAG_E2_ALIGNED:-$RAG_E2E/conditions/e2_gainratio_aligned_sensitivity}
RAG_E3_ALIGNED=${RAG_E3_ALIGNED:-$RAG_E2E/conditions/e3_random_aligned_v2}
RAG_E3_PIPE=${RAG_E3_PIPE:-$RAG_REPO/tools/e3_random_utility_pipeline}
RAG_ATTACK_INPUT=${RAG_ATTACK_INPUT:-$RAG_E2_STRICT/corpus/watermarked_attack_input.jsonl}
RAG_SEEDS=(101 202 303 404 505)

export PYTHONPATH="$RAG_E3_PIPE:$RAG_REPO"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

verification_status() {
  python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

verify_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])

if not verify_path.is_file():
    print("partial")
    raise SystemExit(0)

try:
    rows = json.loads(verify_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("invalid")
    raise SystemExit(0)

def unit_keys(items):
    if not isinstance(items, list):
        raise ValueError("verification JSON must be a list")
    keys = []
    for row in items:
        if not isinstance(row, list) or len(row) < 3 or not isinstance(row[0], list):
            raise ValueError("malformed verification row")
        keys.append(tuple(row[0]))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate verification units")
    return set(keys)

try:
    current_keys = unit_keys(rows)
    baseline_keys = unit_keys(baseline)
except (TypeError, ValueError):
    print("invalid")
    raise SystemExit(0)

if len(baseline) != 30 or not current_keys.issubset(baseline_keys):
    print("invalid")
elif len(rows) == 30 and current_keys == baseline_keys:
    print("complete")
else:
    print("partial")
PY
}

required_files=(
  "$RAG_ATTACK_INPUT"
  "$RAG_E2_ALIGNED/sanitization/gainratio_sanitization_summary.json"
  "$RAG_E2_ALIGNED/sanitization/gainratio_deletion_log.jsonl"
  "$RAG_E0/basepath/wm_prepare/nfcorpus/wmunit.json"
  "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_doc.json"
  "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json"
  "$RAG_E0/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
  "$RAG_E1/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
  "$RAG_E2_ALIGNED/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
)
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$RAG_E3_ALIGNED/budget" "$RAG_E3_ALIGNED/logs"
budget="$RAG_E3_ALIGNED/budget/e3_aligned_budget_manifest.json"

python "$RAG_E3_PIPE/00_freeze_e2_budget.py" \
  --summary "$RAG_E2_ALIGNED/sanitization/gainratio_sanitization_summary.json" \
  --deletion-log "$RAG_E2_ALIGNED/sanitization/gainratio_deletion_log.jsonl" \
  --output "$budget" \
  2>&1 | tee "$RAG_E3_ALIGNED/logs/00_freeze_aligned_budget.log"

python - "$budget" <<'PY'
import json
import sys

budget = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "input_document_count": 3633,
    "target_modified_document_count": 174,
    "target_total_deleted_chars": 37586,
}
actual = {key: budget.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"ALIGNED BUDGET MISMATCH: expected={expected}, actual={actual}")
print("ALIGNED BUDGET LOCKED:", actual)
PY

for seed in "${RAG_SEEDS[@]}"; do
  condition="$RAG_E3_ALIGNED/seed_${seed}"
  corpus_dir="$condition/sanitization"
  database="$RAG_E2E/vectorstores/e3_random_aligned_v2_seed_${seed}"
  build_summary="$RAG_E2E/vectorstores/e3_random_aligned_v2_seed_${seed}_build_summary.json"
  basepath="$condition/basepath"
  verify="$basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"

  mkdir -p "$condition/logs" "$condition/metrics"

  if [[ ! -f "$corpus_dir/random_sanitization_summary.json" ]]; then
    python "$RAG_E3_PIPE/01_build_random_sanitized_corpus.py" \
      --input "$RAG_ATTACK_INPUT" \
      --budget "$budget" \
      --output-dir "$corpus_dir" \
      --seed "$seed" \
      --reset \
      2>&1 | tee "$condition/logs/01_random_sanitize.log"
  else
    echo "[SKIP] seed $seed random corpus already complete"
  fi

  python "$RAG_E3_PIPE/02_posthoc_random_ground_truth_audit.py" \
    --original "$RAG_ATTACK_INPUT" \
    --sanitized "$corpus_dir/random_sanitized_corpus.jsonl" \
    --inject-json "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
    --output "$condition/metrics/posthoc_random_ground_truth_audit.json" \
    2>&1 | tee "$condition/logs/02_posthoc_audit.log"

  if [[ ! -f "$build_summary" ]]; then
    python "$RAG_E3_PIPE/03_build_random_vectorstore.py" \
      --repo "$RAG_REPO" \
      --input "$corpus_dir/random_sanitized_corpus.jsonl" \
      --db "$database" \
      --reset \
      2>&1 | tee "$condition/logs/03_build_vectorstore.log"
  else
    echo "[SKIP] seed $seed vector store already complete"
  fi

  mkdir -p \
    "$basepath/wm_prepare/nfcorpus" \
    "$basepath/wm_generate/nfcorpus/10" \
    "$basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10"
  cp "$RAG_E0/basepath/wm_prepare/nfcorpus/wmunit.json" \
    "$basepath/wm_prepare/nfcorpus/wmunit.json"
  cp "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_doc.json" \
    "$basepath/wm_generate/nfcorpus/10/wmuint_doc.json"
  cp "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
    "$basepath/wm_generate/nfcorpus/10/wmuint_inject.json"

  e0_verify="$RAG_E0/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
  verify_state=$(verification_status "$verify" "$e0_verify")
  if [[ "$verify_state" == "partial" ]]; then
    echo "[RESUME] seed $seed verification is missing or incomplete"
    (
      cd "$RAG_REPO"
      export RAGWM_CHROMA_PATH="$database"
      python src/main.py \
        --eval_dataset nfcorpus \
        --eval_model_code contriever \
        --score_function cosine \
        --top_k 5 \
        --mutual_times 10 \
        --model_name_llm gpt4o_mini_e2e \
        --model_name_rllm gpt4o_mini_e2e \
        --gpu_id 0 \
        --basepath "$basepath" \
        --verify_num 30 \
        --verify_seed 633 \
        --doc 0 --inject 0 --verify 1 --stat 0
    ) 2>&1 | tee "$condition/logs/04_verify.log"
    verify_state=$(verification_status "$verify" "$e0_verify")
  fi

  if [[ "$verify_state" == "complete" ]]; then
    echo "[PASS] seed $seed verification contains the frozen 30 units"
  elif [[ "$verify_state" == "invalid" ]]; then
    echo "INVALID VERIFICATION FILE: $verify" >&2
    echo "Inspect it before any rerun; it was not deleted or overwritten." >&2
    exit 1
  else
    echo "INCOMPLETE VERIFICATION FILE AFTER RUN: $verify" >&2
    exit 1
  fi
done

python "$RAG_E3_PIPE/06_compare_e0_e1_e2_e3.py" \
  --e0 "$RAG_E0/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e1 "$RAG_E1/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e2 "$RAG_E2_ALIGNED/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e3-root "$RAG_E3_ALIGNED" \
  --seeds "${RAG_SEEDS[@]}" \
  --output "$RAG_E3_ALIGNED/e0_e1_e2_aligned_e3_comparison.json" \
  2>&1 | tee "$RAG_E3_ALIGNED/logs/06_compare.log"

echo "ALIGNED E3 PIPELINE COMPLETE"
echo "$RAG_E3_ALIGNED/e0_e1_e2_aligned_e3_comparison.json"

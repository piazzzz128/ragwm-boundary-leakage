#!/usr/bin/env bash
set -euo pipefail

# Phase A for NFCorpus token-aligned E3.
# Freezes the aligned E2 budget, creates five random sanitized corpora, and
# performs post-hoc watermark audits. This phase does not build vector stores
# and does not call the verifier API.

RAG_STORAGE=${RAG_STORAGE:-/root/autodl-tmp/ragwm_storage}
RAG_REPO=${RAG_REPO:-$RAG_STORAGE/repo/ragwm_src}
RAG_E2E=${RAG_E2E:-$RAG_STORAGE/output/sanitization_e2e_nf_strict_v1}
RAG_E0=${RAG_E0:-$RAG_E2E/conditions/e0_watermarked_before}
RAG_E2_STRICT=${RAG_E2_STRICT:-$RAG_E2E/conditions/e2_gainratio_delete}
RAG_E2_ALIGNED=${RAG_E2_ALIGNED:-$RAG_E2E/conditions/e2_gainratio_aligned_sensitivity}
RAG_E3_ALIGNED=${RAG_E3_ALIGNED:-$RAG_E2E/conditions/e3_random_aligned_v2}
RAG_E3_PIPE=${RAG_E3_PIPE:-$RAG_REPO/tools/e3_random_utility_pipeline}
RAG_ATTACK_INPUT=${RAG_ATTACK_INPUT:-$RAG_E2_STRICT/corpus/watermarked_attack_input.jsonl}
RAG_SEEDS=(101 202 303 404 505)

export PYTHONPATH="$RAG_E3_PIPE:$RAG_REPO"
export PYTHONUNBUFFERED=1

required_files=(
  "$RAG_ATTACK_INPUT"
  "$RAG_E2_ALIGNED/sanitization/gainratio_sanitization_summary.json"
  "$RAG_E2_ALIGNED/sanitization/gainratio_deletion_log.jsonl"
  "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json"
  "$RAG_E3_PIPE/00_freeze_e2_budget.py"
  "$RAG_E3_PIPE/01_build_random_sanitized_corpus.py"
  "$RAG_E3_PIPE/02_posthoc_random_ground_truth_audit.py"
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

with open(sys.argv[1], encoding="utf-8") as handle:
    budget = json.load(handle)
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
    echo "[SKIP] seed $seed random corpus already exists"
  fi

  python "$RAG_E3_PIPE/02_posthoc_random_ground_truth_audit.py" \
    --original "$RAG_ATTACK_INPUT" \
    --sanitized "$corpus_dir/random_sanitized_corpus.jsonl" \
    --inject-json "$RAG_E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
    --output "$condition/metrics/posthoc_random_ground_truth_audit.json" \
    2>&1 | tee "$condition/logs/02_posthoc_audit.log"
done

python - "$RAG_E3_ALIGNED" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for seed in (101, 202, 303, 404, 505):
    with (root / f"seed_{seed}/sanitization/random_sanitization_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (root / f"seed_{seed}/metrics/posthoc_random_ground_truth_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)["summary"]
    row = {
        "seed": seed,
        "modified_documents": summary["modified_document_count"],
        "target_deleted_chars": summary["target_total_deleted_chars"],
        "actual_deleted_chars": summary["actual_total_deleted_chars"],
        "relative_budget_error": summary["relative_total_budget_error"],
        "budget_match_pass": summary["budget_match_pass"],
        "randomly_hit_watermarked_documents": audit["randomly_hit_watermarked_documents"],
        "exact_watermark_occurrence_removal_rate": audit["exact_watermark_occurrence_removal_rate"],
        "corpus_sha256": summary["corpus_sha256"],
    }
    rows.append(row)
    if row["modified_documents"] != 174 or not row["budget_match_pass"]:
        raise SystemExit(f"PREFLIGHT FAILED: {row}")

report = {
    "experiment": "NF-ALIGNED-E3-V2 preflight",
    "api_calls_performed": False,
    "vectorstores_built": False,
    "runs": rows,
}
output = root / "aligned_e3_preflight_summary.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("ALIGNED E3 PREFLIGHT PASS")
print(output)
PY

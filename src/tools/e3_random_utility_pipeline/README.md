# E3 Random Baselines and Retrieval Utility

This package continues the NFCorpus end-to-end RAG-WM sanitization experiment after:

- E0 Watermarked-before: WSN = 26
- E1 Oracle clean: WSN = 5
- E2 GainRatio: WSN = 17

## Scientific purpose

E3 tests whether GainRatio performs better than deletion with the same budget.

Each random seed:

1. Uses no watermark labels or injection locations for selection.
2. Modifies exactly the same number of documents as E2.
3. Approximately matches E2's per-document deleted-character distribution.
4. Deletes only complete sentence-boundary suffixes.
5. Rebuilds an independent Chroma database.
6. Uses the same 30 watermark units, verifier, top-k, and verification seed.

Recommended seeds: `101 202 303 404 505`.

## Standard paths

```bash
export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export E0=$E2E/conditions/e0_watermarked_before
export E2=$E2E/conditions/e2_gainratio_delete
export E3ROOT=$E2E/conditions/e3_random_baselines
export PIPE=$REPO/tools/e3_random_utility_pipeline

mkdir -p "$E3ROOT/budget" "$E3ROOT/logs"
cd "$PIPE"

export PYTHONPATH="$PIPE:$REPO"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
```

## 1. Freeze E2 deletion budget

```bash
python 00_freeze_e2_budget.py \
  --summary "$E2/sanitization/gainratio_sanitization_summary.json" \
  --deletion-log "$E2/sanitization/gainratio_deletion_log.jsonl" \
  --output "$E3ROOT/budget/e3_budget_manifest.json" \
  2>&1 | tee "$E3ROOT/logs/00_freeze_budget.log"
```

Check:

```bash
cat "$E3ROOT/budget/e3_budget_manifest.json"
```

## 2. Run one random seed first

```bash
bash 04_run_single_random_seed.sh 101
```

Inspect the budget match:

```bash
python -m json.tool \
"$E3ROOT/seed_101/sanitization/random_sanitization_summary.json"
```

Required:

- `modified_document_count_match: true`
- `budget_match_pass: true`
- `relative_total_budget_error <= 0.10`
- verification has 30 rows and no unknown outputs.

## 3. Run the remaining four seeds

```bash
bash 05_run_five_random_seeds.sh
```

The script is resumable and skips completed artifacts.

## 4. Compare E0/E1/E2/E3

```bash
python 06_compare_e0_e1_e2_e3.py \
  --e0 "$E0/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e1 "$E2E/conditions/e1_oracle_restore/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e2 "$E2/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e3-root "$E3ROOT" \
  --seeds 101 202 303 404 505 \
  --output "$E3ROOT/e0_e1_e2_e3_comparison.json" \
  2>&1 | tee "$E3ROOT/logs/06_compare.log"
```

The key result is:

- `gainratio_WSN_advantage_vs_random_mean > 0`: GainRatio is better.
- `gainratio_beats_random_seed_count`: number of random seeds with higher WSN than E2.
- `random_WSN_mean ± random_WSN_std`.

## 5. Retrieval utility

Run after E3 vector stores exist:

```bash
export SNAPROOT=$E2E/restored/strict_snapshot/root/autodl-tmp/ragwm_storage

python 07_evaluate_retrieval_utility.py \
  --repo "$REPO" \
  --db "e0_watermarked=$SNAPROOT/chromadb_db" \
  --db "e1_oracle=$E2E/vectorstores/e1_oracle_clean" \
  --db "e2_gainratio=$E2E/vectorstores/e2_gainratio_delete" \
  --db "e3_random_101=$E2E/vectorstores/e3_random_seed_101" \
  --db "e3_random_202=$E2E/vectorstores/e3_random_seed_202" \
  --db "e3_random_303=$E2E/vectorstores/e3_random_seed_303" \
  --db "e3_random_404=$E2E/vectorstores/e3_random_seed_404" \
  --db "e3_random_505=$E2E/vectorstores/e3_random_seed_505" \
  --baseline-name e0_watermarked \
  --output "$E3ROOT/retrieval_utility.json" \
  2>&1 | tee "$E3ROOT/logs/07_retrieval_utility.log"
```

Report:

- nDCG@10
- MAP@100
- Recall@100
- MRR@10
- metric deltas versus E0
- top-10 and top-100 retrieval overlap versus E0

## Interpretation

E2 is supported when:

1. E2 WSN is lower than the random mean.
2. E2 beats most or all random seeds.
3. E2 has higher exact-watermark removal than random.
4. Retrieval utility remains close to E0.

Do not replace the fixed E2 threshold after observing E3.

# NFCorpus E2 GainRatio-guided sanitization

This package reproduces the uploaded `nfcorpus_strict_qwen_gainratio.py` scoring behavior exactly, then applies a blind iterative suffix-deletion policy to all 3,633 watermarked NFCorpus documents.

## Important methodological note

The uploaded strict-v1 scorer is **not fully aligned on the first target token**: target-only NLL skips the first target token, while context-conditioned NLL includes it. This package deliberately preserves that behavior for direct comparability with the existing strict-v1 ROC/PR results. Do not describe this implementation as “aligned target-token scoring” in the final paper. A corrected aligned sensitivity run should be added after the primary E2/E3 pipeline.

The attack selection input excludes `change`, `label`, `tuple`, and `watermark_text`. Ground truth is used only by `04_posthoc_ground_truth_audit.py` after sanitization is complete.

## Expected AutoDL paths

```bash
export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export E2=$E2E/conditions/e2_gainratio_delete
export E0=$E2E/conditions/e0_watermarked_before
export SNAPROOT=$E2E/restored/strict_snapshot/root/autodl-tmp/ragwm_storage
export PIPE=$REPO/tools/e2_gainratio_full_pipeline
```

## 1. Preflight

```bash
cd "$PIPE"
python 00_preflight.py \
  --watermarked-db "$SNAPROOT/chromadb_db" \
  --strict-scores "$STORAGE/output/semantic_cliff/nfcorpus_strict_qwen_gainratio_scores.json" \
  | tee "$E2/logs/00_preflight.log"
```

## 2. Export blind attack input

```bash
python 01_export_watermarked_corpus.py \
  --db "$SNAPROOT/chromadb_db" \
  --output-dir "$E2/corpus" \
  | tee "$E2/logs/01_export.log"
```

## 3. Freeze threshold before E2 verification

```bash
python 02_calibrate_threshold.py \
  --scores "$STORAGE/output/semantic_cliff/nfcorpus_strict_qwen_gainratio_scores.json" \
  --output "$E2/scores/fixed_threshold_fpr05.json" \
  | tee "$E2/logs/02_threshold.log"
```

Do not alter the threshold after seeing E2 WSN.

## 4. Twenty-document smoke run

```bash
rm -rf "$E2E/conditions/e2_gainratio_smoke"
python 03_gainratio_iterative_sanitize.py \
  --input "$E2/corpus/watermarked_attack_input.jsonl" \
  --threshold-json "$E2/scores/fixed_threshold_fpr05.json" \
  --output-dir "$E2E/conditions/e2_gainratio_smoke" \
  --limit 20 --max-steps 8 --reset \
  | tee "$E2/logs/03_smoke.log"
```

A smoke pass means 20 completed records and no exception. The number modified is informational.

## 5. Formal full-corpus sanitization

```bash
python 03_gainratio_iterative_sanitize.py \
  --input "$E2/corpus/watermarked_attack_input.jsonl" \
  --threshold-json "$E2/scores/fixed_threshold_fpr05.json" \
  --output-dir "$E2/sanitization" \
  --max-steps 8 --reset \
  | tee "$E2/logs/03_formal_sanitize.log"
```

The script writes one atomic JSON record per document and can be resumed after interruption by rerunning without `--reset`.

## 6. Post-hoc audit only after formal completion

```bash
python 04_posthoc_ground_truth_audit.py \
  --original "$E2/corpus/watermarked_attack_input.jsonl" \
  --sanitized "$E2/sanitization/gainratio_sanitized_corpus.jsonl" \
  --inject-json "$E0/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
  --output "$E2/metrics/posthoc_ground_truth_audit.json" \
  | tee "$E2/logs/04_posthoc_audit.log"
```

## 7. Build isolated E2 vector store

```bash
python 05_build_sanitized_vectorstore.py \
  --repo "$REPO" \
  --input "$E2/sanitization/gainratio_sanitized_corpus.jsonl" \
  --db "$E2E/vectorstores/e2_gainratio_delete" \
  --reset \
  | tee "$E2/logs/05_build_vectorstore.log"
```

Expected count: 3,633.

## 8. Run the same 30 RAG-WM verification queries

```bash
bash 07_run_e2_verification.sh
```

## 9. Compare E0, E1, and E2

```bash
python 06_compare_e0_e1_e2.py \
  --e0 "$E0/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e1 "$E2E/conditions/e1_oracle_restore/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e2 "$E2/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$E2/metrics/e0_e1_e2_comparison.json" \
  | tee "$E2/logs/06_compare.log"
```

Primary interpretation:

- `e2_gainratio_WSN`
- `e2_paired_attack_success_rate`
- `oracle_gap_closure = (26 - WSN_E2) / (26 - 5)`
- document precision/recall and exact watermark removal from the post-hoc audit

Random deletion and retrieval utility should be run only after E2 output and deletion budget are frozen.

# NFCorpus aligned end-to-end E2 sensitivity pipeline

This package performs a **sensitivity analysis**, not a replacement main
experiment.

## Fixed elements

The following are preserved from the sealed strict-v1 E2 pipeline:

- the same 3,633-document watermarked attack input;
- Qwen2.5-7B;
- sentence splitting and last-eligible-sentence selection;
- context tail of at most 1,200 characters;
- 1,024 context tokens and 128 target tokens;
- minimum target length of 25 characters;
- no BOS token and no separator;
- iterative suffix deletion with `max_steps=8`;
- the same 70/30 clean-group calibration split;
- seed `20260714`;
- target clean FPR of 5%;
- Contriever, cosine similarity and top-k=5;
- the same 30 verification units and `verify_seed=633`;
- gpt-4o-mini with the existing formal configuration.

## Only scoring change

The sealed strict-v1 scorer computes:

- target-only NLL over target tokens 2..n;
- conditioned NLL over target tokens 1..n.

This package computes both losses over target tokens 2..n. It does not add BOS
or separator tokens.

## Output isolation

All outputs go to:

```text
/root/autodl-tmp/ragwm_storage/output/sanitization_e2e_nf_strict_v1/
  conditions/e2_gainratio_aligned_sensitivity
```

The vector store goes to:

```text
.../vectorstores/e2_gainratio_aligned_sensitivity
```

The sealed strict-v1 E2 directory is read but never reset or overwritten.

## Installation

Upload and unzip this package, then place the directory at:

```text
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/
  e2_gainratio_aligned_sensitivity_pipeline
```

Set executable permissions:

```bash
chmod +x 07_run_aligned_verification.sh run_aligned_pipeline.sh
```

## Stage A: preflight, threshold and smoke run

```bash
cd /root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/e2_gainratio_aligned_sensitivity_pipeline
bash run_aligned_pipeline.sh
```

Check that:

- preflight reports 3,633 documents;
- aligned scores contain 223 clean and 223 injected rows;
- no row lacks `score_aligned`;
- smoke run completes 20 records without exception.

## Stage B: full-corpus aligned sanitization

Run only after the smoke output is valid:

```bash
export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export EA=$E2E/conditions/e2_gainratio_aligned_sensitivity
export STRICT_E2=$E2E/conditions/e2_gainratio_delete
export PIPEA=$REPO/tools/e2_gainratio_aligned_sensitivity_pipeline

cd "$PIPEA"

python 03_gainratio_iterative_sanitize_aligned.py \
  --input "$STRICT_E2/corpus/watermarked_attack_input.jsonl" \
  --threshold-json "$EA/scores/fixed_threshold_fpr05_aligned.json" \
  --output-dir "$EA/sanitization" \
  --max-steps 8 --reset \
  2>&1 | tee "$EA/logs/03_formal_sanitize.log"
```

The formal run is resumable. If interrupted after records have been written,
rerun the same command **without** `--reset`.

Expected completion:

```text
completed_document_count = 3633
complete = true
ALIGNED GAINRATIO SANITIZATION PASS
```

Do not expect the modified-document count to equal the strict-v1 value of 529.

## Stage C: post-hoc ground-truth audit

```bash
python 04_posthoc_ground_truth_audit_aligned.py \
  --original "$STRICT_E2/corpus/watermarked_attack_input.jsonl" \
  --sanitized "$EA/sanitization/gainratio_sanitized_corpus.jsonl" \
  --inject-json "$E2E/conditions/e0_watermarked_before/basepath/wm_generate/nfcorpus/10/wmuint_inject.json" \
  --output "$EA/metrics/posthoc_ground_truth_audit_aligned.json" \
  2>&1 | tee "$EA/logs/04_posthoc_audit.log"
```

## Stage D: build an isolated aligned vector store

```bash
python 05_build_sanitized_vectorstore_aligned.py \
  --repo "$REPO" \
  --input "$EA/sanitization/gainratio_sanitized_corpus.jsonl" \
  --db "$E2E/vectorstores/e2_gainratio_aligned_sensitivity" \
  --summary-output "$EA/metrics/aligned_vectorstore_build_summary.json" \
  --reset \
  2>&1 | tee "$EA/logs/05_build_vectorstore.log"
```

Expected collection count: 3,633.

## Stage E: run the same RAG-WM verification

```bash
bash 07_run_aligned_verification.sh
```

This step uses the same 30 units and `verify_seed=633`.

## Stage F: compare against E0 and Oracle

```bash
python 06_compare_e0_e1_aligned.py \
  --e0 "$E2E/conditions/e0_watermarked_before/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e1 "$E2E/conditions/e1_oracle_restore/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --aligned "$EA/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$EA/metrics/e0_e1_aligned_comparison.json" \
  2>&1 | tee "$EA/logs/06_compare.log"
```

## Files to upload after completion

Upload these four files:

```text
$EA/scores/fixed_threshold_fpr05_aligned.json
$EA/sanitization/gainratio_sanitization_summary.json
$EA/metrics/posthoc_ground_truth_audit_aligned.json
$EA/metrics/e0_e1_aligned_comparison.json
```

The retrieval-utility sensitivity run should be performed after the aligned WSN
and audit outputs are frozen.

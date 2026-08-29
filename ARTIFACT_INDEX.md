# Artifact Index

This document maps the major experimental claims to the corresponding frozen artifacts in this repository.

## 1. Current Formal Verification Campaign

Primary path:

```text
results/formal/gainratio_random_final_verifier_v1_20260829/
```

Primary report:

```text
gainratio_final_verifier_report.json
```

Key reported values:

| Condition | WSN |
|---|---:|
| Watermarked | 26 / 30 |
| Oracle restore | 4 / 30 |
| GainRatio @ 5% clean FPR | 21 / 30 |

Matched-random WSN:

```text
[22, 25, 25, 22, 24]
mean = 23.6
```

Integrity record:

```text
SHA256SUMS_formal_results.txt
```

Supporting evidence:

```text
supporting_audits/
verifier_campaign/
scripts/
```

---

## 2. Conditional-NLL End-to-End Extension

Path:

```text
results/extensions/conditional_nll_e2e_nf_v1_20260828/
```

Key metric files:

```text
metrics/conditional_nll_complete_report.json
metrics/retrieval_utility_frozen.json
```

Additional evidence:

```text
localization/
thresholds/
scan_records/
conditions/
verifier_refresh/
audit/
logs/
```

Integrity file:

```text
SHA256SUMS_formal_results.txt
```

---

## 3. Cross-Detector Robustness

Path:

```text
results/extensions/cross_detector_lm_robustness_v1/
```

Sub-experiments include:

```text
deepseek_llm_7b_base/
gemma2_2b_base/
qwen2_5_14b_base/
pairwise_qwen_deepseek_v1/
pairwise_qwen_deepseek_v2/
```

These artifacts support detector-LM robustness and cross-detector analyses.

---

## 4. Multi-Budget Evaluation

Path:

```text
results/extensions/sanitization_multibudget_nf_aligned_v1/
```

Used for analyses of threshold/deletion-budget sensitivity.

---

## 5. Semantic-Cliff / Boundary Analysis

Path:

```text
results/extensions/semantic_cliff/
```

Used for supporting boundary-discontinuity analyses.

---

## 6. Historical NFCorpus strict-v1

Path:

```text
results/historical/sanitization_e2e_nf_strict_v1/
```

Important subdirectories:

```text
audit/
audit_inventory/
conditions/
logs/
restored/
```

Historical representative WSN:

```text
E0 = 26 / 30
E1 = 5 / 30
E2 = 17 / 30
```

Historical matched-random:

```text
[23, 24, 24, 23, 23]
```

These results belong to the historical sealed strict-v1 protocol and must not be substituted for the current formal verifier campaign.

---

## 7. Historical TREC-COVID End-to-End Extension

Path:

```text
results/historical/sanitization_e2e_trec_covid_aligned_v1/
```

Retained for provenance and protocol history.

---

## 8. Repository-Wide Integrity Records

```text
manifests/EXPORT_SHA256SUMS.txt
manifests/ORIGINAL_OUTPUT_SHA256SUMS.txt
manifests/FILE_SIZES_BYTES.tsv
manifests/FILES_OVER_99MB.tsv
manifests/REDACTED_FILES.txt
manifests/SECRET_SCAN.txt
```

---

## Evidence Priority

When interpreting results:

```text
Frozen machine output
    >
Experiment-local SHA-256
    >
Formal report
    >
Campaign manifest / audit
    >
Extension artifact
    >
Historical artifact
    >
README summary
```

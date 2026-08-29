# RAG-WM Boundary Leakage

### GainRatio-Guided Targeted Suffix Deletion for Authorized Red-Team Evaluation

[简体中文](README_zh-CN.md)

This repository contains the research code, experimental artifacts, frozen machine-generated outputs, audit records, and reproducibility materials for our study of **boundary leakage in retrieval-augmented generation (RAG) watermarking**.

The central question is:

> When generated watermark text is appended to an original document, does the context–watermark join expose a statistical signal that can be detected without access to the watermark key?

We study this question using **conditional negative log-likelihood (Conditional NLL)**, **DeltaLoss**, and **GainRatio**, and evaluate a GainRatio-guided **targeted suffix deletion** procedure under an **authorized corpus-editing threat model**.

This repository intentionally separates:

- current formal verification results;
- aligned and robustness extensions;
- historical sealed experiments;
- source code;
- frozen machine outputs;
- SHA-256 integrity manifests;
- environment snapshots;
- redaction records.

> **Important:** Results produced under different experimental protocols must not be silently merged. When a prose summary conflicts with a frozen machine-generated report, the machine-generated report and its corresponding integrity manifest take precedence.

---

## Contents

- [Research Question](#research-question)
- [Threat Model](#threat-model)
- [Boundary Leakage](#boundary-leakage)
- [GainRatio](#gainratio)
- [Method Overview](#method-overview)
- [Experimental Scope](#experimental-scope)
- [Result Hierarchy](#result-hierarchy)
- [Current Formal Verification Results](#current-formal-verification-results)
- [Boundary Localization](#boundary-localization)
- [Historical strict-v1 Results](#historical-strict-v1-results)
- [Extensions](#extensions)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Original RAG-WM Pipeline](#original-rag-wm-pipeline)
- [Reproducibility and Integrity](#reproducibility-and-integrity)
- [Environment](#environment)
- [Protocol Notes](#protocol-notes)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Citation](#citation)
- [Licensing](#licensing)

---

## Research Question

RAG watermarking can support corpus ownership verification by inserting or appending generated watermark-related text into a retrieval corpus.

This study investigates whether that construction introduces an unintended side channel at the join between:

```text
Original document context | Generated watermark-related text
                          ^
                    candidate boundary
```

Rather than attempting to recover a watermark key, we ask whether a language model can detect this boundary from conditional likelihood statistics alone.

---

## Threat Model

We consider an:

**Authorized Corpus-Editing Threat Model**

The evaluator has legitimate access to an experimental copy of the corpus and may:

- read the corpus;
- compute language-model scores;
- modify authorized corpus copies;
- rebuild the retrieval index;
- run end-to-end verification after candidate selection.

The core candidate-selection procedure does **not** require the watermark key or verifier feedback.

This work should therefore be interpreted as:

> **authorized red-team evaluation of RAG watermark robustness**

rather than unauthorized modification of third-party systems.

---

## Boundary Leakage

Let:

- \(C\) denote the preceding document context;
- \(T\) denote a candidate target span.

We compare the target loss without context,

\[
L(T),
\]

with its loss when conditioned on the preceding context,

\[
L(T\mid C).
\]

If the join between \(C\) and \(T\) exhibits an unusual statistical relationship, conditioning on \(C\) may produce an identifiable change in target-token loss.

We refer to this phenomenon as **boundary leakage**.

---

## GainRatio

We first define:

\[
\Delta L = L(T)-L(T\mid C).
\]

GainRatio normalizes this loss reduction by the unconditional target loss:

\[
\mathrm{GainRatio}
=
\frac{L(T)-L(T\mid C)}{L(T)}.
\]

The score is used for:

1. candidate boundary scoring;
2. clean-data threshold calibration;
3. backward boundary scanning;
4. targeted suffix deletion;
5. retrieval-index rebuilding;
6. end-to-end watermark verification.

---

## Method Overview

```text
Watermarked Corpus
       │
       ▼
Candidate Boundary Extraction
       │
       ▼
 Context C + Target T
       │
       ├──────────────┐
       ▼              ▼
     L(T)          L(T | C)
       │              │
       └──────┬───────┘
              ▼
          GainRatio
              │
              ▼
 Clean-Data Calibration
              │
              ▼
   Backward Boundary Scan
              │
              ▼
 Targeted Suffix Deletion
              │
              ▼
 Rebuild Retrieval Index
              │
              ▼
End-to-End Verification
              │
              ▼
Matched-Random Comparison
```

---

## Experimental Scope

### Datasets

Experiments in this repository involve:

- **NFCorpus**
- **TREC-COVID**
- sampled **Natural Questions**

Different experiment families may use different dataset sizes and construction procedures. The experiment-local manifest or frozen output should be treated as authoritative.

### Retrieval

The original RAG-WM pipeline primarily uses:

- Contriever
- cosine similarity

### Detector / Scoring Language Models

Experiments include variants using:

- DistilGPT2
- GPT-2 Medium
- Qwen2.5-7B
- Qwen2.5-14B
- DeepSeek-LLM-7B-base
- Gemma-2-2B-base

Cross-detector experiments also include pairwise model analyses.

### API-Based Verification

The current formal verifier campaign records:

```text
provider/model: ephone:gpt-4o-mini
repository alias: gpt4o_mini_e2e
```

API-based outputs are preserved as frozen experimental evidence because hosted inference backends may change over time.

---

## Result Hierarchy

The repository deliberately separates results into three categories:

```text
results/
├── formal/
├── extensions/
└── historical/
```

### `formal/`

Current formal public-reporting artifacts.

```text
results/formal/
└── gainratio_random_final_verifier_v1_20260829/
```

### `extensions/`

Supporting and robustness experiments.

```text
results/extensions/
├── conditional_nll_e2e_nf_v1_20260828/
├── cross_detector_lm_robustness_v1/
├── sanitization_multibudget_nf_aligned_v1/
└── semantic_cliff/
```

### `historical/`

Frozen earlier experiments retained for provenance and auditability.

```text
results/historical/
├── sanitization_e2e_nf_strict_v1/
└── sanitization_e2e_trec_covid_aligned_v1/
```

Historical experiments are preserved rather than overwritten by later protocol revisions.

---

# Current Formal Verification Results

The current formal report is:

```text
results/formal/
└── gainratio_random_final_verifier_v1_20260829/
    └── gainratio_final_verifier_report.json
```

The frozen campaign records:

```text
verify_num  = 30
verify_seed = 633
campaign_status = complete
unit_identity_and_order_match = true
all_unknown_counts_zero = true
```

## Primary Conditions

| Condition | WSN |
|---|---:|
| Watermarked | 26 / 30 |
| Oracle restore | 4 / 30 |
| GainRatio @ 5% clean FPR | 21 / 30 |

Thus, in this campaign:

```text
Watermarked → GainRatio
26 → 21
```

For the paired watermarked-vs-GainRatio comparison:

```text
yes_to_no  = 5
no_to_yes  = 0
discordant = 5
McNemar exact two-sided p = 0.0625
```

Accordingly, this result provides directional evidence of reduced verification success, but the corresponding two-sided exact McNemar test does not cross a conventional \(p<0.05\) threshold.

---

## Budget-Matched Random Baselines

Five random seeds were evaluated under a matched deletion budget:

| Seed | WSN |
|---:|---:|
| 101 | 22 / 30 |
| 202 | 25 / 30 |
| 303 | 25 / 30 |
| 404 | 22 / 30 |
| 505 | 24 / 30 |

Summary:

```text
Random WSN = [22, 25, 25, 22, 24]

Mean             = 23.6
Sample std. dev. ≈ 1.5166
Minimum          = 22
Maximum          = 25

GainRatio WSN    = 21
```

All five random-baseline WSN values are above the GainRatio WSN in this fixed campaign.

However, per-seed paired comparisons should remain descriptive unless their exact paired tests support a stronger statistical claim.

---

## Deletion Budget

The formal GainRatio condition records:

```text
modified_documents        = 174
target_deleted_characters = 37,586
random_character_tolerance = 0.01
random_budget_audit_passed = true
```

The matched-random controls were therefore constructed against the same targeted deletion budget within the specified tolerance.

---

## Claim Boundary

The formal result package explicitly includes the following claim guard:

> Results do not establish ownership-verification bypass.

The supported interpretation is narrower:

> A key-agnostic boundary-localization signal can be detected and, under the evaluated authorized corpus-editing protocol, can affect downstream watermark-verification outcomes.

---

# Boundary Localization

Representative localization results include:

| Dataset / Protocol | ROC-AUC | PR-AUC |
|---|---:|---:|
| TREC-COVID independent boundary set | 0.9245 | 0.9279 |
| TREC-COVID aligned, Qwen2.5-7B | 0.9779 | 0.9781 |
| NFCorpus aligned, Qwen2.5-7B | 0.8346 | 0.8034 |

These experiments support the existence of detectable boundary-level statistical structure across multiple settings.

They should not be treated as a single pooled experiment because their construction and scoring protocols differ.

---

# Historical strict-v1 Results

The historical sealed NFCorpus experiment is preserved under:

```text
results/historical/
└── sanitization_e2e_nf_strict_v1/
```

Representative historical results were:

| Historical condition | WSN |
|---|---:|
| E0 watermarked | 26 / 30 |
| E1 oracle/reference | 5 / 30 |
| E2 GainRatio targeted deletion | 17 / 30 |

Historical matched-random results:

```text
[23, 24, 24, 23, 23]
mean = 23.4
std ≈ 0.55
```

These values belong specifically to the **historical sealed strict-v1 protocol**.

They must not be silently substituted for the current formal result:

```text
GainRatio @ 5% clean FPR = 21 / 30
```

The two result families are retained separately to preserve experimental provenance.

---

# Extensions

## Conditional-NLL End-to-End

```text
results/extensions/
└── conditional_nll_e2e_nf_v1_20260828/
```

Contains:

```text
audit/
conditions/
localization/
logs/
metrics/
scan_records/
thresholds/
verifier_refresh/
SHA256SUMS_formal_results.txt
```

Important metric artifacts include:

```text
metrics/conditional_nll_complete_report.json
metrics/retrieval_utility_frozen.json
```

---

## Cross-Detector Robustness

```text
results/extensions/
└── cross_detector_lm_robustness_v1/
```

Public experiment families include:

```text
deepseek_llm_7b_base/
gemma2_2b_base/
pairwise_qwen_deepseek_v1/
pairwise_qwen_deepseek_v2/
qwen2_5_14b_base/
```

These experiments test whether the observed boundary signal is tied to a single detector language model.

---

## Multi-Budget Analysis

```text
results/extensions/
└── sanitization_multibudget_nf_aligned_v1/
```

This experiment family studies how threshold and deletion-budget choices affect:

- corpus modification;
- watermark verification;
- retrieval utility.

---

## Semantic-Cliff Analysis

```text
results/extensions/
└── semantic_cliff/
```

Contains supporting experiments related to boundary discontinuity and semantic-cliff behavior.

---

# Repository Structure

```text
ragwm-boundary-leakage/
│
├── README.md
├── README_zh-CN.md
├── .gitattributes
│
├── src/
│   ├── README.md
│   ├── requirement.txt
│   ├── analysis/
│   ├── contriever_src/
│   ├── entity_generate/
│   ├── model_configs/
│   ├── rag/
│   ├── src/
│   └── tools/
│
├── results/
│   ├── formal/
│   │   └── gainratio_random_final_verifier_v1_20260829/
│   ├── extensions/
│   │   ├── conditional_nll_e2e_nf_v1_20260828/
│   │   ├── cross_detector_lm_robustness_v1/
│   │   ├── sanitization_multibudget_nf_aligned_v1/
│   │   └── semantic_cliff/
│   └── historical/
│       ├── sanitization_e2e_nf_strict_v1/
│       └── sanitization_e2e_trec_covid_aligned_v1/
│
├── environment/
│   ├── python_packages.txt
│   └── system_environment.txt
│
└── manifests/
    ├── EXPORT_SHA256SUMS.txt
    ├── ORIGINAL_OUTPUT_SHA256SUMS.txt
    ├── FILE_SIZES_BYTES.tsv
    ├── FILES_OVER_99MB.tsv
    ├── REDACTED_FILES.txt
    └── SECRET_SCAN.txt
```

---

# Quick Start

## 1. Install Git LFS

```bash
git lfs install
```

## 2. Clone

```bash
git clone https://github.com/piazzzz128/ragwm-boundary-leakage.git
cd ragwm-boundary-leakage
git lfs pull
```

## 3. Install the original source dependencies

```bash
cd src
pip install -r requirement.txt
```

The original dependency file is currently named `requirement.txt`.

The full public environment snapshot is also available at:

```text
environment/python_packages.txt
```

---

# Original RAG-WM Pipeline

The preserved source-level instructions are available in:

```text
src/README.md
```

A typical base workflow is:

### Dataset preparation

```bash
python rag/prepare_data.py
```

### Vector database creation

Example for NFCorpus:

```bash
python rag/vectorstore.py \
  --eval_dataset nfcorpus \
  --eval_model_code contriever \
  --score_function cosine
```

### Entity extraction

```bash
python entity_generate/generate_entity_llm_check.py \
  --eval_dataset nfcorpus \
  --dataset_prob 1
```

### Watermark-unit generation

```bash
python entity_generate/generate_hash_entity.py \
  --eval_dataset nfcorpus \
  -t scratch \
  --entity_num 100 \
  --edge_prob 0.05
```

### Watermark generation

```bash
python src/main.py \
  --eval_dataset nfcorpus \
  --eval_model_code contriever \
  --score_function cosine \
  --doc 1 \
  --inject 0 \
  --verify 0 \
  --stat 0 \
  --mutual_times 10
```

### Injection

```bash
python src/main.py \
  --eval_dataset nfcorpus \
  --eval_model_code contriever \
  --score_function cosine \
  --doc 0 \
  --inject 1 \
  --verify 0 \
  --stat 0 \
  --mutual_times 10
```

### Verification

```bash
python src/main.py \
  --eval_dataset nfcorpus \
  --eval_model_code contriever \
  --score_function cosine \
  --doc 0 \
  --inject 0 \
  --verify 1 \
  --stat 1 \
  --mutual_times 10
```

Historical code may contain environment-specific absolute paths; these must be changed to the local working directory before rerunning experiments.

---

# Reproducibility and Integrity

This repository uses multiple layers of provenance records.

## Export checksums

```text
manifests/EXPORT_SHA256SUMS.txt
```

## Original-output checksums

```text
manifests/ORIGINAL_OUTPUT_SHA256SUMS.txt
```

## File-size inventory

```text
manifests/FILE_SIZES_BYTES.tsv
```

## Large-file inventory

```text
manifests/FILES_OVER_99MB.tsv
```

## Public-export redaction record

```text
manifests/REDACTED_FILES.txt
```

## Secret-scan record

```text
manifests/SECRET_SCAN.txt
```

Experiment-local formal packages also contain their own SHA-256 manifests.

For example:

```text
results/formal/gainratio_random_final_verifier_v1_20260829/
└── SHA256SUMS_formal_results.txt
```

---

# Git LFS

A large historical ChromaDB binary artifact is managed through Git LFS.

After cloning:

```bash
git lfs pull
git lfs ls-files
```

Without the LFS object, the corresponding file may remain only as an LFS pointer.

---

# Environment

The public export contains:

```text
environment/python_packages.txt
environment/system_environment.txt
```

The export-time system snapshot records approximately:

```text
Python         3.12.3
CUDA           13.2
NVIDIA driver  595.58.03
GPU            NVIDIA RTX PRO 6000 Blackwell
```

This is an **export-time environment snapshot** and does not imply that every historical experiment was executed under an identical hardware/software stack.

For experiment-specific provenance, consult the local:

- logs;
- preflight records;
- manifests;
- model metadata;
- frozen result files.

---

# Protocol Notes

## strict-v1

`strict-v1` refers to a historical sealed experiment/protocol state.

It is retained for provenance and should not be retroactively overwritten.

## aligned

Later experiments use more harmonized scoring conventions, including treatment of:

- scoring ranges;
- target tokens;
- conditional/unconditional losses;
- BOS/separator behavior;
- first-target-token handling.

Aligned extensions are primarily used for:

- detector robustness;
- model-scale analysis;
- multi-budget experiments;
- additional datasets.

`strict-v1` and `aligned` are protocol labels, not aliases for E0 and E2.

---

# Limitations

The current repository has several important limitations.

1. **Multiple protocols coexist.**  
   Historical strict-v1, aligned extensions, and the current formal campaign must be interpreted separately.

2. **Hosted verifier models can change.**  
   Frozen outputs provide stronger provenance than future API reruns.

3. **The repository is primarily a research artifact release.**  
   It is not yet a fully engineered one-command Python package.

4. **Some sensitive configuration files were intentionally redacted.**

5. **Full reproduction may require large pretrained models, BEIR datasets, substantial GPU memory, and authorized API access.**

6. **The reported experiments do not establish a universal ownership-verification bypass.**

---

# Responsible Use

This repository is intended for:

- academic research;
- authorized security evaluation;
- watermark robustness analysis;
- reproducibility research;
- defensive system evaluation.

Users should:

- test only corpora and systems they are authorized to modify;
- respect dataset and model licenses;
- avoid unauthorized corpus manipulation;
- report protocol differences transparently;
- avoid extrapolating the results to all watermarking systems.

---

# Evidence Priority

When different artifacts appear to disagree, use the following evidence order:

```text
1. Frozen machine-generated outputs
2. Experiment-local SHA-256 manifests
3. Formal metric/verifier reports
4. Campaign manifests and supporting audits
5. Extension results
6. Historical archived outputs
7. README or other prose summaries
```

Historical evidence should not be silently rewritten to match newer protocol variants.

---

# Citation

A formal paper citation will be added after publication.

For now:

```bibtex
@misc{ragwm_boundary_leakage_2026,
  title        = {RAG-WM Boundary Leakage: GainRatio-Guided Targeted Suffix Deletion},
  year         = {2026},
  howpublished = {\url{https://github.com/piazzzz128/ragwm-boundary-leakage}},
  note         = {Research code, experimental artifacts, and reproducibility evidence}
}
```

---

# Licensing

This repository currently does **not** declare a software license for the complete codebase.

Parts of the source tree originate from or build on the public RAG-WM research codebase. The upstream repository did not declare a repository license at the time of this release.

Accordingly, public visibility of this repository should not be interpreted as an unrestricted grant to copy, modify, redistribute, or commercially reuse all source files.

A formal licensing decision should be made only after the licensing status of upstream code and third-party components has been resolved.

---

# Version

Initial public research export:

```text
2026-08-29
```

Initial public repository commit:

```text
fd5f665d
Initial public release: RAG-WM boundary leakage experiments
```

---

## Repository

https://github.com/piazzzz128/ragwm-boundary-leakage

**Research focus:** RAG Watermarking · Boundary Leakage · Conditional NLL · GainRatio · Targeted Suffix Deletion · Watermark Robustness · Retrieval-Augmented Generation · Authorized Red-Team Evaluation

# NF-ALIGNED-MULTIBUDGET-V1 — pending registry entry

**Role:** Canonical token-aligned NFCorpus multi-budget attack–utility extension.  
**Status:** Pending until all machine-readable outputs and hashes exist.  
**Main-result guard:** Does not replace the sealed strict-v1 main experiment.

## Frozen configuration

- Detector LM: Qwen2.5-7B local model.
- Alignment: exclude the first target token in both `L(T)` and `L(T|C)`; no BOS; no extra separator.
- Calibration: grouped clean split, seed 20260714, 169 calibration records / 164 groups.
- Budget points: clean-calibration target FPR 1%, 2.5%, 5%, 10%.
- Full-corpus scan: 3,633 documents, maximum 8 steps.
- Verifier: 30 frozen units, gpt-4o-mini, temperature 0, seed 633.
- Retrieval: 323 queries, Contriever/cosine, top-100, 5,000 paired bootstrap replicates, seed 20260609.

## Results

| FPR target | Threshold | Modified docs | Deleted chars | WSN | McNemar p vs E0 | nDCG@10 Δ [95% CI] | MAP@100 Δ [95% CI] | Recall@100 Δ [95% CI] | MRR@10 Δ [95% CI] |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1% | -0.13007188598091768 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |
| 2.5% | -0.1640843956343344 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |
| 5% | -0.1832180580405571 | 174 | 37,586 | 21/30 | 0.0625 | 待从冻结 utility JSON 自动带入 | 待从冻结 utility JSON 自动带入 | 待从冻结 utility JSON 自动带入 | 待从冻结 utility JSON 自动带入 |
| 10% | -0.20519773948364803 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |

## Required hashes

- Threshold manifest: 缺失/待核对
- fpr01 sanitized corpus / verifier / comparison: 缺失/待核对
- fpr025 sanitized corpus / verifier / comparison: 缺失/待核对
- frozen fpr05 source records: 待引用现有登记哈希
- fpr10 sanitized corpus / verifier / comparison: 缺失/待核对
- one-pass multi-budget retrieval utility: 缺失/待核对

## Claim boundary

Do not write a curve claim until every point has completed deletion, re-indexing, verification, and retrieval evaluation. Report verification weakening unless a measured WSN crosses the original threshold. An interval containing zero is not evidence of equivalence.


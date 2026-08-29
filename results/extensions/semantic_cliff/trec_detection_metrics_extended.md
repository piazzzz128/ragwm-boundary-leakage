# TREC-COVID Extended Detection Metrics

All metrics are record-level. Scores are oriented so that larger values indicate a higher probability of inject boundary.

## Main Table

| Dataset | Method | Model | ROC-AUC 95% CI | PR-AUC 95% CI | TPR@FPR=5% | TPR@FPR=10% | Best F1 | N |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TREC-COVID | Contriever + Cosine | Contriever | 0.6891 [0.6410, 0.7382] | 0.6641 [0.6117, 0.7223] | 0.1397 | 0.2271 | 0.7099 | 458 |
| TREC-COVID | DeltaLoss | distilgpt2 | 0.7797 [0.7377, 0.8217] | 0.8094 [0.7753, 0.8465] | 0.4541 | 0.4803 | 0.7169 | 458 |
| TREC-COVID | GainRatio | distilgpt2 | 0.8019 [0.7610, 0.8403] | 0.8234 [0.7885, 0.8587] | 0.4585 | 0.5022 | 0.7337 | 458 |
| TREC-COVID | DeltaLoss | gpt2-medium | 0.8443 [0.8070, 0.8786] | 0.8770 [0.8483, 0.9026] | 0.5677 | 0.6245 | 0.7642 | 458 |
| TREC-COVID | GainRatio | gpt2-medium | 0.8772 [0.8438, 0.9065] | 0.8995 [0.8724, 0.9222] | 0.6026 | 0.6943 | 0.7949 | 458 |
| TREC-COVID | DeltaLoss | Qwen2.5-7B | 0.9588 [0.9411, 0.9739] | 0.9663 [0.9523, 0.9784] | 0.8384 | 0.8996 | 0.9083 | 458 |
| TREC-COVID | GainRatio | Qwen2.5-7B | 0.9779 [0.9654, 0.9883] | 0.9780 [0.9625, 0.9893] | 0.9127 | 0.9432 | 0.9316 | 458 |

## Interpretation Guide

- ROC-AUC measures threshold-free ranking quality.
- PR-AUC is more sensitive to positive-class retrieval quality.
- TPR@FPR=5% reports detection recall when false positives are constrained to at most 5%.
- Best F1 reports the best threshold-dependent binary detection performance.
- Bootstrap 95% CI estimates statistical uncertainty by stratified resampling with replacement.

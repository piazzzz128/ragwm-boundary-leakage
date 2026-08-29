# Main Results: Boundary-level Watermark Detection

| Dataset | Construction | Method | Model | N | Inject / Clean | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | strict v1 | Contriever + Cosine | Contriever | 446 | 223 / 223 | 0.8346 | 0.8034 | 0.2735 | 0.4395 | 0.7900 |
| NFCorpus | strict v1 | GainRatio | Qwen2.5-7B | 446 | 223 / 223 | 0.9777 | 0.9758 | 0.9013 | 0.9552 | 0.9381 |
| TREC-COVID | strict v3 | Contriever + Cosine | Contriever | 458 | 229 / 229 | 0.6891 | 0.6641 | 0.1397 | 0.2271 | 0.7099 |
| TREC-COVID | strict v3 | GainRatio | Qwen2.5-7B | 458 | 229 / 229 | 0.9779 | 0.9780 | 0.9127 | 0.9432 | 0.9316 |
| Natural Questions | sampled strict v2 | Contriever + Cosine | Contriever | 442 | 221 / 221 | 0.8953 | 0.8646 | 0.5068 | 0.6471 | 0.8373 |
| Natural Questions | sampled strict v2 | GainRatio | Qwen2.5-7B | 442 | 221 / 221 | 0.9467 | 0.9465 | 0.7511 | 0.9005 | 0.9005 |

## Notes

- NFCorpus and TREC-COVID are biomedical / medical-related corpora.
- Natural Questions is an open-domain Wikipedia QA dataset. Here we use a 20k-document sampled strict subset, not the full NQ corpus.
- GainRatio is the main method. DeltaLoss is retained as an ablation and intermediate definition.
- Across all three datasets, GainRatio improves ROC-AUC, PR-AUC, low-FPR detection, and Best F1 over Contriever + Cosine.
- The NQ result addresses cross-domain generalization beyond biomedical corpora.

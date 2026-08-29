| Dataset | Method | Model | N | Inject/Clean | ROC-AUC | PR-AUC | TPR@FPR=5% | Best F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | Contriever + Cosine | Contriever | 1000 | 500 / 500 | 0.9153 | 0.8927 | 0.6040 | 0.8564 |
| NFCorpus | GainRatio | Qwen2.5-7B | 1000 | 500 / 500 | 0.9433 | 0.8818 | 0.6960 | 0.9041 |
| TREC-COVID | Contriever + Cosine | Contriever | 458 | 229 / 229 | 0.6891 | 0.6641 | 0.1397 | 0.7099 |
| TREC-COVID | GainRatio | Qwen2.5-7B | 458 | 229 / 229 | 0.9779 | 0.9780 | 0.9127 | 0.9316 |

# Context Length Ablation: Qwen2.5-7B + GainRatio

| Dataset | Construction | Context Tokens | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | strict v1 | 16 | 446 | 0.8800 | 0.8522 | 0.4664 | 0.5830 | 0.8276 |
| NFCorpus | strict v1 | 32 | 446 | 0.9329 | 0.9201 | 0.6233 | 0.8341 | 0.8758 |
| NFCorpus | strict v1 | 64 | 446 | 0.9683 | 0.9641 | 0.8296 | 0.9327 | 0.9211 |
| NFCorpus | strict v1 | 128 | 446 | 0.9787 | 0.9766 | 0.9103 | 0.9596 | 0.9375 |
| NFCorpus | strict v1 | 256 | 446 | 0.9796 | 0.9772 | 0.9103 | 0.9641 | 0.9427 |
| NFCorpus | strict v1 | full | 446 | 0.9776 | 0.9757 | 0.9013 | 0.9552 | 0.9381 |
| TREC-COVID | strict v3 | 16 | 458 | 0.8557 | 0.8425 | 0.4323 | 0.6157 | 0.8000 |
| TREC-COVID | strict v3 | 32 | 458 | 0.9186 | 0.9096 | 0.6157 | 0.7729 | 0.8510 |
| TREC-COVID | strict v3 | 64 | 458 | 0.9473 | 0.9336 | 0.7380 | 0.8384 | 0.8814 |
| TREC-COVID | strict v3 | 128 | 458 | 0.9574 | 0.9433 | 0.7598 | 0.8690 | 0.8931 |
| TREC-COVID | strict v3 | 256 | 458 | 0.9558 | 0.9407 | 0.7511 | 0.8690 | 0.8928 |
| TREC-COVID | strict v3 | full | 458 | 0.9558 | 0.9407 | 0.7511 | 0.8690 | 0.8928 |
| Natural Questions | sampled strict v2 | 16 | 442 | 0.8401 | 0.8254 | 0.4344 | 0.5747 | 0.7838 |
| Natural Questions | sampled strict v2 | 32 | 442 | 0.9069 | 0.9031 | 0.6335 | 0.7647 | 0.8465 |
| Natural Questions | sampled strict v2 | 64 | 442 | 0.9358 | 0.9368 | 0.7330 | 0.8507 | 0.8814 |
| Natural Questions | sampled strict v2 | 128 | 442 | 0.9453 | 0.9455 | 0.7557 | 0.8914 | 0.8975 |
| Natural Questions | sampled strict v2 | 256 | 442 | 0.9458 | 0.9459 | 0.7511 | 0.8959 | 0.8984 |
| Natural Questions | sampled strict v2 | full | 442 | 0.9458 | 0.9459 | 0.7511 | 0.8959 | 0.8984 |

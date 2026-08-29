# Context Length Ablation: Qwen2.5-7B + GainRatio Aligned

| Dataset | Construction | Context Tokens | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | strict v1 | 16 | 446 | 0.9270 | 0.9296 | 0.7444 | 0.8251 | 0.8598 |
| NFCorpus | strict v1 | 32 | 446 | 0.9645 | 0.9654 | 0.8072 | 0.9103 | 0.9062 |
| NFCorpus | strict v1 | 64 | 446 | 0.9852 | 0.9842 | 0.9372 | 0.9686 | 0.9478 |
| NFCorpus | strict v1 | 128 | 446 | 0.9922 | 0.9923 | 0.9596 | 0.9910 | 0.9605 |
| NFCorpus | strict v1 | 256 | 446 | 0.9905 | 0.9910 | 0.9462 | 0.9686 | 0.9524 |
| NFCorpus | strict v1 | full | 446 | 0.9904 | 0.9909 | 0.9462 | 0.9686 | 0.9513 |
| TREC-COVID | strict v3 | 16 | 458 | 0.9173 | 0.9216 | 0.6987 | 0.7642 | 0.8471 |
| TREC-COVID | strict v3 | 32 | 458 | 0.9555 | 0.9562 | 0.8035 | 0.8603 | 0.8874 |
| TREC-COVID | strict v3 | 64 | 458 | 0.9722 | 0.9726 | 0.8734 | 0.9301 | 0.9227 |
| TREC-COVID | strict v3 | 128 | 458 | 0.9780 | 0.9784 | 0.8996 | 0.9432 | 0.9339 |
| TREC-COVID | strict v3 | 256 | 458 | 0.9779 | 0.9781 | 0.8996 | 0.9432 | 0.9318 |
| TREC-COVID | strict v3 | full | 458 | 0.9779 | 0.9781 | 0.8996 | 0.9432 | 0.9318 |
| Natural Questions | sampled strict v2 | 16 | 442 | 0.8818 | 0.8572 | 0.5294 | 0.6335 | 0.8215 |
| Natural Questions | sampled strict v2 | 32 | 442 | 0.9336 | 0.9243 | 0.7195 | 0.8507 | 0.8796 |
| Natural Questions | sampled strict v2 | 64 | 442 | 0.9581 | 0.9592 | 0.8326 | 0.9095 | 0.9108 |
| Natural Questions | sampled strict v2 | 128 | 442 | 0.9645 | 0.9661 | 0.8597 | 0.9186 | 0.9161 |
| Natural Questions | sampled strict v2 | 256 | 442 | 0.9652 | 0.9667 | 0.8597 | 0.9276 | 0.9182 |
| Natural Questions | sampled strict v2 | full | 442 | 0.9652 | 0.9667 | 0.8597 | 0.9276 | 0.9182 |

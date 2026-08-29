# Model Scale Ablation: Aligned Loss-based Detection

| Dataset | Construction | Method | Model | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | strict v1 | DeltaLoss | distilgpt2 | 446 | 0.7847 | 0.8254 | 0.5202 | 0.5336 | 0.7160 |
| NFCorpus | strict v1 | GainRatio | distilgpt2 | 446 | 0.8072 | 0.8378 | 0.5112 | 0.5471 | 0.7329 |
| NFCorpus | strict v1 | DeltaLoss | gpt2-medium | 446 | 0.8952 | 0.9190 | 0.7085 | 0.7534 | 0.8160 |
| NFCorpus | strict v1 | GainRatio | gpt2-medium | 446 | 0.9236 | 0.9367 | 0.7175 | 0.7848 | 0.8488 |
| NFCorpus | strict v1 | DeltaLoss | Qwen2.5-7B | 446 | 0.9668 | 0.9730 | 0.8789 | 0.8924 | 0.9138 |
| NFCorpus | strict v1 | GainRatio | Qwen2.5-7B | 446 | 0.9904 | 0.9909 | 0.9462 | 0.9686 | 0.9513 |
| Natural Questions | sampled strict v2 | DeltaLoss | distilgpt2 | 442 | 0.8945 | 0.9035 | 0.7421 | 0.8009 | 0.8496 |
| Natural Questions | sampled strict v2 | GainRatio | distilgpt2 | 442 | 0.9098 | 0.9069 | 0.7557 | 0.8371 | 0.8680 |
| Natural Questions | sampled strict v2 | DeltaLoss | gpt2-medium | 442 | 0.9181 | 0.9352 | 0.7828 | 0.8281 | 0.8657 |
| Natural Questions | sampled strict v2 | GainRatio | gpt2-medium | 442 | 0.9424 | 0.9496 | 0.8190 | 0.8824 | 0.8944 |
| Natural Questions | sampled strict v2 | DeltaLoss | Qwen2.5-7B | 442 | 0.9436 | 0.9551 | 0.8552 | 0.8824 | 0.9000 |
| Natural Questions | sampled strict v2 | GainRatio | Qwen2.5-7B | 442 | 0.9652 | 0.9667 | 0.8597 | 0.9276 | 0.9182 |
| TREC-COVID | strict v3 | DeltaLoss | distilgpt2 | 458 | 0.7798 | 0.8095 | 0.4585 | 0.4803 | 0.7169 |
| TREC-COVID | strict v3 | GainRatio | distilgpt2 | 458 | 0.8019 | 0.8234 | 0.4585 | 0.4934 | 0.7315 |
| TREC-COVID | strict v3 | DeltaLoss | gpt2-medium | 458 | 0.8445 | 0.8775 | 0.5895 | 0.6245 | 0.7642 |
| TREC-COVID | strict v3 | GainRatio | gpt2-medium | 458 | 0.8775 | 0.8995 | 0.5983 | 0.7031 | 0.7957 |
| TREC-COVID | strict v3 | DeltaLoss | Qwen2.5-7B | 458 | 0.9588 | 0.9660 | 0.8297 | 0.8996 | 0.9071 |
| TREC-COVID | strict v3 | GainRatio | Qwen2.5-7B | 458 | 0.9779 | 0.9781 | 0.8996 | 0.9432 | 0.9318 |

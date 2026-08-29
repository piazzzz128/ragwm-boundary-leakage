# NFCorpus Strict v1 Boundary Detection Results

| Dataset | Method | Model | N | Inject/Clean | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus strict v1 | Contriever + Cosine | Contriever | 446 | 223 / 223 | 0.8346 | 0.8034 | 0.2735 | 0.4395 | 0.7900 |
| NFCorpus strict v1 | GainRatio | Qwen2.5-7B | 446 | 223 / 223 | 0.9777 | 0.9758 | 0.9013 | 0.9552 | 0.9381 |

## Ablation

| Dataset | Method | Model | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---|---:|---:|---:|---:|---:|
| NFCorpus strict v1 | DeltaLoss | Qwen2.5-7B | 0.9604 | 0.9688 | 0.8744 | 0.8924 | 0.9112 |
| NFCorpus strict v1 | GainRatio | Qwen2.5-7B | 0.9777 | 0.9758 | 0.9013 | 0.9552 | 0.9381 |

## Group-level Supplement

| Method | Aggregation | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |
|---|---|---:|---:|---:|---:|---:|
| GainRatio | Group-Mean | 0.9935 | 0.9685 | 0.9800 | 1.0000 | 0.9423 |
| GainRatio | Group-Median | 0.9903 | 0.9576 | 0.9800 | 0.9800 | 0.9231 |
| DeltaLoss | Group-Mean | 0.9923 | 0.9749 | 0.9800 | 0.9800 | 0.9412 |
| DeltaLoss | Group-Median | 0.9849 | 0.9627 | 0.9600 | 0.9600 | 0.9200 |

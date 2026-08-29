import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f)

if isinstance(obj, list):
    records = obj
else:
    records = None
    for k in ["records", "scores", "data", "results", "items"]:
        if k in obj and isinstance(obj[k], list):
            records = obj[k]
            break

if records is None:
    raise RuntimeError("No records found")

y = np.array([int(r["label"]) for r in records])

candidate_keys = sorted(set().union(*[set(r.keys()) for r in records]))
print("FILE:", p)
print("N:", len(records))

for k in candidate_keys:
    if k == "label":
        continue
    vals = []
    ok = True
    for r in records:
        if k not in r or not isinstance(r[k], (int, float)):
            ok = False
            break
        vals.append(float(r[k]))
    if not ok:
        continue

    s = np.array(vals)
    try:
        auc_raw = roc_auc_score(y, s)
        pr_raw = average_precision_score(y, s)
        auc_neg = roc_auc_score(y, -s)
        pr_neg = average_precision_score(y, -s)
        print(f"{k:30s} raw ROC={auc_raw:.4f} PR={pr_raw:.4f} | neg ROC={auc_neg:.4f} PR={pr_neg:.4f}")
    except Exception:
        pass

import argparse, json, math, os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for key in ["records", "scores", "data", "results", "items"]:
        if key in obj and isinstance(obj[key], list):
            return obj[key]
    raise ValueError(f"Cannot find record list in {path}")

def get_label(r):
    for k in ["label", "y", "target", "is_inject"]:
        if k in r:
            return int(r[k])
    raise KeyError(f"No label field in record keys: {list(r.keys())}")

def get_score(r, score_field=None):
    if score_field and score_field in r:
        return float(r[score_field])

    # Prefer explicit anomaly scores if present.
    for k in ["anomaly_score", "score"]:
        if k in r:
            return float(r[k])

    # Formal convention: anomaly score = -GainRatio or -DeltaLoss or -cosine.
    if "gain_ratio" in r:
        return -float(r["gain_ratio"])
    if "record_gain_ratio" in r:
        return -float(r["record_gain_ratio"])
    if "delta_loss" in r:
        return -float(r["delta_loss"])
    if "record_delta_loss" in r:
        return -float(r["record_delta_loss"])
    if "cosine" in r:
        return -float(r["cosine"])
    if "similarity" in r:
        return -float(r["similarity"])

    raise KeyError(f"No usable score field in record keys: {list(r.keys())}")

def tpr_at_fpr(y, s, target_fpr):
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))

def best_f1(y, s):
    precision, recall, _ = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.max(f1))

def metrics(y, s):
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "tpr_at_fpr_5": tpr_at_fpr(y, s, 0.05),
        "tpr_at_fpr_10": tpr_at_fpr(y, s, 0.10),
        "best_f1": best_f1(y, s),
    }

def percentile_ci(vals):
    vals = np.asarray(vals, dtype=float)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--score-field", default=None)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260609)
    args = ap.parse_args()

    records = load_records(args.input)
    y = np.array([get_label(r) for r in records], dtype=int)
    s = np.array([get_score(r, args.score_field) for r in records], dtype=float)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Labels must be binary 0/1, got {np.unique(y)}")

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(args.seed)

    point = metrics(y, s)
    boots = {k: [] for k in point}

    for _ in range(args.n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        yy, ss = y[idx], s[idx]
        try:
            m = metrics(yy, ss)
            for k, v in m.items():
                if not math.isnan(v):
                    boots[k].append(v)
        except Exception:
            continue

    out = {
        "input": args.input,
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "score_orientation": "higher score = more likely injected/anomalous",
        "bootstrap": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "sampling": "stratified bootstrap over positive and negative classes"
        },
        "metrics": {
            k: {"point": point[k], "ci95": percentile_ci(boots[k])}
            for k in point
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out["metrics"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

import argparse, json, os, math
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

def norm(v):
    return str(v).strip().lower()

def keep_record(r, args):
    if args.dataset is not None and norm(r.get("dataset")) != norm(args.dataset):
        return False
    if args.construction is not None and norm(r.get("construction")) != norm(args.construction):
        return False
    if args.model is not None and norm(r.get("model")) != norm(args.model):
        return False
    if args.context_budget is not None and norm(r.get("context_budget")) != norm(args.context_budget):
        return False
    if args.variant is not None and norm(r.get("variant")) != norm(args.variant):
        return False
    if args.attack is not None and norm(r.get("attack")) != norm(args.attack):
        return False
    return True

def get_label(r):
    for k in ["label", "y", "target", "is_inject"]:
        if k in r:
            return int(r[k])
    raise KeyError(f"No label field in record keys: {list(r.keys())}")

def get_score(r, score_field):
    if score_field is None:
        for k in ["anomaly_score", "gain_anomaly_score", "score", "embedding_anomaly_score"]:
            if k in r:
                return float(r[k])
        if "gain_ratio" in r:
            return -float(r["gain_ratio"])
        if "delta_loss" in r:
            return -float(r["delta_loss"])
        if "contriever_cosine" in r:
            return -float(r["contriever_cosine"])
        raise KeyError(f"No usable score field in record keys: {list(r.keys())}")
    if score_field not in r:
        raise KeyError(f"score_field={score_field} not found. Keys={list(r.keys())}")
    return float(r[score_field])

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

def ci(vals):
    vals = np.asarray(vals, dtype=float)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--score-field", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--construction", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--context-budget", default=None)
    ap.add_argument("--attack", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260609)
    args = ap.parse_args()

    records_all = load_records(args.input)
    records = [r for r in records_all if keep_record(r, args)]

    if len(records) == 0:
        raise ValueError("No records after filtering. Check dataset/model/context_budget names.")

    y = np.array([get_label(r) for r in records], dtype=int)
    s = np.array([get_score(r, args.score_field) for r in records], dtype=float)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Labels must contain both 0 and 1. Got {np.unique(y)}")

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
        try:
            m = metrics(y[idx], s[idx])
            for k, v in m.items():
                if not math.isnan(v):
                    boots[k].append(v)
        except Exception:
            continue

    out = {
        "input": args.input,
        "filters": {
            "dataset": args.dataset,
            "construction": args.construction,
            "model": args.model,
            "context_budget": args.context_budget,
            "attack": args.attack,
            "variant": args.variant,
            "score_field": args.score_field,
        },
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
            k: {"point": point[k], "ci95": ci(boots[k])}
            for k in point
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

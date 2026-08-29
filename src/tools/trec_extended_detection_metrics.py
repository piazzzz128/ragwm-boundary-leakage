import os
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


BASE = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff"

OUT_JSON = f"{BASE}/trec_detection_metrics_extended.json"
OUT_MD = f"{BASE}/trec_detection_metrics_extended.md"
OUT_CSV = f"{BASE}/trec_detection_metrics_extended.csv"

N_BOOT = 1000
SEED = 12


METHODS = [
    {
        "dataset": "TREC-COVID",
        "method": "Contriever + Cosine",
        "model": "Contriever",
        "file": f"{BASE}/trec_contriever_cosine_scores.json",
        "score_key": "embedding_anomaly_score",
    },
    {
        "dataset": "TREC-COVID",
        "method": "DeltaLoss",
        "model": "distilgpt2",
        "file": f"{BASE}/trec_delta_loss_scores_distilgpt2.json",
        "score_key": "delta_anomaly_score",
    },
    {
        "dataset": "TREC-COVID",
        "method": "GainRatio",
        "model": "distilgpt2",
        "file": f"{BASE}/trec_delta_loss_scores_distilgpt2.json",
        "score_key": "gain_anomaly_score",
    },
    {
        "dataset": "TREC-COVID",
        "method": "DeltaLoss",
        "model": "gpt2-medium",
        "file": f"{BASE}/trec_delta_loss_scores_gpt2_medium.json",
        "score_key": "delta_anomaly_score",
    },
    {
        "dataset": "TREC-COVID",
        "method": "GainRatio",
        "model": "gpt2-medium",
        "file": f"{BASE}/trec_delta_loss_scores_gpt2_medium.json",
        "score_key": "gain_anomaly_score",
    },
    {
        "dataset": "TREC-COVID",
        "method": "DeltaLoss",
        "model": "Qwen2.5-7B",
        "file": f"{BASE}/trec_delta_loss_scores_qwen2_5_7b.json",
        "score_key": "delta_anomaly_score",
    },
    {
        "dataset": "TREC-COVID",
        "method": "GainRatio",
        "model": "Qwen2.5-7B",
        "file": f"{BASE}/trec_delta_loss_scores_qwen2_5_7b.json",
        "score_key": "gain_anomaly_score",
    },
]


def load_scores(path, score_key):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = json.load(open(path, "r", encoding="utf-8"))

    labels = []
    scores = []
    ids = []

    for i, row in enumerate(data):
        if "label" not in row:
            raise KeyError(f"missing label at index={i} in {path}")
        if score_key not in row:
            raise KeyError(f"missing score_key={score_key} at index={i} in {path}")

        labels.append(int(row["label"]))
        scores.append(float(row[score_key]))
        ids.append(row.get("id", str(i)))

    labels = np.array(labels, dtype=np.int32)
    scores = np.array(scores, dtype=np.float64)

    if len(labels) == 0:
        raise RuntimeError(f"empty score file: {path}")
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise RuntimeError(f"labels must contain both classes: {path}")

    return labels, scores, ids


def tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(labels, scores)

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return {
            "tpr": 0.0,
            "fpr": 0.0,
            "threshold": float("inf"),
        }

    # 在 FPR 约束下取最高 TPR；如并列，取实际 FPR 更低的
    best = valid[np.argmax(tpr[valid])]

    return {
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
        "threshold": float(thresholds[best]),
    }


def best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # precision/recall 长度比 thresholds 多 1，最后一个点没有对应阈值
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)

    best_idx = int(np.nanargmax(f1))

    if best_idx < len(thresholds):
        threshold = float(thresholds[best_idx])
    else:
        threshold = float(np.min(scores) - 1e-12)

    preds = (scores >= threshold).astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    fpr_value = fp / max(fp + tn, 1)
    tpr_value = tp / max(tp + fn, 1)

    return {
        "best_f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fpr": float(fpr_value),
        "tpr": float(tpr_value),
    }


def percentile_ci(values, alpha=0.05):
    arr = np.array(values, dtype=np.float64)
    return {
        "low": float(np.percentile(arr, 100 * alpha / 2)),
        "high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def bootstrap_metrics(labels, scores, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    boot_roc = []
    boot_pr = []
    boot_tpr5 = []
    boot_tpr10 = []
    boot_f1 = []

    for _ in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        rng.shuffle(sample_idx)

        y = labels[sample_idx]
        s = scores[sample_idx]

        boot_roc.append(roc_auc_score(y, s))
        boot_pr.append(average_precision_score(y, s))
        boot_tpr5.append(tpr_at_fpr(y, s, 0.05)["tpr"])
        boot_tpr10.append(tpr_at_fpr(y, s, 0.10)["tpr"])
        boot_f1.append(best_f1(y, s)["best_f1"])

    return {
        "roc_auc_ci95": percentile_ci(boot_roc),
        "pr_auc_ci95": percentile_ci(boot_pr),
        "tpr_at_fpr_5_ci95": percentile_ci(boot_tpr5),
        "tpr_at_fpr_10_ci95": percentile_ci(boot_tpr10),
        "best_f1_ci95": percentile_ci(boot_f1),
        "n_bootstrap": n_boot,
        "seed": seed,
    }


def evaluate_one(spec):
    labels, scores, ids = load_scores(spec["file"], spec["score_key"])

    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))

    tpr5 = tpr_at_fpr(labels, scores, 0.05)
    tpr10 = tpr_at_fpr(labels, scores, 0.10)
    f1 = best_f1(labels, scores)

    boot = bootstrap_metrics(labels, scores)

    result = {
        "dataset": spec["dataset"],
        "method": spec["method"],
        "model": spec["model"],
        "score_file": spec["file"],
        "score_key": spec["score_key"],
        "n": int(len(labels)),
        "positive_inject": int(labels.sum()),
        "negative_clean": int((1 - labels).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tpr_at_fpr_5": tpr5,
        "tpr_at_fpr_10": tpr10,
        "best_f1": f1,
        "bootstrap": boot,
        "score_mean_inject": float(scores[labels == 1].mean()),
        "score_mean_clean": float(scores[labels == 0].mean()),
    }

    return result


def fmt_ci(main, ci):
    return f"{main:.4f} [{ci['low']:.4f}, {ci['high']:.4f}]"


def write_markdown(results):
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# TREC-COVID Extended Detection Metrics\n\n")
        f.write("All metrics are record-level. Scores are oriented so that larger values indicate a higher probability of inject boundary.\n\n")

        f.write("## Main Table\n\n")
        f.write("| Dataset | Method | Model | ROC-AUC 95% CI | PR-AUC 95% CI | TPR@FPR=5% | TPR@FPR=10% | Best F1 | N |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")

        for r in results:
            roc_ci = r["bootstrap"]["roc_auc_ci95"]
            pr_ci = r["bootstrap"]["pr_auc_ci95"]

            f.write(
                f"| {r['dataset']} | {r['method']} | {r['model']} | "
                f"{fmt_ci(r['roc_auc'], roc_ci)} | "
                f"{fmt_ci(r['pr_auc'], pr_ci)} | "
                f"{r['tpr_at_fpr_5']['tpr']:.4f} | "
                f"{r['tpr_at_fpr_10']['tpr']:.4f} | "
                f"{r['best_f1']['best_f1']:.4f} | "
                f"{r['n']} |\n"
            )

        f.write("\n## Interpretation Guide\n\n")
        f.write("- ROC-AUC measures threshold-free ranking quality.\n")
        f.write("- PR-AUC is more sensitive to positive-class retrieval quality.\n")
        f.write("- TPR@FPR=5% reports detection recall when false positives are constrained to at most 5%.\n")
        f.write("- Best F1 reports the best threshold-dependent binary detection performance.\n")
        f.write("- Bootstrap 95% CI estimates statistical uncertainty by stratified resampling with replacement.\n")


def write_csv(results):
    fields = [
        "dataset",
        "method",
        "model",
        "n",
        "positive_inject",
        "negative_clean",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "tpr_at_fpr_5",
        "actual_fpr_at_5",
        "threshold_at_fpr_5",
        "tpr_at_fpr_10",
        "actual_fpr_at_10",
        "threshold_at_fpr_10",
        "best_f1",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1_threshold",
        "best_f1_fpr",
        "score_mean_inject",
        "score_mean_clean",
        "score_file",
        "score_key",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "dataset": r["dataset"],
                "method": r["method"],
                "model": r["model"],
                "n": r["n"],
                "positive_inject": r["positive_inject"],
                "negative_clean": r["negative_clean"],
                "roc_auc": r["roc_auc"],
                "roc_auc_ci_low": r["bootstrap"]["roc_auc_ci95"]["low"],
                "roc_auc_ci_high": r["bootstrap"]["roc_auc_ci95"]["high"],
                "pr_auc": r["pr_auc"],
                "pr_auc_ci_low": r["bootstrap"]["pr_auc_ci95"]["low"],
                "pr_auc_ci_high": r["bootstrap"]["pr_auc_ci95"]["high"],
                "tpr_at_fpr_5": r["tpr_at_fpr_5"]["tpr"],
                "actual_fpr_at_5": r["tpr_at_fpr_5"]["fpr"],
                "threshold_at_fpr_5": r["tpr_at_fpr_5"]["threshold"],
                "tpr_at_fpr_10": r["tpr_at_fpr_10"]["tpr"],
                "actual_fpr_at_10": r["tpr_at_fpr_10"]["fpr"],
                "threshold_at_fpr_10": r["tpr_at_fpr_10"]["threshold"],
                "best_f1": r["best_f1"]["best_f1"],
                "best_f1_precision": r["best_f1"]["precision"],
                "best_f1_recall": r["best_f1"]["recall"],
                "best_f1_threshold": r["best_f1"]["threshold"],
                "best_f1_fpr": r["best_f1"]["fpr"],
                "score_mean_inject": r["score_mean_inject"],
                "score_mean_clean": r["score_mean_clean"],
                "score_file": r["score_file"],
                "score_key": r["score_key"],
            })


def main():
    print("==== Extended detection metrics ====")
    print("Output JSON:", OUT_JSON)
    print("Output MD:", OUT_MD)
    print("Output CSV:", OUT_CSV)
    print("Bootstrap:", N_BOOT)

    results = []

    for spec in METHODS:
        print("\n----")
        print(spec["method"], spec["model"])
        print("file:", spec["file"])
        print("score_key:", spec["score_key"])

        r = evaluate_one(spec)
        results.append(r)

        print("ROC-AUC:", r["roc_auc"])
        print("PR-AUC:", r["pr_auc"])
        print("TPR@FPR=5%:", r["tpr_at_fpr_5"]["tpr"])
        print("TPR@FPR=10%:", r["tpr_at_fpr_10"]["tpr"])
        print("Best F1:", r["best_f1"]["best_f1"])

    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    write_markdown(results)
    write_csv(results)

    print("\n==== DONE ====")
    print("saved:", OUT_JSON)
    print("saved:", OUT_MD)
    print("saved:", OUT_CSV)
    print("\nMarkdown preview:")
    print(open(OUT_MD, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path("/root/autodl-tmp/ragwm_storage")
OUT = ROOT / "output/formal_likelihood_matrix_v1_20260902"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("NFCorpus", "Qwen2.5-7B", ROOT / "output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json", "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218", 446, (223, 223)),
    ("NFCorpus", "Qwen2.5-14B", ROOT / "output/cross_detector_lm_robustness_v1/qwen2_5_14b_base/scores/qwen14b_nfcorpus_aligned_scores.json", "90bb39f9afd874c2f2b3d819cde1b6f56a15919a392946574c60ef9278b90821", 446, (223, 223)),
    ("NFCorpus", "DeepSeek-LLM-7B-base", ROOT / "output/cross_detector_lm_robustness_v1/deepseek_llm_7b_base/scores/deepseek_nfcorpus_aligned_scores.json", "fd29d143ac4c08f23d36b8fac5d4bad7e6d85172f68bb7eb4c1d19275552cc86", 446, (223, 223)),
    ("TREC-COVID", "Qwen2.5-7B", ROOT / "output/sanitization_e2e_trec_covid_aligned_v1/scores/aligned_qwen_v1/trec_aligned_qwen_v1_scores.json", "cc6a22420492cf63ddcaa1be90d2528db907912f905ce2f4425e26a894aed4e2", 458, (229, 229)),
    ("TREC-COVID", "Qwen2.5-14B", ROOT / "output/cross_detector_lm_robustness_v1/qwen2_5_14b_base/scores/qwen14b_trec_aligned_scores.json", "f25ee58f287f585490fe258306e8c73000d327207d3db24a5bf65b4341c18691", 458, (229, 229)),
    ("TREC-COVID", "DeepSeek-LLM-7B-base", ROOT / "output/cross_detector_lm_robustness_v1/deepseek_llm_7b_base/scores/deepseek_trec_aligned_scores.json", "16b89d599fdcd1a792f783174f83560537d3227693f6843218b9c16cdf245118", 458, (229, 229)),
]

METHODS = ("Target-only NLL", "Conditional NLL", "DeltaLoss", "GainRatio")
SEED = 20260609
REPLICATES = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cluster_id(row: dict) -> str:
    value = row.get("source_doc_id")
    if value is None:
        value = row.get("document_id")
    if value is None:
        value = row.get("group_id")
    if value is None:
        value = row.get("id") or row.get("record_id")
    if value is None:
        raise RuntimeError("No source-document cluster identifier")
    return str(value)


def content_signature(row: dict) -> str:
    payload = [str(row.get("context", row.get("context_text", ""))), str(row.get("target", row.get("target_text", ""))), int(row["label"])]
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def cluster_signature(row: dict) -> str:
    payload = [content_signature(row), cluster_id(row)]
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def dataset_signature(rows: list[dict], include_cluster: bool) -> str:
    fn = cluster_signature if include_cluster else content_signature
    values = sorted(fn(row) for row in rows)
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def summarize(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    result = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "best_f1": float(f1.max()),
    }
    for ceiling in (0.01, 0.05, 0.10):
        result[f"tpr_at_fpr_{ceiling:g}"] = float(tpr[fpr <= ceiling + 1e-15].max())
    return result


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


canonical = {}
audits = []
settings = []
raw_path = OUT / "formal_likelihood_raw_scores.jsonl"
raw_stream = raw_path.open("w", encoding="utf-8")

try:
    for dataset, model, path, expected_sha, expected_n, expected_counts in CONFIGS:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{dataset}/{model}: SHA mismatch: {actual_sha}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != expected_n:
            raise RuntimeError(f"{dataset}/{model}: expected {expected_n} rows")
        labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
        counts = tuple(np.bincount(labels, minlength=2))
        if counts != expected_counts:
            raise RuntimeError(f"{dataset}/{model}: label counts {counts}")

        content_sig = dataset_signature(rows, False)
        clustered_sig = dataset_signature(rows, True)
        if dataset not in canonical:
            canonical[dataset] = (content_sig, clustered_sig)
        elif canonical[dataset] != (content_sig, clustered_sig):
            raise RuntimeError(f"{dataset}/{model}: record or cluster identity mismatch")

        policies = sorted({stable_json(row.get("alignment_policy")) for row in rows})
        if len(policies) != 1:
            raise RuntimeError(f"{dataset}/{model}: multiple alignment policies")
        no_bos_values = sorted({row.get("no_bos") for row in rows if "no_bos" in row}, key=str)
        no_separator_values = sorted({row.get("no_separator") for row in rows if "no_separator" in row}, key=str)
        if no_bos_values and no_bos_values != [True]:
            raise RuntimeError(f"{dataset}/{model}: no_bos={no_bos_values}")
        if no_separator_values and no_separator_values != [True]:
            raise RuntimeError(f"{dataset}/{model}: no_separator={no_separator_values}")

        lu = np.asarray([float(row["loss_target_aligned"]) for row in rows], dtype=float)
        lc = np.asarray([float(row["loss_target_given_context_aligned"]) for row in rows], dtype=float)
        if not np.isfinite(lu).all() or not np.isfinite(lc).all() or np.any(lu <= 0) or np.any(lc <= 0):
            raise RuntimeError(f"{dataset}/{model}: invalid NLL values")
        official_delta = lc - lu
        official_gainratio = official_delta / lu
        stored_delta = np.asarray([float(row["delta_loss_aligned"]) for row in rows])
        stored_score = np.asarray([float(row["score_aligned"]) for row in rows])
        delta_error = float(np.max(np.abs(stored_delta - (lu - lc))))
        gainratio_error = float(np.max(np.abs(stored_score - official_gainratio)))
        if delta_error > 1e-7 or gainratio_error > 1e-7:
            raise RuntimeError(f"{dataset}/{model}: formula mismatch delta={delta_error}, gainratio={gainratio_error}")

        scores = {
            "Target-only NLL": lu,
            "Conditional NLL": lc,
            "DeltaLoss": official_delta,
            "GainRatio": official_gainratio,
        }
        point = {name: summarize(labels, values) for name, values in scores.items()}

        clusters = defaultdict(list)
        for i, row in enumerate(rows):
            clusters[cluster_id(row)].append(i)
        cluster_names = sorted(clusters)
        rng = np.random.default_rng(SEED)
        samples = {name: {"roc_auc": [], "pr_auc": []} for name in METHODS}
        differences = {name: {"roc_auc": [], "pr_auc": []} for name in METHODS if name != "GainRatio"}
        skipped = 0
        for _ in range(REPLICATES):
            chosen = rng.choice(cluster_names, size=len(cluster_names), replace=True)
            indices = np.asarray([i for c in chosen for i in clusters[str(c)]], dtype=int)
            y = labels[indices]
            if len(np.unique(y)) < 2:
                skipped += 1
                continue
            rep = {}
            for name, values in scores.items():
                roc = float(roc_auc_score(y, values[indices]))
                pr = float(average_precision_score(y, values[indices]))
                samples[name]["roc_auc"].append(roc)
                samples[name]["pr_auc"].append(pr)
                rep[name] = (roc, pr)
            for name in differences:
                differences[name]["roc_auc"].append(rep["GainRatio"][0] - rep[name][0])
                differences[name]["pr_auc"].append(rep["GainRatio"][1] - rep[name][1])

        ci = {name: {metric: interval(values) for metric, values in by_metric.items()} for name, by_metric in samples.items()}
        paired = {}
        for name, by_metric in differences.items():
            paired[name] = {}
            for metric, values in by_metric.items():
                bounds = interval(values)
                paired[name][metric] = {
                    "gainratio_minus_baseline": point["GainRatio"][metric] - point[name][metric],
                    "ci95": bounds,
                    "ci_excludes_zero": bool(bounds[0] > 0 or bounds[1] < 0),
                }

        audits.append({
            "dataset": dataset,
            "model": model,
            "input_path": str(path),
            "input_sha256": actual_sha,
            "record_count": len(rows),
            "class_counts": {"clean": counts[0], "injected": counts[1]},
            "content_signature_sha256": content_sig,
            "clustered_signature_sha256": clustered_sig,
            "alignment_policy": json.loads(policies[0]),
            "no_bos_values": no_bos_values,
            "no_separator_values": no_separator_values,
            "cluster_count": len(cluster_names),
            "max_formula_error": {"delta_loss": delta_error, "gainratio": gainratio_error},
        })
        settings.append({
            "dataset": dataset,
            "model": model,
            "record_count": len(rows),
            "score_direction": "larger score is more suspicious",
            "definitions": {
                "Target-only NLL": "L_u(T)",
                "Conditional NLL": "L_c(T|C)",
                "DeltaLoss": "L_c(T|C)-L_u(T)",
                "GainRatio": "(L_c(T|C)-L_u(T))/L_u(T)",
            },
            "point_metrics": point,
            "cluster_bootstrap_95ci": ci,
            "paired_gainratio_differences": paired,
            "bootstrap": {"unit": "source document", "cluster_count": len(cluster_names), "requested_replicates": REPLICATES, "valid_replicates": REPLICATES - skipped, "skipped_replicates": skipped, "seed": SEED},
        })
        for i, row in enumerate(rows):
            raw_stream.write(json.dumps({
                "dataset": dataset,
                "model": model,
                "source_input_sha256": actual_sha,
                "source_index": i,
                "record_content_sha256": content_signature(row),
                "source_document_cluster": cluster_id(row),
                "label": int(row["label"]),
                "target_only_nll": float(lu[i]),
                "conditional_nll": float(lc[i]),
                "delta_loss": float(official_delta[i]),
                "gainratio": float(official_gainratio[i]),
            }, ensure_ascii=False, sort_keys=True) + "\n")
finally:
    raw_stream.close()

(OUT / "protocol_identity_audit.json").write_text(json.dumps({"status": "PASS", "settings": audits}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "formal_likelihood_matrix_metrics.json").write_text(json.dumps({"experiment_role": "formal_same_position_likelihood_matrix_v1", "datasets": ["NFCorpus", "TREC-COVID"], "detector_lms": ["Qwen2.5-7B", "Qwen2.5-14B", "DeepSeek-LLM-7B-base"], "methods": list(METHODS), "settings": settings}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

csv_path = OUT / "formal_likelihood_matrix_long.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "model", "method", "n", "roc_auc", "pr_auc", "tpr_at_fpr_0.01", "tpr_at_fpr_0.05", "tpr_at_fpr_0.1", "best_f1"])
    for setting in settings:
        for method in METHODS:
            m = setting["point_metrics"][method]
            writer.writerow([setting["dataset"], setting["model"], method, setting["record_count"], m["roc_auc"], m["pr_auc"], m["tpr_at_fpr_0.01"], m["tpr_at_fpr_0.05"], m["tpr_at_fpr_0.1"], m["best_f1"]])

setting_map = {(s["dataset"], s["model"]): s for s in settings}
columns = [(d, m) for d in ("NFCorpus", "TREC-COVID") for m in ("Qwen2.5-7B", "Qwen2.5-14B", "DeepSeek-LLM-7B-base")]
lines = ["# Formal same-position likelihood matrix", "", "Cells report ROC-AUC / PR-AUC. Larger scores are more suspicious.", "", "| Method | " + " | ".join(f"{d} / {m}" for d, m in columns) + " |", "|---|" + "---|" * len(columns)]
for method in METHODS:
    cells = []
    for key in columns:
        m = setting_map[key]["point_metrics"][method]
        cells.append(f"{m['roc_auc']:.4f} / {m['pr_auc']:.4f}")
    lines.append("| " + method + " | " + " | ".join(cells) + " |")
(OUT / "formal_likelihood_matrix_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("PASS: formal same-position likelihood matrix completed")
print(OUT / "formal_likelihood_matrix_table.md")

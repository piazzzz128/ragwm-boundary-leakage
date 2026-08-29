#!/usr/bin/env python3
"""Compute simple TREC-COVID boundary baselines from the frozen score records.

The script performs no language-model inference. It reuses the target-only and
context-conditioned NLL values already stored for the same 458 records scored
by Qwen2.5-7B. All methods therefore use identical examples and labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


EXPECTED_INPUT_SHA256 = (
    "cc6a22420492cf63ddcaa1be90d2528db907912f905ce2f4425e26a894aed4e2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    result = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "best_f1": float(f1.max()),
    }
    for ceiling in (0.01, 0.05, 0.10):
        eligible = fpr <= ceiling + 1e-15
        result[f"tpr_at_fpr_{ceiling:g}"] = float(tpr[eligible].max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    actual_hash = sha256_file(args.input_json)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"Unexpected input SHA-256: {actual_hash}; expected {EXPECTED_INPUT_SHA256}"
        )

    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    if len(rows) != 458:
        raise RuntimeError(f"Expected 458 records, found {len(rows)}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    if tuple(np.bincount(labels, minlength=2)) != (229, 229):
        raise RuntimeError("Expected a balanced 229/229 boundary set")

    scores = {
        "Target-only NLL": np.asarray(
            [float(row["loss_target_aligned"]) for row in rows]
        ),
        "Context-conditioned NLL": np.asarray(
            [float(row["loss_target_given_context_aligned"]) for row in rows]
        ),
        "DeltaLoss": -np.asarray(
            [float(row["delta_loss_aligned"]) for row in rows]
        ),
        "GainRatio": np.asarray([float(row["score_aligned"]) for row in rows]),
    }
    point = {name: summarize(labels, values) for name, values in scores.items()}

    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        # document_id is the source-document cluster. group_id identifies a
        # watermark unit for injected rows and would merge unrelated documents.
        cluster = str(row.get("document_id") or row["id"])
        clusters[cluster].append(index)
    cluster_names = sorted(clusters)

    rng = np.random.default_rng(args.seed)
    samples: dict[str, dict[str, list[float]]] = {
        name: {"roc_auc": [], "pr_auc": []} for name in scores
    }
    differences: dict[str, dict[str, list[float]]] = {
        name: {"roc_auc": [], "pr_auc": []}
        for name in scores
        if name != "GainRatio"
    }
    skipped = 0
    for _ in range(args.replicates):
        sampled_clusters = rng.choice(
            cluster_names, size=len(cluster_names), replace=True
        )
        indices = np.asarray(
            [i for cluster in sampled_clusters for i in clusters[str(cluster)]],
            dtype=int,
        )
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            skipped += 1
            continue
        replicate_metrics = {}
        for name, values in scores.items():
            roc = float(roc_auc_score(sampled_labels, values[indices]))
            pr = float(average_precision_score(sampled_labels, values[indices]))
            samples[name]["roc_auc"].append(roc)
            samples[name]["pr_auc"].append(pr)
            replicate_metrics[name] = (roc, pr)
        gain_roc, gain_pr = replicate_metrics["GainRatio"]
        for name in differences:
            base_roc, base_pr = replicate_metrics[name]
            differences[name]["roc_auc"].append(gain_roc - base_roc)
            differences[name]["pr_auc"].append(gain_pr - base_pr)

    def interval(values: list[float]) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    auc_intervals = {
        name: {metric: interval(values) for metric, values in by_metric.items()}
        for name, by_metric in samples.items()
    }
    paired_differences = {}
    for name, by_metric in differences.items():
        paired_differences[name] = {}
        for metric, values in by_metric.items():
            observed = point["GainRatio"][metric] - point[name][metric]
            ci = interval(values)
            paired_differences[name][metric] = {
                "gainratio_minus_baseline": observed,
                "ci": ci,
                "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            }

    output = {
        "experiment_role": "trec_simple_boundary_baselines_v1",
        "input_sha256": actual_hash,
        "record_count": len(rows),
        "class_counts": {"clean": 229, "injected": 229},
        "score_direction": "larger score is more suspicious",
        "definitions": {
            "Target-only NLL": "L_u(T)",
            "Context-conditioned NLL": "L_c(T|C)",
            "DeltaLoss": "-(L_u(T)-L_c(T|C))",
            "GainRatio": "-(L_u(T)-L_c(T|C))/L_u(T)",
        },
        "point_metrics": point,
        "auc_cluster_bootstrap_95ci": auc_intervals,
        "paired_gainratio_differences": paired_differences,
        "bootstrap": {
            "unit": "source document",
            "requested_replicates": args.replicates,
            "valid_replicates": args.replicates - skipped,
            "skipped_replicates": skipped,
            "seed": args.seed,
            "cluster_count": len(cluster_names),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

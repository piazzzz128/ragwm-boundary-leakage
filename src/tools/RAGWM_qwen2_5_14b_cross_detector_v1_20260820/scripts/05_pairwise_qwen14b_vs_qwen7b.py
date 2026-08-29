#!/usr/bin/env python3
"""Paired source-cluster comparison of Qwen2.5-14B and Qwen2.5-7B."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cross_detector_common import load_json, sha256_file, write_json_atomic


QWEN7_NF_SHA256 = "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218"
QWEN7_TREC_SHA256 = "cc6a22420492cf63ddcaa1be90d2528db907912f905ce2f4425e26a894aed4e2"


def source_id(row: dict[str, Any]) -> str:
    for key in ("source_doc_id", "doc_id", "document_id", "chroma_id", "id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    raise RuntimeError("Row has no source-document identifier")


def text_value(row: dict[str, Any], key: str) -> str:
    aliases = {
        "context": ("context", "context_text"),
        "target": ("target", "target_text"),
    }
    for candidate in aliases[key]:
        if candidate in row:
            return str(row.get(candidate) or "")
    return ""


def pair_rows(
    qwen7_path: Path,
    qwen14_path: Path,
    dataset: str,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qwen7 = load_json(qwen7_path)
    qwen14 = load_json(qwen14_path)
    if not isinstance(qwen7, list) or not isinstance(qwen14, list):
        raise RuntimeError(f"{dataset}: score files must contain JSON lists")
    if len(qwen7) != expected_count or len(qwen14) != expected_count:
        raise RuntimeError(
            f"{dataset}: expected {expected_count} rows, got "
            f"Qwen7={len(qwen7)}, Qwen14={len(qwen14)}"
        )

    qwen14_by_index = {}
    for fallback_index, row in enumerate(qwen14):
        index = int(row.get("input_index", fallback_index))
        if index in qwen14_by_index:
            raise RuntimeError(f"{dataset}: duplicate Qwen14 input_index {index}")
        qwen14_by_index[index] = row
    if set(qwen14_by_index) != set(range(expected_count)):
        raise RuntimeError(f"{dataset}: incomplete Qwen14 input_index set")

    ordered_qwen14 = []
    label_counts = {0: 0, 1: 0}
    for index, row7 in enumerate(qwen7):
        row14 = qwen14_by_index[index]
        label7 = int(row7["label"])
        label14 = int(row14["label"])
        if label7 != label14:
            raise RuntimeError(f"{dataset}: label mismatch at row {index}")
        label_counts[label7] += 1
        if source_id(row7) != source_id(row14):
            raise RuntimeError(f"{dataset}: source-document mismatch at row {index}")
        for key in ("context", "target"):
            value7 = text_value(row7, key).strip()
            value14 = text_value(row14, key).strip()
            if value7 and value14 and value7 != value14:
                raise RuntimeError(f"{dataset}: {key} mismatch at row {index}")
        for model_name, row in (("Qwen7", row7), ("Qwen14", row14)):
            if "score_aligned" not in row:
                raise RuntimeError(
                    f"{dataset}: {model_name} row {index} lacks score_aligned"
                )
            if not np.isfinite(float(row["score_aligned"])):
                raise RuntimeError(
                    f"{dataset}: {model_name} row {index} has non-finite score"
                )
        ordered_qwen14.append(row14)

    expected_each = expected_count // 2
    if label_counts != {0: expected_each, 1: expected_each}:
        raise RuntimeError(f"{dataset}: unexpected label counts {label_counts}")
    return qwen7, ordered_qwen14


def metric_pair(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def paired_cluster_bootstrap(
    qwen7: list[dict[str, Any]],
    qwen14: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in qwen7], dtype=int)
    scores7 = np.asarray(
        [float(row["score_aligned"]) for row in qwen7], dtype=float
    )
    scores14 = np.asarray(
        [float(row["score_aligned"]) for row in qwen14], dtype=float
    )
    roc7, pr7 = metric_pair(labels, scores7)
    roc14, pr14 = metric_pair(labels, scores14)

    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(qwen14):
        clusters[source_id(row)].append(index)
    names = sorted(clusters)
    rng = np.random.default_rng(seed)
    roc_differences: list[float] = []
    pr_differences: list[float] = []
    skipped = 0
    for _ in range(replicates):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.asarray(
            [index for name in sampled for index in clusters[str(name)]],
            dtype=int,
        )
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            skipped += 1
            continue
        sample_roc7, sample_pr7 = metric_pair(sampled_labels, scores7[indices])
        sample_roc14, sample_pr14 = metric_pair(
            sampled_labels, scores14[indices]
        )
        roc_differences.append(sample_roc14 - sample_roc7)
        pr_differences.append(sample_pr14 - sample_pr7)

    if not roc_differences or not pr_differences:
        raise RuntimeError("All paired cluster-bootstrap replicates were invalid")
    roc_ci = [
        float(np.quantile(roc_differences, 0.025)),
        float(np.quantile(roc_differences, 0.975)),
    ]
    pr_ci = [
        float(np.quantile(pr_differences, 0.025)),
        float(np.quantile(pr_differences, 0.975)),
    ]
    return {
        "difference_direction": "Qwen2.5-14B minus Qwen2.5-7B",
        "qwen2_5_7b": {"roc_auc": roc7, "pr_auc": pr7},
        "qwen2_5_14b": {"roc_auc": roc14, "pr_auc": pr14},
        "paired_difference": {
            "roc_auc": roc14 - roc7,
            "roc_auc_ci": roc_ci,
            "roc_auc_ci_excludes_zero": bool(roc_ci[0] > 0 or roc_ci[1] < 0),
            "pr_auc": pr14 - pr7,
            "pr_auc_ci": pr_ci,
            "pr_auc_ci_excludes_zero": bool(pr_ci[0] > 0 or pr_ci[1] < 0),
        },
        "requested_replicates": replicates,
        "valid_replicates": len(roc_differences),
        "skipped_replicates": skipped,
        "source_document_cluster_count": len(names),
    }


def qwen14_hash(metrics: dict[str, Any], dataset: str) -> str:
    try:
        return str(metrics["datasets"][dataset]["score_file_sha256"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot read frozen Qwen14 {dataset} score hash from metrics"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen7-nf", required=True, type=Path)
    parser.add_argument("--qwen14-nf", required=True, type=Path)
    parser.add_argument("--qwen7-trec", required=True, type=Path)
    parser.add_argument("--qwen14-trec", required=True, type=Path)
    parser.add_argument("--qwen14-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {args.output}")
    for path in (
        args.qwen7_nf,
        args.qwen14_nf,
        args.qwen7_trec,
        args.qwen14_trec,
        args.qwen14_metrics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = load_json(args.qwen14_metrics)
    hashes = {
        "qwen7_nf": sha256_file(args.qwen7_nf),
        "qwen14_nf": sha256_file(args.qwen14_nf),
        "qwen7_trec": sha256_file(args.qwen7_trec),
        "qwen14_trec": sha256_file(args.qwen14_trec),
        "qwen14_metrics": sha256_file(args.qwen14_metrics),
    }
    if hashes["qwen7_nf"] != QWEN7_NF_SHA256:
        raise RuntimeError("Frozen Qwen2.5-7B NFCorpus score hash mismatch")
    if hashes["qwen7_trec"] != QWEN7_TREC_SHA256:
        raise RuntimeError("Frozen Qwen2.5-7B TREC score hash mismatch")
    if hashes["qwen14_nf"] != qwen14_hash(metrics, "nfcorpus"):
        raise RuntimeError("Frozen Qwen2.5-14B NFCorpus score hash mismatch")
    if hashes["qwen14_trec"] != qwen14_hash(metrics, "trec-covid"):
        raise RuntimeError("Frozen Qwen2.5-14B TREC score hash mismatch")

    qwen7_nf, qwen14_nf = pair_rows(
        args.qwen7_nf, args.qwen14_nf, "nfcorpus", 446
    )
    qwen7_trec, qwen14_trec = pair_rows(
        args.qwen7_trec, args.qwen14_trec, "trec-covid", 458
    )
    report = {
        "experiment_role": "paired_qwen2_5_scale_sensitivity_aligned_v1",
        "models": ["Qwen2.5-7B", "Qwen2.5-14B"],
        "interpretation_guard": (
            "This is a two-point same-family scale sensitivity comparison, "
            "not a scaling law or a universal model-size effect."
        ),
        "alignment_policy": "exclude_first_target_token_in_both_conditions",
        "no_bos": True,
        "no_separator": True,
        "score_direction": "larger score is more suspicious",
        "bootstrap": {
            "resampling_unit": "source document cluster",
            "paired": True,
            "requested_replicates": args.replicates,
            "seed": args.seed,
        },
        "inputs": {
            "paths": {
                "qwen7_nf": str(args.qwen7_nf),
                "qwen14_nf": str(args.qwen14_nf),
                "qwen7_trec": str(args.qwen7_trec),
                "qwen14_trec": str(args.qwen14_trec),
                "qwen14_metrics": str(args.qwen14_metrics),
            },
            "sha256": hashes,
        },
        "datasets": {
            "nfcorpus": paired_cluster_bootstrap(
                qwen7_nf, qwen14_nf, args.replicates, args.seed
            ),
            "trec-covid": paired_cluster_bootstrap(
                qwen7_trec, qwen14_trec, args.replicates, args.seed
            ),
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("QWEN2.5 14B-VS-7B PAIRED LOCALIZATION COMPARISON PASS")


if __name__ == "__main__":
    main()

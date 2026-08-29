#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cross_detector_common import load_json, sha256_file, write_json_atomic


def bucket(group_id: str, seed: int) -> float:
    payload = f"{seed}|{group_id}".encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value / float(16**16 - 1)


def quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def point_metrics(rows: list[dict]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray(
        [float(row["score_aligned"]) for row in rows], dtype=float
    )
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "score_mean_clean": float(scores[labels == 0].mean()),
        "score_mean_injected": float(scores[labels == 1].mean()),
    }


def cluster_bootstrap(
    rows: list[dict], replicates: int, seed: int
) -> dict[str, list[float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["source_doc_id"])].append(index)
    group_names = sorted(groups)
    rng = np.random.default_rng(seed)
    roc_values = []
    pr_values = []
    skipped = 0

    for _ in range(replicates):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        labels = np.asarray(
            [int(rows[index]["label"]) for index in indices], dtype=int
        )
        if len(np.unique(labels)) < 2:
            skipped += 1
            continue
        scores = np.asarray(
            [float(rows[index]["score_aligned"]) for index in indices],
            dtype=float,
        )
        roc_values.append(float(roc_auc_score(labels, scores)))
        pr_values.append(float(average_precision_score(labels, scores)))

    if not roc_values or not pr_values:
        raise RuntimeError("All cluster-bootstrap replicates were invalid")
    return {
        "roc_auc_ci": [
            float(np.quantile(roc_values, 0.025)),
            float(np.quantile(roc_values, 0.975)),
        ],
        "pr_auc_ci": [
            float(np.quantile(pr_values, 0.025)),
            float(np.quantile(pr_values, 0.975)),
        ],
        "requested_replicates": replicates,
        "valid_replicates": len(roc_values),
        "skipped_replicates": skipped,
        "cluster_count": len(group_names),
    }


def dataset_report(
    path: Path, dataset: str, replicates: int, seed: int
) -> dict:
    rows = load_json(path)
    expected = 446 if dataset == "nfcorpus" else 458
    expected_label = expected // 2
    labels = {
        "clean": sum(int(row["label"]) == 0 for row in rows),
        "injected": sum(int(row["label"]) == 1 for row in rows),
    }
    if len(rows) != expected or labels != {
        "clean": expected_label,
        "injected": expected_label,
    }:
        raise RuntimeError(
            f"Unexpected {dataset} score rows or labels: {len(rows)}, {labels}"
        )
    if any(row.get("dataset") != dataset for row in rows):
        raise RuntimeError(f"Dataset tag mismatch in {path}")

    return {
        "dataset": dataset,
        "score_file": str(path),
        "score_file_sha256": sha256_file(path),
        "record_count": len(rows),
        "label_counts": labels,
        "score_direction": "larger score is more suspicious",
        **point_metrics(rows),
        "source_document_cluster_bootstrap": cluster_bootstrap(
            rows, replicates, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfcorpus-scores", required=True, type=Path)
    parser.add_argument("--trec-scores", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--threshold-output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260609)
    parser.add_argument("--split-seed", type=int, default=20260714)
    args = parser.parse_args()

    if args.metrics_output.exists() or args.threshold_output.exists():
        raise FileExistsError(
            "Frozen metric/threshold output exists; refusing overwrite"
        )

    nf_report = dataset_report(
        args.nfcorpus_scores,
        "nfcorpus",
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    trec_report = dataset_report(
        args.trec_scores,
        "trec-covid",
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )

    metrics = {
        "experiment_role": "cross_detector_lm_robustness_aligned_v1",
        "model": "DeepSeek-LLM-7B-base",
        "alignment_policy": (
            "exclude_first_target_token_in_both_conditions"
        ),
        "no_bos": True,
        "no_separator": True,
        "datasets": {
            "nfcorpus": nf_report,
            "trec-covid": trec_report,
        },
        "bootstrap_seed": args.bootstrap_seed,
        "warning": (
            "Per-model cluster-bootstrap intervals do not by themselves prove "
            "a significant difference between detector LMs."
        ),
    }
    write_json_atomic(args.metrics_output, metrics)

    rows = load_json(args.nfcorpus_scores)
    clean_groups: dict[str, list[float]] = defaultdict(list)
    injected_scores = []
    for row in rows:
        if int(row["label"]) == 0:
            clean_groups[str(row["source_doc_id"])].append(
                float(row["score_aligned"])
            )
        else:
            injected_scores.append(float(row["score_aligned"]))

    calibration_groups = {
        group
        for group in clean_groups
        if bucket(group, args.split_seed) < 0.70
    }
    holdout_groups = set(clean_groups) - calibration_groups
    calibration = np.asarray(
        [
            score
            for group in calibration_groups
            for score in clean_groups[group]
        ],
        dtype=float,
    )
    holdout = np.asarray(
        [
            score
            for group in holdout_groups
            for score in clean_groups[group]
        ],
        dtype=float,
    )
    injected = np.asarray(injected_scores, dtype=float)

    if len(calibration) != 169 or len(holdout) != 54:
        raise RuntimeError(
            "Frozen group split mismatch: expected 169 calibration and 54 "
            f"holdout records, got {len(calibration)} and {len(holdout)}"
        )

    threshold_value = quantile_higher(calibration, 0.95)
    threshold = {
        "experiment_role": "aligned_end_to_end_sensitivity",
        "extension_role": "cross_detector_lm_robustness_aligned_v1",
        "model": "DeepSeek-LLM-7B-base",
        "budget_id": "model_specific_fpr05",
        "budget_definition": "model-specific clean-calibration target FPR",
        "score_direction": "larger score is more suspicious",
        "score_key": "score_aligned",
        "alignment_policy": (
            "exclude_first_target_token_in_both_conditions"
        ),
        "no_bos": True,
        "no_separator": True,
        "target_clean_fpr": 0.05,
        "quantile": 0.95,
        "threshold": threshold_value,
        "seed": args.split_seed,
        "group_split": True,
        "calibration_fraction": 0.70,
        "calibration_group_count": len(calibration_groups),
        "holdout_group_count": len(holdout_groups),
        "calibration_record_count": len(calibration),
        "holdout_record_count": len(holdout),
        "calibration_empirical_fpr": float(
            np.mean(calibration >= threshold_value)
        ),
        "holdout_empirical_fpr": float(
            np.mean(holdout >= threshold_value)
        ),
        "diagnostic_injected_tpr_at_frozen_threshold": float(
            np.mean(injected >= threshold_value)
        ),
        "scores_path": str(args.nfcorpus_scores),
        "scores_sha256": sha256_file(args.nfcorpus_scores),
        "metrics_path": str(args.metrics_output),
        "metrics_sha256": sha256_file(args.metrics_output),
        "warning": (
            "Threshold frozen before full-corpus deletion and verification. "
            "Do not retune using WSN or retrieval utility. This model-specific "
            "threshold is not the Qwen 5% threshold."
        ),
    }
    write_json_atomic(args.threshold_output, threshold)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(threshold, ensure_ascii=False, indent=2))
    print("DEEPSEEK METRICS AND MODEL-SPECIFIC FPR05 FREEZE PASS")


if __name__ == "__main__":
    main()


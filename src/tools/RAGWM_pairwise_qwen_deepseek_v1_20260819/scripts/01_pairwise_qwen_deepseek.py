#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


QWEN_NF_SHA256 = (
    "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218"
)
QWEN_TREC_SHA256 = (
    "d526aa72f48662521b67339f58849c1ea6c26de4042eb8b3dd842403f8ed293f"
)
DEEPSEEK_NF_SHA256 = (
    "fd29d143ac4c08f23d36b8fac5d4bad7e6d85172f68bb7eb4c1d19275552cc86"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def qwen_source_id(row: dict[str, Any]) -> str:
    for key in (
        "source_doc_id",
        "doc_id",
        "document_id",
        "chroma_id",
        "id",
    ):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    raise RuntimeError("Qwen row has no source-document identifier")


def identity_text(row: dict[str, Any], key: str) -> str:
    aliases = {
        "context": ("context", "context_text"),
        "target": ("target", "target_text"),
    }
    for candidate in aliases[key]:
        if candidate in row:
            return str(row.get(candidate) or "")
    return ""


def pair_rows(
    qwen_path: Path,
    deepseek_path: Path,
    dataset: str,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qwen = load_json(qwen_path)
    deepseek = load_json(deepseek_path)
    if not isinstance(qwen, list) or not isinstance(deepseek, list):
        raise RuntimeError(f"{dataset}: score files must contain JSON lists")
    if len(qwen) != expected_count or len(deepseek) != expected_count:
        raise RuntimeError(
            f"{dataset}: expected {expected_count} rows, got "
            f"Qwen={len(qwen)}, DeepSeek={len(deepseek)}"
        )

    deep_by_index = {}
    for fallback_index, row in enumerate(deepseek):
        index = int(row.get("input_index", fallback_index))
        if index in deep_by_index:
            raise RuntimeError(f"{dataset}: duplicate DeepSeek input_index {index}")
        deep_by_index[index] = row
    if set(deep_by_index) != set(range(expected_count)):
        raise RuntimeError(f"{dataset}: incomplete DeepSeek input_index set")

    ordered_deep = []
    clean = injected = 0
    for index, qrow in enumerate(qwen):
        drow = deep_by_index[index]
        qlabel = int(qrow["label"])
        dlabel = int(drow["label"])
        if qlabel != dlabel:
            raise RuntimeError(f"{dataset}: label mismatch at row {index}")
        clean += qlabel == 0
        injected += qlabel == 1

        for text_key in ("context", "target"):
            qtext = identity_text(qrow, text_key).strip()
            dtext = identity_text(drow, text_key).strip()
            if qtext and dtext and qtext != dtext:
                raise RuntimeError(
                    f"{dataset}: {text_key} mismatch at row {index}"
                )
        if qwen_source_id(qrow) != str(drow["source_doc_id"]):
            raise RuntimeError(
                f"{dataset}: source-document mismatch at row {index}"
            )
        for row_name, row in (("Qwen", qrow), ("DeepSeek", drow)):
            if "score_aligned" not in row:
                raise RuntimeError(
                    f"{dataset}: {row_name} row {index} lacks score_aligned"
                )
            score = float(row["score_aligned"])
            if not np.isfinite(score):
                raise RuntimeError(
                    f"{dataset}: {row_name} row {index} score is non-finite"
                )
        ordered_deep.append(drow)

    expected_each = expected_count // 2
    if (clean, injected) != (expected_each, expected_each):
        raise RuntimeError(
            f"{dataset}: expected balanced labels, got {(clean, injected)}"
        )
    return qwen, ordered_deep


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def paired_cluster_bootstrap(
    qwen: list[dict[str, Any]],
    deepseek: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in qwen], dtype=int)
    qwen_scores = np.asarray(
        [float(row["score_aligned"]) for row in qwen], dtype=float
    )
    deep_scores = np.asarray(
        [float(row["score_aligned"]) for row in deepseek], dtype=float
    )
    qwen_roc, qwen_pr = metrics(labels, qwen_scores)
    deep_roc, deep_pr = metrics(labels, deep_scores)

    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(deepseek):
        clusters[str(row["source_doc_id"])].append(index)
    cluster_names = sorted(clusters)
    rng = np.random.default_rng(seed)
    roc_differences = []
    pr_differences = []
    skipped = 0
    for _ in range(replicates):
        sampled = rng.choice(
            cluster_names, size=len(cluster_names), replace=True
        )
        indices = np.asarray(
            [index for group in sampled for index in clusters[str(group)]],
            dtype=int,
        )
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            skipped += 1
            continue
        q_roc, q_pr = metrics(sampled_labels, qwen_scores[indices])
        d_roc, d_pr = metrics(sampled_labels, deep_scores[indices])
        roc_differences.append(q_roc - d_roc)
        pr_differences.append(q_pr - d_pr)

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
        "score_direction": "larger score is more suspicious",
        "difference_direction": "Qwen2.5-7B minus DeepSeek-LLM-7B-base",
        "qwen": {"roc_auc": qwen_roc, "pr_auc": qwen_pr},
        "deepseek": {"roc_auc": deep_roc, "pr_auc": deep_pr},
        "paired_difference": {
            "roc_auc": qwen_roc - deep_roc,
            "roc_auc_ci": roc_ci,
            "roc_auc_ci_excludes_zero": bool(roc_ci[0] > 0 or roc_ci[1] < 0),
            "pr_auc": qwen_pr - deep_pr,
            "pr_auc_ci": pr_ci,
            "pr_auc_ci_excludes_zero": bool(pr_ci[0] > 0 or pr_ci[1] < 0),
        },
        "requested_replicates": replicates,
        "valid_replicates": len(roc_differences),
        "skipped_replicates": skipped,
        "source_document_cluster_count": len(cluster_names),
    }


def verification_map(path: Path) -> tuple[list[Any], dict[tuple[str, ...], Any]]:
    rows = load_json(path)
    if not isinstance(rows, list) or len(rows) != 30:
        raise RuntimeError(f"Expected 30 verification rows: {path}")
    mapping = {}
    for row in rows:
        unit = tuple(str(value) for value in row[0])
        if unit in mapping:
            raise RuntimeError(f"Duplicate verification unit: {unit}")
        flag = int(row[2][0])
        if flag not in (0, 1, 2):
            raise RuntimeError(f"Unexpected verification flag {flag}: {unit}")
        mapping[unit] = row
    return rows, mapping


def endpoint_pairwise(qwen_path: Path, deepseek_path: Path) -> dict[str, Any]:
    qwen_rows, qwen = verification_map(qwen_path)
    deep_rows, deep = verification_map(deepseek_path)
    if set(qwen) != set(deep):
        raise RuntimeError("Qwen and DeepSeek verification unit sets differ")
    qwen_wsn = sum(int(row[2][0]) == 1 for row in qwen_rows)
    deep_wsn = sum(int(row[2][0]) == 1 for row in deep_rows)
    qwen_unknown = sum(int(row[2][0]) == 2 for row in qwen_rows)
    deep_unknown = sum(int(row[2][0]) == 2 for row in deep_rows)
    if (qwen_wsn, qwen_unknown) != (21, 0):
        raise RuntimeError(
            f"Current-provider Qwen FPR05 is not frozen 21/30: "
            f"WSN={qwen_wsn}, unknown={qwen_unknown}"
        )
    if (deep_wsn, deep_unknown) != (24, 0):
        raise RuntimeError(
            f"DeepSeek FPR05 is not frozen 24/30: "
            f"WSN={deep_wsn}, unknown={deep_unknown}"
        )

    qwen_positive_deep_negative = 0
    qwen_negative_deep_positive = 0
    both_positive = 0
    both_negative = 0
    details = []
    for unit in sorted(qwen):
        qflag = int(qwen[unit][2][0])
        dflag = int(deep[unit][2][0])
        if qflag == 1 and dflag == 0:
            qwen_positive_deep_negative += 1
        elif qflag == 0 and dflag == 1:
            qwen_negative_deep_positive += 1
        elif qflag == 1 and dflag == 1:
            both_positive += 1
        else:
            both_negative += 1
        details.append(
            {
                "watermark_unit": list(unit),
                "qwen_fpr05_flag": qflag,
                "deepseek_fpr05_flag": dflag,
            }
        )

    b = qwen_positive_deep_negative
    c = qwen_negative_deep_positive
    return {
        "comparison_role": "descriptive_non_budget_matched_sensitivity",
        "verifier_reference": "current_provider_verifier_audit_v1_20260818",
        "qwen_fpr05_wsn": qwen_wsn,
        "deepseek_fpr05_wsn": deep_wsn,
        "wsn_difference_deepseek_minus_qwen": deep_wsn - qwen_wsn,
        "both_positive": both_positive,
        "both_negative": both_negative,
        "qwen_positive_deepseek_negative": b,
        "qwen_negative_deepseek_positive": c,
        "discordant_count": b + c,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(b, c),
        "qwen_unknown_count": qwen_unknown,
        "deepseek_unknown_count": deep_unknown,
        "qwen_realized_budget": {
            "modified_documents": 174,
            "deleted_characters": 37586,
        },
        "deepseek_realized_budget": {
            "modified_documents": 96,
            "deleted_characters": 15451,
        },
        "warning": (
            "Detector-specific 5% FPR thresholds produced different realized "
            "deletion budgets. This paired verifier comparison is descriptive "
            "and must not be reported as a budget-matched model comparison."
        ),
        "details": details,
    }


def deepseek_frozen_score_hash(metrics_path: Path, dataset: str) -> str:
    metrics_json = load_json(metrics_path)
    try:
        return str(metrics_json["datasets"][dataset]["score_file_sha256"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Cannot read frozen DeepSeek {dataset} score hash from metrics"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-nf", required=True, type=Path)
    parser.add_argument("--deepseek-nf", required=True, type=Path)
    parser.add_argument("--qwen-trec", required=True, type=Path)
    parser.add_argument("--deepseek-trec", required=True, type=Path)
    parser.add_argument("--deepseek-metrics", required=True, type=Path)
    parser.add_argument("--qwen-verify", required=True, type=Path)
    parser.add_argument("--deepseek-verify", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {args.output}")
    for path in (
        args.qwen_nf,
        args.deepseek_nf,
        args.qwen_trec,
        args.deepseek_trec,
        args.deepseek_metrics,
        args.qwen_verify,
        args.deepseek_verify,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {
        "qwen_nf": sha256_file(args.qwen_nf),
        "deepseek_nf": sha256_file(args.deepseek_nf),
        "qwen_trec": sha256_file(args.qwen_trec),
        "deepseek_trec": sha256_file(args.deepseek_trec),
        "deepseek_metrics": sha256_file(args.deepseek_metrics),
        "qwen_verify": sha256_file(args.qwen_verify),
        "deepseek_verify": sha256_file(args.deepseek_verify),
    }
    if hashes["qwen_nf"] != QWEN_NF_SHA256:
        raise RuntimeError("Frozen Qwen NFCorpus score hash mismatch")
    if hashes["deepseek_nf"] != DEEPSEEK_NF_SHA256:
        raise RuntimeError("Frozen DeepSeek NFCorpus score hash mismatch")
    if hashes["qwen_trec"] != QWEN_TREC_SHA256:
        raise RuntimeError("Frozen Qwen TREC score hash mismatch")
    expected_deep_trec = deepseek_frozen_score_hash(
        args.deepseek_metrics, "trec-covid"
    )
    if hashes["deepseek_trec"] != expected_deep_trec:
        raise RuntimeError("Frozen DeepSeek TREC score hash mismatch")

    qwen_nf, deep_nf = pair_rows(
        args.qwen_nf, args.deepseek_nf, "nfcorpus", 446
    )
    qwen_trec, deep_trec = pair_rows(
        args.qwen_trec, args.deepseek_trec, "trec-covid", 458
    )

    report = {
        "experiment_role": "paired_cross_detector_lm_comparison_v1",
        "models": ["Qwen2.5-7B", "DeepSeek-LLM-7B-base"],
        "alignment_policy": "exclude_first_target_token_in_both_conditions",
        "no_bos": True,
        "no_separator": True,
        "bootstrap": {
            "resampling_unit": "source document cluster",
            "paired": True,
            "requested_replicates": args.replicates,
            "seed": args.seed,
        },
        "inputs": {
            "paths": {
                "qwen_nf": str(args.qwen_nf),
                "deepseek_nf": str(args.deepseek_nf),
                "qwen_trec": str(args.qwen_trec),
                "deepseek_trec": str(args.deepseek_trec),
                "deepseek_metrics": str(args.deepseek_metrics),
                "qwen_verify": str(args.qwen_verify),
                "deepseek_verify": str(args.deepseek_verify),
            },
            "sha256": hashes,
        },
        "boundary_localization": {
            "nfcorpus": paired_cluster_bootstrap(
                qwen_nf, deep_nf, args.replicates, args.seed
            ),
            "trec-covid": paired_cluster_bootstrap(
                qwen_trec, deep_trec, args.replicates, args.seed
            ),
        },
        "end_to_end": endpoint_pairwise(
            args.qwen_verify, args.deepseek_verify
        ),
        "interpretation_guard": (
            "Boundary paired-difference intervals may support model-specific "
            "localization differences. End-to-end results remain a non-budget-"
            "matched sensitivity comparison and do not replace sealed strict-v1."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("QWEN-DEEPSEEK PAIRED STATISTICS PASS")


if __name__ == "__main__":
    main()


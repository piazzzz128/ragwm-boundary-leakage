#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_SCORE_SHA256 = "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218"
EXPECTED_FPR05_THRESHOLD = -0.1832180580405571
FPR_POINTS = (
    ("fpr01", 0.01),
    ("fpr025", 0.025),
    ("fpr05", 0.05),
    ("fpr10", 0.10),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket(group_id: str, seed: int) -> float:
    payload = f"{seed}|{group_id}".encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value / float(16**16 - 1)


def quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    args = parser.parse_args()

    scores_hash = sha256_file(args.scores)
    if scores_hash != EXPECTED_SCORE_SHA256:
        raise RuntimeError(
            "Frozen aligned score hash mismatch. Expected "
            f"{EXPECTED_SCORE_SHA256}, got {scores_hash}"
        )

    rows = json.loads(args.scores.read_text(encoding="utf-8"))
    if len(rows) != 446:
        raise RuntimeError(f"Expected 446 score rows, found {len(rows)}")
    if any("score_aligned" not in row for row in rows):
        raise RuntimeError("At least one row lacks score_aligned")

    clean_groups: dict[str, list[float]] = defaultdict(list)
    inject_scores: list[float] = []
    for row in rows:
        if int(row["label"]) == 0:
            group = str(row.get("source_doc_id") or row.get("group_id") or row["id"])
            clean_groups[group].append(float(row["score_aligned"]))
        else:
            inject_scores.append(float(row["score_aligned"]))

    calibration_groups = {
        group for group in clean_groups
        if bucket(group, args.seed) < args.calibration_fraction
    }
    holdout_groups = set(clean_groups) - calibration_groups
    calibration_scores = np.asarray(
        [score for group in calibration_groups for score in clean_groups[group]],
        dtype=float,
    )
    holdout_scores = np.asarray(
        [score for group in holdout_groups for score in clean_groups[group]],
        dtype=float,
    )
    inject = np.asarray(inject_scores, dtype=float)

    if len(calibration_scores) != 169 or len(holdout_scores) != 54:
        raise RuntimeError(
            "Frozen group split mismatch: expected 169 calibration and 54 holdout "
            f"records, got {len(calibration_scores)} and {len(holdout_scores)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for budget_id, target_fpr in FPR_POINTS:
        threshold = quantile_higher(calibration_scores, 1.0 - target_fpr)
        report = {
            "experiment_role": "aligned_end_to_end_sensitivity",
            "extension_role": "canonical_nfcorpus_aligned_multibudget_v1",
            "budget_id": budget_id,
            "budget_definition": "clean-calibration target FPR",
            "score_direction": "larger score is more suspicious",
            "score_key": "score_aligned",
            "alignment_policy": "exclude_first_target_token_in_both_conditions",
            "no_bos": True,
            "no_separator": True,
            "target_clean_fpr": target_fpr,
            "quantile": 1.0 - target_fpr,
            "threshold": threshold,
            "seed": args.seed,
            "group_split": True,
            "calibration_fraction": args.calibration_fraction,
            "calibration_group_count": len(calibration_groups),
            "holdout_group_count": len(holdout_groups),
            "calibration_record_count": int(len(calibration_scores)),
            "holdout_record_count": int(len(holdout_scores)),
            "calibration_empirical_fpr": float(np.mean(calibration_scores >= threshold)),
            "holdout_empirical_fpr": float(np.mean(holdout_scores >= threshold)),
            "diagnostic_inject_tpr_at_frozen_threshold": float(np.mean(inject >= threshold)),
            "scores_path": str(args.scores),
            "scores_sha256": scores_hash,
            "warning": (
                "Threshold frozen before full-corpus deletion and verification. "
                "Do not retune using WSN or retrieval utility."
            ),
        }
        output = args.output_dir / f"threshold_{budget_id}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["threshold_file"] = str(output)
        report["threshold_file_sha256"] = sha256_file(output)
        reports.append(report)

    fpr05 = next(row for row in reports if row["budget_id"] == "fpr05")
    if abs(float(fpr05["threshold"]) - EXPECTED_FPR05_THRESHOLD) > 1e-12:
        raise RuntimeError(
            "Recomputed 5% threshold does not reproduce the frozen aligned result: "
            f"{fpr05['threshold']} vs {EXPECTED_FPR05_THRESHOLD}"
        )

    manifest = {
        "experiment_role": "canonical_nfcorpus_aligned_multibudget_threshold_manifest_v1",
        "frozen_score_sha256": scores_hash,
        "seed": args.seed,
        "points": reports,
        "reuse_rule": (
            "fpr05 reuses the previously frozen aligned E2 result; only fpr01, "
            "fpr025, and fpr10 require new end-to-end runs."
        ),
    }
    manifest_path = args.output_dir / "threshold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("MULTIBUDGET THRESHOLD FREEZE PASS")


if __name__ == "__main__":
    main()

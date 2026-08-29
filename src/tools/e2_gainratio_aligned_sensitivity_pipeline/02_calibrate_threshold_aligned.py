from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from e2_common_aligned import sha256_file, write_json_atomic


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
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-key", default="score_aligned")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    parser.add_argument("--target-clean-fpr", type=float, default=0.05)
    args = parser.parse_args()

    score_path = Path(args.scores)
    rows = json.loads(score_path.read_text(encoding="utf-8"))

    missing = [row.get("id") for row in rows if args.score_key not in row]
    if missing:
        raise RuntimeError(
            f"{len(missing)} rows lack score key {args.score_key!r}"
        )

    clean = [row for row in rows if int(row["label"]) == 0]
    inject = [row for row in rows if int(row["label"]) == 1]

    clean_groups = defaultdict(list)
    for row in clean:
        group = str(
            row.get("source_doc_id")
            or row.get("group_id")
            or row["id"]
        )
        clean_groups[group].append(float(row[args.score_key]))

    calibration_groups = {
        group
        for group in clean_groups
        if bucket(group, args.seed) < args.calibration_fraction
    }
    holdout_groups = set(clean_groups) - calibration_groups
    if not calibration_groups or not holdout_groups:
        raise RuntimeError("Calibration split produced an empty partition")

    calibration_scores = np.asarray(
        [
            score
            for group in calibration_groups
            for score in clean_groups[group]
        ],
        dtype=float,
    )
    holdout_scores = np.asarray(
        [
            score
            for group in holdout_groups
            for score in clean_groups[group]
        ],
        dtype=float,
    )
    inject_scores = np.asarray(
        [float(row[args.score_key]) for row in inject],
        dtype=float,
    )

    quantile = 1.0 - args.target_clean_fpr
    threshold = quantile_higher(calibration_scores, quantile)

    report = {
        "experiment_role": "aligned_end_to_end_sensitivity",
        "policy": "aligned clean calibration score quantile",
        "score_direction": "larger score is more suspicious",
        "score_key": args.score_key,
        "alignment_policy": (
            "exclude_first_target_token_in_both_conditions"
        ),
        "target_clean_fpr": args.target_clean_fpr,
        "quantile": quantile,
        "threshold": threshold,
        "seed": args.seed,
        "group_split": True,
        "calibration_fraction": args.calibration_fraction,
        "calibration_group_count": len(calibration_groups),
        "holdout_group_count": len(holdout_groups),
        "calibration_record_count": int(len(calibration_scores)),
        "holdout_record_count": int(len(holdout_scores)),
        "calibration_empirical_fpr": float(
            np.mean(calibration_scores >= threshold)
        ),
        "holdout_empirical_fpr": float(
            np.mean(holdout_scores >= threshold)
        ),
        "diagnostic_inject_tpr_at_fixed_threshold": float(
            np.mean(inject_scores >= threshold)
        ),
        "scores_path": str(score_path),
        "scores_sha256": sha256_file(score_path),
        "warning": (
            "Threshold is frozen before aligned E2 verification. "
            "Do not retune it using WSN."
        ),
    }

    output = Path(args.output)
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("ALIGNED THRESHOLD CALIBRATION PASS")


if __name__ == "__main__":
    main()

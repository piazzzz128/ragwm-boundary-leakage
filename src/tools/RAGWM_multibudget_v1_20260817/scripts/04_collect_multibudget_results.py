#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POINTS = ("fpr01", "fpr025", "fpr05", "fpr10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--existing-aligned-dir", required=True, type=Path)
    parser.add_argument("--utility", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    utility = load(args.utility)
    rows = []
    for point in POINTS:
        if point == "fpr05":
            condition = args.existing_aligned_dir
            comparison_path = condition / "metrics/e0_e1_aligned_comparison.json"
        else:
            condition = args.out_root / f"conditions/qwen25_7b_{point}"
            comparison_path = condition / "metrics/e0_e1_budget_comparison.json"

        threshold_path = args.out_root / f"thresholds/threshold_{point}.json"
        sanitize_path = condition / "sanitization/gainratio_sanitization_summary.json"
        required = [threshold_path, sanitize_path, comparison_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing registered artifacts:\n" + "\n".join(missing))

        threshold = load(threshold_path)
        sanitization = load(sanitize_path)
        comparison = load(comparison_path)["summary"]
        utility_name = f"qwen25_7b_{point}"
        metrics = utility["metrics"][utility_name]
        delta = utility["metric_deltas_vs_baseline"][utility_name]
        ci = utility["paired_bootstrap_differences_vs_baseline"][utility_name]

        rows.append({
            "budget_id": point,
            "target_clean_fpr": threshold["target_clean_fpr"],
            "threshold": threshold["threshold"],
            "modified_documents": sanitization["modified_document_count"],
            "modified_document_rate": sanitization["modified_document_rate"],
            "deleted_characters": sanitization["total_deleted_chars"],
            "deleted_character_rate": sanitization["deleted_char_rate"],
            "candidate_evaluations": sanitization["candidate_evaluation_count"],
            "deletion_steps": sanitization["deletion_step_count"],
            "wsn": comparison["aligned_gainratio_WSN"],
            "mcnemar_exact_two_sided_p_vs_e0": comparison["mcnemar_exact_two_sided_p"],
            "unknown": comparison["aligned_unknown_count"],
            "retrieval_metrics": metrics,
            "retrieval_delta_vs_e0": delta,
            "retrieval_paired_bootstrap": ci,
            "artifact_hashes": {
                "threshold": sha256_file(threshold_path),
                "sanitization_summary": sha256_file(sanitize_path),
                "verification_comparison": sha256_file(comparison_path),
            },
        })

    report = {
        "experiment_role": "canonical_nfcorpus_aligned_multibudget_attack_utility_v1",
        "status": "complete",
        "main_result_guard": "Does not overwrite NFCorpus sealed strict-v1.",
        "detector_lm": "Qwen2.5-7B",
        "alignment_policy": "exclude_first_target_token_in_both_conditions",
        "max_steps": 8,
        "verification_units": 30,
        "retrieval_queries": utility["query_count"],
        "bootstrap_replicates": utility["n_boot"],
        "bootstrap_seed": utility["seed"],
        "points": rows,
        "interpretation_guard": (
            "These are complete end-to-end budget points. Report verification "
            "weakening and measured utility changes; do not infer a decision flip "
            "unless WSN is at or below the original threshold."
        ),
        "utility_file": str(args.utility),
        "utility_sha256": sha256_file(args.utility),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("MULTIBUDGET RESULT COLLECTION PASS")


if __name__ == "__main__":
    main()

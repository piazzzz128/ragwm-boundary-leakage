#!/usr/bin/env python3
"""Collect already-generated 14B artifacts without inventing missing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_detector_common import load_json, sha256_file, write_json_atomic


def artifact(path: Path | None):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "content": load_json(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=Path)
    parser.add_argument("--paired-scale", required=True, type=Path)
    parser.add_argument("--endpoint-comparison", type=Path)
    parser.add_argument("--sanitization-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {args.output}")
    preflight = artifact(args.preflight)
    metrics = artifact(args.metrics)
    threshold = artifact(args.threshold)
    paired = artifact(args.paired_scale)
    endpoint = artifact(args.endpoint_comparison)
    sanitization = artifact(args.sanitization_summary)

    if not preflight["content"].get("pass"):
        raise RuntimeError("Preflight is not marked pass")
    if metrics["content"].get("model") != "Qwen2.5-14B":
        raise RuntimeError("Boundary metrics model mismatch")
    if threshold["content"].get("model") != "Qwen2.5-14B":
        raise RuntimeError("Threshold model mismatch")
    if (endpoint is None) != (sanitization is None):
        raise RuntimeError(
            "Endpoint comparison and sanitization summary must be supplied together"
        )

    status = (
        "complete_with_e2e"
        if endpoint is not None
        else "localization_complete_e2e_pending"
    )
    report = {
        "registry_entry_role": "qwen2_5_14b_aligned_extension_v1",
        "status": status,
        "model": "Qwen2.5-14B",
        "protocol_layer": "canonical_aligned_extension",
        "alignment_policy": "exclude_first_target_token_in_both_conditions",
        "no_bos": True,
        "no_separator": True,
        "verifier_reference_if_e2e": {"e0": 25, "e1": 4},
        "must_not_merge_with_sealed_26_5": True,
        "artifacts": {
            "preflight": preflight,
            "boundary_metrics": metrics,
            "fpr05_threshold": threshold,
            "paired_qwen14_minus_qwen7": paired,
            "endpoint_comparison": endpoint,
            "sanitization_summary": sanitization,
        },
        "claim_guards": [
            "do_not_claim_scaling_law_from_two_same_family_points",
            "do_not_claim_universal_cross_model_robustness",
            "do_not_rank_endpoint_wsn_when_realized_edit_budgets_differ",
            "do_not_claim_ownership_bypass_if_wsn_remains_above_2",
            "do_not_replace_negative_results_by_posthoc_score_negation",
        ],
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"QWEN2.5-14B RESULT COLLECTION PASS: {status}")


if __name__ == "__main__":
    main()

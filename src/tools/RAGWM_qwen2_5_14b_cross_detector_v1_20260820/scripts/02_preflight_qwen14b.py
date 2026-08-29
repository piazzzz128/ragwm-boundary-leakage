#!/usr/bin/env python3
"""Freeze the 14B experiment inputs and the aligned-extension WSN reference."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from cross_detector_common import load_json, sha256_file, write_json_atomic


EXPECTED_INPUTS = {
    "nfcorpus_boundaries": "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218",
    "trec_boundaries": "d526aa72f48662521b67339f58849c1ea6c26de4042eb8b3dd842403f8ed293f",
    "attack_input": "52c72f64043e4a668b5084b75ac0cfa92f63b77c3493875cf788ee5021da8e38",
    "aligned_e0": "583eb66725901070f709c6938e22ca3e350d9a39a580e4237a72361f9088d1f5",
    "aligned_e1": "3e0253bdac9cc93878a64d641ddb6ed969144e61e38795f4423a8d0c8079c251",
}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verifier_counts(path: Path) -> dict[str, int]:
    rows = load_json(path)
    return {
        "rows": len(rows),
        "yes": sum(row[2][0] == 1 for row in rows),
        "no": sum(row[2][0] == 0 for row in rows),
        "unknown": sum(row[2][0] == 2 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-freeze", required=True, type=Path)
    parser.add_argument("--nfcorpus-boundaries", required=True, type=Path)
    parser.add_argument("--trec-boundaries", required=True, type=Path)
    parser.add_argument("--attack-input", required=True, type=Path)
    parser.add_argument("--aligned-e0", required=True, type=Path)
    parser.add_argument("--aligned-e1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Preflight output exists; refusing overwrite: {args.output}"
        )
    required = [
        args.model_freeze,
        args.nfcorpus_boundaries,
        args.trec_boundaries,
        args.attack_input,
        args.aligned_e0,
        args.aligned_e1,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    model_freeze = load_json(args.model_freeze)
    input_hashes = {
        "nfcorpus_boundaries": sha256_file(args.nfcorpus_boundaries),
        "trec_boundaries": sha256_file(args.trec_boundaries),
        "attack_input": sha256_file(args.attack_input),
        "aligned_e0": sha256_file(args.aligned_e0),
        "aligned_e1": sha256_file(args.aligned_e1),
    }
    nf_rows = load_json(args.nfcorpus_boundaries)
    trec_rows = load_json(args.trec_boundaries)
    nf_labels = {
        "clean": sum(int(row["label"]) == 0 for row in nf_rows),
        "injected": sum(int(row["label"]) == 1 for row in nf_rows),
    }
    trec_labels = {
        "clean": sum(int(row["label"]) == 0 for row in trec_rows),
        "injected": sum(int(row["label"]) == 1 for row in trec_rows),
    }
    e0_counts = verifier_counts(args.aligned_e0)
    e1_counts = verifier_counts(args.aligned_e1)

    checks = {
        "model_freeze_pass": model_freeze.get("pass") is True,
        "model_is_qwen2_5_14b_base": (
            model_freeze.get("model_name") == "Qwen2.5-14B"
            and model_freeze.get("model_stage") == "base_pretrained"
            and model_freeze.get("quantized") is False
        ),
        "input_hashes_match_registry": input_hashes == EXPECTED_INPUTS,
        "nfcorpus_is_446_balanced": (
            len(nf_rows) == 446
            and nf_labels == {"clean": 223, "injected": 223}
        ),
        "trec_is_458_balanced": (
            len(trec_rows) == 458
            and trec_labels == {"clean": 229, "injected": 229}
        ),
        "attack_input_is_3633": count_jsonl(args.attack_input) == 3633,
        "aligned_e0_is_25_of_30": (
            e0_counts == {"rows": 30, "yes": 25, "no": 5, "unknown": 0}
        ),
        "aligned_e1_is_4_of_30": (
            e1_counts == {"rows": 30, "yes": 4, "no": 26, "unknown": 0}
        ),
    }
    report = {
        "experiment_role": "cross_detector_lm_robustness_aligned_v1",
        "extension_id": "qwen2_5_14b_base_v1_20260820",
        "python": platform.python_version(),
        "model_name": "Qwen2.5-14B",
        "model_dir": model_freeze.get("model_dir"),
        "model_freeze": str(args.model_freeze),
        "model_freeze_sha256": sha256_file(args.model_freeze),
        "alignment_policy": "exclude_first_target_token_in_both_conditions",
        "no_bos": True,
        "no_separator": True,
        "max_context_tokens": 1024,
        "max_target_tokens": 128,
        "score_formula": "score=-(loss_target-loss_target_given_context)/loss_target",
        "score_direction": "larger score is more suspicious",
        "aligned_extension_verifier_reference": {"e0_wsn": 25, "e1_wsn": 4},
        "registry_provenance_reference": "current_provider_verifier_audit_v1_20260818",
        "inputs": {
            "nfcorpus_boundaries": str(args.nfcorpus_boundaries),
            "trec_boundaries": str(args.trec_boundaries),
            "attack_input": str(args.attack_input),
            "aligned_e0": str(args.aligned_e0),
            "aligned_e1": str(args.aligned_e1),
        },
        "input_hashes": input_hashes,
        "nfcorpus_rows": len(nf_rows),
        "nfcorpus_labels": nf_labels,
        "trec_rows": len(trec_rows),
        "trec_labels": trec_labels,
        "attack_input_rows": count_jsonl(args.attack_input),
        "aligned_e0_counts": e0_counts,
        "aligned_e1_counts": e1_counts,
        "checks": checks,
        "pass": all(checks.values()),
        "claim_guard": (
            "This aligned 14B extension must not overwrite or be pooled with "
            "the sealed strict-v1 E0/E1/E2=26/5/17 result."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise RuntimeError("QWEN2.5-14B EXPERIMENT PREFLIGHT FAILED")
    print("QWEN2.5-14B EXPERIMENT PREFLIGHT PASS")


if __name__ == "__main__":
    main()

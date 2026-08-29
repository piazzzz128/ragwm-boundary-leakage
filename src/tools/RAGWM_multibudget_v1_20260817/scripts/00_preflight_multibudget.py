#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


EXPECTED_SCORE_SHA256 = "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218"
EXPECTED_ATTACK_INPUT_SHA256 = "52c72f64043e4a668b5084b75ac0cfa92f63b77c3493875cf788ee5021da8e38"
EXPECTED_EXISTING_FPR05_THRESHOLD = -0.1832180580405571


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--attack-input", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--existing-fpr05", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    required = [
        args.scores,
        args.attack_input,
        args.model_path / "config.json",
        args.existing_fpr05,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    score_rows = json.loads(args.scores.read_text(encoding="utf-8"))
    labels = {
        "clean": sum(int(row["label"]) == 0 for row in score_rows),
        "injected": sum(int(row["label"]) == 1 for row in score_rows),
    }
    missing_score = sum("score_aligned" not in row for row in score_rows)
    attack_input_count = count_jsonl(args.attack_input)
    existing = json.loads(args.existing_fpr05.read_text(encoding="utf-8"))

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover - environment diagnostic
        cuda_available = False
        gpu_name = None
        torch_version = f"unavailable: {exc}"

    report = {
        "experiment_role": "canonical_nfcorpus_aligned_multibudget_preflight_v1",
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "gpu": gpu_name,
        "scores": str(args.scores),
        "scores_sha256": sha256_file(args.scores),
        "score_count": len(score_rows),
        "label_counts": labels,
        "rows_missing_score_aligned": missing_score,
        "attack_input": str(args.attack_input),
        "attack_input_sha256": sha256_file(args.attack_input),
        "attack_input_count": attack_input_count,
        "model_path": str(args.model_path),
        "existing_fpr05": str(args.existing_fpr05),
        "existing_fpr05_threshold": existing.get("threshold"),
        "checks": {
            "scores_hash_matches_frozen": sha256_file(args.scores) == EXPECTED_SCORE_SHA256,
            "attack_input_hash_matches_frozen": sha256_file(args.attack_input) == EXPECTED_ATTACK_INPUT_SHA256,
            "score_count_is_446": len(score_rows) == 446,
            "labels_are_223_plus_223": labels == {"clean": 223, "injected": 223},
            "all_rows_have_score_aligned": missing_score == 0,
            "attack_input_count_is_3633": attack_input_count == 3633,
            "existing_fpr05_threshold_matches": abs(float(existing.get("threshold")) - EXPECTED_EXISTING_FPR05_THRESHOLD) <= 1e-12,
            "cuda_available": cuda_available,
        },
    }
    report["pass"] = all(report["checks"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise RuntimeError("MULTIBUDGET PREFLIGHT FAILED; do not start a formal run")
    print("MULTIBUDGET PREFLIGHT PASS")


if __name__ == "__main__":
    main()

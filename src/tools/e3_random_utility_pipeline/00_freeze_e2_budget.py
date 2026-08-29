from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from e3_common import read_json, read_jsonl, sha256_file, write_json_atomic


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = pos - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--deletion-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    log_path = Path(args.deletion_log)
    summary = read_json(summary_path)
    rows = read_jsonl(log_path)

    required_summary = [
        "input_document_count",
        "modified_document_count",
        "total_deleted_chars",
        "deleted_char_rate",
    ]
    missing = [key for key in required_summary if key not in summary]
    if missing:
        raise RuntimeError(f"Missing E2 summary fields: {missing}")

    lengths = [int(row["deleted_char_count"]) for row in rows]
    steps = [int(row.get("deletion_step_count", 1)) for row in rows]

    if len(rows) != int(summary["modified_document_count"]):
        raise RuntimeError(
            "E2 deletion-log row count does not equal modified_document_count: "
            f"{len(rows)} != {summary['modified_document_count']}"
        )
    if sum(lengths) != int(summary["total_deleted_chars"]):
        raise RuntimeError(
            "E2 deletion-log character total does not equal summary: "
            f"{sum(lengths)} != {summary['total_deleted_chars']}"
        )

    report = {
        "baseline_type": "random_document_sentence_boundary_suffix_deletion",
        "matching_policy": (
            "Match E2 modified-document count exactly and match the per-document "
            "deleted-character distribution approximately at valid sentence boundaries. "
            "No watermark ground truth is used for selection."
        ),
        "input_document_count": int(summary["input_document_count"]),
        "target_modified_document_count": len(rows),
        "target_total_deleted_chars": sum(lengths),
        "target_deleted_char_rate": float(summary["deleted_char_rate"]),
        "target_deleted_char_lengths": lengths,
        "target_deletion_step_counts": steps,
        "deleted_char_distribution": {
            "min": min(lengths) if lengths else None,
            "q25": quantile(lengths, 0.25),
            "median": statistics.median(lengths) if lengths else None,
            "q75": quantile(lengths, 0.75),
            "max": max(lengths) if lengths else None,
            "mean": statistics.mean(lengths) if lengths else None,
        },
        "e2_summary_path": str(summary_path),
        "e2_summary_sha256": sha256_file(summary_path),
        "e2_deletion_log_path": str(log_path),
        "e2_deletion_log_sha256": sha256_file(log_path),
    }

    output = Path(args.output)
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("E3 BUDGET FREEZE PASS")


if __name__ == "__main__":
    main()

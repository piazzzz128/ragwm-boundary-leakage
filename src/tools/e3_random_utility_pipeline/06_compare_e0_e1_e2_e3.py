from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from e3_common import read_json, write_json_atomic


def load_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(row):
    return tuple(row[0])


def summarize_against_e0(e0_rows, rows):
    e0 = {key(row): row for row in e0_rows}
    current = {key(row): row for row in rows}
    if set(e0) != set(current):
        raise RuntimeError("Verification-unit mismatch")

    e0_wsn = sum(row[2][0] == 1 for row in e0.values())
    current_wsn = sum(row[2][0] == 1 for row in current.values())
    yes_to_no = sum(
        e0[unit][2][0] == 1 and current[unit][2][0] == 0
        for unit in e0
    )
    yes_to_yes = sum(
        e0[unit][2][0] == 1 and current[unit][2][0] == 1
        for unit in e0
    )
    no_to_yes = sum(
        e0[unit][2][0] == 0 and current[unit][2][0] == 1
        for unit in e0
    )
    unknown = sum(row[2][0] == 2 for row in current.values())
    return {
        "WSN": current_wsn,
        "absolute_WSN_drop_from_e0": e0_wsn - current_wsn,
        "relative_WSN_reduction_from_e0": (
            (e0_wsn - current_wsn) / e0_wsn if e0_wsn else None
        ),
        "paired_yes_to_no": yes_to_no,
        "paired_yes_to_yes": yes_to_yes,
        "paired_no_to_yes": no_to_yes,
        "paired_attack_success_rate": yes_to_no / e0_wsn if e0_wsn else None,
        "unknown_count": unknown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", required=True)
    parser.add_argument("--e1", required=True)
    parser.add_argument("--e2", required=True)
    parser.add_argument("--e3-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101,202,303,404,505])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e0_rows = load_rows(args.e0)
    e1_rows = load_rows(args.e1)
    e2_rows = load_rows(args.e2)

    e0_wsn = sum(row[2][0] == 1 for row in e0_rows)
    e1_summary = summarize_against_e0(e0_rows, e1_rows)
    e2_summary = summarize_against_e0(e0_rows, e2_rows)

    random_results = []
    for seed in args.seeds:
        verify_path = (
            Path(args.e3_root)
            / f"seed_{seed}/basepath/wm_generate/nfcorpus/"
              "gpt4o_mini_e2e/10/wmuint_verify.json"
        )
        summary_path = (
            Path(args.e3_root)
            / f"seed_{seed}/sanitization/random_sanitization_summary.json"
        )
        audit_path = (
            Path(args.e3_root)
            / f"seed_{seed}/metrics/posthoc_random_ground_truth_audit.json"
        )
        summary = summarize_against_e0(e0_rows, load_rows(verify_path))
        summary["seed"] = seed
        summary["budget"] = read_json(summary_path)
        summary["posthoc_ground_truth"] = read_json(audit_path)["summary"]
        random_results.append(summary)

    random_wsns = [row["WSN"] for row in random_results]
    random_asrs = [row["paired_attack_success_rate"] for row in random_results]
    random_mean = statistics.mean(random_wsns)
    random_std = statistics.stdev(random_wsns) if len(random_wsns) > 1 else 0.0
    e2_wsn = e2_summary["WSN"]

    oracle_gap = e0_wsn - e1_summary["WSN"]
    report = {
        "total": len(e0_rows),
        "e0_watermarked_WSN": e0_wsn,
        "e1_oracle": e1_summary,
        "e2_gainratio": e2_summary,
        "e2_oracle_gap_closure": (
            (e0_wsn - e2_wsn) / oracle_gap if oracle_gap else None
        ),
        "random_seed_count": len(random_results),
        "random_WSN_values": random_wsns,
        "random_WSN_mean": random_mean,
        "random_WSN_std": random_std,
        "random_WSN_min": min(random_wsns),
        "random_WSN_max": max(random_wsns),
        "random_paired_ASR_mean": statistics.mean(random_asrs),
        "gainratio_WSN_advantage_vs_random_mean": random_mean - e2_wsn,
        "gainratio_beats_random_seed_count": sum(e2_wsn < value for value in random_wsns),
        "gainratio_ties_random_seed_count": sum(e2_wsn == value for value in random_wsns),
        "gainratio_worse_than_random_seed_count": sum(e2_wsn > value for value in random_wsns),
        "random_results": random_results,
        "all_runs_valid": all(
            row["unknown_count"] == 0 and row["budget"]["budget_match_pass"]
            for row in random_results
        ),
    }

    write_json_atomic(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_runs_valid"]:
        raise RuntimeError("At least one random run is invalid")
    print("E0/E1/E2/E3 COMPARISON PASS")


if __name__ == "__main__":
    main()

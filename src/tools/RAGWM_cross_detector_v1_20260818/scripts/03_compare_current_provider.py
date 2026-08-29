#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cross_detector_common import load_json, write_json_atomic


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_map(path: Path):
    rows = load_json(path)
    return rows, {tuple(row[0]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", required=True, type=Path)
    parser.add_argument("--e1", required=True, type=Path)
    parser.add_argument("--attack", required=True, type=Path)
    parser.add_argument("--sanitization-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    e0_rows, e0 = load_map(args.e0)
    e1_rows, e1 = load_map(args.e1)
    attack_rows, attack = load_map(args.attack)
    if set(e0) != set(e1) or set(e0) != set(attack):
        raise RuntimeError("Verification watermark-unit sets differ")

    def count(rows, flag):
        return sum(row[2][0] == flag for row in rows)

    wsn0 = count(e0_rows, 1)
    wsn1 = count(e1_rows, 1)
    wsna = count(attack_rows, 1)
    if (len(e0_rows), wsn0, count(e0_rows, 2)) != (30, 25, 0):
        raise RuntimeError("Current-provider E0 is not the frozen 25/30 audit")
    if (len(e1_rows), wsn1, count(e1_rows, 2)) != (30, 4, 0):
        raise RuntimeError("Current-provider E1 is not the frozen 4/30 audit")

    broken = retained = false_gain = b = c = 0
    details = []
    for unit in sorted(e0):
        f0 = e0[unit][2][0]
        f1 = e1[unit][2][0]
        fa = attack[unit][2][0]
        if f0 == 1 and fa == 0:
            broken += 1
            b += 1
        if f0 == 1 and fa == 1:
            retained += 1
        if f0 == 0 and fa == 1:
            false_gain += 1
            c += 1
        details.append(
            {
                "watermark_unit": list(unit),
                "e0_flag": f0,
                "e1_flag": f1,
                "attack_flag": fa,
            }
        )

    sanitation = load_json(args.sanitization_summary)
    oracle_denominator = wsn0 - wsn1
    unknown = count(attack_rows, 2)
    summary = {
        "experiment_role": "cross_detector_lm_robustness_aligned_v1",
        "detector_lm": "DeepSeek-LLM-7B-base",
        "verifier_reference": "current_provider_verifier_audit_v1_20260818",
        "verifier_model": "gpt-4o-mini",
        "total": len(attack_rows),
        "e0_current_provider_WSN": wsn0,
        "e1_current_provider_oracle_WSN": wsn1,
        "deepseek_gainratio_WSN": wsna,
        "absolute_WSN_drop": wsn0 - wsna,
        "relative_WSN_reduction": (
            (wsn0 - wsna) / wsn0 if wsn0 else None
        ),
        "paired_baseline_positive_count": wsn0,
        "paired_broken_count": broken,
        "paired_retained_count": retained,
        "false_gain_count": false_gain,
        "paired_attack_success_rate": broken / wsn0 if wsn0 else None,
        "oracle_gap_closure": (
            (wsn0 - wsna) / oracle_denominator
            if oracle_denominator > 0
            else None
        ),
        "mcnemar_yes_to_no": b,
        "mcnemar_no_to_yes": c,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(b, c),
        "attack_unknown_count": unknown,
        "modified_document_count": sanitation.get("modified_document_count"),
        "total_deleted_chars": sanitation.get("total_deleted_chars"),
        "run_valid": len(attack_rows) == 30 and unknown == 0,
        "warning": (
            "Cross-detector extension under the current-provider verifier "
            "audit. Do not replace sealed strict-v1 E0=26, E1=5, E2=17."
        ),
    }
    write_json_atomic(
        args.output, {"summary": summary, "details": details}
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["run_valid"]:
        raise RuntimeError("DEEPSEEK VERIFIER COMPARISON INVALID")
    print("DEEPSEEK CURRENT-PROVIDER COMPARISON PASS")


if __name__ == "__main__":
    main()


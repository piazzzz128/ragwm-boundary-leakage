from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from e2_common_aligned import write_json_atomic


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_map(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows, {tuple(row[0]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", required=True)
    parser.add_argument("--e1", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e0_rows, e0 = load_map(Path(args.e0))
    e1_rows, e1 = load_map(Path(args.e1))
    ea_rows, ea = load_map(Path(args.aligned))

    if set(e0) != set(e1) or set(e0) != set(ea):
        raise RuntimeError(
            "Verification watermark-unit sets differ across conditions"
        )

    def count_yes(rows):
        return sum(row[2][0] == 1 for row in rows)

    def count_unknown(rows):
        return sum(row[2][0] == 2 for row in rows)

    wsn0 = count_yes(e0_rows)
    wsn1 = count_yes(e1_rows)
    wsna = count_yes(ea_rows)

    broken = retained = false_gain = b = c = 0
    details = []

    for unit in sorted(e0):
        f0 = e0[unit][2][0]
        f1 = e1[unit][2][0]
        fa = ea[unit][2][0]

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
                "aligned_flag": fa,
                "e0_answer": e0[unit][1][3],
                "e1_answer": e1[unit][1][3],
                "aligned_answer": ea[unit][1][3],
            }
        )

    oracle_denominator = wsn0 - wsn1
    summary = {
        "experiment_role": "aligned_end_to_end_sensitivity",
        "total": len(e0_rows),
        "e0_watermarked_WSN": wsn0,
        "e1_oracle_WSN": wsn1,
        "aligned_gainratio_WSN": wsna,
        "aligned_absolute_WSN_drop": wsn0 - wsna,
        "aligned_relative_WSN_reduction": (
            (wsn0 - wsna) / wsn0 if wsn0 else None
        ),
        "aligned_paired_baseline_positive_count": wsn0,
        "aligned_paired_broken_count": broken,
        "aligned_paired_retained_count": retained,
        "aligned_false_gain_count": false_gain,
        "aligned_paired_attack_success_rate": (
            broken / wsn0 if wsn0 else None
        ),
        "oracle_gap_closure": (
            (wsn0 - wsna) / oracle_denominator
            if oracle_denominator > 0
            else None
        ),
        "mcnemar_yes_to_no": b,
        "mcnemar_no_to_yes": c,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(b, c),
        "aligned_unknown_count": count_unknown(ea_rows),
        "run_valid": (
            len(ea_rows) == 30 and count_unknown(ea_rows) == 0
        ),
        "warning": (
            "Sensitivity result only; do not overwrite the sealed "
            "strict-v1 E2 main result."
        ),
    }

    write_json_atomic(
        Path(args.output),
        {"summary": summary, "details": details},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("E0/E1/ALIGNED COMPARISON PASS")


if __name__ == "__main__":
    main()

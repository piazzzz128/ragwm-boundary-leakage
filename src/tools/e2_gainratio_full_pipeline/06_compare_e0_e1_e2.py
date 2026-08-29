from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from e2_common import write_json_atomic


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
    parser.add_argument("--e2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    e0_rows, e0 = load_map(Path(args.e0))
    e1_rows, e1 = load_map(Path(args.e1))
    e2_rows, e2 = load_map(Path(args.e2))
    if set(e0) != set(e1) or set(e0) != set(e2):
        raise RuntimeError("Verification watermark-unit sets differ across E0/E1/E2")

    def count_yes(rows):
        return sum(row[2][0] == 1 for row in rows)

    def count_unknown(rows):
        return sum(row[2][0] == 2 for row in rows)

    wsn0 = count_yes(e0_rows)
    wsn1 = count_yes(e1_rows)
    wsn2 = count_yes(e2_rows)
    broken = retained = false_gain = b = c = 0
    details = []
    for unit in sorted(e0):
        f0, f1, f2 = e0[unit][2][0], e1[unit][2][0], e2[unit][2][0]
        if f0 == 1 and f2 == 0:
            broken += 1
            b += 1
        if f0 == 1 and f2 == 1:
            retained += 1
        if f0 == 0 and f2 == 1:
            false_gain += 1
            c += 1
        details.append(
            {
                "watermark_unit": list(unit),
                "e0_flag": f0,
                "e1_flag": f1,
                "e2_flag": f2,
                "e0_answer": e0[unit][1][3],
                "e1_answer": e1[unit][1][3],
                "e2_answer": e2[unit][1][3],
            }
        )

    oracle_denominator = wsn0 - wsn1
    summary = {
        "total": len(e0_rows),
        "e0_watermarked_WSN": wsn0,
        "e1_oracle_WSN": wsn1,
        "e2_gainratio_WSN": wsn2,
        "e2_absolute_WSN_drop": wsn0 - wsn2,
        "e2_relative_WSN_reduction": (wsn0 - wsn2) / wsn0 if wsn0 else None,
        "e2_paired_baseline_positive_count": wsn0,
        "e2_paired_broken_count": broken,
        "e2_paired_retained_count": retained,
        "e2_false_gain_count": false_gain,
        "e2_paired_attack_success_rate": broken / wsn0 if wsn0 else None,
        "oracle_gap_closure": (
            (wsn0 - wsn2) / oracle_denominator if oracle_denominator > 0 else None
        ),
        "mcnemar_yes_to_no": b,
        "mcnemar_no_to_yes": c,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(b, c),
        "e2_unknown_count": count_unknown(e2_rows),
        "run_valid": len(e2_rows) == 30 and count_unknown(e2_rows) == 0,
    }
    write_json_atomic(Path(args.output), {"summary": summary, "details": details})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("E0/E1/E2 COMPARISON PASS")


if __name__ == "__main__":
    main()

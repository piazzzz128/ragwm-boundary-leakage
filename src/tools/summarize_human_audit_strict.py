import csv
from collections import defaultdict
from pathlib import Path

p = Path("/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples_filled.csv")
rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))

fields = [
    "entity_preserved",
    "relation_preserved",
    "no_new_fact",
    "meaning_preserved",
    "fluency_improved",
    "valid_attack",
]

groups = defaultdict(list)
for r in rows:
    groups[(r["attacker"], r["dataset"])].append(r)

print("| Attacker | Dataset | N rows | N fully labeled | Entity preserved | Relation preserved | No new fact | Meaning preserved | Fluency improved | Valid attack |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

overall_full = 0
overall = {f: [] for f in fields}

for (attacker, dataset), rs in sorted(groups.items()):
    fully = []
    for r in rs:
        if all(str(r.get(f, "")).strip() in {"0", "1"} for f in fields):
            fully.append(r)

    vals = []
    for f in fields:
        xs = [int(str(r[f]).strip()) for r in fully]
        vals.append(sum(xs) / len(xs) if xs else float("nan"))
        overall[f].extend(xs)

    overall_full += len(fully)

    print(
        f"| {attacker} | {dataset} | {len(rs)} | {len(fully)} | "
        + " | ".join(f"{v:.2%}" if v == v else "NA" for v in vals)
        + " |"
    )

print("\nOverall rows =", len(rows))
print("Overall fully labeled =", overall_full)

for f in fields:
    xs = overall[f]
    if xs:
        print(f, f"{sum(xs) / len(xs):.2%}", f"({sum(xs)}/{len(xs)})")
    else:
        print(f, "NA")

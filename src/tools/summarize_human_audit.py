import csv
from collections import defaultdict

p = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples_filled.csv"
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

print("| Attacker | Dataset | N | Entity preserved | Relation preserved | No new fact | Meaning preserved | Fluency improved | Valid attack |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

for (attacker, dataset), rs in sorted(groups.items()):
    vals = []
    for f in fields:
        xs = [int(str(r[f]).strip()) for r in rs if str(r[f]).strip() in {"0", "1"}]
        vals.append(sum(xs) / len(xs) if xs else float("nan"))
    print(
        f"| {attacker} | {dataset} | {len(rs)} | "
        + " | ".join(f"{v:.2%}" for v in vals)
        + " |"
    )

print("\nOverall N =", len(rows))
for f in fields:
    xs = [int(str(r[f]).strip()) for r in rows if str(r[f]).strip() in {"0", "1"}]
    print(f, f"{sum(xs) / len(xs):.2%}", f"({sum(xs)}/{len(xs)})")

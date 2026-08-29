import json
import random
from pathlib import Path
from collections import Counter, defaultdict

SEED = 2026
random.seed(SEED)

IN_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v1.json")
IN_AUDIT = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v1_audit.json")

OUT_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2.json")
OUT_AUDIT = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2_audit.json")

def main():
    data = json.load(open(IN_PATH, "r", encoding="utf-8"))
    old_audit = json.load(open(IN_AUDIT, "r", encoding="utf-8"))

    print("input records:", len(data))
    print("input label counts:", Counter(int(x["label"]) for x in data))

    groups = defaultdict(list)
    for r in data:
        groups[r["full_text"]].append(r)

    dup_groups = {k:v for k,v in groups.items() if len(v) > 1}
    print("duplicate full_text groups:", len(dup_groups))
    for i, (k, v) in enumerate(list(dup_groups.items())[:5]):
        print("DUP SAMPLE", i, "count=", len(v), "labels=", [x["label"] for x in v], "ids=", [x["id"] for x in v])

    # Deterministic de-duplication: keep one record per full_text.
    # Prefer inject if cross-label duplicate, because inject samples are harder to recover;
    # later we rebalance labels to remove class bias.
    deduped = []
    removed = []
    for full_text, rows in groups.items():
        if len(rows) == 1:
            deduped.append(rows[0])
            continue

        rows_sorted = sorted(rows, key=lambda r: (0 if int(r["label"]) == 1 else 1, r.get("id", "")))
        keep = rows_sorted[0]
        drop = rows_sorted[1:]
        deduped.append(keep)
        removed.extend(drop)

    print("after dedup records:", len(deduped))
    print("removed records:", len(removed))
    print("after dedup label counts:", Counter(int(x["label"]) for x in deduped))

    by_label = defaultdict(list)
    for r in deduped:
        by_label[int(r["label"])].append(r)

    n0 = len(by_label[0])
    n1 = len(by_label[1])
    n = min(n0, n1)

    random.shuffle(by_label[0])
    random.shuffle(by_label[1])

    balanced = by_label[0][:n] + by_label[1][:n]
    random.shuffle(balanced)

    # Reassign ids but preserve source identifiers.
    for i, r in enumerate(balanced):
        label = int(r["label"])
        prefix = "clean" if label == 0 else "inject"
        r["id"] = f"nq_sampled_strict_v2_{prefix}_{i}"
        r["construction"] = "sampled_strict_v2"

    full_counter = Counter(r["full_text"] for r in balanced)

    target_in_context = sum(
        1 for r in balanced
        if r.get("target") and r["target"] in r.get("context", "")
    )

    all_wt = [r.get("watermark_text") for r in balanced if r.get("watermark_text")]
    context_contains_any_wt = sum(
        1 for r in balanced
        if int(r["label"]) == 1 and any(wt in r.get("context", "") for wt in all_wt)
    )
    clean_target_contains_any_wt = sum(
        1 for r in balanced
        if int(r["label"]) == 0 and any(wt in r.get("target", "") for wt in all_wt)
    )
    duplicate_full_text = sum(1 for _, v in full_counter.items() if v > 1)

    audit = dict(old_audit)
    audit.update({
        "version": "sampled_strict_v2",
        "source_version": "sampled_strict_v1",
        "dedup_seed": SEED,
        "input_records": len(data),
        "input_label_counts": dict(Counter(int(x["label"]) for x in data)),
        "duplicate_full_text_groups_before": len(dup_groups),
        "removed_duplicate_records": len(removed),
        "records_after_dedup_before_balance": len(deduped),
        "label_counts_after_dedup_before_balance": dict(Counter(int(x["label"]) for x in deduped)),
        "inject_records": sum(1 for r in balanced if int(r["label"]) == 1),
        "clean_records": sum(1 for r in balanced if int(r["label"]) == 0),
        "total_records": len(balanced),
        "label_counts": dict(Counter(int(r["label"]) for r in balanced)),
        "target_in_context": target_in_context,
        "context_contains_any_wt": context_contains_any_wt,
        "clean_target_contains_any_wt": clean_target_contains_any_wt,
        "duplicate_full_text": duplicate_full_text,
        "note": "NQ sampled strict v2 is generated from v1 by deterministic full_text de-duplication and label rebalancing. It should be used for metrics."
    })

    json.dump(balanced, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audit, open(OUT_AUDIT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved:", OUT_PATH)
    print("saved:", OUT_AUDIT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

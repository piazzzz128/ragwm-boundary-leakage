import json
import glob
import os
import random
from pathlib import Path
from collections import defaultdict, Counter

SEED = 2026
random.seed(SEED)

IN_GLOB = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/boundary_*_attack_a2_deepseek-chat_v2.json"
OUT_DIR = Path("/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = OUT_DIR / "adaptive_attack_deepseek_chat_a2_v2_clean_summary.json"
SUMMARY_MD = OUT_DIR / "adaptive_attack_deepseek_chat_a2_v2_clean_summary.md"

def parse_name(path):
    base = os.path.basename(path)

    if base.startswith("boundary_nfcorpus"):
        dataset = "NFCorpus"
    elif base.startswith("boundary_trec"):
        dataset = "TREC-COVID"
    elif base.startswith("boundary_nq"):
        dataset = "Natural Questions"
    else:
        dataset = "unknown"

    return dataset

def clean_one(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    dataset = parse_name(path)

    before_n = len(data)
    before_labels = Counter(int(r["label"]) for r in data)

    groups = defaultdict(list)
    for r in data:
        groups[r.get("full_text", "")].append(r)

    duplicate_groups = {k:v for k,v in groups.items() if len(v) > 1}

    deduped = []
    removed = []

    for full_text, rows in groups.items():
        if len(rows) == 1:
            deduped.append(rows[0])
            continue

        # If duplicate crosses labels, prefer keeping inject; rebalance later.
        rows_sorted = sorted(rows, key=lambda r: (0 if int(r["label"]) == 1 else 1, r.get("id", "")))
        deduped.append(rows_sorted[0])
        removed.extend(rows_sorted[1:])

    by_label = defaultdict(list)
    for r in deduped:
        by_label[int(r["label"])].append(r)

    n = min(len(by_label[0]), len(by_label[1]))

    random.shuffle(by_label[0])
    random.shuffle(by_label[1])

    balanced = by_label[0][:n] + by_label[1][:n]
    random.shuffle(balanced)

    for i, r in enumerate(balanced):
        label = int(r["label"])
        prefix = "clean" if label == 0 else "inject"
        r["id"] = f"{dataset.replace(' ', '_').lower()}_deepseek_a2_cleaned_{prefix}_{i}"
        r["adaptive_attack_cleaned"] = True
        r["attack_model"] = "deepseek-chat"

    full_counter = Counter(r.get("full_text", "") for r in balanced)
    duplicate_full_text = sum(1 for _, c in full_counter.items() if c > 1)
    target_in_context = sum(
        1 for r in balanced
        if r.get("target") and r["target"] in (r.get("context") or "")
    )

    out_path = OUT_DIR / (Path(path).stem + "_clean.json")
    audit_path = OUT_DIR / (Path(path).stem + "_clean_audit.json")

    audit = {
        "dataset": dataset,
        "attack": "a2",
        "attack_model": "deepseek-chat",
        "prompt_version": "v2",
        "source_path": path,
        "output_path": str(out_path),
        "before_n": before_n,
        "before_label_counts": dict(before_labels),
        "duplicate_full_text_groups_before": len(duplicate_groups),
        "removed_duplicate_records": len(removed),
        "after_dedup_before_balance": len(deduped),
        "after_dedup_label_counts": dict(Counter(int(r["label"]) for r in deduped)),
        "after_balance_n": len(balanced),
        "after_balance_label_counts": dict(Counter(int(r["label"]) for r in balanced)),
        "target_in_context": target_in_context,
        "duplicate_full_text": duplicate_full_text,
    }

    json.dump(balanced, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audit, open(audit_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    return audit

def main():
    paths = sorted(glob.glob(IN_GLOB))
    paths = [p for p in paths if not p.endswith("_audit.json")]

    print("input files:", len(paths))
    for p in paths:
        print(" -", p)

    if len(paths) == 0:
        raise RuntimeError("No deepseek-chat A2 attack files found.")

    audits = []
    for p in paths:
        audit = clean_one(p)
        audits.append(audit)
        print("\n", os.path.basename(p))
        print(json.dumps(audit, ensure_ascii=False, indent=2))

    json.dump(audits, open(SUMMARY_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# DeepSeek-chat A2 Cleaned Boundary Summary\n\n")
        f.write("| Dataset | Before N | Dup Groups | Removed | Final N | Labels | target_in_context | duplicate_full_text |\n")
        f.write("|---|---:|---:|---:|---:|---|---:|---:|\n")
        for a in audits:
            f.write(
                f'| {a["dataset"]} | {a["before_n"]} | '
                f'{a["duplicate_full_text_groups_before"]} | {a["removed_duplicate_records"]} | '
                f'{a["after_balance_n"]} | {a["after_balance_label_counts"]} | '
                f'{a["target_in_context"]} | {a["duplicate_full_text"]} |\n'
            )

    print("\nsaved:", SUMMARY_JSON)
    print("saved:", SUMMARY_MD)

if __name__ == "__main__":
    main()

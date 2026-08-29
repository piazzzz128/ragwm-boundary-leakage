import csv
import argparse
from pathlib import Path
import textwrap

DEFAULT_PATH = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples_filled.csv"

FIELDS = [
    "entity_preserved",
    "relation_preserved",
    "no_new_fact",
    "meaning_preserved",
    "fluency_improved",
    "valid_attack",
]

def save_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def is_done(r):
    return all(str(r.get(f, "")).strip() in {"0", "1"} for f in FIELDS)

def short(x, width=900):
    x = str(x).replace("\n", " ").strip()
    return textwrap.fill(x, width=120)[:width]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()

    path = Path(args.path)
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())

    print("Loaded:", path)
    print("N =", len(rows))
    print("Input format:")
    print("  11111 = entity, relation, no_new_fact, meaning, fluency")
    print("  valid_attack is computed automatically from first four fields")
    print("  Example: 11111 good rewrite")
    print("  Example: 01010 entity narrowed")
    print("Commands: q=quit, s=skip\n")

    for idx in range(args.start - 1, len(rows)):
        r = rows[idx]
        if is_done(r):
            continue

        print("\n" + "=" * 100)
        print(f"Sample {idx+1}/{len(rows)}")
        print("dataset:", r["dataset"])
        print("attacker:", r["attacker"])
        print("sample_id:", r["sample_id"])
        print("group_id:", r.get("group_id", ""))

        print("\n[Context]")
        print(short(r["context"], 1000))

        print("\n[Original target]")
        print(short(r["original_target"], 500))

        print("\n[Rewritten target]")
        print(short(r["rewritten_target"], 500))

        print("\nFields:")
        print("1 entity_preserved | 2 relation_preserved | 3 no_new_fact | 4 meaning_preserved | 5 fluency_improved")
        ans = input("Enter 5 bits + optional note: ").strip()

        if ans.lower() == "q":
            save_csv(path, rows, fieldnames)
            print("Saved and quit.")
            return

        if ans.lower() == "s" or not ans:
            print("Skipped.")
            continue

        parts = ans.split(maxsplit=1)
        bits = parts[0].strip()
        note = parts[1].strip() if len(parts) > 1 else ""

        if len(bits) != 5 or any(c not in "01" for c in bits):
            print("Invalid input. Use exactly 5 bits, e.g. 11111 or 01010.")
            continue

        e, rel, nf, mean, flu = bits
        valid = "1" if (e == "1" and rel == "1" and nf == "1" and mean == "1") else "0"

        r["entity_preserved"] = e
        r["relation_preserved"] = rel
        r["no_new_fact"] = nf
        r["meaning_preserved"] = mean
        r["fluency_improved"] = flu
        r["valid_attack"] = valid
        r["notes"] = note

        save_csv(path, rows, fieldnames)
        print(f"Saved. valid_attack={valid}")

    save_csv(path, rows, fieldnames)
    print("\nAll done.")

if __name__ == "__main__":
    main()

import json, csv, os, glob
from pathlib import Path

ROOT = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack"

def load_any(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        for k in ["records", "scores", "data", "results", "items", "samples"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    return []

def is_text_key(k):
    k = k.lower()
    return (
        "context" in k or
        "target" in k or
        "rewrite" in k or
        "rewritten" in k or
        "paraphrase" in k or
        "original" in k or
        "watermark" in k or
        "full_text" in k
    )

files = []
for ext in ["*.json", "*.jsonl", "*.csv"]:
    files.extend(glob.glob(os.path.join(ROOT, "**", ext), recursive=True))

for p in sorted(files):
    try:
        rows = load_any(p)
    except Exception as e:
        print("\nERROR:", p, e)
        continue

    if not rows:
        continue

    keys = sorted(set().union(*[set(r.keys()) for r in rows if isinstance(r, dict)]))
    text_keys = [k for k in keys if is_text_key(k)]
    id_keys = [k for k in keys if k.lower() in {"id", "record_id", "sample_id", "source_id", "original_id"} or "id" in k.lower()]

    if text_keys:
        print("\nFILE:", p)
        print("N:", len(rows))
        print("ID_KEYS:", id_keys)
        print("TEXT_KEYS:", text_keys)
        r = rows[0]
        print("FIRST_KEYS:", list(r.keys()))
        for k in text_keys[:20]:
            if k in r:
                print(f"{k} =", str(r[k])[:250].replace("\n", " "))

import csv, json, os, random, glob
from pathlib import Path

random.seed(20260609)

ROOT = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack"
OUT = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples.csv"

score_jobs = [
    {
        "attacker": "qwen-plus A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/adaptive_attack_qwen_plus_v2_clean_matched_aligned_scores.json",
        "attack": "a2",
    },
    {
        "attacker": "deepseek-chat A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2/adaptive_attack_deepseek_chat_a2_v2_clean_matched_aligned_scores.json",
        "attack": None,
    },
]

datasets = ["NFCorpus", "TREC-COVID", "Natural Questions"]

def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for k in ["records", "scores", "data", "results", "items", "samples"]:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    raise RuntimeError(f"No record list in {path}")

def load_any(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            return load_json_records(path)
        except Exception:
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

def get_first(r, keys):
    for k in keys:
        if k in r and r[k] is not None and str(r[k]).strip():
            return str(r[k])
    return ""

def ids_for(r):
    vals = []
    for k in ["record_id", "id", "sample_id", "source_id", "original_id"]:
        if k in r and r[k] is not None:
            vals.append(str(r[k]))
    return vals

def build_text_index():
    index = {}
    files = []
    for ext in ["*.json", "*.jsonl", "*.csv"]:
        files.extend(glob.glob(os.path.join(ROOT, "**", ext), recursive=True))

    for p in files:
        # scores normally do not contain text; skip CI outputs and score-only files if desired
        if "/ci_" in p or p.endswith("_scores.json"):
            continue
        rows = load_any(p)
        for r in rows:
            if not isinstance(r, dict):
                continue
            has_text = any(
                k in r for k in [
                    "context", "context_text",
                    "original_target", "target_original", "source_target", "clean_target",
                    "rewritten_target", "attacked_target", "target_rewritten", "paraphrased_target",
                    "target", "watermark_text"
                ]
            )
            if not has_text:
                continue
            for rid in ids_for(r):
                index[rid] = r
    return index

text_index = build_text_index()
print("TEXT_INDEX_SIZE =", len(text_index))

rows = []
missing = []

for job in score_jobs:
    score_records = load_json_records(job["path"])

    for ds in datasets:
        subset = []
        for r in score_records:
            if str(r.get("dataset")) != ds:
                continue
            if str(r.get("variant")) != "attacked":
                continue
            if int(r.get("label")) != 1:
                continue
            if job["attack"] is not None and str(r.get("attack")) != job["attack"]:
                continue
            subset.append(r)

        if len(subset) < 20:
            raise RuntimeError(f"Not enough score records for {job['attacker']} {ds}: {len(subset)}")

        sampled = random.sample(subset, 20)

        for r in sampled:
            rid = get_first(r, ["record_id", "id", "sample_id"])
            text_r = text_index.get(rid, {})

            context = get_first(text_r, ["context", "context_text", "source_context"])
            original_target = get_first(text_r, [
                "original_target", "target_original", "source_target",
                "original_watermark_text", "watermark_text_original",
                "watermark_text"
            ])
            rewritten_target = get_first(text_r, [
                "rewritten_target", "attacked_target", "target_rewritten",
                "paraphrased_target", "rewrite", "rewritten_text", "target"
            ])

            if not context or not original_target or not rewritten_target:
                missing.append((job["attacker"], ds, rid, list(text_r.keys()) if text_r else []))

            rows.append({
                "dataset": ds,
                "attacker": job["attacker"],
                "sample_id": rid,
                "group_id": get_first(r, ["group_id"]),
                "context": context,
                "original_target": original_target,
                "rewritten_target": rewritten_target,
                "entity_preserved": "",
                "relation_preserved": "",
                "no_new_fact": "",
                "meaning_preserved": "",
                "fluency_improved": "",
                "valid_attack": "",
                "notes": "",
            })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("WROTE", OUT)
print("N", len(rows))
print("MISSING_TEXT_ROWS", len(missing))
for item in missing[:20]:
    print("MISSING", item)

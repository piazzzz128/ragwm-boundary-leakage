import csv, json, os, random

random.seed(20260609)

jobs = [
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
out_path = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples.csv"

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for k in ["records", "scores", "data", "results", "items"]:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    raise RuntimeError(f"No records in {path}")

def get_first(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return ""

rows = []
for job in jobs:
    records = load_records(job["path"])

    for ds in datasets:
        subset = []
        for r in records:
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
            raise RuntimeError(f"Not enough records for {job['attacker']} {ds}: {len(subset)}")

        sampled = random.sample(subset, 20)
        for r in sampled:
            rows.append({
                "dataset": ds,
                "attacker": job["attacker"],
                "sample_id": get_first(r, ["record_id", "id", "sample_id"]),
                "group_id": get_first(r, ["group_id"]),
                "context": get_first(r, ["context", "context_text"]),
                "original_target": get_first(r, ["original_target", "target_original", "source_target", "watermark_text", "target"]),
                "rewritten_target": get_first(r, ["rewritten_target", "attacked_target", "target_rewritten", "paraphrased_target", "target"]),
                "entity_preserved": "",
                "relation_preserved": "",
                "no_new_fact": "",
                "meaning_preserved": "",
                "fluency_improved": "",
                "valid_attack": "",
                "notes": "",
            })

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("WROTE", out_path)
print("N", len(rows))

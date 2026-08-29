import csv, json, os, random

random.seed(20260609)

OUT = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples.csv"

jobs = [
    {
        "dataset": "NFCorpus",
        "attacker": "qwen-plus A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/boundary_nfcorpus_strict_v1_attack_a2_qwen-plus_v2_clean.json",
    },
    {
        "dataset": "TREC-COVID",
        "attacker": "qwen-plus A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/boundary_trec_strict_v3_attack_a2_qwen-plus_v2_clean.json",
    },
    {
        "dataset": "Natural Questions",
        "attacker": "qwen-plus A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/boundary_nq_sampled_strict_v2_attack_a2_qwen-plus_v2_clean.json",
    },
    {
        "dataset": "NFCorpus",
        "attacker": "deepseek-chat A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2/boundary_nfcorpus_strict_v1_attack_a2_deepseek-chat_v2_clean.json",
    },
    {
        "dataset": "TREC-COVID",
        "attacker": "deepseek-chat A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2/boundary_trec_strict_v3_attack_a2_deepseek-chat_v2_clean.json",
    },
    {
        "dataset": "Natural Questions",
        "attacker": "deepseek-chat A2",
        "path": "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2/boundary_nq_sampled_strict_v2_attack_a2_deepseek-chat_v2_clean.json",
    },
]

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for k in ["records", "data", "items", "samples", "results"]:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    raise RuntimeError(f"No record list in {path}")

def get_first(r, keys):
    for k in keys:
        if k in r and r[k] is not None and str(r[k]).strip():
            return str(r[k])
    return ""

rows = []

for job in jobs:
    records = load_records(job["path"])

    inject_records = []
    for r in records:
        if int(r.get("label", -1)) != 1:
            continue

        context = get_first(r, ["context", "context_text"])
        original_target = get_first(r, ["original_target", "watermark_text"])
        rewritten_target = get_first(r, ["target_text", "target"])

        if not context or not original_target or not rewritten_target:
            continue

        inject_records.append(r)

    if len(inject_records) < 20:
        raise RuntimeError(f"Not enough valid inject records: {job['attacker']} {job['dataset']} = {len(inject_records)}")

    sampled = random.sample(inject_records, 20)

    for r in sampled:
        rows.append({
            "dataset": job["dataset"],
            "attacker": job["attacker"],
            "sample_id": get_first(r, ["id", "record_id", "sample_id"]),
            "group_id": get_first(r, ["group_id"]),
            "watermark_tuple_or_unit": get_first(r, ["tuple", "wmunit", "metadata"]),
            "context": get_first(r, ["context", "context_text"]),
            "original_target": get_first(r, ["original_target", "watermark_text"]),
            "rewritten_target": get_first(r, ["target_text", "target"]),
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

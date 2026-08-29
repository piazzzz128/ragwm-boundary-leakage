import os
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = os.environ.get("RAGWM_API_BASE", "https://api.whatai.cc/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("RAGWM_API_KEY")
MODEL = os.environ.get("RAGWM_ATTACK_MODEL", "qwen-plus")

OUT_DIR = Path("/root/autodl-tmp/ragwm_storage/output/adaptive_attack")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "nfcorpus": {
        "name": "NFCorpus",
        "construction": "strict_v1",
        "boundary": "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nfcorpus_strict_v1.json",
    },
    "trec": {
        "name": "TREC-COVID",
        "construction": "strict_v3",
        "boundary": "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v3.json",
    },
    "nq": {
        "name": "Natural Questions",
        "construction": "sampled_strict_v2",
        "boundary": "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2.json",
    },
}

ATTACKS = {
    "a1": {
        "name": "target_only_paraphrase",
        "system": "You rewrite inserted watermark sentences. Preserve the original meaning and factual relation. Output only one rewritten English sentence.",
        "template": """Rewrite the following sentence into a more natural and fluent English sentence. Preserve its meaning and factual relation. Do not add explanations.

Sentence:
{target}

Output only the rewritten sentence.""",
    },
    "a2": {
        "name": "context_aware_fluency",
        "system": "You rewrite inserted sentences to make them more coherent with the preceding context while preserving the original factual relation. Output only one rewritten English sentence.",
        "template": """Given the preceding context and an inserted sentence, rewrite the inserted sentence so that it reads more logically and fluently after the context. Preserve the original factual relation as much as possible. Do not copy the context. Output only one rewritten English sentence.

Context:
{context}

Inserted sentence:
{target}

Output only the rewritten sentence.""",
    },
    "a3": {
        "name": "strong_adaptive_rewrite",
        "system": "You are an adaptive attacker rewriting inserted watermark sentences to reduce boundary inconsistency. Preserve the watermark relation as much as possible. Output only one rewritten English sentence.",
        "template": """An anomaly detector may flag the inserted sentence if it is not coherent with the preceding context. Rewrite the inserted sentence to maximize local coherence with the context, while preserving the original factual relation as much as possible. Keep it as a single English sentence. Do not add explanations.

Context:
{context}

Inserted sentence:
{target}

Output only the rewritten sentence.""",
    },
}

def norm(s):
    return " ".join((s or "").strip().split())

def cache_key(dataset, attack, model, record_id, target):
    raw = json.dumps({
        "dataset": dataset,
        "attack": attack,
        "model": model,
        "record_id": record_id,
        "target": target,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def call_chat(system, user, retries=5):
    if not API_KEY:
        raise RuntimeError("Missing API key. Please source /root/autodl-tmp/ragwm_storage/.ragwm_api_env first.")

    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 128,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode("utf-8", errors="replace")
                obj = json.loads(body)
                text = obj["choices"][0]["message"]["content"]
                return clean_output(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            wait = min(90, 10 * attempt)
            print(f"[HTTPError] attempt={attempt}/{retries} code={e.code} body={body[:300]} sleep={wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            wait = min(90, 10 * attempt)
            print(f"[ERROR] attempt={attempt}/{retries} err={repr(e)} sleep={wait}s", flush=True)
            time.sleep(wait)

    raise RuntimeError("API call failed after retries")

def clean_output(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    text = text.strip().strip('"').strip("'").strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) > 1:
        # choose first likely sentence, avoiding explanations
        for line in lines:
            if not line.lower().startswith(("here", "rewrite", "output")):
                text = line
                break
        else:
            text = lines[0]
    return norm(text)

def build_attack_boundary(dataset_key, attack_key, limit=None):
    ds = DATASETS[dataset_key]
    attack = ATTACKS[attack_key]

    boundary_path = Path(ds["boundary"])
    data = json.load(open(boundary_path, "r", encoding="utf-8"))

    cache_path = OUT_DIR / f"rewrite_cache_{MODEL}_{dataset_key}_{attack_key}.json"
    out_path = OUT_DIR / f"boundary_{dataset_key}_{ds['construction']}_attack_{attack_key}_{MODEL}.json"
    audit_path = OUT_DIR / f"boundary_{dataset_key}_{ds['construction']}_attack_{attack_key}_{MODEL}_audit.json"

    if cache_path.exists():
        cache = json.load(open(cache_path, "r", encoding="utf-8"))
    else:
        cache = {}

    inject_indices = [i for i, r in enumerate(data) if int(r["label"]) == 1]
    if limit is not None:
        inject_indices = inject_indices[:limit]

    new_data = []
    rewritten_count = 0
    unchanged_count = 0
    failed = []

    for idx, r in enumerate(data):
        rr = dict(r)

        if int(r["label"]) == 1 and idx in inject_indices:
            context = r.get("context") or r.get("context_text") or ""
            target = r.get("target") or r.get("target_text") or ""
            rid = r.get("id", str(idx))
            key = cache_key(dataset_key, attack_key, MODEL, rid, target)

            if key in cache:
                rewritten = cache[key]["rewritten_target"]
            else:
                prompt = attack["template"].format(context=context, target=target)
                try:
                    rewritten = call_chat(attack["system"], prompt)
                    cache[key] = {
                        "record_id": rid,
                        "original_target": target,
                        "rewritten_target": rewritten,
                        "attack": attack_key,
                        "attack_name": attack["name"],
                        "model": MODEL,
                    }
                    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                except Exception as e:
                    failed.append({"index": idx, "id": rid, "reason": repr(e)})
                    rewritten = target

            rewritten = norm(rewritten)
            if not rewritten or rewritten == norm(target):
                unchanged_count += 1
            else:
                rewritten_count += 1

            rr["original_target"] = target
            rr["target"] = rewritten
            rr["target_text"] = rewritten
            rr["adaptive_attack"] = attack_key
            rr["adaptive_attack_name"] = attack["name"]
            rr["adaptive_attack_model"] = MODEL
            rr["full_text"] = norm((rr.get("context") or "") + " " + rewritten)
        else:
            rr["adaptive_attack"] = attack_key
            rr["adaptive_attack_name"] = attack["name"]
            rr["adaptive_attack_model"] = MODEL

        new_data.append(rr)

    # Audit
    label_counts = {
        "0": sum(1 for r in new_data if int(r["label"]) == 0),
        "1": sum(1 for r in new_data if int(r["label"]) == 1),
    }

    target_in_context = sum(
        1 for r in new_data
        if r.get("target") and r["target"] in (r.get("context") or "")
    )
    duplicate_full_text = len(new_data) - len(set(r.get("full_text", "") for r in new_data))

    attack_records = [r for r in new_data if int(r["label"]) == 1 and "original_target" in r]
    avg_len_old = sum(len((r.get("original_target") or "").split()) for r in attack_records) / max(1, len(attack_records))
    avg_len_new = sum(len((r.get("target") or "").split()) for r in attack_records) / max(1, len(attack_records))

    audit = {
        "dataset": ds["name"],
        "construction": ds["construction"],
        "source_boundary": str(boundary_path),
        "attack": attack_key,
        "attack_name": attack["name"],
        "attack_model": MODEL,
        "limit": limit,
        "total_records": len(new_data),
        "label_counts": label_counts,
        "inject_records_selected_for_rewrite": len(inject_indices),
        "rewritten_count": rewritten_count,
        "unchanged_count": unchanged_count,
        "failed_count": len(failed),
        "failed": failed,
        "target_in_context": target_in_context,
        "duplicate_full_text": duplicate_full_text,
        "avg_target_words_before": avg_len_old,
        "avg_target_words_after": avg_len_new,
        "cache_path": str(cache_path),
        "output_path": str(out_path),
    }

    json.dump(new_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audit, open(audit_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved boundary:", out_path)
    print("saved audit:", audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--attack", choices=list(ATTACKS.keys()), required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    build_attack_boundary(args.dataset, args.attack, args.limit)

if __name__ == "__main__":
    main()

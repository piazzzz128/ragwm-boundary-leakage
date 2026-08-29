import json
import glob
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever"

IN_GLOB = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/boundary_*_attack_a*_qwen-plus_v2_clean.json"

OUT_JSON = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/adaptive_rewrite_qwen_plus_v2_clean_coherence_audit.json"
OUT_MD = "/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned/adaptive_rewrite_qwen_plus_v2_clean_coherence_audit.md"

BATCH_SIZE = 64
MAX_LENGTH = 256

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

@torch.no_grad()
def encode(texts, tokenizer, model, device):
    embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        out = model(**inputs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        emb = mean_pool(h, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu())

    return torch.cat(embs, dim=0).numpy()

def parse_file(path):
    base = os.path.basename(path)

    attack = "unknown"
    for a in ["a1", "a2", "a3"]:
        if f"_attack_{a}_" in base:
            attack = a

    if base.startswith("boundary_nfcorpus"):
        dataset = "NFCorpus"
    elif base.startswith("boundary_trec"):
        dataset = "TREC-COVID"
    elif base.startswith("boundary_nq"):
        dataset = "Natural Questions"
    else:
        dataset = "unknown"

    return dataset, attack

def main():
    paths = sorted(glob.glob(IN_GLOB))
    print("input files:", len(paths))
    for p in paths:
        print(" -", p)

    if not paths:
        raise RuntimeError(f"No cleaned attack boundary files found by pattern: {IN_GLOB}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model:", MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    rows = []

    for p in paths:
        dataset, attack = parse_file(p)
        data = json.load(open(p, "r", encoding="utf-8"))

        # only rewritten inject records have original_target
        inj = [
            r for r in data
            if int(r["label"]) == 1 and "original_target" in r
        ]

        print("\nDATASET:", dataset, "ATTACK:", attack)
        print("records:", len(data), "rewritten inject records:", len(inj))

        if not inj:
            print("skip: no rewritten inject records")
            continue

        contexts = [r.get("context") or "" for r in inj]
        old_targets = [r.get("original_target") or "" for r in inj]
        new_targets = [r.get("target") or "" for r in inj]

        c_emb = encode(contexts, tokenizer, model, device)
        old_emb = encode(old_targets, tokenizer, model, device)
        new_emb = encode(new_targets, tokenizer, model, device)

        old_cos = np.sum(c_emb * old_emb, axis=1)
        new_cos = np.sum(c_emb * new_emb, axis=1)
        delta = new_cos - old_cos

        row = {
            "dataset": dataset,
            "attack": attack,
            "n": len(inj),
            "old_cosine_mean": float(np.mean(old_cos)),
            "new_cosine_mean": float(np.mean(new_cos)),
            "delta_cosine_mean": float(np.mean(delta)),
            "delta_cosine_median": float(np.median(delta)),
            "improved_ratio": float(np.mean(delta > 0)),
            "old_words_mean": float(np.mean([len(x.split()) for x in old_targets])),
            "new_words_mean": float(np.mean([len(x.split()) for x in new_targets])),
        }

        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    json.dump(rows, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Adaptive Rewrite Coherence Audit: qwen-plus v2 Clean\n\n")
        f.write("| Dataset | Attack | N | Old Cosine | New Cosine | Δ Cosine | Improved Ratio | Old Words | New Words |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f'| {r["dataset"]} | {r["attack"]} | {r["n"]} | '
                f'{r["old_cosine_mean"]:.4f} | {r["new_cosine_mean"]:.4f} | '
                f'{r["delta_cosine_mean"]:.4f} | {r["improved_ratio"]:.4f} | '
                f'{r["old_words_mean"]:.2f} | {r["new_words_mean"]:.2f} |\n'
            )

    print("\nsaved:", OUT_JSON)
    print("saved:", OUT_MD)

if __name__ == "__main__":
    main()

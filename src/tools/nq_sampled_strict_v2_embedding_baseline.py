import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModel

BOUNDARY_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2.json")
SCORES_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_contriever_cosine_scores.json")
METRICS_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_contriever_cosine_metrics.json")

MODEL_PATH = "/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever"

BATCH_SIZE = 32
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

        outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            h = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            h = outputs[0]
        else:
            h = outputs

        emb = mean_pool(h, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu())

    return torch.cat(embs, dim=0).numpy()

def tpr_at_fpr(y_true, scores, level):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= level)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))

def best_f1(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(np.max(f1))

def main():
    data = json.load(open(BOUNDARY_PATH, "r", encoding="utf-8"))
    print("loaded boundary:", len(data))

    contexts = [x.get("context") or x.get("context_text") or "" for x in data]
    targets = [x.get("target") or x.get("target_text") or "" for x in data]
    labels = np.array([int(x["label"]) for x in data])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model:", MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    ctx_emb = encode(contexts, tokenizer, model, device)
    tgt_emb = encode(targets, tokenizer, model, device)

    cosine = np.sum(ctx_emb * tgt_emb, axis=1)

    # inject is expected to have lower continuity, so anomaly score = -cosine
    scores = -cosine

    records = []
    for item, c, s in zip(data, cosine, scores):
        r = dict(item)
        r["contriever_cosine"] = float(c)
        r["score"] = float(s)
        records.append(r)

    metrics = {
        "dataset": "nq_sampled_strict_v2",
        "method": "Contriever + Cosine",
        "model": "Contriever",
        "n": int(len(labels)),
        "positive_inject": int(labels.sum()),
        "negative_clean": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "tpr_at_fpr_5": tpr_at_fpr(labels, scores, 0.05),
        "tpr_at_fpr_10": tpr_at_fpr(labels, scores, 0.10),
        "best_f1": best_f1(labels, scores),
        "cosine_mean_inject": float(np.mean(cosine[labels == 1])),
        "cosine_mean_clean": float(np.mean(cosine[labels == 0])),
    }

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(SCORES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics, open(METRICS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved scores:", SCORES_PATH)
    print("saved metrics:", METRICS_PATH)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

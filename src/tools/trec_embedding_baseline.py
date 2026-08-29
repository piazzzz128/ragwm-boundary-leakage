import os
import sys
import json
from pathlib import Path

# 关键修复：无论从哪里运行，都把项目根目录加入 Python import 路径
REPO = Path("/root/autodl-tmp/ragwm_storage/repo/ragwm_src")
sys.path.insert(0, str(REPO))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer

from contriever_src.contriever import Contriever


BOUNDARY_PATH = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v3.json"
MODEL_PATH = "/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever"

OUT_SCORE = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/trec_contriever_cosine_scores.json"
OUT_METRIC = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/trec_contriever_cosine_metrics.json"


def assert_file(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def load_boundary():
    assert_file(BOUNDARY_PATH, "boundary file")
    data = json.load(open(BOUNDARY_PATH, "r", encoding="utf-8"))

    n = len(data)
    pos = sum(1 for x in data if int(x["label"]) == 1)
    neg = sum(1 for x in data if int(x["label"]) == 0)
    target_in_context = sum(1 for x in data if x["target"] in x["context"])

    print("boundary file:", BOUNDARY_PATH)
    print("records:", n)
    print("label 1 inject:", pos)
    print("label 0 clean:", neg)
    print("target_in_context:", target_in_context)

    if n == 0:
        raise RuntimeError("empty boundary dataset")
    if pos != neg:
        raise RuntimeError(f"unbalanced dataset: label1={pos}, label0={neg}")
    if target_in_context != 0:
        raise RuntimeError(f"invalid dataset: target_in_context={target_in_context}")

    return data


@torch.no_grad()
def embed_texts(texts, model, tokenizer, device, batch_size=32, max_length=256):
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs)

        # contriever_src 的 Contriever 通常直接返回 dense embedding tensor
        if isinstance(out, torch.Tensor):
            emb = out
        elif isinstance(out, (tuple, list)):
            emb = out[0]
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            raise RuntimeError(f"unknown model output type: {type(out)}")

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0).numpy()


def main():
    print("repo:", REPO)
    print("model path:", MODEL_PATH)

    assert_file(str(REPO / "contriever_src/contriever.py"), "contriever source")
    assert_file(os.path.join(MODEL_PATH, "config.json"), "contriever config")

    data = load_boundary()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Contriever.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()

    contexts = [x["context"] for x in data]
    targets = [x["target"] for x in data]
    labels = np.array([int(x["label"]) for x in data], dtype=np.int32)

    print("embedding contexts...")
    context_emb = embed_texts(contexts, model, tokenizer, device)

    print("embedding targets...")
    target_emb = embed_texts(targets, model, tokenizer, device)

    cosine = (context_emb * target_emb).sum(axis=1)

    # 逻辑：自然边界 context-target 语义更连续，cosine 通常更高；
    # inject 边界更突兀，cosine 通常更低。
    # 所以 anomaly_score = 1 - cosine，分数越高越像 inject。
    anomaly_score = 1.0 - cosine

    roc_auc = roc_auc_score(labels, anomaly_score)
    pr_auc = average_precision_score(labels, anomaly_score)

    metrics = {
        "dataset": "trec-covid",
        "method": "Contriever + Cosine",
        "boundary_file": BOUNDARY_PATH,
        "n": int(len(data)),
        "positive_inject": int(labels.sum()),
        "negative_clean": int((1 - labels).sum()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "cosine_mean_inject": float(cosine[labels == 1].mean()),
        "cosine_mean_clean": float(cosine[labels == 0].mean()),
        "anomaly_score_mean_inject": float(anomaly_score[labels == 1].mean()),
        "anomaly_score_mean_clean": float(anomaly_score[labels == 0].mean()),
    }

    scored = []
    for row, cos, score in zip(data, cosine, anomaly_score):
        item = dict(row)
        item["contriever_cosine"] = float(cos)
        item["embedding_anomaly_score"] = float(score)
        scored.append(item)

    Path(OUT_SCORE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(scored, open(OUT_SCORE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics, open(OUT_METRIC, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n==== METRICS ====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nsaved scores:", OUT_SCORE)
    print("saved metrics:", OUT_METRIC)


if __name__ == "__main__":
    main()

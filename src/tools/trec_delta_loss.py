import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/ragwm_storage/cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_BOUNDARY = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v3.json"
DEFAULT_OUTDIR = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff"


def ensure_model(repo_id: str, local_dir: str):
    local_dir = Path(local_dir)
    if (local_dir / "config.json").exists():
        print("[MODEL] local model exists:", local_dir)
        return str(local_dir)

    print("[MODEL] local model not found, downloading:", repo_id)
    print("[MODEL] target:", local_dir)

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ],
    )

    print("[MODEL] download finished:", local_dir)
    return str(local_dir)


def load_boundary(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = json.load(open(path, "r", encoding="utf-8"))

    n = len(data)
    pos = sum(1 for r in data if int(r["label"]) == 1)
    neg = sum(1 for r in data if int(r["label"]) == 0)
    target_in_context = sum(1 for r in data if r["target"] in r["context"])
    duplicate_full = n - len(set(r["full_text"] for r in data))

    print("==== BOUNDARY CHECK ====")
    print("path:", path)
    print("total:", n)
    print("label 1 inject:", pos)
    print("label 0 clean:", neg)
    print("target_in_context:", target_in_context)
    print("duplicate_full_text:", duplicate_full)

    if n == 0:
        raise RuntimeError("boundary dataset is empty")
    if pos != neg:
        raise RuntimeError(f"dataset not balanced: label1={pos}, label0={neg}")
    if target_in_context != 0:
        raise RuntimeError(f"target_in_context must be 0, got {target_in_context}")
    if duplicate_full != 0:
        raise RuntimeError(f"duplicate_full_text must be 0, got {duplicate_full}")

    return data


def encode(tokenizer, text, max_tokens=None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[-max_tokens:]
    return ids


@torch.no_grad()
def masked_lm_loss(model, input_ids, labels, device):
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)

    out = model(input_ids=input_ids)
    logits = out.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    valid = shift_labels.view(-1) != -100
    if valid.sum().item() == 0:
        return None

    return losses[valid].mean().item()


@torch.no_grad()
def score_one(row, model, tokenizer, device, max_length=512, max_target_tokens=128):
    context = row["context"]
    target = row["target"]

    target_ids = encode(tokenizer, target, max_tokens=max_target_tokens)

    if len(target_ids) < 2:
        return None

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
    if eos is None:
        eos = 50256

    # L(T): unconditional target loss.
    # Add one prefix token so the first target token is also predicted.
    uncond_ids = [eos] + target_ids
    uncond_labels = [-100] + target_ids

    # L(T|C): conditional target loss.
    max_context_tokens = max_length - len(target_ids)
    if max_context_tokens < 8:
        target_ids = target_ids[-(max_length - 8):]
        max_context_tokens = max_length - len(target_ids)

    context_ids = encode(tokenizer, context, max_tokens=max_context_tokens)
    if len(context_ids) == 0:
        context_ids = [eos]

    cond_ids = context_ids + target_ids
    cond_labels = [-100] * len(context_ids) + target_ids

    # Hard safety truncation
    uncond_ids = uncond_ids[-max_length:]
    uncond_labels = uncond_labels[-max_length:]

    cond_ids = cond_ids[-max_length:]
    cond_labels = cond_labels[-max_length:]

    uncond_loss = masked_lm_loss(model, uncond_ids, uncond_labels, device)
    cond_loss = masked_lm_loss(model, cond_ids, cond_labels, device)

    if uncond_loss is None or cond_loss is None:
        return None

    delta_loss = uncond_loss - cond_loss
    gain_ratio = delta_loss / uncond_loss if uncond_loss > 0 else 0.0

    # label=1 是 inject。inject 边界通常上下文帮助更小，因此 delta/gain 更低。
    # AUC 需要 label=1 分数越高，所以 anomaly score 取负。
    return {
        "uncond_loss": float(uncond_loss),
        "cond_loss": float(cond_loss),
        "delta_loss": float(delta_loss),
        "gain_ratio": float(gain_ratio),
        "delta_anomaly_score": float(-delta_loss),
        "gain_anomaly_score": float(-gain_ratio),
    }


def metrics_for_scores(data, score_key):
    labels = np.array([int(r["label"]) for r in data], dtype=np.int32)
    scores = np.array([float(r[score_key]) for r in data], dtype=np.float64)

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "score_mean_inject": float(scores[labels == 1].mean()),
        "score_mean_clean": float(scores[labels == 0].mean()),
    }


def group_metrics(data, score_key, reducer="mean"):
    groups = defaultdict(list)
    labels = {}

    for r in data:
        gid = r.get("group_id") or r.get("id")
        groups[gid].append(float(r[score_key]))
        labels[gid] = int(r["label"])

    y = []
    s = []

    for gid, vals in groups.items():
        y.append(labels[gid])
        if reducer == "median":
            s.append(float(np.median(vals)))
        else:
            s.append(float(np.mean(vals)))

    y = np.array(y, dtype=np.int32)
    s = np.array(s, dtype=np.float64)

    return {
        "groups": int(len(groups)),
        "positive_groups": int(y.sum()),
        "negative_groups": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "score_mean_inject": float(s[y == 1].mean()),
        "score_mean_clean": float(s[y == 0].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", default=DEFAULT_BOUNDARY)
    parser.add_argument("--repo_id", default="distilgpt2")
    parser.add_argument("--model_dir", default="/root/autodl-tmp/ragwm_storage/local_models/distilgpt2")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--tag", default="distilgpt2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_target_tokens", type=int, default=128)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/ragwm_storage/cache/huggingface")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    data = load_boundary(args.boundary)

    if args.limit and args.limit > 0:
        data = data[:args.limit]
        print("[LIMIT] using first records:", len(data))

    model_path = ensure_model(args.repo_id, args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==== MODEL ====")
    print("repo_id:", args.repo_id)
    print("model_path:", model_path)
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    # Robust local loading:
    # - GPT-2 can load pytorch_model.bin.
    # - Qwen2.5-7B usually uses safetensors shards.
    # Therefore do not force use_safetensors=False globally.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    scored = []
    failed = []

    for i, row in enumerate(tqdm(data, desc="DeltaLoss scoring")):
        try:
            s = score_one(
                row,
                model,
                tokenizer,
                device,
                max_length=args.max_length,
                max_target_tokens=args.max_target_tokens,
            )
            if s is None:
                failed.append({"index": i, "id": row.get("id"), "reason": "empty_score"})
                continue

            item = dict(row)
            item.update(s)
            scored.append(item)

        except Exception as e:
            failed.append({"index": i, "id": row.get("id"), "reason": repr(e)})

    if len(scored) == 0:
        raise RuntimeError("No scored records.")

    pos = sum(1 for r in scored if int(r["label"]) == 1)
    neg = sum(1 for r in scored if int(r["label"]) == 0)

    if pos == 0 or neg == 0:
        raise RuntimeError(f"invalid scored labels: pos={pos}, neg={neg}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    score_path = outdir / f"trec_delta_loss_scores_{args.tag}.json"
    metric_path = outdir / f"trec_delta_loss_metrics_{args.tag}.json"
    failed_path = outdir / f"trec_delta_loss_failed_{args.tag}.json"

    metrics = {
        "dataset": "trec-covid",
        "boundary_file": args.boundary,
        "model_repo_id": args.repo_id,
        "model_path": model_path,
        "tag": args.tag,
        "n_input": len(data),
        "n_scored": len(scored),
        "n_failed": len(failed),
        "positive_inject": pos,
        "negative_clean": neg,

        "record_delta_loss": metrics_for_scores(scored, "delta_anomaly_score"),
        "record_gain_ratio": metrics_for_scores(scored, "gain_anomaly_score"),

        "group_mean_delta_loss": group_metrics(scored, "delta_anomaly_score", "mean"),
        "group_median_delta_loss": group_metrics(scored, "delta_anomaly_score", "median"),
        "group_mean_gain_ratio": group_metrics(scored, "gain_anomaly_score", "mean"),
        "group_median_gain_ratio": group_metrics(scored, "gain_anomaly_score", "median"),

        "delta_loss_mean_inject": float(np.mean([r["delta_loss"] for r in scored if r["label"] == 1])),
        "delta_loss_mean_clean": float(np.mean([r["delta_loss"] for r in scored if r["label"] == 0])),
        "gain_ratio_mean_inject": float(np.mean([r["gain_ratio"] for r in scored if r["label"] == 1])),
        "gain_ratio_mean_clean": float(np.mean([r["gain_ratio"] for r in scored if r["label"] == 0])),
    }

    json.dump(scored, open(score_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics, open(metric_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(failed, open(failed_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n==== METRICS ====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nsaved scores:", score_path)
    print("saved metrics:", metric_path)
    print("saved failed:", failed_path)


if __name__ == "__main__":
    main()

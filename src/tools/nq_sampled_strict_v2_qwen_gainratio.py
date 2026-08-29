import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

BOUNDARY_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2.json")
SCORES_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_qwen_gainratio_scores.json")
METRICS_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_qwen_gainratio_metrics.json")
FAILED_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_qwen_gainratio_failed.json")

MODEL_PATH = "/root/autodl-tmp/ragwm_storage/local_models/Qwen2.5-7B"

MAX_CONTEXT_TOKENS = 1024
MAX_TARGET_TOKENS = 128

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

def calc_metrics(y, score):
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "tpr_at_fpr_5": tpr_at_fpr(y, score, 0.05),
        "tpr_at_fpr_10": tpr_at_fpr(y, score, 0.10),
        "best_f1": best_f1(y, score),
    }

def aggregate_group(records, score_key):
    groups = defaultdict(list)
    labels = {}

    for r in records:
        gid = r.get("group_id") or r.get("source_doc_id") or r.get("id")
        groups[gid].append(float(r[score_key]))
        labels[gid] = int(r["label"])

    y = []
    mean_scores = []
    median_scores = []

    for gid, vals in groups.items():
        y.append(labels[gid])
        mean_scores.append(float(np.mean(vals)))
        median_scores.append(float(np.median(vals)))

    return np.array(y), np.array(mean_scores), np.array(median_scores)

@torch.no_grad()
def target_nll(tokenizer, model, device, context, target):
    context = (context or "").strip()
    target = (target or "").strip()

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TARGET_TOKENS,
    )["input_ids"]

    if len(target_ids) < 2:
        return None

    context_ids = []
    if context:
        context_ids = tokenizer(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
        )["input_ids"]

    input_ids = context_ids + target_ids
    if len(input_ids) < 2:
        return None

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    outputs = model(ids)
    logits = outputs.logits

    start = len(context_ids)
    end = len(input_ids)

    pred_positions = []
    label_positions = []

    for p in range(start, end):
        if p == 0:
            continue
        pred_positions.append(p - 1)
        label_positions.append(p)

    if not pred_positions:
        return None

    pred_logits = logits[0, pred_positions, :]
    labels = ids[0, label_positions]

    loss = torch.nn.functional.cross_entropy(
        pred_logits.float(),
        labels,
        reduction="mean",
    )

    return float(loss.item())

def main():
    data = json.load(open(BOUNDARY_PATH, "r", encoding="utf-8"))
    print("loaded boundary:", len(data))
    print("model:", MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    records = []
    failed = []

    for i, item in enumerate(data):
        context = item.get("context") or item.get("context_text") or ""
        target = item.get("target") or item.get("target_text") or ""

        try:
            loss_t = target_nll(tokenizer, model, device, "", target)
            loss_tc = target_nll(tokenizer, model, device, context, target)

            if loss_t is None or loss_tc is None or loss_t <= 0:
                failed.append({"index": i, "id": item.get("id"), "reason": "invalid_loss"})
                continue

            delta_loss = loss_t - loss_tc
            gain_ratio = delta_loss / loss_t

            # label=1 is inject. Inject should have lower gain_ratio.
            score_gain = -gain_ratio
            score_delta = -delta_loss

            r = dict(item)
            r["loss_target"] = loss_t
            r["loss_target_given_context"] = loss_tc
            r["delta_loss"] = delta_loss
            r["gain_ratio"] = gain_ratio
            r["score"] = score_gain
            r["score_delta_loss"] = score_delta
            records.append(r)

            if (i + 1) % 25 == 0:
                print(f"scored {i+1}/{len(data)}, valid={len(records)}, failed={len(failed)}", flush=True)

        except Exception as e:
            failed.append({"index": i, "id": item.get("id"), "reason": repr(e)})
            print("FAILED", i, repr(e), flush=True)

    if not records:
        raise RuntimeError("No valid records scored.")

    y = np.array([int(r["label"]) for r in records])
    score_gain = np.array([float(r["score"]) for r in records])
    score_delta = np.array([float(r["score_delta_loss"]) for r in records])

    metrics = {
        "dataset": "nq_sampled_strict_v2",
        "method": "GainRatio",
        "model": "Qwen2.5-7B",
        "n_input": len(data),
        "n_scored": len(records),
        "n_failed": len(failed),
        "positive_inject": int(y.sum()),
        "negative_clean": int((y == 0).sum()),
        "record_gain_ratio": calc_metrics(y, score_gain),
        "record_delta_loss": calc_metrics(y, score_delta),
        "gain_ratio_mean_inject": float(np.mean([r["gain_ratio"] for r in records if int(r["label"]) == 1])),
        "gain_ratio_mean_clean": float(np.mean([r["gain_ratio"] for r in records if int(r["label"]) == 0])),
        "delta_loss_mean_inject": float(np.mean([r["delta_loss"] for r in records if int(r["label"]) == 1])),
        "delta_loss_mean_clean": float(np.mean([r["delta_loss"] for r in records if int(r["label"]) == 0])),
    }

    gy, gmean_gain, gmedian_gain = aggregate_group(records, "score")
    metrics["group_mean_gain_ratio"] = calc_metrics(gy, gmean_gain)
    metrics["group_median_gain_ratio"] = calc_metrics(gy, gmedian_gain)

    gy2, gmean_delta, gmedian_delta = aggregate_group(records, "score_delta_loss")
    metrics["group_mean_delta_loss"] = calc_metrics(gy2, gmean_delta)
    metrics["group_median_delta_loss"] = calc_metrics(gy2, gmedian_delta)

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(SCORES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics, open(METRICS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(failed, open(FAILED_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved scores:", SCORES_PATH)
    print("saved metrics:", METRICS_PATH)
    print("saved failed:", FAILED_PATH)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

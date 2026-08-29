import json
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/ragwm_storage/local_models/Qwen2.5-7B"

DATASETS = [
    {
        "dataset": "NFCorpus",
        "construction": "strict v1",
        "boundary": "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nfcorpus_strict_v1.json",
    },
    {
        "dataset": "TREC-COVID",
        "construction": "strict v3",
        "boundary": "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v3.json",
    },
    {
        "dataset": "Natural Questions",
        "construction": "sampled strict v2",
        "boundary": "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2.json",
    },
]

CONTEXT_BUDGETS = [16, 32, 64, 128, 256, "full"]
MAX_LENGTH = 512
MAX_TARGET_TOKENS = 128

OUT_DIR = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff")
SCORES_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_aligned_scores.json"
METRICS_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_aligned_metrics.json"
CSV_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_aligned_table.csv"
MD_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_aligned_table.md"

def encode(tokenizer, text, max_tokens=None):
    ids = tokenizer.encode(text or "", add_special_tokens=False)
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
def score_one_with_budget(row, model, tokenizer, device, budget):
    context = row["context"]
    target = row["target"]

    target_ids = encode(tokenizer, target, max_tokens=MAX_TARGET_TOKENS)
    if len(target_ids) < 2:
        return None

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
    if eos is None:
        eos = 50256

    # Old main experiment logic: add one prefix token so first target token is predicted.
    uncond_ids = [eos] + target_ids
    uncond_labels = [-100] + target_ids

    if budget == "full":
        max_context_tokens = MAX_LENGTH - len(target_ids)
        if max_context_tokens < 8:
            target_ids = target_ids[-(MAX_LENGTH - 8):]
            max_context_tokens = MAX_LENGTH - len(target_ids)
    else:
        max_context_tokens = int(budget)

    context_ids = encode(tokenizer, context, max_tokens=max_context_tokens)
    if len(context_ids) == 0:
        context_ids = [eos]

    cond_ids = context_ids + target_ids
    cond_labels = [-100] * len(context_ids) + target_ids

    # Same hard truncation as trec_delta_loss.py
    uncond_ids = uncond_ids[-MAX_LENGTH:]
    uncond_labels = uncond_labels[-MAX_LENGTH:]

    cond_ids = cond_ids[-MAX_LENGTH:]
    cond_labels = cond_labels[-MAX_LENGTH:]

    uncond_loss = masked_lm_loss(model, uncond_ids, uncond_labels, device)
    cond_loss = masked_lm_loss(model, cond_ids, cond_labels, device)

    if uncond_loss is None or cond_loss is None:
        return None

    delta_loss = uncond_loss - cond_loss
    gain_ratio = delta_loss / uncond_loss if uncond_loss > 0 else 0.0

    return {
        "uncond_loss": float(uncond_loss),
        "cond_loss": float(cond_loss),
        "delta_loss": float(delta_loss),
        "gain_ratio": float(gain_ratio),
        "delta_anomaly_score": float(-delta_loss),
        "gain_anomaly_score": float(-gain_ratio),
        "context_tokens_used": len(context_ids),
        "target_tokens": len(target_ids),
    }

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

def main():
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

    all_scores = []
    metrics_rows = []

    for ds in DATASETS:
        dataset = ds["dataset"]
        construction = ds["construction"]
        path = Path(ds["boundary"])

        print("\n====", dataset, construction, "====")
        data = json.load(open(path, "r", encoding="utf-8"))
        print("records:", len(data))

        for budget in CONTEXT_BUDGETS:
            bkey = str(budget)
            records = []
            failed = []

            for i, row in enumerate(data):
                out = score_one_with_budget(row, model, tokenizer, device, budget)

                if out is None:
                    failed.append({"index": i, "id": row.get("id"), "budget": bkey})
                    continue

                rec = {
                    "dataset": dataset,
                    "construction": construction,
                    "record_id": row.get("id"),
                    "label": int(row["label"]),
                    "group_id": row.get("group_id"),
                    "context_budget": bkey,
                    **out,
                }
                records.append(rec)
                all_scores.append(rec)

                if (i + 1) % 100 == 0:
                    print(dataset, "budget", bkey, "scored", i + 1, "/", len(data), flush=True)

            y = np.array([int(r["label"]) for r in records])
            score = np.array([float(r["gain_anomaly_score"]) for r in records])
            gr = np.array([float(r["gain_ratio"]) for r in records])
            dl = np.array([float(r["delta_loss"]) for r in records])

            m = calc_metrics(y, score)

            metrics = {
                "dataset": dataset,
                "construction": construction,
                "model": "Qwen2.5-7B",
                "method": "GainRatio",
                "context_budget_tokens": bkey,
                "n": int(len(records)),
                "positive_inject": int(y.sum()),
                "negative_clean": int((y == 0).sum()),
                "n_failed": int(len(failed)),
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "tpr_at_fpr_5": m["tpr_at_fpr_5"],
                "tpr_at_fpr_10": m["tpr_at_fpr_10"],
                "best_f1": m["best_f1"],
                "gain_ratio_mean_inject": float(np.mean(gr[y == 1])),
                "gain_ratio_mean_clean": float(np.mean(gr[y == 0])),
                "delta_loss_mean_inject": float(np.mean(dl[y == 1])),
                "delta_loss_mean_clean": float(np.mean(dl[y == 0])),
            }

            metrics_rows.append(metrics)

            print(
                dataset,
                "budget=", bkey,
                "ROC=", f'{metrics["roc_auc"]:.4f}',
                "PR=", f'{metrics["pr_auc"]:.4f}',
                "TPR5=", f'{metrics["tpr_at_fpr_5"]:.4f}',
                "F1=", f'{metrics["best_f1"]:.4f}',
                flush=True
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json.dump(all_scores, open(SCORES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics_rows, open(METRICS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    headers = [
        "dataset", "construction", "model", "method", "context_budget_tokens",
        "n", "positive_inject", "negative_clean", "n_failed",
        "roc_auc", "pr_auc", "tpr_at_fpr_5", "tpr_at_fpr_10", "best_f1",
        "gain_ratio_mean_inject", "gain_ratio_mean_clean",
        "delta_loss_mean_inject", "delta_loss_mean_clean",
    ]

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metrics_rows)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Context Length Ablation: Qwen2.5-7B + GainRatio Aligned\n\n")
        f.write("| Dataset | Construction | Context Tokens | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in metrics_rows:
            f.write(
                f'| {r["dataset"]} | {r["construction"]} | {r["context_budget_tokens"]} | '
                f'{r["n"]} | {r["roc_auc"]:.4f} | {r["pr_auc"]:.4f} | '
                f'{r["tpr_at_fpr_5"]:.4f} | {r["tpr_at_fpr_10"]:.4f} | {r["best_f1"]:.4f} |\n'
            )

    print("saved scores:", SCORES_PATH)
    print("saved metrics:", METRICS_PATH)
    print("saved csv:", CSV_PATH)
    print("saved md:", MD_PATH)

if __name__ == "__main__":
    main()

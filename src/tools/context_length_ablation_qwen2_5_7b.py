import json
import csv
from pathlib import Path
from collections import defaultdict

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
MAX_FULL_CONTEXT_TOKENS = 1024
MAX_TARGET_TOKENS = 128

OUT_DIR = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff")
SCORES_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_scores.json"
METRICS_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_metrics.json"
CSV_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_table.csv"
MD_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_table.md"

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

def token_ids(tokenizer, text, max_tokens=None):
    ids = tokenizer(text or "", add_special_tokens=False)["input_ids"]
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[-max_tokens:]
    return ids

def context_ids_for_budget(tokenizer, context, budget):
    if budget == "full":
        return token_ids(tokenizer, context, MAX_FULL_CONTEXT_TOKENS)
    return token_ids(tokenizer, context, int(budget))

@torch.no_grad()
def target_nll_from_ids(model, device, context_ids, target_ids):
    if len(target_ids) < 2:
        return None

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

        print("\n===== DATASET:", dataset, construction, "=====")
        print("boundary:", path)

        data = json.load(open(path, "r", encoding="utf-8"))
        print("records:", len(data))

        labels = np.array([int(r["label"]) for r in data])
        print("positive:", int(labels.sum()), "negative:", int((labels == 0).sum()))

        per_budget_records = {str(b): [] for b in CONTEXT_BUDGETS}
        failed = []

        for i, item in enumerate(data):
            context = item.get("context") or item.get("context_text") or ""
            target = item.get("target") or item.get("target_text") or ""

            try:
                target_ids = token_ids(tokenizer, target, MAX_TARGET_TOKENS)
                loss_t = target_nll_from_ids(model, device, [], target_ids)

                if loss_t is None or loss_t <= 0:
                    failed.append({"dataset": dataset, "index": i, "id": item.get("id"), "reason": "invalid_target_loss"})
                    continue

                for budget in CONTEXT_BUDGETS:
                    bkey = str(budget)
                    cids = context_ids_for_budget(tokenizer, context, budget)
                    loss_tc = target_nll_from_ids(model, device, cids, target_ids)

                    if loss_tc is None:
                        failed.append({"dataset": dataset, "index": i, "id": item.get("id"), "budget": bkey, "reason": "invalid_context_loss"})
                        continue

                    delta_loss = loss_t - loss_tc
                    gain_ratio = delta_loss / loss_t
                    score = -gain_ratio

                    rec = {
                        "dataset": dataset,
                        "construction": construction,
                        "record_id": item.get("id"),
                        "label": int(item["label"]),
                        "group_id": item.get("group_id"),
                        "context_budget": bkey,
                        "context_tokens_used": len(cids),
                        "target_tokens": len(target_ids),
                        "loss_target": loss_t,
                        "loss_target_given_context": loss_tc,
                        "delta_loss": delta_loss,
                        "gain_ratio": gain_ratio,
                        "score": score,
                    }
                    per_budget_records[bkey].append(rec)
                    all_scores.append(rec)

                if (i + 1) % 25 == 0:
                    print(f"{dataset}: scored {i+1}/{len(data)}", flush=True)

            except Exception as e:
                failed.append({"dataset": dataset, "index": i, "id": item.get("id"), "reason": repr(e)})
                print("FAILED", dataset, i, repr(e), flush=True)

        print("failed count:", len(failed))

        for budget in CONTEXT_BUDGETS:
            bkey = str(budget)
            rows = per_budget_records[bkey]
            y = np.array([r["label"] for r in rows])
            score = np.array([r["score"] for r in rows])
            gr = np.array([r["gain_ratio"] for r in rows])
            dl = np.array([r["delta_loss"] for r in rows])

            m = calc_metrics(y, score)

            row = {
                "dataset": dataset,
                "construction": construction,
                "model": "Qwen2.5-7B",
                "method": "GainRatio",
                "context_budget_tokens": bkey,
                "n": int(len(rows)),
                "positive_inject": int(y.sum()),
                "negative_clean": int((y == 0).sum()),
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
            metrics_rows.append(row)

            print(
                dataset,
                "budget=", bkey,
                "ROC=", f'{row["roc_auc"]:.4f}',
                "PR=", f'{row["pr_auc"]:.4f}',
                "TPR5=", f'{row["tpr_at_fpr_5"]:.4f}',
                "F1=", f'{row["best_f1"]:.4f}',
                flush=True
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json.dump(all_scores, open(SCORES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics_rows, open(METRICS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    headers = [
        "dataset", "construction", "model", "method", "context_budget_tokens",
        "n", "positive_inject", "negative_clean",
        "roc_auc", "pr_auc", "tpr_at_fpr_5", "tpr_at_fpr_10", "best_f1",
        "gain_ratio_mean_inject", "gain_ratio_mean_clean",
        "delta_loss_mean_inject", "delta_loss_mean_clean",
    ]

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metrics_rows)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Context Length Ablation: Qwen2.5-7B + GainRatio\n\n")
        f.write("| Dataset | Construction | Context Tokens | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in metrics_rows:
            f.write(
                f'| {r["dataset"]} | {r["construction"]} | {r["context_budget_tokens"]} | '
                f'{r["n"]} | {r["roc_auc"]:.4f} | {r["pr_auc"]:.4f} | '
                f'{r["tpr_at_fpr_5"]:.4f} | {r["tpr_at_fpr_10"]:.4f} | {r["best_f1"]:.4f} |\n'
            )

    print("\nsaved scores:", SCORES_PATH)
    print("saved metrics:", METRICS_PATH)
    print("saved csv:", CSV_PATH)
    print("saved md:", MD_PATH)

if __name__ == "__main__":
    main()

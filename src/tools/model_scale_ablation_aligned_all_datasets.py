import json
import csv
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff")

DATASETS = [
    {
        "dataset": "NFCorpus",
        "construction": "strict v1",
        "boundary": OUT_DIR / "boundary_nfcorpus_strict_v1.json",
    },
    {
        "dataset": "TREC-COVID",
        "construction": "strict v3",
        "boundary": OUT_DIR / "boundary_trec_k5_v3.json",
    },
    {
        "dataset": "Natural Questions",
        "construction": "sampled strict v2",
        "boundary": OUT_DIR / "boundary_nq_sampled_strict_v2.json",
    },
]

MODELS_TO_RUN = [
    {
        "model": "distilgpt2",
        "path": "/root/autodl-tmp/ragwm_storage/local_models/distilgpt2",
    },
    {
        "model": "gpt2-medium",
        "path": "/root/autodl-tmp/ragwm_storage/local_models/gpt2-medium",
    },
]

QWEN_ALIGNED_SCORE_PATH = OUT_DIR / "context_length_ablation_qwen2_5_7b_aligned_scores.json"

SCORES_PATH = OUT_DIR / "model_scale_ablation_aligned_all_datasets_scores.json"
METRICS_PATH = OUT_DIR / "model_scale_ablation_aligned_all_datasets_metrics.json"
CSV_PATH = OUT_DIR / "model_scale_ablation_aligned_all_datasets_table.csv"
MD_PATH = OUT_DIR / "model_scale_ablation_aligned_all_datasets_table.md"

MAX_LENGTH = 512
MAX_TARGET_TOKENS = 128

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
def score_one(row, model, tokenizer, device):
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

    # Unified aligned scoring:
    # L(T): prepend EOS so first target token is predicted.
    uncond_ids = [eos] + target_ids
    uncond_labels = [-100] + target_ids

    # L(T|C): mask context tokens, compute loss only on target tokens.
    max_context_tokens = MAX_LENGTH - len(target_ids)
    if max_context_tokens < 8:
        target_ids = target_ids[-(MAX_LENGTH - 8):]
        max_context_tokens = MAX_LENGTH - len(target_ids)

    context_ids = encode(tokenizer, context, max_tokens=max_context_tokens)
    if len(context_ids) == 0:
        context_ids = [eos]

    cond_ids = context_ids + target_ids
    cond_labels = [-100] * len(context_ids) + target_ids

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

def calc_metrics(records, score_key):
    y = np.array([int(r["label"]) for r in records], dtype=np.int32)
    s = np.array([float(r[score_key]) for r in records], dtype=np.float64)

    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "tpr_at_fpr_5": tpr_at_fpr(y, s, 0.05),
        "tpr_at_fpr_10": tpr_at_fpr(y, s, 0.10),
        "best_f1": best_f1(y, s),
        "score_mean_inject": float(s[y == 1].mean()),
        "score_mean_clean": float(s[y == 0].mean()),
    }

def summarize(dataset_name, construction, model_name, records):
    y = np.array([int(r["label"]) for r in records], dtype=np.int32)

    return [
        {
            "dataset": dataset_name,
            "construction": construction,
            "method": "DeltaLoss",
            "model": model_name,
            "n": int(len(records)),
            "positive_inject": int(y.sum()),
            "negative_clean": int((y == 0).sum()),
            **calc_metrics(records, "delta_anomaly_score"),
            "delta_loss_mean_inject": float(np.mean([r["delta_loss"] for r in records if int(r["label"]) == 1])),
            "delta_loss_mean_clean": float(np.mean([r["delta_loss"] for r in records if int(r["label"]) == 0])),
            "gain_ratio_mean_inject": float(np.mean([r["gain_ratio"] for r in records if int(r["label"]) == 1])),
            "gain_ratio_mean_clean": float(np.mean([r["gain_ratio"] for r in records if int(r["label"]) == 0])),
        },
        {
            "dataset": dataset_name,
            "construction": construction,
            "method": "GainRatio",
            "model": model_name,
            "n": int(len(records)),
            "positive_inject": int(y.sum()),
            "negative_clean": int((y == 0).sum()),
            **calc_metrics(records, "gain_anomaly_score"),
            "delta_loss_mean_inject": float(np.mean([r["delta_loss"] for r in records if int(r["label"]) == 1])),
            "delta_loss_mean_clean": float(np.mean([r["delta_loss"] for r in records if int(r["label"]) == 0])),
            "gain_ratio_mean_inject": float(np.mean([r["gain_ratio"] for r in records if int(r["label"]) == 1])),
            "gain_ratio_mean_clean": float(np.mean([r["gain_ratio"] for r in records if int(r["label"]) == 0])),
        },
    ]

def load_qwen_full_scores():
    if not QWEN_ALIGNED_SCORE_PATH.exists():
        print("[WARN] Qwen aligned score file not found:", QWEN_ALIGNED_SCORE_PATH)
        return [], []

    raw = json.load(open(QWEN_ALIGNED_SCORE_PATH, "r", encoding="utf-8"))
    full = [r for r in raw if str(r.get("context_budget")) == "full"]

    all_scores = []
    metrics = []

    mapping = {
        "NFCorpus": "strict v1",
        "TREC-COVID": "strict v3",
        "Natural Questions": "sampled strict v2",
    }

    for dataset_name, construction in mapping.items():
        rows = [r for r in full if r["dataset"] == dataset_name]
        if not rows:
            print("[WARN] no Qwen full rows for", dataset_name)
            continue

        converted = []
        for r in rows:
            rec = {
                "dataset": dataset_name,
                "construction": construction,
                "model": "Qwen2.5-7B",
                "record_id": r.get("record_id"),
                "label": int(r["label"]),
                "group_id": r.get("group_id"),
                "uncond_loss": float(r["uncond_loss"]),
                "cond_loss": float(r["cond_loss"]),
                "delta_loss": float(r["delta_loss"]),
                "gain_ratio": float(r["gain_ratio"]),
                "delta_anomaly_score": float(r["delta_anomaly_score"]),
                "gain_anomaly_score": float(r["gain_anomaly_score"]),
            }
            converted.append(rec)
            all_scores.append(rec)

        metrics.extend(summarize(dataset_name, construction, "Qwen2.5-7B", converted))

    return all_scores, metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    all_scores = []
    all_metrics = []

    for model_info in MODELS_TO_RUN:
        model_name = model_info["model"]
        model_path = model_info["path"]

        print("\n==============================")
        print("Loading model:", model_name)
        print("Path:", model_path)
        print("==============================")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()

        for ds in DATASETS:
            dataset_name = ds["dataset"]
            construction = ds["construction"]
            boundary_path = Path(ds["boundary"])

            print("\nDATASET:", dataset_name, construction)
            print("boundary:", boundary_path)

            data = json.load(open(boundary_path, "r", encoding="utf-8"))
            print("records:", len(data))

            scored = []
            failed = []

            for i, row in enumerate(data):
                out = score_one(row, model, tokenizer, device)

                if out is None:
                    failed.append({"index": i, "id": row.get("id")})
                    continue

                rec = {
                    "dataset": dataset_name,
                    "construction": construction,
                    "model": model_name,
                    "record_id": row.get("id"),
                    "label": int(row["label"]),
                    "group_id": row.get("group_id"),
                    **out,
                }
                scored.append(rec)
                all_scores.append(rec)

                if (i + 1) % 100 == 0:
                    print(model_name, dataset_name, "scored", i + 1, "/", len(data), flush=True)

            print("valid:", len(scored), "failed:", len(failed))

            metric_rows = summarize(dataset_name, construction, model_name, scored)
            all_metrics.extend(metric_rows)

            for m in metric_rows:
                print(
                    m["dataset"],
                    m["model"],
                    m["method"],
                    "ROC", f'{m["roc_auc"]:.4f}',
                    "PR", f'{m["pr_auc"]:.4f}',
                    "TPR5", f'{m["tpr_at_fpr_5"]:.4f}',
                    "F1", f'{m["best_f1"]:.4f}',
                    flush=True
                )

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    qwen_scores, qwen_metrics = load_qwen_full_scores()
    all_scores.extend(qwen_scores)
    all_metrics.extend(qwen_metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json.dump(all_scores, open(SCORES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(all_metrics, open(METRICS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    headers = [
        "dataset", "construction", "method", "model", "n", "positive_inject", "negative_clean",
        "roc_auc", "pr_auc", "tpr_at_fpr_5", "tpr_at_fpr_10", "best_f1",
        "score_mean_inject", "score_mean_clean",
        "delta_loss_mean_inject", "delta_loss_mean_clean",
        "gain_ratio_mean_inject", "gain_ratio_mean_clean",
    ]

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_metrics)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Scale Ablation: Aligned Loss-based Detection\n\n")
        f.write("| Dataset | Construction | Method | Model | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")

        order = {"distilgpt2": 0, "gpt2-medium": 1, "Qwen2.5-7B": 2}
        method_order = {"DeltaLoss": 0, "GainRatio": 1}
        sorted_metrics = sorted(all_metrics, key=lambda r: (r["dataset"], order.get(r["model"], 99), method_order.get(r["method"], 99)))

        for r in sorted_metrics:
            f.write(
                f'| {r["dataset"]} | {r["construction"]} | {r["method"]} | {r["model"]} | '
                f'{r["n"]} | {r["roc_auc"]:.4f} | {r["pr_auc"]:.4f} | '
                f'{r["tpr_at_fpr_5"]:.4f} | {r["tpr_at_fpr_10"]:.4f} | {r["best_f1"]:.4f} |\n'
            )

    print("\nsaved scores:", SCORES_PATH)
    print("saved metrics:", METRICS_PATH)
    print("saved csv:", CSV_PATH)
    print("saved md:", MD_PATH)

if __name__ == "__main__":
    main()

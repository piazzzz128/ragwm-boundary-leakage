import json
import glob
import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH="/root/autodl-tmp/ragwm_storage/local_models/Qwen2.5-7B"
IN_GLOB="/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2/boundary_*_attack_a2_deepseek-chat_v2_clean.json"

OUT_DIR=Path("/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2")
SCORES_PATH=OUT_DIR/"adaptive_attack_deepseek_chat_a2_v2_clean_matched_aligned_scores.json"
METRICS_PATH=OUT_DIR/"adaptive_attack_deepseek_chat_a2_v2_clean_matched_aligned_metrics.json"
CACHE_PATH=OUT_DIR/"adaptive_attack_deepseek_chat_a2_v2_clean_matched_aligned_score_cache.json"
MD_PATH=OUT_DIR/"adaptive_attack_deepseek_chat_a2_v2_clean_matched_aligned_table.md"
CSV_PATH=OUT_DIR/"adaptive_attack_deepseek_chat_a2_v2_clean_matched_aligned_table.csv"

MAX_LENGTH=512
MAX_TARGET_TOKENS=128

EMBEDDING_BASELINE = {
    "NFCorpus": {"n":446, "roc_auc":0.8346, "pr_auc":0.8034, "tpr_at_fpr_5":0.2735, "tpr_at_fpr_10":0.4395, "best_f1":0.7900},
    "TREC-COVID": {"n":458, "roc_auc":0.6891, "pr_auc":0.6641, "tpr_at_fpr_5":0.1397, "tpr_at_fpr_10":0.2271, "best_f1":0.7099},
    "Natural Questions": {"n":442, "roc_auc":0.8953, "pr_auc":0.8646, "tpr_at_fpr_5":0.5068, "tpr_at_fpr_10":0.6471, "best_f1":0.8373},
}

def sha_key(context, target):
    raw = json.dumps({"context": context or "", "target": target or ""}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def encode(tokenizer, text, max_tokens=None):
    ids=tokenizer.encode(text or "", add_special_tokens=False)
    if max_tokens is not None and len(ids)>max_tokens:
        ids=ids[-max_tokens:]
    return ids

@torch.no_grad()
def masked_lm_loss(model, input_ids, labels, device):
    input_ids=torch.tensor([input_ids], dtype=torch.long, device=device)
    labels=torch.tensor([labels], dtype=torch.long, device=device)

    out=model(input_ids=input_ids)
    logits=out.logits

    shift_logits=logits[:,:-1,:].contiguous()
    shift_labels=labels[:,1:].contiguous()

    loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    losses=loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    valid=shift_labels.view(-1)!=-100
    if valid.sum().item()==0:
        return None

    return losses[valid].mean().item()

@torch.no_grad()
def score_one(context, target, model, tokenizer, device):
    target_ids=encode(tokenizer, target, max_tokens=MAX_TARGET_TOKENS)
    if len(target_ids)<2:
        return None

    eos=tokenizer.eos_token_id
    if eos is None:
        eos=tokenizer.pad_token_id
    if eos is None:
        eos=50256

    uncond_ids=[eos]+target_ids
    uncond_labels=[-100]+target_ids

    max_context_tokens=MAX_LENGTH-len(target_ids)
    if max_context_tokens<8:
        target_ids=target_ids[-(MAX_LENGTH-8):]
        max_context_tokens=MAX_LENGTH-len(target_ids)

    context_ids=encode(tokenizer, context, max_tokens=max_context_tokens)
    if len(context_ids)==0:
        context_ids=[eos]

    cond_ids=context_ids+target_ids
    cond_labels=[-100]*len(context_ids)+target_ids

    uncond_ids=uncond_ids[-MAX_LENGTH:]
    uncond_labels=uncond_labels[-MAX_LENGTH:]
    cond_ids=cond_ids[-MAX_LENGTH:]
    cond_labels=cond_labels[-MAX_LENGTH:]

    uncond_loss=masked_lm_loss(model, uncond_ids, uncond_labels, device)
    cond_loss=masked_lm_loss(model, cond_ids, cond_labels, device)

    if uncond_loss is None or cond_loss is None:
        return None

    delta_loss=uncond_loss-cond_loss
    gain_ratio=delta_loss/uncond_loss if uncond_loss>0 else 0.0

    return {
        "uncond_loss":float(uncond_loss),
        "cond_loss":float(cond_loss),
        "delta_loss":float(delta_loss),
        "gain_ratio":float(gain_ratio),
        "delta_anomaly_score":float(-delta_loss),
        "gain_anomaly_score":float(-gain_ratio),
    }

def score_cached(context, target, model, tokenizer, device, cache):
    key=sha_key(context, target)
    if key in cache:
        return cache[key]
    out=score_one(context, target, model, tokenizer, device)
    if out is not None:
        cache[key]=out
    return out

def tpr_at_fpr(y_true, scores, level):
    fpr,tpr,_=roc_curve(y_true,scores)
    valid=np.where(fpr<=level)[0]
    if len(valid)==0:
        return 0.0
    return float(np.max(tpr[valid]))

def best_f1(y_true, scores):
    p,r,_=precision_recall_curve(y_true,scores)
    f1=2*p*r/np.clip(p+r,1e-12,None)
    return float(np.max(f1))

def metrics(records, score_key):
    y=np.array([int(r["label"]) for r in records])
    s=np.array([float(r[score_key]) for r in records])
    return {
        "roc_auc":float(roc_auc_score(y,s)),
        "pr_auc":float(average_precision_score(y,s)),
        "tpr_at_fpr_5":tpr_at_fpr(y,s,0.05),
        "tpr_at_fpr_10":tpr_at_fpr(y,s,0.10),
        "best_f1":best_f1(y,s),
    }

def parse_file(path):
    base=os.path.basename(path)

    if base.startswith("boundary_nfcorpus"):
        dataset="NFCorpus"
    elif base.startswith("boundary_trec"):
        dataset="TREC-COVID"
    elif base.startswith("boundary_nq"):
        dataset="Natural Questions"
    else:
        dataset="unknown"

    return dataset

def build_variant_rows(data, variant):
    rows=[]
    for r in data:
        rr=dict(r)
        if variant=="original_matched" and int(rr["label"])==1 and "original_target" in rr:
            rr["target"]=rr["original_target"]
            rr["target_text"]=rr["original_target"]
            rr["full_text"]=" ".join(((rr.get("context") or "") + " " + rr["target"]).split())
        rows.append(rr)
    return rows

def main():
    paths=sorted(glob.glob(IN_GLOB))
    print("input files:", len(paths))
    for p in paths:
        print(" -", p)

    if not paths:
        raise RuntimeError("No cleaned deepseek A2 adaptive attack files found.")

    device="cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    if CACHE_PATH.exists():
        cache=json.load(open(CACHE_PATH,"r",encoding="utf-8"))
    else:
        cache={}

    all_scores=[]
    all_metrics=[]

    for p in paths:
        dataset=parse_file(p)
        data=json.load(open(p,"r",encoding="utf-8"))

        for variant in ["original_matched", "attacked"]:
            rows=build_variant_rows(data, variant)

            scored=[]
            failed=[]

            print("\nEVAL", dataset, "a2", variant, "n=", len(rows))

            for i,r in enumerate(rows):
                context=r.get("context") or r.get("context_text") or ""
                target=r.get("target") or r.get("target_text") or ""

                out=score_cached(context, target, model, tokenizer, device, cache)

                if out is None:
                    failed.append({"index":i,"id":r.get("id")})
                    continue

                rec={
                    "dataset":dataset,
                    "attack":"a2",
                    "variant":variant,
                    "detector_model":"Qwen2.5-7B",
                    "attack_model":"deepseek-chat",
                    "prompt_version":"v2",
                    "record_id":r.get("id"),
                    "label":int(r["label"]),
                    "group_id":r.get("group_id"),
                    **out,
                }
                scored.append(rec)
                all_scores.append(rec)

                if (i+1)%100==0:
                    print(dataset, "a2", variant, "scored", i+1, "/", len(rows), "cache", len(cache), flush=True)

            json.dump(cache, open(CACHE_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

            y=np.array([int(r["label"]) for r in scored])
            gr=np.array([float(r["gain_ratio"]) for r in scored])
            dl=np.array([float(r["delta_loss"]) for r in scored])

            for method, key in [("GainRatio","gain_anomaly_score"),("DeltaLoss","delta_anomaly_score")]:
                m=metrics(scored,key)
                row={
                    "dataset":dataset,
                    "attack":"a2",
                    "variant":variant,
                    "method":method,
                    "detector_model":"Qwen2.5-7B",
                    "attack_model":"deepseek-chat",
                    "prompt_version":"v2",
                    "n":len(scored),
                    "positive_inject":int(y.sum()),
                    "negative_clean":int((y==0).sum()),
                    "n_failed":len(failed),
                    **m,
                    "gain_ratio_mean_inject":float(np.mean(gr[y==1])),
                    "gain_ratio_mean_clean":float(np.mean(gr[y==0])),
                    "delta_loss_mean_inject":float(np.mean(dl[y==1])),
                    "delta_loss_mean_clean":float(np.mean(dl[y==0])),
                }
                all_metrics.append(row)

            print(dataset, "a2", variant, "GainRatio", metrics(scored,"gain_anomaly_score"))

    json.dump(all_scores, open(SCORES_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(all_metrics, open(METRICS_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    import csv
    headers=[
        "dataset","attack","variant","method","detector_model","attack_model","prompt_version",
        "n","positive_inject","negative_clean","n_failed",
        "roc_auc","pr_auc","tpr_at_fpr_5","tpr_at_fpr_10","best_f1",
        "gain_ratio_mean_inject","gain_ratio_mean_clean",
        "delta_loss_mean_inject","delta_loss_mean_clean"
    ]
    with open(CSV_PATH,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(all_metrics)

    with open(MD_PATH,"w",encoding="utf-8") as f:
        f.write("# DeepSeek-chat A2 Adaptive Attack: Clean Matched Detection\n\n")
        f.write("| Dataset | Variant | N | ROC-AUC | PR-AUC | TPR@FPR=5% | TPR@FPR=10% | Best F1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")

        for ds in ["NFCorpus","TREC-COVID","Natural Questions"]:
            b=EMBEDDING_BASELINE[ds]
            f.write(f'| {ds} | Contriever+Cosine baseline | {b["n"]} | {b["roc_auc"]:.4f} | {b["pr_auc"]:.4f} | {b["tpr_at_fpr_5"]:.4f} | {b["tpr_at_fpr_10"]:.4f} | {b["best_f1"]:.4f} |\n')
            for variant in ["original_matched","attacked"]:
                rr=[r for r in all_metrics if r["dataset"]==ds and r["variant"]==variant and r["method"]=="GainRatio"]
                if rr:
                    r=rr[0]
                    f.write(f'| {ds} | deepseek-chat A2 {variant} | {r["n"]} | {r["roc_auc"]:.4f} | {r["pr_auc"]:.4f} | {r["tpr_at_fpr_5"]:.4f} | {r["tpr_at_fpr_10"]:.4f} | {r["best_f1"]:.4f} |\n')

    print("saved scores:", SCORES_PATH)
    print("saved metrics:", METRICS_PATH)
    print("saved cache:", CACHE_PATH)
    print("saved csv:", CSV_PATH)
    print("saved md:", MD_PATH)

if __name__=="__main__":
    main()

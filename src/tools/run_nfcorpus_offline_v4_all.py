import os
import re
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import chromadb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contriever_src.contriever import Contriever


SEED = 12
random.seed(SEED)
np.random.seed(SEED)

CHROMA_PATH = "/root/autodl-tmp/ragwm_storage/chromadb_db"
COLLECTION = "nfcorpus_contriever_cosine"

CONTRIEVER_PATH = "/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever"
QWEN_PATH = "/root/autodl-tmp/ragwm_storage/local_models/Qwen2.5-7B"

BASE_OUT = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff")
WM_OUT = Path("/root/autodl-tmp/ragwm_storage/output/wm_generate/nfcorpus/10")
LOG_NOTE = "offline_template_v4_due_to_chat_completion_api_500"

INJECT_FILE = WM_OUT / "wmuint_inject.json"
INJECT_AUDIT = WM_OUT / "wmuint_inject_offline_v4_audit.json"

BOUNDARY_FILE = BASE_OUT / "boundary_nfcorpus_k3_v4.json"
BOUNDARY_AUDIT = BASE_OUT / "boundary_nfcorpus_k3_v4_audit.json"

BASELINE_SCORES = BASE_OUT / "nfcorpus_contriever_cosine_scores_v4.json"
BASELINE_METRICS = BASE_OUT / "nfcorpus_contriever_cosine_metrics_v4.json"

QWEN_SCORES = BASE_OUT / "nfcorpus_delta_loss_scores_qwen2_5_7b_v4.json"
QWEN_METRICS = BASE_OUT / "nfcorpus_delta_loss_metrics_qwen2_5_7b_v4.json"
QWEN_FAILED = BASE_OUT / "nfcorpus_delta_loss_failed_qwen2_5_7b_v4.json"

EXT_JSON = BASE_OUT / "nfcorpus_detection_metrics_extended_v4.json"
EXT_MD = BASE_OUT / "nfcorpus_detection_metrics_extended_v4.md"
EXT_CSV = BASE_OUT / "nfcorpus_detection_metrics_extended_v4.csv"

FINAL_TABLE_MD = BASE_OUT / "main_table_gainratio_nf_trec_v4.md"
FINAL_TABLE_JSON = BASE_OUT / "main_table_gainratio_nf_trec_v4.json"

N_UNITS = 50
DOCS_PER_UNIT = 2
PARAPHRASES_PER_UNIT = 5
MAX_LENGTH = 512
N_BOOT = 1000

RELATIONS = [
    "ASSOCIATED_WITH", "INFLUENCES", "AFFECTS", "RELATED_TO", "MODULATES",
    "CONTRIBUTES_TO", "INTERACTS_WITH", "PREDICTS", "REDUCES", "INCREASES",
]

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "has", "have",
    "study", "studies", "analysis", "effect", "effects", "result", "results", "using",
    "patients", "patient", "health", "disease", "risk", "treatment", "clinical"
}


def normalize(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def split_sentences(text):
    text = normalize(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION)


def get_docs():
    col = get_collection()
    n = col.count()
    ids, docs, metas = [], [], []
    step = 500

    for offset in range(0, n, step):
        got = col.get(limit=step, offset=offset, include=["documents", "metadatas"])
        ids.extend(got["ids"])
        docs.extend(got["documents"])
        metas.extend(got["metadatas"])

    out = []
    for i, d, m in zip(ids, docs, metas):
        d = normalize(d)
        if len(d) >= 150:
            out.append({"id": str(i), "doc": d, "meta": m or {}})
    return out


def candidate_terms(text):
    text = normalize(text)
    chunks = []

    first = text.split("\n")[0] if "\n" in text else text[:220]
    chunks.append(first)

    caps = re.findall(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4}\b", text)
    chunks.extend(caps)

    phrases = re.findall(r"\b[a-zA-Z][a-zA-Z\-]+(?:\s+[a-zA-Z][a-zA-Z\-]+){1,3}\b", text)
    chunks.extend(phrases[:100])

    terms, seen = [], set()
    for c in chunks:
        c = normalize(c)
        words = c.split()
        low = c.lower()

        if not (1 <= len(words) <= 5):
            continue
        if len(c) < 4 or len(c) > 90:
            continue
        if low in STOPWORDS:
            continue
        if len(words) == 1 and words[0].lower() in STOPWORDS:
            continue
        if low not in seen:
            seen.add(low)
            terms.append(c)

    return terms


def make_wt_texts(e1, e2, rel):
    rel_low = rel.replace("_", " ").lower()
    return [
        f"{e1} is {rel_low} {e2} in this biomedical context.",
        f"The relationship between {e1} and {e2} can be described as {rel_low}.",
        f"In biomedical evidence, {e1} is considered {rel_low} {e2}.",
        f"{e2} has a documented connection with {e1} through the relation {rel_low}.",
        f"Research descriptions may characterize {e1} as {rel_low} {e2}.",
    ]


def make_offline_inject_units():
    print("\n[1/6] Make offline NFCorpus inject units")

    WM_OUT.mkdir(parents=True, exist_ok=True)
    docs = get_docs()
    print("docs:", len(docs))

    rng = random.Random(SEED)
    rng.shuffle(docs)

    records = []
    used_pairs = set()
    doc_cursor = 0

    for unit_idx in range(N_UNITS):
        selected_docs = []
        selected_doc_ids = []

        while len(selected_docs) < DOCS_PER_UNIT and doc_cursor < len(docs):
            d = docs[doc_cursor]
            doc_cursor += 1
            terms = candidate_terms(d["doc"])
            if len(terms) >= 2:
                selected_docs.append(d)
                selected_doc_ids.append(d["id"])

        if len(selected_docs) < DOCS_PER_UNIT:
            break

        terms = []
        for d in selected_docs:
            terms.extend(candidate_terms(d["doc"])[:25])
        terms = list(dict.fromkeys(terms))

        if len(terms) < 2:
            continue

        e1, e2 = rng.sample(terms, 2)
        for _ in range(50):
            if e1.lower() != e2.lower() and (e1.lower(), e2.lower()) not in used_pairs:
                break
            e1, e2 = rng.sample(terms, 2)

        used_pairs.add((e1.lower(), e2.lower()))
        rel = RELATIONS[unit_idx % len(RELATIONS)]
        wmunit = [e1, e2, rel]
        wt_items = [[t, 1, 1] for t in make_wt_texts(e1, e2, rel)[:PARAPHRASES_PER_UNIT]]

        records.append([wmunit, selected_doc_ids, wt_items])

    audit = {
        "construction": LOG_NOTE,
        "dataset": "nfcorpus",
        "chroma_path": CHROMA_PATH,
        "collection": COLLECTION,
        "n_units": len(records),
        "docs_per_unit": DOCS_PER_UNIT,
        "paraphrases_per_unit": PARAPHRASES_PER_UNIT,
        "estimated_inject_records": len(records) * DOCS_PER_UNIT * PARAPHRASES_PER_UNIT,
        "output": str(INJECT_FILE),
    }

    json.dump(records, open(INJECT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audit, open(INJECT_AUDIT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("preview:", records[:1])


def remove_wt(text, wt_set):
    out = text or ""
    for wt in sorted(wt_set, key=len, reverse=True):
        wt = normalize(wt)
        if wt and wt in out:
            out = out.replace(wt, " ")
    return normalize(out)


def load_inject_records():
    data = json.load(open(INJECT_FILE, "r", encoding="utf-8"))
    records, all_wt = [], set()

    for unit_idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) < 3:
            continue
        wmunit, doc_ids, wt_items = item[0], item[1], item[2]
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        for wt_item in wt_items:
            if isinstance(wt_item, str):
                wt, ok = wt_item, True
            elif isinstance(wt_item, list) and wt_item:
                wt = wt_item[0]
                ok = True
                if len(wt_item) >= 3:
                    ok = bool(wt_item[1]) and bool(wt_item[2])
            else:
                continue

            wt = normalize(wt)
            if len(wt) < 10 or not ok:
                continue

            all_wt.add(wt)
            for doc_id in doc_ids:
                records.append({
                    "unit_index": unit_idx,
                    "doc_id": str(doc_id),
                    "wmunit": wmunit,
                    "target": wt,
                })

    return records, all_wt


def build_boundary():
    print("\n[2/6] Build NFCorpus boundary dataset")

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    inject_records, all_wt = load_inject_records()
    docs = get_docs()
    doc_map = {d["id"]: {"document": d["doc"], "metadata": d["meta"]} for d in docs}

    inject, seen = [], set()

    for idx, r in enumerate(inject_records):
        doc_id = r["doc_id"]
        target = normalize(r["target"])

        if doc_id not in doc_map:
            continue

        context = remove_wt(doc_map[doc_id]["document"], all_wt)
        if len(context) < 30:
            continue
        if target in context:
            continue

        context = context[-900:].strip()
        full_text = normalize(context + " " + target)
        key = (context, target, 1)
        if key in seen:
            continue
        seen.add(key)

        inject.append({
            "id": f"inject_{r['unit_index']}_{doc_id}_{idx}",
            "label": 1,
            "dataset": "nfcorpus",
            "kind": "inject_boundary",
            "unit_index": r["unit_index"],
            "doc_id": doc_id,
            "metadata": doc_map[doc_id]["metadata"],
            "wmunit": r["wmunit"],
            "context": context,
            "target": target,
            "full_text": full_text,
        })

    rng = random.Random(SEED)
    clean_candidates = []

    for doc_id, item in doc_map.items():
        clean_doc = remove_wt(item["document"], all_wt)
        sents = split_sentences(clean_doc)
        if len(sents) < 2:
            continue

        for i in range(1, len(sents)):
            target = normalize(sents[i])
            context = normalize(" ".join(sents[:i]))[-900:].strip()

            if len(context) < 30 or len(target) < 20:
                continue
            if target in context:
                continue
            if any(wt in context for wt in all_wt):
                continue
            if any(wt in target for wt in all_wt):
                continue

            clean_candidates.append({
                "id": f"clean_{doc_id}_{i}",
                "label": 0,
                "dataset": "nfcorpus",
                "kind": "clean_boundary",
                "doc_id": doc_id,
                "metadata": item["metadata"],
                "context": context,
                "target": target,
                "full_text": normalize(context + " " + target),
            })

    rng.shuffle(clean_candidates)

    clean, seen_clean = [], set()
    for r in clean_candidates:
        key = (r["context"], r["target"], 0)
        if key in seen_clean:
            continue
        seen_clean.add(key)
        clean.append(r)
        if len(clean) >= len(inject):
            break

    n = min(len(inject), len(clean))
    inject, clean = inject[:n], clean[:n]
    samples = inject + clean
    rng.shuffle(samples)

    audit = {
        "construction": LOG_NOTE,
        "chroma_path": CHROMA_PATH,
        "collection": COLLECTION,
        "inject_file": str(INJECT_FILE),
        "total": len(samples),
        "label_1": sum(1 for r in samples if r["label"] == 1),
        "label_0": sum(1 for r in samples if r["label"] == 0),
        "all_watermark_texts": len(all_wt),
        "target_in_context": sum(1 for r in samples if r["target"] in r["context"]),
        "context_contains_any_wt": sum(1 for r in samples if any(wt in r["context"] for wt in all_wt)),
        "clean_target_contains_any_wt": sum(1 for r in samples if r["label"] == 0 and any(wt in r["target"] for wt in all_wt)),
        "duplicate_full_text": len(samples) - len(set(r["full_text"] for r in samples)),
    }

    json.dump(samples, open(BOUNDARY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audit, open(BOUNDARY_AUDIT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("preview:", samples[:2])

    if audit["label_1"] != audit["label_0"]:
        raise RuntimeError("unbalanced labels")
    for k in ["target_in_context", "context_contains_any_wt", "clean_target_contains_any_wt", "duplicate_full_text"]:
        if audit[k] != 0:
            raise RuntimeError(f"audit failed: {k}={audit[k]}")


def mean_pooling(token_embeddings, mask):
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.no_grad()
def encode_contriever(model, tokenizer, texts, device, batch=64):
    arr = []
    for i in tqdm(range(0, len(texts), batch)):
        inputs = tokenizer(
            texts[i:i+batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        if torch.is_tensor(outputs):
            emb = outputs
        elif hasattr(outputs, "last_hidden_state"):
            emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        elif isinstance(outputs, (tuple, list)):
            emb = outputs[0]
            if emb.dim() == 3:
                emb = mean_pooling(emb, inputs["attention_mask"])
        else:
            raise TypeError(type(outputs))

        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if emb.dim() != 2:
            raise RuntimeError(f"bad embedding shape: {tuple(emb.shape)}")

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        arr.append(emb.cpu().numpy())
    return np.vstack(arr)


def run_embedding_baseline():
    print("\n[3/6] Run NFCorpus Contriever + Cosine baseline")

    data = json.load(open(BOUNDARY_FILE, "r", encoding="utf-8"))
    contexts = [r["context"] for r in data]
    targets = [r["target"] for r in data]
    labels = np.array([int(r["label"]) for r in data], dtype=np.int32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_PATH, local_files_only=True)
    model = Contriever.from_pretrained(CONTRIEVER_PATH, local_files_only=True).to(device)
    model.eval()

    c_emb = encode_contriever(model, tokenizer, contexts, device)
    t_emb = encode_contriever(model, tokenizer, targets, device)

    cosine = np.sum(c_emb * t_emb, axis=1)
    anomaly = -cosine

    scores = []
    for r, cos, anom in zip(data, cosine, anomaly):
        item = dict(r)
        item["cosine"] = float(cos)
        item["embedding_anomaly_score"] = float(anom)
        scores.append(item)

    metrics = {
        "dataset": "nfcorpus",
        "method": "Contriever + Cosine",
        "boundary_file": str(BOUNDARY_FILE),
        "n": int(len(labels)),
        "positive_inject": int(labels.sum()),
        "negative_clean": int((1-labels).sum()),
        "roc_auc": float(roc_auc_score(labels, anomaly)),
        "pr_auc": float(average_precision_score(labels, anomaly)),
        "cosine_mean_inject": float(cosine[labels == 1].mean()),
        "cosine_mean_clean": float(cosine[labels == 0].mean()),
    }

    json.dump(scores, open(BASELINE_SCORES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics, open(BASELINE_METRICS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def target_nll(model, tokenizer, context, target, device):
    context = (context or "").strip()
    target = (target or "").strip()

    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(target_ids) < 1:
        raise ValueError("empty target ids")

    t_inputs = tokenizer(target, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    t_ids = t_inputs["input_ids"].to(device)
    t_attn = t_inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=t_ids, attention_mask=t_attn, labels=t_ids.clone())
    loss_t = float(out.loss.detach().cpu())

    joined = context + "\n" + target
    ct_inputs = tokenizer(joined, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    ct_ids = ct_inputs["input_ids"].to(device)
    ct_attn = ct_inputs["attention_mask"].to(device)

    labels = torch.full_like(ct_ids, -100)
    n_t = min(len(target_ids), ct_ids.shape[1])
    labels[:, -n_t:] = ct_ids[:, -n_t:]

    with torch.no_grad():
        out = model(input_ids=ct_ids, attention_mask=ct_attn, labels=labels)
    loss_ct = float(out.loss.detach().cpu())

    return loss_t, loss_ct


def metric_from_scores(rows, key):
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    scores = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "score_mean_inject": float(scores[labels == 1].mean()),
        "score_mean_clean": float(scores[labels == 0].mean()),
    }


def run_qwen_gainratio():
    print("\n[4/6] Run NFCorpus Qwen2.5-7B DeltaLoss/GainRatio")

    data = json.load(open(BOUNDARY_FILE, "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    scores, failed = [], []

    for i, r in enumerate(tqdm(data)):
        try:
            loss_t, loss_ct = target_nll(model, tokenizer, r["context"], r["target"], device)
            delta = loss_t - loss_ct
            gain = delta / max(loss_t, 1e-12)

            item = dict(r)
            item["loss_target"] = loss_t
            item["loss_context_target"] = loss_ct
            item["delta_loss"] = delta
            item["gain_ratio"] = gain
            item["delta_anomaly_score"] = -delta
            item["gain_anomaly_score"] = -gain
            scores.append(item)
        except Exception as e:
            failed.append({"index": i, "id": r.get("id"), "error": repr(e)})

    metrics = {
        "dataset": "nfcorpus",
        "boundary_file": str(BOUNDARY_FILE),
        "model_repo_id": "Qwen/Qwen2.5-7B",
        "model_path": QWEN_PATH,
        "tag": "qwen2_5_7b_v4_offline",
        "max_length": MAX_LENGTH,
        "n_input": len(data),
        "n_scored": len(scores),
        "n_failed": len(failed),
        "positive_inject": sum(1 for r in scores if int(r["label"]) == 1),
        "negative_clean": sum(1 for r in scores if int(r["label"]) == 0),
        "record_delta_loss": metric_from_scores(scores, "delta_anomaly_score"),
        "record_gain_ratio": metric_from_scores(scores, "gain_anomaly_score"),
        "delta_loss_mean_inject": float(np.mean([r["delta_loss"] for r in scores if int(r["label"]) == 1])),
        "delta_loss_mean_clean": float(np.mean([r["delta_loss"] for r in scores if int(r["label"]) == 0])),
        "gain_ratio_mean_inject": float(np.mean([r["gain_ratio"] for r in scores if int(r["label"]) == 1])),
        "gain_ratio_mean_clean": float(np.mean([r["gain_ratio"] for r in scores if int(r["label"]) == 0])),
    }

    json.dump(scores, open(QWEN_SCORES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics, open(QWEN_METRICS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(failed, open(QWEN_FAILED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics["n_failed"] != 0:
        raise RuntimeError("Qwen scoring has failed samples")


def tpr_at_fpr(y, s, target):
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.where(fpr <= target)[0]
    if len(idx) == 0:
        return 0.0, 0.0, float("inf")
    best = idx[np.argmax(tpr[idx])]
    return float(tpr[best]), float(fpr[best]), float(thr[best])


def best_f1(y, s):
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1))
    threshold = float(thr[i]) if i < len(thr) else float(np.min(s) - 1e-12)
    pred = (s >= threshold).astype(np.int32)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    return {
        "best_f1": float(f1[i]),
        "precision": float(p[i]),
        "recall": float(r[i]),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def ci(vals):
    vals = np.array(vals, dtype=float)
    return {
        "low": float(np.percentile(vals, 2.5)),
        "high": float(np.percentile(vals, 97.5)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def bootstrap(y, s):
    rng = np.random.default_rng(SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    rocs, prs, tpr5s, tpr10s, f1s = [], [], [], [], []

    for _ in range(N_BOOT):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        rng.shuffle(idx)
        yy, ss = y[idx], s[idx]
        rocs.append(roc_auc_score(yy, ss))
        prs.append(average_precision_score(yy, ss))
        tpr5s.append(tpr_at_fpr(yy, ss, 0.05)[0])
        tpr10s.append(tpr_at_fpr(yy, ss, 0.10)[0])
        f1s.append(best_f1(yy, ss)["best_f1"])

    return {
        "roc_auc_ci95": ci(rocs),
        "pr_auc_ci95": ci(prs),
        "tpr_at_fpr_5_ci95": ci(tpr5s),
        "tpr_at_fpr_10_ci95": ci(tpr10s),
        "best_f1_ci95": ci(f1s),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }


def extended_metrics_one(dataset, method, model, score_file, score_key):
    data = json.load(open(score_file, "r", encoding="utf-8"))
    y = np.array([int(r["label"]) for r in data], dtype=np.int32)
    s = np.array([float(r[score_key]) for r in data], dtype=np.float64)

    roc = float(roc_auc_score(y, s))
    pr = float(average_precision_score(y, s))
    tpr5, fpr5, thr5 = tpr_at_fpr(y, s, 0.05)
    tpr10, fpr10, thr10 = tpr_at_fpr(y, s, 0.10)
    f1 = best_f1(y, s)
    boot = bootstrap(y, s)

    return {
        "dataset": dataset,
        "method": method,
        "model": model,
        "score_file": str(score_file),
        "score_key": score_key,
        "n": int(len(y)),
        "positive_inject": int(y.sum()),
        "negative_clean": int((1-y).sum()),
        "roc_auc": roc,
        "pr_auc": pr,
        "tpr_at_fpr_5": {"tpr": tpr5, "fpr": fpr5, "threshold": thr5},
        "tpr_at_fpr_10": {"tpr": tpr10, "fpr": fpr10, "threshold": thr10},
        "best_f1": f1,
        "bootstrap": boot,
    }


def fmt(main, c):
    return f"{main:.4f} [{c['low']:.4f}, {c['high']:.4f}]"


def run_extended_metrics():
    print("\n[5/6] Compute NFCorpus extended metrics")

    results = [
        extended_metrics_one("NFCorpus", "Contriever + Cosine", "Contriever", BASELINE_SCORES, "embedding_anomaly_score"),
        extended_metrics_one("NFCorpus", "GainRatio", "Qwen2.5-7B", QWEN_SCORES, "gain_anomaly_score"),
    ]

    json.dump(results, open(EXT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with open(EXT_MD, "w", encoding="utf-8") as f:
        f.write("# NFCorpus v4 Extended Detection Metrics\n\n")
        f.write("| Dataset | Method | Model | ROC-AUC 95% CI | PR-AUC 95% CI | TPR@FPR=5% | TPR@FPR=10% | Best F1 | N |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['dataset']} | {r['method']} | {r['model']} | "
                f"{fmt(r['roc_auc'], r['bootstrap']['roc_auc_ci95'])} | "
                f"{fmt(r['pr_auc'], r['bootstrap']['pr_auc_ci95'])} | "
                f"{r['tpr_at_fpr_5']['tpr']:.4f} | "
                f"{r['tpr_at_fpr_10']['tpr']:.4f} | "
                f"{r['best_f1']['best_f1']:.4f} | "
                f"{r['n']} |\n"
            )

    with open(EXT_CSV, "w", encoding="utf-8", newline="") as f:
        fields = ["dataset", "method", "model", "n", "positive_inject", "negative_clean", "roc_auc", "pr_auc", "tpr_at_fpr_5", "tpr_at_fpr_10", "best_f1", "score_file", "score_key"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "dataset": r["dataset"],
                "method": r["method"],
                "model": r["model"],
                "n": r["n"],
                "positive_inject": r["positive_inject"],
                "negative_clean": r["negative_clean"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
                "tpr_at_fpr_5": r["tpr_at_fpr_5"]["tpr"],
                "tpr_at_fpr_10": r["tpr_at_fpr_10"]["tpr"],
                "best_f1": r["best_f1"]["best_f1"],
                "score_file": r["score_file"],
                "score_key": r["score_key"],
            })

    print(open(EXT_MD, "r", encoding="utf-8").read())


def pick(rows, dataset, method, model):
    for r in rows:
        if r["dataset"].lower() == dataset.lower() and r["method"] == method and r["model"] == model:
            return r
    raise KeyError((dataset, method, model))


def make_final_table():
    print("\n[6/6] Make final NF + TREC main table")

    nf = json.load(open(EXT_JSON, "r", encoding="utf-8"))
    trec_path = BASE_OUT / "trec_detection_metrics_extended.json"
    if not trec_path.exists():
        raise FileNotFoundError(f"Missing TREC extended metrics: {trec_path}")

    trec = json.load(open(trec_path, "r", encoding="utf-8"))

    rows = [
        pick(nf, "NFCorpus", "Contriever + Cosine", "Contriever"),
        pick(nf, "NFCorpus", "GainRatio", "Qwen2.5-7B"),
        pick(trec, "TREC-COVID", "Contriever + Cosine", "Contriever"),
        pick(trec, "TREC-COVID", "GainRatio", "Qwen2.5-7B"),
    ]

    json.dump(rows, open(FINAL_TABLE_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with open(FINAL_TABLE_MD, "w", encoding="utf-8") as f:
        f.write("| Dataset | Method | Model | N | Inject/Clean | ROC-AUC | PR-AUC | TPR@FPR=5% | Best F1 |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['method']} | {r['model']} | "
                f"{r['n']} | {r['positive_inject']} / {r['negative_clean']} | "
                f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
                f"{r['tpr_at_fpr_5']['tpr']:.4f} | "
                f"{r['best_f1']['best_f1']:.4f} |\n"
            )

    print(open(FINAL_TABLE_MD, "r", encoding="utf-8").read())


def preflight():
    print("REPO_ROOT:", REPO_ROOT)
    print("CHROMA_PATH:", CHROMA_PATH)
    print("COLLECTION:", COLLECTION)
    print("CONTRIEVER_PATH:", CONTRIEVER_PATH)
    print("QWEN_PATH:", QWEN_PATH)

    if not Path(CONTRIEVER_PATH, "config.json").exists():
        raise FileNotFoundError(CONTRIEVER_PATH)
    if not Path(QWEN_PATH, "config.json").exists():
        raise FileNotFoundError(QWEN_PATH)

    col = get_collection()
    print("collection count:", col.count())
    if col.count() <= 0:
        raise RuntimeError("empty NFCorpus collection")


def main():
    preflight()
    make_offline_inject_units()
    build_boundary()
    run_embedding_baseline()
    run_qwen_gainratio()
    run_extended_metrics()
    make_final_table()

    print("\nDONE.")
    print("Boundary:", BOUNDARY_FILE)
    print("NFCorpus extended metrics:", EXT_MD)
    print("Final main table:", FINAL_TABLE_MD)


if __name__ == "__main__":
    main()

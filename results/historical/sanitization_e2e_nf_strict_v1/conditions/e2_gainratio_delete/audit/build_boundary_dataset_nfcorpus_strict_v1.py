import json
import re
import random
from pathlib import Path
from collections import Counter
import chromadb

SEED = 2026
random.seed(SEED)

DB_PATH = "/root/autodl-tmp/ragwm_storage/chromadb_db"
COLLECTION = "nfcorpus_contriever_cosine"

INJECT_PATH = Path("/root/autodl-tmp/ragwm_storage/output/wm_generate/nfcorpus/10/wmuint_inject.json")
OUT_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nfcorpus_strict_v1.json")
AUDIT_PATH = Path("/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nfcorpus_strict_v1_audit.json")

MAX_CONTEXT_CHARS = 1200
MIN_TARGET_CHARS = 25

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def split_sentences(text: str):
    text = norm_space(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) >= MIN_TARGET_CHARS]
    return parts

def tail_context(text: str, max_chars: int = MAX_CONTEXT_CHARS):
    text = norm_space(text)
    if len(text) <= max_chars:
        return text
    cut = text[-max_chars:]
    # avoid starting in the middle of a word/sentence too badly
    first_space = cut.find(" ")
    if first_space > 0:
        cut = cut[first_space+1:]
    return cut.strip()

def load_inject_units():
    data = json.load(open(INJECT_PATH, "r", encoding="utf-8"))
    units = []
    all_wt = []
    for unit_idx, item in enumerate(data):
        tup = item[0]
        doc_ids = item[1]
        candidates = item[2]
        for cand_idx, cand in enumerate(candidates):
            wt = norm_space(cand[0])
            if not wt:
                continue
            units.append({
                "unit_idx": unit_idx,
                "candidate_idx": cand_idx,
                "tuple": tup,
                "doc_ids": doc_ids,
                "watermark_text": wt,
                "wd1": cand[1] if len(cand) > 1 else None,
                "wd2": cand[2] if len(cand) > 2 else None,
            })
            all_wt.append(wt)
    return units, all_wt

def main():
    assert INJECT_PATH.exists(), f"missing {INJECT_PATH}"

    units, all_wt = load_inject_units()
    print("inject units:", len(json.load(open(INJECT_PATH, "r", encoding="utf-8"))))
    print("watermark candidates:", len(units))

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection(COLLECTION)
    print("collection count:", col.count())

    # get all docs
    all_data = col.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    by_source_id = {}
    changed_docs = []
    clean_docs = []

    for chroma_id, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        source_id = meta.get("id", chroma_id)
        rec = {
            "chroma_id": chroma_id,
            "source_doc_id": source_id,
            "document": doc or "",
            "metadata": meta,
            "change": bool(meta.get("change", False)),
        }
        by_source_id.setdefault(source_id, []).append(rec)
        if rec["change"]:
            changed_docs.append(rec)
        else:
            clean_docs.append(rec)

    print("changed docs:", len(changed_docs))
    print("clean docs:", len(clean_docs))

    # Build a quick searchable list of changed documents.
    changed_by_docid = {}
    for rec in changed_docs:
        changed_by_docid.setdefault(rec["source_doc_id"], []).append(rec)

    inject_records = []
    missing = []

    for idx, u in enumerate(units):
        wt = u["watermark_text"]
        candidate_recs = []

        # Prefer the doc_ids recorded in wmuint_inject.json
        for did in u["doc_ids"]:
            candidate_recs.extend(changed_by_docid.get(did, []))

        # Fallback: scan all changed docs
        if not candidate_recs:
            candidate_recs = changed_docs

        found = False
        for rec in candidate_recs:
            doc = rec["document"]
            pos = doc.find(wt)
            if pos < 0:
                continue

            # To avoid previous appended WT leaking into context, cut context at earliest WT occurrence in this doc.
            wt_positions = [doc.find(x) for x in all_wt if x and doc.find(x) >= 0]
            earliest_wt_pos = min(wt_positions) if wt_positions else pos
            original_context = doc[:earliest_wt_pos]
            context = tail_context(original_context)

            if not context:
                continue
            if wt in context:
                continue

            full_text = norm_space(context + " " + wt)
            inject_records.append({
                "id": f"nfcorpus_strict_inject_{len(inject_records)}",
                "dataset": "nfcorpus",
                "source": "strict_inject",
                "label": 1,
                "boundary_type": "inject",
                "group_id": "||".join(map(str, u["tuple"])),
                "tuple": u["tuple"],
                "source_doc_id": rec["source_doc_id"],
                "chroma_id": rec["chroma_id"],
                "unit_idx": u["unit_idx"],
                "candidate_idx": u["candidate_idx"],
                "context": context,
                "target": wt,
                "context_text": context,
                "target_text": wt,
                "watermark_text": wt,
                "full_text": full_text,
                "wd1": u["wd1"],
                "wd2": u["wd2"],
            })
            found = True
            break

        if not found:
            missing.append(u)

    print("inject boundary records:", len(inject_records))
    print("missing watermark candidates in changed docs:", len(missing))

    # Build clean boundary records from unchanged docs.
    clean_candidates = []
    for rec in clean_docs:
        text = rec["document"]
        sents = split_sentences(text)
        if len(sents) < 2:
            continue

        # choose a later sentence as target; context is preceding text
        for j in range(1, min(len(sents), 5)):
            target = sents[j]
            if any(wt in target for wt in all_wt):
                continue
            context = tail_context(" ".join(sents[:j]))
            if not context or target in context:
                continue
            full_text = norm_space(context + " " + target)
            clean_candidates.append({
                "id": f"nfcorpus_strict_clean_candidate_{len(clean_candidates)}",
                "dataset": "nfcorpus",
                "source": "clean",
                "label": 0,
                "boundary_type": "clean",
                "group_id": rec["source_doc_id"],
                "tuple": None,
                "source_doc_id": rec["source_doc_id"],
                "chroma_id": rec["chroma_id"],
                "context": context,
                "target": target,
                "context_text": context,
                "target_text": target,
                "watermark_text": None,
                "full_text": full_text,
            })

    print("clean candidates:", len(clean_candidates))

    if not inject_records:
        raise RuntimeError("No inject boundary records generated.")
    if len(clean_candidates) < len(inject_records):
        raise RuntimeError(f"Not enough clean candidates: {len(clean_candidates)} < {len(inject_records)}")

    random.shuffle(clean_candidates)
    clean_records = clean_candidates[:len(inject_records)]
    for i, rec in enumerate(clean_records):
        rec["id"] = f"nfcorpus_strict_clean_{i}"

    records = inject_records + clean_records
    random.shuffle(records)

    # Audit
    full_texts = [r["full_text"] for r in records]
    full_counter = Counter(full_texts)

    target_in_context = sum(1 for r in records if r["target"] and r["target"] in r["context"])
    context_contains_any_wt = sum(1 for r in inject_records if any(wt in r["context"] for wt in all_wt))
    clean_target_contains_any_wt = sum(1 for r in clean_records if any(wt in r["target"] for wt in all_wt))
    duplicate_full_text = sum(1 for k, v in full_counter.items() if v > 1)

    audit = {
        "dataset": "nfcorpus",
        "version": "strict_v1",
        "seed": SEED,
        "collection": COLLECTION,
        "collection_count": col.count(),
        "changed_docs": len(changed_docs),
        "clean_docs": len(clean_docs),
        "inject_units": len(json.load(open(INJECT_PATH, "r", encoding="utf-8"))),
        "watermark_candidates": len(units),
        "inject_records": len(inject_records),
        "clean_records": len(clean_records),
        "total_records": len(records),
        "label_counts": dict(Counter(r["label"] for r in records)),
        "missing_watermark_candidates_in_changed_docs": len(missing),
        "target_in_context": target_in_context,
        "context_contains_any_wt": context_contains_any_wt,
        "clean_target_contains_any_wt": clean_target_contains_any_wt,
        "duplicate_full_text": duplicate_full_text,
        "candidate_count_distribution_in_inject_file": dict(Counter(len(x[2]) for x in json.load(open(INJECT_PATH, "r", encoding="utf-8")))),
        "note": "Strict NFCorpus boundary dataset built from current wmuint_inject.json and current Chroma collection. Context for inject records is cut before the earliest appended watermark text in the changed document to avoid watermark leakage into context."
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audit, open(AUDIT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved:", OUT_PATH)
    print("saved:", AUDIT_PATH)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

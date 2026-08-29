import os
import re
import json
import random
from pathlib import Path

import chromadb

COLLECTION = "trec-covid_contriever_cosine"
INJECT_PATH = "/root/autodl-tmp/ragwm_storage/output/wm_generate/trec-covid/10/wmuint_inject.json"
OUT_PATH = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v3.json"
AUDIT_PATH = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v3_audit.json"

RANDOM_SEED = 12
random.seed(RANDOM_SEED)

CHROMA_CANDIDATES = [
    "/root/autodl-tmp/ragwm_storage/chromadb_db",
    "/workspace/ragwm/ragwm/chromadb_db",
]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def sent_split(text):
    text = norm(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def extract_wt(x):
    if isinstance(x, list) and len(x) > 0:
        return norm(x[0])
    return norm(x)


def get_chroma_collection():
    for p in CHROMA_CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            client = chromadb.PersistentClient(path=p)
            names = [c.name for c in client.list_collections()]
            if COLLECTION in names:
                col = client.get_collection(COLLECTION)
                print("USE CHROMA:", p)
                print("COLLECTION COUNT:", col.count())
                return p, col
        except Exception as e:
            print("[WARN] chroma open failed:", p, repr(e))
    raise RuntimeError("Cannot find target Chroma collection.")


def remove_all_wt(doc, all_wts):
    doc = norm(doc)
    # 长句优先，避免短句先删破坏长句
    for wt in sorted(all_wts, key=len, reverse=True):
        if wt and wt in doc:
            doc = doc.replace(wt, " ")
    return norm(doc)


def contains_any_wt(text, all_wts):
    text = norm(text)
    return any(wt and wt in text for wt in all_wts)


def tail_context(text, max_chars=900):
    text = norm(text)
    sents = sent_split(text)
    if len(sents) >= 2:
        ctx = " ".join(sents[-3:])
    else:
        ctx = text[-max_chars:]
    return norm(ctx)[-max_chars:]


def clean_boundary_from_doc(text, max_chars=900):
    sents = sent_split(text)
    if len(sents) < 2:
        return None

    candidates = []
    for i in range(1, len(sents)):
        left = " ".join(sents[max(0, i - 3):i]).strip()
        right = sents[i].strip()
        if len(left) >= 80 and len(right) >= 25:
            candidates.append((left[-max_chars:], right))

    if not candidates:
        return None

    return random.choice(candidates)


def main():
    if not os.path.exists(INJECT_PATH):
        raise FileNotFoundError(INJECT_PATH)

    chroma_path, col = get_chroma_collection()

    inject_data = json.load(open(INJECT_PATH, "r", encoding="utf-8"))
    print("inject units:", len(inject_data))

    all_wts = []
    injected_doc_ids = set()

    for item in inject_data:
        if not isinstance(item, list) or len(item) < 3:
            continue
        doc_ids = item[1]
        wm_items = item[2]
        if isinstance(doc_ids, list):
            injected_doc_ids.update(map(str, doc_ids))
        if isinstance(wm_items, list):
            for z in wm_items:
                wt = extract_wt(z)
                if len(wt) >= 10:
                    all_wts.append(wt)

    all_wts = sorted(set(all_wts), key=len, reverse=True)
    print("all watermark texts:", len(all_wts))
    print("injected doc ids:", len(injected_doc_ids))

    inject_records = []

    for unit_idx, item in enumerate(inject_data):
        if not isinstance(item, list) or len(item) < 3:
            continue

        wmunit = item[0]
        doc_ids = item[1]
        wm_items = item[2]
        if not isinstance(doc_ids, list) or not isinstance(wm_items, list):
            continue

        n = min(len(doc_ids), len(wm_items))

        for j in range(n):
            doc_id = str(doc_ids[j])
            wt = extract_wt(wm_items[j])

            if len(wt) < 10:
                continue

            got = col.get(ids=[doc_id], include=["documents", "metadatas"])
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            if not docs:
                continue

            raw_doc = docs[0]
            meta = metas[0] if metas else {}

            clean_doc = remove_all_wt(raw_doc, all_wts)
            ctx = tail_context(clean_doc)

            if len(ctx) < 50:
                continue

            # inject 样本的 context 不能已经包含任何水印句
            if contains_any_wt(ctx, all_wts):
                continue

            inject_records.append({
                "id": f"inject_{unit_idx}_{j}",
                "label": 1,
                "dataset": "trec-covid",
                "kind": "inject_boundary",
                "group_id": f"wm_{unit_idx}",
                "unit_index": unit_idx,
                "doc_rank": j,
                "doc_id": doc_id,
                "metadata": meta,
                "wmunit": wmunit,
                "context": ctx,
                "target": wt,
                "full_text": ctx + " " + wt,
            })

    print("valid inject records:", len(inject_records))

    # clean 样本从非注入文档中抽，避免污染
    clean_records = []
    count = col.count()
    ids = list(range(count))
    random.shuffle(ids)

    batch_size = 200
    target_clean = len(inject_records)

    for start in range(0, len(ids), batch_size):
        if len(clean_records) >= target_clean:
            break

        batch_nums = ids[start:start + batch_size]
        batch_ids = [f"id_{i}" for i in batch_nums if f"id_{i}" not in injected_doc_ids]
        if not batch_ids:
            continue

        got = col.get(ids=batch_ids, include=["documents", "metadatas"])
        got_ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []

        for doc_id, raw_doc, meta in zip(got_ids, docs, metas):
            if len(clean_records) >= target_clean:
                break

            clean_doc = remove_all_wt(raw_doc, all_wts)

            # 保险：clean doc 不应包含任何水印句
            if contains_any_wt(clean_doc, all_wts):
                continue

            cb = clean_boundary_from_doc(clean_doc)
            if cb is None:
                continue

            cctx, ctgt = cb

            # clean target 不能是任何 watermark text
            if contains_any_wt(cctx, all_wts) or contains_any_wt(ctgt, all_wts):
                continue

            clean_records.append({
                "id": f"clean_{len(clean_records)}",
                "label": 0,
                "dataset": "trec-covid",
                "kind": "clean_boundary",
                "group_id": f"clean_{doc_id}",
                "unit_index": None,
                "doc_rank": None,
                "doc_id": doc_id,
                "metadata": meta,
                "wmunit": None,
                "context": cctx,
                "target": ctgt,
                "full_text": cctx + " " + ctgt,
            })

    print("valid clean records:", len(clean_records))

    m = min(len(inject_records), len(clean_records))
    if m == 0:
        raise RuntimeError("No valid records generated.")

    random.shuffle(inject_records)
    random.shuffle(clean_records)

    final = inject_records[:m] + clean_records[:m]
    random.shuffle(final)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(final, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    audit = {
        "chroma_path": chroma_path,
        "collection": COLLECTION,
        "collection_count": count,
        "inject_units": len(inject_data),
        "all_watermark_texts": len(all_wts),
        "injected_doc_ids": len(injected_doc_ids),
        "raw_valid_inject_records": len(inject_records),
        "raw_valid_clean_records": len(clean_records),
        "balanced_total": len(final),
        "label_1": sum(1 for r in final if r["label"] == 1),
        "label_0": sum(1 for r in final if r["label"] == 0),
        "target_in_context": sum(1 for r in final if r["target"] in r["context"]),
        "context_contains_any_wt": sum(1 for r in final if contains_any_wt(r["context"], all_wts)),
        "clean_target_contains_any_wt": sum(1 for r in final if r["label"] == 0 and contains_any_wt(r["target"], all_wts)),
        "duplicate_full_text": len(final) - len(set(r["full_text"] for r in final)),
    }

    json.dump(audit, open(AUDIT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\nAUDIT:")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    print("\nPREVIEW:")
    for r in final[:5]:
        print("----")
        print("label:", r["label"], r["kind"], "doc:", r["doc_id"])
        print("context:", r["context"][-250:])
        print("target:", r["target"])


if __name__ == "__main__":
    main()

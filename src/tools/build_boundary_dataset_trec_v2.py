import os
import re
import json
import random
from pathlib import Path

import chromadb

CHROMA_PATH = "/root/autodl-tmp/ragwm_storage/chromadb_db"
COLLECTION = "trec-covid_contriever_cosine"

INJECT_PATH = "/root/autodl-tmp/ragwm_storage/output/wm_generate/trec-covid/10/wmuint_inject.json"
OUT_PATH = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5_v2.json"

random.seed(12)


def norm_space(s):
    return re.sub(r"\s+", " ", s or "").strip()


def sent_split(text):
    text = norm_space(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def extract_wm_text(x):
    if isinstance(x, list) and len(x) > 0:
        return norm_space(str(x[0]))
    return norm_space(str(x))


def remove_wm_from_doc(doc, wt):
    """
    当前 Chroma 里的文档可能已经被 inject 修改过。
    如果 doc 中已经包含 wt，则把 wt 前面的部分作为原始 context。
    """
    doc = norm_space(doc)
    wt = norm_space(wt)

    if not wt:
        return doc

    idx = doc.find(wt)
    if idx >= 0:
        before = doc[:idx].strip()
        if len(before) >= 50:
            return before

    # 兜底：删除所有完全匹配的 wt
    cleaned = doc.replace(wt, " ")
    cleaned = norm_space(cleaned)
    return cleaned


def tail_context(text, max_chars=900):
    text = norm_space(text)
    sents = sent_split(text)
    if len(sents) >= 2:
        ctx = " ".join(sents[-3:])
    else:
        ctx = text[-max_chars:]
    return norm_space(ctx)[-max_chars:]


def clean_boundary_from_doc(text, max_chars=900):
    sents = sent_split(text)
    if len(sents) < 2:
        return None

    candidates = []
    for i in range(1, len(sents)):
        left = " ".join(sents[max(0, i - 3):i]).strip()
        right = sents[i].strip()
        if len(left) >= 80 and len(right) >= 20:
            candidates.append((left[-max_chars:], right))

    if not candidates:
        return None

    return random.choice(candidates)


def main():
    print("CHROMA_PATH:", CHROMA_PATH)
    print("COLLECTION:", COLLECTION)
    print("INJECT_PATH:", INJECT_PATH)
    print("OUT_PATH:", OUT_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)
    print("collection count:", col.count())

    inject_data = json.load(open(INJECT_PATH, "r", encoding="utf-8"))
    print("inject units:", len(inject_data))

    inject_records = []
    clean_records = []
    used_clean = set()

    for unit_idx, item in enumerate(inject_data):
        if not isinstance(item, list) or len(item) < 3:
            continue

        wmunit = item[0]
        doc_ids = item[1]
        wm_text_items = item[2]

        n = min(len(doc_ids), len(wm_text_items))

        for j in range(n):
            doc_id = str(doc_ids[j])
            wt = extract_wm_text(wm_text_items[j])
            if len(wt) < 10:
                continue

            got = col.get(ids=[doc_id], include=["documents", "metadatas"])
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []

            if not docs:
                continue

            raw_doc = docs[0]
            meta = metas[0] if metas else {}

            original_part = remove_wm_from_doc(raw_doc, wt)
            ctx = tail_context(original_part)

            # 关键检查：context 不能已经包含 target
            if wt in ctx:
                print("[SKIP_DUP]", unit_idx, j, doc_id, wt[:80])
                continue

            if len(ctx) < 50:
                continue

            inject_records.append({
                "id": f"inject_{unit_idx}_{j}",
                "label": 1,
                "dataset": "trec-covid",
                "kind": "inject_boundary",
                "unit_index": unit_idx,
                "doc_rank": j,
                "doc_id": doc_id,
                "metadata": meta,
                "wmunit": wmunit,
                "context": ctx,
                "target": wt,
                "full_text": ctx + " " + wt,
            })

            if doc_id not in used_clean:
                cb = clean_boundary_from_doc(original_part)
                if cb is not None:
                    cctx, ctgt = cb
                    clean_records.append({
                        "id": f"clean_{unit_idx}_{j}",
                        "label": 0,
                        "dataset": "trec-covid",
                        "kind": "clean_boundary",
                        "unit_index": unit_idx,
                        "doc_rank": j,
                        "doc_id": doc_id,
                        "metadata": meta,
                        "wmunit": None,
                        "context": cctx,
                        "target": ctgt,
                        "full_text": cctx + " " + ctgt,
                    })
                    used_clean.add(doc_id)

    print("raw inject records:", len(inject_records))
    print("raw clean records:", len(clean_records))

    m = min(len(inject_records), len(clean_records))
    if m == 0:
        raise RuntimeError("No valid boundary records generated.")

    random.shuffle(inject_records)
    random.shuffle(clean_records)

    final = inject_records[:m] + clean_records[:m]
    random.shuffle(final)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(final, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("balanced total:", len(final))
    print("label 1:", sum(1 for r in final if r["label"] == 1))
    print("label 0:", sum(1 for r in final if r["label"] == 0))
    print("saved:", OUT_PATH)

    print("\npreview:")
    for r in final[:3]:
        print(json.dumps({
            "id": r["id"],
            "label": r["label"],
            "kind": r["kind"],
            "doc_id": r["doc_id"],
            "context": r["context"][:180],
            "target": r["target"][:120],
            "target_in_context": r["target"] in r["context"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import os
import re
import json
import random
from pathlib import Path

import chromadb


CHROMA_PATH = "/root/autodl-tmp/ragwm_storage/chromadb_db"
COLLECTION = "trec-covid_contriever_cosine"

INJECT_PATH = "/root/autodl-tmp/ragwm_storage/output/wm_generate/trec-covid/10/wmuint_inject.json"
OUT_PATH = "/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_trec_k5.json"

RANDOM_SEED = 12
random.seed(RANDOM_SEED)


def sent_split(text: str):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    parts = [p.strip() for p in parts if len(p.strip()) >= 20]
    return parts


def tail_context(text: str, max_chars: int = 900):
    sents = sent_split(text)
    if len(sents) >= 2:
        ctx = " ".join(sents[-3:])
    else:
        ctx = text[-max_chars:]
    ctx = re.sub(r"\s+", " ", ctx).strip()
    return ctx[-max_chars:]


def clean_boundary_from_doc(text: str, max_chars: int = 900):
    sents = sent_split(text)
    if len(sents) < 2:
        return None

    # 优先取中间自然边界，避免开头/结尾异常
    candidates = []
    for i in range(1, len(sents)):
        left = " ".join(sents[max(0, i-3):i]).strip()
        right = sents[i].strip()
        if len(left) >= 80 and len(right) >= 20:
            candidates.append((left[-max_chars:], right))

    if not candidates:
        return None

    return random.choice(candidates)


def extract_wm_text(x):
    # 常见格式：["sentence", 1, 1]
    if isinstance(x, list) and len(x) > 0:
        return str(x[0]).strip()
    return str(x).strip()


def main():
    print("CHROMA_PATH:", CHROMA_PATH)
    print("COLLECTION:", COLLECTION)
    print("INJECT_PATH:", INJECT_PATH)
    print("OUT_PATH:", OUT_PATH)

    if not os.path.exists(INJECT_PATH):
        raise FileNotFoundError(INJECT_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)
    print("collection count:", col.count())

    inject_data = json.load(open(INJECT_PATH, "r", encoding="utf-8"))
    print("inject units:", len(inject_data))

    records = []
    clean_records = []

    used_clean_doc_ids = set()

    for unit_idx, item in enumerate(inject_data):
        if not isinstance(item, list) or len(item) < 3:
            continue

        wmunit = item[0]
        doc_ids = item[1]
        wm_text_items = item[2]

        if not isinstance(doc_ids, list) or not isinstance(wm_text_items, list):
            continue

        n = min(len(doc_ids), len(wm_text_items))

        for j in range(n):
            doc_id = str(doc_ids[j])
            wt = extract_wm_text(wm_text_items[j])
            if not wt or len(wt) < 10:
                continue

            try:
                got = col.get(ids=[doc_id], include=["documents", "metadatas"])
            except Exception as e:
                print("[WARN] chroma get failed:", doc_id, repr(e))
                continue

            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            if not docs:
                print("[WARN] empty doc:", doc_id)
                continue

            original_doc = docs[0]
            meta = metas[0] if metas else {}

            ctx = tail_context(original_doc)
            if len(ctx) < 50:
                continue

            records.append({
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

            # 同一个被注入文档内构造一条 clean boundary
            cb = clean_boundary_from_doc(original_doc)
            if cb is not None and doc_id not in used_clean_doc_ids:
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
                used_clean_doc_ids.add(doc_id)

    print("raw inject records:", len(records))
    print("raw clean records:", len(clean_records))

    m = min(len(records), len(clean_records))
    if m == 0:
        raise RuntimeError("No boundary pairs generated. Check Chroma ids and wmuint_inject format.")

    random.shuffle(records)
    random.shuffle(clean_records)

    final = records[:m] + clean_records[:m]
    random.shuffle(final)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(final, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("balanced total:", len(final))
    print("inject used:", m)
    print("clean used:", m)
    print("saved:", OUT_PATH)

    print("\npreview:")
    for r in final[:3]:
        print(json.dumps({
            "id": r["id"],
            "label": r["label"],
            "kind": r["kind"],
            "doc_id": r["doc_id"],
            "context": r["context"][:120],
            "target": r["target"][:120],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

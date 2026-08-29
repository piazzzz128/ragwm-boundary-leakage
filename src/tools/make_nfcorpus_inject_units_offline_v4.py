import re
import json
import random
from pathlib import Path

import chromadb


SEED = 12
CHROMA_PATH = "/root/autodl-tmp/ragwm_storage/chromadb_db"
COLLECTION = "nfcorpus_contriever_cosine"

OUT_DIR = Path("/root/autodl-tmp/ragwm_storage/output/wm_generate/nfcorpus/10")
OUT_FILE = OUT_DIR / "wmuint_inject.json"
OUT_AUDIT = OUT_DIR / "wmuint_inject_offline_v4_audit.json"

N_UNITS = 50
DOCS_PER_UNIT = 2
PARAPHRASES_PER_UNIT = 5


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "has", "have",
    "study", "studies", "analysis", "effect", "effects", "result", "results", "using",
    "patients", "patient", "health", "disease", "risk", "treatment", "clinical"
}

RELATIONS = [
    "ASSOCIATED_WITH",
    "INFLUENCES",
    "AFFECTS",
    "RELATED_TO",
    "MODULATES",
    "CONTRIBUTES_TO",
    "INTERACTS_WITH",
    "PREDICTS",
    "REDUCES",
    "INCREASES",
]


def normalize(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def get_docs():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)
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
        if len(d) >= 200:
            out.append({
                "id": str(i),
                "doc": d,
                "meta": m or {},
            })
    return out


def candidate_terms(text):
    text = normalize(text)
    # Prefer title-like first line terms.
    chunks = []
    first = text.split("\n")[0] if "\n" in text else text[:200]
    chunks.append(first)

    # Capitalized phrases.
    caps = re.findall(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4}\b", text)
    chunks.extend(caps)

    # Biomedical-looking lower/mixed phrases.
    phrases = re.findall(r"\b[a-zA-Z][a-zA-Z\-]+(?:\s+[a-zA-Z][a-zA-Z\-]+){1,3}\b", text)
    chunks.extend(phrases[:80])

    terms = []
    seen = set()
    for c in chunks:
        c = normalize(c)
        words = c.split()
        if not (1 <= len(words) <= 5):
            continue
        low = c.lower()
        if low in STOPWORDS:
            continue
        if any(w.lower() in STOPWORDS for w in words) and len(words) == 1:
            continue
        if len(c) < 4 or len(c) > 80:
            continue
        if low not in seen:
            seen.add(low)
            terms.append(c)

    return terms


def make_texts(e1, e2, rel):
    rel_low = rel.replace("_", " ").lower()

    return [
        f"{e1} is {rel_low} {e2} in this biomedical context.",
        f"The relationship between {e1} and {e2} can be described as {rel_low}.",
        f"In biomedical evidence, {e1} is considered {rel_low} {e2}.",
        f"{e2} has a documented connection with {e1} through the relation {rel_low}.",
        f"Research descriptions may characterize {e1} as {rel_low} {e2}.",
    ]


def main():
    rng = random.Random(SEED)
    docs = get_docs()
    print("docs:", len(docs))

    rng.shuffle(docs)

    records = []
    used_pairs = set()

    doc_cursor = 0

    for unit_idx in range(N_UNITS):
        selected_doc_ids = []
        selected_docs = []

        while len(selected_docs) < DOCS_PER_UNIT and doc_cursor < len(docs):
            d = docs[doc_cursor]
            doc_cursor += 1
            terms = candidate_terms(d["doc"])
            if len(terms) >= 2:
                selected_docs.append(d)
                selected_doc_ids.append(d["id"])

        if len(selected_docs) < DOCS_PER_UNIT:
            break

        # Build terms from selected docs.
        terms = []
        for d in selected_docs:
            terms.extend(candidate_terms(d["doc"])[:20])

        terms = list(dict.fromkeys(terms))
        if len(terms) < 2:
            continue

        for _ in range(50):
            e1, e2 = rng.sample(terms, 2)
            if e1.lower() != e2.lower() and (e1.lower(), e2.lower()) not in used_pairs:
                break

        used_pairs.add((e1.lower(), e2.lower()))
        rel = RELATIONS[unit_idx % len(RELATIONS)]

        wmunit = [e1, e2, rel]
        texts = make_texts(e1, e2, rel)

        wt_items = [[t, 1, 1] for t in texts[:PARAPHRASES_PER_UNIT]]
        records.append([wmunit, selected_doc_ids, wt_items])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    audit = {
        "construction": "offline_template_v4",
        "dataset": "nfcorpus",
        "chroma_path": CHROMA_PATH,
        "collection": COLLECTION,
        "n_units": len(records),
        "docs_per_unit": DOCS_PER_UNIT,
        "paraphrases_per_unit": PARAPHRASES_PER_UNIT,
        "estimated_inject_records": len(records) * DOCS_PER_UNIT * PARAPHRASES_PER_UNIT,
        "output": str(OUT_FILE),
        "note": "This is an offline RAG-WM-style construction used because the configured chat completion API returned HTTP 500.",
    }
    json.dump(audit, open(OUT_AUDIT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved:", OUT_FILE)
    print("saved:", OUT_AUDIT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("preview:", records[:2])


if __name__ == "__main__":
    main()

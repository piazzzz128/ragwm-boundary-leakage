from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import chromadb
import torch

from e2_common_aligned import read_jsonl, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--collection", default="nfcorpus_contriever_cosine")
    parser.add_argument("--retriever", default="contriever")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, args.repo)
    from src.utils import load_models

    rows = read_jsonl(Path(args.input))
    if len(rows) != 3633:
        raise RuntimeError(
            f"Expected 3633 sanitized documents, found {len(rows)}"
        )

    db_path = Path(args.db)
    if args.reset and db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer, get_emb = load_models(args.retriever)
    model.eval()
    model.to(device)

    client = chromadb.PersistentClient(path=str(db_path))
    existing_names = [
        getattr(item, "name", str(item))
        for item in client.list_collections()
    ]
    if args.collection in existing_names:
        client.delete_collection(name=args.collection)

    collection = client.create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [row["text"] for row in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {
            key: value.to(device)
            for key, value in tokenized.items()
        }

        with torch.no_grad():
            embeddings = get_emb(model, tokenized)
        embeddings = embeddings.detach().cpu().tolist()

        ids = [row["internal_id"] for row in batch]
        metadatas = [
            {
                "id": row["source_doc_id"],
                "title": row.get("title", ""),
                "change": False,
                "sanitization": (
                    "gainratio_aligned_iterative_suffix_sensitivity"
                ),
            }
            for row in batch
        ]
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(
            f"embedded {min(start + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )

    count = collection.count()
    report = {
        "experiment_role": "aligned_end_to_end_sensitivity",
        "database": str(db_path),
        "collection": args.collection,
        "count": count,
        "expected_count": len(rows),
        "retriever": args.retriever,
        "distance": "cosine",
    }
    write_json_atomic(Path(args.summary_output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if count != len(rows):
        raise RuntimeError(
            f"Vector store count mismatch: {count} != {len(rows)}"
        )
    print("ALIGNED E2 VECTORSTORE BUILD PASS")


if __name__ == "__main__":
    main()

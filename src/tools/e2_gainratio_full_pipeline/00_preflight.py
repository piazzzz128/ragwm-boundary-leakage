from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import torch

from e2_common import DEFAULT_MODEL_PATH, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermarked-db", required=True)
    parser.add_argument("--collection", default="nfcorpus_contriever_cosine")
    parser.add_argument("--strict-scores", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    db_path = Path(args.watermarked_db)
    score_path = Path(args.strict_scores)
    model_path = Path(args.model_path)

    required = [db_path, score_path, model_path / "config.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=args.collection)
    scores = json.loads(score_path.read_text(encoding="utf-8"))

    labels = {0: 0, 1: 0}
    for row in scores:
        labels[int(row["label"])] = labels.get(int(row["label"]), 0) + 1

    report = {
        "watermarked_db": str(db_path),
        "collection": args.collection,
        "collection_count": collection.count(),
        "strict_scores": str(score_path),
        "strict_scores_sha256": sha256_file(score_path),
        "strict_score_count": len(scores),
        "label_counts": labels,
        "model_path": str(model_path),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if collection.count() != 3633:
        raise RuntimeError(f"Expected 3633 documents, found {collection.count()}")
    if labels.get(0) != 223 or labels.get(1) != 223:
        raise RuntimeError(f"Expected 223 clean and 223 inject scores, found {labels}")
    print("E2 PREFLIGHT PASS")


if __name__ == "__main__":
    main()

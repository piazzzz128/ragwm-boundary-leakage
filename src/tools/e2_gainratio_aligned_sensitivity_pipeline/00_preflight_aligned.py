from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import torch

from e2_common_aligned import DEFAULT_MODEL_PATH, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermarked-db", required=True)
    parser.add_argument("--collection", default="nfcorpus_contriever_cosine")
    parser.add_argument("--aligned-scores", required=True)
    parser.add_argument("--attack-input", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    db_path = Path(args.watermarked_db)
    score_path = Path(args.aligned_scores)
    input_path = Path(args.attack_input)
    model_path = Path(args.model_path)

    required = [db_path, score_path, input_path, model_path / "config.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=args.collection)
    scores = json.loads(score_path.read_text(encoding="utf-8"))

    labels = {0: 0, 1: 0}
    missing_score = 0
    for row in scores:
        labels[int(row["label"])] = labels.get(int(row["label"]), 0) + 1
        if "score_aligned" not in row:
            missing_score += 1

    input_count = sum(
        1 for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    report = {
        "experiment_role": "aligned_end_to_end_sensitivity",
        "watermarked_db": str(db_path),
        "collection": args.collection,
        "collection_count": collection.count(),
        "aligned_scores": str(score_path),
        "aligned_scores_sha256": sha256_file(score_path),
        "aligned_score_count": len(scores),
        "aligned_score_key": "score_aligned",
        "rows_missing_aligned_score": missing_score,
        "label_counts": labels,
        "attack_input": str(input_path),
        "attack_input_sha256": sha256_file(input_path),
        "attack_input_count": input_count,
        "model_path": str(model_path),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if collection.count() != 3633:
        raise RuntimeError(
            f"Expected 3633 documents, found {collection.count()}"
        )
    if input_count != 3633:
        raise RuntimeError(
            f"Expected 3633 attack-input rows, found {input_count}"
        )
    if labels.get(0) != 223 or labels.get(1) != 223:
        raise RuntimeError(
            f"Expected 223 clean and 223 inject scores, found {labels}"
        )
    if missing_score:
        raise RuntimeError(
            f"{missing_score} rows lack score_aligned"
        )

    print("ALIGNED E2 PREFLIGHT PASS")


if __name__ == "__main__":
    main()

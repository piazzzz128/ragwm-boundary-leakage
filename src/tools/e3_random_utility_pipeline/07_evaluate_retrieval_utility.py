from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import chromadb
import torch

from e3_common import write_json_atomic


def normalize_query(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "query", "title"):
            if value.get(key):
                return str(value[key])
    return str(value)


def normalize_qrels(qrels):
    output = {}
    for qid, rels in qrels.items():
        if isinstance(rels, dict):
            output[str(qid)] = {
                str(docid): float(score)
                for docid, score in rels.items()
                if float(score) > 0
            }
    return output


def dcg(relevances, k):
    total = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        total += (2.0 ** rel - 1.0) / math.log2(rank + 1)
    return total


def metrics_for_ranking(ranking, relevance, ndcg_k=10, eval_k=100):
    binary_relevant = {docid for docid, score in relevance.items() if score > 0}
    gains = [relevance.get(docid, 0.0) for docid in ranking]
    ideal = sorted(relevance.values(), reverse=True)

    actual_dcg = dcg(gains, ndcg_k)
    ideal_dcg = dcg(ideal, ndcg_k)
    ndcg = actual_dcg / ideal_dcg if ideal_dcg else 0.0

    hits = 0
    precision_sum = 0.0
    reciprocal_rank = 0.0
    for rank, docid in enumerate(ranking[:eval_k], start=1):
        if docid in binary_relevant:
            hits += 1
            precision_sum += hits / rank
            if reciprocal_rank == 0.0 and rank <= 10:
                reciprocal_rank = 1.0 / rank

    recall = hits / len(binary_relevant) if binary_relevant else 0.0
    ap = precision_sum / len(binary_relevant) if binary_relevant else 0.0

    return {
        "ndcg_at_10": ndcg,
        "map_at_100": ap,
        "recall_at_100": recall,
        "mrr_at_10": reciprocal_rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--db",
        action="append",
        required=True,
        help="Repeat as NAME=/path/to/chromadb",
    )
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--retriever", default="contriever")
    parser.add_argument("--collection", default="nfcorpus_contriever_cosine")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline-name", default="e0_watermarked")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    from src.utils import load_beir_datasets, load_models

    dbs = {}
    for value in args.db:
        if "=" not in value:
            raise ValueError(f"Invalid --db value: {value}")
        name, path = value.split("=", 1)
        dbs[name] = path

    corpus, queries, qrels = load_beir_datasets(args.dataset, args.split)
    qrels = normalize_qrels(qrels)
    query_ids = [
        str(qid)
        for qid in queries
        if str(qid) in qrels and qrels[str(qid)]
    ]
    query_texts = [normalize_query(queries[qid]) for qid in queries if str(qid) in set(query_ids)]
    # Preserve query_ids/query_texts alignment explicitly.
    aligned = [
        (str(qid), normalize_query(queries[qid]))
        for qid in queries
        if str(qid) in qrels and qrels[str(qid)]
    ]
    query_ids = [qid for qid, _ in aligned]
    query_texts = [text for _, text in aligned]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer, get_emb = load_models(args.retriever)
    model.eval()
    model.to(device)

    all_query_embeddings = []
    for start in range(0, len(query_texts), args.batch_size):
        batch = query_texts[start : start + args.batch_size]
        tokenized = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            embeddings = get_emb(model, tokenized)
        all_query_embeddings.extend(embeddings.detach().cpu().tolist())
        print(
            f"encoded queries {min(start + len(batch), len(query_texts))}/"
            f"{len(query_texts)}",
            flush=True,
        )

    rankings = {}
    aggregate = {}

    for name, db_path in dbs.items():
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=args.collection)
        condition_rankings = {}

        for start in range(0, len(query_ids), args.batch_size):
            batch_ids = query_ids[start : start + args.batch_size]
            batch_embeddings = all_query_embeddings[start : start + args.batch_size]
            result = collection.query(
                query_embeddings=batch_embeddings,
                n_results=min(args.top_k, collection.count()),
                include=["metadatas"],
            )
            for offset, qid in enumerate(batch_ids):
                ids = result["ids"][offset]
                metadatas = result["metadatas"][offset]
                canonical = []
                for internal_id, metadata in zip(ids, metadatas):
                    metadata = metadata or {}
                    canonical.append(str(metadata.get("id", internal_id)))
                condition_rankings[qid] = canonical

        rows = [
            metrics_for_ranking(condition_rankings[qid], qrels[qid])
            for qid in query_ids
        ]
        aggregate[name] = {
            "query_count": len(rows),
            "ndcg_at_10": statistics.mean(row["ndcg_at_10"] for row in rows),
            "map_at_100": statistics.mean(row["map_at_100"] for row in rows),
            "recall_at_100": statistics.mean(row["recall_at_100"] for row in rows),
            "mrr_at_10": statistics.mean(row["mrr_at_10"] for row in rows),
        }
        rankings[name] = condition_rankings
        print(name, json.dumps(aggregate[name], indent=2))

    if args.baseline_name not in rankings:
        raise RuntimeError(
            f"Baseline {args.baseline_name!r} not found in database names"
        )

    baseline = rankings[args.baseline_name]
    overlap = {}
    for name, current in rankings.items():
        if name == args.baseline_name:
            continue
        top10_fractions = []
        top100_fractions = []
        for qid in query_ids:
            base10 = set(baseline[qid][:10])
            cur10 = set(current[qid][:10])
            base100 = set(baseline[qid][:100])
            cur100 = set(current[qid][:100])
            top10_fractions.append(len(base10 & cur10) / 10.0)
            top100_fractions.append(len(base100 & cur100) / 100.0)
        overlap[name] = {
            "mean_top10_overlap_fraction_vs_baseline": statistics.mean(top10_fractions),
            "mean_top100_overlap_fraction_vs_baseline": statistics.mean(top100_fractions),
        }

    baseline_metrics = aggregate[args.baseline_name]
    deltas = {}
    for name, values in aggregate.items():
        if name == args.baseline_name:
            continue
        deltas[name] = {
            metric: values[metric] - baseline_metrics[metric]
            for metric in ("ndcg_at_10", "map_at_100", "recall_at_100", "mrr_at_10")
        }

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "retriever": args.retriever,
        "query_count": len(query_ids),
        "baseline_name": args.baseline_name,
        "metrics": aggregate,
        "metric_deltas_vs_baseline": deltas,
        "retrieval_overlap": overlap,
    }
    write_json_atomic(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("RETRIEVAL UTILITY EVALUATION PASS")


if __name__ == "__main__":
    main()

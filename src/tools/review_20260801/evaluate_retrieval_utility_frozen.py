#!/usr/bin/env python3
"""Freeze NFCorpus rankings, per-query metrics, aggregates, and paired CIs.

This evaluator performs retrieval exactly once for each condition.  Every point
estimate and percentile interval is then derived from the same frozen per-query
rows, preventing the cross-run mismatch found in the earlier aggregate and
bootstrap files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch


METRICS = ("ndcg_at_10", "map_at_100", "recall_at_100", "mrr_at_10")


def normalize_query(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "query", "title"):
            if value.get(key):
                return str(value[key])
    return str(value)


def normalize_qrels(qrels: dict[Any, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for qid, rels in qrels.items():
        if isinstance(rels, dict):
            output[str(qid)] = {
                str(docid): float(score)
                for docid, score in rels.items()
                if float(score) > 0
            }
    return output


def dcg(relevances: list[float], k: int) -> float:
    return sum(
        (2.0**rel - 1.0) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def metrics_for_ranking(
    ranking: list[str], relevance: dict[str, float], ndcg_k: int = 10, eval_k: int = 100
) -> dict[str, float]:
    binary_relevant = {docid for docid, score in relevance.items() if score > 0}
    gains = [relevance.get(docid, 0.0) for docid in ranking]
    ideal = sorted(relevance.values(), reverse=True)
    ideal_dcg = dcg(ideal, ndcg_k)
    ndcg = dcg(gains, ndcg_k) / ideal_dcg if ideal_dcg else 0.0

    hits = 0
    precision_sum = 0.0
    reciprocal_rank = 0.0
    for rank, docid in enumerate(ranking[:eval_k], start=1):
        if docid in binary_relevant:
            hits += 1
            precision_sum += hits / rank
            if reciprocal_rank == 0.0 and rank <= 10:
                reciprocal_rank = 1.0 / rank

    return {
        "ndcg_at_10": ndcg,
        "map_at_100": precision_sum / len(binary_relevant) if binary_relevant else 0.0,
        "recall_at_100": hits / len(binary_relevant) if binary_relevant else 0.0,
        "mrr_at_10": reciprocal_rank,
    }


def parse_databases(values: list[str]) -> dict[str, str]:
    databases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --db value: {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"Invalid --db value: {value!r}")
        if name in databases:
            raise ValueError(f"Duplicate database name: {name!r}")
        databases[name] = path
    return databases


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def rankings_sha256(rankings: dict[str, dict[str, list[str]]]) -> str:
    payload = json.dumps(
        rankings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--db", action="append", required=True)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--retriever", default="contriever")
    parser.add_argument("--collection", default="nfcorpus_contriever_cosine")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline-name", default="e0_watermarked")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.top_k < 100:
        raise ValueError("--top-k must be at least 100 for @100 metrics")
    databases = parse_databases(args.db)
    if args.baseline_name not in databases:
        raise ValueError("The baseline name must be one of the supplied databases")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, args.repo)
    from src.utils import load_beir_datasets, load_models

    _, queries, raw_qrels = load_beir_datasets(args.dataset, args.split)
    qrels = normalize_qrels(raw_qrels)
    aligned_queries = [
        (str(qid), normalize_query(value))
        for qid, value in queries.items()
        if str(qid) in qrels and qrels[str(qid)]
    ]
    query_ids = [qid for qid, _ in aligned_queries]
    query_texts = [text for _, text in aligned_queries]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer, get_emb = load_models(args.retriever)
    model.eval()
    model.to(device)

    embeddings: list[list[float]] = []
    for start in range(0, len(query_texts), args.batch_size):
        batch = query_texts[start : start + args.batch_size]
        tokenized = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            encoded = get_emb(model, tokenized)
        embeddings.extend(encoded.detach().cpu().tolist())
        print(f"encoded {min(start + len(batch), len(query_texts))}/{len(query_texts)}")

    rankings: dict[str, dict[str, list[str]]] = {}
    collection_manifest: dict[str, dict[str, Any]] = {}
    for name, db_path in databases.items():
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=args.collection)
        condition_rankings: dict[str, list[str]] = {}
        collection_manifest[name] = {
            "database_path": str(Path(db_path).resolve()),
            "collection": args.collection,
            "collection_count": int(collection.count()),
            "collection_metadata": collection.metadata,
        }
        for start in range(0, len(query_ids), args.batch_size):
            batch_ids = query_ids[start : start + args.batch_size]
            result = collection.query(
                query_embeddings=embeddings[start : start + args.batch_size],
                n_results=min(args.top_k, collection.count()),
                include=["metadatas"],
            )
            for offset, qid in enumerate(batch_ids):
                canonical: list[str] = []
                for internal_id, metadata in zip(
                    result["ids"][offset], result["metadatas"][offset]
                ):
                    metadata = metadata or {}
                    canonical.append(str(metadata.get("id", internal_id)))
                condition_rankings[qid] = canonical
        rankings[name] = condition_rankings

    per_query: list[dict[str, Any]] = []
    metric_arrays: dict[str, dict[str, np.ndarray]] = {
        name: {metric: np.empty(len(query_ids), dtype=float) for metric in METRICS}
        for name in databases
    }
    for query_index, qid in enumerate(query_ids):
        row: dict[str, Any] = {"query_id": qid, "conditions": {}}
        for name in databases:
            ranking = rankings[name][qid]
            metrics = metrics_for_ranking(ranking, qrels[qid])
            for metric in METRICS:
                metric_arrays[name][metric][query_index] = metrics[metric]
            row["conditions"][name] = {
                **metrics,
                "top10_doc_ids": ranking[:10],
                "top100_doc_ids": ranking[:100],
            }
        per_query.append(row)

    aggregate = {
        name: {
            "query_count": len(query_ids),
            **{
                metric: float(metric_arrays[name][metric].mean())
                for metric in METRICS
            },
        }
        for name in databases
    }

    baseline_arrays = metric_arrays[args.baseline_name]
    deltas: dict[str, dict[str, float]] = {}
    overlap: dict[str, dict[str, float]] = {}
    differences: dict[str, dict[str, np.ndarray]] = {}
    for name in databases:
        if name == args.baseline_name:
            continue
        deltas[name] = {
            metric: float(
                (metric_arrays[name][metric] - baseline_arrays[metric]).mean()
            )
            for metric in METRICS
        }
        differences[name] = {
            metric: metric_arrays[name][metric] - baseline_arrays[metric]
            for metric in METRICS
        }
        overlap[name] = {
            "mean_top10_overlap_fraction_vs_baseline": statistics.mean(
                len(
                    set(rankings[args.baseline_name][qid][:10])
                    & set(rankings[name][qid][:10])
                )
                / 10.0
                for qid in query_ids
            ),
            "mean_top100_overlap_fraction_vs_baseline": statistics.mean(
                len(
                    set(rankings[args.baseline_name][qid][:100])
                    & set(rankings[name][qid][:100])
                )
                / 100.0
                for qid in query_ids
            ),
        }

    rng = np.random.default_rng(args.seed)
    bootstrap_samples = {
        name: {metric: np.empty(args.n_boot, dtype=float) for metric in METRICS}
        for name in differences
    }
    for replicate in range(args.n_boot):
        sampled = rng.integers(0, len(query_ids), size=len(query_ids))
        for name in differences:
            for metric in METRICS:
                bootstrap_samples[name][metric][replicate] = float(
                    differences[name][metric][sampled].mean()
                )

    paired_bootstrap: dict[str, dict[str, Any]] = {}
    for name, metric_samples in bootstrap_samples.items():
        paired_bootstrap[name] = {}
        for metric, samples in metric_samples.items():
            lower, upper = np.quantile(samples, (0.025, 0.975))
            diff = differences[name][metric]
            paired_bootstrap[name][metric] = {
                "observed_mean_difference_vs_baseline": float(diff.mean()),
                "bootstrap_95_percentile_ci": [float(lower), float(upper)],
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "query_level_positive_count": int(np.count_nonzero(diff > 0.0)),
                "query_level_negative_count": int(np.count_nonzero(diff < 0.0)),
                "query_level_tie_count": int(np.count_nonzero(diff == 0.0)),
            }

    report: dict[str, Any] = {
        "experiment_role": "frozen_ranking_retrieval_utility_and_paired_bootstrap",
        "dataset": args.dataset,
        "split": args.split,
        "retriever": args.retriever,
        "device": device,
        "query_count": len(query_ids),
        "baseline_name": args.baseline_name,
        "top_k": args.top_k,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "bootstrap_rng": "numpy.random.default_rng (PCG64)",
        "collection_manifest": collection_manifest,
        "ranking_sha256": rankings_sha256(rankings),
        "metrics": aggregate,
        "metric_deltas_vs_baseline": deltas,
        "paired_bootstrap_differences_vs_baseline": paired_bootstrap,
        "retrieval_overlap": overlap,
        "per_query": per_query,
        "interpretation_note": (
            "All rankings, point estimates, per-query differences, and confidence "
            "intervals in this file derive from one frozen retrieval pass."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, indent=2))
    print("FROZEN RETRIEVAL UTILITY EVALUATION PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from e3_common import (
    read_json,
    read_jsonl,
    sha256_file,
    valid_suffix_candidates,
    write_json_atomic,
    write_jsonl,
)


def choose_assignment(
    target_length,
    available_ids,
    candidate_map,
    rng,
    candidate_pool_size,
    random_top_k,
):
    if not available_ids:
        raise RuntimeError("No unused eligible documents remain")

    pool_size = min(candidate_pool_size, len(available_ids))
    sampled_ids = rng.sample(available_ids, pool_size)

    scored = []
    for internal_id in sampled_ids:
        candidates = candidate_map[internal_id]
        best = min(
            candidates,
            key=lambda item: abs(item["deleted_char_count"] - target_length),
        )
        error = abs(best["deleted_char_count"] - target_length)
        scored.append((error, rng.random(), internal_id, best))

    scored.sort(key=lambda item: (item[0], item[1]))
    top = scored[: min(random_top_k, len(scored))]
    return rng.choice(top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-pool-size", type=int, default=400)
    parser.add_argument("--random-top-k", type=int, default=5)
    parser.add_argument("--max-budget-error-rate", type=float, default=0.10)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    budget_path = Path(args.budget)
    output_dir = Path(args.output_dir)

    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    budget = read_json(budget_path)
    rng = random.Random(args.seed)

    if len(rows) != int(budget["input_document_count"]):
        raise RuntimeError(
            f"Input document count mismatch: {len(rows)} != "
            f"{budget['input_document_count']}"
        )

    forbidden = {"change", "label", "watermark_text", "tuple"}
    leaked = forbidden.intersection(rows[0].keys())
    if leaked:
        raise RuntimeError(f"Ground-truth fields leaked into random input: {leaked}")

    by_id = {str(row["internal_id"]): row for row in rows}
    candidate_map = {}
    for row in rows:
        internal_id = str(row["internal_id"])
        candidates = valid_suffix_candidates(row["text"])
        if candidates:
            candidate_map[internal_id] = candidates

    target_lengths = list(map(int, budget["target_deleted_char_lengths"]))
    rng.shuffle(target_lengths)
    available_ids = list(candidate_map)
    rng.shuffle(available_ids)

    if len(available_ids) < len(target_lengths):
        raise RuntimeError(
            f"Only {len(available_ids)} eligible documents for "
            f"{len(target_lengths)} required modifications"
        )

    assignments = []
    for target_length in target_lengths:
        error, _, internal_id, candidate = choose_assignment(
            target_length=target_length,
            available_ids=available_ids,
            candidate_map=candidate_map,
            rng=rng,
            candidate_pool_size=args.candidate_pool_size,
            random_top_k=args.random_top_k,
        )
        available_ids.remove(internal_id)
        row = by_id[internal_id]
        text = row["text"]
        start = int(candidate["boundary_start"])
        sanitized_text = text[:start].rstrip()
        actual_deleted = len(text) - len(sanitized_text)
        assignments.append(
            {
                "internal_id": internal_id,
                "source_doc_id": row["source_doc_id"],
                "title": row.get("title", ""),
                "target_deleted_char_count": target_length,
                "actual_deleted_char_count": actual_deleted,
                "absolute_budget_error": abs(actual_deleted - target_length),
                "relative_budget_error": (
                    abs(actual_deleted - target_length) / target_length
                    if target_length
                    else 0.0
                ),
                "boundary_start": start,
                "selected_terminal_sentence": candidate["target"],
                "original_char_count": len(text),
                "sanitized_char_count": len(sanitized_text),
                "sanitized_text": sanitized_text,
            }
        )

    assignment_map = {row["internal_id"]: row for row in assignments}
    sanitized_rows = []
    for row in rows:
        internal_id = str(row["internal_id"])
        assignment = assignment_map.get(internal_id)
        sanitized_rows.append(
            {
                "internal_id": internal_id,
                "source_doc_id": row["source_doc_id"],
                "title": row.get("title", ""),
                "text": assignment["sanitized_text"] if assignment else row["text"],
            }
        )

    target_total = int(budget["target_total_deleted_chars"])
    actual_total = sum(row["actual_deleted_char_count"] for row in assignments)
    budget_error_rate = abs(actual_total - target_total) / target_total if target_total else 0.0

    corpus_path = output_dir / "random_sanitized_corpus.jsonl"
    log_path = output_dir / "random_deletion_log.jsonl"
    summary_path = output_dir / "random_sanitization_summary.json"

    write_jsonl(corpus_path, sanitized_rows)
    write_jsonl(log_path, assignments)

    summary = {
        "seed": args.seed,
        "input_document_count": len(rows),
        "eligible_document_count": len(candidate_map),
        "modified_document_count": len(assignments),
        "target_modified_document_count": int(
            budget["target_modified_document_count"]
        ),
        "modified_document_count_match": (
            len(assignments) == int(budget["target_modified_document_count"])
        ),
        "target_total_deleted_chars": target_total,
        "actual_total_deleted_chars": actual_total,
        "absolute_total_budget_error": abs(actual_total - target_total),
        "relative_total_budget_error": budget_error_rate,
        "budget_match_pass": budget_error_rate <= args.max_budget_error_rate,
        "selection_uses_ground_truth": False,
        "selection_policy": (
            "Uniformly randomized eligible-document pool with sentence-boundary "
            "suffix cuts chosen to approximately match E2 per-document deletion lengths."
        ),
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "deletion_log": str(log_path),
        "deletion_log_sha256": sha256_file(log_path),
    }
    write_json_atomic(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["modified_document_count_match"]:
        raise RuntimeError("Modified-document count does not match E2")
    if not summary["budget_match_pass"]:
        raise RuntimeError(
            "Random deletion character budget differs too much from E2; "
            "increase --candidate-pool-size or --max-budget-error-rate only with justification"
        )
    print("RANDOM SANITIZATION PASS")


if __name__ == "__main__":
    main()

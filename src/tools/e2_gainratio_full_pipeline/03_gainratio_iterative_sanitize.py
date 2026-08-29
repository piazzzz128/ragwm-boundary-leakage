from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from e2_common import (
    DEFAULT_MODEL_PATH,
    gainratio_score_exact_strict_v1,
    last_eligible_sentence_span,
    load_json,
    load_qwen_model,
    norm_space,
    read_jsonl,
    sha256_file,
    tail_context,
    write_json_atomic,
    write_jsonl,
)


def process_document(row, tokenizer, model, device, threshold, max_steps):
    original_text = row["text"]
    current_text = original_text
    candidates: List[Dict[str, Any]] = []
    deletions: List[Dict[str, Any]] = []
    stop_reason = "max_steps_reached"

    for step in range(1, max_steps + 1):
        span = last_eligible_sentence_span(current_text)
        if span is None:
            stop_reason = "no_eligible_sentence"
            break
        start, end = span
        target = norm_space(current_text[start:end])
        context = tail_context(current_text[:start])
        if not context:
            stop_reason = "empty_context"
            break

        scoring = gainratio_score_exact_strict_v1(
            tokenizer, model, device, context, target
        )
        if scoring is None:
            stop_reason = "invalid_loss"
            break

        suspicious = float(scoring["score"]) >= threshold
        candidate = {
            "step": step,
            "boundary_start": start,
            "boundary_end": end,
            "context": context,
            "target": target,
            **scoring,
            "threshold": threshold,
            "selected_for_deletion": suspicious,
        }
        candidates.append(candidate)

        if not suspicious:
            stop_reason = "score_below_threshold"
            break

        deleted_suffix = current_text[start:]
        deletions.append(
            {
                "step": step,
                "boundary_start": start,
                "target": target,
                "score": scoring["score"],
                "deleted_suffix": deleted_suffix,
                "deleted_char_count": len(deleted_suffix),
            }
        )
        current_text = current_text[:start].rstrip()
        stop_reason = "deleted_and_continue"

    return {
        "internal_id": row["internal_id"],
        "source_doc_id": row["source_doc_id"],
        "title": row.get("title", ""),
        "original_text": original_text,
        "sanitized_text": current_text,
        "modified": current_text != original_text,
        "original_char_count": len(original_text),
        "sanitized_char_count": len(current_text),
        "deleted_char_count": len(original_text) - len(current_text),
        "candidate_evaluations": candidates,
        "deletions": deletions,
        "deletion_step_count": len(deletions),
        "stop_reason": stop_reason,
    }


def compile_outputs(records_dir: Path, input_count: int, output_dir: Path) -> Dict[str, Any]:
    record_files = sorted(records_dir.glob("*.json"))
    records = [load_json(path) for path in record_files]

    sanitized_rows = [
        {
            "internal_id": row["internal_id"],
            "source_doc_id": row["source_doc_id"],
            "title": row.get("title", ""),
            "text": row["sanitized_text"],
        }
        for row in records
    ]
    candidate_rows = []
    deletion_rows = []
    for row in records:
        for candidate in row["candidate_evaluations"]:
            candidate_rows.append(
                {
                    "internal_id": row["internal_id"],
                    "source_doc_id": row["source_doc_id"],
                    **candidate,
                }
            )
        if row["modified"]:
            deletion_rows.append(
                {
                    "internal_id": row["internal_id"],
                    "source_doc_id": row["source_doc_id"],
                    "title": row.get("title", ""),
                    "original_char_count": row["original_char_count"],
                    "sanitized_char_count": row["sanitized_char_count"],
                    "deleted_char_count": row["deleted_char_count"],
                    "deletion_step_count": row["deletion_step_count"],
                    "deletions": row["deletions"],
                }
            )

    sanitized_path = output_dir / "gainratio_sanitized_corpus.jsonl"
    candidates_path = output_dir / "gainratio_candidate_scores.jsonl"
    deletions_path = output_dir / "gainratio_deletion_log.jsonl"
    write_jsonl(sanitized_path, sanitized_rows)
    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(deletions_path, deletion_rows)

    total_original_chars = sum(row["original_char_count"] for row in records)
    total_deleted_chars = sum(row["deleted_char_count"] for row in records)
    summary = {
        "input_document_count": input_count,
        "completed_document_count": len(records),
        "complete": len(records) == input_count,
        "modified_document_count": sum(row["modified"] for row in records),
        "modified_document_rate": (
            sum(row["modified"] for row in records) / len(records) if records else None
        ),
        "candidate_evaluation_count": len(candidate_rows),
        "deletion_step_count": sum(row["deletion_step_count"] for row in records),
        "total_original_chars": total_original_chars,
        "total_deleted_chars": total_deleted_chars,
        "deleted_char_rate": (
            total_deleted_chars / total_original_chars if total_original_chars else None
        ),
        "stop_reason_counts": dict(Counter(row["stop_reason"] for row in records)),
        "sanitized_corpus": str(sanitized_path),
        "sanitized_corpus_sha256": sha256_file(sanitized_path),
        "candidate_scores": str(candidates_path),
        "deletion_log": str(deletions_path),
    }
    write_json_atomic(output_dir / "gainratio_sanitization_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    records_dir = output_dir / "records"
    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)
    records_dir.mkdir(parents=True, exist_ok=True)

    threshold_report = load_json(Path(args.threshold_json))
    threshold = float(threshold_report["threshold"])
    rows = read_jsonl(input_path)
    selected_rows = rows[: args.limit] if args.limit > 0 else rows

    print(f"Input documents: {len(rows)}")
    print(f"Documents selected this run: {len(selected_rows)}")
    print(f"Threshold: {threshold}")
    print(f"Model: {args.model_path}")
    print("Selection input fields:", sorted(rows[0].keys()))
    forbidden = {"change", "label", "watermark_text", "tuple"}
    leaked = forbidden.intersection(rows[0].keys())
    if leaked:
        raise RuntimeError(f"Forbidden ground-truth fields in attack input: {leaked}")

    tokenizer, model, device = load_qwen_model(args.model_path)

    for index, row in enumerate(selected_rows):
        record_path = records_dir / f"{index:05d}.json"
        if record_path.exists():
            continue
        result = process_document(
            row, tokenizer, model, device, threshold, args.max_steps
        )
        write_json_atomic(record_path, result)
        if (index + 1) % 25 == 0 or index + 1 == len(selected_rows):
            print(
                f"processed {index + 1}/{len(selected_rows)}; "
                f"modified={result['modified']}; "
                f"steps={result['deletion_step_count']}",
                flush=True,
            )
        if torch.cuda.is_available() and (index + 1) % 200 == 0:
            torch.cuda.empty_cache()

    summary = compile_outputs(records_dir, len(selected_rows), output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.limit == 0 and not summary["complete"]:
        raise RuntimeError("Formal sanitization is incomplete")
    print("GAINRATIO SANITIZATION PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2_common_aligned import norm_space, read_jsonl, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--sanitized", required=True)
    parser.add_argument("--inject-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    original_rows = read_jsonl(Path(args.original))
    sanitized_rows = read_jsonl(Path(args.sanitized))
    inject = json.loads(Path(args.inject_json).read_text(encoding="utf-8"))

    original = {row["internal_id"]: row for row in original_rows}
    sanitized = {row["internal_id"]: row for row in sanitized_rows}
    if set(original) != set(sanitized):
        raise RuntimeError("Original and sanitized document IDs differ")

    known_texts = sorted(
        {
            norm_space(candidate[0])
            for unit in inject
            for candidate in unit[2]
            if candidate and norm_space(candidate[0])
        },
        key=len,
        reverse=True,
    )

    details = []
    tp = fp = fn = tn = 0
    occurrences_before = occurrences_after = 0
    fully_cleaned_watermarked = 0
    partially_cleaned_watermarked = 0

    for internal_id in sorted(original):
        before_text = norm_space(original[internal_id]["text"])
        after_text = norm_space(sanitized[internal_id]["text"])
        present_before = [
            text for text in known_texts if text in before_text
        ]
        present_after = [
            text for text in known_texts if text in after_text
        ]
        watermarked = bool(present_before)
        modified = before_text != after_text

        occurrences_before += len(present_before)
        occurrences_after += len(present_after)

        if watermarked and modified:
            tp += 1
        elif not watermarked and modified:
            fp += 1
        elif watermarked and not modified:
            fn += 1
        else:
            tn += 1

        if watermarked and not present_after:
            fully_cleaned_watermarked += 1
        elif watermarked and len(present_after) < len(present_before):
            partially_cleaned_watermarked += 1

        if watermarked or modified or present_after:
            details.append(
                {
                    "internal_id": internal_id,
                    "source_doc_id": original[internal_id]["source_doc_id"],
                    "watermarked_by_exact_scan": watermarked,
                    "modified": modified,
                    "watermark_texts_before": present_before,
                    "watermark_texts_after": present_after,
                }
            )

    watermarked_count = tp + fn
    modified_count = tp + fp
    summary = {
        "experiment_role": "aligned_end_to_end_sensitivity",
        "scoring_version": "aligned_minimal_v1",
        "note": (
            "Ground truth is used only after attack completion for audit, "
            "never for selection."
        ),
        "document_count": len(original),
        "known_watermark_text_count": len(known_texts),
        "watermarked_document_count_by_exact_scan": watermarked_count,
        "modified_document_count": modified_count,
        "true_positive_modified_watermarked": tp,
        "false_positive_modified_clean": fp,
        "false_negative_unmodified_watermarked": fn,
        "true_negative_unmodified_clean": tn,
        "document_precision": tp / modified_count if modified_count else None,
        "document_recall": tp / watermarked_count if watermarked_count else None,
        "clean_document_modification_rate": (
            fp / (fp + tn) if fp + tn else None
        ),
        "exact_watermark_occurrences_before": occurrences_before,
        "exact_watermark_occurrences_after": occurrences_after,
        "exact_watermark_occurrence_removal_rate": (
            (occurrences_before - occurrences_after) / occurrences_before
            if occurrences_before
            else None
        ),
        "documents_with_any_exact_watermark_after": sum(
            bool(row["watermark_texts_after"]) for row in details
        ),
        "fully_cleaned_watermarked_document_count": (
            fully_cleaned_watermarked
        ),
        "fully_cleaned_watermarked_document_rate": (
            fully_cleaned_watermarked / watermarked_count
            if watermarked_count
            else None
        ),
        "partially_cleaned_watermarked_document_count": (
            partially_cleaned_watermarked
        ),
    }

    write_json_atomic(
        Path(args.output),
        {"summary": summary, "details": details},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("ALIGNED POSTHOC GROUND-TRUTH AUDIT PASS")


if __name__ == "__main__":
    main()

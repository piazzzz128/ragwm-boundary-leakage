from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb

from e2_common import sha256_file, write_json_atomic, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--collection", default="nfcorpus_contriever_cosine")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_path = output_dir / "watermarked_attack_input.jsonl"
    audit_path = output_dir / "watermarked_export_audit.jsonl"
    summary_path = output_dir / "export_summary.json"

    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_collection(name=args.collection)
    data = collection.get(include=["documents", "metadatas"])

    attack_rows = []
    audit_rows = []
    for internal_id, document, metadata in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        metadata = metadata or {}
        source_doc_id = str(metadata.get("id", internal_id))
        title = str(metadata.get("title", "") or "")
        text = document or ""

        # Attack input deliberately excludes the change flag and watermark labels.
        attack_rows.append(
            {
                "internal_id": str(internal_id),
                "source_doc_id": source_doc_id,
                "title": title,
                "text": text,
            }
        )
        audit_rows.append(
            {
                "internal_id": str(internal_id),
                "source_doc_id": source_doc_id,
                "title": title,
                "text": text,
                "metadata": metadata,
            }
        )

    attack_rows.sort(key=lambda row: row["internal_id"])
    audit_rows.sort(key=lambda row: row["internal_id"])
    write_jsonl(attack_path, attack_rows)
    write_jsonl(audit_path, audit_rows)

    summary = {
        "database": args.db,
        "collection": args.collection,
        "document_count": len(attack_rows),
        "attack_input": str(attack_path),
        "attack_input_sha256": sha256_file(attack_path),
        "audit_export": str(audit_path),
        "audit_export_sha256": sha256_file(audit_path),
        "attack_input_fields": sorted(attack_rows[0].keys()) if attack_rows else [],
        "selection_input_contains_change_flag": False,
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(attack_rows) != 3633:
        raise RuntimeError(f"Expected 3633 documents, exported {len(attack_rows)}")
    print("WATERMARKED CORPUS EXPORT PASS")


if __name__ == "__main__":
    main()

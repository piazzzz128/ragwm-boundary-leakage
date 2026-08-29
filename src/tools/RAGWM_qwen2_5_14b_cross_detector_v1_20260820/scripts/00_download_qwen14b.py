#!/usr/bin/env python3
"""Download and revision-freeze the official Qwen2.5-14B base snapshot.

This helper is optional when an already complete local snapshot exists. It
resolves the requested revision to an immutable commit before downloading and
records that commit for the subsequent SHA-256 freeze step.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from cross_detector_common import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if args.manifest_output.exists():
        raise FileExistsError(
            "Download manifest already exists; refusing to overwrite frozen "
            f"provenance: {args.manifest_output}"
        )

    api = HfApi(endpoint=args.endpoint)
    info = api.model_info(args.repo_id, revision=args.revision)
    resolved_revision = str(info.sha)
    if len(resolved_revision) != 40:
        raise RuntimeError(
            f"Expected a 40-character immutable revision, got {resolved_revision!r}"
        )

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=resolved_revision,
        local_dir=str(args.local_dir),
        endpoint=args.endpoint,
        max_workers=args.max_workers,
    )

    files = sorted(
        [
            {
                "path": str(path.relative_to(args.local_dir)),
                "size_bytes": path.stat().st_size,
            }
            for path in args.local_dir.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        ],
        key=lambda item: item["path"],
    )
    manifest = {
        "artifact_role": "qwen2_5_14b_base_download_provenance_v1",
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "local_dir": str(args.local_dir),
        "download_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
        "next_step": (
            "Run 01_freeze_model_snapshot.py to hash every model and tokenizer "
            "file before scoring."
        ),
    }
    write_json_atomic(args.manifest_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("QWEN2.5-14B DOWNLOAD AND REVISION FREEZE PASS")


if __name__ == "__main__":
    main()

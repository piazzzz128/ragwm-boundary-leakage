from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Same sentence-boundary rule used by the E2 strict-v1 pipeline."""
    if not text:
        return []
    boundaries = list(re.finditer(r"(?<=[.!?])\s+", text))
    raw_spans: List[Tuple[int, int]] = []
    start = 0
    for match in boundaries:
        raw_spans.append((start, match.start()))
        start = match.end()
    raw_spans.append((start, len(text)))

    spans: List[Tuple[int, int]] = []
    for raw_start, raw_end in raw_spans:
        segment = text[raw_start:raw_end]
        if not segment:
            continue
        left_trim = len(segment) - len(segment.lstrip())
        right_trim = len(segment) - len(segment.rstrip())
        start_pos = raw_start + left_trim
        end_pos = raw_end - right_trim
        if end_pos > start_pos:
            spans.append((start_pos, end_pos))
    return spans


def valid_suffix_candidates(text: str, min_target_chars: int = 25):
    """Candidate suffix cuts that retain nonempty context and begin at a valid sentence."""
    candidates = []
    for start, end in sentence_spans(text):
        target = norm_space(text[start:end])
        prefix = text[:start].rstrip()
        if not prefix:
            continue
        if len(target) < min_target_chars:
            continue
        candidates.append(
            {
                "boundary_start": start,
                "boundary_end": end,
                "target": target,
                "deleted_char_count": len(text) - start,
            }
        )
    return candidates

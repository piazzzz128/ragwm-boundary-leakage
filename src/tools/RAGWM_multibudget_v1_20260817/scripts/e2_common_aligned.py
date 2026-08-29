from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_CONTEXT_CHARS = 1200
MIN_TARGET_CHARS = 25
MAX_CONTEXT_TOKENS = 1024
MAX_TARGET_TOKENS = 128
DEFAULT_MODEL_PATH = "/root/autodl-tmp/ragwm_storage/local_models/Qwen2.5-7B"


def norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def tail_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = norm_space(text)
    if len(text) <= max_chars:
        return text
    cut = text[-max_chars:]
    first_space = cut.find(" ")
    if first_space > 0:
        cut = cut[first_space + 1 :]
    return cut.strip()


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return sentence-like spans while preserving original-text positions."""
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


def last_eligible_sentence_span(
    text: str,
    min_target_chars: int = MIN_TARGET_CHARS,
) -> Optional[Tuple[int, int]]:
    for start, end in reversed(sentence_spans(text)):
        target = norm_space(text[start:end])
        if len(target) >= min_target_chars:
            return start, end
    return None


def load_json(path: Path) -> Any:
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


def load_qwen_model(model_path: str = DEFAULT_MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def target_nll_aligned(
    tokenizer,
    model,
    device: str,
    context: str,
    target: str,
):
    """Minimal correction: score target tokens 2..n in both conditions.

    No BOS token and no separator are introduced. This preserves every other
    strict-v1 scoring choice while removing first-target-token asymmetry.
    """
    context = (context or "").strip()
    target = (target or "").strip()

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TARGET_TOKENS,
    )["input_ids"]
    if len(target_ids) < 2:
        return None

    context_ids = []
    if context:
        context_ids = tokenizer(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
        )["input_ids"]

    input_ids = context_ids + target_ids
    start = len(context_ids)
    end = len(input_ids)

    # Exclude the first target token in BOTH target-only and conditioned NLL.
    label_positions = list(range(start + 1, end))
    if not label_positions:
        return None
    pred_positions = [position - 1 for position in label_positions]

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(ids).logits
    pred_logits = logits[0, pred_positions, :]
    labels = ids[0, label_positions]

    loss = torch.nn.functional.cross_entropy(
        pred_logits.float(),
        labels,
        reduction="mean",
    )
    return float(loss.item())


def gainratio_score_aligned(
    tokenizer,
    model,
    device: str,
    context: str,
    target: str,
):
    loss_target = target_nll_aligned(
        tokenizer, model, device, "", target
    )
    loss_conditioned = target_nll_aligned(
        tokenizer, model, device, context, target
    )
    if loss_target is None or loss_conditioned is None or loss_target <= 0:
        return None

    delta_loss = loss_target - loss_conditioned
    gain_ratio = delta_loss / loss_target
    score = -gain_ratio

    return {
        # Generic names are retained for compatibility with the original logs.
        "loss_target": loss_target,
        "loss_target_given_context": loss_conditioned,
        "delta_loss": delta_loss,
        "gain_ratio": gain_ratio,
        "score": score,
        "scoring_version": "aligned_minimal_v1",
        "alignment_policy": "exclude_first_target_token_in_both_conditions",
    }

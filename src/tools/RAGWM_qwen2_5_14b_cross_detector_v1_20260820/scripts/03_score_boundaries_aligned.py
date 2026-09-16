#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cross_detector_common import (
    load_json,
    sha256_file,
    write_json_atomic,
)
from protocol_contract import (
    MAX_CONTEXT_TOKENS,
    MAX_TARGET_TOKENS,
    aligned_label_positions,
)


@torch.inference_mode()
def target_nll_aligned(
    tokenizer,
    model,
    device: str,
    context: str,
    target: str,
) -> Optional[float]:
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
    label_positions = aligned_label_positions(start, len(target_ids))
    if not label_positions:
        return None
    prediction_positions = [position - 1 for position in label_positions]

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False).logits
    prediction_logits = logits[0, prediction_positions, :]
    labels = ids[0, label_positions]
    loss = torch.nn.functional.cross_entropy(
        prediction_logits.float(), labels, reduction="mean"
    )
    if not torch.isfinite(loss):
        return None
    return float(loss.item())


@torch.inference_mode()
def score_one(
    row: Dict[str, Any], tokenizer, model, device: str
) -> Optional[Dict[str, float]]:
    context = str(row.get("context") or row.get("context_text") or "")
    target = str(row.get("target") or row.get("target_text") or "")

    loss_target = target_nll_aligned(
        tokenizer, model, device, "", target
    )
    loss_conditioned = target_nll_aligned(
        tokenizer, model, device, context, target
    )
    if (
        loss_target is None
        or loss_conditioned is None
        or loss_target <= 0
    ):
        return None

    delta_loss = loss_target - loss_conditioned
    gain_ratio = delta_loss / loss_target
    return {
        "loss_target_aligned": float(loss_target),
        "loss_target_given_context_aligned": float(loss_conditioned),
        "delta_loss_aligned": float(delta_loss),
        "gain_ratio_aligned": float(gain_ratio),
        "score_aligned": float(-gain_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-dir", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen2.5-14B")
    parser.add_argument(
       "--dataset", required=True, choices=["nfcorpus", "trec-covid", "nq"]
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    preflight = load_json(args.preflight)
    if not preflight.get("pass"):
        raise RuntimeError("Preflight is not marked pass")
    if Path(preflight["model_dir"]) != args.model_path:
        raise RuntimeError("Model path differs from frozen preflight")
    if preflight.get("model_name") != "Qwen2.5-14B":
        raise RuntimeError("Preflight model is not Qwen2.5-14B")

    if args.reset:
        if args.records_dir.exists():
            shutil.rmtree(args.records_dir)
        for path in (args.output, args.manifest_output):
            if path.exists():
                path.unlink()
    if args.output.exists() and args.limit == 0:
        raise FileExistsError(
            f"Frozen score output already exists; refusing overwrite: {args.output}"
        )

    rows = load_json(args.input)
    expected_count = {
    "nfcorpus": 446,
    "trec-covid": 458,
    "nq": 442,
}[args.dataset]
    expected_each_label = expected_count // 2
    label_counts = {
        "clean": sum(int(row["label"]) == 0 for row in rows),
        "injected": sum(int(row["label"]) == 1 for row in rows),
    }
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {args.dataset} rows, found {len(rows)}"
        )
    if label_counts != {
        "clean": expected_each_label,
        "injected": expected_each_label,
    }:
        raise RuntimeError(f"Unexpected label counts: {label_counts}")

    selected = rows[: args.limit] if args.limit > 0 else rows
    args.records_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("Formal Qwen2.5-14B scoring requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal Qwen2.5-14B scoring requires BF16 support")
    dtype = torch.bfloat16
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.eval()
    model.config.use_cache = False

    failures = []
    for index, row in enumerate(selected):
        record_path = args.records_dir / f"{index:05d}.json"
        if record_path.exists():
            existing = load_json(record_path)
            if (
                existing.get("input_index") != index
                or existing.get("dataset") != args.dataset
                or existing.get("model") != args.model_name
            ):
                raise RuntimeError(
                    f"Resume record identity mismatch: {record_path}"
                )
            continue

        scoring = score_one(row, tokenizer, model, device)
        if scoring is None:
            failures.append(
                {"input_index": index, "record_id": row.get("id")}
            )
            continue

        context = str(row.get("context") or row.get("context_text") or "")
        target = str(row.get("target") or row.get("target_text") or "")
        source_doc_id = str(
            row.get("source_doc_id")
            or row.get("doc_id")
            or row.get("document_id")
            or row.get("chroma_id")
            or row.get("id")
        )
        output_row = {
            "experiment_role": "cross_detector_lm_robustness_aligned_v1",
            "dataset": args.dataset,
            "model": args.model_name,
            "model_path": str(args.model_path),
            "input_index": index,
            "record_id": row.get("id"),
            "label": int(row["label"]),
            "group_id": row.get("group_id"),
            "source_doc_id": source_doc_id,
            "boundary_type": row.get("boundary_type") or row.get("kind"),
            "context": context,
            "target": target,
            "context_char_count": len(context),
            "target_char_count": len(target),
            "alignment_policy": (
                "exclude_first_target_token_in_both_conditions"
            ),
            "no_bos": True,
            "no_separator": True,
            **scoring,
        }
        write_json_atomic(record_path, output_row)

        if (index + 1) % 25 == 0 or index + 1 == len(selected):
            print(
                f"{args.dataset}: scored {index + 1}/{len(selected)}",
                flush=True,
            )
        if torch.cuda.is_available() and (index + 1) % 100 == 0:
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record_paths = sorted(args.records_dir.glob("*.json"))
    scored = [load_json(path) for path in record_paths]
    scored.sort(key=lambda row: int(row["input_index"]))

    complete = len(scored) == len(selected) and not failures
    if args.limit == 0 and not complete:
        raise RuntimeError(
            f"Formal scoring incomplete: {len(scored)}/{len(selected)}, "
            f"failures={failures}"
        )

    write_json_atomic(args.output, scored)
    manifest = {
        "experiment_role": "cross_detector_lm_robustness_aligned_v1",
        "dataset": args.dataset,
        "model": args.model_name,
        "model_path": str(args.model_path),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "preflight": str(args.preflight),
        "preflight_sha256": sha256_file(args.preflight),
        "precision": "bfloat16",
        "quantized": False,
        "alignment_policy": (
            "exclude_first_target_token_in_both_conditions"
        ),
        "no_bos": True,
        "no_separator": True,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_target_tokens": MAX_TARGET_TOKENS,
        "requested_record_count": len(selected),
        "completed_record_count": len(scored),
        "failed_record_count": len(failures),
        "failures": failures,
        "complete": complete,
        "scores": str(args.output),
        "scores_sha256": sha256_file(args.output),
        "warning": (
            "Qwen2.5-14B cross-detector extension only; do not overwrite "
            "Qwen2.5-7B or sealed strict-v1 scores."
        ),
    }
    write_json_atomic(args.manifest_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if complete:
        print(f"QWEN2.5-14B {args.dataset.upper()} ALIGNED SCORING PASS")


if __name__ == "__main__":
    main()

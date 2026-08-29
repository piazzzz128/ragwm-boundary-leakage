#!/usr/bin/env python3
"""Validate and SHA-256-freeze a local Qwen2.5-14B base snapshot."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from packaging.version import Version
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from cross_detector_common import load_json, sha256_file, write_json_atomic


CORE_CANDIDATES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors.index.json",
)


def model_files(model_dir: Path) -> tuple[list[Path], list[str], int | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(
            "Formal run requires the official unquantized safetensors index: "
            f"{index_path}"
        )
    index = load_json(index_path)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise RuntimeError("Weight index contains no shard names")
    shards = [model_dir / name for name in shard_names]
    missing_shards = [str(path) for path in shards if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(
            "Missing model shards:\n" + "\n".join(missing_shards)
        )

    core = [model_dir / name for name in CORE_CANDIDATES]
    core = [path for path in core if path.is_file()]
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(model_dir / "config.json")
    if not (model_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(model_dir / "tokenizer.json")

    total_size = index.get("metadata", {}).get("total_size")
    return sorted(set(core + shards)), shard_names, total_size


def tensor_dtype_counts(model_dir: Path, shards: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard_name in shards:
        shard = model_dir / shard_name
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtype = str(handle.get_slice(key).get_dtype())
                counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def gpu_report(min_gpu_gib: float) -> tuple[dict, dict[str, bool]]:
    if not torch.cuda.is_available():
        return {"available": False}, {
            "cuda_available": False,
            "bf16_supported": False,
            "single_gpu_memory_meets_minimum": False,
        }
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024**3)
    report = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_0_name": props.name,
        "device_0_total_memory_gib": total_gib,
        "device_0_compute_capability": [props.major, props.minor],
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "required_execution_mode": "single_gpu_bfloat16_unquantized",
        "minimum_total_memory_gib": min_gpu_gib,
        "recommended_total_memory_gib": 48,
    }
    checks = {
        "cuda_available": True,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "single_gpu_memory_meets_minimum": total_gib >= min_gpu_gib,
    }
    return report, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--min-gpu-gib", type=float, default=40.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Frozen model manifest exists; refusing overwrite: {args.output}"
        )
    if not args.model_dir.is_dir():
        raise NotADirectoryError(args.model_dir)

    files, shards, index_total_size = model_files(args.model_dir)
    file_hashes = {
        str(path.relative_to(args.model_dir)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    }
    model_content_bytes = sum(item["size_bytes"] for item in file_hashes.values())
    weight_shard_bytes = sum((args.model_dir / name).stat().st_size for name in shards)
    dtype_counts = tensor_dtype_counts(args.model_dir, shards)

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer_ids = list(tokenizer.get_vocab().values())
    tokenizer_max_id = max(tokenizer_ids)

    source = None
    source_sha256 = None
    source_checks = {"source_manifest_optional_or_valid": True}
    if args.source_manifest is not None:
        source = load_json(args.source_manifest)
        source_sha256 = sha256_file(args.source_manifest)
        source_checks = {
            "source_manifest_optional_or_valid": (
                source.get("repo_id") == "Qwen/Qwen2.5-14B"
                and len(str(source.get("resolved_revision", ""))) == 40
                and Path(source.get("local_dir", "")) == args.model_dir
            )
        }

    gpu, gpu_checks = gpu_report(args.min_gpu_gib)
    checks = {
        "model_type_is_qwen2": getattr(config, "model_type", None) == "qwen2",
        "architecture_is_qwen2_for_causal_lm": (
            getattr(config, "architectures", None) == ["Qwen2ForCausalLM"]
        ),
        "hidden_layers_are_48": getattr(config, "num_hidden_layers", None) == 48,
        "attention_heads_are_40": getattr(config, "num_attention_heads", None) == 40,
        "kv_heads_are_8": getattr(config, "num_key_value_heads", None) == 8,
        "tokenizer_ids_fit_model_vocab": tokenizer_max_id < int(config.vocab_size),
        "transformers_version_at_least_4_37": Version(transformers.__version__) >= Version("4.37.0"),
        "weight_shards_present": len(shards) > 0,
        "all_weight_tensors_are_bfloat16": set(dtype_counts) == {"BF16"},
        "no_quantization_config": not bool(
            getattr(config, "quantization_config", None)
        ),
        "weight_size_matches_unquantized_14b_range": 25_000_000_000
        <= weight_shard_bytes
        <= 35_000_000_000,
        **source_checks,
        **gpu_checks,
    }

    report = {
        "artifact_role": "qwen2_5_14b_base_model_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "Qwen2.5-14B",
        "official_repo_id": "Qwen/Qwen2.5-14B",
        "model_stage": "base_pretrained",
        "official_parameter_count": "14.7B",
        "precision": "bfloat16",
        "quantized": False,
        "model_dir": str(args.model_dir),
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "source_manifest_sha256": source_sha256,
        "resolved_revision": source.get("resolved_revision") if source else None,
        "remote_revision_status": "recorded" if source else "missing_not_recorded",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "config_vocab_size": int(config.vocab_size),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "tokenizer_size": len(tokenizer),
        "tokenizer_max_id": int(tokenizer_max_id),
        "weight_index_total_size": index_total_size,
        "weight_shard_bytes": weight_shard_bytes,
        "tensor_dtype_counts": dtype_counts,
        "frozen_model_content_bytes": model_content_bytes,
        "weight_shards": shards,
        "files": file_hashes,
        "gpu": gpu,
        "checks": checks,
        "pass": all(checks.values()),
        "guard": (
            "This manifest freezes an unquantized BF16 Qwen2.5-14B base "
            "snapshot. Do not substitute an Instruct or quantized checkpoint."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise RuntimeError("QWEN2.5-14B MODEL FREEZE FAILED")
    print("QWEN2.5-14B MODEL FREEZE PASS")


if __name__ == "__main__":
    main()

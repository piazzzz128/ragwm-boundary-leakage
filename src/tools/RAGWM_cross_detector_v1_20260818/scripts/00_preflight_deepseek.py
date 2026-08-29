#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from cross_detector_common import load_json, sha256_file, write_json_atomic


EXPECTED = {
    "config.json": "80cafaa279bc26b96cad0976ec81d9bba7dc1ef55bd9c3428c15c2d56d9b32ac",
    "generation_config.json": "f95ae1c1a3518c924cc3b093a2fc97a5ef0a52a0b65b78505f1107fca598a17b",
    "tokenizer.json": "a08b02921f08548065a7b2ec13b2ffeed873231add60f9c3c7b08b04f2cc212a",
    "tokenizer_config.json": "4fd5a605324b47a246bd72153628256581196802396bf8f3e0259fd55b8f37ce",
    "pytorch_model.bin.index.json": "bbcc3c9995867aba35987f43cb4deeb37684c985a2e01cdbafd2b14e03575de4",
    "pytorch_model-00001-of-00002.bin": "950392f338e46bc0807acfedd86dc7da12745b8835b78e704e98d7e26d2e618d",
    "pytorch_model-00002-of-00002.bin": "9883470e0517cadf59a6f163e0c58e0854de662de0f7a68a1b28d95e209de34c",
}

EXPECTED_INPUTS = {
    "nfcorpus_boundaries": "e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218",
    "trec_boundaries": "d526aa72f48662521b67339f58849c1ea6c26de4042eb8b3dd842403f8ed293f",
    "attack_input": "52c72f64043e4a668b5084b75ac0cfa92f63b77c3493875cf788ee5021da8e38",
    "current_provider_e0": "583eb66725901070f709c6938e22ca3e350d9a39a580e4237a72361f9088d1f5",
    "current_provider_e1": "3e0253bdac9cc93878a64d641ddb6ed969144e61e38795f4423a8d0c8079c251",
}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verifier_counts(path: Path) -> dict[str, int]:
    rows = load_json(path)
    return {
        "rows": len(rows),
        "yes": sum(row[2][0] == 1 for row in rows),
        "no": sum(row[2][0] == 0 for row in rows),
        "unknown": sum(row[2][0] == 2 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--nfcorpus-boundaries", required=True, type=Path)
    parser.add_argument("--trec-boundaries", required=True, type=Path)
    parser.add_argument("--attack-input", required=True, type=Path)
    parser.add_argument("--current-provider-e0", required=True, type=Path)
    parser.add_argument("--current-provider-e1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    required_paths = [
        *(args.model_dir / name for name in EXPECTED),
        args.nfcorpus_boundaries,
        args.trec_boundaries,
        args.attack_input,
        args.current_provider_e0,
        args.current_provider_e1,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    index = load_json(args.model_dir / "pytorch_model.bin.index.json")
    shards = sorted(set(index.get("weight_map", {}).values()))
    expected_shards = [
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
    ]

    model_hashes = {
        name: sha256_file(args.model_dir / name) for name in EXPECTED
    }
    input_hashes = {
        "nfcorpus_boundaries": sha256_file(args.nfcorpus_boundaries),
        "trec_boundaries": sha256_file(args.trec_boundaries),
        "attack_input": sha256_file(args.attack_input),
        "current_provider_e0": sha256_file(args.current_provider_e0),
        "current_provider_e1": sha256_file(args.current_provider_e1),
    }

    nf_rows = load_json(args.nfcorpus_boundaries)
    trec_rows = load_json(args.trec_boundaries)
    nf_labels = {
        "clean": sum(int(row["label"]) == 0 for row in nf_rows),
        "injected": sum(int(row["label"]) == 1 for row in nf_rows),
    }
    trec_labels = {
        "clean": sum(int(row["label"]) == 0 for row in trec_rows),
        "injected": sum(int(row["label"]) == 1 for row in trec_rows),
    }

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer_vocab_ids = list(tokenizer.get_vocab().values())
    tokenizer_max_id = max(tokenizer_vocab_ids)

    e0_counts = verifier_counts(args.current_provider_e0)
    e1_counts = verifier_counts(args.current_provider_e1)

    checks = {
        "model_hashes_match_frozen": model_hashes == EXPECTED,
        "model_index_has_two_expected_shards": shards == expected_shards,
        "model_type_is_llama": getattr(config, "model_type", None) == "llama",
        "architecture_is_llama_for_causal_lm": (
            getattr(config, "architectures", None) == ["LlamaForCausalLM"]
        ),
        "tokenizer_ids_fit_padded_model_vocab": (
            tokenizer_max_id < config.vocab_size
        ),
        "input_hashes_match_frozen": input_hashes == EXPECTED_INPUTS,
        "nfcorpus_is_446_balanced": (
            len(nf_rows) == 446
            and nf_labels == {"clean": 223, "injected": 223}
        ),
        "trec_is_458_balanced": (
            len(trec_rows) == 458
            and trec_labels == {"clean": 229, "injected": 229}
        ),
        "attack_input_is_3633": count_jsonl(args.attack_input) == 3633,
        "current_provider_e0_is_frozen_25_of_30": (
            e0_counts == {"rows": 30, "yes": 25, "no": 5, "unknown": 0}
        ),
        "current_provider_e1_is_frozen_4_of_30": (
            e1_counts == {"rows": 30, "yes": 4, "no": 26, "unknown": 0}
        ),
    }

    report = {
        "experiment_role": "cross_detector_lm_robustness_preflight_v1",
        "python": platform.python_version(),
        "model_name": "DeepSeek-LLM-7B-base",
        "model_dir": str(args.model_dir),
        "model_source": "ModelScope snapshot",
        "model_remote_revision": None,
        "model_remote_revision_status": "missing_not_recorded_at_download",
        "model_hashes": model_hashes,
        "model_index_parameter_entries": len(index.get("weight_map", {})),
        "model_index_shards": shards,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "config_vocab_size": int(config.vocab_size),
        "tokenizer_size": len(tokenizer),
        "tokenizer_max_id": int(tokenizer_max_id),
        "vocab_note": (
            "The model vocabulary is padded to 102400 rows; tokenizer size "
            "100015 is valid because all tokenizer IDs remain in range."
        ),
        "inputs": {
            "nfcorpus_boundaries": str(args.nfcorpus_boundaries),
            "trec_boundaries": str(args.trec_boundaries),
            "attack_input": str(args.attack_input),
            "current_provider_e0": str(args.current_provider_e0),
            "current_provider_e1": str(args.current_provider_e1),
        },
        "input_hashes": input_hashes,
        "nfcorpus_rows": len(nf_rows),
        "nfcorpus_labels": nf_labels,
        "trec_rows": len(trec_rows),
        "trec_labels": trec_labels,
        "attack_input_rows": count_jsonl(args.attack_input),
        "current_provider_e0_counts": e0_counts,
        "current_provider_e1_counts": e1_counts,
        "checks": checks,
        "pass": all(checks.values()),
        "warning": (
            "This extension must not replace the sealed strict-v1 main result "
            "or the Qwen2.5-7B aligned multi-budget result."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise RuntimeError("DEEPSEEK CROSS-DETECTOR PREFLIGHT FAILED")
    print("DEEPSEEK CROSS-DETECTOR PREFLIGHT PASS")


if __name__ == "__main__":
    main()

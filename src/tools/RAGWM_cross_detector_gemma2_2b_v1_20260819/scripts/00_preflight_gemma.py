#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from cross_detector_common import load_json, sha256_file, write_json_atomic


EXPECTED = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "README.md": "8ee47bb6d2af9b4638afbd0683ded6c6bac5e9e01b2b8b4fd1c218182be4d2e9",
    "config.json": "56ee4e528fedfe70b2fabb0704eec9f2ef94461de936928220a3d6e7fac5ef98",
    "configuration.json": "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9",
    "generation_config.json": "4c8b8739d9583289c138eb8f412bdd4854b6fed6ba65ba28a323c0721c9a5973",
    "model-00001-of-00003.safetensors": "1425aa066ec77e3eb79aac14a5bdea3ebcec46aa5c96cd40608c5c1fd70d193d",
    "model-00002-of-00003.safetensors": "96c111d3dcdbde9271595e463b5d9f7fc4810ad8b79e736309c0a1833e6c0d35",
    "model-00003-of-00003.safetensors": "4e08abc64d1767fdacd2c94da7f2ec4b8c65b25b19a53e87d19dc432901b5f02",
    "model.safetensors.index.json": "e5c6f26fbe40fc3712f4df57da9b89c58b033714316a4547d052b37f43217e8e",
    "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
    "tokenizer.json": "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": "bb8245b8d39a065fdf869ee4fce48db9cc491473eb7ff1b41d321fb4caf540d1",
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

    index = load_json(args.model_dir / "model.safetensors.index.json")
    shards = sorted(set(index.get("weight_map", {}).values()))
    expected_shards = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
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
        "model_index_has_three_expected_shards": shards == expected_shards,
        "model_type_is_gemma2": getattr(config, "model_type", None) == "gemma2",
        "architecture_is_gemma2_for_causal_lm": (
            getattr(config, "architectures", None) == ["Gemma2ForCausalLM"]
        ),
        "tokenizer_ids_fit_model_vocab": (
            tokenizer_max_id < config.vocab_size
        ),
        "config_and_tokenizer_vocab_are_256000": (
            int(config.vocab_size) == 256000 and len(tokenizer) == 256000
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
        "model_name": "Gemma-2-2B-base",
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
        "vocab_note": "Config and tokenizer both report 256000 entries.",
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
        raise RuntimeError("GEMMA CROSS-DETECTOR PREFLIGHT FAILED")
    print("GEMMA CROSS-DETECTOR PREFLIGHT PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module():
    path = SCRIPTS / "05_pairwise_qwen14b_vs_qwen7b.py"
    spec = importlib.util.spec_from_file_location("qwen_scale_pairwise", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pairwise module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_module()
    rows7 = [
        {"label": 0, "score_aligned": 0.1, "source_doc_id": "a"},
        {"label": 1, "score_aligned": 0.8, "source_doc_id": "a"},
        {"label": 0, "score_aligned": 0.2, "source_doc_id": "b"},
        {"label": 1, "score_aligned": 0.7, "source_doc_id": "b"},
    ]
    rows14 = [
        {"label": 0, "score_aligned": 0.05, "source_doc_id": "a"},
        {"label": 1, "score_aligned": 0.9, "source_doc_id": "a"},
        {"label": 0, "score_aligned": 0.15, "source_doc_id": "b"},
        {"label": 1, "score_aligned": 0.85, "source_doc_id": "b"},
    ]
    report = module.paired_cluster_bootstrap(
        rows7, rows14, replicates=100, seed=20260609
    )
    assert report["difference_direction"] == "Qwen2.5-14B minus Qwen2.5-7B"
    assert report["qwen2_5_7b"]["roc_auc"] == 1.0
    assert report["qwen2_5_14b"]["roc_auc"] == 1.0
    assert report["valid_replicates"] == 100
    print("QWEN2.5 14B-VS-7B PAIRED SMOKE TEST PASS")


if __name__ == "__main__":
    main()

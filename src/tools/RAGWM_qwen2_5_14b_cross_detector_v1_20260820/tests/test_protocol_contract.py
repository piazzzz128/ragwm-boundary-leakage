#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def main() -> None:
    import protocol_contract as module
    assert module.MAX_CONTEXT_TOKENS == 1024
    assert module.MAX_TARGET_TOKENS == 128

    target_only_labels = module.aligned_label_positions(0, 3)
    conditioned_labels = module.aligned_label_positions(2, 3)
    assert target_only_labels == [1, 2]
    assert conditioned_labels == [3, 4]
    assert [position - 1 for position in target_only_labels] == [0, 1]
    assert [position - 1 for position in conditioned_labels] == [2, 3]
    assert module.aligned_label_positions(0, 1) == []
    print("QWEN2.5-14B ALIGNED PROTOCOL CONTRACT TEST PASS")


if __name__ == "__main__":
    main()

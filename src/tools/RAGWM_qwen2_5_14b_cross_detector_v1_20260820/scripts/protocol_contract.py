from __future__ import annotations


MAX_CONTEXT_TOKENS = 1024
MAX_TARGET_TOKENS = 128
ALIGNMENT_POLICY = "exclude_first_target_token_in_both_conditions"
NO_BOS = True
NO_SEPARATOR = True
SCORE_FORMULA = "score=-(loss_target-loss_target_given_context)/loss_target"


def aligned_label_positions(context_length: int, target_length: int) -> list[int]:
    """Return target-token label positions after excluding target token one."""
    if context_length < 0 or target_length < 2:
        return []
    start = context_length
    return list(range(start + 1, start + target_length))

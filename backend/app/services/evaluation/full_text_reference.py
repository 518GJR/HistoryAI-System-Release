"""Deterministic helpers for edition-controlled full-text references."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def normalize_layout_whitespace(text: str) -> str:
    """Ignore layout whitespace only; punctuation and wording stay significant."""

    return re.sub(r"\s+", "", text)


def inspect_reference_text(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        raise ValueError("Reference text is empty")
    return {
        "encoding": "utf-8",
        "byte_length": len(raw),
        "character_length": len(text),
        "line_count": len(lines),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first_nonempty_line": nonempty[0],
        "last_nonempty_line": nonempty[-1],
        "normalized_character_length": len(normalize_layout_whitespace(text)),
    }


def compare_full_text(candidate: str, reference: str) -> dict:
    """Return exact normalized agreement or a mandatory human-review result.

    The comparison deliberately does not repair, substitute, or guess textual
    variants. Any non-layout difference remains visible for manual review.
    """

    candidate_normalized = normalize_layout_whitespace(candidate)
    reference_normalized = normalize_layout_whitespace(reference)
    if candidate_normalized == reference_normalized:
        return {
            "status": "exact_after_layout_normalization",
            "matches": True,
            "requires_human_review": False,
            "first_mismatch_index": None,
        }
    limit = min(len(candidate_normalized), len(reference_normalized))
    mismatch = next(
        (index for index in range(limit) if candidate_normalized[index] != reference_normalized[index]),
        limit,
    )
    return {
        "status": "textual_variant_requires_human_review",
        "matches": False,
        "requires_human_review": True,
        "first_mismatch_index": mismatch,
        "candidate_normalized_length": len(candidate_normalized),
        "reference_normalized_length": len(reference_normalized),
    }

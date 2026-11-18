import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / (
    "docs/evaluation/workbench/unit5-20-formalization-20261030/formal-import-20261101/"
    "scripts/build_candidate_vector_index_lesson4.py"
)
SPEC = importlib.util.spec_from_file_location("lesson4_vector_builder", SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def document(source_id, lesson_number, excerpt=None):
    return {
        "source_id": source_id,
        "lesson_number": lesson_number,
        "excerpt": excerpt or f"正文-{source_id}",
        "embedding_text": f"教材原文：{excerpt or source_id}",
    }


def test_partition_associations_keeps_active_and_pending_separate():
    rows = [
        {"knowledge_id": "K1", "fragment_id": "F1", "source_id": "S1", "relation_status": "active"},
        {
            "knowledge_id": "K2",
            "fragment_id": "F2",
            "source_id": "S2",
            "relation_status": "pending_manual_source_review",
        },
    ]
    active, pending = BUILDER.partition_associations(rows)
    assert active == [rows[0]]
    assert pending == [rows[1]]


def test_partition_associations_rejects_unknown_status():
    with pytest.raises(RuntimeError, match="Unexpected relation_status"):
        BUILDER.partition_associations([
            {"knowledge_id": "K1", "fragment_id": "F1", "source_id": "S1", "relation_status": "ignored"}
        ])


def test_derive_incremental_documents_reuses_old_and_selects_only_16_lesson4():
    formal = [document("UNIT1", 1)]
    reused = [document("OLD-1", 5), document("OLD-2", 20)]
    new = [document(f"L04-F{number:02d}", 4) for number in range(1, 17)]
    base = formal + reused
    current = reused + new
    active_ids = {row["source_id"] for row in current}
    fragments = {
        row["source_id"]: {"lesson_number": row["lesson_number"]} for row in current
    }

    reused_result, new_result = BUILDER.derive_incremental_documents(
        base, current, 1, active_ids, fragments
    )

    assert reused_result == reused
    assert new_result == new
    assert all(row["lesson_number"] == 4 for row in new_result)


def test_derive_incremental_documents_rejects_changed_reused_metadata():
    old = document("OLD-1", 5)
    changed = document("OLD-1", 5, excerpt="已改变")
    new = [document(f"L04-F{number:02d}", 4) for number in range(1, 17)]
    fragments = {
        row["source_id"]: {"lesson_number": row["lesson_number"]}
        for row in [changed] + new
    }
    with pytest.raises(RuntimeError, match="metadata changed"):
        BUILDER.derive_incremental_documents(
            [document("UNIT1", 1), old],
            [changed] + new,
            1,
            set(fragments),
            fragments,
        )


def test_combine_reused_and_new_preserves_all_base_rows_exactly():
    base_documents = [document("BASE-1", 1), document("BASE-2", 5)]
    new_documents = [document("L04-F01", 4)]
    base_vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
    new_vectors = np.asarray([[10, 11, 12, 13]], dtype=np.float32)

    combined, documents = BUILDER.combine_reused_and_new(
        base_vectors, base_documents, new_vectors, new_documents
    )

    assert np.array_equal(combined[:2], base_vectors)
    assert np.array_equal(combined[2:], new_vectors)
    assert documents[:2] == base_documents
    assert documents[2:] == new_documents


def test_combine_rejects_duplicate_fragment_id():
    duplicate = document("DUPLICATE", 4)
    with pytest.raises(RuntimeError, match="not unique"):
        BUILDER.combine_reused_and_new(
            np.ones((1, 4), dtype=np.float32),
            [duplicate],
            np.ones((1, 4), dtype=np.float32),
            [duplicate],
        )

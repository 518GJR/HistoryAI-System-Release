import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from app.models import (
    HistoryKnowledge,
    SignedTextbookKnowledge,
    SignedTextbookKnowledgeSource,
    SignedTextbookSourceFragment,
)
from app.scripts.import_signed_textbook_knowledge import (
    import_signed_points,
    inspect_import_plan,
    load_repaired_bundle,
)
from app.services.signed_textbook_service import load_retrievable_signed_points


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LESSON4_BUNDLE = PROJECT_ROOT / (
    "docs/evaluation/workbench/unit5-20-formalization-20261030/formal-import-20261101/"
    "lesson4-source-crosswalk-20261101/lesson4_formal_import_candidate.json"
)


def _create_signed_tables(database_url):
    engine = create_engine(database_url)
    SignedTextbookKnowledge.__table__.create(engine, checkfirst=True)
    SignedTextbookSourceFragment.__table__.create(engine, checkfirst=True)
    SignedTextbookKnowledgeSource.__table__.create(engine, checkfirst=True)
    engine.dispose()


def _write_bundle(tmp_path, payload, name="lesson4-mutated.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_repaired_bundle_has_complete_signed_counts_and_lesson_coverage():
    bundle = load_repaired_bundle()

    assert len(bundle["points"]) == 338
    assert len({point["knowledge_id"] for point in bundle["points"]}) == 338
    assert all(point["content"].strip() for point in bundle["points"])
    assert len(bundle["sources"]) == 340
    assert len(bundle["associations"]) == 369
    assert len(bundle["expanded_associations"]) == 372
    assert set(bundle["active_by_point"]) == {
        point["knowledge_id"]
        for point in bundle["points"]
        if point["retrieval_status"] == "eligible"
    }
    assert set(bundle["active_by_point"]) == {
        point["knowledge_id"] for point in bundle["points"]
    } - {"L16-Q015"}
    assert {
        lesson: sum(point["lesson_number"] == lesson for point in bundle["points"])
        for lesson in range(5, 21)
    } == {
        5: 15,
        6: 28,
        7: 29,
        8: 30,
        9: 27,
        10: 19,
        11: 22,
        12: 22,
        13: 18,
        14: 22,
        15: 22,
        16: 19,
        17: 13,
        18: 18,
        19: 16,
        20: 18,
    }


def test_repaired_import_is_idempotent_and_preserves_existing_legacy_rows(tmp_path):
    database_url = "sqlite:///" + (tmp_path / "formal-import-rehearsal.sqlite").as_posix()
    engine = create_engine(database_url)
    HistoryKnowledge.__table__.create(engine, checkfirst=True)
    with Session(engine) as session, session.begin():
        session.add(HistoryKnowledge(title="现有第一单元保护行", content="不得被签核导入覆盖。"))
    engine.dispose()
    _create_signed_tables(database_url)

    first = import_signed_points(database_url)
    second = import_signed_points(database_url)
    retrievable = load_retrievable_signed_points(database_url)

    assert first == {
        "knowledge": {"inserted": 338, "updated": 0, "unchanged": 0},
        "fragments": {"inserted": 340, "updated": 0, "unchanged": 0},
        "associations": {"inserted": 372, "updated": 0, "unchanged": 0},
    }
    assert second == {
        "knowledge": {"inserted": 0, "updated": 0, "unchanged": 338},
        "fragments": {"inserted": 0, "updated": 0, "unchanged": 340},
        "associations": {"inserted": 0, "updated": 0, "unchanged": 372},
    }
    assert len(retrievable) == 337
    assert "L16-Q015" not in {point["knowledge_id"] for point in retrievable}

    engine = create_engine(database_url)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(HistoryKnowledge)) == 1
        assert session.scalar(select(func.count()).select_from(SignedTextbookKnowledge)) == 338
        assert session.scalar(select(func.count()).select_from(SignedTextbookSourceFragment)) == 340
        assert session.scalar(select(func.count()).select_from(SignedTextbookKnowledgeSource)) == 372
        pending = session.get(SignedTextbookKnowledge, "L16-Q015")
        assert pending is not None
        assert pending.content
        assert pending.content_status == "signed_fact_summary_pending_manual_source_review"
        assert pending.source_status == "pending_manual_source_review"
        l16_relations = session.scalars(
            select(SignedTextbookKnowledgeSource).where(
                SignedTextbookKnowledgeSource.knowledge_id == "L16-Q015"
            )
        ).all()
        assert {row.relation_status for row in l16_relations} == {
            "pending_manual_source_review"
        }
    engine.dispose()


def test_lesson4_incremental_first_import_dry_run_and_repeat(tmp_path):
    database_url = "sqlite:///" + (tmp_path / "lesson4-incremental.sqlite").as_posix()
    _create_signed_tables(database_url)
    bundle = load_repaired_bundle(bundle_file=LESSON4_BUNDLE)

    dry_run = inspect_import_plan(database_url, bundle)
    assert dry_run["safe_to_apply"] is True
    assert dry_run["database_before"] == {"knowledge": 0, "fragments": 0, "associations": 0}
    assert dry_run["database_expected_after"] == {
        "knowledge": 14,
        "fragments": 16,
        "associations": 20,
    }

    first = import_signed_points(database_url, bundle_file=LESSON4_BUNDLE, include_report=True)
    second = import_signed_points(database_url, bundle_file=LESSON4_BUNDLE, include_report=True)
    assert first["counts"] == {
        "knowledge": {"inserted": 14, "updated": 0, "unchanged": 0},
        "fragments": {"inserted": 16, "updated": 0, "unchanged": 0},
        "associations": {"inserted": 20, "updated": 0, "unchanged": 0},
    }
    assert first["postcheck"]["database_before"] == {
        "knowledge": 14,
        "fragments": 16,
        "associations": 20,
    }
    assert second["counts"] == {
        "knowledge": {"inserted": 0, "updated": 0, "unchanged": 14},
        "fragments": {"inserted": 0, "updated": 0, "unchanged": 16},
        "associations": {"inserted": 0, "updated": 0, "unchanged": 20},
    }


def test_lesson4_changed_existing_id_is_rejected_without_writes(tmp_path):
    database_url = "sqlite:///" + (tmp_path / "lesson4-conflict.sqlite").as_posix()
    _create_signed_tables(database_url)
    import_signed_points(database_url, bundle_file=LESSON4_BUNDLE)
    payload = json.loads(LESSON4_BUNDLE.read_text(encoding="utf-8"))
    payload["knowledge_points"][0]["content"] += "冲突内容"
    conflict_bundle = _write_bundle(tmp_path, payload)

    with pytest.raises(ValueError, match="Import precheck found conflicts"):
        import_signed_points(database_url, bundle_file=conflict_bundle)

    engine = create_engine(database_url)
    with Session(engine) as session:
        assert session.get(SignedTextbookKnowledge, "L04-001").content == payload["knowledge_points"][0]["summary"]
        assert _counts(session) == (14, 16, 20)
    engine.dispose()


def test_lesson4_missing_association_is_rejected_offline(tmp_path):
    payload = json.loads(LESSON4_BUNDLE.read_text(encoding="utf-8"))
    payload["associations"] = [
        row for row in payload["associations"] if row["knowledge_id"] != "L04-014"
    ]
    broken_bundle = _write_bundle(tmp_path, payload)

    with pytest.raises(ValueError, match="Active source coverage must match"):
        load_repaired_bundle(bundle_file=broken_bundle)


def test_lesson4_pending_relation_is_stored_but_not_retrievable(tmp_path):
    database_url = "sqlite:///" + (tmp_path / "lesson4-pending.sqlite").as_posix()
    _create_signed_tables(database_url)
    payload = json.loads(LESSON4_BUNDLE.read_text(encoding="utf-8"))
    point = next(row for row in payload["knowledge_points"] if row["knowledge_id"] == "L04-014")
    point["retrieval_status"] = "pending_manual_source_review"
    point["source_status"] = "pending_manual_source_review"
    point["pending_source_refs"] = point.pop("source_refs")
    for relation in payload["associations"]:
        if relation["knowledge_id"] == "L04-014":
            relation["relation_status"] = "pending_manual_source_review"
    pending_bundle = _write_bundle(tmp_path, payload)

    bundle = load_repaired_bundle(bundle_file=pending_bundle)
    assert "L04-014" not in bundle["active_by_point"]
    assert "L04-014" in bundle["pending_by_point"]
    import_signed_points(database_url, bundle_file=pending_bundle)
    retrievable_ids = {
        row["knowledge_id"] for row in load_retrievable_signed_points(database_url)
    }
    assert "L04-014" not in retrievable_ids


def test_lesson4_increment_preserves_338_rows_and_l16_pending_state(tmp_path):
    database_url = "sqlite:///" + (tmp_path / "lesson4-over-338.sqlite").as_posix()
    _create_signed_tables(database_url)
    import_signed_points(database_url)
    engine = create_engine(database_url)
    with Session(engine) as session:
        before = _snapshot_338(session)
    engine.dispose()

    result = import_signed_points(database_url, bundle_file=LESSON4_BUNDLE)
    assert result["knowledge"]["inserted"] == 14
    assert result["fragments"]["inserted"] == 16
    assert result["associations"]["inserted"] == 20

    engine = create_engine(database_url)
    with Session(engine) as session:
        assert _snapshot_338(session) == before
        assert _counts(session) == (352, 356, 392)
        statuses = set(
            session.scalars(
                select(SignedTextbookKnowledgeSource.relation_status).where(
                    SignedTextbookKnowledgeSource.knowledge_id == "L16-Q015"
                )
            ).all()
        )
        assert statuses == {"pending_manual_source_review"}
    engine.dispose()


def test_lesson4_transaction_rolls_back_on_insert_failure(tmp_path):
    database_url = "sqlite:///" + (tmp_path / "lesson4-rollback.sqlite").as_posix()
    _create_signed_tables(database_url)

    def fail_relation_insert(*_args, **_kwargs):
        raise RuntimeError("forced relation failure")

    event.listen(SignedTextbookKnowledgeSource, "before_insert", fail_relation_insert)
    try:
        with pytest.raises(RuntimeError, match="forced relation failure"):
            import_signed_points(database_url, bundle_file=LESSON4_BUNDLE)
    finally:
        event.remove(SignedTextbookKnowledgeSource, "before_insert", fail_relation_insert)

    engine = create_engine(database_url)
    with Session(engine) as session:
        assert _counts(session) == (0, 0, 0)
    engine.dispose()


def _counts(session):
    return (
        session.scalar(select(func.count()).select_from(SignedTextbookKnowledge)),
        session.scalar(select(func.count()).select_from(SignedTextbookSourceFragment)),
        session.scalar(select(func.count()).select_from(SignedTextbookKnowledgeSource)),
    )


def _snapshot_338(session):
    knowledge = session.scalars(
        select(SignedTextbookKnowledge).where(SignedTextbookKnowledge.lesson_number >= 5)
    ).all()
    fragments = session.scalars(
        select(SignedTextbookSourceFragment).where(SignedTextbookSourceFragment.lesson_number >= 5)
    ).all()
    relations = session.scalars(
        select(SignedTextbookKnowledgeSource).where(
            SignedTextbookKnowledgeSource.knowledge_id.in_([row.knowledge_id for row in knowledge])
        )
    ).all()
    return {
        "knowledge": sorted(
            (row.knowledge_id, row.content, row.provenance_json, row.source_status) for row in knowledge
        ),
        "fragments": sorted(
            (row.fragment_id, row.source_id, row.excerpt, row.metadata_json) for row in fragments
        ),
        "relations": sorted(
            (row.knowledge_id, row.fragment_id, row.source_id, row.relation_status) for row in relations
        ),
    }

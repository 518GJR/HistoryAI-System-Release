import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.models import (
    SignedTextbookKnowledge,
    SignedTextbookKnowledgeSource,
    SignedTextbookSourceFragment,
)
from app.services.signed_textbook_service import (
    load_active_fragment_associations,
    load_retrievable_signed_corpus,
    load_retrievable_signed_points,
)
from app.services.vector_service import build_source_documents


def _database_url(tmp_path, name):
    return "sqlite:///" + (tmp_path / name).as_posix()


def _insert(database_url, knowledge_id, trace):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(
            SignedTextbookKnowledge(
                knowledge_id=knowledge_id,
                lesson_number=5,
                title="测试知识点",
                content="测试知识点具有完整事实内容。",
                content_status="signed_fact_summary",
                signoff_status="approved",
                source_status="fragment_linked_repaired",
                provenance_json=json.dumps(trace, ensure_ascii=False),
            )
        )
    engine.dispose()


def _add_fragment_relation(
    database_url,
    knowledge_id,
    fragment_id,
    relation_status="active",
    verification_status="pdf_text_layer_lesson_page_content_match",
    pages=(28, 33),
    parent_source_id=None,
):
    engine = create_engine(database_url)
    with Session(engine) as session, session.begin():
        if session.get(SignedTextbookSourceFragment, fragment_id) is None:
            session.add(
                SignedTextbookSourceFragment(
                    fragment_id=fragment_id,
                    source_id=fragment_id,
                    lesson_number=5,
                    book_id="H7A",
                    lesson_id="H7A-L05",
                    excerpt=f"{fragment_id}教材原文。",
                    textbook_pages_json=json.dumps([pages[0]]),
                    pdf_pages_json=json.dumps([pages[1]]),
                    source_type="教材PDF片段",
                    verification_status=verification_status,
                    support_score=1.0,
                    historical_source_anomaly=False,
                    metadata_json=json.dumps(
                        {"section_id": "H7A-L05-S01", "locator": fragment_id},
                        ensure_ascii=False,
                    ),
                )
            )
        session.add(
            SignedTextbookKnowledgeSource(
                knowledge_id=knowledge_id,
                fragment_id=fragment_id,
                source_id=parent_source_id or fragment_id,
                relation_status=relation_status,
            )
        )
    engine.dispose()


def _insert_relation_fixture(database_url):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        for knowledge_id in ("K-1", "K-2", "L13-Q011", "L16-Q015"):
            session.add(
                SignedTextbookKnowledge(
                    knowledge_id=knowledge_id,
                    lesson_number=13 if knowledge_id == "L13-Q011" else 16,
                    title=f"标题{knowledge_id}",
                    content=f"内容{knowledge_id}",
                    content_status="signed_fact_summary",
                    signoff_status="approved",
                    source_status="fragment_linked_repaired",
                    provenance_json="{}",
                )
            )
        for fragment_id, lesson_number in (
            ("F-SHARED", 16),
            ("F-L13", 13),
            ("F-PENDING", 16),
        ):
            session.add(
                SignedTextbookSourceFragment(
                    fragment_id=fragment_id,
                    source_id=f"PRIMARY-{fragment_id}",
                    lesson_number=lesson_number,
                    book_id="H7A",
                    lesson_id=f"H7A-L{lesson_number:02d}",
                    excerpt=f"{fragment_id}完整教材原文。",
                    textbook_pages_json=json.dumps([39, 40]),
                    pdf_pages_json=json.dumps([44, 45]),
                    source_type="教材PDF片段",
                    verification_status="verified",
                    support_score=1.0,
                    historical_source_anomaly=False,
                    metadata_json="{}",
                )
            )
        relations = (
            ("K-1", "F-SHARED", "S-1", "active"),
            ("K-2", "F-SHARED", "S-1", "active"),
            ("K-2", "F-SHARED", "S-2", "active"),
            ("L16-Q015", "F-SHARED", "S-PENDING-SHARED", "pending_manual_source_review"),
            ("L13-Q011", "F-L13", "S-L13", "active"),
            ("L16-Q015", "F-PENDING", "S-L16", "pending_manual_source_review"),
        )
        for knowledge_id, fragment_id, source_id, status in relations:
            session.add(
                SignedTextbookKnowledgeSource(
                    knowledge_id=knowledge_id,
                    fragment_id=fragment_id,
                    source_id=source_id,
                    relation_status=status,
                )
            )
    engine.dispose()


def test_mixed_active_pending_relations_keep_only_active_fragment(tmp_path):
    database_url = _database_url(tmp_path, "mixed.sqlite")
    _insert(database_url, "TEST-001", {"qualifiers": ["约"]})
    _add_fragment_relation(database_url, "TEST-001", "F-ACTIVE", pages=(28, 33))
    _add_fragment_relation(
        database_url,
        "TEST-001",
        "F-PENDING",
        relation_status="pending_manual_source_review",
        verification_status="pending_manual_source_review",
        pages=(29, 34),
    )

    points, audit = load_retrievable_signed_corpus(database_url)

    assert len(points) == 1
    assert [ref["source_id"] for ref in points[0]["source_refs"]] == ["F-ACTIVE"]
    assert audit == {
        "signed_knowledge_total": 1,
        "retrievable_knowledge_count": 1,
        "active_fragment_count": 1,
        "pending_association_count": 1,
        "excluded_knowledge_count": 0,
    }


def test_pending_only_knowledge_is_excluded_from_index_corpus(tmp_path):
    database_url = _database_url(tmp_path, "pending-only.sqlite")
    _insert(database_url, "TEST-002", {"source_fragment_ids": ["F-PENDING"]})
    _add_fragment_relation(
        database_url,
        "TEST-002",
        "F-PENDING",
        relation_status="pending_manual_source_review",
        verification_status="pending_manual_source_review",
    )

    points, audit = load_retrievable_signed_corpus(database_url)

    assert points == []
    assert audit["pending_association_count"] == 1
    assert audit["excluded_knowledge_count"] == 1


def test_provenance_without_active_database_relation_is_not_evidence(tmp_path):
    database_url = _database_url(tmp_path, "provenance-only.sqlite")
    _insert(
        database_url,
        "TEST-003",
        {
            "source_fragment_ids": ["F-ONLY-IN-JSON"],
            "source_refs": [
                {
                    "fragment_id": "F-ONLY-IN-JSON",
                    "excerpt": "provenance中的文字不能替代数据库active关系。",
                    "textbook_pages": "28",
                    "pdf_pages": "33",
                }
            ],
        },
    )

    assert load_retrievable_signed_points(database_url) == []


def test_multiple_parent_relations_do_not_duplicate_fragment_document(tmp_path):
    database_url = _database_url(tmp_path, "deduplicate.sqlite")
    _insert(database_url, "TEST-004", {})
    _add_fragment_relation(
        database_url, "TEST-004", "F-SHARED", parent_source_id="S-1"
    )
    _add_fragment_relation(
        database_url, "TEST-004", "F-SHARED", parent_source_id="S-2"
    )

    points = load_retrievable_signed_points(database_url)
    documents = build_source_documents(points)

    assert len(points[0]["source_refs"]) == 1
    assert len(documents) == 1
    assert documents[0]["source_id"] == "F-SHARED"


def test_active_relation_still_requires_retrievable_verification_status(tmp_path):
    database_url = _database_url(tmp_path, "invalid-verification.sqlite")
    _insert(database_url, "TEST-005", {})
    _add_fragment_relation(
        database_url,
        "TEST-005",
        "F-UNVERIFIED",
        relation_status="active",
        verification_status="pending_manual_source_review",
    )

    points, audit = load_retrievable_signed_corpus(database_url)

    assert points == []
    assert audit["active_fragment_count"] == 0
    assert audit["excluded_knowledge_count"] == 1


def test_active_fragment_associations_batch_preserves_all_active_triples(tmp_path):
    database_url = _database_url(tmp_path, "relations.sqlite")
    _insert_relation_fixture(database_url)

    result = load_active_fragment_associations(
        ["F-SHARED", "F-L13", "F-PENDING"], database_url
    )

    assert set(result) == {"F-SHARED", "F-L13", "F-PENDING"}
    assert [
        (row["knowledge_id"], row["fragment_id"], row["source_id"])
        for row in result["F-SHARED"]["active_associations"]
    ] == [
        ("K-1", "F-SHARED", "S-1"),
        ("K-2", "F-SHARED", "S-1"),
        ("K-2", "F-SHARED", "S-2"),
    ]
    assert result["F-SHARED"]["textbook_pages"] == "39;40"
    assert result["F-SHARED"]["pdf_pages"] == "44;45"
    assert result["F-SHARED"]["verification_status"] == "verified"
    assert result["F-L13"]["active_associations"][0]["knowledge_id"] == "L13-Q011"
    assert result["F-PENDING"]["active_associations"] == []
    assert all(
        row["knowledge_id"] != "L16-Q015"
        for entry in result.values()
        for row in entry["active_associations"]
    )

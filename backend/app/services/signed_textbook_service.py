"""Opt-in retrieval view over signed textbook points."""

import json
import logging
import os

from sqlalchemy import and_, create_engine, select
from sqlalchemy.orm import Session

from app.models import (
    SignedTextbookKnowledge,
    SignedTextbookKnowledgeSource,
    SignedTextbookSourceFragment,
)


LOGGER = logging.getLogger(__name__)

ACTIVE_RELATION_STATUS = "active"
PENDING_RELATION_STATUS = "pending_manual_source_review"
RETRIEVABLE_CONTENT_STATUS = "signed_fact_summary"
RETRIEVABLE_SOURCE_STATUSES = {
    "fragment_linked_repaired",
    "fragment_linked_pdf_verified",
}
RETRIEVABLE_VERIFICATION_STATUSES = {
    "candidate_lesson_page_excerpt_checked_against_pdf",
    "inherited_existing_retrieval_evidence",
    "pdf_text_layer_lesson_page_content_match",
    "verified_against_original_pdf_pages_27_32",
    "verified",  # Legacy explicit verified status retained for compatibility.
}


def _page_string(value: str) -> str:
    try:
        pages = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(pages, list):
        return ""
    return ";".join(str(page).strip() for page in pages if str(page).strip())


def load_active_fragment_associations(fragment_ids, database_url=None):
    """Batch-load signed fragment metadata and every active parent relation.

    Keys are returned for signed fragments even when no active relation exists,
    allowing vector retrieval to reject pending-only signed evidence. The query
    is a single read-only SELECT across all three signed textbook tables.
    """
    normalized_ids = list(
        dict.fromkeys(
            str(fragment_id).strip()
            for fragment_id in fragment_ids
            if str(fragment_id).strip()
        )
    )
    url = database_url or os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip()
    if not url or not normalized_ids:
        return {}

    relation = SignedTextbookKnowledgeSource
    knowledge = SignedTextbookKnowledge
    fragment = SignedTextbookSourceFragment
    statement = (
        select(
            fragment.fragment_id,
            fragment.lesson_id,
            fragment.lesson_number,
            fragment.excerpt,
            fragment.source_type,
            fragment.textbook_pages_json,
            fragment.pdf_pages_json,
            fragment.verification_status,
            fragment.historical_source_anomaly,
            relation.knowledge_id,
            relation.source_id,
            knowledge.title.label("knowledge_title"),
            knowledge.content_status,
            knowledge.source_status,
        )
        .select_from(fragment)
        .outerjoin(
            relation,
            and_(
                relation.fragment_id == fragment.fragment_id,
                relation.relation_status == ACTIVE_RELATION_STATUS,
            ),
        )
        .outerjoin(knowledge, knowledge.knowledge_id == relation.knowledge_id)
        .where(fragment.fragment_id.in_(normalized_ids))
        .order_by(fragment.fragment_id, relation.knowledge_id, relation.source_id)
    )

    engine = create_engine(url)
    try:
        with Session(engine) as session:
            rows = session.execute(statement).mappings().all()
    finally:
        engine.dispose()

    result = {}
    for row in rows:
        fragment_id = row["fragment_id"]
        entry = result.setdefault(
            fragment_id,
            {
                "fragment_id": fragment_id,
                "lesson_id": row["lesson_id"],
                "lesson_number": row["lesson_number"],
                "excerpt": row["excerpt"],
                "material_type": row["source_type"],
                "textbook_pages": _page_string(row["textbook_pages_json"]),
                "pdf_pages": _page_string(row["pdf_pages_json"]),
                "verification_status": row["verification_status"],
                "active_associations": [],
            },
        )
        if (
            row["knowledge_id"]
            and row["source_id"]
            and row["content_status"] == RETRIEVABLE_CONTENT_STATUS
            and row["source_status"] in RETRIEVABLE_SOURCE_STATUSES
            and row["verification_status"] in RETRIEVABLE_VERIFICATION_STATUSES
            and not row["historical_source_anomaly"]
        ):
            entry["active_associations"].append(
                {
                    "knowledge_id": row["knowledge_id"],
                    "fragment_id": fragment_id,
                    "source_id": row["source_id"],
                    "knowledge_title": row["knowledge_title"] or "",
                }
            )
    return result


def _json_object(value) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_retrievable_signed_corpus(database_url=None):
    """Load signed points from one batched read, filtering by normalized relations."""
    url = database_url or os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip()
    if not url:
        return [], {
            "signed_knowledge_total": 0,
            "retrievable_knowledge_count": 0,
            "active_fragment_count": 0,
            "pending_association_count": 0,
            "excluded_knowledge_count": 0,
        }
    knowledge = SignedTextbookKnowledge
    relation = SignedTextbookKnowledgeSource
    fragment = SignedTextbookSourceFragment
    statement = (
        select(
            knowledge.knowledge_id,
            knowledge.lesson_number,
            knowledge.title,
            knowledge.content,
            knowledge.content_status,
            knowledge.source_status,
            knowledge.provenance_json,
            relation.fragment_id,
            relation.source_id.label("parent_source_id"),
            relation.relation_status,
            fragment.lesson_id,
            fragment.excerpt,
            fragment.textbook_pages_json,
            fragment.pdf_pages_json,
            fragment.source_type,
            fragment.verification_status,
            fragment.historical_source_anomaly,
            fragment.metadata_json,
        )
        .select_from(knowledge)
        .outerjoin(relation, relation.knowledge_id == knowledge.knowledge_id)
        .outerjoin(fragment, fragment.fragment_id == relation.fragment_id)
        .order_by(knowledge.knowledge_id, relation.fragment_id, relation.source_id)
    )
    engine = create_engine(url)
    try:
        with Session(engine) as session:
            rows = session.execute(statement).mappings().all()
    finally:
        engine.dispose()

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["knowledge_id"], []).append(dict(row))
    result = []
    active_fragment_ids = set()
    pending_count = sum(
        row["relation_status"] == PENDING_RELATION_STATUS
        for row in rows
    )
    for knowledge_id, knowledge_rows in grouped.items():
        first = knowledge_rows[0]
        trace = _json_object(first["provenance_json"])
        if (
            first["content_status"] != RETRIEVABLE_CONTENT_STATUS
            or first["source_status"] not in RETRIEVABLE_SOURCE_STATUSES
        ):
            continue
        refs_by_fragment = {}
        for row in knowledge_rows:
            if (
                row["relation_status"] != ACTIVE_RELATION_STATUS
                or not row["fragment_id"]
                or not str(row["excerpt"] or "").strip()
                or row["verification_status"] not in RETRIEVABLE_VERIFICATION_STATUSES
                or row["historical_source_anomaly"]
            ):
                continue
            textbook_pages = _page_string(row["textbook_pages_json"])
            pdf_pages = _page_string(row["pdf_pages_json"])
            if not textbook_pages or not pdf_pages:
                continue
            metadata = _json_object(row["metadata_json"])
            fragment_id = str(row["fragment_id"])
            refs_by_fragment.setdefault(fragment_id, {
                "source_id": fragment_id,
                "fragment_id": fragment_id,
                "lesson_id": str(row["lesson_id"] or f"H7A-L{first['lesson_number']:02d}"),
                "section_id": str(metadata.get("section_id") or "signed-staging"),
                "material_type": str(metadata.get("material_type") or row["source_type"] or "教材片段"),
                "textbook_pages": textbook_pages,
                "pdf_pages": pdf_pages,
                "excerpt": str(row["excerpt"]).strip(),
                "verification_status": str(row["verification_status"]),
                "locator": str(metadata.get("locator") or ""),
            })
        refs = list(refs_by_fragment.values())
        if not refs:
            continue
        active_fragment_ids.update(refs_by_fragment)
        lesson_number = int(first["lesson_number"])
        result.append({
            "knowledge_id": knowledge_id,
            "book_id": str(trace.get("book_id") or "H7A"),
            "book_title": str(trace.get("book_title") or "义务教育教科书·中国历史 七年级上册"),
            "unit_id": str(trace.get("unit_id") or f"H7A-U{2 if lesson_number <= 8 else 3 if lesson_number <= 15 else 4:02d}"),
            "unit_title": str(trace.get("unit_title") or "第4—20课签核知识点"),
            "lesson_id": str(trace.get("lesson_id") or f"H7A-L{lesson_number:02d}"),
            "lesson_number": lesson_number,
            "lesson_title": str(trace.get("lesson_title") or f"第{lesson_number}课"),
            "section_title": str(trace.get("section_title") or "签核知识点"),
            "title": first["title"],
            "content": first["content"],
            "summary": first["content"],
            "keywords": [first["title"]],
            "people": [],
            "qualifiers": trace.get("qualifiers") or [],
            "certainty": first["source_status"],
            "source_refs": refs,
        })
    audit = {
        "signed_knowledge_total": len(grouped),
        "retrievable_knowledge_count": len(result),
        "active_fragment_count": len(active_fragment_ids),
        "pending_association_count": pending_count,
        "excluded_knowledge_count": len(grouped) - len(result),
    }
    return result, audit


def load_retrievable_signed_points(database_url=None):
    points, _ = load_retrievable_signed_corpus(database_url)
    return points

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import QuestionHistory
from app.schemas import (
    QuestionHistoryListResponse,
    QuestionHistoryResponse,
    QuestionRequest,
    QuestionResponse,
    KnowledgeSource,
)
from app.services import rag_service
from app.services.ai_service import (
    AIProviderError,
    generate_answer_result,
    generate_direct_answer,
    template_answer,
)
from app.services.knowledge_service import (
    assemble_textbook_answer,
    build_knowledge_outline,
)
from app.services.rag_service import (
    KnowledgeDataError,
    get_knowledge_mode,
    knowledge_item_to_source,
    load_textbook_knowledge_from_json,
    retrieve_history_knowledge,
    get_retrieval_mode,
)
from app.services.support_assessment import (
    analyze_question_requirements,
    assess_support,
    evidence_covers_assessment,
    supporting_answer_items,
)


router = APIRouter()
logger = logging.getLogger(__name__)
MAX_SUPPLEMENTAL_QUERIES = 4

SUPPLEMENT_QUERY_HINTS = {
    "time": "年代 距今",
    "place": "地点 位于 地区 流域",
    "tools": "工具 石器 骨器 技术",
    "agriculture": "农业 农作物 种植 水稻 粟 黍",
    "housing": "房屋 干栏 半地穴",
    "livelihood": "生产生活 饲养 渔猎 采集 纺织 制衣 手工业",
    "urban": "城市建设 大型城址 高等级建筑 宫殿区 内城 外城 宫城",
    "burial": "墓葬 随葬品 贵族墓 普通墓",
    "economy_culture": "经济 文化联系 交流 融合 多元一体",
    "process": "过程 经历 形成",
    "result_impact": "结果 影响 说明 表明",
    "certainty": "可能 相传 约",
}


def _retrieve_for_request(question: str, requested_mode: str | None):
    """Use existing retrieval implementations without mutating global config."""
    if requested_mode is None:
        return retrieve_history_knowledge(question), get_retrieval_mode()
    if requested_mode == "keyword":
        return rag_service._retrieve_keyword_history_knowledge(question), "keyword"
    if get_knowledge_mode() != "textbook":
        raise KnowledgeDataError("Vector retrieval requires KNOWLEDGE_MODE=textbook")

    from app.services.vector_service import VectorIndexError, retrieve_textbook_vectors

    try:
        return retrieve_textbook_vectors(question, top_k=rag_service.TOP_K), "vector"
    except VectorIndexError as exc:
        raise KnowledgeDataError(str(exc)) from exc


def _candidate_identity(item: dict) -> tuple:
    knowledge_ids = tuple(
        sorted(
            str(value)
            for value in (item.get("knowledge_ids") or [item.get("knowledge_id")])
            if value
        )
    )
    source_ids = tuple(
        sorted(
            str(reference.get("source_id") or "")
            for reference in item.get("source_refs") or []
            if isinstance(reference, dict) and reference.get("source_id")
        )
    )
    return knowledge_ids, source_ids


def _missing_comparison_queries(assessment) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for facet in assessment.unsupported_facets:
        parts = facet.split(":")
        if len(parts) != 4 or parts[0] != "subject":
            continue
        subject, value = parts[1], parts[3]
        hint = SUPPLEMENT_QUERY_HINTS.get(value, value)
        grouped.setdefault(subject, [])
        if hint not in grouped[subject]:
            grouped[subject].append(hint)
    return [
        " ".join([subject, *hints]).strip()
        for subject, hints in grouped.items()
    ][:MAX_SUPPLEMENTAL_QUERIES]


def _supplement_comparison_evidence(
    question: str,
    retrieval_mode: str,
    candidates: list[dict],
    assessment,
    textbook_points: list[dict],
):
    """Run bounded same-mode retrieval only for missing comparison cells."""
    requirements = analyze_question_requirements(question, textbook_points)
    if not requirements["is_comparison"] or assessment.status == "sufficient":
        return candidates, assessment

    combined = list(candidates)
    seen = {_candidate_identity(item) for item in combined}
    used_supplement = False
    previous_covered = len(assessment.covered_facets)
    for supplemental_query in _missing_comparison_queries(assessment):
        supplemental, executed_mode = _retrieve_for_request(
            supplemental_query, retrieval_mode
        )
        if executed_mode != retrieval_mode:
            raise KnowledgeDataError(
                "Supplemental retrieval changed the requested mode"
            )
        added = []
        for item in supplemental:
            identity = _candidate_identity(item)
            if identity in seen:
                continue
            seen.add(identity)
            combined.append(item)
            added.append(item)
        if not added:
            continue
        used_supplement = True
        updated = assess_support(
            question, retrieval_mode, combined, textbook_points
        )
        assessment = updated
        if assessment.status == "sufficient":
            break
        if len(assessment.covered_facets) <= previous_covered:
            continue
        previous_covered = len(assessment.covered_facets)

    if used_supplement and "supplemental_retrieval_used" not in assessment.reason_codes:
        assessment.reason_codes.append("supplemental_retrieval_used")
    return combined, assessment


def _serialize_created_at(value: datetime, dialect_name: str) -> datetime:
    """Mark known SQLite UTC values; preserve unknown legacy DB semantics."""
    if value.tzinfo is None and dialect_name == "sqlite":
        return value.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc)


def _parse_sources(sources_text: str | None) -> list[dict]:
    if not sources_text:
        return []

    try:
        parsed_sources = json.loads(sources_text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed_sources, list):
        return []

    sources = []
    for source in parsed_sources:
        if not isinstance(source, dict):
            continue

        source_snapshot = dict(source)
        try:
            source_snapshot["score"] = float(source.get("score", 0) or 0)
        except (TypeError, ValueError):
            source_snapshot["score"] = 0

        try:
            sources.append(KnowledgeSource.model_validate(source_snapshot).model_dump())
        except ValidationError:
            continue

    return sources


@router.post("/question", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Run the requested answer mode and save history without blocking delivery."""
    if request.mode == "none":
        try:
            answer = generate_direct_answer(request.question)
        except AIProviderError as exc:
            payload = QuestionResponse(
                requested_mode="none",
                executed_mode="none",
                retrieval_mode="none",
                answer_mode="error",
                answer="",
                sources=[],
                used_evidence=[],
                support_assessment=None,
                knowledge_outline=None,
                history_saved=False,
                error=str(exc),
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )

        question_history = QuestionHistory(
            user_id=None,
            question=request.question,
            answer=answer,
            sources="[]",
        )
        history_saved = True
        try:
            db.add(question_history)
            db.commit()
        except Exception as exc:
            history_saved = False
            try:
                db.rollback()
            except Exception:
                logger.exception("Question history rollback also failed")
            logger.warning("History save failed; returning generated answer: %s", exc)

        return QuestionResponse(
            requested_mode="none",
            executed_mode="none",
            retrieval_mode="none",
            answer_mode="model_generation",
            answer=answer,
            sources=[],
            used_evidence=[],
            support_assessment=None,
            knowledge_outline=None,
            history_saved=history_saved,
            error=None,
        )

    try:
        knowledge_items, retrieval_mode = _retrieve_for_request(
            request.question, request.mode
        )
    except KnowledgeDataError as exc:
        raise HTTPException(status_code=503, detail=f"检索不可用：{exc}") from exc
    support_assessment = None
    answer_items = knowledge_items
    textbook_points = None
    if get_knowledge_mode() == "textbook":
        # Keep support assessment aligned with the opt-in retrieval corpus.
        textbook_points = rag_service.load_history_knowledge()
        support_assessment = assess_support(
            request.question, retrieval_mode, knowledge_items, textbook_points
        )
        knowledge_items, support_assessment = _supplement_comparison_evidence(
            request.question,
            retrieval_mode,
            knowledge_items,
            support_assessment,
            textbook_points,
        )
        answer_items = (
            supporting_answer_items(knowledge_items, support_assessment)
            if support_assessment.status in {"sufficient", "partial"}
            else []
        )

    if support_assessment is not None and textbook_points is not None:
        answer, answer_mode, used_evidence = assemble_textbook_answer(
            request.question,
            answer_items,
            support_assessment,
            textbook_points,
        )
        if not answer_items:
            answer = template_answer(request.question, [])
            answer_mode = "insufficient_support"
            used_evidence = []
        elif support_assessment.status == "sufficient" and not evidence_covers_assessment(
            support_assessment, used_evidence
        ):
            answer = template_answer(request.question, [])
            answer_mode = "insufficient_support"
            used_evidence = []
    else:
        answer, answer_mode, used_evidence = generate_answer_result(
            request.question, answer_items
        )

    sources = [knowledge_item_to_source(item) for item in knowledge_items]
    supporting_source_ids = (
        set(support_assessment.supporting_source_ids) if support_assessment else set()
    )
    for source in sources:
        source["supporting"] = any(
            reference.get("source_id") in supporting_source_ids
            for reference in source.get("source_refs", [])
        )
    sources_by_id = {
        knowledge_id: source
        for source in sources
        for knowledge_id in source.get("knowledge_ids") or [source["knowledge_id"]]
    }
    for evidence in used_evidence:
        for knowledge_id in evidence.get("knowledge_ids", []):
            source = sources_by_id.get(knowledge_id)
            if source is not None:
                source.setdefault("used_evidence", []).append(evidence)
    outline = build_knowledge_outline(request.question, answer_items)
    if outline is not None:
        outline["sources"] = [
            source
            for source in sources
            if support_assessment is None or source.get("supporting")
        ]

    question_history = QuestionHistory(
        user_id=None,
        question=request.question,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False),
    )
    history_saved = True
    try:
        db.add(question_history)
        db.commit()
    except Exception as exc:
        history_saved = False
        try:
            db.rollback()
        except Exception:
            logger.exception("Question history rollback also failed")
        logger.warning("History save failed; returning generated answer: %s", exc)

    return QuestionResponse(
        requested_mode=request.mode,
        executed_mode=retrieval_mode,
        answer=answer,
        sources=sources,
        history_saved=history_saved,
        answer_mode=answer_mode,
        retrieval_mode=retrieval_mode,
        used_evidence=used_evidence,
        support_assessment=(support_assessment.public() if support_assessment else None),
        knowledge_outline=outline,
        error=None,
    )


@router.get("/question/history", response_model=QuestionHistoryListResponse)
def get_question_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return recent question history records."""
    history_rows = (
        db.query(QuestionHistory)
        .order_by(QuestionHistory.created_at.desc(), QuestionHistory.id.desc())
        .limit(limit)
        .all()
    )

    dialect_name = db.get_bind().dialect.name
    histories = [
        QuestionHistoryResponse(
            id=row.id,
            user_id=row.user_id,
            question=row.question,
            answer=row.answer,
            sources=_parse_sources(row.sources),
            created_at=_serialize_created_at(row.created_at, dialect_name),
        )
        for row in history_rows
    ]

    return QuestionHistoryListResponse(histories=histories)

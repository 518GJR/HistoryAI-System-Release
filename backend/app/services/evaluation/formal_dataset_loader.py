"""Adapt the formal 100-question candidate into the existing rubric models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    CriterionDefinition,
    RubricDataset,
    RubricQuestion,
    SourceReference,
)
from .rubric_loader import load_rubric


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rules_as_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _criterion(raw: dict, question: dict, source_path: Path) -> CriterionDefinition:
    # Frozen Unit 1 criteria already use the native evaluator schema.
    if "classification" in raw:
        return CriterionDefinition.model_validate(raw)
    scoring = question["scoring_rule"]
    return CriterionDefinition(
        criterion_id=raw["criterion_id"],
        criterion_description=raw["criterion_description"],
        criterion_weight=raw["criterion_weight"],
        original_criterion_text=raw["criterion_description"],
        original_weight_text=f'{raw["criterion_weight"]}分',
        classification="C",
        scoring_method=(
            "whole_full_zero"
            if question["question_group"] == "R"
            else "whole_full_half_zero"
        ),
        equivalence_rules_raw=str(scoring.get("equivalence_rules") or ""),
        major_error_rules=_rules_as_list(scoring.get("major_error_rules")),
        source_references=[SourceReference(source_file=str(source_path))],
        review_status="source_verified",
    )


def load_formal_evaluation(path: Path) -> RubricDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "historyai.formal-evaluation-candidate.v1":
        raise ValueError("Not a formal 100-question evaluation dataset")
    questions = []
    for raw in payload.get("questions", []):
        scoring = raw["scoring_rule"]
        criteria = [_criterion(item, raw, path) for item in scoring["criteria"]]
        group = raw["question_group"]
        standard_answer = raw.get("standard_answer") or raw.get("expected_answer") or raw.get("reference_answer")
        questions.append(RubricQuestion(
            question_id=raw["question_id"],
            question_text=raw.get("question_text") or raw["question"],
            standard_answer=standard_answer,
            rubric_version=payload["dataset_version"],
            question_weight=float(raw.get("max_score") or raw.get("total_score")),
            criteria=criteria,
            textbook_references=[json.dumps(raw.get("textbook_references", {}), ensure_ascii=False)],
            lesson=str(raw.get("lesson") or raw.get("lesson_id") or ""),
            knowledge_point_ids=list(raw.get("source_mapping", {}).get("knowledge_ids", [])),
            question_type=str(raw.get("task_type") or ""),
            original_major_error_text=str(scoring.get("major_error_rules") or ""),
            original_equivalence_text=str(scoring.get("equivalence_rules") or ""),
            original_source_record={"dataset_path": str(path), "signoff_status": str(raw.get("signoff_status") or raw.get("signoff", {}).get("status") or "")},
            question_group=group,
            mode_specific_guidance=dict(raw.get("expected_behavior_by_mode") or {}),
            reference_text_verification=dict(raw.get("full_text_reference") or {}),
        ))
    expected = payload.get("counts", {}).get("total", 100)
    if len(questions) != expected or len({item.question_id for item in questions}) != len(questions):
        raise ValueError("Formal evaluation question count or IDs are invalid")
    return RubricDataset(
        rubric_version=payload["dataset_version"],
        source_release=payload["dataset_id"],
        generated_at=datetime.fromisoformat(payload["generated_date"]).replace(tzinfo=timezone.utc),
        source_files=[SourceReference(source_file=str(path), sha256=_sha256(path))],
        questions=questions,
        classification_counts={"Q": sum(q.question_group == "Q" for q in questions), "K": sum(q.question_group == "K" for q in questions), "R": sum(q.question_group == "R" for q in questions)},
        conversion_status="formal_candidate_adapter",
        conversion_notes=["Reuses the existing deterministic scorer and batch runner; signed question text, answers, weights, and rules are not rewritten."],
    )


def load_evaluation_dataset(path: Path) -> RubricDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "historyai.formal-evaluation-candidate.v1":
        return load_formal_evaluation(path)
    return load_rubric(path)


def question_for_mode(question: RubricQuestion, mode: str) -> RubricQuestion:
    guidance = question.mode_specific_guidance.get(mode)
    if not guidance:
        return question
    adapted = question.model_copy(deep=True)
    adapted.standard_answer = f"{question.standard_answer}\n本模式签核行为：{guidance}"
    for criterion in adapted.criteria:
        if criterion.criterion_id.endswith("-BEHAVIOR"):
            criterion.criterion_description = f"行为符合度：{guidance}"
            criterion.original_criterion_text = criterion.criterion_description
    return adapted

import json
from collections import Counter
from pathlib import Path

import pytest

from app.services.evaluation.rubric_loader import (
    RubricValidationError,
    build_rubric,
    load_rubric,
    validate_rubric,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUBRIC_PATH = PROJECT_ROOT / "docs/evaluation/workbench/unit1_v1.0_rubric.json"


def test_frozen_conversion_has_exact_counts_and_weights():
    rubric = build_rubric(PROJECT_ROOT)
    criteria = [criterion for question in rubric.questions for criterion in question.criteria]

    assert len(rubric.questions) == 20
    assert len(criteria) == 109
    assert Counter(item.classification for item in criteria) == {
        "A": 26,
        "B": 8,
        "C": 37,
        "D": 38,
    }
    assert all(sum(c.criterion_weight for c in q.criteria) == 10 for q in rubric.questions)
    assert validate_rubric(rubric, expected_question_count=20)


def test_explicit_pending_b_and_d_rules_remain_distinct():
    rubric = load_rubric(RUBRIC_PATH)
    criteria = {c.criterion_id: c for q in rubric.questions for c in q.criteria}

    assert [fact.weight for fact in criteria["Q002-C02"].atomic_facts] == [1, 1, 1, 1]
    assert all(
        fact.weight is None and fact.proposed_weight
        for fact in criteria["Q005-C05"].atomic_facts
    )
    b_items = [c for c in criteria.values() if c.classification == "B"]
    assert len(b_items) == 8
    assert all(len(c.scoring_options) == 2 for c in b_items)
    assert all(not option.selected for c in b_items for option in c.scoring_options)
    d_items = [c for c in criteria.values() if c.classification == "D"]
    assert len(d_items) == 38
    assert all(not c.atomic_facts for c in d_items)
    assert all(c.review_status.startswith("blocked_pending_adjudication") for c in d_items)


def test_original_text_and_coordinates_are_preserved():
    rubric = load_rubric(RUBRIC_PATH)
    q1 = rubric.questions[0]
    criterion = q1.criteria[0]

    assert q1.original_source_record["关键事实要点"].startswith("①元谋人")
    assert criterion.original_criterion_text == "①元谋人：云南元谋、约170万年。"
    assert criterion.original_weight_text == "①3分（地点1.5＋年代1.5）"
    assert criterion.source_references[0].cells == ["H5", "I5", "J5", "K5", "L5"]
    assert criterion.equivalence_rules_raw == q1.original_equivalence_text
    assert criterion.major_error_rules == [q1.original_major_error_text]


def test_weight_change_and_duplicate_criterion_are_rejected():
    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions[0].criteria[0].criterion_weight = 2.9
    with pytest.raises(RubricValidationError, match="weight sum"):
        validate_rubric(rubric)

    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions[0].criteria[1].criterion_id = rubric.questions[0].criteria[0].criterion_id
    with pytest.raises(RubricValidationError, match="duplicate criterion"):
        validate_rubric(rubric)


def test_configurable_machine_readable_format_can_load_120_questions(tmp_path):
    base = load_rubric(RUBRIC_PATH)
    template = base.questions[0]
    questions = []
    for index in range(1, 121):
        question = template.model_copy(deep=True)
        question.question_id = f"F{index:03d}"
        for criterion_index, criterion in enumerate(question.criteria, start=1):
            old_prefix = criterion.criterion_id
            criterion.criterion_id = f"F{index:03d}-C{criterion_index:02d}"
            for fact_index, fact in enumerate(criterion.atomic_facts, start=1):
                fact.fact_id = f"F{index:03d}-C{criterion_index:02d}-F{fact_index:02d}"
        questions.append(question)
    future = base.model_copy(update={
        "questions": questions,
        "source_release": "future-configurable-input",
        "classification_counts": {"A": 360, "C": 120},
    })
    path = tmp_path / "future_120.json"
    path.write_text(json.dumps(future.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")

    loaded = load_rubric(path)

    assert len(loaded.questions) == 120
    assert validate_rubric(loaded, expected_question_count=120)

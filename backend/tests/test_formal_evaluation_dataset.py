from pathlib import Path

from app.services.evaluation.formal_dataset_loader import (
    load_evaluation_dataset,
    question_for_mode,
)
from app.services.evaluation.full_text_reference import (
    compare_full_text,
    inspect_reference_text,
)
from app.services.evaluation.models import CriterionJudgment, FactStatus
from app.services.evaluation.scoring_engine import score_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "docs/evaluation/releases/formal_100_v1.0/formal_evaluation_100.json"
LISAO = PROJECT_ROOT / "docs/evaluation/references/R03_lisao_1998_ma_maoyuan.txt"


def test_formal_100_loads_through_existing_rubric_models():
    rubric = load_evaluation_dataset(DATASET)
    assert len(rubric.questions) == 100
    assert rubric.classification_counts == {"Q": 20, "K": 60, "R": 20}
    assert {item.question_weight for item in rubric.questions if item.question_group == "R"} == {3.0}
    assert {item.question_weight for item in rubric.questions if item.question_group != "R"} == {10.0}


def test_r03_mode_guidance_keeps_none_and_rag_behaviors_distinct():
    question = next(item for item in load_evaluation_dataset(DATASET).questions if item.question_id == "R03")
    none = question_for_mode(question, "none")
    keyword = question_for_mode(question, "keyword")
    assert "允许使用模型自身知识提供全文" in none.standard_answer
    assert "教材没有收录" in keyword.standard_answer
    assert "全文" in keyword.standard_answer
    assert none.criteria[0].criterion_description != keyword.criteria[0].criterion_description


def test_existing_scorer_accepts_signed_three_point_rubric():
    question = next(item for item in load_evaluation_dataset(DATASET).questions if item.question_id == "R03")
    judgments = [CriterionJudgment(criterion_id=item.criterion_id, status=FactStatus.FULL, evidence_quotes=["回答片段"], rationale="满足") for item in question.criteria]
    result = score_answer(run_id="test", question=question, requested_mode="none", executed_mode="none", answer_text="回答片段", judgments=judgments, directly_addresses_question=True, has_verifiable_fact=True)
    assert result.earned_score == 3
    assert result.max_score == 3


def test_r_dimensions_do_not_invent_unsigned_half_points():
    question = next(item for item in load_evaluation_dataset(DATASET).questions if item.question_id == "R03")
    judgments = [CriterionJudgment(criterion_id=item.criterion_id, status=FactStatus.FULL, evidence_quotes=["回答片段"], rationale="满足") for item in question.criteria]
    judgments[0] = CriterionJudgment(criterion_id=question.criteria[0].criterion_id, status=FactStatus.PARTIAL, evidence_quotes=["回答片段"], rationale="部分")
    result = score_answer(run_id="test", question=question, requested_mode="none", executed_mode="none", answer_text="回答片段", judgments=judgments, directly_addresses_question=True, has_verifiable_fact=True)
    assert result.earned_score is None
    assert result.needs_review is True


def test_r03_reference_is_utf8_complete_and_variants_require_review():
    audit = inspect_reference_text(LISAO)
    text = LISAO.read_text(encoding="utf-8")
    assert audit["first_nonempty_line"] == "#### 离骚"
    assert audit["last_nonempty_line"].endswith("吾将从彭咸之所居！")
    assert compare_full_text(text.replace("\n", " "), text)["matches"] is True
    assert compare_full_text(text.replace("彭咸", "彭某"), text)["requires_human_review"] is True

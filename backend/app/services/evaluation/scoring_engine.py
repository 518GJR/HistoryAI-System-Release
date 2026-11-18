"""Deterministic scoring; semantic components may judge facts but never set weights."""

from __future__ import annotations

from .models import (
    CriterionDefinition,
    CriterionJudgment,
    CriterionScore,
    FactStatus,
    MajorErrorConfirmation,
    RubricQuestion,
    ScoreResult,
)

SCORER_VERSION = "deterministic-v1"


class ScoringInputError(ValueError):
    """Semantic results cannot be mapped safely to the rubric."""


def _requires_evidence(status: FactStatus) -> bool:
    return status in {FactStatus.FULL, FactStatus.PARTIAL, FactStatus.INCORRECT}


def _explicit_subfact_score(
    criterion: CriterionDefinition, judgment: CriterionJudgment
) -> tuple[float | None, float, str | None]:
    if judgment.status == FactStatus.FULL:
        return criterion.criterion_weight, criterion.criterion_weight, None
    if judgment.status in {FactStatus.MISSING, FactStatus.INCORRECT}:
        return 0.0, 0.0, None
    if judgment.status == FactStatus.UNCERTAIN:
        return None, 0.0, "semantic judgment is uncertain"

    expected = {fact.fact_id: fact for fact in criterion.atomic_facts}
    supplied = {result.fact_id: result for result in judgment.atomic_results}
    if len(supplied) != len(judgment.atomic_results) or set(supplied) != set(expected):
        return None, 0.0, "partial explicit-subfact result is missing or duplicates atomic facts"
    score = 0.0
    for fact_id, fact in expected.items():
        result = supplied[fact_id]
        if result.status == FactStatus.FULL:
            if not result.evidence_quotes:
                return None, score, f"{fact_id} lacks answer evidence"
            score += fact.weight or 0.0
        elif result.status in {FactStatus.PARTIAL, FactStatus.UNCERTAIN}:
            return None, score, f"{fact_id} cannot be assigned an invented subfact partial weight"
    return round(score, 8), round(score, 8), None


def _score_criterion(
    criterion: CriterionDefinition,
    judgment: CriterionJudgment,
    forced_zero: bool,
) -> CriterionScore:
    if forced_zero:
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            status=FactStatus.INCORRECT,
            earned_score=0.0,
            provisional_score=0.0,
            max_score=criterion.criterion_weight,
            needs_review=False,
            evidence_quotes=judgment.evidence_quotes,
            rationale="Confirmed major error affects this criterion; scored zero once.",
        )
    if judgment.contradiction_detected:
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=None, provisional_score=0.0, max_score=criterion.criterion_weight,
            needs_review=True, review_reason="contradictory statements require human review",
            evidence_quotes=judgment.evidence_quotes, rationale=judgment.rationale,
        )
    if judgment.major_error_candidate:
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=None, provisional_score=0.0, max_score=criterion.criterion_weight,
            needs_review=True, review_reason="major-error candidate is not confirmed",
            evidence_quotes=judgment.evidence_quotes, rationale=judgment.rationale,
        )
    if _requires_evidence(judgment.status) and not judgment.evidence_quotes:
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=None, provisional_score=0.0, max_score=criterion.criterion_weight,
            needs_review=True, review_reason="semantic result lacks an answer quotation",
            evidence_quotes=[], rationale=judgment.rationale,
        )
    if criterion.classification in {"B", "D"}:
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=None, provisional_score=0.0, max_score=criterion.criterion_weight,
            needs_review=True,
            review_reason=f"class {criterion.classification} rule remains {criterion.review_status}",
            evidence_quotes=judgment.evidence_quotes, rationale=judgment.rationale,
        )
    if criterion.scoring_method == "equal_subfacts_pending_signoff":
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=None, provisional_score=0.0, max_score=criterion.criterion_weight,
            needs_review=True, review_reason="equal split has not been approved",
            evidence_quotes=judgment.evidence_quotes, rationale=judgment.rationale,
        )
    if criterion.scoring_method in {"explicit_subfacts", "equal_subfacts"}:
        earned, provisional, reason = _explicit_subfact_score(criterion, judgment)
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=earned, provisional_score=provisional,
            max_score=criterion.criterion_weight, needs_review=reason is not None,
            review_reason=reason, evidence_quotes=judgment.evidence_quotes,
            rationale=judgment.rationale,
        )
    if judgment.status == FactStatus.FULL:
        earned = criterion.criterion_weight
    elif (
        judgment.status == FactStatus.PARTIAL
        and criterion.scoring_method == "whole_full_half_zero"
        and criterion.review_status == "source_verified"
    ):
        earned = criterion.criterion_weight * 0.5
    elif judgment.status in {FactStatus.MISSING, FactStatus.INCORRECT}:
        earned = 0.0
    else:
        # C-class whole-point partial is deliberately not converted to 50% until R02 signoff.
        return CriterionScore(
            criterion_id=criterion.criterion_id, status=judgment.status,
            earned_score=None, provisional_score=0.0, max_score=criterion.criterion_weight,
            needs_review=True,
            review_reason=(
                "whole-point partial rule awaits global R02 signoff"
                if judgment.status == FactStatus.PARTIAL
                and "pending_global_R02_signoff" in criterion.scoring_method
                else "partial status has no signed numeric value for this criterion"
                if judgment.status == FactStatus.PARTIAL
                else "semantic judgment is uncertain"
            ),
            evidence_quotes=judgment.evidence_quotes, rationale=judgment.rationale,
        )
    return CriterionScore(
        criterion_id=criterion.criterion_id, status=judgment.status,
        earned_score=earned, provisional_score=earned, max_score=criterion.criterion_weight,
        needs_review=False, evidence_quotes=judgment.evidence_quotes,
        rationale=judgment.rationale,
    )


def score_answer(
    *,
    run_id: str,
    question: RubricQuestion,
    requested_mode: str,
    executed_mode: str,
    answer_text: str,
    judgments: list[CriterionJudgment],
    directly_addresses_question: bool | None,
    has_verifiable_fact: bool | None,
    answer_level_contradiction: bool = False,
    qualifier_issue_detected: bool = False,
    confirmed_major_errors: list[MajorErrorConfirmation] | None = None,
    prompt_version: str | None = None,
    scorer_config_id: str | None = None,
    raw_semantic_output: dict | None = None,
) -> ScoreResult:
    if requested_mode not in {"none", "keyword", "vector"} or executed_mode not in {"none", "keyword", "vector"}:
        raise ScoringInputError("Unknown execution mode")
    if requested_mode != executed_mode:
        raise ScoringInputError("Requested and executed modes differ")
    if round(sum(item.criterion_weight for item in question.criteria), 8) != question.question_weight:
        raise ScoringInputError("Rubric criterion weights do not equal the question weight")
    if question.question_weight <= 0:
        raise ScoringInputError("Question weight must be positive")
    expected = {criterion.criterion_id: criterion for criterion in question.criteria}
    if len(expected) != len(question.criteria):
        raise ScoringInputError("Duplicate criterion definition")
    supplied = {judgment.criterion_id: judgment for judgment in judgments}
    if len(supplied) != len(judgments):
        raise ScoringInputError("Duplicate criterion judgment")
    missing = sorted(set(expected) - set(supplied))
    extra = sorted(set(supplied) - set(expected))
    if missing or extra:
        raise ScoringInputError(f"Criterion judgment mismatch; missing={missing}, extra={extra}")

    confirmations = confirmed_major_errors or []
    affected = {criterion_id for item in confirmations for criterion_id in item.affected_criterion_ids}
    unknown_affected = affected - set(expected)
    if unknown_affected:
        raise ScoringInputError(f"Major error references unknown criteria: {sorted(unknown_affected)}")

    criterion_scores = [
        _score_criterion(criterion, supplied[criterion.criterion_id], criterion.criterion_id in affected)
        for criterion in question.criteria
    ]
    provisional_score = round(sum(item.provisional_score for item in criterion_scores), 8)
    review_reasons = [
        f"{item.criterion_id}: {item.review_reason}"
        for item in criterion_scores if item.needs_review and item.review_reason
    ]
    if directly_addresses_question is None or has_verifiable_fact is None:
        review_reasons.append("answer-level semantic checks are uncertain")
    if answer_level_contradiction:
        review_reasons.append("answer contains contradictory statements")
    if qualifier_issue_detected:
        review_reasons.append("required qualifier may be missing or overstated")
    needs_review = bool(review_reasons)
    major_error_detected = bool(confirmations)

    if needs_review:
        earned_score = None
        completeness = None
        final_correctness = "incorrect" if major_error_detected else None
    else:
        earned_score = round(sum((item.earned_score or 0.0) for item in criterion_scores), 8)
        completeness = round(earned_score / question.question_weight, 8)
        if major_error_detected or not directly_addresses_question or not has_verifiable_fact:
            final_correctness = "incorrect"
        elif completeness >= 0.85:
            final_correctness = "correct"
        elif completeness >= 0.35:
            final_correctness = "partial"
        else:
            final_correctness = "incorrect"

    return ScoreResult(
        run_id=run_id, question_id=question.question_id,
        requested_mode=requested_mode, executed_mode=executed_mode,
        answer_text=answer_text, criterion_results=criterion_scores,
        earned_score=earned_score, provisional_earned_score=provisional_score,
        max_score=question.question_weight, completeness_score=completeness,
        provisional_completeness_score=round(provisional_score / question.question_weight, 8),
        major_error_detected=major_error_detected,
        major_error_confirmations=confirmations, final_correctness=final_correctness,
        needs_review=needs_review, review_reason=review_reasons,
        scorer_version=SCORER_VERSION, rubric_version=question.rubric_version,
        prompt_version=prompt_version, scorer_config_id=scorer_config_id,
        raw_semantic_output=raw_semantic_output,
    )

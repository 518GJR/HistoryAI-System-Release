"""Versioned, auditable data models used by the offline evaluator."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictStr,
    model_validator,
)


class FactStatus(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    MISSING = "missing"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


class SourceReference(BaseModel):
    source_file: str
    sheet: str | None = None
    cells: list[str] = Field(default_factory=list)
    section: str | None = None
    sha256: str | None = None


class AtomicFact(BaseModel):
    fact_id: str
    description: str
    weight: float | None = None
    proposed_weight: str | None = None
    approval_status: str = "source_verified"


class ScoringOption(BaseModel):
    option_id: str
    description: str
    status: str = "pending_adjudication"
    selected: bool = False


class CriterionDefinition(BaseModel):
    criterion_id: str
    criterion_description: str
    criterion_weight: float
    original_criterion_text: str
    original_weight_text: str
    classification: Literal["A", "B", "C", "D"]
    scoring_method: str
    atomic_facts: list[AtomicFact] = Field(default_factory=list)
    scoring_options: list[ScoringOption] = Field(default_factory=list)
    allowed_equivalences: list[str] = Field(default_factory=list)
    disallowed_equivalences: list[str] = Field(default_factory=list)
    equivalence_rules_raw: str = ""
    major_error_rules: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    review_status: str
    blocking_reason: str | None = None
    classification_record: dict[str, str] = Field(default_factory=dict)


class RubricQuestion(BaseModel):
    question_id: str
    question_text: str
    standard_answer: str
    rubric_version: str
    question_weight: float = 10.0
    criteria: list[CriterionDefinition]
    textbook_references: list[str] = Field(default_factory=list)
    lesson: str = ""
    knowledge_point_ids: list[str] = Field(default_factory=list)
    question_type: str = ""
    difficulty: str = ""
    original_major_error_text: str = ""
    original_equivalence_text: str = ""
    original_source_record: dict[str, str] = Field(default_factory=dict)
    question_group: Literal["Q", "K", "R"] | None = None
    mode_specific_guidance: dict[str, str] = Field(default_factory=dict)
    reference_text_verification: dict[str, Any] = Field(default_factory=dict)


class RubricDataset(BaseModel):
    schema_version: str = "evaluation-rubric-v1"
    rubric_version: str
    source_release: str
    generated_at: datetime
    source_files: list[SourceReference]
    questions: list[RubricQuestion]
    classification_counts: dict[str, int]
    conversion_status: str
    conversion_notes: list[str] = Field(default_factory=list)


class AtomicJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    status: FactStatus
    evidence_quotes: list[str] = Field(default_factory=list)
    rationale: str = ""
    contradiction_detected: StrictBool = False
    major_error_candidate: StrictBool = False


class CriterionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    status: FactStatus
    evidence_quotes: list[str] = Field(default_factory=list)
    rationale: str = ""
    atomic_results: list[AtomicJudgment] = Field(default_factory=list)
    contradiction_detected: StrictBool = False
    major_error_candidate: StrictBool = False
    raw_model_output: dict[str, Any] | None = None

    @model_validator(mode="after")
    def propagate_atomic_review_flags(self):
        # Atomic flags are formal review signals. Preserve them on each atomic
        # result and promote them so the criterion-level scoring path cannot
        # silently lose a contradiction or major-error candidate.
        if any(item.contradiction_detected for item in self.atomic_results):
            self.contradiction_detected = True
        if any(item.major_error_candidate for item in self.atomic_results):
            self.major_error_candidate = True
        return self


class MajorErrorCandidate(BaseModel):
    """Unconfirmed model observation; it can only trigger human review."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: StrictStr
    answer_quote: StrictStr
    rationale: StrictStr
    # V2 compatibility only. This is untrusted model prose and never selects or
    # overrides a rule. V3 prompts omit it and reference the criterion ID only.
    model_rule_text: StrictStr | None = Field(default=None, validation_alias="rule_text")
    _frozen_rule_texts: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="after")
    def require_non_empty_trace(self):
        for field_name in ("criterion_id", "answer_quote", "rationale"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"major error candidate has empty {field_name}")
        if self.model_rule_text is not None and not self.model_rule_text.strip():
            raise ValueError("major error candidate has empty model rule text")
        return self

    @property
    def frozen_rule_texts(self) -> tuple[str, ...]:
        return self._frozen_rule_texts

    def bind_frozen_rules(self, rules: list[str]) -> None:
        """Attach authoritative rules after rubric ID validation."""

        self._frozen_rule_texts = tuple(rules)

    def audit_dump(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "answer_quote": self.answer_quote,
            "rationale": self.rationale,
            "model_rule_text": self.model_rule_text,
            "frozen_rule_texts": list(self.frozen_rule_texts),
            "rule_source": "rubric",
        }


class MajorErrorConfirmation(BaseModel):
    rule_text: str
    answer_quote: str
    textbook_fact: str
    source_reference: str
    affected_criterion_ids: list[str]
    confirmed_by: str


class CriterionScore(BaseModel):
    criterion_id: str
    status: FactStatus
    earned_score: float | None
    max_score: float
    provisional_score: float
    needs_review: bool
    review_reason: str | None = None
    evidence_quotes: list[str] = Field(default_factory=list)
    rationale: str = ""


class ScoreResult(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    run_id: str
    question_id: str
    requested_mode: Literal["none", "keyword", "vector"]
    executed_mode: Literal["none", "keyword", "vector"]
    answer_text: str
    criterion_results: list[CriterionScore]
    earned_score: float | None
    provisional_earned_score: float
    max_score: float
    completeness_score: float | None
    provisional_completeness_score: float
    major_error_detected: bool
    major_error_confirmations: list[MajorErrorConfirmation] = Field(default_factory=list)
    final_correctness: Literal["correct", "partial", "incorrect"] | None
    needs_review: bool
    review_reason: list[str] = Field(default_factory=list)
    scorer_version: str
    rubric_version: str
    prompt_version: str | None = None
    scorer_config_id: str | None = None
    raw_semantic_output: dict[str, Any] | None = None
    automatic_result_preserved: bool = True


class ReviewRecord(BaseModel):
    review_id: str
    run_id: str
    question_id: str
    requested_mode: Literal["none", "keyword", "vector"]
    needs_review: bool
    review_reason: list[str]
    review_status: Literal["pending", "confirmed", "overridden"]
    reviewer_decision: Literal["correct", "partial", "incorrect"] | None = None
    reviewer_score: float | None = Field(default=None, ge=0, le=10)
    reviewer_comment: str = ""
    reviewed_at: datetime | None = None
    automatic_result: dict[str, Any]

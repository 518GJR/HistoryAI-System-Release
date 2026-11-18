import json
from pathlib import Path

import httpx
import pytest

from app.services.evaluation.models import (
    AtomicFact,
    AtomicJudgment,
    CriterionDefinition,
    CriterionJudgment,
    FactStatus,
    MajorErrorConfirmation,
    RubricQuestion,
)
from app.services.evaluation.scoring_engine import ScoringInputError, score_answer
from app.services.evaluation.rubric_loader import load_rubric
from app.services.evaluation.semantic_judge import (
    OpenAICompatibleSemanticJudge,
    SemanticJudgeError,
)
from app.services.evaluation import semantic_judge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUBRIC_PATH = PROJECT_ROOT / "docs/evaluation/workbench/unit1_v1.0_rubric.json"


def criterion(
    identifier,
    weight,
    method="whole_full_half_zero",
    classification="A",
    facts=None,
    review="source_verified",
    major_error_rules=None,
):
    return CriterionDefinition(
        criterion_id=identifier,
        criterion_description=identifier,
        criterion_weight=weight,
        original_criterion_text=identifier,
        original_weight_text=f"{weight}分",
        classification=classification,
        scoring_method=method,
        atomic_facts=facts or [],
        major_error_rules=major_error_rules or [],
        review_status=review,
    )


def question(criteria):
    return RubricQuestion(
        question_id="QX", question_text="question", standard_answer="answer",
        rubric_version="test-v1", criteria=criteria,
    )


def judgment(identifier, status, evidence=True, **kwargs):
    return CriterionJudgment(
        criterion_id=identifier, status=status,
        evidence_quotes=["answer quote"] if evidence else [], **kwargs,
    )


def score(q, judgments, **kwargs):
    return score_answer(
        run_id="run", question=q, requested_mode="keyword", executed_mode="keyword",
        answer_text="answer quote", judgments=judgments,
        directly_addresses_question=kwargs.pop("directly_addresses_question", True),
        has_verifiable_fact=kwargs.pop("has_verifiable_fact", True), **kwargs,
    )


def test_explicit_subweights_accumulate_without_second_half_penalty():
    facts = [
        AtomicFact(fact_id="QX-C01-F01", description="a", weight=1),
        AtomicFact(fact_id="QX-C01-F02", description="b", weight=1),
        AtomicFact(fact_id="QX-C01-F03", description="c", weight=1),
    ]
    q = question([
        criterion("QX-C01", 3, "explicit_subfacts", facts=facts),
        criterion("QX-C02", 7),
    ])
    partial = judgment("QX-C01", FactStatus.PARTIAL)
    partial.atomic_results = [
        AtomicJudgment(fact_id="QX-C01-F01", status="full", evidence_quotes=["a"]),
        AtomicJudgment(fact_id="QX-C01-F02", status="missing"),
        AtomicJudgment(fact_id="QX-C01-F03", status="full", evidence_quotes=["c"]),
    ]
    result = score(q, [partial, judgment("QX-C02", FactStatus.FULL)])

    assert result.earned_score == 9
    assert result.criterion_results[0].earned_score == 2
    assert result.final_correctness == "correct"


def test_approved_equal_split_accumulates_subfacts():
    facts = [
        AtomicFact(fact_id="QX-C01-F01", description="a", weight=2),
        AtomicFact(fact_id="QX-C01-F02", description="b", weight=2),
    ]
    q = question([
        criterion("QX-C01", 4, "equal_subfacts", facts=facts),
        criterion("QX-C02", 6),
    ])
    partial = judgment("QX-C01", FactStatus.PARTIAL)
    partial.atomic_results = [
        AtomicJudgment(fact_id="QX-C01-F01", status="full", evidence_quotes=["a"]),
        AtomicJudgment(fact_id="QX-C01-F02", status="missing"),
    ]
    result = score(q, [partial, judgment("QX-C02", FactStatus.FULL)])

    assert result.criterion_results[0].earned_score == 2
    assert result.earned_score == 8
    assert result.final_correctness == "partial"


def test_approved_whole_point_partial_uses_half_and_threshold_edges_are_exact():
    q = question([criterion("QX-C01", 7), criterion("QX-C02", 3)])
    result = score(q, [judgment("QX-C01", FactStatus.FULL), judgment("QX-C02", FactStatus.PARTIAL)])
    assert result.earned_score == 8.5
    assert result.completeness_score == 0.85
    assert result.final_correctness == "correct"

    q = question([criterion("QX-C01", 3.5), criterion("QX-C02", 6.5)])
    result = score(q, [judgment("QX-C01", FactStatus.FULL), judgment("QX-C02", FactStatus.MISSING, evidence=False)])
    assert result.completeness_score == 0.35
    assert result.final_correctness == "partial"


def test_major_error_is_applied_once_and_independent_score_is_retained():
    q = question([criterion("QX-C01", 4), criterion("QX-C02", 6)])
    confirmation = MajorErrorConfirmation(
        rule_text="frozen rule", answer_quote="wrong", textbook_fact="correct",
        source_reference="page 1", affected_criterion_ids=["QX-C01"], confirmed_by="human",
    )
    result = score(q, [judgment("QX-C01", "full"), judgment("QX-C02", "full")], confirmed_major_errors=[confirmation])

    assert result.earned_score == 6
    assert result.completeness_score == 0.6
    assert result.final_correctness == "incorrect"
    assert result.criterion_results[1].earned_score == 6


@pytest.mark.parametrize("reason", ["contradiction", "uncertain", "D"])
def test_ambiguous_and_d_class_results_require_review(reason):
    classification = "D" if reason == "D" else "A"
    review = "blocked_pending_adjudication" if reason == "D" else "source_verified"
    q = question([criterion("QX-C01", 10, classification=classification, review=review)])
    item = judgment(
        "QX-C01",
        FactStatus.UNCERTAIN if reason == "uncertain" else FactStatus.FULL,
        evidence=reason != "uncertain",
        contradiction_detected=reason == "contradiction",
    )
    result = score(q, [item])

    assert result.needs_review is True
    assert result.earned_score is None
    assert result.final_correctness is None


def test_missing_evidence_and_major_error_candidate_do_not_become_definite_scores():
    q = question([criterion("QX-C01", 10)])
    no_evidence = score(q, [judgment("QX-C01", FactStatus.FULL, evidence=False)])
    candidate = score(q, [judgment("QX-C01", FactStatus.INCORRECT, major_error_candidate=True)])

    assert no_evidence.needs_review and no_evidence.final_correctness is None
    assert candidate.needs_review and not candidate.major_error_detected


def test_answer_level_contradiction_and_qualifier_issue_require_review():
    q = question([criterion("QX-C01", 10)])
    contradiction = score(q, [judgment("QX-C01", FactStatus.FULL)], answer_level_contradiction=True)
    qualifier = score(q, [judgment("QX-C01", FactStatus.FULL)], qualifier_issue_detected=True)

    assert contradiction.needs_review and contradiction.final_correctness is None
    assert qualifier.needs_review and qualifier.final_correctness is None


def test_unanswered_or_non_verifiable_answer_is_incorrect_not_execution_failure():
    q = question([criterion("QX-C01", 10)])
    result = score(
        q,
        [judgment("QX-C01", FactStatus.MISSING, evidence=False)],
        directly_addresses_question=False,
        has_verifiable_fact=False,
    )

    assert result.needs_review is False
    assert result.final_correctness == "incorrect"
    assert result.completeness_score == 0


@pytest.mark.parametrize(
    ("method", "classification", "review"),
    [
        ("whole_full_half_zero_pending_global_R02_signoff", "C", "pending_global_R02_signoff"),
        ("equal_subfacts_pending_signoff", "A", "pending_owner_signoff"),
    ],
)
def test_unapproved_partial_rules_remain_review_only(method, classification, review):
    q = question([criterion("QX-C01", 10, method=method, classification=classification, review=review)])
    result = score(q, [judgment("QX-C01", FactStatus.PARTIAL)])

    assert result.earned_score is None
    assert result.needs_review is True
    assert result.final_correctness is None


def test_missing_duplicate_and_mode_mismatch_inputs_are_rejected():
    q = question([criterion("QX-C01", 5), criterion("QX-C02", 5)])
    with pytest.raises(ScoringInputError, match="missing"):
        score(q, [judgment("QX-C01", "full")])
    with pytest.raises(ScoringInputError, match="Duplicate"):
        score(q, [judgment("QX-C01", "full"), judgment("QX-C01", "full")])
    with pytest.raises(ScoringInputError, match="differ"):
        score_answer(
            run_id="run", question=q, requested_mode="keyword", executed_mode="vector",
            answer_text="x", judgments=[judgment("QX-C01", "full"), judgment("QX-C02", "full")],
            directly_addresses_question=True, has_verifiable_fact=True,
        )


def test_scoring_engine_rejects_invalid_rubric_weight_and_duplicate_definition():
    invalid_weight = question([criterion("QX-C01", 9)])
    with pytest.raises(ScoringInputError, match="weights"):
        score(invalid_weight, [judgment("QX-C01", "full")])

    duplicate = question([criterion("QX-C01", 5), criterion("QX-C01", 5)])
    with pytest.raises(ScoringInputError, match="Duplicate criterion definition"):
        score(duplicate, [judgment("QX-C01", "full"), judgment("QX-C02", "full")])


def test_deterministic_result_is_identical_for_same_input():
    q = question([criterion("QX-C01", 10)])
    args = [judgment("QX-C01", FactStatus.FULL)]
    assert score(q, args).model_dump() == score(q, args).model_dump()


def valid_semantic_payload():
    return {
        "criterion_results": [
            {
                "criterion_id": "QX-C01",
                "status": "full",
                "evidence_quotes": ["answer"],
                "rationale": "The answer states the required fact.",
                "atomic_results": [],
                "contradiction_detected": False,
                "major_error_candidate": False,
            }
        ],
        "directly_addresses_question": True,
        "has_verifiable_fact": True,
        "answer_level_contradiction": False,
        "qualifier_issue_detected": False,
        "major_error_candidates": [],
    }


def configured_semantic_judge(retries=0):
    return OpenAICompatibleSemanticJudge(config={
        "provider": "openai_compatible", "key": "test", "model": "deepseek-flash",
        "url": "https://api.deepseek.com/chat/completions", "timeout": 1,
    }, retries=retries)


def install_semantic_mock(monkeypatch, handler):
    monkeypatch.setattr(
        semantic_judge,
        "_create_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout),
    )


def test_semantic_judge_accepts_valid_deepseek_shape(monkeypatch):
    def handler(request):
        request_payload = json.loads(request.content)
        assert request.url.path == "/chat/completions"
        assert request_payload["model"] == "deepseek-flash"
        assert request_payload["response_format"] == {"type": "json_object"}
        assert "criterion_results" in request_payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(valid_semantic_payload())}}]},
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, raw = configured_semantic_judge().judge(
        question([criterion("QX-C01", 10)]), "answer"
    )

    assert assessment.criterion_results[0].status == FactStatus.FULL
    assert assessment.criterion_results[0].evidence_quotes == ["answer"]
    assert raw == valid_semantic_payload()


def test_semantic_judge_accepts_atomic_review_flags_and_promotes_them(monkeypatch):
    payload = valid_semantic_payload()
    payload["criterion_results"][0]["atomic_results"] = [
        {
            "fact_id": "QX-C01-F01",
            "status": "incorrect",
            "evidence_quotes": ["contradictory answer quote"],
            "rationale": "The atomic fact conflicts with the required fact.",
            "contradiction_detected": True,
            "major_error_candidate": True,
        }
    ]

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    q = question([
        criterion(
            "QX-C01",
            10,
            method="explicit_subfacts",
            facts=[AtomicFact(fact_id="QX-C01-F01", description="required fact", weight=10)],
            major_error_rules=["Reversing the required atomic fact is a major-error candidate."],
        )
    ])
    assessment, raw = configured_semantic_judge().judge(q, "contradictory answer quote")
    result = assessment.criterion_results[0]

    assert result.atomic_results[0].contradiction_detected is True
    assert result.atomic_results[0].major_error_candidate is True
    assert result.contradiction_detected is True
    assert result.major_error_candidate is True
    assert raw == payload


def test_semantic_judge_accepts_structured_major_error_candidate_and_preserves_review_signal(monkeypatch):
    payload = valid_semantic_payload()
    payload["major_error_candidates"] = [
        {
            "criterion_id": "QX-C01",
            "rule_text": "The model paraphrases the frozen rule instead of copying it.",
            "answer_quote": "answer",
            "rationale": "The quoted answer may reverse the required fact.",
        }
    ]

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, raw = configured_semantic_judge().judge(
        question([
            criterion(
                "QX-C01",
                10,
                major_error_rules=["Changing the core fact is a major-error candidate."],
            )
        ]),
        "answer",
    )
    scored = score(question([criterion("QX-C01", 10)]), assessment.criterion_results)

    assert assessment.major_error_candidates[0].criterion_id == "QX-C01"
    assert assessment.major_error_candidates[0].model_rule_text.startswith("The model paraphrases")
    assert assessment.major_error_candidates[0].frozen_rule_texts == (
        "Changing the core fact is a major-error candidate.",
    )
    assert assessment.criterion_results[0].major_error_candidate is True
    assert scored.needs_review is True
    assert scored.review_reason == ["QX-C01: major-error candidate is not confirmed"]
    assert raw["major_error_candidates"][0]["model_rule_text"].startswith(
        "The model paraphrases"
    )
    assert raw["major_error_candidates"][0]["frozen_rule_texts"] == [
        "Changing the core fact is a major-error candidate."
    ]
    assert raw["major_error_candidates"][0]["rule_source"] == "rubric"


@pytest.mark.parametrize(
    ("question_id", "criterion_id", "model_rule_text"),
    [
        ("Q011", "Q011-C05", "把两个遗址的关键信息混写，可能构成重大错误。"),
        ("Q016", "Q016-C06", "把陶寺遗址的地点或观象台结论说错。"),
    ],
)
def test_real_recovery_candidate_ids_bind_frozen_rules_and_require_review(
    monkeypatch, question_id, criterion_id, model_rule_text
):
    rubric = load_rubric(RUBRIC_PATH)
    q = next(item for item in rubric.questions if item.question_id == question_id)
    criterion_by_id = {item.criterion_id: item for item in q.criteria}
    payload = {
        "criterion_results": [
            {
                "criterion_id": item.criterion_id,
                "status": "missing",
                "evidence_quotes": [],
                "rationale": "The mock answer does not state this criterion.",
                "atomic_results": [],
                "contradiction_detected": False,
                "major_error_candidate": False,
            }
            for item in q.criteria
        ],
        "directly_addresses_question": True,
        "has_verifiable_fact": True,
        "answer_level_contradiction": False,
        "qualifier_issue_detected": False,
        "major_error_candidates": [
            {
                "criterion_id": criterion_id,
                "rule_text": model_rule_text,
                "answer_quote": "mock risky statement",
                "rationale": "This statement may trigger a frozen major-error rule.",
            }
        ],
    }

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, raw = configured_semantic_judge().judge(q, "mock risky statement")
    candidate = assessment.major_error_candidates[0]
    result = score_answer(
        run_id="recovery-v3",
        question=q,
        requested_mode="none",
        executed_mode="none",
        answer_text="mock risky statement",
        judgments=assessment.criterion_results,
        directly_addresses_question=assessment.directly_addresses_question,
        has_verifiable_fact=assessment.has_verifiable_fact,
        answer_level_contradiction=assessment.answer_level_contradiction,
        qualifier_issue_detected=assessment.qualifier_issue_detected,
        confirmed_major_errors=[],
        prompt_version=semantic_judge.PROMPT_VERSION,
        scorer_config_id="mock-config",
        raw_semantic_output=raw,
    )

    assert candidate.model_rule_text == model_rule_text
    assert candidate.frozen_rule_texts == tuple(
        criterion_by_id[criterion_id].major_error_rules
    )
    assert raw["major_error_candidates"][0]["frozen_rule_texts"] == (
        criterion_by_id[criterion_id].major_error_rules
    )
    assert assessment.criterion_results[
        next(i for i, item in enumerate(q.criteria) if item.criterion_id == criterion_id)
    ].major_error_candidate is True
    assert result.major_error_detected is False
    assert result.needs_review is True
    assert result.final_correctness is None
    assert f"{criterion_id}: major-error candidate is not confirmed" in result.review_reason


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": 201},
        {**valid_semantic_payload(), "unexpected": True},
        {
            **valid_semantic_payload(),
            "criterion_results": [
                {**valid_semantic_payload()["criterion_results"][0], "unexpected": True}
            ],
        },
        {
            **valid_semantic_payload(),
            "criterion_results": [
                {
                    **valid_semantic_payload()["criterion_results"][0],
                    "atomic_results": [
                        {
                            "fact_id": "QX-C01-F01",
                            "status": "missing",
                            "evidence_quotes": [],
                            "rationale": "not stated",
                            "contradiction_detected": False,
                            "major_error_candidate": False,
                            "unexpected": True,
                        }
                    ],
                }
            ],
        },
        {
            **valid_semantic_payload(),
            "major_error_candidates": [
                {
                    "criterion_id": "QX-C01",
                    "rule_text": "rule",
                    "answer_quote": "answer",
                    "rationale": "reason",
                    "unexpected": True,
                }
            ],
        },
        {
            **valid_semantic_payload(),
            "major_error_candidates": [
                {
                    "criterion_id": "QX-C01",
                    "answer_quote": "answer",
                    "rationale": "reason",
                    "frozen_rule_texts": ["model must not provide authority fields"],
                }
            ],
        },
        {
            **valid_semantic_payload(),
            "criterion_results": [
                {**valid_semantic_payload()["criterion_results"][0], "status": "invented"}
            ],
        },
        {
            **valid_semantic_payload(),
            "criterion_results": [
                {
                    **valid_semantic_payload()["criterion_results"][0],
                    "evidence_quotes": [],
                }
            ],
        },
        {
            **valid_semantic_payload(),
            "criterion_results": [
                {**valid_semantic_payload()["criterion_results"][0], "rationale": ""}
            ],
        },
        {
            **valid_semantic_payload(),
            "criterion_results": [
                {
                    **valid_semantic_payload()["criterion_results"][0],
                    "contradiction_detected": "false",
                }
            ],
        },
        {**valid_semantic_payload(), "major_error_candidates": ["QX-C01"]},
        {
            **valid_semantic_payload(),
            "major_error_candidates": [
                {
                    "criterion_id": "QX-C01",
                    "rule_text": "rule",
                    "answer_quote": "answer",
                }
            ],
        },
    ],
)
def test_semantic_judge_rejects_invalid_scoring_structures(monkeypatch, payload):
    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError):
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")


def test_semantic_judge_rejects_unknown_atomic_and_candidate_ids(monkeypatch):
    payloads = []
    unknown_atomic = valid_semantic_payload()
    unknown_atomic["criterion_results"][0]["atomic_results"] = [
        {
            "fact_id": "QX-C01-F99",
            "status": "missing",
            "evidence_quotes": [],
            "rationale": "not stated",
            "contradiction_detected": False,
            "major_error_candidate": False,
        }
    ]
    payloads.append(unknown_atomic)
    unknown_candidate = valid_semantic_payload()
    unknown_candidate["major_error_candidates"] = [
        {
            "criterion_id": "QX-C99",
            "rule_text": "rule",
            "answer_quote": "answer",
            "rationale": "reason",
        }
    ]
    payloads.append(unknown_candidate)

    for payload in payloads:
        def handler(request, payload=payload):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        install_semantic_mock(monkeypatch, handler)
        with pytest.raises(SemanticJudgeError, match="IDs do not match"):
            configured_semantic_judge().judge(
                question([
                    criterion(
                        "QX-C01",
                        10,
                        facts=[AtomicFact(fact_id="QX-C01-F01", description="fact", weight=10)],
                    )
                ]),
                "answer",
            )


def test_model_rule_prose_cannot_override_frozen_rule(monkeypatch):
    payload = valid_semantic_payload()
    payload["major_error_candidates"] = [
        {
            "criterion_id": "QX-C01",
            "rule_text": "invented rule",
            "answer_quote": "answer",
            "rationale": "reason",
        }
    ]

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, raw = configured_semantic_judge().judge(
        question([criterion("QX-C01", 10, major_error_rules=["frozen rule"])]),
        "answer",
    )

    assert assessment.major_error_candidates[0].model_rule_text == "invented rule"
    assert assessment.major_error_candidates[0].frozen_rule_texts == ("frozen rule",)
    assert raw["major_error_candidates"][0]["frozen_rule_texts"] == ["frozen rule"]


def test_semantic_judge_rejects_candidate_without_frozen_rule(monkeypatch):
    payload = valid_semantic_payload()
    payload["major_error_candidates"] = [
        {
            "criterion_id": "QX-C01",
            "answer_quote": "answer",
            "rationale": "reason",
        }
    ]

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    with pytest.raises(SemanticJudgeError, match="has no frozen rule"):
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")


def test_semantic_judge_rejects_cross_criterion_rule_reference(monkeypatch):
    payload = valid_semantic_payload()
    second = dict(payload["criterion_results"][0])
    second["criterion_id"] = "QX-C02"
    payload["criterion_results"].append(second)
    payload["major_error_candidates"] = [
        {
            "criterion_id": "QX-C01",
            "rule_text": "frozen rule B",
            "answer_quote": "answer",
            "rationale": "reason",
        }
    ]

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    q = question([
        criterion("QX-C01", 5, major_error_rules=["frozen rule A"]),
        criterion("QX-C02", 5, major_error_rules=["frozen rule B"]),
    ])
    with pytest.raises(SemanticJudgeError, match="conflicts with QX-C01"):
        configured_semantic_judge().judge(q, "answer")


def test_semantic_schema_error_keeps_safe_diagnostic_path(monkeypatch):
    payload = valid_semantic_payload()
    payload["major_error_candidates"] = [{"criterion_id": "QX-C01"}]

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError) as captured:
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")

    message = str(captured.value)
    assert "major_error_candidates" in message
    assert "answer_quote" in message
    assert "test" not in message


@pytest.mark.parametrize("content", ["", "   ", "not-json", None])
def test_semantic_judge_rejects_empty_or_non_json_content(monkeypatch, content):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError):
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")


@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_semantic_judge_rejects_http_errors(monkeypatch, status_code):
    def handler(request):
        return httpx.Response(status_code, json={"error": {"message": "mock failure"}})

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError):
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not-response-json"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {}}]}),
    ],
)
def test_semantic_judge_rejects_invalid_api_response_shapes(monkeypatch, response):
    def handler(request):
        return response

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError):
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")


def test_semantic_judge_wraps_transport_failures(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("mock connection failed", request=request)

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError):
        configured_semantic_judge().judge(question([criterion("QX-C01", 10)]), "answer")


def test_semantic_judge_retries_ssl_eof_then_succeeds(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError(
                "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol",
                request=request,
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(valid_semantic_payload())}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, _ = configured_semantic_judge(retries=1).judge(
        question([criterion("QX-C01", 10)]), "answer"
    )

    assert calls == 2
    assert assessment.criterion_results[0].status == FactStatus.FULL


def test_semantic_judge_stops_after_repeated_ssl_eof(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadError(
            "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol",
            request=request,
        )

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError, match="UNEXPECTED_EOF_WHILE_READING"):
        configured_semantic_judge(retries=1).judge(
            question([criterion("QX-C01", 10)]), "answer"
        )
    assert calls == 2


@pytest.mark.parametrize("status_code", [401, 403])
def test_semantic_judge_does_not_retry_authentication_errors(monkeypatch, status_code):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json={"error": {"message": "denied"}})

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError):
        configured_semantic_judge(retries=2).judge(
            question([criterion("QX-C01", 10)]), "answer"
        )
    assert calls == 1


@pytest.mark.parametrize("status_code", [429, 500])
def test_semantic_judge_retries_retryable_http_status(monkeypatch, status_code):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, json={"error": {"message": "retry"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(valid_semantic_payload())}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, _ = configured_semantic_judge(retries=1).judge(
        question([criterion("QX-C01", 10)]), "answer"
    )

    assert calls == 2
    assert assessment.criterion_results[0].status == FactStatus.FULL


def test_semantic_judge_repairs_one_schema_error_without_migrating_extra_field(monkeypatch):
    calls = []
    invalid = {**valid_semantic_payload(), "major_error_candidate": False}

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        content = invalid if len(calls) == 1 else valid_semantic_payload()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, raw = configured_semantic_judge(retries=3).judge(
        question([criterion("QX-C01", 10)]), "answer"
    )

    assert len(calls) == 2
    assert len(calls[1]["messages"]) == 4
    assert "major_error_candidate" in calls[1]["messages"][-1]["content"]
    assert "不得迁移或推断非法字段" in calls[1]["messages"][-1]["content"]
    assert assessment.criterion_results[0].status == FactStatus.FULL
    assert "major_error_candidate" not in raw


def test_semantic_judge_allows_only_one_schema_correction(monkeypatch):
    calls = 0
    invalid = {**valid_semantic_payload(), "major_error_candidate": False}

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(invalid)}}]}
        )

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError, match="extra_forbidden"):
        configured_semantic_judge(retries=3).judge(
            question([criterion("QX-C01", 10)]), "answer"
        )
    assert calls == 2


def test_schema_validation_error_context_is_json_safe(monkeypatch):
    calls = 0
    invalid = valid_semantic_payload()
    invalid["criterion_results"][0]["rationale"] = ""

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(invalid)}}]}
        )

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError) as captured:
        configured_semantic_judge().judge(
            question([criterion("QX-C01", 10)]), "answer"
        )

    assert calls == 2
    assert "Invalid semantic judge response schema" in str(captured.value)
    assert "Object of type ValueError is not JSON serializable" not in str(captured.value)


def test_illegal_major_error_candidate_gets_one_bounded_correction(monkeypatch):
    calls = []
    invalid = valid_semantic_payload()
    invalid["criterion_results"][0]["major_error_candidate"] = True

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        content = invalid if len(calls) == 1 else valid_semantic_payload()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, _ = configured_semantic_judge().judge(
        question([criterion("QX-C01", 10)]), "answer"
    )

    assert len(calls) == 2
    assert "no frozen rule" in calls[1]["messages"][-1]["content"]
    assert assessment.criterion_results[0].major_error_candidate is False


def test_illegal_major_error_candidate_allows_only_one_correction(monkeypatch):
    calls = 0
    invalid = valid_semantic_payload()
    invalid["criterion_results"][0]["major_error_candidate"] = True

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(invalid)}}]}
        )

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(SemanticJudgeError, match="has no frozen rule"):
        configured_semantic_judge().judge(
            question([criterion("QX-C01", 10)]), "answer"
        )

    assert calls == 2


def test_repeated_timeout_has_distinct_exhausted_retry_diagnostic(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("mock read timeout", request=request)

    install_semantic_mock(monkeypatch, handler)

    with pytest.raises(
        SemanticJudgeError,
        match="Semantic judge request timed out after 2 attempts",
    ):
        configured_semantic_judge(retries=1).judge(
            question([criterion("QX-C01", 10)]), "answer"
        )

    assert calls == 2


def test_semantic_judge_preserves_legal_criterion_major_error_candidate(monkeypatch):
    payload = valid_semantic_payload()
    payload["criterion_results"][0]["major_error_candidate"] = True

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    install_semantic_mock(monkeypatch, handler)
    assessment, _ = configured_semantic_judge().judge(
        question([
            criterion(
                "QX-C01",
                10,
                major_error_rules=["Changing the core fact is a major-error candidate."],
            )
        ]),
        "answer",
    )

    assert assessment.criterion_results[0].major_error_candidate is True


def test_prompt_forbids_unapproved_equivalence_expansion():
    assert "不得扩充输入中没有的等价表达" in semantic_judge.SYSTEM_PROMPT
    assert "不得填写字符串或规则文本" in semantic_judge.SYSTEM_PROMPT
    assert "程序将按 criterion_id 绑定权威冻结规则" in semantic_judge.SYSTEM_PROMPT
    assert semantic_judge.PROMPT_VERSION == "history-fact-judge-v3"

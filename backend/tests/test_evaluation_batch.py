import json
from pathlib import Path

import httpx
import pytest

from app.services.evaluation.models import CriterionJudgment, FactStatus, ReviewRecord, ScoreResult
from app.services.evaluation.formal_dataset_loader import load_evaluation_dataset
from app.services.evaluation.rubric_loader import load_rubric
from app.services.evaluation.semantic_judge import SemanticAssessment, SemanticJudgeError
from scripts.run_evaluation import EvaluationRunner, build_summary, parse_args, select_questions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUBRIC_PATH = PROJECT_ROOT / "docs/evaluation/workbench/unit1_v1.0_rubric.json"
FORMAL_DATASET_PATH = PROJECT_ROOT / "docs/evaluation/releases/formal_100_v1.0/formal_evaluation_100.json"
PILOT_QUESTION_IDS = ["K05-01", "K09-02", "R01", "R03"]


class MockJudge:
    prompt_version = "mock-v1"
    config_id = "mock-config"

    def judge(self, question, answer_text):
        assessment = SemanticAssessment(
            criterion_results=[
                CriterionJudgment(
                    criterion_id=c.criterion_id,
                    status=FactStatus.MISSING,
                    rationale="mock criterion is absent",
                )
                for c in question.criteria
            ],
            directly_addresses_question=True,
            has_verifiable_fact=True,
        )
        return assessment, assessment.model_dump(mode="json")


def api_payload(mode):
    return {
        "requested_mode": mode, "executed_mode": mode, "answer": f"mock {mode}",
        "sources": [], "error": None, "answer_mode": "mock", "retrieval_mode": mode,
        "history_saved": True, "used_evidence": [], "support_assessment": None,
        "knowledge_outline": None,
    }


def test_question_ids_default_to_full_formal_dataset_and_select_requested_subset():
    rubric = load_evaluation_dataset(FORMAL_DATASET_PATH)

    assert len(rubric.questions) == 100
    assert select_questions(rubric, None) is rubric

    selected = select_questions(rubric, PILOT_QUESTION_IDS)
    assert [question.question_id for question in selected.questions] == PILOT_QUESTION_IDS
    assert [question.question_weight for question in selected.questions] == [10, 10, 3, 3]
    assert selected.classification_counts == {"Q": 0, "K": 2, "R": 2}
    assert len(rubric.questions) == 100

    args = parse_args(["--question-ids", *PILOT_QUESTION_IDS, "--expected-questions", "4"])
    assert args.question_ids == PILOT_QUESTION_IDS
    assert args.expected_questions == 4


@pytest.mark.parametrize(
    ("question_ids", "message"),
    [
        (["K05-01", "K05-01"], "Duplicate question IDs: K05-01"),
        (["K05-01", "NOT-A-FORMAL-ID"], "Unknown question IDs: NOT-A-FORMAL-ID"),
    ],
)
def test_question_ids_reject_duplicates_and_unknown_ids(question_ids, message):
    rubric = load_evaluation_dataset(FORMAL_DATASET_PATH)

    with pytest.raises(ValueError, match=message):
        select_questions(rubric, question_ids)


def test_four_question_three_mode_plan_resumes_without_duplicate_tasks(tmp_path):
    rubric = select_questions(load_evaluation_dataset(FORMAL_DATASET_PATH), PILOT_QUESTION_IDS)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((body["question"], body["mode"]))
        return httpx.Response(200, json=api_payload(body["mode"]))

    kwargs = dict(
        rubric=rubric,
        output_dir=tmp_path / "formal-four",
        run_id="formal-four",
        base_url="https://mock.example",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        semantic_judge=MockJudge(),
    )
    summary = EvaluationRunner(**kwargs).run()

    assert summary["overall"]["planned_count"] == 12
    assert len(calls) == 12
    assert {(question_id, mode) for question_id in PILOT_QUESTION_IDS for mode in ("none", "keyword", "vector")} == {
        (record["question_id"], record["requested_mode"])
        for record in map(json.loads, (tmp_path / "formal-four/responses.jsonl").read_text(encoding="utf-8").splitlines())
    }

    EvaluationRunner(**kwargs).run()
    assert len(calls) == 12
    assert len((tmp_path / "formal-four/responses.jsonl").read_text(encoding="utf-8").splitlines()) == 12
    assert len((tmp_path / "formal-four/scores.jsonl").read_text(encoding="utf-8").splitlines()) == 12


def test_filtered_run_refuses_mismatched_existing_run_without_overwriting_results(tmp_path):
    output_dir = tmp_path / "existing-first-unit-run"
    output_dir.mkdir()
    metadata_path = output_dir / "run_metadata.json"
    responses_path = output_dir / "responses.jsonl"
    scores_path = output_dir / "scores.jsonl"
    metadata_path.write_text('{"existing":"first-unit"}\n', encoding="utf-8")
    responses_path.write_text('{"existing":"response"}\n', encoding="utf-8")
    scores_path.write_text('{"existing":"score"}\n', encoding="utf-8")
    before = {path: path.read_bytes() for path in (metadata_path, responses_path, scores_path)}
    rubric = select_questions(load_evaluation_dataset(FORMAL_DATASET_PATH), PILOT_QUESTION_IDS)

    with pytest.raises(ValueError, match="Existing run metadata does not match this resume request"):
        EvaluationRunner(
            rubric=rubric,
            output_dir=output_dir,
            run_id="existing-first-unit-run",
            base_url="https://mock.example",
            client=httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("must not call API"))),
            semantic_judge=MockJudge(),
        ).run()

    assert {path: path.read_bytes() for path in before} == before


def test_20_question_three_mode_mock_integration_and_resume(tmp_path):
    rubric = load_rubric(RUBRIC_PATH)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        assert body["mode"] in {"none", "keyword", "vector"}
        calls.append((body["question"], body["mode"]))
        return httpx.Response(200, json=api_payload(body["mode"]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs = dict(
        rubric=rubric, output_dir=tmp_path / "mock-run", run_id="mock-run",
        base_url="https://mock.example", client=client, semantic_judge=MockJudge(),
    )
    summary = EvaluationRunner(**kwargs).run()

    assert len(calls) == 60
    assert summary["overall"]["planned_count"] == 60
    assert summary["overall"]["execution_success_count"] == 60
    assert summary["overall"]["needs_review_count"] == 54
    assert summary["overall"]["valid_final_decision_count"] == 6
    responses = (tmp_path / "mock-run/responses.jsonl").read_text(encoding="utf-8").splitlines()
    scores = (tmp_path / "mock-run/scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(responses) == len(scores) == 60
    assert len((tmp_path / "mock-run/review_queue.jsonl").read_text(encoding="utf-8").splitlines()) == 54
    assert all((tmp_path / "mock-run" / name).exists() for name in [
        "summary.json", "summary.md", "per_question.csv", "per_mode.csv", "run_metadata.json"
    ])

    EvaluationRunner(**kwargs).run()
    assert len(calls) == 60
    assert len((tmp_path / "mock-run/responses.jsonl").read_text(encoding="utf-8").splitlines()) == 60
    assert len((tmp_path / "mock-run/review_queue.jsonl").read_text(encoding="utf-8").splitlines()) == 54


def test_mode_mismatch_and_single_failure_do_not_stop_other_questions(tmp_path):
    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions = rubric.questions[:3]
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        if call_count == 1:
            return httpx.Response(503, json={"detail": "mock unavailable"})
        if call_count == 2:
            return httpx.Response(200, text="not-json")
        payload = api_payload(body["mode"])
        payload["executed_mode"] = "vector"
        return httpx.Response(200, json=payload)

    runner = EvaluationRunner(
        rubric=rubric, output_dir=tmp_path / "failures", run_id="failures",
        base_url="https://mock.example", modes=["keyword"],
        client=httpx.Client(transport=httpx.MockTransport(handler)), semantic_judge=MockJudge(),
    )
    summary = runner.run()
    statuses = [json.loads(line)["execution_status"] for line in (tmp_path / "failures/responses.jsonl").read_text(encoding="utf-8").splitlines()]

    assert call_count == 3
    assert statuses == ["http_error", "invalid_response", "mode_mismatch"]
    assert summary["overall"]["execution_success_count"] == 0
    assert (tmp_path / "failures/scores.jsonl").read_text(encoding="utf-8") == ""


def test_each_result_is_durable_before_next_request_and_failure_does_not_stop_batch(tmp_path):
    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions = [rubric.questions[2], rubric.questions[0], rubric.questions[1]]
    output_dir = tmp_path / "incremental"
    responses_path = output_dir / "responses.jsonl"
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        if calls == 2:
            persisted = responses_path.read_text(encoding="utf-8").splitlines()
            assert len(persisted) == 1
            assert json.loads(persisted[0])["execution_status"] == "success"
            assert len((output_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()) == 1
            assert len((output_dir / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()) == 1
            return httpx.Response(503, json={"detail": "mock unavailable"})
        if calls == 3:
            persisted = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
            assert [item["execution_status"] for item in persisted] == ["success", "http_error"]
        return httpx.Response(200, json=api_payload(body["mode"]))

    runner = EvaluationRunner(
        rubric=rubric, output_dir=output_dir, run_id="incremental",
        base_url="https://mock.example", modes=["none"],
        client=httpx.Client(transport=httpx.MockTransport(handler)), semantic_judge=MockJudge(),
    )
    summary = runner.run()
    responses = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
    scores = [json.loads(line) for line in (output_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    reviews = [json.loads(line) for line in (output_dir / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()]

    assert calls == 3
    assert [item["execution_status"] for item in responses] == ["success", "http_error", "success"]
    assert responses[1]["error"] == "mock unavailable"
    assert responses[1]["raw_api_response"] == {"detail": "mock unavailable"}
    assert all(item["run_id"] == "incremental" and item["requested_mode"] == "none" for item in responses)
    assert len(scores) == 2
    assert len(reviews) == sum(bool(item["needs_review"]) for item in scores)
    assert all(ScoreResult.model_validate(item) for item in scores)
    assert all(ReviewRecord.model_validate(item) for item in reviews)
    assert summary["overall"]["execution_success_count"] == 2
    assert summary["overall"]["execution_failed_count"] == 1


def test_resume_retries_failed_task_but_never_repeats_completed_task(tmp_path):
    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions = rubric.questions[:2]
    output_dir = tmp_path / "resume-after-failure"
    calls = []
    fail_second = True

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["question"])
        if fail_second and body["question"] == rubric.questions[1].question_text:
            return httpx.Response(503, json={"detail": "retry later"})
        return httpx.Response(200, json=api_payload(body["mode"]))

    kwargs = dict(
        rubric=rubric, output_dir=output_dir, run_id="resume-after-failure",
        base_url="https://mock.example", modes=["keyword"],
        client=httpx.Client(transport=httpx.MockTransport(handler)), semantic_judge=MockJudge(),
    )
    EvaluationRunner(**kwargs).run()
    assert calls == [question.question_text for question in rubric.questions]

    fail_second = False
    EvaluationRunner(**kwargs).run()
    assert calls == [
        rubric.questions[0].question_text,
        rubric.questions[1].question_text,
        rubric.questions[1].question_text,
    ]
    response_lines = [json.loads(line) for line in (output_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["execution_status"] for item in response_lines] == ["success", "http_error", "success"]

    EvaluationRunner(**kwargs).run()
    assert len(calls) == 3
    assert len((output_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_single_semantic_scoring_failure_is_persisted_and_batch_continues(tmp_path):
    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions = rubric.questions[:3]
    judged = []

    class IntermittentJudge(MockJudge):
        def judge(self, question, answer_text):
            judged.append(question.question_id)
            if question.question_id == rubric.questions[1].question_id:
                raise SemanticJudgeError(
                    "Invalid semantic judge response schema: major_error_candidates.rule_text missing"
                )
            return super().judge(question, answer_text)

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json=api_payload(body["mode"]))

    output_dir = tmp_path / "semantic-failure"
    summary = EvaluationRunner(
        rubric=rubric,
        output_dir=output_dir,
        run_id="semantic-failure",
        base_url="https://mock.example",
        modes=["none"],
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        semantic_judge=IntermittentJudge(),
    ).run()
    scores = [
        json.loads(line)
        for line in (output_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = scores[1]

    assert judged == [question.question_id for question in rubric.questions]
    assert [item["scoring_status"] for item in scores] == ["completed", "failed", "completed"]
    assert failed["question_id"] == rubric.questions[1].question_id
    assert failed["needs_review"] is True
    assert "major_error_candidates.rule_text missing" in failed["review_reason"][0]
    assert summary["overall"]["execution_success_count"] == 3
    assert len((output_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_timeout_is_recorded_separately(tmp_path):
    rubric = load_rubric(RUBRIC_PATH).model_copy(deep=True)
    rubric.questions = rubric.questions[:1]

    def handler(request):
        raise httpx.ReadTimeout("mock timeout", request=request)

    runner = EvaluationRunner(
        rubric=rubric, output_dir=tmp_path / "timeout", run_id="timeout",
        base_url="https://mock.example", modes=["none"], retries=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)), semantic_judge=MockJudge(),
    )
    summary = runner.run()
    record = json.loads((tmp_path / "timeout/responses.jsonl").read_text(encoding="utf-8"))

    assert record["execution_status"] == "timeout"
    assert record["retry_count"] == 1
    assert summary["overall"]["execution_failed_count"] == 1


def test_statistics_exclude_failures_and_reviews_from_correctness_denominator():
    metadata = {"run_id": "r", "rubric_version": "v", "modes": ["none"], "question_count": 4, "planned_count": 4}
    responses = [
        {"requested_mode": "none", "execution_status": "success"},
        {"requested_mode": "none", "execution_status": "success"},
        {"requested_mode": "none", "execution_status": "success"},
        {"requested_mode": "none", "execution_status": "http_error"},
    ]
    scores = [
        {"requested_mode": "none", "scoring_status": "completed", "final_correctness": "correct", "needs_review": False, "completeness_score": 1},
        {"requested_mode": "none", "scoring_status": "completed", "final_correctness": "incorrect", "needs_review": False, "completeness_score": 0.2},
        {"requested_mode": "none", "scoring_status": "completed", "final_correctness": None, "needs_review": True, "completeness_score": None},
    ]

    metrics = build_summary(responses, scores, metadata)["overall"]

    assert metrics["execution_failed_count"] == 1
    assert metrics["valid_final_decision_count"] == 2
    assert metrics["correct_rate"] == 0.5
    assert metrics["incorrect_rate"] == 0.5
    assert metrics["needs_review_count"] == 1
    assert metrics["not_finalized_count"] == 1

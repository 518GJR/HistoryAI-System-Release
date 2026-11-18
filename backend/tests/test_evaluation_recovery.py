import hashlib
import json
import shutil
from pathlib import Path

import pytest

from app.services.evaluation.models import CriterionJudgment, FactStatus, ReviewRecord
from app.services.evaluation.rubric_loader import load_rubric
from app.services.evaluation.semantic_judge import SemanticAssessment, SemanticJudgeError
from scripts.recover_failed_scores import ScoreRecoveryRunner, load_recovery_targets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN_DIR = (
    PROJECT_ROOT
    / "docs/evaluation/workbench/results/unit1-v1.0-real-pilot-20261026-05"
)
RUBRIC_PATH = PROJECT_ROOT / "docs/evaluation/workbench/unit1_v1.0_rubric.json"
EXPECTED_FAILED_IDS = {
    "unit1-v1.0-real-pilot-20261026-05:Q005:keyword",
    "unit1-v1.0-real-pilot-20261026-05:Q007:none",
    "unit1-v1.0-real-pilot-20261026-05:Q007:keyword",
    "unit1-v1.0-real-pilot-20261026-05:Q008:none",
    "unit1-v1.0-real-pilot-20261026-05:Q008:keyword",
    "unit1-v1.0-real-pilot-20261026-05:Q011:none",
    "unit1-v1.0-real-pilot-20261026-05:Q016:none",
    "unit1-v1.0-real-pilot-20261026-05:Q016:keyword",
    "unit1-v1.0-real-pilot-20261026-05:Q018:vector",
}
PROTECTED_FILES = (
    "run_metadata.json",
    "responses.jsonl",
    "scores.jsonl",
    "review_queue.jsonl",
    "summary.json",
)


def file_hashes():
    return {
        name: hashlib.sha256((SOURCE_RUN_DIR / name).read_bytes()).hexdigest()
        for name in PROTECTED_FILES
    }


class MockRecoveryJudge:
    prompt_version = "history-fact-judge-v3"
    config_id = "mock-recovery-config"
    config = {"model": "mock-deepseek-v2"}

    def __init__(self, fail_question_id=None, interrupt_after=None):
        self.fail_question_id = fail_question_id
        self.interrupt_after = interrupt_after
        self.calls = []

    def judge(self, question, answer_text):
        self.calls.append((question.question_id, answer_text))
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt("simulated interruption")
        if question.question_id == self.fail_question_id:
            raise SemanticJudgeError(
                "Invalid semantic judge response schema: simulated V2 structure error"
            )
        judgments = [
            CriterionJudgment(
                criterion_id=criterion.criterion_id,
                status=FactStatus.MISSING,
                rationale="The saved answer does not state this criterion in the mock.",
            )
            for criterion in question.criteria
        ]
        if self.calls and len(self.calls) == 1:
            judgments[0].contradiction_detected = True
            judgments[0].major_error_candidate = True
        assessment = SemanticAssessment(
            criterion_results=judgments,
            directly_addresses_question=True,
            has_verifiable_fact=True,
        )
        return assessment, assessment.model_dump(mode="json")


def make_runner(
    tmp_path,
    judge,
    recovery_run_id="mock-score-recovery",
    *,
    record_ids=None,
    expected_failed_count=9,
):
    return ScoreRecoveryRunner(
        rubric=load_rubric(RUBRIC_PATH),
        source_run_dir=SOURCE_RUN_DIR,
        output_dir=tmp_path / "recovery",
        recovery_run_id=recovery_run_id,
        semantic_judge=judge,
        expected_failed_count=expected_failed_count,
        record_ids=record_ids,
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_exactly_selects_nine_failed_scores_and_preserves_source_linkage():
    rubric = load_rubric(RUBRIC_PATH)
    metadata, targets = load_recovery_targets(SOURCE_RUN_DIR, rubric)

    assert metadata["run_id"] == "unit1-v1.0-real-pilot-20261026-05"
    assert len(targets) == 9
    assert {target.source_record_id for target in targets} == EXPECTED_FAILED_IDS
    for target in targets:
        assert target.failed_score["scoring_status"] == "failed"
        assert target.response["execution_status"] == "success"
        assert target.response["question_id"] == target.question.question_id
        assert target.response["requested_mode"] == target.failed_score["requested_mode"]
        assert target.response["executed_mode"] == target.failed_score["executed_mode"]
        assert isinstance(target.response["answer_text"], str)


def test_selects_only_requested_failed_records_in_requested_order():
    _, targets = load_recovery_targets(
        SOURCE_RUN_DIR,
        load_rubric(RUBRIC_PATH),
        expected_failed_count=2,
        record_ids=["Q011:none", "Q016:none"],
    )

    assert [
        (target.question.question_id, target.response["requested_mode"])
        for target in targets
    ] == [("Q011", "none"), ("Q016", "none")]
    assert all(target.failed_score["scoring_status"] == "failed" for target in targets)
    assert [target.source_record_id for target in targets] == [
        "unit1-v1.0-real-pilot-20261026-05:Q011:none",
        "unit1-v1.0-real-pilot-20261026-05:Q016:none",
    ]


@pytest.mark.parametrize(
    ("record_ids", "message"),
    [
        (["Q011"], "expected QUESTION_ID:MODE"),
        (["Q011:invalid"], "Invalid recovery mode"),
        (["Q999:none"], "Unknown recovery question ID"),
        (["Q011:none", "Q011:none"], "Duplicate recovery record ID"),
    ],
)
def test_rejects_invalid_record_id_selection(record_ids, message):
    with pytest.raises(ValueError, match=message):
        load_recovery_targets(
            SOURCE_RUN_DIR,
            load_rubric(RUBRIC_PATH),
            expected_failed_count=len(record_ids),
            record_ids=record_ids,
        )


def test_rejects_nonexistent_question_mode_combination(tmp_path):
    source_copy = tmp_path / "source"
    source_copy.mkdir()
    shutil.copy2(SOURCE_RUN_DIR / "run_metadata.json", source_copy / "run_metadata.json")
    for name in ("responses.jsonl", "scores.jsonl"):
        records = [
            item
            for item in read_jsonl(SOURCE_RUN_DIR / name)
            if (item["question_id"], item["requested_mode"]) != ("Q001", "vector")
        ]
        (source_copy / name).write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="does not exist in source run: Q001:vector"):
        load_recovery_targets(
            source_copy,
            load_rubric(RUBRIC_PATH),
            expected_failed_count=1,
            record_ids=["Q001:vector"],
        )


def test_rejects_completed_score_selection():
    with pytest.raises(ValueError, match="not a failed score.*Q001:none"):
        load_recovery_targets(
            SOURCE_RUN_DIR,
            load_rubric(RUBRIC_PATH),
            expected_failed_count=1,
            record_ids=["Q001:none"],
        )


def test_mock_recovery_persists_independent_results_and_keeps_source_immutable(tmp_path):
    before = file_hashes()
    judge = MockRecoveryJudge()
    runner = make_runner(tmp_path, judge)
    summary = runner.run()
    scores = read_jsonl(runner.scores_path)
    reviews = read_jsonl(runner.review_queue_path)

    assert len(judge.calls) == 9
    assert len(scores) == 9
    assert {item["recovered_from_record_id"] for item in scores} == EXPECTED_FAILED_IDS
    assert all(item["recovery_run_id"] == "mock-score-recovery" for item in scores)
    assert all(item["source_run_id"] == "unit1-v1.0-real-pilot-20261026-05" for item in scores)
    assert all(item["prompt_version"] == "history-fact-judge-v3" for item in scores)
    assert all(item["scorer_config_id"] == "mock-recovery-config" for item in scores)
    assert all(item["scoring_model"] == "mock-deepseek-v2" for item in scores)
    assert all(item["question_text"] and isinstance(item["answer_text"], str) for item in scores)
    assert len(reviews) == sum(bool(item["needs_review"]) for item in scores)
    assert all(ReviewRecord.model_validate(item) for item in reviews)
    first_raw = scores[0]["raw_semantic_output"]["criterion_results"][0]
    assert first_raw["contradiction_detected"] is True
    assert first_raw["major_error_candidate"] is True
    assert scores[0]["needs_review"] is True
    assert summary["target_count"] == summary["processed_count"] == 9
    assert summary["remaining_count"] == 0
    assert summary["scoring_completed_count"] == 9
    assert summary["scoring_failed_count"] == 0
    assert "human gold standard" in summary["accuracy_note"]
    assert summary["source_file_sha256"] == {
        name: digest for name, digest in before.items() if name in PROTECTED_FILES[:3]
    }
    assert file_hashes() == before


def test_resume_after_interruption_skips_durable_results(tmp_path):
    interrupted = MockRecoveryJudge(interrupt_after=2)
    runner = make_runner(tmp_path, interrupted)

    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        runner.run()
    persisted = read_jsonl(runner.scores_path)
    assert len(persisted) == 2

    resumed = MockRecoveryJudge()
    summary = make_runner(tmp_path, resumed).run()
    assert len(resumed.calls) == 7
    assert len(read_jsonl(runner.scores_path)) == 9
    assert summary["processed_count"] == 9

    no_repeat = MockRecoveryJudge()
    make_runner(tmp_path, no_repeat).run()
    assert no_repeat.calls == []
    assert len(read_jsonl(runner.scores_path)) == 9


def test_filtered_resume_after_interruption_does_not_repeat_completed_target(tmp_path):
    selected = ["Q011:none", "Q016:none"]
    interrupted = MockRecoveryJudge(interrupt_after=1)
    runner = make_runner(
        tmp_path,
        interrupted,
        record_ids=selected,
        expected_failed_count=2,
    )

    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        runner.run()
    persisted = read_jsonl(runner.scores_path)
    assert [(item["question_id"], item["requested_mode"]) for item in persisted] == [
        ("Q011", "none")
    ]

    resumed = MockRecoveryJudge()
    summary = make_runner(
        tmp_path,
        resumed,
        record_ids=selected,
        expected_failed_count=2,
    ).run()

    assert [question_id for question_id, _ in resumed.calls] == ["Q016"]
    assert summary["target_count"] == summary["processed_count"] == 2
    scores = read_jsonl(runner.scores_path)
    assert len(scores) == 2
    assert {item["recovered_from_record_id"] for item in scores} == {
        "unit1-v1.0-real-pilot-20261026-05:Q011:none",
        "unit1-v1.0-real-pilot-20261026-05:Q016:none",
    }


def test_filtered_recovery_refuses_first_recovery_output_and_run_id(tmp_path):
    source_name = SOURCE_RUN_DIR.name
    previous_output = SOURCE_RUN_DIR.parent / f"{source_name}-score-recovery-v2"
    runner = ScoreRecoveryRunner(
        rubric=load_rubric(RUBRIC_PATH),
        source_run_dir=SOURCE_RUN_DIR,
        output_dir=previous_output,
        recovery_run_id="safe-new-run-id",
        semantic_judge=MockRecoveryJudge(),
        expected_failed_count=2,
        record_ids=["Q011:none", "Q016:none"],
    )
    with pytest.raises(ValueError, match="new output directory"):
        runner.run()

    runner = make_runner(
        tmp_path,
        MockRecoveryJudge(),
        recovery_run_id="unit1-v1.0-real-pilot-20261026-05-score-recovery-v2",
        record_ids=["Q011:none", "Q016:none"],
        expected_failed_count=2,
    )
    with pytest.raises(ValueError, match="new recovery run ID"):
        runner.run()


def test_single_recovery_failure_is_recorded_and_remaining_targets_continue(tmp_path):
    judge = MockRecoveryJudge(fail_question_id="Q011")
    runner = make_runner(tmp_path, judge)
    summary = runner.run()
    scores = read_jsonl(runner.scores_path)
    failed = [item for item in scores if item["scoring_status"] == "failed"]

    assert len(judge.calls) == 9
    assert len(scores) == 9
    assert len(failed) == 1
    assert failed[0]["question_id"] == "Q011"
    assert failed[0]["needs_review"] is True
    assert "simulated V2 structure error" in failed[0]["review_reason"][0]
    assert failed[0]["answer_text"]
    assert summary["scoring_completed_count"] == 8
    assert summary["scoring_failed_count"] == 1

    second_run = MockRecoveryJudge()
    make_runner(tmp_path, second_run).run()
    assert second_run.calls == []


def test_completed_needs_review_scores_are_not_selected():
    _, targets = load_recovery_targets(
        SOURCE_RUN_DIR,
        load_rubric(RUBRIC_PATH),
    )
    source_scores = read_jsonl(SOURCE_RUN_DIR / "scores.jsonl")
    completed_review_ids = {
        item["record_id"]
        for item in source_scores
        if item["scoring_status"] == "completed" and item["needs_review"]
    }

    assert len(completed_review_ids) == 45
    assert completed_review_ids.isdisjoint(
        {target.source_record_id for target in targets}
    )

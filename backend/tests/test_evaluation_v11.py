from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.services.evaluation.adjudication_v11 import (
    AdjudicationError,
    REVIEW_FIELDS,
    export_review_template,
    import_adjudications,
    read_jsonl,
)
from app.services.evaluation.statistics_v11 import StatisticsError, build_statistics
from app.services.evaluation.v11_rules import (
    V11RuleError,
    load_v11_rules,
    score_confirmed_rule,
    score_confirmed_d_rule,
)
from scripts.build_unit1_v11_scores import build_offline_replay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = PROJECT_ROOT / "docs/evaluation/workbench/unit1_v1.1/scoring_rules.json"
RUBRIC_PATH = PROJECT_ROOT / "docs/evaluation/workbench/unit1_v1.0_rubric.json"
REAL_RUN = PROJECT_ROOT / "docs/evaluation/workbench/results/unit1-v1.0-real-pilot-20261026-05"
RECOVERY_V2 = PROJECT_ROOT / "docs/evaluation/workbench/results/unit1-v1.0-real-pilot-20261026-05-score-recovery-v2/scores.jsonl"
RECOVERY_V3 = PROJECT_ROOT / "docs/evaluation/workbench/results/unit1-v1.0-real-pilot-20261026-05-score-recovery-v3-final2/scores.jsonl"


@pytest.fixture(scope="module")
def rules():
    return load_v11_rules(RULES_PATH, RUBRIC_PATH)


def _judgment(status: str, evidence: str = "可核验事实") -> dict:
    return {
        "status": status,
        "earned_score": None,
        "needs_review": True,
        "review_reason": "class D rule remains blocked_pending_adjudication",
        "evidence_quotes": [] if status in {"missing", "uncertain"} else [evidence],
        "rationale": "mock semantic rationale",
    }


def test_v11_config_has_exact_confirmed_and_manual_partition(rules):
    assert rules["confirmed_rule_count"] == 70
    assert rules["confirmed_rule_counts"] == {"A": 9, "B": 8, "C": 37, "D": 16}
    assert len(rules["confirmed_non_d_rules"]) == 54
    assert len(rules["confirmed_rules"]) == 16
    assert len(rules["manual_rules"]) == 22
    assert len(rules["preserved_existing_a_rules"]) == 15
    assert {item["criterion_id"] for item in rules["deferred_a_rules"]} == {"Q014-C06", "Q015-C05"}
    assert rules["provenance"]["formal_signoff_gap"] is False
    assert len(rules["provenance"]["signoff_files"]) == 4
    assert rules["global_rule_signoffs"] == {
        "G01": "pending_not_signed_off",
        "G02": "pending_not_signed_off",
        "G03": "user_confirmed_v1.1",
        "G04": "user_confirmed_v1.1",
        "G05": "pending_not_signed_off",
    }
    assert {item["criterion_id"] for item in rules["confirmed_rules"]}.isdisjoint(
        {item["criterion_id"] for item in rules["manual_rules"]}
    )


@pytest.mark.parametrize("criterion_index", range(16))
def test_each_confirmed_d_rule_has_full_zero_and_review_boundaries(rules, criterion_index):
    rule = rules["confirmed_rules"][criterion_index]
    full = score_confirmed_d_rule(rule, _judgment("full"), answer_text="正常上下文", precision=2)
    zero = score_confirmed_d_rule(rule, _judgment("missing"), answer_text="", precision=2)
    review = score_confirmed_d_rule(rule, _judgment("uncertain"), answer_text="", precision=2)
    assert full["score"] == rule["weight"]
    assert full["needs_review"] is False
    assert zero["score"] == 0
    assert zero["needs_review"] is False
    assert review["score"] is None
    assert review["needs_review"] is True


def test_q014_c05_missing_either_required_action_is_strict_zero(rules):
    rule = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q014-C05")
    result = score_confirmed_d_rule(rule, _judgment("partial"), answer_text="仅写对外战争并形成国家", precision=2)
    assert result["score"] == 0
    assert result["needs_review"] is False
    assert "strict rule" in result["reason"]


def test_q014_c05_both_actions_without_state_conclusion_scores_half(rules):
    rule = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q014-C05")
    result = score_confirmed_d_rule(
        rule,
        _judgment("partial", "统治阶级对内统治并对外战争"),
        answer_text="统治阶级对内统治并对外战争",
        precision=2,
    )
    assert result["score"] == 0.75
    assert result["needs_review"] is False


def test_q017_c06_special_example_scores_one_and_does_not_generalize(rules):
    rule = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q017-C06")
    exact = score_confirmed_d_rule(
        rule,
        _judgment("partial", "都是早期文明的重要遗址。"),
        answer_text="比较回答",
        precision=2,
    )
    nearby = score_confirmed_d_rule(
        rule,
        _judgment("partial", "都是重要文明遗址"),
        answer_text="比较回答",
        precision=2,
    )
    assert exact["score"] == 1.0
    assert exact["decision"] == "special_partial"
    assert nearby["score"] is None
    assert nearby["needs_review"] is True


def test_q019_c06_only_human_ancestor_phrase_requires_review(rules):
    rule = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q019-C06")
    result = score_confirmed_d_rule(
        rule,
        _judgment("partial", "炎黄二帝是人文初祖。"),
        answer_text="炎黄二帝是人文初祖。",
        precision=2,
    )
    assert result["score"] is None
    assert result["needs_review"] is True


def test_q018_c06_center_phrase_requires_context_review_even_if_model_says_full(rules):
    rule = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q018-C06")
    result = score_confirmed_d_rule(
        rule,
        _judgment("full", "以中原为中心，走向一体"),
        answer_text="以中原为中心，走向一体",
        precision=2,
    )
    assert result["score"] is None
    assert "contextual human review" in result["reason"]


def test_q009_c03_partial_remains_review_and_q013_c05_supports_1_25(rules):
    q009 = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q009-C03")
    q013 = next(item for item in rules["confirmed_rules"] if item["criterion_id"] == "Q013-C05")
    pending = score_confirmed_d_rule(q009, _judgment("partial"), answer_text="", precision=2)
    precise = score_confirmed_d_rule(q013, _judgment("partial"), answer_text="", precision=2)
    assert pending["score"] is None and pending["needs_review"]
    assert precise["score"] == 1.25 and not precise["needs_review"]


def test_signed_abc_rules_apply_only_their_confirmed_partial_boundaries(rules):
    rules_by_id = {item["criterion_id"]: item for item in rules["confirmed_non_d_rules"]}
    split = score_confirmed_rule(
        rules_by_id["Q015-C01"], _judgment("partial"), answer_text="古城结构", precision=2
    )
    whole = score_confirmed_rule(
        rules_by_id["Q005-C06"], _judgment("partial"), answer_text="只覆盖一端", precision=2
    )
    component = score_confirmed_rule(
        rules_by_id["Q016-C03"], _judgment("partial"), answer_text="小型墓无随葬品", precision=2
    )
    c_half = score_confirmed_rule(
        rules_by_id["Q005-C03"], _judgment("partial"), answer_text="大量打制石器", precision=2
    )
    assert split["score"] == 0.75 and not split["needs_review"]
    assert whole["score"] is None and whole["needs_review"]
    assert component["score"] is None and component["needs_review"]
    assert c_half["score"] == 0.5 and not c_half["needs_review"]


def test_q016_c06_and_other_22_candidates_are_never_auto_approved(rules):
    manual = {item["criterion_id"] for item in rules["manual_rules"]}
    assert "Q016-C06" in manual
    assert len(manual) == 22
    assert all(item.get("approval_status") is None for item in rules["manual_rules"])


def _pending_result() -> dict:
    return {
        "run_id": "run-v11",
        "record_id": "source:Q001:none",
        "question_id": "Q001",
        "retrieval_mode": "none",
        "answer_text": "原始回答不可修改",
        "rule_version": "unit1_v1.1",
        "criteria": [
            {
                "criterion_id": "Q001-C01",
                "max_score": 2.0,
                "automatic": {"decision": "pending", "reason": "unsigned", "score": None},
                "final": {
                    "score": None,
                    "decision": "pending",
                    "source": "pending",
                    "reason": "unsigned",
                    "automatic_preserved": True,
                    "difference_from_automatic": None,
                },
            },
            {
                "criterion_id": "Q001-C02",
                "max_score": 8.0,
                "automatic": {"decision": "full", "reason": "automatic", "score": 8.0},
                "final": {
                    "score": 8.0,
                    "decision": "full",
                    "source": "automatic",
                    "reason": "automatic",
                    "automatic_preserved": True,
                    "difference_from_automatic": None,
                },
            },
        ],
        "automatic_score": None,
        "automatic_label": None,
        "final_score": None,
        "final_label": None,
        "scoring_completed": False,
        "needs_review": True,
        "major_error_confirmed": False,
        "major_error_confirmations": [],
        "human_confirmed_final": False,
    }


def _edit_csv(path: Path, editor) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    editor(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_adjudication_import_preserves_automatic_and_records_override(tmp_path, rules):
    source = [_pending_result()]
    template = tmp_path / "review.csv"
    assert export_review_template(source, template, include_all=True) == 2

    def approve(rows):
        rows[0].update({
            "human_score": "1.25",
            "human_reason": "教材证据支持部分得分",
            "textbook_citation": "教材第1页",
            "major_error_confirmation": "no",
            "reviewer": "reviewer-a",
            "second_reviewer": "reviewer-b",
            "adjudication_status": "overridden",
        })
        rows[1].update({
            "human_score": "7.5",
            "human_reason": "人工复核后扣0.5分",
            "textbook_citation": "教材第1页",
            "major_error_confirmation": "no",
            "reviewer": "reviewer-a",
            "second_reviewer": "reviewer-b",
            "adjudication_status": "overridden",
        })

    _edit_csv(template, approve)
    merged = import_adjudications(source, template, rules)[0]
    assert merged["answer_text"] == "原始回答不可修改"
    assert merged["criteria"][0]["automatic"]["score"] is None
    assert merged["criteria"][0]["final"]["score"] == 1.25
    assert merged["criteria"][0]["final"]["automatic_preserved"] is True
    assert merged["criteria"][1]["automatic"]["score"] == 8.0
    assert merged["criteria"][1]["final"]["difference_from_automatic"] == -0.5
    assert merged["final_score"] == 8.75
    assert merged["human_confirmed_final"] is True


def test_pending_adjudication_is_not_filled_with_zero(tmp_path, rules):
    source = [_pending_result()]
    template = tmp_path / "review.csv"
    export_review_template(source, template)
    merged = import_adjudications(source, template, rules)[0]
    assert merged["criteria"][0]["final"]["score"] is None
    assert merged["final_score"] is None
    assert merged["human_confirmed_final"] is False


@pytest.mark.parametrize("case", ["duplicate", "out_of_range", "precision", "version"])
def test_adjudication_rejects_duplicate_abnormal_score_and_version(tmp_path, rules, case):
    source = [_pending_result()]
    template = tmp_path / "review.csv"
    export_review_template(source, template)

    def corrupt(rows):
        row = rows[0]
        row.update({
            "human_score": "1",
            "human_reason": "reason",
            "major_error_confirmation": "no",
            "reviewer": "reviewer",
            "adjudication_status": "overridden",
        })
        if case == "duplicate":
            rows.append(deepcopy(row))
        elif case == "out_of_range":
            row["human_score"] = "2.01"
        elif case == "precision":
            row["human_score"] = "1.251"
        elif case == "version":
            row["rule_version"] = "unit1_v1.0"

    _edit_csv(template, corrupt)
    with pytest.raises(AdjudicationError):
        import_adjudications(source, template, rules)


def _sixty_inputs(rule_version: str = "unit1_v1.1"):
    responses = []
    scores = []
    for mode in ("none", "keyword", "vector"):
        for index in range(1, 21):
            question_id = f"Q{index:03d}"
            record_id = f"source:{question_id}:{mode}"
            responses.append({
                "record_id": record_id,
                "question_id": question_id,
                "requested_mode": mode,
                "execution_status": "success",
            })
            scores.append({
                "record_id": record_id,
                "question_id": question_id,
                "retrieval_mode": mode,
                "rule_version": rule_version,
                "scoring_completed": False,
                "needs_review": True,
                "human_confirmed_final": False,
                "automatic_score": None,
                "automatic_label": None,
                "final_score": None,
                "final_label": None,
                "major_error_confirmed": False,
            })
    return responses, scores


def test_statistics_do_not_publish_fake_accuracy_before_human_completion():
    responses, scores = _sixty_inputs()
    report = build_statistics(responses, scores)
    for mode in ("none", "keyword", "vector"):
        item = report["per_mode"][mode]
        assert item["total_answer_count"] == 20
        assert item["human_valid_final_count"] == 0
        assert item["answer_correct_rate"] is None
        assert item["planned_completion_coverage_rate"] == 0
        assert item["formal_20_question_result"] is False
        assert "0/20" in item["formal_result_note"]
        assert "不得称为完整实验结果" in item["formal_result_note"]


def test_statistics_block_rule_version_conflict():
    responses, scores = _sixty_inputs()
    scores[0]["rule_version"] = "unit1_v1.0"
    with pytest.raises(StatisticsError, match="Rule version conflict"):
        build_statistics(responses, scores)


@pytest.mark.skipif(not all(p.exists() for p in (REAL_RUN / "responses.jsonl", REAL_RUN / "scores.jsonl", RECOVERY_V2, RECOVERY_V3)), reason="Optional local experiment outputs are excluded from the public release")
def test_real_60_answer_offline_replay_without_external_api(tmp_path, rules):
    report = build_offline_replay(
        responses_path=REAL_RUN / "responses.jsonl",
        base_scores_path=REAL_RUN / "scores.jsonl",
        recovery_scores_paths=[RECOVERY_V2, RECOVERY_V3],
        rubric_path=RUBRIC_PATH,
        rules_path=RULES_PATH,
        output_dir=tmp_path / "v11",
        replay_run_id="pytest-unit1-v1.1",
    )
    scores = read_jsonl(tmp_path / "v11/unified_scores.jsonl")
    assert report["external_model_calls"] == 0
    assert report["response_count"] == report["unified_score_count"] == 60
    assert report["per_mode_count"] == {"none": 20, "keyword": 20, "vector": 20}
    assert len({(item["question_id"], item["retrieval_mode"]) for item in scores}) == 60
    assert report["human_confirmed_final_count"] == 0
    assert report["formal_accuracy_available"] is False
    assert report["encoded_confirmed_rule_count"] == 70
    assert report["scoring_completed_answer_count"] > 6
    assert report["pending_answer_count"] < 54
    assert report["pending_criterion_instance_count"] < 128
    assert all(item["rule_version"] == "unit1_v1.1" for item in scores)
    assert all(item["source_automatic_result_preserved"] for item in scores)
    manual_ids = {rule["criterion_id"] for rule in rules["manual_rules"]}
    deferred_ids = {rule["criterion_id"] for rule in rules["deferred_a_rules"]}
    for item in scores:
        for criterion in item["criteria"]:
            if criterion["criterion_id"] in manual_ids | deferred_ids:
                assert criterion["automatic"]["needs_review"] is True


def test_signoff_hash_change_is_rejected(tmp_path):
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    raw["provenance"]["signoff_files"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad-signoff-hash.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(V11RuleError, match="Signoff hash mismatch"):
        load_v11_rules(path, RUBRIC_PATH)


def test_rule_loader_rejects_version_conflict(tmp_path):
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    raw["base_rubric_version"] = "unit1_v0"
    path = tmp_path / "bad-rules.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(V11RuleError, match="base_rubric_version"):
        load_v11_rules(path, RUBRIC_PATH)

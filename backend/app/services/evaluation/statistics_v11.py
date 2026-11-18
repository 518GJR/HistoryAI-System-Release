"""Formal, denominator-safe statistics for Unit 1 v1.1."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class StatisticsError(ValueError):
    """Input coverage or version mismatch would make statistics misleading."""


MODES = ("none", "keyword", "vector")


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def validate_experiment_inputs(
    responses: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    rule_version: str,
    planned_per_mode: int = 20,
) -> None:
    response_keys = [(item.get("question_id"), item.get("requested_mode")) for item in responses]
    if len(response_keys) != len(set(response_keys)):
        raise StatisticsError("Duplicate response question/mode identity")
    score_ids = [item.get("record_id") for item in scores]
    if len(score_ids) != len(set(score_ids)):
        raise StatisticsError("Duplicate score record_id")
    for mode in MODES:
        mode_responses = [item for item in responses if item.get("requested_mode") == mode]
        question_ids = {item.get("question_id") for item in mode_responses}
        if len(mode_responses) != planned_per_mode or len(question_ids) != planned_per_mode:
            raise StatisticsError(
                f"Expected {planned_per_mode} unique responses for mode={mode}, "
                f"found rows={len(mode_responses)}, questions={len(question_ids)}"
            )
    if len(responses) != planned_per_mode * len(MODES):
        raise StatisticsError(f"Expected 60 responses, found {len(responses)}")
    if len(scores) != len(responses):
        raise StatisticsError(f"Expected one unified score per response: {len(scores)} != {len(responses)}")
    versions = {item.get("rule_version") for item in scores}
    if versions != {rule_version}:
        raise StatisticsError(f"Rule version conflict: expected={rule_version}, actual={sorted(versions)}")
    score_keys = {(item.get("question_id"), item.get("retrieval_mode")) for item in scores}
    if score_keys != set(response_keys):
        raise StatisticsError("Response/score question-mode coverage mismatch")


def _mode_metrics(
    responses: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    planned_count: int,
) -> dict[str, Any]:
    successful = [item for item in responses if item.get("execution_status") == "success"]
    scoring_complete = [item for item in scores if item.get("scoring_completed")]
    pending = [item for item in scores if item.get("needs_review")]
    human_final = [item for item in scores if item.get("human_confirmed_final")]
    correct = [item for item in human_final if item.get("final_label") == "correct"]
    partial = [item for item in human_final if item.get("final_label") == "partial"]
    incorrect = [item for item in human_final if item.get("final_label") == "incorrect"]
    automatic_final = [
        item for item in scores
        if item.get("automatic_score") is not None
        and item.get("automatic_label") in {"correct", "partial", "incorrect"}
    ]
    agreement_eligible = [
        item for item in human_final
        if item.get("automatic_label") in {"correct", "partial", "incorrect"}
    ]
    agreement = [item for item in agreement_eligible if item["automatic_label"] == item["final_label"]]
    human_score_total = sum(float(item["final_score"]) for item in human_final)
    auto_score_total = sum(float(item["automatic_score"]) for item in automatic_final)
    metrics = {
        "total_answer_count": len(responses),
        "qa_success_count": len(successful),
        "scoring_completed_count": len(scoring_complete),
        "pending_human_review_count": len(pending),
        "human_valid_final_count": len(human_final),
        "correct_count": len(correct),
        "partial_count": len(partial),
        "incorrect_count": len(incorrect),
        "answer_correct_rate": _ratio(len(correct), len(human_final)),
        "answer_correct_rate_denominator": len(human_final),
        "planned_completion_coverage_rate": _ratio(len(human_final), planned_count),
        "planned_count": planned_count,
        "mean_completeness_score_rate": _ratio(human_score_total, len(human_final) * 10),
        "mean_completeness_denominator": len(human_final),
        "major_factual_error_rate": _ratio(
            sum(bool(item.get("major_error_confirmed")) for item in human_final),
            len(human_final),
        ),
        "major_factual_error_denominator": len(human_final),
        "automatic_human_agreement_rate": _ratio(len(agreement), len(agreement_eligible)),
        "automatic_human_agreement_denominator": len(agreement_eligible),
        "automatic_valid_estimate_count": len(automatic_final),
        "automatic_correct_estimate_count": sum(item["automatic_label"] == "correct" for item in automatic_final),
        "automatic_correct_estimate_rate": _ratio(
            sum(item["automatic_label"] == "correct" for item in automatic_final),
            len(automatic_final),
        ),
        "automatic_mean_completeness_score_rate": _ratio(
            auto_score_total,
            len(automatic_final) * 10,
        ),
        "formal_20_question_result": len(human_final) == planned_count,
    }
    metrics["formal_result_note"] = (
        f"{planned_count}条预定样本的人工终判完整，可报告正式结果。"
        if metrics["formal_20_question_result"]
        else f"仅有{len(human_final)}/{planned_count}条有效人工终判，pending={len(pending)}；不得称为完整实验结果。"
    )
    return metrics


def build_statistics(
    responses: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    rule_version: str = "unit1_v1.1",
    planned_per_mode: int = 20,
) -> dict[str, Any]:
    validate_experiment_inputs(
        responses,
        scores,
        rule_version=rule_version,
        planned_per_mode=planned_per_mode,
    )
    response_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in responses:
        response_groups[item["requested_mode"]].append(item)
    for item in scores:
        score_groups[item["retrieval_mode"]].append(item)
    return {
        "schema_version": "historyai-unit1-v1.1-statistics-v1",
        "rule_version": rule_version,
        "metric_scope": {
            "formal_metrics": "human_confirmed_final only",
            "automatic_metrics": "automatic terminal estimates only",
            "correct_rate_formula": "human confirmed correct / valid human final",
            "completeness_formula": "sum valid final score / (valid scored answers * 10)",
        },
        "per_mode": {
            mode: _mode_metrics(response_groups[mode], score_groups[mode], planned_per_mode)
            for mode in MODES
        },
        "overall": _mode_metrics(responses, scores, planned_per_mode * len(MODES)),
    }


def write_statistics_reports(output_dir: Path, report: dict[str, Any], *, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "statistics.json",
        "csv": output_dir / "statistics.csv",
        "md": output_dir / "statistics.md",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite statistics outputs: {existing}")
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [{"mode": mode, **values} for mode, values in report["per_mode"].items()]
    fields = list(rows[0])
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 第一单元 v1.1 评分统计",
        "",
        f"规则版本：`{report['rule_version']}`",
        "",
        "正式指标只使用有效人工终判；自动估计单独列示。",
        "",
        "| 模式 | 回答 | 问答成功 | 评分完成 | Pending | 人工终判 | 正确/部分/错误 | 正确率 | 完成覆盖率 | 平均完整性 | 重大错误率 | 自动-人工一致率 | 自动有效估计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = report["per_mode"][mode]
        rate = "N/A" if item["answer_correct_rate"] is None else f"{item['answer_correct_rate']:.2%} (n={item['answer_correct_rate_denominator']})"
        coverage = f"{item['planned_completion_coverage_rate']:.2%}"
        completeness = "N/A" if item["mean_completeness_score_rate"] is None else f"{item['mean_completeness_score_rate']:.2%}"
        major = "N/A" if item["major_factual_error_rate"] is None else f"{item['major_factual_error_rate']:.2%}"
        agreement = "N/A" if item["automatic_human_agreement_rate"] is None else f"{item['automatic_human_agreement_rate']:.2%} (n={item['automatic_human_agreement_denominator']})"
        lines.append(
            f"| {mode} | {item['total_answer_count']} | {item['qa_success_count']} | "
            f"{item['scoring_completed_count']} | {item['pending_human_review_count']} | "
            f"{item['human_valid_final_count']} | {item['correct_count']}/{item['partial_count']}/{item['incorrect_count']} | "
            f"{rate} | {coverage} | {completeness} | {major} | {agreement} | "
            f"{item['automatic_valid_estimate_count']} |"
        )
    lines.extend(["", "## 完整性声明", ""])
    lines.extend(f"- `{mode}`：{report['per_mode'][mode]['formal_result_note']}" for mode in MODES)
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

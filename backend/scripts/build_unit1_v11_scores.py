"""Offline replay of the 60 frozen answers through Unit 1 v1.1 rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.evaluation.adjudication_v11 import (
    export_review_template,
    read_jsonl,
    write_jsonl,
)
from app.services.evaluation.v11_rules import (
    V11RuleError,
    apply_v11_overlay,
    canonical_record_id,
    load_v11_rules,
)


def select_latest_completed_scores(
    responses: list[dict[str, Any]],
    base_scores: list[dict[str, Any]],
    recovery_generations: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    response_map = {item["record_id"]: item for item in responses}
    if len(response_map) != len(responses):
        raise V11RuleError("Duplicate response record_id")
    selected: dict[str, dict[str, Any]] = {}
    for generation_index, generation in enumerate([base_scores, *recovery_generations]):
        generation_ids: set[str] = set()
        for score in generation:
            canonical_id = canonical_record_id(score)
            if not canonical_id:
                raise V11RuleError(f"Score in generation {generation_index} has no canonical record ID")
            if canonical_id in generation_ids:
                raise V11RuleError(f"Duplicate score for {canonical_id} in generation {generation_index}")
            generation_ids.add(canonical_id)
            if canonical_id not in response_map:
                raise V11RuleError(f"Score references unknown response: {canonical_id}")
            if score.get("scoring_status") == "completed":
                response = response_map[canonical_id]
                if score.get("answer_text") != response.get("answer_text"):
                    raise V11RuleError(f"Source answer text changed for {canonical_id}")
                if score.get("question_id") != response.get("question_id"):
                    raise V11RuleError(f"Question identity changed for {canonical_id}")
                if score.get("requested_mode") != response.get("requested_mode"):
                    raise V11RuleError(f"Retrieval mode changed for {canonical_id}")
                selected[canonical_id] = score
    missing = sorted(set(response_map) - set(selected))
    if missing:
        raise V11RuleError(f"No completed semantic score after recovery for: {missing}")
    return [selected[item["record_id"]] for item in responses]


def build_offline_replay(
    *,
    responses_path: Path,
    base_scores_path: Path,
    recovery_scores_paths: list[Path],
    rubric_path: Path,
    rules_path: Path,
    output_dir: Path,
    replay_run_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    responses = read_jsonl(responses_path)
    base_scores = read_jsonl(base_scores_path)
    recoveries = [read_jsonl(path) for path in recovery_scores_paths]
    rules = load_v11_rules(rules_path, rubric_path)
    selected = select_latest_completed_scores(responses, base_scores, recoveries)
    if len(responses) != 60:
        raise V11RuleError(f"Expected 60 frozen responses, found {len(responses)}")
    coverage = Counter((item["requested_mode"] for item in responses))
    if coverage != Counter({"none": 20, "keyword": 20, "vector": 20}):
        raise V11RuleError(f"Expected 20 responses per mode, found {dict(coverage)}")
    question_coverage = {
        mode: {item["question_id"] for item in responses if item["requested_mode"] == mode}
        for mode in ("none", "keyword", "vector")
    }
    if any(len(ids) != 20 for ids in question_coverage.values()):
        raise V11RuleError("Each retrieval mode must contain Q001-Q020 exactly once")

    unified = []
    for source_score in selected:
        result = apply_v11_overlay(source_score, rules, rubric_path)
        result["source_run_id"] = result["run_id"]
        result["run_id"] = replay_run_id
        unified.append(result)
    paths = {
        "scores": output_dir / "unified_scores.jsonl",
        "template": output_dir / "human_adjudication_template.csv",
        "all_criteria_template": output_dir / "human_adjudication_all_criteria.csv",
        "report": output_dir / "offline_validation_report.json",
        "markdown": output_dir / "offline_validation_report.md",
        "signoff": output_dir / "signoff_validation.json",
    }
    if output_dir.exists() and not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite v1.1 outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["scores"], unified, overwrite=overwrite)
    if paths["template"].exists() and overwrite:
        paths["template"].unlink()
    if paths["all_criteria_template"].exists() and overwrite:
        paths["all_criteria_template"].unlink()
    template_rows = export_review_template(unified, paths["template"])
    all_criteria_template_rows = export_review_template(
        unified,
        paths["all_criteria_template"],
        include_all=True,
    )
    pending_records = [item for item in unified if item["needs_review"]]
    complete_records = [item for item in unified if item["scoring_completed"]]
    pending_criteria_count = sum(item["pending_criterion_count"] for item in unified)
    per_mode_summary = {}
    for mode in ("none", "keyword", "vector"):
        mode_records = [item for item in unified if item["retrieval_mode"] == mode]
        per_mode_summary[mode] = {
            "total_answer_count": len(mode_records),
            "automatic_complete_answer_count": sum(item["scoring_completed"] for item in mode_records),
            "needs_review_answer_count": sum(item["needs_review"] for item in mode_records),
            "pending_criterion_instance_count": sum(item["pending_criterion_count"] for item in mode_records),
        }
    prompt_versions = Counter(str(item.get("prompt_version")) for item in unified)
    report = {
        "schema_version": "historyai-unit1-v1.1-offline-validation-v1",
        "replay_run_id": replay_run_id,
        "rule_version": rules["rule_version"],
        "external_model_calls": 0,
        "response_count": len(responses),
        "unified_score_count": len(unified),
        "question_count": len({item["question_id"] for item in unified}),
        "per_mode_count": dict(Counter(item["retrieval_mode"] for item in unified)),
        "source_prompt_versions": dict(prompt_versions),
        "encoded_confirmed_rule_count": rules["confirmed_rule_count"],
        "encoded_confirmed_rule_counts": rules["confirmed_rule_counts"],
        "preserved_existing_a_rule_count": len(rules["preserved_existing_a_rules"]),
        "deferred_a_rule_count": len(rules["deferred_a_rules"]),
        "encoded_confirmed_d_rule_count": len(rules["confirmed_rules"]),
        "manual_d_rule_count": len(rules["manual_rules"]),
        "scoring_completed_answer_count": len(complete_records),
        "pending_answer_count": len(pending_records),
        "pending_criterion_instance_count": pending_criteria_count,
        "pending_criterion_template_row_count": template_rows,
        "all_criteria_template_row_count": all_criteria_template_rows,
        "human_confirmed_final_count": 0,
        "formal_accuracy_available": False,
        "formal_accuracy_block_reason": "No completed human adjudication import; pending items are not treated as zero.",
        "source_files_preserved": True,
        "formal_signoff_gap": rules["provenance"]["formal_signoff_gap"],
        "per_mode_summary": per_mode_summary,
        "comparison_to_previous": {
            "previous": {
                "automatic_complete_answer_count": 6,
                "needs_review_answer_count": 54,
                "pending_criterion_instance_count": 128,
            },
            "current": {
                "automatic_complete_answer_count": len(complete_records),
                "needs_review_answer_count": len(pending_records),
                "pending_criterion_instance_count": pending_criteria_count,
            },
            "delta": {
                "automatic_complete_answer_count": len(complete_records) - 6,
                "needs_review_answer_count": len(pending_records) - 54,
                "pending_criterion_instance_count": pending_criteria_count - 128,
            },
        },
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gap = {
        "rule_version": rules["rule_version"],
        "formal_signoff_file_found": True,
        "validated_signoff_file_count": len(rules["provenance"]["signoff_files"]),
        "confirmed_rule_counts": rules["confirmed_rule_counts"],
        "confirmed_rule_count": rules["confirmed_rule_count"],
        "global_rule_signoffs": rules["global_rule_signoffs"],
        "note": rules["provenance"]["formal_signoff_gap_note"],
    }
    paths["signoff"].write_text(json.dumps(gap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# 第一单元 v1.1 离线数据贯通验证",
        "",
        f"- 规则版本：`{rules['rule_version']}`",
        f"- 原始回答：{len(responses)}（none/keyword/vector 各20）",
        f"- 已统一重放评分：{len(unified)}",
        f"- 已编码用户确认规则：{rules['confirmed_rule_count']}（A9/B8/C37/D16）",
        f"- 保持既有A类子权重：{len(rules['preserved_existing_a_rules'])}",
        f"- 暂缓A类规则：{len(rules['deferred_a_rules'])}",
        f"- 持续人工D类规则：{len(rules['manual_rules'])}",
        f"- 已获得完整最终分数的回答：{len(complete_records)}",
        f"- 仍待人工复核的回答：{len(pending_records)}",
        f"- 人工模板待裁决评分点：{template_rows}",
        f"- 全评分点复核模板行数：{all_criteria_template_rows}",
        "- 外部模型调用：0",
        "- 正式回答正确率：不可计算（有效人工终判为0）",
        "",
        "## 分模式自动评分状态",
        "",
        "| 模式 | 回答 | 自动完整 | needs_review | 待裁决评分点 |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {mode} | {item['total_answer_count']} | {item['automatic_complete_answer_count']} | {item['needs_review_answer_count']} | {item['pending_criterion_instance_count']} |"
            for mode, item in per_mode_summary.items()
        ],
        "",
        "## 与上一版离线结果对比",
        "",
        f"- 自动完整评分回答：6 → {len(complete_records)}（{len(complete_records) - 6:+d}）",
        f"- needs_review回答：54 → {len(pending_records)}（{len(pending_records) - 54:+d}）",
        f"- 待人工裁决评分点：128 → {pending_criteria_count}（{pending_criteria_count - 128:+d}）",
        "",
        "## 保护与阻断",
        "",
        "旧回答、旧评分和冻结题集均只读。所有pending评分点保持pending，没有静默按0分。",
        "四份签核文件已按配置哈希和ID集合验证。G03/G04已确认；G01/G02/G05继续保持未签核。",
    ]
    paths["markdown"].write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--recovery-scores", type=Path, action="append", default=[])
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="unit1-v1.1-offline-replay-20261027")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_offline_replay(
        responses_path=args.responses,
        base_scores_path=args.base_scores,
        recovery_scores_paths=args.recovery_scores,
        rubric_path=args.rubric,
        rules_path=args.rules,
        output_dir=args.output_dir,
        replay_run_id=args.run_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

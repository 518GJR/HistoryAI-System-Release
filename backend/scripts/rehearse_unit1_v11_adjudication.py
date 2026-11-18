"""Run an isolated mock rehearsal of the v1.1 adjudication importer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from app.services.evaluation.adjudication_v11 import (
    AdjudicationError,
    REVIEW_FIELDS,
    export_review_template,
    import_adjudications,
    write_jsonl,
)
from app.services.evaluation.v11_rules import load_v11_rules


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mock_result() -> dict[str, Any]:
    return {
        "run_id": "mock-unit1-v1.1",
        "record_id": "mock:Q001:none",
        "question_id": "Q001",
        "retrieval_mode": "none",
        "answer_text": "仅用于导入演练的Mock回答",
        "rule_version": "unit1_v1.1",
        "criteria": [
            {
                "criterion_id": "Q001-C01",
                "max_score": 2.0,
                "automatic": {"decision": "pending", "reason": "mock pending", "score": None},
                "final": {
                    "score": None,
                    "decision": "pending",
                    "source": "pending",
                    "reason": "mock pending",
                    "automatic_preserved": True,
                    "difference_from_automatic": None,
                },
            },
            {
                "criterion_id": "Q001-C02",
                "max_score": 8.0,
                "automatic": {"decision": "full", "reason": "mock automatic", "score": 8.0},
                "final": {
                    "score": 8.0,
                    "decision": "full",
                    "source": "automatic",
                    "reason": "mock automatic",
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


def _rewrite_csv(path: Path, edit: Callable[[list[dict[str, str]]], None]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    edit(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _final_row(row: dict[str, str], *, score: str = "2", reason: str = "Mock合法理由") -> None:
    row.update({
        "human_score": score,
        "human_reason": reason,
        "textbook_citation": "Mock教材依据",
        "major_error_confirmation": "no",
        "reviewer": "mock-reviewer",
        "second_reviewer": "mock-second-reviewer",
        "adjudication_status": "overridden",
    })


def run_rehearsal(
    *,
    rules_path: Path,
    protected_files: list[Path],
    report_dir: Path,
) -> dict[str, Any]:
    if report_dir.exists():
        raise FileExistsError(f"Refusing to reuse rehearsal result directory: {report_dir}")
    before = {str(path): _sha256(path) for path in protected_files}
    rules = load_v11_rules(rules_path)
    mock = [_mock_result()]
    checks: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="historyai-v11-mock-") as temporary:
        temp = Path(temporary)

        legal = temp / "legal.csv"
        export_review_template(mock, legal)
        _rewrite_csv(legal, lambda rows: _final_row(rows[0]))
        merged = import_adjudications(mock, legal, rules)[0]
        checks.append({
            "check": "legal_score_import",
            "passed": merged["final_score"] == 10.0 and merged["human_confirmed_final"] is True,
            "detail": f"final_score={merged['final_score']}, human_confirmed_final={merged['human_confirmed_final']}",
        })

        out_of_range = temp / "out_of_range.csv"
        export_review_template(mock, out_of_range)
        _rewrite_csv(out_of_range, lambda rows: _final_row(rows[0], score="2.01"))
        try:
            import_adjudications(mock, out_of_range, rules)
            range_rejected = False
            range_detail = "unexpectedly accepted"
        except AdjudicationError as exc:
            range_rejected = "out of range" in str(exc)
            range_detail = str(exc)
        checks.append({"check": "out_of_range_rejected", "passed": range_rejected, "detail": range_detail})

        no_reason = temp / "no_reason.csv"
        export_review_template(mock, no_reason)
        _rewrite_csv(no_reason, lambda rows: _final_row(rows[0], reason=""))
        try:
            import_adjudications(mock, no_reason, rules)
            reason_rejected = False
            reason_detail = "unexpectedly accepted"
        except AdjudicationError as exc:
            reason_rejected = "requires human_reason" in str(exc)
            reason_detail = str(exc)
        checks.append({"check": "missing_reason_rejected", "passed": reason_rejected, "detail": reason_detail})

        duplicate = temp / "duplicate.csv"
        export_review_template(mock, duplicate)
        def duplicate_edit(rows: list[dict[str, str]]) -> None:
            _final_row(rows[0])
            rows.append(deepcopy(rows[0]))
        _rewrite_csv(duplicate, duplicate_edit)
        try:
            import_adjudications(mock, duplicate, rules)
            duplicate_rejected = False
            duplicate_detail = "unexpectedly accepted"
        except AdjudicationError as exc:
            duplicate_rejected = "Duplicate adjudication ID" in str(exc)
            duplicate_detail = str(exc)
        checks.append({"check": "duplicate_id_rejected", "passed": duplicate_rejected, "detail": duplicate_detail})

        incomplete = temp / "incomplete.csv"
        export_review_template(mock, incomplete)
        incomplete_result = import_adjudications(mock, incomplete, rules)[0]
        pending_preserved = (
            incomplete_result["criteria"][0]["final"]["score"] is None
            and incomplete_result["final_score"] is None
            and incomplete_result["human_confirmed_final"] is False
        )
        checks.append({
            "check": "incomplete_remains_pending_not_zero",
            "passed": pending_preserved,
            "detail": "pending criterion score and answer final score remain null",
        })

        collision = temp / "existing.jsonl"
        collision.write_text("existing protected content\n", encoding="utf-8")
        try:
            write_jsonl(collision, [merged], overwrite=False)
            overwrite_refused = False
            overwrite_detail = "unexpectedly overwritten"
        except FileExistsError as exc:
            overwrite_refused = collision.read_text(encoding="utf-8") == "existing protected content\n"
            overwrite_detail = str(exc)
        checks.append({
            "check": "existing_output_not_overwritten",
            "passed": overwrite_refused,
            "detail": overwrite_detail,
        })

    after = {str(path): _sha256(path) for path in protected_files}
    source_unchanged = before == after
    checks.append({
        "check": "original_experiment_files_unchanged",
        "passed": source_unchanged,
        "detail": "SHA256 unchanged for all protected source files",
    })
    report = {
        "schema_version": "historyai-unit1-v1.1-mock-rehearsal-v1",
        "rule_version": rules["rule_version"],
        "mock_scope": "temporary_directory_only",
        "mock_written_to_formal_gold": False,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "checks": checks,
        "passed_count": sum(bool(item["passed"]) for item in checks),
        "check_count": len(checks),
        "all_passed": all(bool(item["passed"]) for item in checks),
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "mock_import_rehearsal.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 第一单元 v1.1 Mock人工裁决导入演练",
        "",
        f"规则版本：`{rules['rule_version']}`",
        "",
        "Mock记录只存在于系统临时目录，没有写入正式人工金标准。",
        "",
        "| 检查 | 结果 | 说明 |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['check']} | {'通过' if item['passed'] else '失败'} | {item['detail']} |")
    (report_dir / "mock_import_rehearsal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--protected-file", type=Path, action="append", default=[])
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_rehearsal(
        rules_path=args.rules,
        protected_files=args.protected_file,
        report_dir=args.report_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

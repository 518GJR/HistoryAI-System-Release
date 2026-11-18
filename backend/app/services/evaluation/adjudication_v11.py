"""CSV export/import for criterion-level Unit 1 v1.1 adjudication."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


class AdjudicationError(ValueError):
    """Human-review data failed identity, range, or version validation."""


REVIEW_FIELDS = [
    "run_id",
    "record_id",
    "question_id",
    "retrieval_mode",
    "criterion_id",
    "criterion_weight",
    "answer_text",
    "automatic_decision",
    "automatic_reason",
    "automatic_score",
    "human_score",
    "human_reason",
    "textbook_citation",
    "major_error_confirmation",
    "reviewer",
    "second_reviewer",
    "adjudication_status",
    "rule_version",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AdjudicationError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _source_index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for result in results:
        record_id = str(result.get("record_id") or "")
        if not record_id:
            raise AdjudicationError("Score result is missing record_id")
        if record_id in index:
            raise AdjudicationError(f"Duplicate score record_id: {record_id}")
        index[record_id] = result
    return index


def export_review_template(
    results: list[dict[str, Any]],
    output_path: Path,
    *,
    include_all: bool = False,
) -> int:
    """Export pending criteria, or every criterion for optional human override."""

    _source_index(results)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing template: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: (item["question_id"], item["retrieval_mode"])):
        for criterion in result["criteria"]:
            if not include_all and criterion["final"]["score"] is not None:
                continue
            automatic = criterion["automatic"]
            rows.append({
                "run_id": result["run_id"],
                "record_id": result["record_id"],
                "question_id": result["question_id"],
                "retrieval_mode": result["retrieval_mode"],
                "criterion_id": criterion["criterion_id"],
                "criterion_weight": criterion["max_score"],
                "answer_text": result["answer_text"],
                "automatic_decision": automatic["decision"],
                "automatic_reason": automatic["reason"],
                "automatic_score": "" if automatic["score"] is None else automatic["score"],
                "human_score": "",
                "human_reason": "",
                "textbook_citation": "",
                "major_error_confirmation": "pending",
                "reviewer": "",
                "second_reviewer": "",
                "adjudication_status": "pending",
                "rule_version": result["rule_version"],
            })
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _decimal_score(raw: str, maximum: float, *, precision: int, location: str) -> float:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise AdjudicationError(f"Invalid human_score at {location}: {raw!r}") from exc
    if not value.is_finite() or value < 0 or value > Decimal(str(maximum)):
        raise AdjudicationError(f"human_score out of range at {location}: {value} not in [0, {maximum}]")
    if max(0, -value.as_tuple().exponent) > precision:
        raise AdjudicationError(
            f"human_score has more than {precision} decimal places at {location}: {value}"
        )
    return float(value)


def _label(score: float, major_error: bool, config: dict[str, Any]) -> str:
    thresholds = config["answer_label_thresholds"]
    if major_error and thresholds.get("major_error_forces_incorrect", True):
        return "incorrect"
    if score >= float(thresholds["correct_min"]):
        return "correct"
    if score >= float(thresholds["partial_min"]):
        return "partial"
    return "incorrect"


def import_adjudications(
    results: list[dict[str, Any]],
    csv_path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge final CSV rows while preserving both automatic and human decisions."""

    merged = deepcopy(results)
    result_index = _source_index(merged)
    criterion_index = {
        (result["record_id"], criterion["criterion_id"]): (result, criterion)
        for result in merged
        for criterion in result["criteria"]
    }
    seen: set[tuple[str, str]] = set()
    major_error_reviews: dict[str, list[bool]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REVIEW_FIELDS:
            raise AdjudicationError(
                f"Unexpected CSV columns; expected={REVIEW_FIELDS}, actual={reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            record_id = row["record_id"].strip()
            criterion_id = row["criterion_id"].strip()
            key = (record_id, criterion_id)
            location = f"line {line_number} ({record_id}/{criterion_id})"
            if key in seen:
                raise AdjudicationError(f"Duplicate adjudication ID at {location}")
            seen.add(key)
            if key not in criterion_index:
                raise AdjudicationError(f"Unknown record/criterion at {location}")
            result, criterion = criterion_index[key]
            identity_checks = {
                "run_id": str(result["run_id"]),
                "question_id": str(result["question_id"]),
                "retrieval_mode": str(result["retrieval_mode"]),
                "criterion_weight": str(criterion["max_score"]),
                "answer_text": str(result["answer_text"]),
                "rule_version": str(result["rule_version"]),
                "automatic_decision": str(criterion["automatic"]["decision"]),
                "automatic_reason": str(criterion["automatic"]["reason"]),
                "automatic_score": (
                    "" if criterion["automatic"]["score"] is None
                    else str(criterion["automatic"]["score"])
                ),
            }
            for field, expected in identity_checks.items():
                if row[field] != expected:
                    raise AdjudicationError(
                        f"Identity field {field} changed at {location}; expected={expected!r}"
                    )
            if row["rule_version"] != config["rule_version"]:
                raise AdjudicationError(f"Rule version conflict at {location}")

            status = row["adjudication_status"].strip().lower()
            if status not in {"pending", "confirmed", "overridden"}:
                raise AdjudicationError(f"Invalid adjudication_status at {location}: {status!r}")
            major_raw = row["major_error_confirmation"].strip().lower()
            if major_raw not in {"", "pending", "yes", "no"}:
                raise AdjudicationError(f"Invalid major_error_confirmation at {location}: {major_raw!r}")
            if major_raw in {"yes", "no"}:
                major_error_reviews.setdefault(record_id, []).append(major_raw == "yes")

            score_raw = row["human_score"].strip()
            if status == "pending" or not score_raw:
                if status != "pending" or score_raw:
                    raise AdjudicationError(
                        f"Pending rows must have a blank human_score at {location}"
                    )
                continue
            if not row["human_reason"].strip() or not row["reviewer"].strip():
                raise AdjudicationError(f"Final adjudication requires human_reason and reviewer at {location}")
            score = _decimal_score(
                score_raw,
                criterion["max_score"],
                precision=int(config["score_precision"]),
                location=location,
            )
            auto_score = criterion["automatic"]["score"]
            actual_status = "confirmed" if auto_score == score and auto_score is not None else "overridden"
            if status != actual_status:
                raise AdjudicationError(
                    f"adjudication_status must be {actual_status!r} at {location}"
                )
            difference = None if auto_score is None else round(score - float(auto_score), int(config["score_precision"]))
            criterion["human_adjudication"] = {
                "score": score,
                "reason": row["human_reason"].strip(),
                "textbook_citation": row["textbook_citation"].strip(),
                "major_error_confirmation": major_raw or "pending",
                "reviewer": row["reviewer"].strip(),
                "second_reviewer": row["second_reviewer"].strip(),
                "adjudication_status": status,
                "rule_version": row["rule_version"],
            }
            criterion["final"] = {
                "score": score,
                "decision": status,
                "source": "human",
                "reason": row["human_reason"].strip(),
                "automatic_preserved": True,
                "difference_from_automatic": difference,
            }

    for result in merged:
        human_count = sum("human_adjudication" in item for item in result["criteria"])
        pending_count = sum(item["final"]["score"] is None for item in result["criteria"])
        final_scores = [item["final"]["score"] for item in result["criteria"] if item["final"]["score"] is not None]
        major_reviews = major_error_reviews.get(result["record_id"], [])
        if any(major_reviews):
            result["major_error_confirmed"] = True
        major_error_reviewed = bool(major_reviews) or bool(result.get("major_error_confirmations"))
        score_complete = pending_count == 0
        score = round(sum(final_scores), int(config["score_precision"])) if score_complete else None
        if score is not None and not (0 <= score <= float(config["question_max_score"])):
            raise AdjudicationError(f"Final answer score out of range for {result['record_id']}: {score}")
        human_final = human_count > 0 and score_complete and major_error_reviewed
        result["human_adjudication_count"] = human_count
        result["pending_criterion_count"] = pending_count
        result["scoring_completed"] = score_complete
        result["final_score"] = score
        result["final_label"] = (
            _label(score, bool(result["major_error_confirmed"]), config) if score is not None else None
        )
        result["major_error_human_reviewed"] = major_error_reviewed
        result["human_confirmed_final"] = human_final
        result["needs_review"] = pending_count > 0 or (human_count > 0 and not major_error_reviewed)
        if human_count > 0 and not major_error_reviewed:
            result["answer_level_review_reason"] = "major factual error confirmation remains pending"
    return merged

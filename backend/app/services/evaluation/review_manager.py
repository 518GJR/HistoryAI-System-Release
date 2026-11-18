"""Append-only human review records; automatic results are never overwritten."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ReviewRecord, ScoreResult


class ReviewManager:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: ReviewRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = {item.review_id for item in self.read_all()}
        if record.review_id in existing:
            raise ValueError(f"Duplicate review_id: {record.review_id}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def read_all(self) -> list[ReviewRecord]:
        if not self.path.exists():
            return []
        records = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    records.append(ReviewRecord.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid review JSONL at line {line_number}: {exc}") from exc
        return records

    def pending_record(self, score: ScoreResult, review_id: str) -> ReviewRecord:
        return ReviewRecord(
            review_id=review_id, run_id=score.run_id, question_id=score.question_id,
            requested_mode=score.requested_mode, needs_review=score.needs_review,
            review_reason=score.review_reason, review_status="pending",
            automatic_result=score.model_dump(mode="json"),
        )

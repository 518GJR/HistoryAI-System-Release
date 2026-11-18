"""Recover only failed semantic scores from an immutable evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.services.evaluation.models import RubricDataset, RubricQuestion
from app.services.evaluation.rubric_loader import load_rubric
from app.services.evaluation.scoring_engine import SCORER_VERSION, score_answer
from app.services.evaluation.semantic_judge import (
    OpenAICompatibleSemanticJudge,
    SemanticJudge,
    SemanticJudgeError,
)
from app.services.provider_config import provider_config


ALLOWED_MODES = {"none", "keyword", "vector"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required recovery input is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"Required recovery input is missing: {path}") from exc
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _key(record: dict) -> tuple[str, str]:
    try:
        return str(record["question_id"]), str(record["requested_mode"])
    except KeyError as exc:
        raise ValueError(f"Recovery record is missing {exc.args[0]}") from exc


def _unique_by_key(records: list[dict], label: str) -> dict[tuple[str, str], dict]:
    result = {}
    for record in records:
        key = _key(record)
        if key in result:
            raise ValueError(f"Duplicate {label} record for {key[0]}:{key[1]}")
        result[key] = record
    return result


def _parse_record_ids(record_ids: list[str] | None) -> list[tuple[str, str]] | None:
    if record_ids is None:
        return None
    if not record_ids:
        raise ValueError("--record-ids requires at least one QUESTION_ID:MODE value")

    parsed = []
    seen = set()
    for record_id in record_ids:
        parts = record_id.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Invalid recovery record ID {record_id!r}; expected QUESTION_ID:MODE"
            )
        key = (parts[0], parts[1])
        if key[1] not in ALLOWED_MODES:
            raise ValueError(
                f"Invalid recovery mode {key[1]!r} in {record_id!r}; "
                f"expected one of {sorted(ALLOWED_MODES)}"
            )
        if key in seen:
            raise ValueError(f"Duplicate recovery record ID: {record_id}")
        seen.add(key)
        parsed.append(key)
    return parsed


@dataclass(frozen=True)
class RecoveryTarget:
    question: RubricQuestion
    response: dict
    failed_score: dict

    @property
    def source_record_id(self) -> str:
        return str(self.failed_score["record_id"])


def load_recovery_targets(
    source_run_dir: Path,
    rubric: RubricDataset,
    *,
    expected_failed_count: int | None = 9,
    record_ids: list[str] | None = None,
) -> tuple[dict, list[RecoveryTarget]]:
    """Validate source linkage and return only scoring_status=failed records."""

    metadata = _read_json(source_run_dir / "run_metadata.json")
    responses = _read_jsonl(source_run_dir / "responses.jsonl")
    scores = _read_jsonl(source_run_dir / "scores.jsonl")
    response_map = _unique_by_key(responses, "response")
    score_map = _unique_by_key(scores, "score")

    if set(response_map) != set(score_map):
        missing_scores = sorted(set(response_map) - set(score_map))
        missing_responses = sorted(set(score_map) - set(response_map))
        raise ValueError(
            "Source response/score keys differ; "
            f"missing_scores={missing_scores}, missing_responses={missing_responses}"
        )
    if metadata.get("rubric_version") != rubric.rubric_version:
        raise ValueError("Source run and recovery rubric versions do not match")

    question_map = {question.question_id: question for question in rubric.questions}
    if len(question_map) != len(rubric.questions):
        raise ValueError("Recovery rubric contains duplicate question IDs")

    selected_keys = _parse_record_ids(record_ids)
    if selected_keys is not None:
        for question_id, requested_mode in selected_keys:
            selection = f"{question_id}:{requested_mode}"
            if question_id not in question_map:
                raise ValueError(f"Unknown recovery question ID: {question_id}")
            score = score_map.get((question_id, requested_mode))
            if score is None:
                raise ValueError(f"Recovery record does not exist in source run: {selection}")
            if score.get("scoring_status") != "failed":
                raise ValueError(
                    f"Recovery record is not a failed score and cannot be selected: {selection}"
                )

    targets = []
    for key, failed_score in score_map.items():
        if failed_score.get("scoring_status") != "failed":
            continue
        response = response_map[key]
        question_id, requested_mode = key
        question = question_map.get(question_id)
        if question is None:
            raise ValueError(f"Failed score references unknown question {question_id}")
        if response.get("execution_status") != "success":
            raise ValueError(f"Failed score {question_id}:{requested_mode} has no successful answer")
        if response.get("record_id") != failed_score.get("record_id"):
            raise ValueError(f"Source record IDs differ for {question_id}:{requested_mode}")
        if response.get("run_id") != metadata.get("run_id") or failed_score.get("run_id") != metadata.get("run_id"):
            raise ValueError(f"Source run IDs differ for {question_id}:{requested_mode}")
        if response.get("executed_mode") != failed_score.get("executed_mode"):
            raise ValueError(f"Executed modes differ for {question_id}:{requested_mode}")
        if response.get("requested_mode") != failed_score.get("requested_mode"):
            raise ValueError(f"Requested modes differ for {question_id}:{requested_mode}")
        if response.get("question_text") != question.question_text:
            raise ValueError(f"Question text differs from rubric for {question_id}:{requested_mode}")
        if not isinstance(response.get("answer_text"), str):
            raise ValueError(f"Source answer is not a string for {question_id}:{requested_mode}")
        targets.append(RecoveryTarget(question, response, failed_score))

    if selected_keys is not None:
        targets_by_key = {
            (target.question.question_id, target.response["requested_mode"]): target
            for target in targets
        }
        targets = [targets_by_key[key] for key in selected_keys]

    if expected_failed_count is not None and len(targets) != expected_failed_count:
        raise ValueError(
            f"Expected {expected_failed_count} failed scores, found {len(targets)}"
        )
    return metadata, targets


class ScoreRecoveryRunner:
    def __init__(
        self,
        *,
        rubric: RubricDataset,
        source_run_dir: Path,
        output_dir: Path,
        recovery_run_id: str,
        semantic_judge: SemanticJudge,
        expected_failed_count: int | None = 9,
        record_ids: list[str] | None = None,
    ):
        self.rubric = rubric
        self.source_run_dir = source_run_dir
        self.output_dir = output_dir
        self.recovery_run_id = recovery_run_id
        self.semantic_judge = semantic_judge
        self.expected_failed_count = expected_failed_count
        self.record_ids = record_ids
        self.scores_path = output_dir / "scores.jsonl"
        self.review_queue_path = output_dir / "review_queue.jsonl"
        self.summary_path = output_dir / "recovery_summary.json"

    def _identity(self, source_metadata: dict) -> dict:
        config = getattr(self.semantic_judge, "config", {})
        return {
            "schema_version": "evaluation-score-recovery-v1",
            "recovery_run_id": self.recovery_run_id,
            "source_run_id": source_metadata["run_id"],
            "source_run_directory": str(self.source_run_dir.resolve()),
            "source_file_sha256": {
                name: _sha256(self.source_run_dir / name)
                for name in ("run_metadata.json", "responses.jsonl", "scores.jsonl")
            },
            "source_rubric_version": source_metadata["rubric_version"],
            "scorer_prompt_version": self.semantic_judge.prompt_version,
            "scorer_config_id": self.semantic_judge.config_id,
            "scoring_model": config.get("model") if isinstance(config, dict) else None,
        }

    def _score_one(self, target: RecoveryTarget, source_metadata: dict) -> dict:
        started_at = _utc_now()
        response = target.response
        base = {
            "record_id": f"{self.recovery_run_id}:{target.question.question_id}:{response['requested_mode']}",
            "run_id": self.recovery_run_id,
            "recovery_run_id": self.recovery_run_id,
            "recovered_from_record_id": target.source_record_id,
            "source_run_id": source_metadata["run_id"],
            "source_response_record_id": response["record_id"],
            "source_failure_reason": target.failed_score.get("review_reason", []),
            "question_id": target.question.question_id,
            "question_text": target.question.question_text,
            "answer_text": response["answer_text"],
            "requested_mode": response["requested_mode"],
            "executed_mode": response["executed_mode"],
            "rubric_version": self.rubric.rubric_version,
            "scorer_version": SCORER_VERSION,
            "prompt_version": self.semantic_judge.prompt_version,
            "scorer_config_id": self.semantic_judge.config_id,
            "scoring_model": (
                getattr(self.semantic_judge, "config", {}).get("model")
                if isinstance(getattr(self.semantic_judge, "config", {}), dict)
                else None
            ),
            "started_at": started_at,
        }
        try:
            assessment, raw = self.semantic_judge.judge(
                target.question, response["answer_text"]
            )
            result = score_answer(
                run_id=self.recovery_run_id,
                question=target.question,
                requested_mode=response["requested_mode"],
                executed_mode=response["executed_mode"],
                answer_text=response["answer_text"],
                judgments=assessment.criterion_results,
                directly_addresses_question=assessment.directly_addresses_question,
                has_verifiable_fact=assessment.has_verifiable_fact,
                answer_level_contradiction=assessment.answer_level_contradiction,
                qualifier_issue_detected=assessment.qualifier_issue_detected,
                confirmed_major_errors=[],
                prompt_version=self.semantic_judge.prompt_version,
                scorer_config_id=self.semantic_judge.config_id,
                raw_semantic_output=raw,
            )
            record = result.model_dump(mode="json")
            record.update(base)
            record["scoring_status"] = "completed"
        except (SemanticJudgeError, ValueError) as exc:
            record = {
                **base,
                "scoring_status": "failed",
                "needs_review": True,
                "review_reason": [str(exc)],
                "final_correctness": None,
                "completeness_score": None,
                "raw_semantic_output": None,
            }
        record["finished_at"] = _utc_now()
        return record

    def _review_record(self, score: dict) -> dict:
        return {
            "review_id": f'review:{score["record_id"]}',
            "recovery_run_id": self.recovery_run_id,
            "recovered_from_record_id": score["recovered_from_record_id"],
            "run_id": self.recovery_run_id,
            "question_id": score["question_id"],
            "requested_mode": score["requested_mode"],
            "needs_review": True,
            "review_reason": score.get("review_reason", []),
            "review_status": "pending",
            "reviewer_decision": None,
            "reviewer_score": None,
            "reviewer_comment": "",
            "reviewed_at": None,
            "automatic_result": score,
        }

    def _summary(self, identity: dict, targets: list[RecoveryTarget], scores: list[dict]) -> dict:
        completed = [item for item in scores if item.get("scoring_status") == "completed"]
        failed = [item for item in scores if item.get("scoring_status") == "failed"]
        processed = {item["recovered_from_record_id"] for item in scores}
        return {
            **identity,
            "generated_at": _utc_now(),
            "target_count": len(targets),
            "processed_count": len(processed),
            "remaining_count": len(targets) - len(processed),
            "scoring_completed_count": len(completed),
            "scoring_failed_count": len(failed),
            "needs_review_count": sum(bool(item.get("needs_review")) for item in scores),
            "valid_final_decision_count": sum(
                item.get("final_correctness") in {"correct", "partial", "incorrect"}
                and not item.get("needs_review")
                for item in completed
            ),
            "target_source_record_ids": [target.source_record_id for target in targets],
            "processed_source_record_ids": sorted(processed),
            "failed_source_record_ids": [
                item["recovered_from_record_id"] for item in failed
            ],
            "accuracy_note": (
                "Recovery validates scoring execution and structure only; it does not establish "
                "agreement with an independent human gold standard."
            ),
        }

    def run(self) -> dict:
        source_resolved = self.source_run_dir.resolve()
        output_resolved = self.output_dir.resolve()
        if source_resolved == output_resolved or source_resolved in output_resolved.parents:
            raise ValueError(
                "Recovery output directory must be outside the source run directory"
            )
        source_metadata, targets = load_recovery_targets(
            self.source_run_dir,
            self.rubric,
            expected_failed_count=self.expected_failed_count,
            record_ids=self.record_ids,
        )
        if self.record_ids is not None:
            previous_output = Path(str(source_resolved) + "-score-recovery-v2")
            previous_run_id = f'{source_metadata["run_id"]}-score-recovery-v2'
            if output_resolved == previous_output.resolve():
                raise ValueError(
                    "Filtered recovery requires a new output directory; refusing the first "
                    "recovery batch directory"
                )
            if self.recovery_run_id == previous_run_id:
                raise ValueError(
                    "Filtered recovery requires a new recovery run ID; refusing the first "
                    "recovery batch ID"
                )
        identity = self._identity(source_metadata)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scores_path.touch(exist_ok=True)
        self.review_queue_path.touch(exist_ok=True)

        if self.summary_path.exists():
            prior_summary = _read_json(self.summary_path)
            for field, expected in identity.items():
                if prior_summary.get(field) != expected:
                    raise ValueError(f"Existing recovery summary has different {field}")

        target_ids = {target.source_record_id for target in targets}
        score_lines = _read_jsonl(self.scores_path)
        existing = {}
        for score in score_lines:
            source_id = score.get("recovered_from_record_id")
            if source_id not in target_ids:
                raise ValueError(f"Recovery output contains an unknown source record ID: {source_id}")
            if source_id in existing:
                raise ValueError(f"Duplicate recovered score for source record ID: {source_id}")
            existing[source_id] = score

        review_lines = _read_jsonl(self.review_queue_path)
        queued = set()
        for review in review_lines:
            source_id = review.get("recovered_from_record_id")
            if source_id not in target_ids:
                raise ValueError(
                    f"Recovery review queue contains an unknown source record ID: {source_id}"
                )
            if source_id in queued:
                raise ValueError(f"Duplicate recovered review for source record ID: {source_id}")
            queued.add(source_id)

        # Repair an interruption after the score append but before its review append.
        for source_id, score in existing.items():
            if score.get("needs_review") and source_id not in queued:
                _append_jsonl(self.review_queue_path, self._review_record(score))
                queued.add(source_id)

        summary = self._summary(identity, targets, list(existing.values()))
        _write_json_atomic(self.summary_path, summary)
        for target in targets:
            if target.source_record_id in existing:
                continue
            score = self._score_one(target, source_metadata)
            _append_jsonl(self.scores_path, score)
            existing[target.source_record_id] = score
            if score.get("needs_review"):
                _append_jsonl(self.review_queue_path, self._review_record(score))
                queued.add(target.source_record_id)
            summary = self._summary(identity, targets, list(existing.values()))
            _write_json_atomic(self.summary_path, summary)
        return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_source = (
        project_root
        / "docs/evaluation/workbench/results/unit1-v1.0-real-pilot-20261026-05"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=default_source)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(default_source) + "-score-recovery-v2"),
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=project_root / "docs/evaluation/workbench/unit1_v1.0_rubric.json",
    )
    parser.add_argument(
        "--recovery-run-id",
        default="unit1-v1.0-real-pilot-20261026-05-score-recovery-v2",
    )
    parser.add_argument("--expected-failed", type=int, default=9)
    parser.add_argument(
        "--record-ids",
        nargs="+",
        metavar="QUESTION_ID:MODE",
        help=(
            "Recover only these source scoring_status=failed records, for example "
            "Q011:none Q016:none"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rubric = load_rubric(args.rubric)
    config = provider_config("EVALUATION")
    if not config:
        raise SystemExit(
            "EVALUATION_* scoring configuration is unavailable; recovery was not started"
        )
    judge = OpenAICompatibleSemanticJudge(config=config)
    runner = ScoreRecoveryRunner(
        rubric=rubric,
        source_run_dir=args.source_run_dir,
        output_dir=args.output_dir,
        recovery_run_id=args.recovery_run_id,
        semantic_judge=judge,
        expected_failed_count=(
            len(args.record_ids) if args.record_ids is not None else args.expected_failed
        ),
        record_ids=args.record_ids,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

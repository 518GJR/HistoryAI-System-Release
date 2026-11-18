"""Resumable batch evaluation through the existing POST /api/question endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

from app.services.evaluation.formal_dataset_loader import (
    load_evaluation_dataset,
    question_for_mode,
)
from app.services.evaluation.scoring_engine import score_answer
from app.services.evaluation.semantic_judge import (
    OpenAICompatibleSemanticJudge,
    SemanticJudge,
    SemanticJudgeError,
)
from app.services.evaluation.models import RubricDataset, RubricQuestion

MODES = ("none", "keyword", "vector")
RESPONSE_FIELDS = {
    "requested_mode", "executed_mode", "answer", "sources", "error",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _review_record(score: dict) -> dict:
    return {
        "review_id": f'review:{score["record_id"]}',
        "run_id": score["run_id"],
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


def _append_review_if_needed(path: Path, score: dict, queued: set[str]) -> None:
    if not score.get("needs_review"):
        return
    record = _review_record(score)
    if record["review_id"] in queued:
        return
    _append_jsonl(path, record)
    queued.add(record["review_id"])


class EvaluationRunner:
    def __init__(
        self,
        *,
        rubric: RubricDataset,
        output_dir: Path,
        run_id: str,
        base_url: str,
        modes: Iterable[str] = MODES,
        timeout: float = 60.0,
        retries: int = 0,
        concurrency: int = 1,
        client: httpx.Client | None = None,
        semantic_judge: SemanticJudge | None = None,
    ):
        self.rubric = rubric
        self.output_dir = output_dir
        self.run_id = run_id
        self.base_url = base_url.rstrip("/")
        self.modes = tuple(modes)
        if not self.modes or any(mode not in MODES for mode in self.modes):
            raise ValueError("modes must be a non-empty subset of none, keyword, vector")
        if not 1 <= concurrency <= 4:
            raise ValueError("concurrency must be between 1 and 4")
        self.timeout = timeout
        self.retries = retries
        self.concurrency = concurrency
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self.semantic_judge = semantic_judge
        self.responses_path = output_dir / "responses.jsonl"
        self.scores_path = output_dir / "scores.jsonl"
        self.review_queue_path = output_dir / "review_queue.jsonl"

    @staticmethod
    def _key(record: dict) -> tuple[str, str]:
        return str(record["question_id"]), str(record["requested_mode"])

    def _request_one(self, question: RubricQuestion, mode: str) -> dict:
        started_at = _utc_now()
        started = time.perf_counter()
        status = "request_error"
        error = None
        payload = None
        http_status = None
        attempt = 0
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}/api/question",
                    json={"question": question.question_text, "mode": mode},
                    timeout=self.timeout,
                )
                http_status = response.status_code
                if response.status_code >= 500 and attempt < self.retries:
                    continue
                if response.status_code >= 400:
                    status = "http_error"
                    try:
                        body = response.json()
                        payload = body
                        if isinstance(body, dict):
                            error = body.get("detail") or body.get("error") or response.text
                        else:
                            error = response.text
                    except ValueError:
                        payload = response.text
                        error = response.text
                    break
                try:
                    payload = response.json()
                except ValueError as exc:
                    payload = response.text
                    status, error = "invalid_response", f"invalid JSON: {exc}"
                    break
                if not isinstance(payload, dict) or not RESPONSE_FIELDS.issubset(payload):
                    status, error = "invalid_response", "required response fields are missing"
                elif payload.get("requested_mode") != mode or payload.get("executed_mode") != mode:
                    status, error = "mode_mismatch", (
                        f"sent {mode}, response requested {payload.get('requested_mode')}, "
                        f"executed {payload.get('executed_mode')}"
                    )
                elif not isinstance(payload.get("answer"), str):
                    status, error = "invalid_response", "answer is not a string"
                elif not isinstance(payload.get("sources"), list):
                    status, error = "invalid_response", "sources is not a list"
                else:
                    status = "success"
                break
            except httpx.TimeoutException as exc:
                status, error = "timeout", str(exc) or "request timed out"
                if attempt >= self.retries:
                    break
            except httpx.HTTPError as exc:
                status, error = "request_error", str(exc)
                if attempt >= self.retries:
                    break
        finished_at = _utc_now()
        return {
            "record_id": f"{self.run_id}:{question.question_id}:{mode}",
            "run_id": self.run_id, "question_id": question.question_id,
            "requested_mode": mode,
            "executed_mode": payload.get("executed_mode") if isinstance(payload, dict) else None,
            "question_text": question.question_text,
            "answer_text": payload.get("answer", "") if isinstance(payload, dict) else "",
            "sources": payload.get("sources", []) if isinstance(payload, dict) else [],
            "raw_api_response": payload,
            "error": error or (payload.get("error") if isinstance(payload, dict) else None),
            "execution_status": status, "http_status": http_status,
            "started_at": started_at, "finished_at": finished_at,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "retry_count": attempt,
            "rubric_version": self.rubric.rubric_version,
        }

    def _score(self, question: RubricQuestion, response_record: dict) -> dict:
        key = f"{self.run_id}:{question.question_id}:{response_record['requested_mode']}"
        if self.semantic_judge is None:
            return {
                "record_id": key, "run_id": self.run_id, "question_id": question.question_id,
                "requested_mode": response_record["requested_mode"],
                "executed_mode": response_record["executed_mode"],
                "scoring_status": "not_configured", "needs_review": True,
                "review_reason": ["semantic scorer was not configured"],
                "final_correctness": None, "completeness_score": None,
                "rubric_version": self.rubric.rubric_version,
            }
        try:
            scoring_question = question_for_mode(question, response_record["requested_mode"])
            assessment, raw = self.semantic_judge.judge(scoring_question, response_record["answer_text"])
            result = score_answer(
                run_id=self.run_id, question=scoring_question,
                requested_mode=response_record["requested_mode"],
                executed_mode=response_record["executed_mode"],
                answer_text=response_record["answer_text"],
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
            record.update({"record_id": key, "scoring_status": "completed"})
            return record
        except (SemanticJudgeError, ValueError) as exc:
            return {
                "record_id": key, "run_id": self.run_id, "question_id": question.question_id,
                "requested_mode": response_record["requested_mode"],
                "executed_mode": response_record["executed_mode"],
                "scoring_status": "failed", "needs_review": True,
                "review_reason": [str(exc)], "final_correctness": None,
                "completeness_score": None, "rubric_version": self.rubric.rubric_version,
            }

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.responses_path.touch(exist_ok=True)
        self.scores_path.touch(exist_ok=True)
        metadata_path = self.output_dir / "run_metadata.json"
        metadata = {
            "run_id": self.run_id, "rubric_version": self.rubric.rubric_version,
            "base_url": self.base_url, "modes": list(self.modes),
            "question_count": len(self.rubric.questions), "planned_count": len(self.rubric.questions) * len(self.modes),
            "concurrency": self.concurrency, "timeout_seconds": self.timeout,
            "max_retries": self.retries,
            "scoring_enabled": self.semantic_judge is not None,
            "scorer_prompt_version": getattr(self.semantic_judge, "prompt_version", None),
            "scorer_config_id": getattr(self.semantic_judge, "config_id", None),
        }
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Existing run metadata does not match this resume request")
        else:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        response_lines = _read_jsonl(self.responses_path)
        existing_responses = {}
        successful_responses = {}
        for item in response_lines:
            key = self._key(item)
            existing_responses[key] = item
            if item.get("execution_status") == "success":
                if key in successful_responses:
                    raise ValueError("Duplicate successful response records detected")
                successful_responses[key] = item
        tasks = [
            (question, mode) for question in self.rubric.questions for mode in self.modes
            if (question.question_id, mode) not in successful_responses
        ]
        score_lines = _read_jsonl(self.scores_path)
        existing_scores = {self._key(item): item for item in score_lines}
        if len(existing_scores) != len(score_lines):
            raise ValueError("Duplicate score records detected")
        question_map = {question.question_id: question for question in self.rubric.questions}
        queued = {item.get("review_id") for item in _read_jsonl(self.review_queue_path)}

        # Repair the narrow interruption window between a durable score append
        # and its corresponding review-queue append.
        for score_record in existing_scores.values():
            _append_review_if_needed(self.review_queue_path, score_record, queued)

        def persist_score(question: RubricQuestion, response_record: dict) -> None:
            key = self._key(response_record)
            if key in existing_scores or response_record["execution_status"] != "success":
                return
            score_record = self._score(question, response_record)
            _append_jsonl(self.scores_path, score_record)
            existing_scores[key] = score_record
            _append_review_if_needed(self.review_queue_path, score_record, queued)

        # Finish any scoring left incomplete by an earlier interruption before
        # issuing new question requests.
        for key, response_record in successful_responses.items():
            persist_score(question_map[key[0]], response_record)

        def persist_response(question: RubricQuestion, mode: str, record: dict) -> None:
            key = (question.question_id, mode)
            _append_jsonl(self.responses_path, record)
            existing_responses[key] = record
            if record["execution_status"] == "success":
                successful_responses[key] = record
                persist_score(question, record)

        if self.concurrency == 1:
            for question, mode in tasks:
                persist_response(question, mode, self._request_one(question, mode))
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._request_one, question, mode): (question, mode)
                    for question, mode in tasks
                }
                for future in as_completed(futures):
                    question, mode = futures[future]
                    persist_response(question, mode, future.result())

        summary = build_summary(list(existing_responses.values()), list(existing_scores.values()), metadata)
        write_reports(self.output_dir, list(existing_responses.values()), list(existing_scores.values()), summary)
        if self._owns_client:
            self.client.close()
        return summary


def _mode_metrics(responses: list[dict], scores: list[dict], planned: int) -> dict:
    success = [item for item in responses if item["execution_status"] == "success"]
    completed = [item for item in scores if item.get("scoring_status") == "completed"]
    valid = [item for item in completed if item.get("final_correctness") in {"correct", "partial", "incorrect"} and not item.get("needs_review")]
    completeness = [float(item["completeness_score"]) for item in valid if item.get("completeness_score") is not None]
    labels = Counter(item["final_correctness"] for item in valid)
    finalized_scores = [item for item in valid if item.get("earned_score") is not None and item.get("max_score") is not None]
    total_earned = sum(float(item["earned_score"]) for item in finalized_scores)
    total_max = sum(float(item["max_score"]) for item in finalized_scores)
    return {
        "planned_count": planned, "execution_success_count": len(success),
        "execution_success_rate": len(success) / planned if planned else None,
        "execution_failed_count": planned - len(success),
        "valid_final_decision_count": len(valid),
        "correct_count": labels["correct"], "correct_rate": labels["correct"] / len(valid) if valid else None,
        "partial_count": labels["partial"], "partial_rate": labels["partial"] / len(valid) if valid else None,
        "incorrect_count": labels["incorrect"], "incorrect_rate": labels["incorrect"] / len(valid) if valid else None,
        "mean_completeness": sum(completeness) / len(completeness) if completeness else None,
        "total_earned_score": total_earned,
        "total_max_score": total_max,
        "mean_earned_score": total_earned / len(finalized_scores) if finalized_scores else None,
        "score_denominator": len(finalized_scores),
        "completeness_denominator": len(completeness),
        "needs_review_count": sum(bool(item.get("needs_review")) for item in scores),
        "needs_review_rate": (sum(bool(item.get("needs_review")) for item in scores) / len(success) if success else None),
        "needs_review_denominator": len(success),
        "not_finalized_count": len(success) - len(valid),
    }


def build_summary(responses: list[dict], scores: list[dict], metadata: dict) -> dict:
    by_mode_responses: dict[str, list[dict]] = defaultdict(list)
    by_mode_scores: dict[str, list[dict]] = defaultdict(list)
    for item in responses:
        by_mode_responses[item["requested_mode"]].append(item)
    for item in scores:
        by_mode_scores[item["requested_mode"]].append(item)
    per_mode = {
        mode: _mode_metrics(by_mode_responses[mode], by_mode_scores[mode], metadata["question_count"])
        for mode in metadata["modes"]
    }
    return {
        "run_id": metadata["run_id"], "rubric_version": metadata["rubric_version"],
        "generated_at": _utc_now(),
        "overall": _mode_metrics(responses, scores, metadata["planned_count"]),
        "per_mode": per_mode,
        "denominator_note": "Execution failures are excluded from correctness. needs_review records are excluded from valid final decisions.",
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_reports(output_dir: Path, responses: list[dict], scores: list[dict], summary: dict) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_map = {(item["question_id"], item["requested_mode"]): item for item in scores}
    per_question = []
    for response in sorted(responses, key=lambda item: (item["question_id"], item["requested_mode"])):
        score = score_map.get((response["question_id"], response["requested_mode"]), {})
        per_question.append({
            "question_id": response["question_id"], "requested_mode": response["requested_mode"],
            "executed_mode": response.get("executed_mode"), "execution_status": response["execution_status"],
            "latency_ms": response["latency_ms"], "scoring_status": score.get("scoring_status"),
            "final_correctness": score.get("final_correctness"), "completeness_score": score.get("completeness_score"),
            "earned_score": score.get("earned_score"), "max_score": score.get("max_score"),
            "needs_review": score.get("needs_review"), "error": response.get("error"),
        })
    question_fields = ["question_id", "requested_mode", "executed_mode", "execution_status", "latency_ms", "scoring_status", "earned_score", "max_score", "final_correctness", "completeness_score", "needs_review", "error"]
    _write_csv(output_dir / "per_question.csv", per_question, question_fields)
    mode_rows = [{"mode": mode, **metrics} for mode, metrics in summary["per_mode"].items()]
    mode_fields = ["mode", "planned_count", "execution_success_count", "execution_success_rate", "execution_failed_count", "valid_final_decision_count", "total_earned_score", "total_max_score", "mean_earned_score", "score_denominator", "correct_count", "correct_rate", "partial_count", "partial_rate", "incorrect_count", "incorrect_rate", "mean_completeness", "completeness_denominator", "needs_review_count", "needs_review_rate", "needs_review_denominator", "not_finalized_count"]
    _write_csv(output_dir / "per_mode.csv", mode_rows, mode_fields)
    lines = [
        f"# Evaluation summary: {summary['run_id']}", "",
        f"Rubric: `{summary['rubric_version']}`", "",
        "Correctness denominators include only records with a valid final decision. Execution failures and `needs_review` records are reported separately.", "",
        "| Mode | Planned | Executed | Failed | Valid final | Score | Average | Correct | Partial | Incorrect | Mean completeness | Needs review |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in summary["per_mode"].items():
        mean = "n/a" if metrics["mean_completeness"] is None else f'{metrics["mean_completeness"]:.4f}'
        average = "n/a" if metrics["mean_earned_score"] is None else f'{metrics["mean_earned_score"]:.4f}'
        lines.append(f"| {mode} | {metrics['planned_count']} | {metrics['execution_success_count']} | {metrics['execution_failed_count']} | {metrics['valid_final_decision_count']} | {metrics['total_earned_score']}/{metrics['total_max_score']} | {average} | {metrics['correct_count']} | {metrics['partial_count']} | {metrics['incorrect_count']} | {mean} (n={metrics['completeness_denominator']}) | {metrics['needs_review_count']} (n={metrics['needs_review_denominator']}) |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _try_write_chart(output_dir, summary)


def _try_write_chart(output_dir: Path, summary: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    modes = list(summary["per_mode"])
    metrics = summary["per_mode"]
    labels = ["correct_rate", "partial_rate", "incorrect_rate", "mean_completeness"]
    x = range(len(modes)); width = 0.18
    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, label in enumerate(labels):
        values = [metrics[mode][label] or 0 for mode in modes]
        axis.bar([value + (offset - 1.5) * width for value in x], values, width, label=label)
    axis.set_xticks(list(x), [f"{mode}\nreview={metrics[mode]['needs_review_count']}" for mode in modes])
    axis.set_ylim(0, 1); axis.set_ylabel("Rate / score"); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "per_mode_metrics.png", dpi=150); plt.close(fig)


def select_questions(rubric: RubricDataset, question_ids: list[str] | None) -> RubricDataset:
    """Return the full rubric or a validated subset in the requested ID order."""
    if question_ids is None:
        return rubric
    duplicates = sorted(question_id for question_id, count in Counter(question_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {', '.join(duplicates)}")
    question_map = {question.question_id: question for question in rubric.questions}
    missing = [question_id for question_id in question_ids if question_id not in question_map]
    if missing:
        raise ValueError(f"Unknown question IDs: {', '.join(missing)}")
    selected = rubric.model_copy(deep=True)
    selected.questions = [question_map[question_id] for question_id in question_ids]
    selected.classification_counts = {
        group: sum(question.question_group == group for question in selected.questions)
        for group in ("Q", "K", "R")
    }
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--rubric", type=Path, default=project_root / "docs/evaluation/workbench/unit1_v1.0_rubric.json")
    parser.add_argument("--output-root", type=Path, default=project_root / "docs/evaluation/workbench/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--expected-questions", type=int, default=20)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Run only these validated formal question IDs, in the supplied order",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rubric = load_evaluation_dataset(args.rubric)
    try:
        rubric = select_questions(rubric, args.question_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(rubric.questions) != args.expected_questions:
        raise SystemExit(f"Expected {args.expected_questions} questions, found {len(rubric.questions)}")
    run_id = args.run_id or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    judge = None if args.skip_scoring else OpenAICompatibleSemanticJudge(retries=args.retries)
    runner = EvaluationRunner(
        rubric=rubric, output_dir=args.output_root / run_id, run_id=run_id,
        base_url=args.base_url, modes=args.modes, timeout=args.timeout,
        retries=args.retries, concurrency=args.concurrency, semantic_judge=judge,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Unit 1 v1.1 deterministic scoring overlay.

The v1.0 semantic judgment and answer text are immutable inputs.  This module
only maps those judgments to versioned deterministic rules; it never calls a
model and never edits the source score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .rubric_loader import load_rubric


class V11RuleError(ValueError):
    """Raised when a v1.1 rule set or source score cannot be applied safely."""


FINAL_STATUSES = {"full", "partial", "missing", "incorrect"}
MODEL_REVIEW_REASONS = (
    "major-error candidate",
    "contradictory statements",
    "lacks an answer quotation",
    "semantic judgment is uncertain",
)


def _quantize(value: float | int | Decimal, precision: int) -> float:
    quantum = Decimal(1).scaleb(-precision)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _normalise_boundary_text(value: str) -> str:
    return re.sub(r"[\s，。；：、！？,.!?;:\"'“”‘’（）()]+", "", value or "")


def load_v11_rules(path: Path, rubric_path: Path | None = None) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("rule_version") != "unit1_v1.1":
        raise V11RuleError("Expected rule_version unit1_v1.1")
    if config.get("base_rubric_version") != "unit1_v1.0":
        raise V11RuleError("v1.1 rules must declare base_rubric_version unit1_v1.0")
    confirmed_d = config.get("confirmed_rules", [])
    confirmed_non_d = config.get("confirmed_non_d_rules", [])
    manual = config.get("manual_rules", [])
    deferred_a = config.get("deferred_a_rules", [])
    preserved_a = config.get("preserved_existing_a_rules", [])
    if len(confirmed_d) != 16:
        raise V11RuleError("v1.1 must contain exactly 16 confirmed D rules")
    if len(confirmed_non_d) != 54:
        raise V11RuleError("v1.1 must contain exactly 54 confirmed A/B/C rules")
    if len(confirmed_d) + len(confirmed_non_d) != config.get("confirmed_rule_count"):
        raise V11RuleError("v1.1 confirmed rule count must be 70")
    actual_counts = {
        classification: sum(
            item.get("classification", "D") == classification
            for item in [*confirmed_non_d, *confirmed_d]
        )
        for classification in ("A", "B", "C", "D")
    }
    if actual_counts != config.get("confirmed_rule_counts"):
        raise V11RuleError(
            f"Confirmed rule class counts mismatch: {actual_counts}"
        )
    if len(manual) != config.get("manual_rule_count") or len(manual) != 22:
        raise V11RuleError("v1.1 must contain exactly 22 manual D rules")
    if len(deferred_a) != config.get("deferred_a_rule_count") or len(deferred_a) != 2:
        raise V11RuleError("v1.1 must contain exactly 2 deferred A rules")
    if len(preserved_a) != config.get("preserved_existing_a_rule_count") or len(preserved_a) != 15:
        raise V11RuleError("v1.1 must preserve exactly 15 existing A rules")
    confirmed_ids = [item["criterion_id"] for item in [*confirmed_non_d, *confirmed_d]]
    manual_ids = [item["criterion_id"] for item in manual]
    deferred_ids = [item["criterion_id"] for item in deferred_a]
    all_ids = confirmed_ids + manual_ids + deferred_ids + preserved_a
    if len(all_ids) != len(set(all_ids)):
        raise V11RuleError("Duplicate criterion_id in v1.1 configuration")
    if set(confirmed_ids) & set(manual_ids):
        raise V11RuleError("A D criterion cannot be both confirmed and manual")
    global_signoffs = config.get("global_rule_signoffs", {})
    if global_signoffs.get("G03") != "user_confirmed_v1.1" or global_signoffs.get("G04") != "user_confirmed_v1.1":
        raise V11RuleError("G03 and G04 must be signed off for this configuration")
    if any(global_signoffs.get(item) != "pending_not_signed_off" for item in ("G01", "G02", "G05")):
        raise V11RuleError("G01/G02/G05 must remain unsigned")
    if config.get("provenance", {}).get("formal_signoff_gap"):
        raise V11RuleError("The four formal signoff files are required for this configuration")
    if rubric_path is not None:
        validate_v11_rules_against_rubric(config, rubric_path)
        validate_signoff_provenance(config, rubric_path)
    return config


def _markdown_table_ids(path: Path) -> set[str]:
    return set(re.findall(r"^\|\s*(Q\d{3}-C\d{2})\s*\|", path.read_text(encoding="utf-8"), re.MULTILINE))


def validate_signoff_provenance(config: dict[str, Any], rubric_path: Path) -> None:
    project_root = rubric_path.resolve().parents[3]
    files = config.get("provenance", {}).get("signoff_files", [])
    if len(files) != 4:
        raise V11RuleError("Expected exactly four signoff files")
    resolved: dict[str, Path] = {}
    for item in files:
        signoff_path = project_root / item["path"]
        if not signoff_path.is_file():
            raise V11RuleError(f"Missing signoff file: {item['path']}")
        digest = hashlib.sha256(signoff_path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise V11RuleError(f"Signoff hash mismatch: {item['path']}")
        resolved[signoff_path.name] = signoff_path

    ab_path = next(path for name, path in resolved.items() if "AB_17" in name)
    c_csv_path = next(path for name, path in resolved.items() if "C类37" in name and path.suffix == ".csv")
    c_md_path = next(path for name, path in resolved.items() if "C类37" in name and path.suffix == ".md")
    d_path = next(path for name, path in resolved.items() if "d_16" in name)
    confirmed_non_d = config["confirmed_non_d_rules"]
    configured_ab = {
        item["criterion_id"] for item in confirmed_non_d if item["classification"] in {"A", "B"}
    } | {item["criterion_id"] for item in config["deferred_a_rules"]}
    if _markdown_table_ids(ab_path) != configured_ab:
        raise V11RuleError("A/B signoff IDs do not match the v1.1 configuration")
    configured_c = {
        item["criterion_id"] for item in confirmed_non_d if item["classification"] == "C"
    }
    with c_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        c_rows = list(csv.DictReader(handle))
    c_csv_ids = {row["id"] for row in c_rows}
    if len(c_rows) != 37 or c_csv_ids != configured_c or _markdown_table_ids(c_md_path) != configured_c:
        raise V11RuleError("C signoff Markdown/CSV/config IDs are inconsistent")
    if any(row["status"] != "user_confirmed_v1.1" for row in c_rows):
        raise V11RuleError("C signoff CSV contains an unsigned rule")
    configured_d = {item["criterion_id"] for item in config["confirmed_rules"]}
    if _markdown_table_ids(d_path) != configured_d:
        raise V11RuleError("D signoff IDs do not match the v1.1 configuration")


def validate_v11_rules_against_rubric(config: dict[str, Any], rubric_path: Path) -> None:
    rubric = load_rubric(rubric_path)
    if rubric.rubric_version != config["base_rubric_version"]:
        raise V11RuleError(
            f"Rule/rubric version mismatch: {config['base_rubric_version']} != {rubric.rubric_version}"
        )
    all_criteria = {
        criterion.criterion_id: criterion
        for question in rubric.questions
        for criterion in question.criteria
    }
    configured_scored = {
        item["criterion_id"]: item
        for item in [
            *config["confirmed_non_d_rules"],
            *config["confirmed_rules"],
            *config["manual_rules"],
            *config["deferred_a_rules"],
        ]
    }
    configured_ids = set(configured_scored) | set(config["preserved_existing_a_rules"])
    if configured_ids != set(all_criteria):
        missing = sorted(set(all_criteria) - configured_ids)
        extra = sorted(configured_ids - set(all_criteria))
        raise V11RuleError(f"Rule coverage mismatch; missing={missing}, extra={extra}")
    precision = int(config["score_precision"])
    for criterion_id, item in configured_scored.items():
        expected = _quantize(all_criteria[criterion_id].criterion_weight, precision)
        actual = _quantize(item["weight"], precision)
        if actual != expected:
            raise V11RuleError(
                f"Weight changed for {criterion_id}: configured={actual}, rubric={expected}"
            )
        declared_class = item.get("classification")
        if declared_class and declared_class != all_criteria[criterion_id].classification:
            raise V11RuleError(f"Classification changed for {criterion_id}")
        components = item.get("component_weights")
        if components and _quantize(sum(components), precision) != expected:
            raise V11RuleError(f"Component weights do not sum to {expected} for {criterion_id}")
    if any(all_criteria[item].classification != "A" for item in config["preserved_existing_a_rules"]):
        raise V11RuleError("Only A rules may be preserved as existing subweights")


def _base_review_requires_preservation(judgment: dict[str, Any]) -> str | None:
    reason = str(judgment.get("review_reason") or "")
    for marker in MODEL_REVIEW_REASONS:
        if marker in reason:
            return reason
    return None


def score_confirmed_rule(
    rule: dict[str, Any],
    judgment: dict[str, Any],
    *,
    answer_text: str,
    precision: int,
    major_error_confirmed: bool = False,
) -> dict[str, Any]:
    """Map one immutable semantic judgment through one signed v1.1 rule."""

    status = str(judgment.get("status") or "")
    evidence = [str(item) for item in judgment.get("evidence_quotes", [])]
    trace = {
        "criterion_id": rule["criterion_id"],
        "model_status": status,
        "evidence_quotes": evidence,
        "model_rationale": str(judgment.get("rationale") or ""),
        "rule_version": "unit1_v1.1",
        "rule_status": rule.get("approval_status", "user_confirmed_v1.1"),
        "scoring_policy": rule["scoring_policy"],
        "score": None,
        "decision": "pending",
        "needs_review": True,
        "reason": None,
    }
    if major_error_confirmed:
        trace.update(
            score=0.0,
            decision="incorrect",
            needs_review=False,
            reason="confirmed major factual error independently forces this criterion to zero",
        )
        return trace

    preserved = _base_review_requires_preservation(judgment)
    if preserved:
        trace["reason"] = f"preserved model review signal: {preserved}"
        return trace
    if status not in FINAL_STATUSES | {"uncertain"}:
        trace["reason"] = f"unsupported semantic status: {status!r}"
        return trace

    combined_text = "\n".join([answer_text, *evidence])
    context_phrase = rule.get("context_review_phrase")
    if context_phrase and context_phrase in combined_text:
        trace["reason"] = f"phrase {context_phrase!r} requires contextual human review"
        return trace
    manual_exact = rule.get("manual_review_exact_evidence")
    if manual_exact and any(
        _normalise_boundary_text(item) == _normalise_boundary_text(manual_exact)
        for item in evidence
    ):
        trace["reason"] = f"exact boundary {manual_exact!r} requires human review"
        return trace
    manual_contains = rule.get("manual_review_when_contains")
    manual_unless = rule.get("manual_review_unless_contains_any", [])
    if (
        manual_contains
        and manual_contains in combined_text
        and not any(item in combined_text for item in manual_unless)
    ):
        trace["reason"] = f"{manual_contains!r} without a legend qualifier requires human review"
        return trace

    weight = _quantize(rule["weight"], precision)
    if status == "full":
        trace.update(score=weight, decision="full", needs_review=False, reason="confirmed full boundary")
        return trace
    if status in {"missing", "incorrect"}:
        trace.update(score=0.0, decision=status, needs_review=False, reason="confirmed zero boundary")
        return trace
    if status == "uncertain":
        trace["reason"] = "uncertain semantic judgment cannot be terminally scored"
        return trace

    policy = rule["scoring_policy"]
    if policy == "strict_all_or_zero_review":
        marker_groups = rule.get("required_partial_markers", {})
        has_all_required_actions = bool(marker_groups) and all(
            any(marker in combined_text for marker in alternatives)
            for alternatives in marker_groups.values()
        )
        if has_all_required_actions:
            trace.update(
                score=_quantize(rule["partial_score"], precision),
                decision="partial",
                needs_review=False,
                reason="Q014-C05: both internal rule and external war are present; state conclusion is omitted",
            )
        else:
            trace.update(
                score=0.0,
                decision="missing_required_component",
                needs_review=False,
                reason="Q014-C05 strict rule: missing either internal rule or external war scores zero",
            )
    elif policy == "full_zero_partial_review":
        trace["reason"] = "partial score is not signed off and must be adjudicated"
    elif policy == "component_sum_review":
        trace["reason"] = "signed component weights require component-level evidence not present in the legacy judgment"
    elif policy == "full_special_partial_zero_review":
        special = rule["special_partial"]
        expected = _normalise_boundary_text(special["exact_evidence"])
        if any(_normalise_boundary_text(item) == expected for item in evidence):
            trace.update(
                score=_quantize(special["score"], precision),
                decision="special_partial",
                needs_review=False,
                reason="non-generalizable exact boundary example",
            )
        else:
            trace["reason"] = "partial does not match the non-generalizable special example"
    elif policy in {"full_half_zero_review", "full_half_zero_context_review"}:
        partial_score = rule.get("partial_score", Decimal(str(rule["weight"])) / Decimal("2"))
        trace.update(
            score=_quantize(partial_score, precision),
            decision="partial",
            needs_review=False,
            reason="confirmed partial boundary",
        )
    else:
        raise V11RuleError(f"Unknown scoring policy for {rule['criterion_id']}: {policy}")
    return trace


def score_confirmed_d_rule(
    rule: dict[str, Any],
    judgment: dict[str, Any],
    *,
    answer_text: str,
    precision: int,
    major_error_confirmed: bool = False,
) -> dict[str, Any]:
    """Backward-compatible name for callers and tests of the D-rule overlay."""

    return score_confirmed_rule(
        rule,
        judgment,
        answer_text=answer_text,
        precision=precision,
        major_error_confirmed=major_error_confirmed,
    )


def _label(score: float, *, major_error: bool, thresholds: dict[str, Any]) -> str:
    if major_error and thresholds.get("major_error_forces_incorrect", True):
        return "incorrect"
    if score >= float(thresholds["correct_min"]):
        return "correct"
    if score >= float(thresholds["partial_min"]):
        return "partial"
    return "incorrect"


def canonical_record_id(source_score: dict[str, Any]) -> str:
    return str(source_score.get("recovered_from_record_id") or source_score.get("record_id") or "")


def apply_v11_overlay(
    source_score: dict[str, Any],
    config: dict[str, Any],
    rubric_path: Path,
) -> dict[str, Any]:
    """Create one v1.1 result without mutating the source v1.0 score."""

    if source_score.get("scoring_status") != "completed":
        raise V11RuleError("Only completed source scores can be replayed")
    if source_score.get("rubric_version") != config["base_rubric_version"]:
        raise V11RuleError(
            f"Source score rubric version {source_score.get('rubric_version')!r} does not match "
            f"{config['base_rubric_version']!r}"
        )
    rubric = load_rubric(rubric_path)
    question_map = {item.question_id: item for item in rubric.questions}
    question_id = str(source_score["question_id"])
    if question_id not in question_map:
        raise V11RuleError(f"Unknown question_id: {question_id}")
    question = question_map[question_id]
    definition_map = {item.criterion_id: item for item in question.criteria}
    supplied = {item["criterion_id"]: item for item in source_score.get("criterion_results", [])}
    if len(supplied) != len(source_score.get("criterion_results", [])):
        raise V11RuleError(f"Duplicate source criterion result for {question_id}")
    if set(supplied) != set(definition_map):
        raise V11RuleError(f"Source criterion coverage mismatch for {question_id}")

    confirmed = {
        item["criterion_id"]: item
        for item in [*config["confirmed_non_d_rules"], *config["confirmed_rules"]]
    }
    manual = {item["criterion_id"]: item for item in config["manual_rules"]}
    deferred_a = {item["criterion_id"]: item for item in config["deferred_a_rules"]}
    preserved_a = set(config["preserved_existing_a_rules"])
    confirmed_error_ids = {
        criterion_id
        for item in source_score.get("major_error_confirmations", [])
        for criterion_id in item.get("affected_criterion_ids", [])
    }
    precision = int(config["score_precision"])
    criteria: list[dict[str, Any]] = []
    for definition in question.criteria:
        source = deepcopy(supplied[definition.criterion_id])
        common = {
            "criterion_id": definition.criterion_id,
            "max_score": _quantize(definition.criterion_weight, precision),
            "classification": definition.classification,
            "model_judgment": {
                "status": source.get("status"),
                "evidence_quotes": deepcopy(source.get("evidence_quotes", [])),
                "rationale": source.get("rationale", ""),
                "prompt_version": source_score.get("prompt_version"),
                "scoring_source": "existing_v1_v2_v3_semantic_judgment",
            },
            "original_v1_score": source,
        }
        if definition.criterion_id in confirmed:
            automatic = score_confirmed_rule(
                confirmed[definition.criterion_id],
                source,
                answer_text=str(source_score.get("answer_text") or ""),
                precision=precision,
                major_error_confirmed=definition.criterion_id in confirmed_error_ids,
            )
        elif definition.criterion_id in manual:
            automatic = {
                "criterion_id": definition.criterion_id,
                "model_status": source.get("status"),
                "evidence_quotes": deepcopy(source.get("evidence_quotes", [])),
                "model_rationale": source.get("rationale", ""),
                "rule_version": config["rule_version"],
                "rule_status": "pending_manual_adjudication",
                "scoring_policy": "manual_only",
                "score": None,
                "decision": "pending",
                "needs_review": True,
                "reason": "D criterion remains in the 22-item manual group",
            }
        elif definition.criterion_id in deferred_a:
            automatic = {
                "criterion_id": definition.criterion_id,
                "model_status": source.get("status"),
                "evidence_quotes": deepcopy(source.get("evidence_quotes", [])),
                "model_rationale": source.get("rationale", ""),
                "rule_version": config["rule_version"],
                "rule_status": "deferred_a_signoff",
                "scoring_policy": "manual_only",
                "score": None,
                "decision": "pending",
                "needs_review": True,
                "reason": deferred_a[definition.criterion_id]["reason"],
            }
        elif definition.criterion_id in preserved_a:
            score = source.get("earned_score")
            needs_review = bool(source.get("needs_review")) or score is None
            automatic = {
                "criterion_id": definition.criterion_id,
                "model_status": source.get("status"),
                "evidence_quotes": deepcopy(source.get("evidence_quotes", [])),
                "model_rationale": source.get("rationale", ""),
                "rule_version": config["rule_version"],
                "rule_status": definition.review_status,
                "scoring_policy": "preserve_v1_non_d",
                "score": _quantize(score, precision) if score is not None else None,
                "decision": source.get("status") if not needs_review else "pending",
                "needs_review": needs_review,
                "reason": source.get("review_reason") if needs_review else "preserved v1 deterministic result",
            }
        else:
            raise V11RuleError(f"No v1.1 rule coverage for {definition.criterion_id}")
        common["automatic"] = automatic
        common["final"] = {
            "score": automatic["score"] if not automatic["needs_review"] else None,
            "decision": automatic["decision"] if not automatic["needs_review"] else "pending",
            "source": "automatic" if not automatic["needs_review"] else "pending",
            "reason": automatic["reason"],
            "automatic_preserved": True,
            "difference_from_automatic": None,
        }
        criteria.append(common)

    pending = [item for item in criteria if item["final"]["score"] is None]
    known_scores = [item["final"]["score"] for item in criteria if item["final"]["score"] is not None]
    major_error = bool(source_score.get("major_error_detected"))
    final_score = None if pending else _quantize(sum(known_scores), precision)
    if final_score is not None and final_score > float(config["question_max_score"]):
        raise V11RuleError(f"Final score exceeds 10 for {canonical_record_id(source_score)}")
    final_label = (
        _label(final_score, major_error=major_error, thresholds=config["answer_label_thresholds"])
        if final_score is not None
        else None
    )
    result = {
        "schema_version": "historyai-unit1-v1.1-score-v1",
        "run_id": source_score.get("run_id"),
        "record_id": canonical_record_id(source_score),
        "question_id": question_id,
        "retrieval_mode": source_score.get("requested_mode"),
        "executed_mode": source_score.get("executed_mode"),
        "answer_text": source_score.get("answer_text", ""),
        "rule_version": config["rule_version"],
        "base_rubric_version": config["base_rubric_version"],
        "prompt_version": source_score.get("prompt_version"),
        "scorer_version": "deterministic-v1.1-overlay",
        "scoring_source": "offline_replay_existing_v1_v2_v3_judgments",
        "source_score_record_id": source_score.get("record_id"),
        "source_recovery_run_id": source_score.get("recovery_run_id"),
        "source_automatic_result_preserved": True,
        "criteria": criteria,
        "automatic_known_score": _quantize(sum(known_scores), precision),
        "automatic_scored_criterion_count": len(known_scores),
        "automatic_total_criterion_count": len(criteria),
        "automatic_score": final_score,
        "automatic_label": final_label,
        "final_score": final_score,
        "final_label": final_label,
        "scoring_completed": not pending,
        "pending_criterion_count": len(pending),
        "needs_review": bool(pending),
        "major_error_confirmed": major_error,
        "major_error_confirmations": deepcopy(source_score.get("major_error_confirmations", [])),
        "human_adjudication_count": 0,
        "human_confirmed_final": False,
        "formal_signoff_gap": bool(config["provenance"]["formal_signoff_gap"]),
    }
    return result

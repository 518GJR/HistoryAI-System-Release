"""Convert the frozen XLSX plus classification report into an audited JSON rubric."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from .models import (
    AtomicFact,
    CriterionDefinition,
    RubricDataset,
    RubricQuestion,
    ScoringOption,
    SourceReference,
)

RUBRIC_VERSION = "unit1_v1.0"
SCHEMA_VERSION = "evaluation-rubric-v1"
EXPECTED_HEADERS = [
    "question_id", "对应课次", "对应知识点 ID", "问题原文", "问题类型", "难度",
    "标准答案", "关键事实要点", "关键事实评分分值或权重", "重大事实错误及判错规则",
    "允许的等价表达", "教材依据与页码", "是否可直接进入评估集", "核验状态", "待人工核验事项",
]
CLASSIFICATION_COLUMNS = [
    "criterion_id", "source_coordinate", "original_criterion_text", "original_weight_text",
    "proposed_atomic_facts", "classification", "decision_boundary", "subweights_or_options",
    "scoring_method", "classification_reason", "frozen_basis", "major_error_or_qualifier",
    "pending_question", "machine_readiness", "review_status",
]


class RubricValidationError(ValueError):
    """Frozen input or converted rubric failed a structural check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise RubricValidationError(f"Invalid XLSX cell reference: {cell_ref}")
    result = 0
    for letter in letters.group(0):
        result = result * 26 + ord(letter) - 64
    return result


def _xlsx_rows(path: Path) -> list[list[str]]:
    """Read the first worksheet with stdlib only (no new XLSX dependency)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.findall(".//m:t", ns)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//m:sheetData/m:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "")
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//m:is/m:t", ns))
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text or "0")]
                else:
                    value = value_node.text or ""
                values[_column_number(ref)] = value
            if values:
                rows.append([values.get(index, "") for index in range(1, max(values) + 1)])
    return rows


def read_frozen_workbook(path: Path) -> list[dict[str, str]]:
    rows = _xlsx_rows(path)
    header_index = next((i for i, row in enumerate(rows) if row and row[0] == "question_id"), None)
    if header_index is None:
        raise RubricValidationError("Frozen workbook header row was not found")
    headers = rows[header_index]
    if headers != EXPECTED_HEADERS:
        raise RubricValidationError(f"Unexpected workbook headers: {headers!r}")
    records = []
    for row in rows[header_index + 1 :]:
        padded = row + [""] * (len(headers) - len(row))
        if padded[0]:
            records.append(dict(zip(headers, padded, strict=True)))
    if len(records) != 20:
        raise RubricValidationError(f"Expected 20 question rows, found {len(records)}")
    return records


def read_classification_report(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    marker = "## 6. Q001—Q020完整评分点分类表"
    if marker not in text:
        raise RubricValidationError("Classification table section was not found")
    section = text.split(marker, 1)[1].split("## 7.", 1)[0]
    records = []
    for line in section.splitlines():
        if not line.startswith("| Q"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(CLASSIFICATION_COLUMNS) or "-C" not in cells[0]:
            continue
        record = dict(zip(CLASSIFICATION_COLUMNS, cells, strict=True))
        record["classification"] = record["classification"].replace("**", "")
        record["scoring_method"] = record["scoring_method"].strip("`")
        record["review_status"] = record["review_status"].strip("`")
        records.append(record)
    if len(records) != 109:
        raise RubricValidationError(f"Expected 109 classification records, found {len(records)}")
    return records


def _weight_from_text(text: str) -> float:
    match = re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]?\s*(\d+(?:\.\d+)?)分", text)
    if not match:
        raise RubricValidationError(f"Cannot parse criterion weight: {text}")
    return float(match.group(1))


def _strip_number(text: str) -> str:
    return re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", text).strip()


def _parse_explicit_atomic_facts(record: dict[str, str]) -> list[AtomicFact]:
    facts = []
    pattern = r"(?:^|；)([^；]+?)（(\d+(?:\.\d+)?)分）"
    for index, match in enumerate(re.finditer(pattern, record["proposed_atomic_facts"]), start=1):
        facts.append(AtomicFact(
            fact_id=f'{record["criterion_id"]}-F{index:02d}',
            description=match.group(1).strip(),
            weight=float(match.group(2)),
            approval_status="source_verified",
        ))
    if not facts:
        raise RubricValidationError(f'Cannot parse explicit subweights for {record["criterion_id"]}')
    return facts


def _parse_pending_atomic_facts(record: dict[str, str]) -> list[AtomicFact]:
    proposal = record["subweights_or_options"].split("合计=", 1)[0]
    facts = []
    for part in proposal.split("；"):
        if "=" not in part:
            continue
        description, proposed_weight = part.rsplit("=", 1)
        facts.append(AtomicFact(
            fact_id=f'{record["criterion_id"]}-F{len(facts) + 1:02d}',
            description=description.strip(),
            weight=None,
            proposed_weight=proposed_weight.strip(),
            approval_status="pending_owner_signoff",
        ))
    return facts


def _scoring_options(record: dict[str, str]) -> list[ScoringOption]:
    if record["classification"] != "B":
        return []
    first, separator, remainder = record["subweights_or_options"].partition("方案二：")
    if not separator:
        raise RubricValidationError(f'B options missing for {record["criterion_id"]}')
    return [
        ScoringOption(option_id=f'{record["criterion_id"]}-OPTION-EQUAL', description=first.strip()),
        ScoringOption(option_id=f'{record["criterion_id"]}-OPTION-WHOLE', description=(separator + remainder).strip()),
    ]


def _manifest_entries(manifest_path: Path, project_root: Path) -> list[SourceReference]:
    text = manifest_path.read_text(encoding="utf-8")
    pattern = re.compile(r"\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|")
    references = []
    for relative, expected_size, expected_hash in pattern.findall(text):
        source = project_root / Path(relative)
        if not source.exists() or source.stat().st_size != int(expected_size) or _sha256(source) != expected_hash:
            raise RubricValidationError(f"Frozen file checksum mismatch: {relative}")
        references.append(SourceReference(source_file=relative, sha256=expected_hash))
    if len(references) != 4:
        raise RubricValidationError("Freeze manifest did not contain four controlled files")
    return references


def build_rubric(project_root: Path, *, generated_at: datetime | None = None) -> RubricDataset:
    release_dir = project_root / "docs/evaluation/releases/unit1_v1.0"
    workbook_path = release_dir / "unit1_candidate_questions.xlsx"
    classification_path = project_root / "docs/evaluation/project_audit/unit1_scoring_point_classification.md"
    manifest_path = release_dir / "FREEZE_MANIFEST.md"
    source_files = _manifest_entries(manifest_path, project_root)
    workbook_sha = _sha256(workbook_path)
    classification_sha = _sha256(classification_path)
    source_files.extend([
        SourceReference(source_file="docs/evaluation/project_audit/unit1_scoring_point_classification.md", section="§6 Q001—Q020完整评分点分类表", sha256=classification_sha),
        SourceReference(source_file="docs/evaluation/releases/unit1_v1.0/FREEZE_MANIFEST.md", sha256=_sha256(manifest_path)),
    ])

    workbook_records = read_frozen_workbook(workbook_path)
    classification_records = read_classification_report(classification_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in classification_records:
        grouped.setdefault(record["criterion_id"].split("-", 1)[0], []).append(record)

    questions = []
    for row_number, source_record in enumerate(workbook_records, start=5):
        question_id = source_record["question_id"]
        descriptions = [line.strip() for line in source_record["关键事实要点"].splitlines() if line.strip()]
        weights = [line.strip() for line in source_record["关键事实评分分值或权重"].splitlines() if line.strip()]
        classified = grouped.get(question_id, [])
        if not (len(descriptions) == len(weights) == len(classified)):
            raise RubricValidationError(f"{question_id} count mismatch: H={len(descriptions)}, I={len(weights)}, classification={len(classified)}")
        criteria = []
        for index, (description, weight_text, record) in enumerate(zip(descriptions, weights, classified, strict=True), start=1):
            expected_id = f"{question_id}-C{index:02d}"
            if record["criterion_id"] != expected_id:
                raise RubricValidationError(f"Expected {expected_id}, found {record['criterion_id']}")
            if description != record["original_criterion_text"] or weight_text != record["original_weight_text"]:
                raise RubricValidationError(f"Source text changed for {expected_id}")
            method = record["scoring_method"]
            atomic_facts = (
                _parse_explicit_atomic_facts(record) if method == "explicit_subfacts"
                else _parse_pending_atomic_facts(record) if method == "equal_subfacts_pending_signoff"
                else []
            )
            is_pending = record["classification"] in {"B", "D"} or "pending" in record["review_status"]
            criteria.append(CriterionDefinition(
                criterion_id=expected_id,
                criterion_description=_strip_number(description),
                criterion_weight=_weight_from_text(weight_text),
                original_criterion_text=description,
                original_weight_text=weight_text,
                classification=record["classification"],
                scoring_method=method,
                atomic_facts=atomic_facts,
                scoring_options=_scoring_options(record),
                allowed_equivalences=[],
                disallowed_equivalences=[],
                equivalence_rules_raw=source_record["允许的等价表达"],
                major_error_rules=[source_record["重大事实错误及判错规则"]],
                source_references=[
                    SourceReference(source_file="docs/evaluation/releases/unit1_v1.0/unit1_candidate_questions.xlsx", sheet="候选题", cells=[f"H{row_number}", f"I{row_number}", f"J{row_number}", f"K{row_number}", f"L{row_number}"], sha256=workbook_sha),
                    SourceReference(source_file="docs/evaluation/project_audit/unit1_scoring_point_classification.md", section=f"§6 {question_id} / {expected_id}", sha256=classification_sha),
                ],
                review_status=record["review_status"],
                blocking_reason=record["pending_question"] if is_pending else None,
                classification_record=record,
            ))
        questions.append(RubricQuestion(
            question_id=question_id,
            question_text=source_record["问题原文"], standard_answer=source_record["标准答案"],
            rubric_version=RUBRIC_VERSION, criteria=criteria,
            textbook_references=[source_record["教材依据与页码"]], lesson=source_record["对应课次"],
            knowledge_point_ids=[item.strip() for item in re.split(r"[、,，]", source_record["对应知识点 ID"]) if item.strip()],
            question_type=source_record["问题类型"], difficulty=source_record["难度"],
            original_major_error_text=source_record["重大事实错误及判错规则"],
            original_equivalence_text=source_record["允许的等价表达"],
            original_source_record=source_record,
        ))

    dataset = RubricDataset(
        schema_version=SCHEMA_VERSION, rubric_version=RUBRIC_VERSION,
        source_release="docs/evaluation/releases/unit1_v1.0",
        generated_at=generated_at or datetime.now(timezone.utc), source_files=source_files,
        questions=questions,
        classification_counts=dict(Counter(record["classification"] for record in classification_records)),
        conversion_status="partial_pending_human_signoff",
        conversion_notes=[
            "明确子权重的A类可确定性计分。",
            "仅建议等权拆分的A类保留 proposed_weight，正式 weight 为空。",
            "B类保留两种未选定方案。",
            "C类整体结构保留，partial=50%仍待R02统一约定签核。",
            "D类保留顶层原始文本和权重，不拆分、不分配子权重。",
        ],
    )
    validate_rubric(dataset, expected_question_count=20)
    return dataset


def validate_rubric(dataset: RubricDataset, *, expected_question_count: int | None = None) -> list[dict[str, object]]:
    anomalies: list[str] = []
    question_ids = [question.question_id for question in dataset.questions]
    if len(question_ids) != len(set(question_ids)):
        anomalies.append("duplicate question_id")
    if expected_question_count is not None and len(question_ids) != expected_question_count:
        anomalies.append(f"expected {expected_question_count} questions, found {len(question_ids)}")
    actual_classifications = Counter(
        criterion.classification
        for question in dataset.questions
        for criterion in question.criteria
    )
    if dict(actual_classifications) != dataset.classification_counts:
        anomalies.append(
            f"classification counts changed: declared={dataset.classification_counts}, "
            f"actual={dict(actual_classifications)}"
        )
    criterion_ids: list[str] = []
    per_question = []
    for question in dataset.questions:
        local_ids = [criterion.criterion_id for criterion in question.criteria]
        criterion_ids.extend(local_ids)
        original_weight_sum = round(sum(_weight_from_text(c.original_weight_text) for c in question.criteria), 8)
        weight_sum = round(sum(c.criterion_weight for c in question.criteria), 8)
        local_anomalies = []
        if not question.question_text or not question.standard_answer:
            local_anomalies.append("missing question or standard answer")
        if not question.criteria:
            local_anomalies.append("missing criteria")
        if question.question_weight != 10:
            local_anomalies.append(f"question weight {question.question_weight} != 10")
        if len(local_ids) != len(set(local_ids)):
            local_anomalies.append("duplicate criterion_id")
        if any(not identifier.startswith(f"{question.question_id}-C") for identifier in local_ids):
            local_anomalies.append("criterion_id does not map to question_id")
        if weight_sum != question.question_weight:
            local_anomalies.append(f"weight sum {weight_sum} != question weight {question.question_weight}")
        if original_weight_sum != weight_sum:
            local_anomalies.append(f"original weight sum {original_weight_sum} != JSON weight sum {weight_sum}")
        for criterion in question.criteria:
            if criterion.classification == "D" and criterion.atomic_facts:
                local_anomalies.append(f"{criterion.criterion_id} D class was split")
            if criterion.classification == "B" and any(option.selected for option in criterion.scoring_options):
                local_anomalies.append(f"{criterion.criterion_id} B option was selected")
            if criterion.scoring_method == "explicit_subfacts":
                child_sum = round(sum(fact.weight or 0 for fact in criterion.atomic_facts), 8)
                if child_sum != criterion.criterion_weight:
                    local_anomalies.append(f"{criterion.criterion_id} explicit subweight sum changed")
            if criterion.scoring_method == "equal_subfacts_pending_signoff" and any(fact.weight is not None for fact in criterion.atomic_facts):
                local_anomalies.append(f"{criterion.criterion_id} pending equal split became operative")
        per_question.append({
            "question_id": question.question_id,
            "criterion_count": len(question.criteria),
            "original_weight_sum": original_weight_sum,
            "json_weight_sum": weight_sum,
            "pending_review_count": sum(c.review_status != "source_verified" for c in question.criteria),
            "anomalies": local_anomalies,
        })
        anomalies.extend(f"{question.question_id}: {item}" for item in local_anomalies)
    if len(criterion_ids) != len(set(criterion_ids)):
        anomalies.append("duplicate criterion_id across questions")
    if anomalies:
        raise RubricValidationError("; ".join(anomalies))
    return per_question


def write_rubric_and_audit(project_root: Path, output_path: Path, audit_path: Path) -> tuple[RubricDataset, dict[str, object]]:
    dataset = build_rubric(project_root)
    checks = validate_rubric(dataset, expected_question_count=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8")); existing.pop("generated_at", None)
        candidate = dataset.model_dump(mode="json"); candidate.pop("generated_at", None)
        if existing != candidate:
            raise FileExistsError(f"Refusing to overwrite a different rubric: {output_path}")
    else:
        output_path.write_text(json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_criteria = [criterion for question in dataset.questions for criterion in question.criteria]
    audit = {
        "rubric_version": dataset.rubric_version,
        "question_count": len(dataset.questions),
        "original_criterion_count": len(all_criteria), "json_criterion_count": len(all_criteria),
        "classification_counts": dataset.classification_counts,
        "b_criteria": {"count": sum(c.classification == "B" for c in all_criteria), "all_options_preserved": all(len(c.scoring_options) == 2 and not any(o.selected for o in c.scoring_options) for c in all_criteria if c.classification == "B")},
        "d_criteria": {"count": sum(c.classification == "D" for c in all_criteria), "all_blocked_without_subweights": all(c.review_status.startswith("blocked_pending_adjudication") and not c.atomic_facts for c in all_criteria if c.classification == "D")},
        "pending_equal_a_operational_weight_count": sum(1 for c in all_criteria if c.scoring_method == "equal_subfacts_pending_signoff" for fact in c.atomic_facts if fact.weight is not None),
        "omissions": [], "duplicates": [], "weight_changes": [], "unauthorized_rule_additions": [],
        "per_question": checks, "structure_acceptance": "passed",
        "formal_scoring_readiness": "partial_pending_human_signoff",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")) != audit:
        raise FileExistsError(f"Refusing to overwrite a different audit report: {audit_path}")
    if not audit_path.exists():
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dataset, audit


def load_rubric(path: Path) -> RubricDataset:
    dataset = RubricDataset.model_validate_json(path.read_text(encoding="utf-8"))
    validate_rubric(dataset)
    return dataset

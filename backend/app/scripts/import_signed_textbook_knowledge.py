"""Import signed textbook points into an explicitly selected database.

This command never uses the application's default production DATABASE_URL.
Example: python -m app.scripts.import_signed_textbook_knowledge --database-url sqlite:///tmp/rehearsal.db
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session

from app.models import (
    SignedTextbookKnowledge,
    SignedTextbookKnowledgeSource,
    SignedTextbookSourceFragment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = PROJECT_ROOT / "docs/evaluation/workbench/unit5-20-formalization-20261030"
STAGING = WORKBENCH / "staging"
POINTS_FILE = STAGING / "seventh_grade_history_retrieval_repaired_338.json"
SOURCES_FILE = STAGING / "seventh_grade_history_sources_repaired.json"
DEFAULT_PREP_DIR = WORKBENCH / "formal-import-20261101/preflight"
PENDING_KNOWLEDGE_ID = "L16-Q015"
ALLOWED_RELATION_STATUSES = {"active", "pending_manual_source_review"}
SIGNED_TABLES = {
    SignedTextbookKnowledge.__tablename__,
    SignedTextbookSourceFragment.__tablename__,
    SignedTextbookKnowledgeSource.__tablename__,
}


def load_repaired_bundle(points_file=POINTS_FILE, sources_file=SOURCES_FILE, bundle_file=None):
    """Load and validate a signed corpus without writing a database.

    ``bundle_file`` supports incremental candidates containing knowledge points,
    sources, and associations in one JSON document. The original two-file 338
    corpus remains the default for backward compatibility.
    """
    if bundle_file:
        payload = json.loads(Path(bundle_file).read_text(encoding="utf-8-sig"))
        point_payload = payload
        source_payload = payload
    else:
        point_payload = json.loads(Path(points_file).read_text(encoding="utf-8-sig"))
        source_payload = json.loads(Path(sources_file).read_text(encoding="utf-8-sig"))
    points = point_payload["knowledge_points"]
    sources = source_payload["sources"]
    associations = source_payload["associations"]
    knowledge_ids = [str(row.get("knowledge_id") or "").strip() for row in points]
    if not points or any(not value for value in knowledge_ids) or len(set(knowledge_ids)) != len(points):
        raise ValueError("knowledge_id values must be non-empty and unique")
    if any(not str(row.get("content") or "").strip() for row in points):
        raise ValueError("Every signed point must retain complete non-empty content")
    source_map = {row["fragment_id"]: row for row in sources}
    fragment_ids = [str(row.get("fragment_id") or "").strip() for row in sources]
    if not sources or any(not value for value in fragment_ids) or len(source_map) != len(sources):
        raise ValueError("fragment_id values must be non-empty and unique")
    stable_source_ids = [str(row.get("source_id") or "").strip() for row in sources]
    if any(not value for value in stable_source_ids) or len(set(stable_source_ids)) != len(stable_source_ids):
        raise ValueError("source_id values must be non-empty and unique")

    point_ids = {row["knowledge_id"] for row in points}
    expanded = []
    active_by_point = defaultdict(set)
    pending_by_point = defaultdict(set)
    for relation in associations:
        knowledge_id = relation["knowledge_id"]
        fragment_id = relation["fragment_id"]
        source_ids = relation.get("source_ids") or []
        status = relation["relation_status"]
        if status not in ALLOWED_RELATION_STATUSES:
            raise ValueError(f"Unsupported relation_status: {status}")
        if knowledge_id not in point_ids or fragment_id not in source_map:
            raise ValueError(f"Broken association: {knowledge_id}/{fragment_id}")
        if not source_ids:
            raise ValueError(f"Association lacks parent source_id: {knowledge_id}/{fragment_id}")
        if not set(source_ids) <= set(source_map[fragment_id].get("parent_source_ids") or []):
            raise ValueError(f"Association source_id mismatch: {knowledge_id}/{fragment_id}")
        target = active_by_point if status == "active" else pending_by_point
        target[knowledge_id].add(fragment_id)
        for source_id in source_ids:
            expanded.append(
                {
                    "knowledge_id": knowledge_id,
                    "fragment_id": fragment_id,
                    "source_id": source_id,
                    "relation_status": status,
                }
            )
    keys = {(r["knowledge_id"], r["fragment_id"], r["source_id"]) for r in expanded}
    if len(keys) != len(expanded):
        raise ValueError("Duplicate knowledge_id/fragment_id/source_id association")

    eligible_ids = {r["knowledge_id"] for r in points if r["retrieval_status"] == "eligible"}
    if set(active_by_point) != eligible_ids:
        raise ValueError("Active source coverage must match the eligible knowledge points")
    if set(active_by_point) | set(pending_by_point) != point_ids:
        raise ValueError("Every knowledge point must have an active or pending association")
    associated_fragments = {row["fragment_id"] for row in expanded}
    if associated_fragments != set(source_map):
        raise ValueError("Every source fragment must participate in an association")
    if PENDING_KNOWLEDGE_ID in point_ids and PENDING_KNOWLEDGE_ID in active_by_point:
        raise ValueError(f"{PENDING_KNOWLEDGE_ID} must not have an active source")
    for point in points:
        knowledge_id = point["knowledge_id"]
        active_refs = {r["fragment_id"] for r in point.get("source_refs") or []}
        pending_refs = {r["fragment_id"] for r in point.get("pending_source_refs") or []}
        if active_refs != active_by_point.get(knowledge_id, set()):
            raise ValueError(f"Active source_refs mismatch: {knowledge_id}")
        if pending_refs != pending_by_point.get(knowledge_id, set()):
            raise ValueError(f"Pending source_refs mismatch: {knowledge_id}")
    return {
        "points": points,
        "sources": sources,
        "associations": associations,
        "expanded_associations": expanded,
        "active_by_point": active_by_point,
        "pending_by_point": pending_by_point,
        "bundle_file": Path(bundle_file) if bundle_file else None,
    }


def _point_trace(point: dict, relations: list[dict]) -> dict:
    active_refs = []
    for reference in point.get("source_refs") or []:
        normalized = dict(reference)
        normalized["locator"] = ""
        active_refs.append(normalized)
    source_ids = sorted(
        {
            relation["source_id"]
            for relation in relations
            if relation["knowledge_id"] == point["knowledge_id"]
        }
    )
    trace = dict(point)
    trace.update(
        {
            "content_status": (
                "signed_fact_summary"
                if point["retrieval_status"] == "eligible"
                else "signed_fact_summary_pending_manual_source_review"
            ),
            "source_fragment_ids": [r["fragment_id"] for r in active_refs],
            "source_excerpts": [r["excerpt"] for r in active_refs],
            "source_refs": active_refs,
            "source_ids": source_ids,
            "import_status": "formal_repaired_ready",
        }
    )
    return trace


def _knowledge_values(point: dict, bundle: dict) -> dict:
    trace = _point_trace(point, bundle["expanded_associations"])
    return {
        "lesson_number": point["lesson_number"],
        "title": point["title"],
        "content": point["content"],
        "content_status": trace["content_status"],
        "signoff_status": point["signoff_status"],
        "source_status": point["source_status"],
        "provenance_json": json.dumps(trace, ensure_ascii=False, sort_keys=True),
    }


def _fragment_values(source: dict) -> dict:
    return {
        "source_id": source["source_id"],
        "lesson_number": source["lesson_number"],
        "book_id": source["book_id"],
        "lesson_id": source["lesson_id"],
        "excerpt": source["excerpt"],
        "textbook_pages_json": json.dumps(source["textbook_pages"], ensure_ascii=False),
        "pdf_pages_json": json.dumps(source["pdf_pages"], ensure_ascii=False),
        "source_type": source["source_type"],
        "verification_status": source["verification_status"],
        "support_score": source.get("support_score"),
        "historical_source_anomaly": bool(source.get("historical_source_anomaly")),
        "metadata_json": json.dumps(source, ensure_ascii=False, sort_keys=True),
    }


def _row_matches(row, values: dict) -> bool:
    return all(getattr(row, field) == value for field, value in values.items())


def _assert_signed_tables_exist(engine) -> None:
    existing = set(inspect(engine).get_table_names())
    missing = sorted(SIGNED_TABLES - existing)
    if missing:
        raise ValueError(f"Signed textbook tables must exist before import: {missing}")


def _database_counts(session: Session) -> dict:
    return {
        "knowledge": session.scalar(select(func.count()).select_from(SignedTextbookKnowledge)),
        "fragments": session.scalar(select(func.count()).select_from(SignedTextbookSourceFragment)),
        "associations": session.scalar(select(func.count()).select_from(SignedTextbookKnowledgeSource)),
    }


def inspect_import_plan(database_url: str, bundle: dict) -> dict:
    """Build a SELECT-only import plan against the database's actual state."""
    if not database_url:
        raise ValueError("An explicit database URL is required")
    engine = create_engine(database_url)
    try:
        _assert_signed_tables_exist(engine)
        with Session(engine) as session:
            before = _database_counts(session)
            point_ids = [row["knowledge_id"] for row in bundle["points"]]
            fragment_ids = [row["fragment_id"] for row in bundle["sources"]]
            candidate_source_ids = [row["source_id"] for row in bundle["sources"]]
            existing_points = {
                row.knowledge_id: row
                for row in session.scalars(
                    select(SignedTextbookKnowledge).where(
                        SignedTextbookKnowledge.knowledge_id.in_(point_ids)
                    )
                ).all()
            }
            existing_fragments = {
                row.fragment_id: row
                for row in session.scalars(
                    select(SignedTextbookSourceFragment).where(
                        SignedTextbookSourceFragment.fragment_id.in_(fragment_ids)
                    )
                ).all()
            }
            source_id_owners = defaultdict(set)
            for row in session.scalars(
                select(SignedTextbookSourceFragment).where(
                    SignedTextbookSourceFragment.source_id.in_(candidate_source_ids)
                )
            ).all():
                source_id_owners[row.source_id].add(row.fragment_id)
            existing_relations = {
                (row.knowledge_id, row.fragment_id, row.source_id): row
                for row in session.scalars(
                    select(SignedTextbookKnowledgeSource).where(
                        SignedTextbookKnowledgeSource.knowledge_id.in_(point_ids)
                    )
                ).all()
            }
            l16_statuses = set(
                session.scalars(
                    select(SignedTextbookKnowledgeSource.relation_status).where(
                        SignedTextbookKnowledgeSource.knowledge_id == PENDING_KNOWLEDGE_ID
                    )
                ).all()
            )

            plan = {
                "database_before": before,
                "knowledge": {"inserted": 0, "unchanged": 0, "conflicts": []},
                "fragments": {"inserted": 0, "unchanged": 0, "conflicts": []},
                "associations": {"inserted": 0, "unchanged": 0, "conflicts": []},
                "l16_q015_relation_statuses": sorted(l16_statuses),
            }
            for point in bundle["points"]:
                knowledge_id = point["knowledge_id"]
                row = existing_points.get(knowledge_id)
                if row is None:
                    plan["knowledge"]["inserted"] += 1
                elif _row_matches(row, _knowledge_values(point, bundle)):
                    plan["knowledge"]["unchanged"] += 1
                else:
                    plan["knowledge"]["conflicts"].append(knowledge_id)
            for source in bundle["sources"]:
                fragment_id = source["fragment_id"]
                source_id = source["source_id"]
                foreign_owners = source_id_owners.get(source_id, set()) - {fragment_id}
                row = existing_fragments.get(fragment_id)
                if foreign_owners:
                    plan["fragments"]["conflicts"].append(
                        {"source_id": source_id, "owned_by_fragment_ids": sorted(foreign_owners)}
                    )
                elif row is None:
                    plan["fragments"]["inserted"] += 1
                elif _row_matches(row, _fragment_values(source)):
                    plan["fragments"]["unchanged"] += 1
                else:
                    plan["fragments"]["conflicts"].append(fragment_id)
            for relation in bundle["expanded_associations"]:
                key = (relation["knowledge_id"], relation["fragment_id"], relation["source_id"])
                row = existing_relations.get(key)
                if row is None:
                    plan["associations"]["inserted"] += 1
                elif row.relation_status == relation["relation_status"]:
                    plan["associations"]["unchanged"] += 1
                else:
                    plan["associations"]["conflicts"].append(
                        {"key": list(key), "existing": row.relation_status, "incoming": relation["relation_status"]}
                    )
            if "active" in l16_statuses:
                plan["global_conflict"] = f"{PENDING_KNOWLEDGE_ID} unexpectedly has an active relation"
            else:
                plan["global_conflict"] = None
            plan["database_expected_after"] = {
                "knowledge": before["knowledge"] + plan["knowledge"]["inserted"],
                "fragments": before["fragments"] + plan["fragments"]["inserted"],
                "associations": before["associations"] + plan["associations"]["inserted"],
            }
            plan["safe_to_apply"] = not (
                plan["knowledge"]["conflicts"]
                or plan["fragments"]["conflicts"]
                or plan["associations"]["conflicts"]
                or plan["global_conflict"]
            )
            return plan
    finally:
        engine.dispose()


def write_preflight_files(bundle: dict, output_dir=DEFAULT_PREP_DIR):
    """Write the 338-row UTF-8-SIG manifest, full relations, and validation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relations = bundle["expanded_associations"]
    active_sources = defaultdict(set)
    pending_sources = defaultdict(set)
    for relation in relations:
        target = active_sources if relation["relation_status"] == "active" else pending_sources
        target[relation["knowledge_id"]].add(relation["source_id"])
    manifest_rows = []
    for point in bundle["points"]:
        knowledge_id = point["knowledge_id"]
        manifest_rows.append(
            {
                "knowledge_id": knowledge_id,
                "lesson_number": point["lesson_number"],
                "title": point["title"],
                "content": point["content"],
                "signoff_status": point["signoff_status"],
                "source_status": point["source_status"],
                "retrieval_status": point["retrieval_status"],
                "content_status": (
                    "signed_fact_summary"
                    if point["retrieval_status"] == "eligible"
                    else "signed_fact_summary_pending_manual_source_review"
                ),
                "active_fragment_ids": ";".join(sorted(bundle["active_by_point"].get(knowledge_id, set()))),
                "pending_fragment_ids": ";".join(sorted(bundle["pending_by_point"].get(knowledge_id, set()))),
                "active_source_ids": ";".join(sorted(active_sources.get(knowledge_id, set()))),
                "pending_source_ids": ";".join(sorted(pending_sources.get(knowledge_id, set()))),
                "rag_included": "yes" if point["retrieval_status"] == "eligible" else "no",
                "expected_action": "insert",
            }
        )
    manifest_file = output_dir / "formal_signed_338_import.csv"
    with manifest_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    relation_file = output_dir / "formal_signed_338_source_associations.json"
    relation_file.write_text(
        json.dumps(
            {
                "sources": bundle["sources"],
                "associations": bundle["associations"],
                "expanded_associations": relations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lesson_counts = Counter(point["lesson_number"] for point in bundle["points"])
    lesson_eligible = Counter(
        point["lesson_number"]
        for point in bundle["points"]
        if point["retrieval_status"] == "eligible"
    )
    report = [
        "# 338条签核知识点正式导入前校验",
        "",
        "- knowledge_id：338条，全部唯一。",
        "- 完整事实内容：338条，均非空。",
        "- 可靠来源覆盖：337条；L16-Q015保留但不进入RAG。",
        f"- 稳定来源片段：{len(bundle['sources'])}条（活跃338条，待核验2条）。",
        f"- 知识点—片段关联：{len(bundle['associations'])}条。",
        f"- knowledge_id—fragment_id—source_id三元关联：{len(relations)}条。",
        "- 正式库当前无签核表，预计首次导入：知识点338、来源片段340、三元关联372；更新0。",
        "",
        "| 课次 | 知识点 | 可检索 | 待核验 |",
        "|---:|---:|---:|---:|",
    ]
    for lesson in range(5, 21):
        total = lesson_counts[lesson]
        eligible = lesson_eligible[lesson]
        report.append(f"| 第{lesson}课 | {total} | {eligible} | {total - eligible} |")
    report_file = output_dir / "formal_signed_338_preflight_report.md"
    report_file.write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest_file, relation_file, report_file


def import_signed_points(
    database_url: str,
    points_file=POINTS_FILE,
    sources_file=SOURCES_FILE,
    bundle_file=None,
    include_report=False,
):
    """Insert a validated bundle atomically; exact repeats are no-ops.

    Changed rows sharing an existing stable ID are treated as conflicts rather
    than silently overwriting signed data. Schema migration is deliberately not
    performed here.
    """
    if not database_url:
        raise ValueError("An explicit database URL is required")
    bundle = load_repaired_bundle(points_file, sources_file, bundle_file)
    precheck = inspect_import_plan(database_url, bundle)
    if not precheck["safe_to_apply"]:
        raise ValueError(f"Import precheck found conflicts: {json.dumps(precheck, ensure_ascii=False)}")
    engine = create_engine(database_url)
    counts = {
        "knowledge": {"inserted": 0, "updated": 0, "unchanged": 0},
        "fragments": {"inserted": 0, "updated": 0, "unchanged": 0},
        "associations": {"inserted": 0, "updated": 0, "unchanged": 0},
    }
    try:
        _assert_signed_tables_exist(engine)
        with Session(engine) as session, session.begin():
            for point in bundle["points"]:
                values = _knowledge_values(point, bundle)
                row = session.get(SignedTextbookKnowledge, point["knowledge_id"])
                if row is None:
                    session.add(SignedTextbookKnowledge(knowledge_id=point["knowledge_id"], **values))
                    counts["knowledge"]["inserted"] += 1
                else:
                    counts["knowledge"]["unchanged"] += 1
            session.flush()
            for source in bundle["sources"]:
                values = _fragment_values(source)
                row = session.get(SignedTextbookSourceFragment, source["fragment_id"])
                if row is None:
                    session.add(SignedTextbookSourceFragment(fragment_id=source["fragment_id"], **values))
                    counts["fragments"]["inserted"] += 1
                else:
                    counts["fragments"]["unchanged"] += 1
            session.flush()
            for relation in bundle["expanded_associations"]:
                key = (relation["knowledge_id"], relation["fragment_id"], relation["source_id"])
                row = session.get(SignedTextbookKnowledgeSource, key)
                if row is None:
                    session.add(SignedTextbookKnowledgeSource(**relation))
                    counts["associations"]["inserted"] += 1
                else:
                    counts["associations"]["unchanged"] += 1
            session.flush()
            if _database_counts(session) != precheck["database_expected_after"]:
                raise RuntimeError("In-transaction database counts do not match the preflight expectation")
            l16_statuses = set(
                session.scalars(
                    select(SignedTextbookKnowledgeSource.relation_status).where(
                        SignedTextbookKnowledgeSource.knowledge_id == PENDING_KNOWLEDGE_ID
                    )
                ).all()
            )
            if "active" in l16_statuses:
                raise RuntimeError(f"{PENDING_KNOWLEDGE_ID} unexpectedly gained an active relation")
    finally:
        engine.dispose()

    postcheck = inspect_import_plan(database_url, bundle)
    if postcheck["database_before"] != precheck["database_expected_after"]:
        raise RuntimeError("Post-import database counts do not match the preflight expectation")
    if not postcheck["safe_to_apply"] or any(
        postcheck[group]["inserted"] for group in ("knowledge", "fragments", "associations")
    ):
        raise RuntimeError("Post-import read-only verification failed")
    if include_report:
        return {"counts": counts, "precheck": precheck, "postcheck": postcheck}
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument("--points-file", type=Path, default=POINTS_FILE)
    parser.add_argument("--sources-file", type=Path, default=SOURCES_FILE)
    parser.add_argument(
        "--bundle-file",
        type=Path,
        help="Incremental JSON containing knowledge_points, sources, and associations",
    )
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREP_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    bundle = load_repaired_bundle(args.points_file, args.sources_file, args.bundle_file)
    files = [] if args.bundle_file else write_preflight_files(bundle, args.prepare_dir)
    result = {
        "knowledge_points": len(bundle["points"]),
        "source_fragments": len(bundle["sources"]),
        "association_rows": len(bundle["expanded_associations"]),
        "preflight_files": [str(path) for path in files],
        "database_applied": False,
    }
    if args.dry_run:
        if not args.database_url:
            raise ValueError("--database-url is required with --dry-run")
        result["dry_run"] = inspect_import_plan(args.database_url, bundle)
    if args.apply:
        if not args.database_url:
            raise ValueError("--database-url is required with --apply")
        result["import_result"] = import_signed_points(
            args.database_url,
            args.points_file,
            args.sources_file,
            args.bundle_file,
            include_report=True,
        )
        result["database_applied"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))

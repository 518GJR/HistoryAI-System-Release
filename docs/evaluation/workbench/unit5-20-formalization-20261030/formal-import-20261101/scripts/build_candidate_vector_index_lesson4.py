"""Build an isolated 389-vector candidate by appending only Lesson 4 vectors."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "backend/app/main.py").is_file())

def local_path(variable: str, default: Path) -> Path:
    value = Path(os.environ[variable]).expanduser() if os.getenv(variable) else default
    return (value if value.is_absolute() else ROOT / value).resolve()
BACKEND = ROOT / "backend"
FORMAL_JSON = local_path("HISTORYAI_FORMAL_JSON", ROOT / "data/textbook_history_knowledge.json")
FORMAL_INDEX = local_path("HISTORYAI_FORMAL_INDEX", ROOT / "data/vector_index/textbook_bge_small_zh_v1_5.npz")
BASE_DIR = ROOT / (
    "docs/evaluation/workbench/unit5-20-formalization-20261030/"
    "formal-import-20261101/candidate-vector-index"
)
BASE_INDEX = local_path("HISTORYAI_BASE_INDEX", BASE_DIR / "textbook_bge_small_zh_v1_5_candidate_fullbook.npz")
OUTPUT_DIR = ROOT / (
    "docs/evaluation/workbench/unit5-20-formalization-20261030/"
    "formal-import-20261101/candidate-vector-index-lesson4-20261102"
)
OUTPUT_DIR = local_path("HISTORYAI_CANDIDATE_OUTPUT_DIR", OUTPUT_DIR)
CANDIDATE_INDEX = OUTPUT_DIR / "textbook_bge_small_zh_v1_5_candidate_lessons1_20.npz"
FRAGMENT_MAP = OUTPUT_DIR / "fragment_vector_position_map.json"
ASSOCIATION_MAP = OUTPUT_DIR / "knowledge_source_association_map.json"
MANIFEST = OUTPUT_DIR / "candidate_index_build_manifest.json"
HASHES = OUTPUT_DIR / "candidate_index_sha256.json"
REPORT = OUTPUT_DIR / "candidate_vector_index_build_report.md"

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
FORMAL_JSON_SHA256 = "C7B206D12FE315664389A1516F8C822B62D90C32229836F9F5F2417D81B1EEB5"
FORMAL_INDEX_SHA256 = "5D66EBF1AB39DCAD631E230BF38B4ABC7B1CF43A379CA9F4988E32004370AAFA"
BASE_INDEX_SHA256 = "455BEF5746FAA8A7787EC3287D1450C685E8D28A77F3D922495A9A8D78A9DB76"
EXPECTED_DATABASE_COUNTS = {
    "signed_textbook_knowledge": 352,
    "signed_textbook_source_fragment": 356,
    "signed_textbook_knowledge_source": 392,
}
EXPECTED_RELATION_COUNTS = Counter({"active": 390, "pending_manual_source_review": 2})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def page_text(value: str) -> str:
    pages = json.loads(value)
    if not isinstance(pages, list) or not pages:
        raise ValueError("Page metadata must be a non-empty JSON list")
    return ";".join(str(page) for page in pages)


def load_npz(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        vectors = np.asarray(payload["vectors"], dtype=np.float32).copy()
        metadata = json.loads(str(payload["metadata"].item()))
    return vectors, metadata


def load_database_snapshot(database_url: str) -> dict:
    """Read the three signed tables using SELECT only."""
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            counts = {
                table: connection.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar_one()
                for table in (
                    "signed_textbook_knowledge",
                    "signed_textbook_source_fragment",
                    "signed_textbook_knowledge_source",
                )
            }
            knowledge = {
                row["knowledge_id"]: dict(row)
                for row in connection.execute(text(
                    "SELECT knowledge_id, lesson_number, title, content, content_status, "
                    "signoff_status, source_status, provenance_json "
                    "FROM signed_textbook_knowledge ORDER BY knowledge_id"
                )).mappings()
            }
            fragments = {
                row["fragment_id"]: dict(row)
                for row in connection.execute(text(
                    "SELECT fragment_id, source_id, lesson_number, book_id, lesson_id, "
                    "excerpt, textbook_pages_json, pdf_pages_json, source_type, "
                    "verification_status, support_score, historical_source_anomaly, metadata_json "
                    "FROM signed_textbook_source_fragment ORDER BY fragment_id"
                )).mappings()
            }
            associations = [
                dict(row)
                for row in connection.execute(text(
                    "SELECT knowledge_id, fragment_id, source_id, relation_status "
                    "FROM signed_textbook_knowledge_source "
                    "ORDER BY knowledge_id, fragment_id, source_id"
                )).mappings()
            ]
    finally:
        engine.dispose()
    return {
        "counts": counts,
        "knowledge": knowledge,
        "fragments": fragments,
        "associations": associations,
    }


def partition_associations(associations: list[dict]) -> tuple[list[dict], list[dict]]:
    active = [row for row in associations if row["relation_status"] == "active"]
    pending = [
        row for row in associations if row["relation_status"] == "pending_manual_source_review"
    ]
    if len(active) + len(pending) != len(associations):
        raise RuntimeError("Unexpected relation_status in formal database")
    return active, pending


def derive_incremental_documents(
    base_documents: list[dict],
    current_signed_documents: list[dict],
    formal_vector_count: int,
    active_fragment_ids: set[str],
    fragments: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Validate all reused documents and return exactly the new Lesson 4 documents."""
    base_ids = [document["source_id"] for document in base_documents]
    if len(base_ids) != len(set(base_ids)):
        raise RuntimeError("Base candidate contains duplicate fragment IDs")
    current_by_id = {document["source_id"]: document for document in current_signed_documents}
    if len(current_by_id) != len(current_signed_documents):
        raise RuntimeError("Current signed documents contain duplicate fragment IDs")
    if set(current_by_id) != active_fragment_ids:
        raise RuntimeError("Current retrieval documents do not exactly match active fragments")

    reused_signed = base_documents[formal_vector_count:]
    reused_signed_ids = {document["source_id"] for document in reused_signed}
    if not reused_signed_ids <= active_fragment_ids:
        raise RuntimeError("A reused signed fragment is no longer active")
    for old_document in reused_signed:
        fragment_id = old_document["source_id"]
        if current_by_id[fragment_id] != old_document:
            raise RuntimeError(f"Reused signed document metadata changed: {fragment_id}")

    new_ids = active_fragment_ids - reused_signed_ids
    if len(new_ids) != 16:
        raise RuntimeError(f"Expected exactly 16 new active fragments, got {len(new_ids)}")
    if any(fragments[fragment_id]["lesson_number"] != 4 for fragment_id in new_ids):
        raise RuntimeError("A newly encoded fragment is not from Lesson 4")
    new_documents = [
        document for document in current_signed_documents if document["source_id"] in new_ids
    ]
    if len(new_documents) != 16:
        raise RuntimeError("Lesson 4 document selection is incomplete")
    return reused_signed, new_documents


def combine_reused_and_new(
    base_vectors: np.ndarray,
    base_documents: list[dict],
    new_vectors: np.ndarray,
    new_documents: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    if base_vectors.ndim != 2 or new_vectors.ndim != 2:
        raise RuntimeError("Vector arrays must be two-dimensional")
    if len(base_vectors) != len(base_documents) or len(new_vectors) != len(new_documents):
        raise RuntimeError("Vector/document count mismatch")
    if base_vectors.shape[1] != new_vectors.shape[1]:
        raise RuntimeError("New vector dimension does not match reused vectors")
    documents = list(base_documents) + list(new_documents)
    ids = [document["source_id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Combined fragment IDs are not unique")
    return np.vstack([base_vectors, new_vectors]).astype(np.float32, copy=False), documents


def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["EMBEDDING_MODEL"] = MODEL_ID
    os.environ["EMBEDDING_MODEL_REVISION"] = MODEL_REVISION

    database_url = os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("Set SIGNED_TEXTBOOK_DATABASE_URL explicitly; this tool never reads a project .env")
    for label, path in (("HISTORYAI_FORMAL_JSON", FORMAL_JSON), ("HISTORYAI_FORMAL_INDEX", FORMAL_INDEX), ("HISTORYAI_BASE_INDEX", BASE_INDEX)):
        if not path.is_file():
            raise RuntimeError(f"Missing local input {label}; prepare legally held data/indexes before building")
    parsed = make_url(database_url)
    if (parsed.host, parsed.port or 3306, parsed.database) != ("localhost", 3306, "history_ai"):
        raise RuntimeError("DATABASE_URL does not target localhost:3306/history_ai")
    os.environ["SIGNED_TEXTBOOK_DATABASE_URL"] = database_url
    sys.path.insert(0, str(BACKEND))

    from app.services.signed_textbook_service import load_retrievable_signed_points
    from app.services.vector_service import (
        INDEX_FORMAT_VERSION,
        PREPROCESSING,
        _validate_token_lengths,
        _validate_vectors,
        build_source_documents,
        configured_model_id,
        configured_model_revision,
        get_embedding_model,
        textbook_data_sha256,
    )

    outputs = (CANDIDATE_INDEX, FRAGMENT_MAP, ASSOCIATION_MAP, MANIFEST, HASHES, REPORT)
    if any(path.exists() for path in outputs):
        raise RuntimeError("Lesson 4 candidate output already exists; refusing to overwrite")

    if sha256(FORMAL_JSON) != FORMAL_JSON_SHA256:
        raise RuntimeError("Formal first-unit JSON hash changed")
    if sha256(FORMAL_INDEX) != FORMAL_INDEX_SHA256:
        raise RuntimeError("Formal first-unit index hash changed")
    if sha256(BASE_INDEX) != BASE_INDEX_SHA256:
        raise RuntimeError("Validated 373-vector base candidate hash changed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    formal_vectors, formal_metadata = load_npz(FORMAL_INDEX)
    base_vectors, base_metadata = load_npz(BASE_INDEX)
    formal_vectors = _validate_vectors(formal_vectors, len(formal_metadata["documents"]))
    base_vectors = _validate_vectors(base_vectors, len(base_metadata["documents"]))
    if formal_vectors.shape != (35, 512) or base_vectors.shape != (373, 512):
        raise RuntimeError("Formal/base vector shape mismatch")
    if not np.array_equal(base_vectors[:35], formal_vectors):
        raise RuntimeError("Base candidate no longer preserves all 35 formal vectors")
    if base_metadata["documents"][:35] != formal_metadata["documents"]:
        raise RuntimeError("Base candidate first-unit metadata changed")
    for metadata in (formal_metadata, base_metadata):
        if metadata["model_id"] != MODEL_ID or metadata["model_revision"] != MODEL_REVISION:
            raise RuntimeError("Embedding model/revision mismatch")
        if metadata["preprocessing"] != PREPROCESSING or metadata["vector_dimension"] != 512:
            raise RuntimeError("Preprocessing or vector dimension mismatch")

    snapshot = load_database_snapshot(database_url)
    if snapshot["counts"] != EXPECTED_DATABASE_COUNTS:
        raise RuntimeError(f"Unexpected formal table counts: {snapshot['counts']}")
    relation_counts = Counter(row["relation_status"] for row in snapshot["associations"])
    if relation_counts != EXPECTED_RELATION_COUNTS:
        raise RuntimeError(f"Unexpected relation statuses: {dict(relation_counts)}")
    active, pending = partition_associations(snapshot["associations"])
    active_fragment_ids = {row["fragment_id"] for row in active}
    active_knowledge_ids = {row["knowledge_id"] for row in active}
    if len(active_fragment_ids) != 354 or len(active_knowledge_ids) != 351:
        raise RuntimeError("Unexpected active fragment/knowledge coverage")
    if "L13-Q011" not in active_knowledge_ids or "L16-Q015" in active_knowledge_ids:
        raise RuntimeError("Special knowledge-point active status is incorrect")

    signed_points = load_retrievable_signed_points()
    signed_documents = build_source_documents(signed_points)
    if {point["knowledge_id"] for point in signed_points} != active_knowledge_ids:
        raise RuntimeError("Retrieval view does not exactly match active knowledge IDs")
    _, lesson4_documents = derive_incremental_documents(
        base_metadata["documents"], signed_documents, len(formal_vectors),
        active_fragment_ids, snapshot["fragments"],
    )
    lesson4_ids = {document["source_id"] for document in lesson4_documents}
    lesson4_knowledge_ids = {
        row["knowledge_id"] for row in active if row["fragment_id"] in lesson4_ids
    }
    lesson4_relations = [row for row in active if row["fragment_id"] in lesson4_ids]
    if len(lesson4_knowledge_ids) != 14 or len(lesson4_relations) != 20:
        raise RuntimeError("Lesson 4 knowledge/association coverage mismatch")
    for document in lesson4_documents:
        fragment = snapshot["fragments"][document["source_id"]]
        if document["excerpt"] != fragment["excerpt"]:
            raise RuntimeError(f"Excerpt mismatch: {document['source_id']}")
        if document["textbook_pages"] != page_text(fragment["textbook_pages_json"]):
            raise RuntimeError(f"Textbook page mismatch: {document['source_id']}")
        if document["pdf_pages"] != page_text(fragment["pdf_pages_json"]):
            raise RuntimeError(f"PDF page mismatch: {document['source_id']}")

    embedding_model = get_embedding_model()
    token_stats = _validate_token_lengths(embedding_model, lesson4_documents)
    lesson4_vectors = embedding_model.encode(
        [document["embedding_text"] for document in lesson4_documents],
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    lesson4_vectors = _validate_vectors(lesson4_vectors, 16)
    combined_vectors, combined_documents = combine_reused_and_new(
        base_vectors, base_metadata["documents"], lesson4_vectors, lesson4_documents
    )
    combined_vectors = _validate_vectors(combined_vectors, len(combined_documents))
    if combined_vectors.shape != (389, 512):
        raise RuntimeError(f"Unexpected candidate shape: {combined_vectors.shape}")

    active_by_fragment: dict[str, list[dict]] = defaultdict(list)
    for row in active:
        active_by_fragment[row["fragment_id"]].append(row)
    fragment_rows = []
    for position, document in enumerate(combined_documents):
        if position < 35:
            corpus = "unit1_formal_reused"
        elif position < 373:
            corpus = "signed_lessons_5_20_reused"
        else:
            corpus = "signed_lesson_4_new"
        fragment_rows.append({
            "vector_position": position,
            "fragment_id": document["source_id"],
            "corpus": corpus,
            "lesson_number": document["lesson_number"],
            "textbook_pages": document["textbook_pages"],
            "pdf_pages": document["pdf_pages"],
        })

    unit1_relations = []
    for document in formal_metadata["documents"]:
        for knowledge_id in document.get("knowledge_ids") or []:
            unit1_relations.append({
                "knowledge_id": knowledge_id,
                "fragment_id": document["source_id"],
                "source_id": document.get("parent_source_id") or document["source_id"],
                "relation_status": "active",
                "corpus": "unit1_formal",
            })
    relation_payload = {
        "schema_version": "historyai-candidate-associations-v2",
        "active_associations": unit1_relations + active,
        "signed_active_associations": active,
        "excluded_pending_associations": pending,
        "signed_active_by_fragment": {
            fragment_id: rows for fragment_id, rows in sorted(active_by_fragment.items())
        },
    }

    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "candidate_schema": "historyai-lessons1-20-candidate-v2",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_id": configured_model_id(),
        "model_revision": configured_model_revision(),
        "model_max_tokens": token_stats["model_max_tokens"],
        "maximum_document_tokens": token_stats["maximum_document_tokens"],
        "vector_dimension": 512,
        "similarity": "cosine_via_dot_product",
        "normalization": "l2_unit_length",
        "preprocessing": PREPROCESSING,
        "textbook_data_sha256": textbook_data_sha256(),
        "vector_count": 389,
        "reused_candidate_vector_count": 373,
        "reused_formal_vector_count": 35,
        "newly_encoded_vector_count": 16,
        "coverage_lessons": list(range(1, 21)),
        "coverage_note": "Covers lessons 1-20; first 373 vectors are reused byte-for-byte.",
        "association_map_file": ASSOCIATION_MAP.name,
        "documents": combined_documents,
    }
    with tempfile.NamedTemporaryFile(
        prefix=".candidate-lesson4-", suffix=".npz", dir=OUTPUT_DIR, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(
            temporary_path,
            vectors=combined_vectors,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        os.replace(temporary_path, CANDIDATE_INDEX)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    json_dump(FRAGMENT_MAP, {
        "schema_version": "historyai-fragment-vector-map-v2",
        "vector_count": 389,
        "items": fragment_rows,
    })
    json_dump(ASSOCIATION_MAP, relation_payload)

    reloaded_vectors, reloaded_metadata = load_npz(CANDIDATE_INDEX)
    reloaded_vectors = _validate_vectors(reloaded_vectors, len(reloaded_metadata["documents"]))
    if not np.array_equal(reloaded_vectors[:373], base_vectors):
        raise RuntimeError("The 373 reused vectors changed after reload")
    if reloaded_metadata["documents"][:373] != base_metadata["documents"]:
        raise RuntimeError("The 373 reused document metadata rows changed")
    if {document["source_id"] for document in reloaded_metadata["documents"][373:]} != lesson4_ids:
        raise RuntimeError("Reloaded Lesson 4 fragment set mismatch")
    pending_only_ids = {row["fragment_id"] for row in pending} - active_fragment_ids
    if {document["source_id"] for document in combined_documents[35:]} & pending_only_ids:
        raise RuntimeError("Pending-only fragment entered the candidate index")

    lesson_counts = Counter(document["lesson_number"] for document in combined_documents)
    manifest = {
        "status": "passed",
        "built_at": metadata["built_at"],
        "inputs": {
            "formal_index": str(FORMAL_INDEX),
            "base_candidate_index": str(BASE_INDEX),
            "database": "mysql://localhost:3306/history_ai (credentials redacted)",
        },
        "input_sha256": {
            FORMAL_INDEX.name: sha256(FORMAL_INDEX),
            BASE_INDEX.name: sha256(BASE_INDEX),
        },
        "database_counts": snapshot["counts"],
        "relation_status_counts": dict(relation_counts),
        "active_knowledge_count": len(active_knowledge_ids),
        "active_signed_fragment_count": len(active_fragment_ids),
        "signed_active_association_count": len(active),
        "excluded_pending_association_count": len(pending),
        "reused_candidate_vectors": 373,
        "reused_formal_vectors": 35,
        "newly_encoded_lesson4_vectors": 16,
        "candidate_vector_count": 389,
        "vector_dimension": 512,
        "lesson4_knowledge_count": len(lesson4_knowledge_ids),
        "lesson4_fragment_count": len(lesson4_ids),
        "lesson4_active_association_count": len(lesson4_relations),
        "lesson_vector_counts": {str(key): value for key, value in sorted(lesson_counts.items())},
        "coverage_lessons": list(range(1, 21)),
        "L13_Q011_active": "L13-Q011" in active_knowledge_ids,
        "L16_Q015_indexed": "L16-Q015" in active_knowledge_ids,
        "base_373_vectors_exact_after_reload": True,
        "formal_35_vectors_exact_after_reload": True,
        "candidate_reload_passed": True,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dimension": 512,
            "preprocessing": PREPROCESSING,
            "normalization": "l2_unit_length",
        },
    }
    json_dump(MANIFEST, manifest)

    report_lines = [
        "# 第4课增量候选向量索引构建报告",
        "",
        f"构建时间：{metadata['built_at']}。状态：通过。未覆盖正式索引或旧373候选索引。",
        "",
        "## 构建结果",
        "",
        "- 复用旧候选向量：373条，重新加载后逐行完全一致。",
        "- 其中第一单元正式向量：35条，与正式NPZ逐行完全一致。",
        "- 第4课新编码active唯一片段：16条。",
        "- 新候选索引：389×512。",
        "- 正式签核知识点：352条；具有active关联：351条。",
        "- 正式来源片段：356条；active唯一片段：354条。",
        "- 三元关联：392条，其中active 390、pending 2。",
        "- 第4课：14条知识点、16个片段、20条active关系。",
        "- L13-Q011的active来源已纳入；L16-Q015未进入有效索引。",
        "",
        "## 模型与安全",
        "",
        f"模型 `{MODEL_ID}`，revision `{MODEL_REVISION}`；512维、L2归一化。仅第4课16个片段在Hugging Face离线模式下使用本地模型编码。正式MySQL仅执行SELECT。",
        "",
        "## 课程覆盖",
        "",
        "候选索引覆盖第1—20课。向量以fragment_id去重；390条active三元关系保存在独立映射中，2条pending关系仅保存在排除清单。",
        "",
        "| 课次 | 向量数 |",
        "|---:|---:|",
    ]
    report_lines.extend(
        f"| 第{lesson}课 | {lesson_counts[lesson]} |" for lesson in sorted(lesson_counts)
    )
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    hash_payload = {
        "inputs_unchanged": {
            str(FORMAL_INDEX): sha256(FORMAL_INDEX),
            str(BASE_INDEX): sha256(BASE_INDEX),
        },
        "candidate_outputs": {
            path.name: sha256(path)
            for path in (CANDIDATE_INDEX, FRAGMENT_MAP, ASSOCIATION_MAP, MANIFEST, REPORT)
        },
    }
    json_dump(HASHES, hash_payload)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Cannot build local candidate: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

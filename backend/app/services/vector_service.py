"""Local dense retrieval over deduplicated textbook source excerpts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEXTBOOK_DATA_FILE = PROJECT_ROOT / "data" / "textbook_history_knowledge.json"
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "data" / "vector_index" / "textbook_bge_small_zh_v1_5.npz"
)
DEFAULT_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
INDEX_FORMAT_VERSION = "historyai-textbook-vector-v1"
PREPROCESSING = {
    "document_template": "书名：{book_title}\n第{lesson_number}课：{lesson_title}\n小标题：{section_title}\n教材原文：{excerpt}",
    "document_prefix": "",
    "query_prefix": QUERY_PREFIX,
    "whitespace": "strip-fields-preserve-excerpt",
    "chunking": "none-one-deduplicated-source-per-vector",
}


class VectorIndexError(RuntimeError):
    """The vector runtime or index is unavailable, invalid, or incompatible."""


_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


_FACET_LABELS = {
    "definition": "定义",
    "historical_status": "历史地位",
    "time": "时间 年代",
    "founding_agent": "建立者 人物",
    "reason": "原因",
    "result_impact": "影响 作用",
    "content": "内容",
}


def _query_subject(clause: str, previous: str) -> str:
    quoted = re.search(r"[“\"]([^”\"]{2,24})[”\"]", clause)
    if quoted:
        return quoted.group(1).strip()
    if clause.startswith(("它", "其", "该")):
        return previous
    match = re.match(
        r"(.{2,30}?)(?=是什么|是指什么|指什么|的?历史地位|地位如何|"
        r"于哪一年|在哪一年|何时|什么时候|的?建立者|建立者|创立者|"
        r"为什么|为何|的?原因|的?影响|有何影响|有什么影响)",
        clause,
    )
    if match:
        subject = match.group(1).strip(" ，,：:")
        if subject not in {"它", "其", "该", "建立者"}:
            return subject
    return previous


def _clause_aspects(clause: str) -> list[str]:
    aspects: list[tuple[int, str]] = []

    def add(aspect: str, cues: tuple[str, ...]) -> None:
        positions = [clause.find(cue) for cue in cues if cue in clause]
        if positions:
            aspects.append((min(positions), aspect))

    if "是什么" in clause:
        add("definition", ("是什么",))
    elif "指什么" in clause or "是指" in clause:
        add("content", ("指什么", "是指"))
    add("historical_status", ("历史地位", "文字史上的地位", "地位如何"))
    add("time", ("哪一年", "何时", "什么时候", "年代", "时间"))
    add("founding_agent", ("建立者", "创立者", "创建者", "谁建立", "谁创立"))
    add("reason", ("为什么", "为何", "原因", "背景"))
    add("result_impact", ("影响", "作用", "意义", "结果", "价值"))
    if not aspects:
        return ["content"]
    return [aspect for _, aspect in sorted(dict.fromkeys(aspects))]


def decompose_vector_query(question: str) -> list[dict[str, str]]:
    """Split only the user's wording into bounded semantic retrieval facets."""
    original = question.strip()
    if not original:
        return []
    clauses = [
        value.strip(" ，,：:")
        for value in re.split(r"[？?。；;！!]", original)
        if value.strip(" ，,：:")
    ]
    if not clauses:
        clauses = [original]

    queries: list[dict[str, str]] = []
    previous_subject = ""
    for clause in clauses:
        subject = _query_subject(clause, previous_subject)
        if subject:
            previous_subject = subject
        resolved = clause
        if subject and clause.startswith(("它", "其", "该")):
            resolved = re.sub(r"^(?:它|其|该)(?:在|的)?", f"{subject} ", clause, count=1)
        elif subject and clause.startswith(("建立者", "创立者", "创建者")):
            resolved = f"{subject} {clause}"
        for aspect in _clause_aspects(clause):
            query = resolved
            label = _FACET_LABELS[aspect]
            if label not in query and aspect in {"reason", "result_impact"}:
                query = f"{query} {label}"
            entry = {"aspect": aspect, "query": query.strip(), "subject": subject}
            if entry not in queries:
                queries.append(entry)
    return queries or [{"aspect": "content", "query": original, "subject": ""}]


def _document_supports_subquestion(document: dict, subquestion: dict) -> bool:
    from app.services.support_assessment import _evidence_supports_subject_aspect

    return _evidence_supports_subject_aspect(
        str(subquestion.get("subject") or ""),
        subquestion["aspect"],
        str(document.get("excerpt") or ""),
    )


def configured_model_id() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def configured_model_revision() -> str:
    return (
        os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_MODEL_REVISION).strip()
        or DEFAULT_MODEL_REVISION
    )


def configured_index_file() -> Path:
    configured = os.getenv("VECTOR_INDEX_PATH", "").strip()
    if not configured:
        return DEFAULT_INDEX_FILE
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def textbook_data_sha256(path: Path | None = None) -> str:
    source_path = path or TEXTBOOK_DATA_FILE
    try:
        digest = hashlib.sha256(source_path.read_bytes())
        if os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip():
            from app.services.signed_textbook_service import load_retrievable_signed_points
            signed = load_retrievable_signed_points()
            digest.update(json.dumps(signed, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()
    except OSError as exc:
        raise VectorIndexError(
            f"Cannot read textbook data for hashing: {source_path}"
        ) from exc


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_source_documents(knowledge_items: list[dict]) -> list[dict]:
    """Deduplicate by source_id and merge knowledge-point associations.

    Conflicting excerpt or source metadata is rejected rather than silently
    choosing one version.
    """
    documents: dict[str, dict] = {}
    for item in knowledge_items:
        knowledge_id = str(item.get("knowledge_id") or "").strip()
        if not knowledge_id:
            raise VectorIndexError("Knowledge point without knowledge_id")
        for reference in item.get("source_refs") or []:
            if not isinstance(reference, dict):
                raise VectorIndexError(f"Invalid source reference on {knowledge_id}")
            source_id = str(reference.get("source_id") or "").strip()
            excerpt = str(reference.get("excerpt") or "").strip()
            if not source_id or not excerpt:
                raise VectorIndexError(f"Incomplete source reference on {knowledge_id}")

            source_metadata = {
                "source_id": source_id,
                "parent_source_id": source_id,
                "fragment_id": source_id,
                "fragment_start": 0,
                "fragment_end": len(excerpt),
                "excerpt": excerpt,
                "book_id": str(item.get("book_id") or ""),
                "book_title": str(item.get("book_title") or ""),
                "unit_id": str(item.get("unit_id") or ""),
                "unit_title": str(item.get("unit_title") or ""),
                "lesson_id": str(item.get("lesson_id") or reference.get("lesson_id") or ""),
                "lesson_number": item.get("lesson_number"),
                "lesson_title": str(item.get("lesson_title") or ""),
                "section_id": str(reference.get("section_id") or ""),
                "section_title": str(item.get("section_title") or ""),
                "material_type": str(reference.get("material_type") or ""),
                "textbook_pages": str(reference.get("textbook_pages") or ""),
                "pdf_pages": str(reference.get("pdf_pages") or ""),
                "locator": str(reference.get("locator") or ""),
            }
            existing = documents.get(source_id)
            if existing is None:
                documents[source_id] = {
                    **source_metadata,
                    "knowledge_ids": [knowledge_id],
                    "knowledge_titles": [str(item.get("title") or "")],
                    "periods": [str(item.get("dynasty") or "")],
                    "qualifiers": list(item.get("qualifiers") or []),
                    "certainty_notes": [str(item.get("certainty") or "")],
                }
                continue

            for field, value in source_metadata.items():
                if existing[field] != value:
                    raise VectorIndexError(
                        f"Conflicting source metadata for {source_id}: {field}"
                    )
            existing["knowledge_ids"].append(knowledge_id)
            existing["knowledge_titles"].append(str(item.get("title") or ""))
            existing["periods"].append(str(item.get("dynasty") or ""))
            existing["qualifiers"].extend(item.get("qualifiers") or [])
            existing["certainty_notes"].append(str(item.get("certainty") or ""))

    result = []
    for document in documents.values():
        for field in (
            "knowledge_ids",
            "knowledge_titles",
            "periods",
            "qualifiers",
            "certainty_notes",
        ):
            document[field] = _stable_unique(document[field])
        document["embedding_text"] = PREPROCESSING["document_template"].format(
            **document
        )
        result.append(document)
    return result


def get_embedding_model():
    """Load one CPU model instance per configured id/revision per process."""
    key = (configured_model_id(), configured_model_revision())
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                key[0],
                revision=key[1],
                device="cpu",
                trust_remote_code=False,
            )
        except Exception as exc:
            raise VectorIndexError(
                f"Failed to load local embedding model {key[0]} at {key[1]}: {exc}"
            ) from exc
        _MODEL_CACHE[key] = model
        return model


def _validate_token_lengths(model, documents: list[dict]) -> dict:
    tokenizer = getattr(model, "tokenizer", None)
    max_length = int(getattr(model, "max_seq_length", 0) or 0)
    if tokenizer is None or max_length <= 0:
        raise VectorIndexError("Embedding model does not expose tokenizer/max_seq_length")
    maximum = 0
    for document in documents:
        token_ids = tokenizer(
            document["embedding_text"], add_special_tokens=True, truncation=False
        )["input_ids"]
        token_count = len(token_ids)
        maximum = max(maximum, token_count)
        if token_count > max_length:
            raise VectorIndexError(
                f"Source {document['source_id']} has {token_count} tokens, exceeding "
                f"model limit {max_length}; explicit chunking is required"
            )
    return {"model_max_tokens": max_length, "maximum_document_tokens": maximum}


def _validate_vectors(vectors: np.ndarray, expected_count: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != expected_count or values.shape[1] <= 0:
        raise VectorIndexError(
            f"Invalid vector shape {values.shape}; expected {expected_count} rows"
        )
    if not np.isfinite(values).all():
        raise VectorIndexError("Vector index contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise VectorIndexError("Vector index contains zero-length vectors")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise VectorIndexError("Vector index rows are not L2 normalized")
    return values


def build_vector_index(index_file: Path | None = None, model=None) -> dict:
    """Build and atomically replace the local source-level vector index."""
    from app.services.rag_service import load_textbook_knowledge_from_json

    if os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip() and index_file is None:
        raise VectorIndexError("Signed textbook staging requires an explicit index_file")
    target = index_file or configured_index_file()
    points = load_textbook_knowledge_from_json()
    signed_audit = None
    if os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip():
        from app.services.signed_textbook_service import load_retrievable_signed_corpus
        signed_points, signed_audit = load_retrievable_signed_corpus()
        points.extend(signed_points)
    documents = build_source_documents(points)
    embedding_model = model or get_embedding_model()
    token_stats = _validate_token_lengths(embedding_model, documents)
    try:
        vectors = embedding_model.encode(
            [document["embedding_text"] for document in documents],
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise VectorIndexError(f"Failed to encode textbook sources: {exc}") from exc
    vectors = _validate_vectors(vectors, len(documents))

    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_id": configured_model_id(),
        "model_revision": configured_model_revision(),
        "model_max_tokens": token_stats["model_max_tokens"],
        "maximum_document_tokens": token_stats["maximum_document_tokens"],
        "vector_dimension": int(vectors.shape[1]),
        "similarity": "cosine_via_dot_product",
        "normalization": "l2_unit_length",
        "preprocessing": PREPROCESSING,
        "textbook_data_sha256": textbook_data_sha256(),
        "vector_count": int(vectors.shape[0]),
        "documents": documents,
    }
    if signed_audit is not None:
        metadata["signed_source_audit"] = {
            **signed_audit,
            "final_vector_count": int(vectors.shape[0]),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.", suffix=".npz", dir=target.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            vectors=vectors,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        os.replace(temporary_path, target)
    except OSError as exc:
        raise VectorIndexError(f"Failed to atomically write vector index: {target}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return metadata


def load_vector_index(index_file: Path | None = None) -> tuple[np.ndarray, dict]:
    """Load and fully validate an index against current data and configuration."""
    target = index_file or configured_index_file()
    if not target.exists():
        raise VectorIndexError(
            f"Vector index not found: {target}. Run the explicit build command first"
        )
    try:
        with np.load(target, allow_pickle=False) as payload:
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            metadata = json.loads(str(payload["metadata"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise VectorIndexError(f"Vector index is unreadable or corrupt: {target}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("documents"), list):
        raise VectorIndexError("Vector index metadata is invalid")

    expected = {
        "format_version": INDEX_FORMAT_VERSION,
        "model_id": configured_model_id(),
        "model_revision": configured_model_revision(),
        "preprocessing": PREPROCESSING,
        "textbook_data_sha256": textbook_data_sha256(),
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise VectorIndexError(
            "Vector index does not match current configuration/data "
            f"({', '.join(mismatches)}); rebuild it"
        )
    vectors = _validate_vectors(vectors, len(metadata["documents"]))
    if metadata.get("vector_count") != vectors.shape[0]:
        raise VectorIndexError("Vector count metadata does not match stored vectors")
    if metadata.get("vector_dimension") != vectors.shape[1]:
        raise VectorIndexError("Vector dimension metadata does not match stored vectors")
    source_ids = [document.get("source_id") for document in metadata["documents"]]
    if any(not source_id for source_id in source_ids) or len(source_ids) != len(set(source_ids)):
        raise VectorIndexError("Vector document metadata has missing/duplicate source IDs")
    return vectors, metadata


def _document_to_knowledge_item(
    document: dict, score: float, signed_fragment: dict | None = None
) -> dict:
    associations = (
        list(signed_fragment.get("active_associations") or [])
        if signed_fragment is not None
        else []
    )
    knowledge_ids = _stable_unique(
        [relation["knowledge_id"] for relation in associations]
        or list(document["knowledge_ids"])
    )
    titles = _stable_unique(
        [relation.get("knowledge_title", "") for relation in associations]
        or list(document["knowledge_titles"])
    )
    excerpt = (
        str(signed_fragment.get("excerpt") or "")
        if signed_fragment is not None
        else document["excerpt"]
    )
    material_type = str(
        document.get("material_type")
        or (signed_fragment.get("material_type") if signed_fragment is not None else "")
        or ""
    )
    textbook_pages = (
        str(signed_fragment.get("textbook_pages") or "")
        if signed_fragment is not None
        else document["textbook_pages"]
    )
    pdf_pages = (
        str(signed_fragment.get("pdf_pages") or "")
        if signed_fragment is not None
        else document["pdf_pages"]
    )
    if associations:
        source_refs = [
            {
                "knowledge_id": relation["knowledge_id"],
                "fragment_id": relation["fragment_id"],
                "source_id": relation["source_id"],
                "lesson_id": signed_fragment["lesson_id"],
                "section_id": document["section_id"],
                "material_type": material_type,
                "textbook_pages": textbook_pages,
                "pdf_pages": pdf_pages,
                "excerpt": excerpt,
                "verification_status": signed_fragment["verification_status"],
                "locator": document["locator"],
            }
            for relation in associations
        ]
    else:
        source_refs = [
            {
                "knowledge_id": knowledge_ids[0] if knowledge_ids else "",
                "fragment_id": document["source_id"],
                "source_id": document["source_id"],
                "lesson_id": document["lesson_id"],
                "section_id": document["section_id"],
                "material_type": material_type,
                "textbook_pages": textbook_pages,
                "pdf_pages": pdf_pages,
                "excerpt": excerpt,
                "verification_status": "",
                "locator": document["locator"],
            }
        ]
    return {
        "knowledge_id": knowledge_ids[0] if knowledge_ids else "",
        "knowledge_ids": knowledge_ids,
        "title": " / ".join(titles) or document["locator"],
        "dynasty": " / ".join(document["periods"]),
        "time": "",
        "people": [],
        "event": excerpt,
        "reason": "",
        "impact": "",
        "content": excerpt,
        "keywords": [],
        "matched_terms": [],
        "score": float(score),
        "retrieval_mode": "vector",
        "retrieval_unit": "source_excerpt",
        "book_id": document["book_id"],
        "book_title": document["book_title"],
        "unit_id": document["unit_id"],
        "unit_title": document["unit_title"],
        "lesson_id": document["lesson_id"],
        "lesson_number": document["lesson_number"],
        "lesson_title": document["lesson_title"],
        "section_title": document["section_title"],
        "material_types": [material_type],
        "textbook_pages": textbook_pages,
        "pdf_pages": pdf_pages,
        "qualifiers": list(document["qualifiers"]),
        "evidence_type": "textbook_excerpt",
        "certainty": " / ".join(document["certainty_notes"]),
        "source_refs": source_refs,
    }


def retrieve_textbook_vectors(question: str, top_k: int = 3) -> list[dict]:
    """Retrieve each question facet independently, then fairly merge by source."""
    query = question.strip()
    limit = max(0, int(top_k))
    if not query or limit == 0:
        return []
    vectors, metadata = load_vector_index()
    model = get_embedding_model()
    subquestions = decompose_vector_query(query)
    query_texts = [f"{QUERY_PREFIX}{item['query']}" for item in subquestions]
    tokenizer = getattr(model, "tokenizer", None)
    max_length = int(getattr(model, "max_seq_length", 0) or 0)
    if tokenizer is None or max_length <= 0:
        raise VectorIndexError("Embedding model does not expose tokenizer/max_seq_length")
    for query_text in query_texts:
        token_count = len(
            tokenizer(query_text, add_special_tokens=True, truncation=False)["input_ids"]
        )
        if token_count > max_length:
            raise VectorIndexError(
                f"Query has {token_count} tokens, exceeding model limit {max_length}"
            )
    try:
        query_vectors = model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise VectorIndexError(f"Failed to encode query: {exc}") from exc
    query_vectors = _validate_vectors(query_vectors, len(query_texts))
    score_rows = query_vectors @ vectors.T
    facet_validity = [
        [
            _document_supports_subquestion(document, subquestion)
            for document in metadata["documents"]
        ]
        for subquestion in subquestions
    ]
    ranked = [
        np.asarray(
            sorted(
                range(len(row)),
                key=lambda index: (
                    facet_validity[facet_index][index],
                    float(row[index]),
                ),
                reverse=True,
            ),
            dtype=np.int64,
        )
        for facet_index, row in enumerate(score_rows)
    ]

    # Round-robin over facet rankings. A strong neighbor for one facet cannot
    # consume the complete result budget before the other facets are visited.
    selected: dict[int, dict] = {}
    depth = 0
    document_count = len(metadata["documents"])
    while len(selected) < min(limit, document_count) and depth < document_count:
        for facet_index, order in enumerate(ranked):
            document_index = int(order[depth])
            entry = selected.get(document_index)
            if entry is None:
                if len(selected) >= min(limit, document_count):
                    continue
                entry = {
                    "score": float(score_rows[facet_index, document_index]),
                    "matched_subquestions": [],
                }
                selected[document_index] = entry
            entry["score"] = max(
                entry["score"], float(score_rows[facet_index, document_index])
            )
            aspect = subquestions[facet_index]["aspect"]
            if (
                facet_validity[facet_index][document_index]
                and aspect not in entry["matched_subquestions"]
            ):
                entry["matched_subquestions"].append(aspect)
        depth += 1

    selected_documents = [
        (metadata["documents"][index], entry)
        for index, entry in selected.items()
    ]
    signed_fragments = {}
    if os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip():
        from app.services.signed_textbook_service import (
            load_active_fragment_associations,
        )

        signed_fragments = load_active_fragment_associations(
            [document["source_id"] for document, _ in selected_documents]
        )

    results = []
    for document, selection in selected_documents:
        fragment_id = document["source_id"]
        signed_fragment = signed_fragments.get(fragment_id)
        if signed_fragment is not None and not signed_fragment["active_associations"]:
            continue
        result = _document_to_knowledge_item(
            document, selection["score"], signed_fragment
        )
        result["matched_subquestions"] = selection["matched_subquestions"]
        results.append(result)
    return results

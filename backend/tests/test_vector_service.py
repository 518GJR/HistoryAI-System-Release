import json

import numpy as np
import pytest

from app.services import rag_service, vector_service


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": list(range(max(1, len(text) // 2)))}


class FakeEmbeddingModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 512

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            if "灰烬、烧石和烧骨" in text or (
                text.startswith(vector_service.QUERY_PREFIX) and "火" in text
            ):
                rows.append([1.0, 0.0, 0.0])
            elif "北京人" in text:
                rows.append([0.8, 0.6, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        values = np.asarray(rows, dtype=np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values


def test_source_documents_deduplicate_and_preserve_associations(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    items = rag_service.load_textbook_knowledge_from_json()

    documents = vector_service.build_source_documents(items)

    assert len(documents) == 35
    assert len({document["source_id"] for document in documents}) == 35
    shared = next(d for d in documents if d["source_id"] == "H7A-P004-A")
    assert len(shared["knowledge_ids"]) == 4
    assert shared["excerpt"].startswith("化石是研究古人类历史")
    assert shared["textbook_pages"] == "4"
    assert shared["pdf_pages"] == "9"
    assert "教材原文：" in shared["embedding_text"]


def test_source_document_conflict_fails_explicitly(monkeypatch):
    items = rag_service.load_textbook_knowledge_from_json()
    conflicting = dict(items[0])
    conflicting["knowledge_id"] = "CONFLICT"
    conflicting["source_refs"] = [dict(items[0]["source_refs"][0])]
    conflicting["source_refs"][0]["excerpt"] = "不一致原文"

    with pytest.raises(vector_service.VectorIndexError, match="Conflicting source"):
        vector_service.build_source_documents([items[0], conflicting])


def test_mock_index_build_load_and_vector_ranking(monkeypatch, tmp_path):
    index_file = tmp_path / "index.npz"
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("VECTOR_INDEX_PATH", str(index_file))
    monkeypatch.setattr(vector_service, "get_embedding_model", lambda: FakeEmbeddingModel())

    metadata = vector_service.build_vector_index(index_file, FakeEmbeddingModel())
    results = rag_service.retrieve_history_knowledge("北京人用火的意义是什么？")

    assert metadata["vector_count"] == 35
    assert metadata["vector_dimension"] == 3
    assert metadata["maximum_document_tokens"] <= metadata["model_max_tokens"]
    assert results[0]["source_refs"][0]["source_id"] == "H7A-P006-B"
    assert results[0]["retrieval_mode"] == "vector"
    assert results[0]["retrieval_unit"] == "source_excerpt"
    assert len(results) == 3


def test_vector_results_batch_expand_active_source_relations_and_skip_pending(monkeypatch):
    documents = [
        {
            "source_id": "H7A-P006-B",
            "knowledge_ids": ["H7A-L01-010"],
            "knowledge_titles": ["北京人用火"],
            "excerpt": "第一单元教材原文。",
            "book_id": "H7A",
            "book_title": "中国历史",
            "unit_id": "H7A-U01",
            "unit_title": "第一单元",
            "lesson_id": "H7A-L01",
            "lesson_number": 1,
            "lesson_title": "第1课",
            "section_id": "S-1",
            "section_title": "北京人",
            "material_type": "教材正文",
            "textbook_pages": "6",
            "pdf_pages": "11",
            "locator": "",
            "periods": [],
            "qualifiers": [],
            "certainty_notes": [],
        },
        {
            "source_id": "F-SHARED",
            "knowledge_ids": ["STALE-K"],
            "knowledge_titles": ["旧标题"],
            "excerpt": "索引中的教材原文。",
            "book_id": "H7A",
            "book_title": "中国历史",
            "unit_id": "H7A-U03",
            "unit_title": "签核知识点",
            "lesson_id": "H7A-L13",
            "lesson_number": 13,
            "lesson_title": "第13课",
            "section_id": "signed-staging",
            "section_title": "签核知识点",
            "material_type": "旧类型",
            "textbook_pages": "旧页码",
            "pdf_pages": "旧PDF页码",
            "locator": "",
            "periods": [],
            "qualifiers": [],
            "certainty_notes": [],
        },
        {
            "source_id": "F-PENDING",
            "knowledge_ids": ["L16-Q015"],
            "knowledge_titles": ["待核验知识点"],
            "excerpt": "待核验原文。",
            "book_id": "H7A",
            "book_title": "中国历史",
            "unit_id": "H7A-U04",
            "unit_title": "签核知识点",
            "lesson_id": "H7A-L16",
            "lesson_number": 16,
            "lesson_title": "第16课",
            "section_id": "signed-staging",
            "section_title": "签核知识点",
            "material_type": "教材片段",
            "textbook_pages": "80",
            "pdf_pages": "85",
            "locator": "",
            "periods": [],
            "qualifiers": [],
            "certainty_notes": [],
        },
    ]
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.9, 0.4358899, 0.0], [0.8, 0.6, 0.0]],
        dtype=np.float32,
    )
    calls = []

    def batch_lookup(fragment_ids):
        calls.append(list(fragment_ids))
        return {
            "F-SHARED": {
                "fragment_id": "F-SHARED",
                "lesson_id": "H7A-L13",
                "lesson_number": 13,
                "excerpt": "数据库完整教材原文。",
                "material_type": "教材PDF片段",
                "textbook_pages": "72",
                "pdf_pages": "77",
                "verification_status": "verified",
                "active_associations": [
                    {"knowledge_id": "L13-Q011", "fragment_id": "F-SHARED", "source_id": "S-1", "knowledge_title": "活跃标题一"},
                    {"knowledge_id": "L13-Q012", "fragment_id": "F-SHARED", "source_id": "S-1", "knowledge_title": "活跃标题二"},
                    {"knowledge_id": "L13-Q012", "fragment_id": "F-SHARED", "source_id": "S-2", "knowledge_title": "活跃标题二"},
                ],
            },
            "F-PENDING": {
                "fragment_id": "F-PENDING",
                "lesson_id": "H7A-L16",
                "lesson_number": 16,
                "excerpt": "待核验原文。",
                "material_type": "教材片段",
                "textbook_pages": "80",
                "pdf_pages": "85",
                "verification_status": "pending_manual_source_review",
                "active_associations": [],
            },
        }

    monkeypatch.setenv("SIGNED_TEXTBOOK_DATABASE_URL", "sqlite:///unused.sqlite")
    monkeypatch.setattr(vector_service, "load_vector_index", lambda: (vectors, {"documents": documents}))
    monkeypatch.setattr(vector_service, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        "app.services.signed_textbook_service.load_active_fragment_associations",
        batch_lookup,
    )

    results = vector_service.retrieve_textbook_vectors("火", top_k=3)

    assert calls == [["H7A-P006-B", "F-SHARED", "F-PENDING"]]
    assert len(results) == 2
    signed = next(result for result in results if result["lesson_number"] == 13)
    assert signed["knowledge_ids"] == ["L13-Q011", "L13-Q012"]
    assert [
        (ref["knowledge_id"], ref["fragment_id"], ref["source_id"])
        for ref in signed["source_refs"]
    ] == [
        ("L13-Q011", "F-SHARED", "S-1"),
        ("L13-Q012", "F-SHARED", "S-1"),
        ("L13-Q012", "F-SHARED", "S-2"),
    ]
    assert all(ref["excerpt"] == "数据库完整教材原文。" for ref in signed["source_refs"])
    assert all(ref["textbook_pages"] == "72" for ref in signed["source_refs"])
    assert all(ref["pdf_pages"] == "77" for ref in signed["source_refs"])
    assert all(ref["verification_status"] == "verified" for ref in signed["source_refs"])
    assert not any("L16-Q015" in result["knowledge_ids"] for result in results)


def test_signed_associations_preserve_index_source_metadata_and_database_evidence():
    excerpt = (
        "为稳定周初的政治形势，周王室根据血缘关系远近和功劳大小，"
        "将宗亲和功臣等分封到各地，授予他们管理土地和民众的权力，"
        "建立诸侯国。受封的诸侯承担向周王进献贡赋的义务，其军队也要"
        "服从周王调遣；同时具有一定的独立性，可以在自己的封地内进行再分封。"
    )
    document = {
        "source_id": "H7A-P025-C",
        "knowledge_ids": ["STALE-K"],
        "knowledge_titles": ["旧标题"],
        "excerpt": "索引构建时教材原文。",
        "book_id": "H7A",
        "book_title": "中国历史",
        "unit_id": "H7A-U02",
        "unit_title": "第二单元",
        "lesson_id": "H7A-L04",
        "lesson_number": 4,
        "lesson_title": "第4课",
        "section_id": "H7A-L04-S03",
        "section_title": "西周的兴衰",
        "material_type": "正文",
        "textbook_pages": "旧页码",
        "pdf_pages": "旧PDF页码",
        "locator": "西周分封制、诸侯义务与贵族等级",
        "periods": ["西周"],
        "qualifiers": ["一定"],
        "certainty_notes": ["教材明确说明"],
    }
    signed_fragment = {
        "fragment_id": "H7A-P025-C",
        "lesson_id": "H7A-L04",
        "lesson_number": 4,
        "excerpt": excerpt,
        "material_type": "lesson4_signed_crosswalk_pdf_fragment",
        "textbook_pages": "25",
        "pdf_pages": "30",
        "verification_status": "verified_against_original_pdf_pages_27_32",
        "active_associations": [
            {
                "knowledge_id": "L04-010",
                "fragment_id": "H7A-P025-C",
                "source_id": "H7A-P025-C",
                "knowledge_title": "西周分封制",
            },
            {
                "knowledge_id": "L04-011",
                "fragment_id": "H7A-P025-C",
                "source_id": "H7A-P025-C",
                "knowledge_title": "西周等级与礼乐",
            },
        ],
    }

    result = vector_service._document_to_knowledge_item(
        document, 0.91, signed_fragment
    )

    assert result["knowledge_ids"] == ["L04-010", "L04-011"]
    assert result["material_types"] == ["正文"]
    assert len(result["source_refs"]) == 2
    for reference in result["source_refs"]:
        assert reference["source_id"] == "H7A-P025-C"
        assert reference["fragment_id"] == "H7A-P025-C"
        assert reference["material_type"] == "正文"
        assert reference["locator"] == "西周分封制、诸侯义务与贵族等级"
        assert reference["textbook_pages"] == "25"
        assert reference["pdf_pages"] == "30"
        assert reference["excerpt"] == excerpt
        assert (
            reference["verification_status"]
            == "verified_against_original_pdf_pages_27_32"
        )


def test_unsigned_vector_result_keeps_document_source_metadata():
    document = {
        "source_id": "H7A-P006-B",
        "knowledge_ids": ["H7A-L01-010"],
        "knowledge_titles": ["北京人用火"],
        "excerpt": "北京人会使用火，还会长时间保存火种。",
        "book_id": "H7A",
        "book_title": "中国历史",
        "unit_id": "H7A-U01",
        "unit_title": "第一单元",
        "lesson_id": "H7A-L01",
        "lesson_number": 1,
        "lesson_title": "第1课",
        "section_id": "H7A-L01-S02",
        "section_title": "北京人",
        "material_type": "正文",
        "textbook_pages": "6",
        "pdf_pages": "11",
        "locator": "北京人的用火证据",
        "periods": ["史前时期"],
        "qualifiers": [],
        "certainty_notes": ["教材明确说明"],
    }

    result = vector_service._document_to_knowledge_item(document, 0.8)

    assert result["knowledge_ids"] == ["H7A-L01-010"]
    assert result["material_types"] == ["正文"]
    assert result["source_refs"] == [
        {
            "knowledge_id": "H7A-L01-010",
            "fragment_id": "H7A-P006-B",
            "source_id": "H7A-P006-B",
            "lesson_id": "H7A-L01",
            "section_id": "H7A-L01-S02",
            "material_type": "正文",
            "textbook_pages": "6",
            "pdf_pages": "11",
            "excerpt": "北京人会使用火，还会长时间保存火种。",
            "verification_status": "",
            "locator": "北京人的用火证据",
        }
    ]


def test_index_data_hash_mismatch_requires_rebuild(monkeypatch, tmp_path):
    index_file = tmp_path / "index.npz"
    monkeypatch.setenv("VECTOR_INDEX_PATH", str(index_file))
    vector_service.build_vector_index(index_file, FakeEmbeddingModel())
    changed_data = tmp_path / "changed.json"
    changed_data.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(vector_service, "TEXTBOOK_DATA_FILE", changed_data)

    with pytest.raises(vector_service.VectorIndexError, match="rebuild it"):
        vector_service.load_vector_index(index_file)


def test_corrupt_index_fails_without_keyword_fallback(monkeypatch, tmp_path):
    corrupt = tmp_path / "index.npz"
    corrupt.write_text("not npz", encoding="utf-8")
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("VECTOR_INDEX_PATH", str(corrupt))

    with pytest.raises(rag_service.KnowledgeDataError, match="corrupt"):
        rag_service.retrieve_history_knowledge("北京人")


def test_missing_index_fails_without_keyword_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("VECTOR_INDEX_PATH", str(tmp_path / "missing.npz"))

    with pytest.raises(rag_service.KnowledgeDataError, match="not found"):
        rag_service.retrieve_history_knowledge("北京人")


def test_model_load_failure_is_exposed_without_fallback(monkeypatch, tmp_path):
    index_file = tmp_path / "index.npz"
    vector_service.build_vector_index(index_file, FakeEmbeddingModel())
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("VECTOR_INDEX_PATH", str(index_file))

    def fail_model_load():
        raise vector_service.VectorIndexError("mock model load failure")

    monkeypatch.setattr(vector_service, "get_embedding_model", fail_model_load)
    with pytest.raises(rag_service.KnowledgeDataError, match="model load failure"):
        rag_service.retrieve_history_knowledge("北京人")


def test_keyword_mode_does_not_touch_vector_runtime(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("keyword mode must not use vector runtime")

    monkeypatch.setattr(vector_service, "retrieve_textbook_vectors", fail_if_called)
    results = rag_service.retrieve_history_knowledge("北京人用火有什么作用？")

    assert results
    assert all(result["retrieval_mode"] == "keyword" for result in results)


def test_vector_mode_rejects_demo_knowledge(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "demo")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")

    with pytest.raises(rag_service.KnowledgeDataError, match="requires"):
        rag_service.retrieve_history_knowledge("秦始皇")


def test_index_metadata_json_contains_no_summary_or_certainty_in_embedding_text():
    items = rag_service.load_textbook_knowledge_from_json()
    documents = vector_service.build_source_documents(items)
    raw_payload = json.loads(vector_service.TEXTBOOK_DATA_FILE.read_text(encoding="utf-8"))
    excerpts = {source["source_id"]: source["excerpt"] for source in raw_payload["sources"]}

    assert all(document["excerpt"] == excerpts[document["source_id"]] for document in documents)
    assert all("certainty" not in document["embedding_text"] for document in documents)


def test_compound_question_is_split_from_question_text_only():
    k08 = vector_service.decompose_vector_query(
        "甲骨文是什么？它在中国文字发展史上的地位如何？"
    )
    k13 = vector_service.decompose_vector_query(
        "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？"
    )

    assert [item["aspect"] for item in k08] == [
        "definition",
        "historical_status",
    ]
    assert all("甲骨文" in item["query"] for item in k08)
    assert [item["aspect"] for item in k13] == [
        "time",
        "founding_agent",
        "content",
    ]
    assert [item["subject"] for item in k13] == [
        "东汉建立",
        "东汉建立",
        "光武中兴",
    ]
    assert "东汉建立" in k13[1]["query"]
    assert "光武中兴" in k13[2]["query"]


class FacetedEmbeddingModel(FakeEmbeddingModel):
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            if "定义" in text or "是什么" in text:
                rows.append([1.0, 0.0, 0.0])
            elif "历史地位" in text or "地位如何" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def _facet_document(source_id, knowledge_id, title, excerpt):
    return {
        "source_id": source_id,
        "knowledge_ids": [knowledge_id],
        "knowledge_titles": [title],
        "excerpt": excerpt,
        "book_id": "H7A",
        "book_title": "中国历史",
        "unit_id": "H7A-U02",
        "unit_title": "第二单元",
        "lesson_id": "H7A-L08",
        "lesson_number": 8,
        "lesson_title": "第8课",
        "section_id": "signed-staging",
        "section_title": "签核知识点",
        "material_type": "正文",
        "textbook_pages": "46",
        "pdf_pages": "51",
        "locator": title,
        "periods": [],
        "qualifiers": [],
        "certainty_notes": [],
    }


def test_faceted_vector_retrieval_reserves_candidates_for_each_aspect(monkeypatch):
    documents = [
        _facet_document("SRC-DEF", "L08-Q013", "甲骨文的定义", "甲骨文是刻写在龟甲和兽骨上的文字。"),
        _facet_document("SRC-STATUS", "L08-Q016", "甲骨文的历史地位", "甲骨文是中国已发现的古代文字中年代最早、体系较为完整的文字。"),
        _facet_document("SRC-NEAR-1", "N-1", "相关材料一", "相关材料一。"),
        _facet_document("SRC-NEAR-2", "N-2", "相关材料二", "相关材料二。"),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.delenv("SIGNED_TEXTBOOK_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        vector_service, "load_vector_index", lambda: (vectors, {"documents": documents})
    )
    monkeypatch.setattr(
        vector_service, "get_embedding_model", lambda: FacetedEmbeddingModel()
    )

    results = vector_service.retrieve_textbook_vectors(
        "甲骨文是什么？它在中国文字发展史上的地位如何？", top_k=3
    )

    by_source = {
        item["source_refs"][0]["source_id"]: item for item in results
    }
    assert {"SRC-DEF", "SRC-STATUS"} <= set(by_source)
    assert by_source["SRC-DEF"]["matched_subquestions"] == ["definition"]
    assert by_source["SRC-STATUS"]["matched_subquestions"] == [
        "historical_status"
    ]


def test_faceted_vector_merge_deduplicates_source_and_keeps_subquestions(monkeypatch):
    document = _facet_document(
        "SRC-SHARED", "L08-Q013", "甲骨文综合材料", "甲骨文教材原文。"
    )
    vectors = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.delenv("SIGNED_TEXTBOOK_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        vector_service, "load_vector_index", lambda: (vectors, {"documents": [document]})
    )
    monkeypatch.setattr(
        vector_service, "get_embedding_model", lambda: FacetedEmbeddingModel()
    )

    results = vector_service.retrieve_textbook_vectors(
        "甲骨文是什么？它在中国文字发展史上的地位如何？", top_k=3
    )

    assert len(results) == 1
    assert results[0]["matched_subquestions"] == []


def test_facet_labels_require_excerpt_support_and_preserve_dynasty_subject(monkeypatch):
    documents = [
        _facet_document(
            "SRC-WESTERN",
            "L11-Q005",
            "西汉建立及都城",
            "公元前202年，刘邦建立汉朝，后定都长安，史称西汉。",
        ),
        _facet_document(
            "SRC-EASTERN",
            "L13-Q003",
            "东汉建立与刘秀完成全国统一",
            "25年，刘秀称帝，后定都洛阳，史称东汉。",
        ),
        _facet_document(
            "SRC-PROSPERITY",
            "L13-Q005",
            "“光武中兴”的表现",
            "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。",
        ),
    ]
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.7, 0.7141428, 0.0], [0.8, 0.6, 0.0]],
        dtype=np.float32,
    )
    monkeypatch.delenv("SIGNED_TEXTBOOK_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        vector_service, "load_vector_index", lambda: (vectors, {"documents": documents})
    )
    monkeypatch.setattr(vector_service, "get_embedding_model", lambda: FakeEmbeddingModel())

    results = vector_service.retrieve_textbook_vectors(
        "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？", top_k=3
    )
    by_source = {item["source_refs"][0]["source_id"]: item for item in results}

    assert {"time", "founding_agent"} <= set(
        by_source["SRC-EASTERN"]["matched_subquestions"]
    )
    assert not set(by_source["SRC-WESTERN"]["matched_subquestions"]) & {
        "time",
        "founding_agent",
    }
    assert by_source["SRC-PROSPERITY"]["matched_subquestions"] == ["content"]


class K13FacetEmbeddingModel(FakeEmbeddingModel):
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            if "哪一年" in text:
                rows.append([1.0, 0.0, 0.0])
            elif "建立者" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_k13_vector_prefers_founding_fact_over_other_eastern_han_events(
    monkeypatch,
):
    documents = [
        _facet_document(
            "SRC-WESTERN-REGIONS",
            "L14-Q018",
            "东汉设置西域长史府",
            "123年，东汉改西域都护府为西域长史府，继续管理西域。",
        ),
        _facet_document(
            "SRC-POWERFUL-FAMILIES",
            "L13-Q011",
            "东汉豪强大族势力膨胀及其影响",
            "曾帮助刘秀建立东汉政权的豪强大族，此时势力不断壮大。",
        ),
        _facet_document(
            "SRC-EASTERN-FOUNDING",
            "L13-Q003",
            "东汉建立与刘秀完成全国统一",
            "25年，刘秀称帝，后定都洛阳，史称东汉。",
        ),
        _facet_document(
            "SRC-PROSPERITY",
            "L13-Q005",
            "“光武中兴”的表现",
            "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。",
        ),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7071068, 0.7071068, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.delenv("SIGNED_TEXTBOOK_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        vector_service, "load_vector_index", lambda: (vectors, {"documents": documents})
    )
    monkeypatch.setattr(
        vector_service, "get_embedding_model", lambda: K13FacetEmbeddingModel()
    )

    results = vector_service.retrieve_textbook_vectors(
        "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？", top_k=3
    )
    by_source = {item["source_refs"][0]["source_id"]: item for item in results}

    assert "SRC-EASTERN-FOUNDING" in by_source
    assert set(by_source["SRC-EASTERN-FOUNDING"]["matched_subquestions"]) == {
        "time",
        "founding_agent",
    }
    assert by_source["SRC-PROSPERITY"]["matched_subquestions"] == ["content"]
    for source_id in {"SRC-WESTERN-REGIONS", "SRC-POWERFUL-FAMILIES"} & set(
        by_source
    ):
        assert not set(by_source[source_id]["matched_subquestions"]) & {
            "time",
            "founding_agent",
        }

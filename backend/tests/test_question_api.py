import json
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import QuestionHistory
from app.api import question as question_api
from app.services import ai_service, rag_service, vector_service


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_question_history():
    db = TestingSessionLocal()
    try:
        db.query(QuestionHistory).delete()
        db.commit()
    finally:
        db.close()

    yield

    db = TestingSessionLocal()
    try:
        db.query(QuestionHistory).delete()
        db.commit()
    finally:
        db.close()


def test_question_api_hit():
    response = client.post(
        "/api/question",
        json={"question": "\u79e6\u59cb\u7687\u4e3a\u4ec0\u4e48\u7edf\u4e00\u516d\u56fd\uff1f"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "\u79e6\u59cb\u7687\u7edf\u4e00\u516d\u56fd" in data["answer"]
    assert data["sources"]
    assert data["sources"][0]["title"] == "\u79e6\u59cb\u7687\u7edf\u4e00\u516d\u56fd"
    assert data["sources"][0]["score"] > 0


def test_question_api_multiple_sources_sorted():
    response = client.post(
        "/api/question",
        json={
            "question": "\u5927\u4e00\u7edf \u4e2d\u592e\u96c6\u6743 "
            "\u6c49\u6b66\u5e1d \u79e6\u59cb\u7687"
        },
    )

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert 1 <= len(sources) <= 3
    scores = [source["score"] for source in sources]
    assert scores == sorted(scores, reverse=True)


def test_question_api_no_match():
    response = client.post(
        "/api/question",
        json={"question": "\u4eca\u5929\u5348\u996d\u5403\u4ec0\u4e48\u6bd4\u8f83\u597d\uff1f"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "当前检索资料中暂时没有匹配到" in data["answer"]
    assert "不表示整本教材或知识库没有相关内容" in data["answer"]
    assert data["sources"] == []


def test_question_history_api_returns_200_and_histories_key():
    response = client.get("/api/question/history")

    assert response.status_code == 200
    data = response.json()
    assert "histories" in data
    assert isinstance(data["histories"], list)


def test_question_history_api_returns_sources_as_list():
    db = TestingSessionLocal()
    try:
        db.add(
            QuestionHistory(
                user_id=None,
                question="test question",
                answer="test answer",
                sources="not valid json",
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get("/api/question/history")

    assert response.status_code == 200
    histories = response.json()["histories"]
    assert len(histories) == 1
    assert histories[0]["sources"] == []


def test_question_history_api_accepts_legacy_source_snapshot():
    db = TestingSessionLocal()
    try:
        db.add(
            QuestionHistory(
                user_id=None,
                question="legacy question",
                answer="legacy answer",
                sources=json.dumps(
                    [{"title": "旧来源", "dynasty": "秦朝", "score": 3}],
                    ensure_ascii=False,
                ),
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get("/api/question/history")

    assert response.status_code == 200
    source = response.json()["histories"][0]["sources"][0]
    assert source["title"] == "旧来源"
    assert source["book_title"] == ""
    assert source["source_refs"] == []


def test_textbook_source_metadata_reaches_api_and_history(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")

    post_response = client.post(
        "/api/question", json={"question": "北京人用火有什么作用？"}
    )
    history_response = client.get("/api/question/history")

    assert post_response.status_code == 200
    source = post_response.json()["sources"][0]
    assert source["book_id"] == "H7A"
    assert source["lesson_number"] == 1
    assert source["section_title"]
    assert source["textbook_pages"]
    assert source["pdf_pages"]
    assert source["source_refs"][0]["excerpt"]
    assert post_response.json()["knowledge_outline"]["sources"][0] == source
    assert history_response.json()["histories"][0]["sources"] == post_response.json()["sources"]


def test_textbook_support_assessment_uses_the_active_retrieval_corpus(monkeypatch):
    """Extended textbook evidence must be assessed against the same corpus."""
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")

    def fail_if_model_called(*args, **kwargs):
        raise AssertionError("textbook corpus assessment must not call a model")

    monkeypatch.setattr(ai_service, "_create_client", fail_if_model_called)
    monkeypatch.setattr(question_api, "generate_answer_result", fail_if_model_called)
    point = {
        "knowledge_id": "SIGNED-CORPUS-TEST",
        "book_id": "H7A",
        "book_title": "义务教育教科书·中国历史 七年级上册",
        "unit_id": "H7A-U02",
        "unit_title": "夏商周时期",
        "lesson_id": "H7A-L05",
        "lesson_number": 5,
        "lesson_title": "青铜器与甲骨文",
        "section_title": "签核知识点",
        "title": "西周分封制加强地方控制",
        "content": (
            "西周时期，为加强对地方的控制，周王把土地和人民分封给宗亲和功臣，"
            "建立诸侯国；诸侯职位可以世袭。"
        ),
        "summary": "西周通过分封宗亲和功臣、建立诸侯国来加强地方控制。",
        "keywords": ["西周", "分封制", "地方控制"],
        "people": ["周王"],
        "qualifiers": [],
        "certainty": "signed",
        "source_refs": [
            {
                "source_id": "SIGNED-CORPUS-SRC",
                "lesson_id": "H7A-L05",
                "section_id": "signed-staging",
                "material_type": "教材片段",
                "textbook_pages": "28",
                "pdf_pages": "33",
                "excerpt": (
                    "西周时期，为加强对地方的控制，周王把土地和人民分封给宗亲和功臣，"
                    "建立诸侯国；诸侯职位可以世袭。"
                ),
                "locator": "H7A-L05-P028",
            }
        ],
    }
    candidate = {
        **point,
        "knowledge_ids": [point["knowledge_id"]],
        "score": 0.91,
        "retrieval_mode": "vector",
        "retrieval_unit": "source_excerpt",
    }
    monkeypatch.setattr(
        question_api,
        "_retrieve_for_request",
        lambda question, requested_mode: ([candidate], "vector"),
    )
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: [point])
    monkeypatch.setattr(rag_service, "load_textbook_knowledge_from_json", lambda: [])

    response = client.post(
        "/api/question",
        json={"question": "西周时期，周王为加强地方控制采取了什么措施？"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["support_assessment"]["reason_codes"] != ["no_question_anchor"]
    assert data["support_assessment"]["status"] in {"sufficient", "partial"}
    assert data["answer_mode"] in {"controlled_evidence", "controlled_partial"}
    assert [item["knowledge_id"] for item in data["used_evidence"]] == [
        "SIGNED-CORPUS-TEST"
    ]
    assert data["used_evidence"][0]["source_ids"] == ["SIGNED-CORPUS-SRC"]


def test_textbook_controlled_evidence_reaches_question_api(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setenv("AI_MODEL", "test-model")
    monkeypatch.setenv("AI_BASE_URL", "https://mock.example/v1")

    def fail_if_model_called(*args, **kwargs):
        raise AssertionError("textbook controlled assembly must not call the model")

    monkeypatch.setattr(ai_service, "_create_client", fail_if_model_called)
    monkeypatch.setattr(question_api, "generate_answer_result", fail_if_model_called)

    response = client.post(
        "/api/question", json={"question": "北京人用火有什么作用和意义？"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer_mode"] == "controlled_evidence"
    assert "北京人用火烧烤食物、防寒、照明、驱兽" in data["answer"]
    assert "改善了生存条件" in data["answer"]
    assert "人类演化史上的里程碑" in data["answer"]
    assert "（教材第6页，H7A-P006-B）" in data["answer"]
    assert not data["answer"].startswith("教材原文：")
    assert len(data["used_evidence"]) == 1
    assert data["used_evidence"][0]["knowledge_id"] == "H7A-L01-010"
    assert data["used_evidence"][0]["knowledge_ids"] == ["H7A-L01-010"]
    assert data["used_evidence"][0]["source_ids"] == ["H7A-P006-B"]
    assert data["sources"][0]["used_evidence"] == data["used_evidence"]
    assert data["sources"][0]["knowledge_id"] == "H7A-L01-010"
    assert data["sources"][0]["textbook_pages"] == "6"
    assert data["sources"][0]["pdf_pages"] == "11"


def test_textbook_data_error_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setattr(rag_service, "TEXTBOOK_DATA_FILE", tmp_path / "missing.json")

    response = client.post("/api/question", json={"question": "北京人"})

    assert response.status_code == 503
    assert response.json()["detail"].startswith("检索不可用：")


def test_question_api_exposes_actual_vector_mode(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    point = next(
        item for item in rag_service.load_history_knowledge()
        if item["knowledge_id"] == "H7A-L01-010"
    )
    reference = next(
        ref for ref in point["source_refs"]
        if ref["source_id"] == "H7A-P006-B"
    )
    # Use the linked corpus excerpt, including the requested subject/topic,
    # rather than a shortened paraphrase that cannot cover the same facets.
    excerpt = reference["excerpt"]

    def fake_vector_results(question, top_k):
        return [
            {
                "knowledge_id": "H7A-L01-010",
                "knowledge_ids": ["H7A-L01-010"],
                "title": "北京人用火",
                "dynasty": "史前时期",
                "event": excerpt,
                "content": excerpt,
                "score": 0.8123,
                "retrieval_mode": "vector",
                "retrieval_unit": "source_excerpt",
                "textbook_pages": "6",
                "pdf_pages": "11",
                "source_refs": [dict(reference)],
            }
        ]

    monkeypatch.setattr(
        "app.services.vector_service.retrieve_textbook_vectors", fake_vector_results
    )
    response = client.post("/api/question", json={"question": "北京人用火"})

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_mode"] == "vector"
    assert data["support_assessment"]["status"] == "sufficient"
    assert data["sources"][0]["supporting"] is True
    assert data["sources"][0]["retrieval_mode"] == "vector"
    assert data["sources"][0]["retrieval_unit"] == "source_excerpt"
    assert data["sources"][0]["score"] == pytest.approx(0.8123)
    assert excerpt in data["answer"]
    assert data["used_evidence"] == [
        {
            "evidence_id": "E1",
            "text": excerpt,
            "knowledge_id": "H7A-L01-010",
            "knowledge_ids": ["H7A-L01-010"],
            "source_ids": ["H7A-P006-B"],
            "source_title": "北京人用火",
            "source_titles": ["北京人用火"],
            "textbook_pages": "6",
            "pdf_pages": "11",
        }
    ]
    assert data["sources"][0]["used_evidence"] == data["used_evidence"]

    history = client.get("/api/question/history").json()["histories"][0]
    assert history["sources"][0]["used_evidence"] == data["used_evidence"]


def test_vector_api_supports_xia_founding_agent_from_verbatim_evidence(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    excerpt = "约公元前2070年，禹建立起我国历史上第一个奴隶制王朝——夏朝。"
    point = {
        "knowledge_id": "L04-001",
        "book_id": "H7A",
        "book_title": "义务教育教科书·中国历史 七年级上册",
        "unit_id": "H7A-U02",
        "unit_title": "夏商周时期",
        "lesson_id": "H7A-L04",
        "lesson_number": 4,
        "lesson_title": "夏商周的更替",
        "section_title": "夏朝的建立",
        "title": "夏朝建立",
        "content": excerpt,
        "summary": excerpt,
        "keywords": ["夏朝建立"],
        "people": [],
        "qualifiers": ["约公元前2070年"],
        "source_refs": [
            {
                "source_id": "H7A-P022-A",
                "fragment_id": "H7A-P022-A",
                "knowledge_id": "L04-001",
                "lesson_id": "H7A-L04",
                "section_id": "H7A-L04-S01",
                "material_type": "教材正文",
                "textbook_pages": "22",
                "pdf_pages": "27",
                "excerpt": excerpt,
                "verification_status": "verified",
            }
        ],
    }
    candidate = {
        **point,
        "knowledge_ids": ["L04-001"],
        "score": 0.93,
        "retrieval_mode": "vector",
        "retrieval_unit": "source_excerpt",
        "event": excerpt,
    }
    monkeypatch.setattr(
        question_api,
        "_retrieve_for_request",
        lambda question, requested_mode: ([candidate], "vector"),
    )
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: [point])

    response = client.post(
        "/api/question",
        json={"question": "夏朝的建立者是谁？", "mode": "vector"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["support_assessment"]["status"] == "sufficient"
    assert "no_question_anchor" not in data["support_assessment"]["reason_codes"]
    assert data["answer_mode"] == "controlled_evidence"
    assert "禹" in data["answer"]
    assert "约公元前2070年" in data["answer"]
    assert data["used_evidence"]
    assert data["used_evidence"][0]["text"] == excerpt
    assert data["used_evidence"][0]["source_ids"] == ["H7A-P022-A"]
    assert data["sources"][0]["supporting"] is True
    assert data["sources"][0]["used_evidence"] == data["used_evidence"]


def test_vector_nearest_neighbors_are_returned_but_cannot_become_an_answer(
    monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    points = rag_service.load_textbook_knowledge_from_json()
    point_map = {point["knowledge_id"]: point for point in points}

    def fake_vector_results(question, top_k):
        results = []
        for score, knowledge_id in zip(
            (0.478, 0.460, 0.454),
            ("H7A-L02-013", "H7A-L03-015", "H7A-L02-010"),
        ):
            item = dict(point_map[knowledge_id])
            item.update(
                {
                    "knowledge_ids": [knowledge_id],
                    "score": score,
                    "retrieval_mode": "vector",
                    "retrieval_unit": "source_excerpt",
                }
            )
            results.append(item)
        return results

    monkeypatch.setattr(
        "app.services.vector_service.retrieve_textbook_vectors", fake_vector_results
    )

    response = client.post(
        "/api/question", json={"question": "秦始皇统一六国采取了哪些措施？"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 3
    assert all(source["retrieval_mode"] == "vector" for source in data["sources"])
    assert all(source["supporting"] is False for source in data["sources"])
    assert data["answer_mode"] == "insufficient_support"
    assert "当前检索资料中暂时没有匹配到" in data["answer"]
    assert data["used_evidence"] == []
    assert data["support_assessment"]["status"] == "insufficient"
    assert data["support_assessment"]["reason_codes"] == ["no_question_anchor"]
    assert data["knowledge_outline"] is None


def test_question_history_api_can_query_record_after_post():
    question = "\u79e6\u59cb\u7687\u4e3a\u4ec0\u4e48\u7edf\u4e00\u516d\u56fd\uff1f"

    post_response = client.post("/api/question", json={"question": question})
    history_response = client.get("/api/question/history")

    assert post_response.status_code == 200
    assert history_response.status_code == 200

    histories = history_response.json()["histories"]
    assert histories
    assert histories[0]["question"] == question
    assert isinstance(histories[0]["sources"], list)
    assert histories[0]["sources"] == post_response.json()["sources"]


def test_sqlite_history_timestamp_is_serialized_as_explicit_utc():
    db = TestingSessionLocal()
    try:
        db.add(
            QuestionHistory(
                question="timezone test",
                answer="timezone answer",
                sources="[]",
                created_at=datetime(2026, 10, 19, 8, 30),
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get("/api/question/history")

    assert response.status_code == 200
    assert response.json()["histories"][0]["created_at"] == "2026-10-19T08:30:00Z"


def test_question_answer_survives_history_save_failure():
    class BrokenDb:
        def add(self, value):
            pass

        def commit(self):
            raise RuntimeError("mock database failure")

        def rollback(self):
            pass

        def close(self):
            pass

    def override_broken_db():
        yield BrokenDb()

    previous = app.dependency_overrides[get_db]
    app.dependency_overrides[get_db] = override_broken_db
    try:
        response = client.post("/api/question", json={"question": "秦始皇统一六国"})
    finally:
        app.dependency_overrides[get_db] = previous

    assert response.status_code == 200
    assert response.json()["answer"]
    assert response.json()["history_saved"] is False


def test_no_rag_bypasses_all_retrieval_and_evidence_paths(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("No-RAG must bypass retrieval and evidence paths")

    monkeypatch.setattr(question_api, "generate_direct_answer", lambda question: "Mock 自由回答")
    monkeypatch.setattr(question_api, "retrieve_history_knowledge", fail_if_called)
    monkeypatch.setattr(
        question_api.rag_service, "_retrieve_keyword_history_knowledge", fail_if_called
    )
    monkeypatch.setattr(vector_service, "retrieve_textbook_vectors", fail_if_called)
    monkeypatch.setattr(question_api, "load_textbook_knowledge_from_json", fail_if_called)
    monkeypatch.setattr(question_api, "assess_support", fail_if_called)
    monkeypatch.setattr(question_api, "supporting_answer_items", fail_if_called)
    monkeypatch.setattr(question_api, "evidence_covers_assessment", fail_if_called)
    monkeypatch.setattr(question_api, "generate_answer_result", fail_if_called)
    monkeypatch.setattr(question_api, "template_answer", fail_if_called)
    monkeypatch.setattr(question_api, "build_knowledge_outline", fail_if_called)

    response = client.post(
        "/api/question", json={"question": "北京人是谁？", "mode": "none"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "history_saved": True,
        "answer_mode": "model_generation",
        "retrieval_mode": "none",
        "requested_mode": "none",
        "executed_mode": "none",
        "answer": "Mock 自由回答",
        "sources": [],
        "used_evidence": [],
        "support_assessment": None,
        "knowledge_outline": None,
        "error": None,
    }
    history = client.get("/api/question/history").json()["histories"][0]
    assert history["answer"] == "Mock 自由回答"
    assert history["sources"] == []


def test_no_rag_provider_unavailable_fails_without_fallback(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Unavailable No-RAG must not fall back")

    monkeypatch.setenv("AI_PROVIDER", "disabled")
    monkeypatch.setattr(question_api, "retrieve_history_knowledge", fail_if_called)
    monkeypatch.setattr(question_api, "generate_answer_result", fail_if_called)
    monkeypatch.setattr(question_api, "template_answer", fail_if_called)

    response = client.post(
        "/api/question", json={"question": "北京人是谁？", "mode": "none"}
    )

    assert response.status_code == 503
    data = response.json()
    assert data["requested_mode"] == "none"
    assert data["executed_mode"] == "none"
    assert data["answer_mode"] == "error"
    assert data["answer"] == ""
    assert data["sources"] == []
    assert data["history_saved"] is False
    assert "AI 模型供应商" in data["error"]
    assert client.get("/api/question/history").json()["histories"] == []


def test_explicit_keyword_uses_existing_keyword_flow(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    original = rag_service._retrieve_keyword_history_knowledge
    calls = []

    def keyword_spy(question):
        calls.append(question)
        return original(question)

    monkeypatch.setattr(rag_service, "_retrieve_keyword_history_knowledge", keyword_spy)
    monkeypatch.setattr(
        vector_service,
        "retrieve_textbook_vectors",
        lambda *args, **kwargs: pytest.fail("keyword must not call vector retrieval"),
    )

    response = client.post(
        "/api/question", json={"question": "秦始皇统一六国", "mode": "keyword"}
    )

    assert response.status_code == 200
    assert calls == ["秦始皇统一六国"]
    assert response.json()["requested_mode"] == "keyword"
    assert response.json()["executed_mode"] == "keyword"
    assert response.json()["retrieval_mode"] == "keyword"


def test_explicit_vector_uses_existing_vector_flow(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
    point = rag_service.load_textbook_knowledge_from_json()[9]
    calls = []

    def vector_spy(question, top_k):
        calls.append((question, top_k))
        result = dict(point)
        result.update(
            {
                "knowledge_ids": [point["knowledge_id"]],
                "score": 0.8,
                "retrieval_mode": "vector",
                "retrieval_unit": "source_excerpt",
            }
        )
        return [result]

    monkeypatch.setattr(vector_service, "retrieve_textbook_vectors", vector_spy)
    response = client.post(
        "/api/question", json={"question": "北京人用火", "mode": "vector"}
    )

    assert response.status_code == 200
    assert calls == [("北京人用火", rag_service.TOP_K)]
    assert response.json()["requested_mode"] == "vector"
    assert response.json()["executed_mode"] == "vector"
    assert response.json()["retrieval_mode"] == "vector"
    assert response.json()["sources"][0]["retrieval_mode"] == "vector"


def test_omitted_mode_preserves_original_configured_dispatch(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
    original = question_api.retrieve_history_knowledge
    calls = []

    def retrieval_spy(question):
        calls.append(question)
        return original(question)

    monkeypatch.setattr(question_api, "retrieve_history_knowledge", retrieval_spy)
    response = client.post("/api/question", json={"question": "秦始皇统一六国"})

    assert response.status_code == 200
    assert calls == ["秦始皇统一六国"]
    assert response.json()["requested_mode"] is None
    assert response.json()["executed_mode"] == "keyword"
    assert response.json()["retrieval_mode"] == "keyword"

import pytest
import httpx
from app.services import rag_service

@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "demo")
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
    # Unit tests build the fixed 35-source JSON fixture unless a test opts in
    # explicitly. Never inherit the developer's signed staging database.
    monkeypatch.setenv("SIGNED_TEXTBOOK_DATABASE_URL", "")
    for prefix in ("AI", "ASR", "TTS"):
        monkeypatch.setenv(f"{prefix}_PROVIDER", "disabled")
        monkeypatch.setenv(f"{prefix}_API_KEY", "")
    def blocked(*args, **kwargs):
        raise AssertionError("External HTTP forbidden in tests; use MockTransport")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(rag_service, "load_history_knowledge_from_db", lambda: [])

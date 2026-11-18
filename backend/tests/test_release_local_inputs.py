"""Public release safety contract for the optional local candidate builder."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/evaluation/workbench/unit5-20-formalization-20261030/formal-import-20261101/scripts/build_candidate_vector_index_lesson4.py"
spec = importlib.util.spec_from_file_location("release_local_builder", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_builder_resolves_current_project():
    assert builder.ROOT == ROOT
    assert builder.BACKEND == ROOT / "backend"


def test_builder_requires_explicit_database(monkeypatch):
    monkeypatch.delenv("SIGNED_TEXTBOOK_DATABASE_URL", raising=False)
    monkeypatch.setattr(builder, "create_engine", lambda *a, **k: pytest.fail("No database access allowed"))
    with pytest.raises(RuntimeError, match="Set SIGNED_TEXTBOOK_DATABASE_URL explicitly"):
        builder.main()


def test_builder_reports_missing_local_input_before_database_access(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNED_TEXTBOOK_DATABASE_URL", "mysql+pymysql://localhost/history_ai")
    monkeypatch.setattr(builder, "FORMAL_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(builder, "create_engine", lambda *a, **k: pytest.fail("No database access allowed"))
    with pytest.raises(RuntimeError, match="Missing local input HISTORYAI_FORMAL_JSON"):
        builder.main()

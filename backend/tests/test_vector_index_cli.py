import json

import pytest

from app.scripts import build_textbook_vector_index as build_cli
from app.scripts import verify_textbook_retrieval as verify_cli


def _metadata():
    return {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "model_revision": "revision",
        "vector_count": 42,
        "vector_dimension": 512,
        "textbook_data_sha256": "data-hash",
        "maximum_document_tokens": 80,
        "model_max_tokens": 512,
        "built_at": "2026-11-02T00:00:00Z",
        "signed_source_audit": {
            "signed_knowledge_total": 10,
            "retrievable_knowledge_count": 9,
            "active_fragment_count": 7,
            "pending_association_count": 2,
            "excluded_knowledge_count": 1,
            "final_vector_count": 42,
        },
    }


def test_signed_build_requires_explicit_output(monkeypatch):
    monkeypatch.setenv("SIGNED_TEXTBOOK_DATABASE_URL", "sqlite:///signed.sqlite")
    with pytest.raises(SystemExit) as exc:
        build_cli.main([])
    assert exc.value.code == 2


def test_signed_build_refuses_to_overwrite_existing_output(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNED_TEXTBOOK_DATABASE_URL", "sqlite:///signed.sqlite")
    output = tmp_path / "existing.npz"
    output.write_bytes(b"keep")
    monkeypatch.setattr(
        build_cli,
        "build_vector_index",
        lambda **kwargs: pytest.fail("builder must not run"),
    )

    with pytest.raises(SystemExit) as exc:
        build_cli.main(["--output", str(output)])

    assert exc.value.code == 2
    assert output.read_bytes() == b"keep"


def test_signed_build_passes_real_output_and_prints_audit(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SIGNED_TEXTBOOK_DATABASE_URL", "sqlite:///signed.sqlite")
    output = tmp_path / "new-candidate.npz"
    called = {}

    def fake_build(index_file):
        called["index_file"] = index_file
        return _metadata()

    monkeypatch.setattr(build_cli, "build_vector_index", fake_build)

    assert build_cli.main(["--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert called["index_file"] == output
    assert report["index_file"] == str(output)
    assert report["signed_source_audit"]["active_fragment_count"] == 7


def test_retrieval_verifier_loads_explicit_index_path(monkeypatch, tmp_path):
    index_file = tmp_path / "candidate.npz"
    captured = {}

    def fake_load(path):
        captured["path"] = path
        return None, {
            "model_id": "model",
            "model_revision": "revision",
            "format_version": "format",
            "vector_count": 10,
            "vector_dimension": 512,
            "textbook_data_sha256": "hash",
            "similarity": "cosine",
            "normalization": "l2",
        }

    monkeypatch.setattr(verify_cli, "load_vector_index", fake_load)
    monkeypatch.setattr(
        verify_cli.rag_service, "retrieve_history_knowledge", lambda question: []
    )

    report = verify_cli.run_verification(index_file)

    assert captured["path"] == index_file.resolve()
    assert report["index_file"] == str(index_file.resolve())

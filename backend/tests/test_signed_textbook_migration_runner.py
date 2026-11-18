from pathlib import Path

import pytest

from app.scripts.run_signed_textbook_migration import (
    DEFAULT_SQL_FILE,
    EXPECTED_TABLES,
    MigrationValidationError,
    main,
    split_and_validate_migration,
)


def test_current_migration_preserves_all_three_create_table_statements():
    sql = DEFAULT_SQL_FILE.read_text(encoding="utf-8")

    statements = split_and_validate_migration(sql)

    assert len(statements) == 3
    assert [
        statement.split("(", 1)[0].split()[-1].lower() for statement in statements
    ] == list(EXPECTED_TABLES)
    assert statements[0].startswith("CREATE TABLE IF NOT EXISTS signed_textbook_knowledge")
    assert statements[1].startswith(
        "CREATE TABLE IF NOT EXISTS signed_textbook_source_fragment"
    )
    assert statements[2].startswith(
        "CREATE TABLE IF NOT EXISTS signed_textbook_knowledge_source"
    )


@pytest.mark.parametrize("keyword", ["DROP", "DELETE", "TRUNCATE"])
def test_destructive_statements_are_rejected(keyword):
    sql = DEFAULT_SQL_FILE.read_text(encoding="utf-8") + f"\n{keyword} TABLE forbidden;"

    with pytest.raises(MigrationValidationError, match="Destructive keyword"):
        split_and_validate_migration(sql)


def test_missing_required_index_is_rejected():
    sql = DEFAULT_SQL_FILE.read_text(encoding="utf-8").replace(
        "KEY ix_signed_textbook_knowledge_source_source_id (source_id),", ""
    )

    with pytest.raises(MigrationValidationError, match="Association index is missing"):
        split_and_validate_migration(sql)


def test_dry_run_prints_all_sql_without_calling_apply(monkeypatch, capsys, tmp_path):
    sql_file = tmp_path / "migration.sql"
    sql_file.write_text(DEFAULT_SQL_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    def forbidden_apply(*args, **kwargs):
        raise AssertionError("dry-run must not create a database connection")

    monkeypatch.setattr(
        "app.scripts.run_signed_textbook_migration.apply_migration", forbidden_apply
    )

    assert main(["--dry-run", "--sql-file", str(sql_file)]) == 0
    output = capsys.readouterr().out
    assert "VALIDATION_OK statements=3" in output
    assert output.count("-- STATEMENT ") == 3
    assert "database_connection_created=false" in output

"""Validate and optionally apply the signed-textbook additive migration.

Dry-run never creates a database engine or opens a database connection.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SQL_FILE = Path(__file__).with_name("migration_signed_textbook_knowledge.sql")
EXPECTED_TABLES = (
    "signed_textbook_knowledge",
    "signed_textbook_source_fragment",
    "signed_textbook_knowledge_source",
)
DESTRUCTIVE_KEYWORDS = ("DROP", "DELETE", "TRUNCATE")


class MigrationValidationError(ValueError):
    """The migration SQL does not match the approved additive schema."""


def _without_line_comments(sql: str) -> str:
    """Remove complete ``--`` comment lines without dropping adjacent SQL."""
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def _normalized(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def _require(pattern: str, statement: str, message: str) -> None:
    if not re.search(pattern, statement, flags=re.IGNORECASE | re.DOTALL):
        raise MigrationValidationError(message)


def split_and_validate_migration(sql: str) -> list[str]:
    """Return exactly the three approved CREATE TABLE statements."""
    cleaned = _without_line_comments(sql)
    destructive = re.search(
        rf"\b({'|'.join(DESTRUCTIVE_KEYWORDS)})\b", cleaned, flags=re.IGNORECASE
    )
    if destructive:
        raise MigrationValidationError(
            f"Destructive keyword is forbidden: {destructive.group(1).upper()}"
        )

    statements = [part.strip() for part in cleaned.split(";") if part.strip()]
    if len(statements) != 3:
        raise MigrationValidationError(
            f"Expected exactly 3 SQL statements, found {len(statements)}"
        )

    actual_tables = []
    for statement in statements:
        match = re.match(
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z0-9_]+)\s*\(",
            statement,
            flags=re.IGNORECASE,
        )
        if not match:
            raise MigrationValidationError("Every statement must be CREATE TABLE IF NOT EXISTS")
        actual_tables.append(match.group(1).lower())
    if tuple(actual_tables) != EXPECTED_TABLES:
        raise MigrationValidationError(
            f"Unexpected table order: {actual_tables}; expected {list(EXPECTED_TABLES)}"
        )

    knowledge = _normalized(statements[0])
    _require(
        r"knowledge_id\s+VARCHAR\s*\(\s*64\s*\)\s+PRIMARY\s+KEY",
        knowledge,
        "signed_textbook_knowledge must use knowledge_id as primary key",
    )
    _require(
        r"KEY\s+ix_signed_textbook_knowledge_lesson_number\s*\(\s*lesson_number\s*\)",
        knowledge,
        "signed_textbook_knowledge lesson_number index is missing",
    )

    fragment = _normalized(statements[1])
    _require(
        r"fragment_id\s+VARCHAR\s*\(\s*128\s*\)\s+PRIMARY\s+KEY",
        fragment,
        "signed_textbook_source_fragment must use fragment_id as primary key",
    )
    _require(
        r"KEY\s+ix_signed_textbook_source_fragment_lesson_number\s*\(\s*lesson_number\s*\)",
        fragment,
        "signed_textbook_source_fragment lesson_number index is missing",
    )

    association = _normalized(statements[2])
    _require(
        r"PRIMARY\s+KEY\s*\(\s*knowledge_id\s*,\s*fragment_id\s*,\s*source_id\s*\)",
        association,
        "signed_textbook_knowledge_source composite primary key is missing",
    )
    for index_name, column in (
        ("ix_signed_textbook_knowledge_source_fragment_id", "fragment_id"),
        ("ix_signed_textbook_knowledge_source_source_id", "source_id"),
        ("ix_signed_textbook_knowledge_source_relation_status", "relation_status"),
    ):
        _require(
            rf"KEY\s+{index_name}\s*\(\s*{column}\s*\)",
            association,
            f"Association index is missing: {index_name}",
        )
    _require(
        r"FOREIGN\s+KEY\s*\(\s*knowledge_id\s*\)\s+REFERENCES\s+signed_textbook_knowledge\s*\(\s*knowledge_id\s*\)",
        association,
        "knowledge_id foreign key is missing",
    )
    _require(
        r"FOREIGN\s+KEY\s*\(\s*fragment_id\s*\)\s+REFERENCES\s+signed_textbook_source_fragment\s*\(\s*fragment_id\s*\)",
        association,
        "fragment_id foreign key is missing",
    )
    return statements


def apply_migration(database_url: str, statements: list[str]) -> None:
    """Open a connection only for an explicitly authorized apply operation."""
    if not database_url:
        raise ValueError("An explicit database URL is required for --apply")
    from sqlalchemy import create_engine

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql-file", type=Path, default=DEFAULT_SQL_FILE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url")
    args = parser.parse_args(argv)

    sql = args.sql_file.read_text(encoding="utf-8")
    statements = split_and_validate_migration(sql)
    print(f"VALIDATION_OK statements={len(statements)}")
    for index, (table, statement) in enumerate(
        zip(EXPECTED_TABLES, statements), start=1
    ):
        print(f"\n-- STATEMENT {index}: {table}\n{statement};")

    if args.dry_run:
        print("\nDRY_RUN_COMPLETE database_connection_created=false")
        return 0
    apply_migration(args.database_url or "", statements)
    print("\nMIGRATION_APPLIED=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

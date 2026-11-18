"""Explicit CLI for building the local textbook vector index."""

import argparse
import json
import os
from pathlib import Path

from app.services.vector_service import build_vector_index, configured_index_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit destination NPZ; required when signed MySQL data is enabled",
    )
    args = parser.parse_args(argv)
    signed_enabled = bool(os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip())
    if signed_enabled and args.output is None:
        parser.error("--output is required when SIGNED_TEXTBOOK_DATABASE_URL is configured")
    if signed_enabled and args.output.exists():
        parser.error(f"refusing to overwrite existing signed-corpus index: {args.output}")
    index_file = args.output or configured_index_file()
    metadata = build_vector_index(index_file=index_file)
    print(
        json.dumps(
            {
                "index_file": str(index_file),
                "model_id": metadata["model_id"],
                "model_revision": metadata["model_revision"],
                "vector_count": metadata["vector_count"],
                "vector_dimension": metadata["vector_dimension"],
                "textbook_data_sha256": metadata["textbook_data_sha256"],
                "maximum_document_tokens": metadata["maximum_document_tokens"],
                "model_max_tokens": metadata["model_max_tokens"],
                "built_at": metadata["built_at"],
                "signed_source_audit": metadata.get("signed_source_audit"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

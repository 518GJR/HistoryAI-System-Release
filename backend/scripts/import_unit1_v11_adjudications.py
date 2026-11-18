"""Validate and merge Unit 1 v1.1 criterion-level human adjudications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.evaluation.adjudication_v11 import (
    import_adjudications,
    read_jsonl,
    write_jsonl,
)
from app.services.evaluation.v11_rules import load_v11_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = load_v11_rules(args.rules)
    source = read_jsonl(args.scores)
    merged = import_adjudications(source, args.adjudications, rules)
    write_jsonl(args.output, merged, overwrite=args.overwrite)
    summary = {
        "record_count": len(merged),
        "human_confirmed_final_count": sum(bool(item["human_confirmed_final"]) for item in merged),
        "pending_record_count": sum(bool(item["needs_review"]) for item in merged),
        "rule_version": rules["rule_version"],
        "automatic_results_preserved": all(
            criterion["final"].get("automatic_preserved")
            for item in merged for criterion in item["criteria"]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Write CSV, JSON, and Markdown statistics for Unit 1 v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.evaluation.adjudication_v11 import (
    import_adjudications,
    read_jsonl,
    write_jsonl,
)
from app.services.evaluation.statistics_v11 import build_statistics, write_statistics_reports
from app.services.evaluation.v11_rules import load_v11_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = load_v11_rules(args.rules)
    responses = read_jsonl(args.responses)
    scores = read_jsonl(args.scores)
    if args.adjudications:
        scores = import_adjudications(scores, args.adjudications, rules)
        write_jsonl(
            args.output_dir / "final_scores.jsonl",
            scores,
            overwrite=args.overwrite,
        )
    report = build_statistics(responses, scores, rule_version=rules["rule_version"])
    write_statistics_reports(args.output_dir, report, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

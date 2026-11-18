"""Build the audited machine-readable workbench copy from frozen inputs."""

from pathlib import Path

from app.services.evaluation.rubric_loader import write_rubric_and_audit


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    workbench = project_root / "docs/evaluation/workbench"
    dataset, audit = write_rubric_and_audit(
        project_root,
        workbench / "unit1_v1.0_rubric.json",
        workbench / "unit1_v1.0_fidelity_report.json",
    )
    print(
        f"Built {len(dataset.questions)} questions / "
        f"{sum(len(q.criteria) for q in dataset.questions)} criteria; "
        f"structure={audit['structure_acceptance']}."
    )


if __name__ == "__main__":
    main()

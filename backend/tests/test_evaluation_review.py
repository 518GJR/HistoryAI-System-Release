from datetime import datetime, timezone

from app.services.evaluation.models import ReviewRecord
from app.services.evaluation.review_manager import ReviewManager


def test_append_only_review_preserves_automatic_result(tmp_path):
    manager = ReviewManager(tmp_path / "reviews.jsonl")
    automatic = {"final_correctness": None, "needs_review": True, "earned_score": None}
    pending = ReviewRecord(
        review_id="review-1", run_id="run", question_id="Q001",
        requested_mode="keyword", needs_review=True,
        review_reason=["uncertain"], review_status="pending",
        automatic_result=automatic,
    )
    override = ReviewRecord(
        review_id="review-2", run_id="run", question_id="Q001",
        requested_mode="keyword", needs_review=False,
        review_reason=["uncertain"], review_status="overridden",
        reviewer_decision="partial", reviewer_score=6,
        reviewer_comment="human gold review",
        reviewed_at=datetime.now(timezone.utc), automatic_result=automatic,
    )

    manager.append(pending)
    manager.append(override)
    records = manager.read_all()

    assert len(records) == 2
    assert records[0].automatic_result == automatic
    assert records[1].automatic_result == automatic
    assert records[1].reviewer_decision == "partial"

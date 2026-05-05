import pandas as pd

from manual_recalc.storage import build_batch_progress


def test_build_batch_progress_marks_done_partial_and_new_batches() -> None:
    batches = [
        ["r1", "r2"],
        ["r3", "r4"],
        ["r5"],
    ]
    status_df = pd.DataFrame(
        [
            {"review_id": "r1", "status": "done", "committed": 1},
            {"review_id": "r2", "status": "done", "committed": 1},
            {"review_id": "r3", "status": "done", "committed": 1},
            {"review_id": "r4", "status": "not_started", "committed": 0},
        ]
    )

    summaries = build_batch_progress(batches, status_df)

    assert [item["status"] for item in summaries] == ["done", "partial", "new"]
    assert summaries[0]["done"] == 2
    assert summaries[0]["dirty"] == 0
    assert summaries[0]["label"] == "batch_001 [done 2/2] (2 reviews)"
    assert summaries[1]["done"] == 1
    assert summaries[1]["dirty"] == 1
    assert summaries[1]["label"] == "batch_002 [partial 1/2 | draft 1] (2 reviews)"
    assert summaries[2]["done"] == 0
    assert summaries[2]["dirty"] == 0
    assert summaries[2]["label"] == "batch_003 [new] (1 reviews)"


def test_build_batch_progress_returns_new_batches_without_status_rows() -> None:
    summaries = build_batch_progress([["r1", "r2"]], pd.DataFrame())

    assert summaries == [
        {
            "batch_id": "batch_001",
            "total": 2,
            "done": 0,
            "dirty": 0,
            "status": "new",
            "label": "batch_001 [new] (2 reviews)",
        }
    ]


def test_build_batch_progress_keeps_custom_batch_id() -> None:
    summaries = build_batch_progress(
        [{"batch_id": "repair_batch", "review_ids": ["r1", "r2"]}],
        pd.DataFrame([{"review_id": "r1", "status": "done", "committed": 1}]),
    )

    assert summaries == [
        {
            "batch_id": "repair_batch",
            "total": 2,
            "done": 1,
            "dirty": 0,
            "status": "partial",
            "label": "repair_batch [partial 1/2 | draft 0] (2 reviews)",
        }
    ]

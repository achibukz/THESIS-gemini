"""Shared synthetic-CSV builders for build_dataset tests.

Every test in this package uses small in-memory CSVs written to tmp_path.
These helpers keep the test bodies focused on the assertion, not on
boilerplate row construction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


VIDEO_COLUMNS = [
    "video_id", "post_date", "post_time", "caption", "duration_ms",
    "comments", "shares", "ECR", "avg_watch_time_s",
    "NAWP", "watched_full_pct", "traffic_foryou_pct", "traffic_follow_pct",
    "traffic_profile_pct", "traffic_search_pct", "new_followers",
    "data_quality",
]

FOLLOWER_COLUMNS = [
    "date", "follower_count", "daily_net", "creator_handle",
    "creator_uid", "data_quality",
]


def _row(columns: list[str], **kwargs) -> dict:
    row = {c: "" for c in columns}
    row.update(kwargs)
    return row


@pytest.fixture
def write_videos_csv(tmp_path):
    """Return a function (handle, rows) -> Path that writes a video CSV."""
    def _write(handle: str, rows: list[dict], date: str = "2026-06-20") -> Path:
        df = pd.DataFrame([_row(VIDEO_COLUMNS, **r) for r in rows])
        path = tmp_path / f"tiktok_videos_{handle}_{date}.csv"
        df.to_csv(path, index=False)
        return path
    return _write


@pytest.fixture
def write_followers_csv(tmp_path):
    """Return a function (handle, rows) -> Path that writes a follower CSV."""
    def _write(handle: str, rows: list[dict], date: str = "2026-06-20") -> Path:
        df = pd.DataFrame([_row(FOLLOWER_COLUMNS, **r) for r in rows])
        path = tmp_path / f"tiktok_followers_{handle}_{date}.csv"
        df.to_csv(path, index=False)
        return path
    return _write


@pytest.fixture
def write_roster_csv(tmp_path):
    """Return a function (rows) -> Path that writes a roster CSV."""
    def _write(rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        path = tmp_path / "roster.csv"
        df.to_csv(path, index=False)
        return path
    return _write

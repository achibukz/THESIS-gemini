#!/usr/bin/env python3
"""Merge video-analytics and follower-history CSVs from the TikTok Analytics Exporter.

Joins the two datasets on date so that each video row gains a
``follower_count_at_post`` column — the creator's follower count on the
day the video was published.

Usage
-----
    # From the pipeline/ directory:
    uv run python merge_csvs.py \
        --videos  ../tiktok-analytics-exporter/input/tiktok_videos_*.csv \
        --followers ../tiktok-analytics-exporter/input/tiktok_followers_*.csv \
        --output  ../outputs/merged_analytics.csv

    # Or with multiple creator files (glob-expanded by the shell):
    uv run python merge_csvs.py \
        --videos  ../data/videos_creator_*.csv \
        --followers ../data/followers_creator_*.csv \
        --output  ../outputs/merged_analytics.csv

Design decisions
----------------
* **Exact-date join only.**  If a video's ``post_date`` has no matching row
  in the follower history, the ``follower_count_at_post`` column is left
  as NaN and a warning is printed.  The row is *not* dropped — downstream
  code can decide how to handle missing values.
* **Multi-creator safe.**  When multiple video/follower files are supplied,
  we concatenate them and join on ``(post_date, creator_uid)`` when creator
  UIDs are populated, falling back to date-only joins for single-creator
  data.
* **Idempotent column.**  If the video CSV already contains a
  ``follower_count`` column (the extension's snapshot-at-export value),
  we keep it as-is and add the *new* ``follower_count_at_post`` beside it
  so both are available for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_videos(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate one or more video-analytics CSVs."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"video_id": str})
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["post_date"] = pd.to_datetime(combined["post_date"], format="%Y-%m-%d")
    return combined


def load_followers(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate one or more follower-history CSVs.

    Rows with ``data_quality == 'no_data'`` are dropped — they carry no
    usable follower count.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"creator_uid": str})
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Drop rows the extension flagged as having no data
    combined = combined[combined["data_quality"] != "no_data"].copy()

    # Normalise the join key
    combined["date"] = pd.to_datetime(combined["date"], format="%Y-%m-%d")
    return combined


def merge(
    videos: pd.DataFrame,
    followers: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join videos with follower history on exact date match.

    Returns the videos DataFrame with a new ``follower_count_at_post``
    column.  Rows that had no follower-history match keep NaN.
    """
    # Decide join keys: if creator_uid is present and non-empty in both
    # DataFrames, use (date, creator_uid) for a multi-creator-safe join.
    videos_have_uid = (
        "creator_uid" in videos.columns
        and videos["creator_uid"].notna().any()
        and (videos["creator_uid"] != "").any()
    )
    followers_have_uid = (
        "creator_uid" in followers.columns
        and followers["creator_uid"].notna().any()
        and (followers["creator_uid"] != "").any()
    )

    # Prepare the follower lookup table — keep only the columns we need
    follower_lookup = followers[["date", "follower_count", "creator_uid"]].copy()
    follower_lookup = follower_lookup.rename(
        columns={"follower_count": "follower_count_at_post"}
    )

    if videos_have_uid and followers_have_uid:
        # Multi-creator join
        merged = videos.merge(
            follower_lookup,
            left_on=["post_date", "creator_uid"],
            right_on=["date", "creator_uid"],
            how="left",
        )
    else:
        # Single-creator fallback (date-only)
        follower_lookup = follower_lookup.drop(columns=["creator_uid"])
        # Deduplicate in case multiple creator rows share the same date
        follower_lookup = follower_lookup.drop_duplicates(subset=["date"])
        merged = videos.merge(
            follower_lookup,
            left_on="post_date",
            right_on="date",
            how="left",
        )

    # Clean up the extra 'date' column from the follower side
    if "date" in merged.columns:
        merged = merged.drop(columns=["date"])

    return merged


def report_missing(merged: pd.DataFrame) -> None:
    """Print warnings for video rows that could not be matched."""
    missing = merged[merged["follower_count_at_post"].isna()]
    if missing.empty:
        return

    n_missing = len(missing)
    n_total = len(merged)
    print(
        f"⚠  {n_missing}/{n_total} video(s) had no follower-history match "
        f"on their post_date — follower_count_at_post is NaN for these rows.",
        file=sys.stderr,
    )

    # Show the unmatched dates (deduplicated)
    dates = sorted(missing["post_date"].dropna().unique())
    if len(dates) <= 10:
        for d in dates:
            print(f"   • {pd.Timestamp(d).strftime('%Y-%m-%d')}", file=sys.stderr)
    else:
        for d in dates[:5]:
            print(f"   • {pd.Timestamp(d).strftime('%Y-%m-%d')}", file=sys.stderr)
        print(f"   … and {len(dates) - 5} more dates", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge video-analytics and follower-history CSVs."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to video-analytics CSV file(s).",
    )
    parser.add_argument(
        "--followers",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to follower-history CSV file(s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the merged output CSV.",
    )
    args = parser.parse_args(argv)

    # Validate inputs exist
    for p in args.videos:
        if not p.exists():
            parser.error(f"Video CSV not found: {p}")
    for p in args.followers:
        if not p.exists():
            parser.error(f"Follower CSV not found: {p}")

    # Load
    print(f"Loading {len(args.videos)} video file(s)…")
    videos = load_videos(args.videos)
    print(f"  → {len(videos)} video rows")

    print(f"Loading {len(args.followers)} follower file(s)…")
    followers = load_followers(args.followers)
    print(f"  → {len(followers)} follower-history rows (after dropping no_data)")

    # Merge
    merged = merge(videos, followers)

    # Report
    report_missing(merged)

    # Convert post_date back to string for CSV output
    merged["post_date"] = merged["post_date"].dt.strftime("%Y-%m-%d")

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"✓ Wrote {len(merged)} rows to {args.output}")

    # Quick summary
    matched = merged["follower_count_at_post"].notna().sum()
    print(f"  {matched}/{len(merged)} videos matched with follower data")


if __name__ == "__main__":
    main()

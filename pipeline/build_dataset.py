#!/usr/bin/env python3
"""Build the conclusive multi-creator dataset.

Joins N creators' video-analytics and follower-history CSVs against a
roster CSV (intake from the consent Google Form), producing one anonymized
``outputs/dataset.csv`` keyed by ``(post_date, pseudonymous_id)``.

See docs/2026-06-21-multi-creator-merge-design.md for the design spec.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


# Anchored to match the extension's: tiktok_<kind>_<handle>_<YYYY-MM-DD>.csv
# Handle slug is everything between the kind and the trailing date.
_FILENAME_RE = re.compile(
    r"^tiktok_(?P<kind>videos|followers)_(?P<slug>.+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$"
)


class ParseError(ValueError):
    """Raised when a filename does not match the expected pattern."""


def sanitize_handle(handle: str) -> str:
    """Apply the same sanitization the Chrome extension applies.

    See tiktok-analytics-exporter/popup.js:sanitize — replaces every
    character outside [a-zA-Z0-9_-] with underscore, truncates to 64.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", handle)[:64]


def parse_filename(path: Path) -> tuple[str, str, str]:
    """Return (kind, slug, date) parsed from a CSV filename.

    Raises ParseError if the filename does not match
    tiktok_{videos|followers}_<slug>_<YYYY-MM-DD>.csv.
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ParseError(f"Filename does not match expected pattern: {path.name}")
    return m.group("kind"), m.group("slug"), m.group("date")


class InputError(ValueError):
    """Raised when the discovered input set is invalid for the merge."""


def discover_inputs(
    roster: pd.DataFrame,
    inputs_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Discover per-creator video/follower CSVs in ``inputs_dir``.

    Returns ``{pseudonymous_id: {"videos": Path, "followers": Path}}``.

    Raises ``InputError`` with the full list of problems if anything is
    wrong: unknown handle in filename, missing pair, orphan roster row,
    or duplicate donation for the same handle+kind.
    """
    handle_to_id: dict[str, str] = {}
    for _, row in roster.iterrows():
        slug = sanitize_handle(row["creator_handle"])
        handle_to_id[slug] = row["pseudonymous_id"]

    files_by_slug_kind: dict[tuple[str, str], list[Path]] = defaultdict(list)
    problems: list[str] = []

    for path in sorted(inputs_dir.glob("tiktok_*.csv")):
        try:
            kind, slug, _ = parse_filename(path)
        except ParseError as e:
            problems.append(str(e))
            continue
        files_by_slug_kind[(slug, kind)].append(path)

    for (slug, kind), paths in files_by_slug_kind.items():
        if len(paths) > 1:
            problems.append(
                f"duplicate donation: {len(paths)} {kind} files for handle "
                f"'{slug}': {[p.name for p in paths]}"
            )

    seen_slugs = {slug for (slug, _) in files_by_slug_kind}
    for slug in seen_slugs:
        if slug not in handle_to_id:
            problems.append(
                f"unknown handle in filename: '{slug}' is not in roster"
            )

    for slug in seen_slugs:
        if slug not in handle_to_id:
            continue
        has_videos = (slug, "videos") in files_by_slug_kind
        has_followers = (slug, "followers") in files_by_slug_kind
        if has_videos and not has_followers:
            problems.append(
                f"missing follower file for handle '{slug}'"
            )
        if has_followers and not has_videos:
            problems.append(
                f"missing video file for handle '{slug}'"
            )

    for slug, pseudo_id in handle_to_id.items():
        if slug not in seen_slugs:
            problems.append(
                f"orphan roster row: pseudonymous_id={pseudo_id} "
                f"(handle slug '{slug}') has no input files"
            )

    if problems:
        joined = "\n  • ".join(problems)
        raise InputError(
            f"Discovered {len(problems)} input problem(s):\n  • {joined}"
        )

    result: dict[str, dict[str, Path]] = {}
    for slug, pseudo_id in handle_to_id.items():
        result[pseudo_id] = {
            "videos": files_by_slug_kind[(slug, "videos")][0],
            "followers": files_by_slug_kind[(slug, "followers")][0],
        }
    return result


REQUIRED_ROSTER_COLUMNS = [
    "pseudonymous_id",
    "creator_handle",
    "consent_form_id",
    "donation_date",
]


class RosterError(ValueError):
    """Raised when the roster CSV is invalid."""


def load_roster(path: Path) -> pd.DataFrame:
    """Read and validate the roster CSV.

    Required columns: pseudonymous_id, creator_handle, consent_form_id,
    donation_date. All must be non-empty. pseudonymous_id and
    creator_handle must each be unique.
    """
    df = pd.read_csv(path, dtype=str).fillna("")

    missing = [c for c in REQUIRED_ROSTER_COLUMNS if c not in df.columns]
    if missing:
        raise RosterError(
            f"Roster CSV is missing required column(s): {', '.join(missing)}"
        )

    for col in ("pseudonymous_id", "creator_handle"):
        empties = df[df[col].str.strip() == ""]
        if not empties.empty:
            raise RosterError(
                f"Roster CSV has {len(empties)} row(s) with empty {col}"
            )
        dupes = df[df.duplicated(subset=[col], keep=False)]
        if not dupes.empty:
            raise RosterError(
                f"Roster CSV has duplicate {col} value(s): "
                f"{sorted(set(dupes[col]))}"
            )

    return df

#!/usr/bin/env python3
"""Build the conclusive multi-creator dataset.

Joins N creators' video-analytics and follower-history CSVs against a
roster CSV (intake from the consent Google Form), producing one anonymized
``outputs/dataset.csv`` keyed by ``(post_date, pseudonymous_id)``.

See docs/2026-06-21-multi-creator-merge-design.md for the design spec.
"""
from __future__ import annotations

import re
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

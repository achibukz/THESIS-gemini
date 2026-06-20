#!/usr/bin/env python3
"""Build the conclusive multi-creator dataset.

Joins N creators' video-analytics and follower-history CSVs against a
roster CSV (intake from the consent Google Form), producing one anonymized
``outputs/dataset.csv`` keyed by ``(post_date, pseudonymous_id)``.

See docs/2026-06-21-multi-creator-merge-design.md for the design spec.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


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

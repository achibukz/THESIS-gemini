# Multi-Creator Merge Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pipeline/build_dataset.py` that merges N creators' video and follower CSVs against a roster CSV (Google Form intake), producing one anonymized `outputs/dataset.csv` keyed by `(post_date, pseudonymous_id)` — closing the cartesian-blow-up gap documented in Caveat #1.

**Architecture:** Pre-stamp then one big join (approach B). For each video and follower CSV, parse the creator handle from the filename slug, look it up in the roster to get `pseudonymous_id`, stamp that column onto every row, drop `creator_handle` / `creator_uid`. Concatenate all videos, concatenate all followers, then do a single left-join on `(post_date, pseudonymous_id)`. Finally left-join the roster to add `consent_form_id` and `donation_date`. Strict validation up front — collect every problem before aborting, no partial output.

**Tech Stack:** Python 3.12, pandas, pytest, `uv` for env management. Chrome extension touch is a single-line safety check in `tiktok-analytics-exporter/popup.js`.

**Spec:** `docs/2026-06-21-multi-creator-merge-design.md` (commit `826e4a6` on this branch).

**Branch:** `feat/multi-creator-merge` (already cut from `main`).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `pipeline/build_dataset.py` | Create | CLI + all merge logic. Single file; small enough to stay readable. |
| `pipeline/tests/__init__.py` | Create | Empty marker so pytest discovers the package. |
| `pipeline/tests/conftest.py` | Create | Shared synthetic-CSV builders used by every test. |
| `pipeline/tests/test_build_dataset.py` | Create | All tests for `build_dataset.py`. |
| `pipeline/pyproject.toml` | Modify | Add `pytest` to a `[dependency-groups] dev` group. |
| `tiktok-analytics-exporter/popup.js` | Modify | Refuse `saveVideoCSV` / `saveFollowerCSV` when handle resolves to `'unknown'`. |
| `tiktok-analytics-exporter/SMOKE.md` | Modify | Add a smoke step that verifies the unknown-handle save guard. |
| `outputs/2026-06-21-extension-cleanup-caveats.md` | Modify | Note that Caveat #1 is now addressed by `build_dataset.py`. |
| `log-2026-06-21.md` | Modify | Record branch, spec, plan, and storage-hardening TODO. |

`pipeline/merge_csvs.py` is **not** touched. It stays as the quick single-creator path.

---

### Task 1: Pytest dev dep + test scaffolding

**Files:**
- Modify: `pipeline/pyproject.toml`
- Create: `pipeline/tests/__init__.py`
- Create: `pipeline/tests/conftest.py`

- [ ] **Step 1: Add pytest as a dev dependency**

Run from `pipeline/`:

```bash
cd pipeline
uv add --dev pytest
```

Expected: `pyproject.toml` gains a `[dependency-groups] dev = ["pytest>=…"]` block, `uv.lock` updates.

- [ ] **Step 2: Create the tests package marker**

Create `pipeline/tests/__init__.py` as an empty file.

- [ ] **Step 3: Create `pipeline/tests/conftest.py` with synthetic-CSV builders**

```python
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
    "views", "likes", "comments", "shares", "ECR", "avg_watch_time_s",
    "NAWP", "watched_full_pct", "traffic_foryou_pct", "traffic_follow_pct",
    "traffic_profile_pct", "traffic_search_pct", "new_followers",
    "creator_uid", "creator_handle", "follower_count",
    "account_created_date", "data_quality",
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
```

- [ ] **Step 4: Verify pytest runs (zero tests is fine)**

Run from `pipeline/`:

```bash
uv run pytest tests/ -q
```

Expected: `no tests ran` (exit 0 or 5). If exit 5, that's fine — pytest's "no tests collected" code.

- [ ] **Step 5: Commit**

```bash
git add pipeline/pyproject.toml pipeline/uv.lock pipeline/tests/__init__.py pipeline/tests/conftest.py
git commit -m "test: add pytest dev dep and synthetic-CSV fixtures"
```

---

### Task 2: Roster loader

**Files:**
- Create: `pipeline/build_dataset.py`
- Modify: `pipeline/tests/test_build_dataset.py`

- [ ] **Step 1: Write the failing tests for `load_roster`**

Create `pipeline/tests/test_build_dataset.py` with:

```python
"""Tests for build_dataset.py."""
from __future__ import annotations

import pytest

from build_dataset import load_roster, RosterError


def test_load_roster_happy_path(write_roster_csv):
    path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice.dev",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
        {"pseudonymous_id": "C002", "creator_handle": "bob_creator",
         "consent_form_id": "CF-002", "donation_date": "2026-06-16"},
    ])
    roster = load_roster(path)
    assert set(roster["pseudonymous_id"]) == {"C001", "C002"}
    assert set(roster["creator_handle"]) == {"alice.dev", "bob_creator"}


def test_load_roster_rejects_missing_required_column(write_roster_csv):
    path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "donation_date": "2026-06-15"},  # missing consent_form_id
    ])
    with pytest.raises(RosterError, match="consent_form_id"):
        load_roster(path)


def test_load_roster_rejects_empty_pseudonymous_id(write_roster_csv):
    path = write_roster_csv([
        {"pseudonymous_id": "", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    with pytest.raises(RosterError, match="pseudonymous_id"):
        load_roster(path)


def test_load_roster_rejects_duplicate_handle(write_roster_csv):
    path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
        {"pseudonymous_id": "C002", "creator_handle": "alice",
         "consent_form_id": "CF-002", "donation_date": "2026-06-16"},
    ])
    with pytest.raises(RosterError, match="duplicate"):
        load_roster(path)


def test_load_roster_rejects_duplicate_pseudonymous_id(write_roster_csv):
    path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
        {"pseudonymous_id": "C001", "creator_handle": "bob",
         "consent_form_id": "CF-002", "donation_date": "2026-06-16"},
    ])
    with pytest.raises(RosterError, match="duplicate"):
        load_roster(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: 5 ERRORS — `ModuleNotFoundError: No module named 'build_dataset'`.

- [ ] **Step 3: Create `pipeline/build_dataset.py` with `load_roster` and `RosterError`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_dataset.py pipeline/tests/test_build_dataset.py
git commit -m "feat(build_dataset): roster loader with strict validation"
```

---

### Task 3: Filename slug parser

**Files:**
- Modify: `pipeline/build_dataset.py`
- Modify: `pipeline/tests/test_build_dataset.py`

The extension's sanitize: `s.replace(/[^a-z0-9_-]/gi, '_').slice(0, 64)` (case-insensitive, replaces anything outside `[a-zA-Z0-9_-]` with `_`). The pipeline must apply the **same** transform to the roster's `creator_handle` to build the slug-lookup column, so periods in handles (`alice.dev` → `alice_dev`) match the slug the extension actually writes.

- [ ] **Step 1: Write failing tests**

Append to `pipeline/tests/test_build_dataset.py`:

```python
from pathlib import Path

from build_dataset import sanitize_handle, parse_filename, ParseError


def test_sanitize_handle_matches_extension_rules():
    # Mirrors tiktok-analytics-exporter/popup.js: replaces anything outside
    # [a-zA-Z0-9_-] with '_', truncates to 64 chars.
    assert sanitize_handle("alice.dev") == "alice_dev"
    assert sanitize_handle("Bob_Creator") == "Bob_Creator"
    assert sanitize_handle("user@name") == "user_name"
    assert sanitize_handle("a" * 100) == "a" * 64


def test_parse_filename_videos():
    kind, slug, date = parse_filename(
        Path("/x/tiktok_videos_alice_dev_2026-06-20.csv")
    )
    assert kind == "videos"
    assert slug == "alice_dev"
    assert date == "2026-06-20"


def test_parse_filename_followers():
    kind, slug, date = parse_filename(
        Path("/x/tiktok_followers_bob_creator_2026-06-20.csv")
    )
    assert kind == "followers"
    assert slug == "bob_creator"
    assert date == "2026-06-20"


def test_parse_filename_rejects_bad_pattern():
    import pytest
    with pytest.raises(ParseError):
        parse_filename(Path("/x/garbage.csv"))


def test_parse_filename_rejects_unknown_kind():
    import pytest
    with pytest.raises(ParseError):
        parse_filename(Path("/x/tiktok_other_alice_2026-06-20.csv"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py::test_sanitize_handle_matches_extension_rules -v
```

Expected: ImportError on `sanitize_handle` / `parse_filename` / `ParseError`.

- [ ] **Step 3: Add `sanitize_handle`, `parse_filename`, `ParseError` to `build_dataset.py`**

Append to `pipeline/build_dataset.py`:

```python
import re


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: all tests pass (10 total now).

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_dataset.py pipeline/tests/test_build_dataset.py
git commit -m "feat(build_dataset): filename slug parser + sanitize"
```

---

### Task 4: Input discovery + strict validation pass

**Files:**
- Modify: `pipeline/build_dataset.py`
- Modify: `pipeline/tests/test_build_dataset.py`

`discover_inputs(roster, inputs_dir)` returns `{pseudonymous_id: {"videos": Path, "followers": Path}}`. If anything is wrong it raises `InputError` with every problem in a single message.

- [ ] **Step 1: Write failing tests**

Append to `pipeline/tests/test_build_dataset.py`:

```python
from build_dataset import discover_inputs, InputError


def test_discover_inputs_happy_path(tmp_path, write_videos_csv,
                                    write_followers_csv, write_roster_csv):
    write_videos_csv("alice", [{}])
    write_followers_csv("alice", [{}])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    roster = load_roster(roster_path)
    inputs = discover_inputs(roster, tmp_path)
    assert set(inputs.keys()) == {"C001"}
    assert inputs["C001"]["videos"].name.startswith("tiktok_videos_alice_")
    assert inputs["C001"]["followers"].name.startswith("tiktok_followers_alice_")


def test_discover_inputs_aborts_on_missing_follower(
        tmp_path, write_videos_csv, write_roster_csv):
    import pytest
    write_videos_csv("alice", [{}])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    roster = load_roster(roster_path)
    with pytest.raises(InputError, match="follower"):
        discover_inputs(roster, tmp_path)


def test_discover_inputs_aborts_on_unknown_handle(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    import pytest
    write_videos_csv("ghost", [{}])
    write_followers_csv("ghost", [{}])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    roster = load_roster(roster_path)
    with pytest.raises(InputError, match="ghost"):
        discover_inputs(roster, tmp_path)


def test_discover_inputs_aborts_on_orphan_roster_row(
        tmp_path, write_roster_csv):
    import pytest
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    roster = load_roster(roster_path)
    with pytest.raises(InputError, match="C001"):
        discover_inputs(roster, tmp_path)


def test_discover_inputs_aborts_on_duplicate_donation(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    import pytest
    write_videos_csv("alice", [{}], date="2026-06-20")
    write_videos_csv("alice", [{}], date="2026-06-21")
    write_followers_csv("alice", [{}])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    roster = load_roster(roster_path)
    with pytest.raises(InputError, match="duplicate"):
        discover_inputs(roster, tmp_path)


def test_discover_inputs_collects_all_problems(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    """A single abort message lists every problem, not just the first."""
    import pytest
    write_videos_csv("alice", [{}])  # missing followers for alice
    write_videos_csv("ghost", [{}])  # ghost is not in roster
    write_followers_csv("ghost", [{}])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
        {"pseudonymous_id": "C002", "creator_handle": "bob",
         "consent_form_id": "CF-002", "donation_date": "2026-06-16"},
    ])
    roster = load_roster(roster_path)
    with pytest.raises(InputError) as exc_info:
        discover_inputs(roster, tmp_path)
    msg = str(exc_info.value)
    assert "alice" in msg      # missing follower
    assert "ghost" in msg      # unknown handle
    assert "C002" in msg or "bob" in msg  # orphan roster row
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: 6 new test failures with ImportError on `discover_inputs` / `InputError`.

- [ ] **Step 3: Add `InputError` and `discover_inputs` to `build_dataset.py`**

Append to `pipeline/build_dataset.py`:

```python
from collections import defaultdict


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
    # Build sanitized-handle → pseudonymous_id lookup
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

    # Detect duplicate donations (same slug + kind appears in 2+ files)
    for (slug, kind), paths in files_by_slug_kind.items():
        if len(paths) > 1:
            problems.append(
                f"duplicate donation: {len(paths)} {kind} files for handle "
                f"'{slug}': {[p.name for p in paths]}"
            )

    # Detect unknown handles
    seen_slugs = {slug for (slug, _) in files_by_slug_kind}
    for slug in seen_slugs:
        if slug not in handle_to_id:
            problems.append(
                f"unknown handle in filename: '{slug}' is not in roster"
            )

    # Detect mismatched pairs (videos without followers or vice versa)
    for slug in seen_slugs:
        if slug not in handle_to_id:
            continue  # already reported
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

    # Detect orphan roster rows (in roster but no input files)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: all 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_dataset.py pipeline/tests/test_build_dataset.py
git commit -m "feat(build_dataset): strict input discovery and validation"
```

---

### Task 5: Load + stamp helpers

**Files:**
- Modify: `pipeline/build_dataset.py`
- Modify: `pipeline/tests/test_build_dataset.py`

- [ ] **Step 1: Write failing tests**

Append to `pipeline/tests/test_build_dataset.py`:

```python
from build_dataset import load_and_stamp_videos, load_and_stamp_followers


def test_load_and_stamp_videos_strips_handle_and_uid(
        write_videos_csv):
    path = write_videos_csv("alice", [
        {"video_id": "v1", "post_date": "2026-06-01",
         "creator_handle": "alice", "creator_uid": "U-123"},
    ])
    df = load_and_stamp_videos(path, "C001")
    assert list(df["pseudonymous_id"]) == ["C001"]
    assert "creator_handle" not in df.columns
    assert "creator_uid" not in df.columns
    assert df.iloc[0]["video_id"] == "v1"


def test_load_and_stamp_followers_drops_no_data_rows(
        write_followers_csv):
    path = write_followers_csv("alice", [
        {"date": "2026-06-01", "follower_count": "100",
         "data_quality": ""},
        {"date": "2026-06-02", "follower_count": "",
         "data_quality": "no_data"},
        {"date": "2026-06-03", "follower_count": "102",
         "data_quality": ""},
    ])
    df = load_and_stamp_followers(path, "C001")
    assert list(df["pseudonymous_id"]) == ["C001", "C001"]
    assert "creator_handle" not in df.columns
    assert "creator_uid" not in df.columns
    assert set(df["date"]) == {"2026-06-01", "2026-06-03"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py::test_load_and_stamp_videos_strips_handle_and_uid tests/test_build_dataset.py::test_load_and_stamp_followers_drops_no_data_rows -v
```

Expected: 2 failures (ImportError).

- [ ] **Step 3: Add `load_and_stamp_videos` and `load_and_stamp_followers`**

Append to `pipeline/build_dataset.py`:

```python
def load_and_stamp_videos(path: Path, pseudonymous_id: str) -> pd.DataFrame:
    """Read a video CSV, stamp pseudonymous_id, drop handle+uid columns."""
    df = pd.read_csv(path, dtype={"video_id": str}).fillna("")
    df["pseudonymous_id"] = pseudonymous_id
    return df.drop(columns=[c for c in ("creator_handle", "creator_uid")
                            if c in df.columns])


def load_and_stamp_followers(path: Path, pseudonymous_id: str) -> pd.DataFrame:
    """Read a follower CSV, drop no_data rows, stamp pseudonymous_id,
    drop handle+uid columns."""
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["data_quality"] != "no_data"].copy()
    df["pseudonymous_id"] = pseudonymous_id
    return df.drop(columns=[c for c in ("creator_handle", "creator_uid")
                            if c in df.columns])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: all 18 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_dataset.py pipeline/tests/test_build_dataset.py
git commit -m "feat(build_dataset): load and stamp helpers"
```

---

### Task 6: The big join (videos × followers × roster)

**Files:**
- Modify: `pipeline/build_dataset.py`
- Modify: `pipeline/tests/test_build_dataset.py`

This is the headline task. Concatenates all stamped videos and followers, does one `(post_date, pseudonymous_id)` left-join, then a roster join for provenance.

- [ ] **Step 1: Write failing tests — including the cartesian regression guard**

Append to `pipeline/tests/test_build_dataset.py`:

```python
import pandas as pd

from build_dataset import build_merged_dataset


def test_build_merged_dataset_happy_path(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    write_videos_csv("alice", [
        {"video_id": "v1", "post_date": "2026-06-01"},
    ])
    write_followers_csv("alice", [
        {"date": "2026-06-01", "follower_count": "1000", "data_quality": ""},
    ])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    df = build_merged_dataset(roster_path, tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["follower_count_at_post"] == 1000
    assert df.iloc[0]["pseudonymous_id"] == "C001"
    assert df.iloc[0]["consent_form_id"] == "CF-001"
    assert "creator_handle" not in df.columns
    assert "creator_uid" not in df.columns


def test_build_merged_dataset_cartesian_regression_guard(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    """The bug from Caveat #1: two creators posting on the same date
    must not cross-match each other's follower counts.
    """
    write_videos_csv("alice", [
        {"video_id": "v_alice", "post_date": "2026-06-01"},
    ])
    write_videos_csv("bob", [
        {"video_id": "v_bob", "post_date": "2026-06-01"},
    ])
    write_followers_csv("alice", [
        {"date": "2026-06-01", "follower_count": "1000",
         "data_quality": ""},
    ])
    write_followers_csv("bob", [
        {"date": "2026-06-01", "follower_count": "5000",
         "data_quality": ""},
    ])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
        {"pseudonymous_id": "C002", "creator_handle": "bob",
         "consent_form_id": "CF-002", "donation_date": "2026-06-15"},
    ])
    df = build_merged_dataset(roster_path, tmp_path)

    # Exactly 2 rows — one per video — no cartesian blow-up
    assert len(df) == 2

    alice_row = df[df["video_id"] == "v_alice"].iloc[0]
    bob_row = df[df["video_id"] == "v_bob"].iloc[0]
    assert alice_row["follower_count_at_post"] == 1000
    assert bob_row["follower_count_at_post"] == 5000


def test_build_merged_dataset_nan_when_no_follower_match(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    write_videos_csv("alice", [
        {"video_id": "v1", "post_date": "2025-12-01"},  # before any follower row
    ])
    write_followers_csv("alice", [
        {"date": "2026-06-01", "follower_count": "1000", "data_quality": ""},
    ])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    df = build_merged_dataset(roster_path, tmp_path)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["follower_count_at_post"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py::test_build_merged_dataset_happy_path -v
```

Expected: ImportError on `build_merged_dataset`.

- [ ] **Step 3: Add `build_merged_dataset` to `build_dataset.py`**

Append to `pipeline/build_dataset.py`:

```python
def build_merged_dataset(roster_path: Path, inputs_dir: Path) -> pd.DataFrame:
    """End-to-end merge: roster + per-creator CSVs → one dataset.

    Returns the merged DataFrame. Does not write to disk — the CLI
    main() handles I/O.
    """
    roster = load_roster(roster_path)
    inputs = discover_inputs(roster, inputs_dir)

    video_frames = [
        load_and_stamp_videos(paths["videos"], pseudo_id)
        for pseudo_id, paths in inputs.items()
    ]
    follower_frames = [
        load_and_stamp_followers(paths["followers"], pseudo_id)
        for pseudo_id, paths in inputs.items()
    ]

    videos_all = pd.concat(video_frames, ignore_index=True)
    followers_all = pd.concat(follower_frames, ignore_index=True)

    # Coerce follower_count to numeric so NaN propagates correctly
    followers_all["follower_count"] = pd.to_numeric(
        followers_all["follower_count"], errors="coerce"
    )

    follower_lookup = followers_all[
        ["date", "pseudonymous_id", "follower_count"]
    ].rename(columns={"follower_count": "follower_count_at_post"})

    # The cartesian-safe join: key includes pseudonymous_id
    merged = videos_all.merge(
        follower_lookup,
        left_on=["post_date", "pseudonymous_id"],
        right_on=["date", "pseudonymous_id"],
        how="left",
    ).drop(columns=["date"])

    # Attach roster provenance (drop creator_handle from output)
    roster_cols = roster.drop(columns=["creator_handle"])
    merged = merged.merge(roster_cols, on="pseudonymous_id", how="left")

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: all 21 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_dataset.py pipeline/tests/test_build_dataset.py
git commit -m "feat(build_dataset): cartesian-safe merge with pseudonymous_id key"
```

---

### Task 7: CLI wiring + NaN warning + integration test

**Files:**
- Modify: `pipeline/build_dataset.py`
- Modify: `pipeline/tests/test_build_dataset.py`

- [ ] **Step 1: Write failing integration test**

Append to `pipeline/tests/test_build_dataset.py`:

```python
import subprocess
import sys


def test_cli_writes_output_csv(
        tmp_path, write_videos_csv, write_followers_csv, write_roster_csv):
    write_videos_csv("alice", [
        {"video_id": "v1", "post_date": "2026-06-01"},
    ])
    write_followers_csv("alice", [
        {"date": "2026-06-01", "follower_count": "1000", "data_quality": ""},
    ])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    output = tmp_path / "out" / "dataset.csv"

    result = subprocess.run(
        [
            sys.executable, "-m", "build_dataset",
            "--inputs", str(tmp_path),
            "--roster", str(roster_path),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    df = pd.read_csv(output)
    assert df.iloc[0]["video_id"] == "v1"
    assert df.iloc[0]["follower_count_at_post"] == 1000
    assert "creator_handle" not in df.columns
    assert "creator_uid" not in df.columns


def test_cli_exits_nonzero_on_input_error(
        tmp_path, write_videos_csv, write_roster_csv):
    # Video file present, follower file missing → strict abort
    write_videos_csv("alice", [{"video_id": "v1", "post_date": "2026-06-01"}])
    roster_path = write_roster_csv([
        {"pseudonymous_id": "C001", "creator_handle": "alice",
         "consent_form_id": "CF-001", "donation_date": "2026-06-15"},
    ])
    output = tmp_path / "out" / "dataset.csv"

    result = subprocess.run(
        [
            sys.executable, "-m", "build_dataset",
            "--inputs", str(tmp_path),
            "--roster", str(roster_path),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "follower" in result.stderr.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py::test_cli_writes_output_csv -v
```

Expected: fails — `build_dataset` has no `__main__` entry point yet.

- [ ] **Step 3: Add CLI to `build_dataset.py`**

Append to `pipeline/build_dataset.py`:

```python
import argparse
import sys


def _warn_unmatched(merged: pd.DataFrame) -> None:
    """Print a stderr summary of rows with NaN follower_count_at_post."""
    missing_mask = merged["follower_count_at_post"].isna()
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return
    n_total = len(merged)
    print(
        f"⚠  {n_missing}/{n_total} video(s) had no follower-history match "
        f"on their post_date — follower_count_at_post is NaN.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the conclusive multi-creator dataset."
    )
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Directory of per-creator CSVs.")
    parser.add_argument("--roster", type=Path, required=True,
                        help="Path to the roster CSV.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the merged dataset CSV.")
    args = parser.parse_args(argv)

    try:
        merged = build_merged_dataset(args.roster, args.inputs)
    except (RosterError, InputError, ParseError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    _warn_unmatched(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    matched = int(merged["follower_count_at_post"].notna().sum())
    creators = merged["pseudonymous_id"].nunique()
    print(
        f"✓ Wrote {len(merged)} video rows across {creators} creator(s) "
        f"to {args.output} ({matched} matched, {len(merged) - matched} NaN)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd pipeline
uv run pytest tests/test_build_dataset.py -v
```

Expected: all 23 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_dataset.py pipeline/tests/test_build_dataset.py
git commit -m "feat(build_dataset): CLI entry point + NaN warning"
```

---

### Task 8: Extension safety — refuse save when handle is unknown

**Files:**
- Modify: `tiktok-analytics-exporter/popup.js`
- Modify: `tiktok-analytics-exporter/SMOKE.md`

Today `saveVideoCSV` and `saveFollowerCSV` fall back to `'unknown'` when `state.profile?.creator_handle` is empty. That produces files the pipeline must reject. Make the save refuse and prompt the user to visit their profile page first so profile interception fires.

- [ ] **Step 1: Modify `saveVideoCSV` in `tiktok-analytics-exporter/popup.js`**

Find:

```javascript
async function saveVideoCSV() {
  const res = await sendBg({ type: 'get-state' });
  const rows = res?.state?.videoStep?.rows || [];
  if (!rows.length) return;
  const handle = res.state.profile?.creator_handle || 'unknown';
  const today = isoDate(new Date());
  const filename = `tiktok_videos_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, VIDEO_CSV_COLUMNS));
}
```

Replace with:

```javascript
async function saveVideoCSV() {
  const res = await sendBg({ type: 'get-state' });
  const rows = res?.state?.videoStep?.rows || [];
  if (!rows.length) return;
  const handle = res.state.profile?.creator_handle;
  if (!handle) {
    showErr('m1-error',
      'Creator handle not detected. Open your TikTok profile in a new tab ' +
      'to let the extension capture it, then try saving again.');
    return;
  }
  const today = isoDate(new Date());
  const filename = `tiktok_videos_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, VIDEO_CSV_COLUMNS));
}
```

- [ ] **Step 2: Modify `saveFollowerCSV` in the same file**

Find:

```javascript
  const handle = res.state.profile?.creator_handle || 'unknown';
  const today = isoDate(new Date());
  const filename = `tiktok_followers_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, FOLLOWER_CSV_COLUMNS));
```

Replace with:

```javascript
  const handle = res.state.profile?.creator_handle;
  if (!handle) {
    showErr('m2-error',
      'Creator handle not detected. Open your TikTok profile in a new tab ' +
      'to let the extension capture it, then try saving again.');
    return;
  }
  const today = isoDate(new Date());
  const filename = `tiktok_followers_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, FOLLOWER_CSV_COLUMNS));
```

- [ ] **Step 3: Add a smoke step to `tiktok-analytics-exporter/SMOKE.md`**

Append to the bottom of `tiktok-analytics-exporter/SMOKE.md`:

```markdown

## Save guard: unknown handle is refused

1. Reload the extension fresh (so `state.profile` is empty).
2. Run a video extract on TikTok Studio (don't open the profile tab).
3. Press **Save CSV** in the Videos panel.
4. **Expect:** the popup shows "Creator handle not detected…" in the
   m1-error area and no file is downloaded.
5. Open your TikTok profile in a new tab; wait for it to render.
6. Press **Save CSV** again.
7. **Expect:** download succeeds, filename is `tiktok_videos_<your_handle>_<date>.csv`.
8. Repeat for the Followers panel (m2-error).
```

- [ ] **Step 4: Manual smoke (no automated test runner for the extension)**

Run the steps in the SMOKE.md addition above against an unpacked extension load. Confirm both Video and Followers panels refuse with the message and the m1/m2 error areas show it.

- [ ] **Step 5: Commit**

```bash
git add tiktok-analytics-exporter/popup.js tiktok-analytics-exporter/SMOKE.md
git commit -m "feat(extension): refuse CSV save when creator handle is unknown"
```

---

### Task 9: Update caveats doc + session log

**Files:**
- Modify: `outputs/2026-06-21-extension-cleanup-caveats.md`
- Modify: `log-2026-06-21.md`

- [ ] **Step 1: Update Caveat #1 to reflect the fix**

In `outputs/2026-06-21-extension-cleanup-caveats.md`, find the closing paragraph of section 1:

```markdown
**When you add a second creator:** either run the merge once per creator (recommended) or restore a creator identity column to the video CSV (e.g., derive it from the filename and pass it through `merge_csvs.py`).
```

Replace with:

```markdown
**When you add a second creator:** use `pipeline/build_dataset.py` (added 2026-06-21 on `feat/multi-creator-merge`). It pre-stamps `pseudonymous_id` from a roster CSV onto every video and follower row by parsing the filename slug, then joins on `(post_date, pseudonymous_id)` — the join key includes the creator, so the cartesian blow-up cannot happen. `merge_csvs.py` is unchanged and stays as the single-creator quick path.
```

- [ ] **Step 2: Append a section to `log-2026-06-21.md`**

Append at the bottom (under the existing entry's "Open items" or as a new sub-section):

```markdown

## Multi-creator merge pipeline — design + plan (branch: feat/multi-creator-merge)

**Decisions made**
- Identity source: Google Form intake roster (`sensitive_data/roster.csv`).
- File↔creator binding: filename slug (handle, sanitized like the extension).
- Roster schema: `pseudonymous_id`, `creator_handle` (join-only, dropped from output), `consent_form_id`, `donation_date`. Categorical fields (`niche`, `follower_bracket`, `audience_geo`) — schema TBD; tracked as open item.
- Anonymization: strip `creator_handle` and `creator_uid` from `dataset.csv`.
- Scope: merger only — no engineered features (cyclic encodings, `creator_age_at_post_days`, `videos_posted_before`) in this pass.
- Validation: strict-only mode. Collect every problem, abort once, no partial output.
- New script `pipeline/build_dataset.py`; `pipeline/merge_csvs.py` unchanged.

**Open items**
- **Storage hardening (deferred).** Per-creator raw inputs, `roster.csv`, and `dataset.csv` should live under `sensitive_data/` (already gitignored) to match Methodology §4.1.3 governance. For now they stay in `tiktok-analytics-exporter/input/` and `outputs/`. Revisit before the real-study merge.
- **Roster categorical schema (TBD).** `niche`, `follower_bracket`, `audience_geo` — pre-publication categorical features. Finalize before the real-study merge.

**Artifacts**
- Spec: `docs/2026-06-21-multi-creator-merge-design.md` (commit 826e4a6)
- Plan: `docs/2026-06-21-multi-creator-merge-plan.md`
```

- [ ] **Step 3: Commit**

```bash
git add outputs/2026-06-21-extension-cleanup-caveats.md log-2026-06-21.md
git commit -m "docs: log multi-creator merge design, plan, and open items"
```

---

## Wrap-up — manual checks before opening a PR

1. From `pipeline/`: `uv run pytest tests/ -v` → all 23 tests pass.
2. From repo root: `git log --oneline main..feat/multi-creator-merge` → 9 commits (1 spec + 8 implementation).
3. Manually run the smoke from Task 8 step 3 against an unpacked extension.
4. `git diff --stat main..feat/multi-creator-merge` → no files outside the table at the top of this plan should appear.

When all four pass, the branch is ready to push and PR into `main`.

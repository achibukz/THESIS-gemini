"""Tests for build_dataset.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from build_dataset import (
    load_roster, RosterError,
    sanitize_handle, parse_filename, ParseError,
)


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
    with pytest.raises(ParseError):
        parse_filename(Path("/x/garbage.csv"))


def test_parse_filename_rejects_unknown_kind():
    with pytest.raises(ParseError):
        parse_filename(Path("/x/tiktok_other_alice_2026-06-20.csv"))

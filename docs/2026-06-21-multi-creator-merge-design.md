# Multi-Creator Merge Pipeline — Design Spec

**Date:** 2026-06-21
**Branch:** `feat/multi-creator-merge`
**Status:** Spec — awaiting user review before implementation plan.

## Context

The existing `pipeline/merge_csvs.py` joins one creator's video-analytics CSV and follower-history CSV on `post_date`, producing `outputs/merged_analytics.csv` with a new `follower_count_at_post` column.

Caveat #1 in `outputs/2026-06-21-extension-cleanup-caveats.md` documents that the script is **unsafe for multiple creators**: TikTok does not populate `creator_uid` in the video payload, so the join falls through to date-only matching. Concatenating two creators' video CSVs and running the date-only join produces a cartesian blow-up — every video on date *D* matches every follower row on date *D* across all creators — and silently wrong `follower_count_at_post` values.

The study needs one conclusive dataset across 30–50 participants. This spec defines how to build it safely.

## Anchor

The pipeline produces `outputs/dataset.csv`, one row per video, anonymized, enriched with:
- `follower_count_at_post` from the matching creator's follower history.
- `pseudonymous_id` from a roster CSV (intake from the consent Google Form).
- Provenance columns (`consent_form_id`, `donation_date`) from the roster.

No engineered features (cyclic encodings, `creator_age_at_post_days`, `videos_posted_before`). Those stay in a later feature-engineering step so the merge step is reusable and auditable.

## What to Produce

### Components

#### 1. `pipeline/build_dataset.py` (new)

CLI:

```
uv run python build_dataset.py \
    --inputs   <dir>          # contains per-creator videos_<handle>_*.csv and followers_<handle>_*.csv
    --roster   <path>         # roster CSV (Google Form intake export)
    --output   <path>         # final dataset.csv
```

Stages (pre-stamp then one big join — approach B):

1. **Load roster.** Required columns: `pseudonymous_id`, `creator_handle`, `consent_form_id`, `donation_date`. Build a `handle → pseudonymous_id` lookup. Reject the run if `creator_handle` or `pseudonymous_id` is empty for any row, or if either column has duplicates.
2. **Discover inputs.** Glob `<inputs>/tiktok_videos_*.csv` and `<inputs>/tiktok_followers_*.csv`. Parse the handle slug from each filename. Filename convention assumed: `tiktok_{videos|followers}_<handle>_<YYYY-MM-DD>.csv` (handle = sanitized `[a-z0-9._-]+`).
3. **Strict validation pass.** Collect every problem first; only abort once with the full list. Failures (any one aborts the run):
   - Filename does not match the expected pattern.
   - Handle parsed from a filename is not present in the roster.
   - Handle present in a video filename has no matching follower filename, or vice versa.
   - Roster row has no corresponding pair of input files.
   - Same `(handle, kind)` appears in two filenames (ambiguous donation).
4. **Load + stamp.** For each video CSV: read, stamp `pseudonymous_id`, drop `creator_handle` and `creator_uid`. Same for each follower CSV; additionally drop rows with `data_quality == 'no_data'`.
5. **Concatenate** all video frames into `videos_all` and all follower frames into `followers_all`. Verify no duplicate `(post_date, pseudonymous_id)` rows on the follower side (would indicate two donations from the same creator covering overlapping dates — strict abort).
6. **One big left-join** of `videos_all` against `followers_all` on `(post_date == date, pseudonymous_id)`. This is the cartesian-safe join because the key includes the creator.
7. **Left-join to roster** on `pseudonymous_id` to attach `consent_form_id` and `donation_date`. Drop `creator_handle` from the roster columns — it stays only in `sensitive_data/roster.csv`.
8. **Warn** to stderr for any video row with NaN `follower_count_at_post` (video posted on a date the follower history has no usable count for). Keep the rows; do not drop.
9. **Write** `outputs/dataset.csv`. Print a summary (`N videos × M creators, X matched / Y NaN`).

#### 2. Chrome extension filename-slug fix

The extension currently saves files like `tiktok_videos_unknown_2026-06-20.csv` because the slug position was filled with the literal string `"unknown"`. The follower-history flow already captures `creator_handle` via profile interception (see Caveat #6 — `parseFollowerHistoryResponse` attaches `creator_handle` / `creator_uid` to follower-history rows).

Change: use the captured `creator_handle`, lowercased and sanitized to `[a-z0-9._-]+`, in the filename slug for both `tiktok_videos_*.csv` and `tiktok_followers_*.csv`. Fall back to `"unknown"` only if the profile interception genuinely failed (in which case the file should be discarded — the pipeline cannot use it).

Scope: minimal edit to the filename builder + a smoke note in `tiktok-analytics-exporter/SMOKE.md`. Out of scope for this branch: any other extension behavior.

#### 3. `pipeline/merge_csvs.py` (unchanged)

Kept as the quick single-creator path for ad-hoc debugging. Not deleted, not modified.

### Roster CSV schema (intake Google Form export)

| Column | Required | Purpose | Survives to `dataset.csv`? |
|---|---|---|---|
| `pseudonymous_id` | yes | Stable anonymous creator ID (e.g., `C001`) | yes |
| `creator_handle` | yes | TikTok @handle, used only to match filename slug | no — dropped |
| `consent_form_id` | yes | Which consent form was signed | yes |
| `donation_date` | yes | When the data was donated | yes |
| `niche`, `follower_bracket`, `audience_geo` | TBD | Pre-publication categorical features | TBD — schema not finalized; log open item |

### Storage layout (deferred hardening)

For this branch the script writes to `outputs/dataset.csv` and reads from whatever `--inputs` path the user passes (today that's `tiktok-analytics-exporter/input/`). The roster lives wherever the user puts it.

**Open item to log:** before the dataset is built for the real study, move per-creator raw inputs and `dataset.csv` into `sensitive_data/` (already gitignored) to match Methodology §4.1.3's "AES-256 encrypted environment, access restricted to primary research team." This is a security/sensitivity hardening pass to be tracked separately.

### Error handling

- **Strict mode is the only mode.** Every validation failure listed above aborts the run after collecting *all* problems and prints a single error block. No partial output is written.
- **NaN `follower_count_at_post`** is the one non-fatal warning condition: the video was posted on a date the follower history has no usable count for (most often because the account was too new on that date). Row preserved, count is `NaN`, warning printed.
- **Empty roster categorical fields** (niche, follower_bracket, audience_geo) are passed through as empty cells — downstream code decides what to do with them.

### Testing

Tests in `pipeline/tests/test_build_dataset.py` (pytest, kept light per project conventions). Cases:

1. **Happy path.** Two synthetic creators × small CSVs → one merged dataset, no cartesian rows, correct `follower_count_at_post` per creator.
2. **Cartesian regression guard.** Two creators posting on the same date with different follower counts → assert each video maps only to its own creator's count. This is the direct test of the bug fix.
3. **Strict abort: missing follower file.** Video CSV present for handle X, no follower CSV for X → exit non-zero, error message names handle X.
4. **Strict abort: unknown handle in filename.** Filename slug not in roster → exit non-zero, error names the handle.
5. **Strict abort: orphan roster row.** Roster row with no input files → exit non-zero, error names the `pseudonymous_id`.
6. **Anonymization.** `creator_handle` and `creator_uid` columns are absent from `dataset.csv`.
7. **NaN tolerance.** Video posted before follower history starts → row preserved with NaN, warning emitted to stderr.
8. **Duplicate-donation abort.** Two video files for the same handle → strict abort.

Tests use small in-memory or `tmp_path` CSVs; no fixtures that require real TikTok data.

## Sources

- `pipeline/merge_csvs.py` — existing single-creator merger (reused conceptually; `build_dataset.py` does not import it).
- `outputs/2026-06-21-extension-cleanup-caveats.md` Caveat #1 — defines the cartesian bug this spec closes.
- `outputs/2026-06-09-feature-engineering-notes.md` — confirms `follower_count_at_post` is the right column name and explains the temporal-leakage motivation.
- `~/Documents/Obsidian/schoolMem/wiki/AY2526-T3/THSST1-Thesis-in-Software-Technology-1/topics/dataset-collection.md` — anonymization commitments, IRB framing for the roster CSV.
- `CLAUDE.md` Locked Decisions — pipeline order (Consent → Extension Export → Submission → Video Download → Anonymization → Verification) places the roster CSV at the "Submission" step.

## Out of Scope

- Engineered features (`creator_age_at_post_days`, `videos_posted_before`, cyclic encodings of `post_hour`/`post_dow`). Deferred to a later `feature_engineering.py`.
- Roster categorical schema (`niche`, `follower_bracket`, `audience_geo`). Logged as an open item.
- Moving inputs/outputs into `sensitive_data/`. Logged as an open item.
- Any other Chrome-extension behavior change beyond the filename slug fix.
- Replacing or rewriting `pipeline/merge_csvs.py`. It stays as the quick single-creator path.

## Format Conventions

- Filename slug pattern: `tiktok_{videos|followers}_<handle>_<YYYY-MM-DD>.csv`, handle sanitized to `[a-z0-9._-]+`, lowercased.
- Output columns: union of `videos_all` columns (minus `creator_handle`, `creator_uid`) + `follower_count_at_post` + roster columns (minus `creator_handle`).
- Date columns serialized as `YYYY-MM-DD` strings in the output CSV (matches current `merge_csvs.py` behavior).

## Open Items to Log

1. Roster categorical schema (`niche`, `follower_bracket`, `audience_geo`) — finalize before real-study merge.
2. Storage hardening — move per-creator raw inputs and `dataset.csv` into `sensitive_data/` before the real-study merge.

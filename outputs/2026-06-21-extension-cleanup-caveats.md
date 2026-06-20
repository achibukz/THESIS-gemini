# Extension + Merge Cleanup — Caveats

**Date:** 2026-06-21
**Scope:** Caveats discovered while removing unextracted columns from the video CSV, restoring `data_quality`, and reviewing the Chrome extension after the tab-open buttons were dropped.

---

## 1. Multi-creator merging is not safe after the cleanup

`creator_uid` was removed from the video CSV because TikTok's insight payload never populated it for us. As a result, `pipeline/merge_csvs.py` can no longer use `(post_date, creator_uid)` as the join key — `videos_have_uid` is always `False`, so every run falls through to the date-only branch.

This is fine for a single participant. The moment two participants' video CSVs are concatenated and joined against a combined follower file, every video on date *D* will match every follower row on date *D* — producing a cartesian blow-up and silently wrong `follower_count_at_post` values.

**When you add a second creator:** either run the merge once per creator (recommended) or restore a creator identity column to the video CSV (e.g., derive it from the filename and pass it through `merge_csvs.py`).

---

## 2. `data_quality` is partial-truth, not always populated

Restored on 2026-06-21. It only carries one of two values when present:

- `insufficient_data` — TikTok flagged the whole insight payload as `status: 2` (account too new or video too fresh).
- `missing_ecr` — the 5-second retention point was not in the payload (most often paired with a `views: 0` row).

Empty string means "no problem detected". Downstream code that filters on quality should match these exact tokens, not test for truthiness.

---

## 3. True account-creation date is unobtainable

`account_created_date` was dropped because TikTok does not expose it through any creator-side surface. The agreed proxy lives in `outputs/2026-06-09-feature-engineering-notes.md` (§ "Creator Age"):

```
creator_age_at_post_days = post_date − min(post_date) per creator_uid
```

Computed downstream of `merge_csvs.py`, not in the extension. Underestimates real age when TikTok Studio truncates the export — document the limitation in Methodology §4.1.

---

## 4. `follower_count_at_post` can be `NaN` and rows are kept

`merge_csvs.py` does a left-join and warns to stderr for unmatched dates instead of dropping them. Any feature-engineering or model code reading `merged_analytics.csv` must decide what to do with NaN (drop, impute, or fail loudly). Don't assume the column is always populated.

The follower side already drops `data_quality == 'no_data'` rows before the join, so NaN on the merged side specifically means "the video was posted on a date the follower history has no usable count for" — most often because the account was too new on that date.

---

## 5. Chrome extension no longer auto-opens TikTok Studio tabs

TikTok blocks/redirects when the extension calls `chrome.tabs.create` / `chrome.tabs.update` against `tiktokstudio` URLs, so the "Open my Content page" / "Open my Followers analytics" buttons and their `is-studio-page` / `is-followers-page` checks were removed.

User-facing impact: the popup now assumes the active tab is already on the correct TikTok Studio page. The two `m1-ready` / `m2-ready` panes carry guidance text reminding the user to navigate there before extracting. If they extract from the wrong page, the orchestration will fail at the page-fetch step and surface the error in the popup. No silent corruption.

---

## 6. Profile interception is still live (for the follower CSV)

The video pipeline no longer fetches the profile or back-fills `follower_count` / `creator_handle` / `creator_uid` onto video rows. But `background.js` still intercepts profile URLs and populates `state.profile`, because `parseFollowerHistoryResponse` uses it to attach `creator_handle` / `creator_uid` to follower-history rows.

Don't strip the `PROFILE_RE` / `ACCOUNT_INFO_RE` interception or `ingestProfile` thinking they're orphaned — the follower CSV depends on them.

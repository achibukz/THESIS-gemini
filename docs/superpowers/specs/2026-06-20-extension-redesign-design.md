# Chrome Extension Redesign — Design Spec

**Date:** 2026-06-20
**Scope:** `tiktok-analytics-exporter/`
**Status:** Approved for implementation
**Mockup of record:** `tiktok-analytics-exporter/mockups/option-b-modular-v2.html`

## Context

The TikTok Analytics Exporter is a creator-side Chrome extension that lets Filipino micro-creators in the thesis study back up their own TikTok Studio analytics. The current version (v0.1.0) exports a single CSV of per-video performance data via passive interception of TikTok's `/aweme/v2/data/insight/` endpoint, orchestrated from `background.js` with a polling popup.

The 2026-06-09 adviser meeting added a **second deliverable**: a per-day follower-count history CSV covering the last 365 days. Creators will now hand off **two CSV files**. The popup also needs a "helper, not scraper" visual refresh — the current UI reads as a generic dev tool, which is a credibility problem with both the ethics board and non-technical participants.

This spec covers both: the new follower-history feature and a full popup redesign to mockup `option-b-modular-v2.html`.

## Goal

Ship a Chrome extension where, in one session, a Filipino micro-creator can:

1. See a popup that clearly frames the tool as a personal backup of their own data, with consent / ethics / privacy disclosures one tab away.
2. Be guided to open TikTok Studio → Content, then extract a `tiktok_videos_*.csv` of per-video performance.
3. Be guided to open TikTok Studio → Analytics → Followers (365-day filter pre-applied), then extract a `tiktok_followers_*.csv` of daily follower counts.
4. Send both files to the research team.

## Architectural decision

**Approach A — extend the existing pipeline.** `background.js` remains the sole orchestrator. A second flow (`runFollowersExport`) sits next to the existing video flow, reusing the same intercept infrastructure, the same `state.insightTemplate`, and the same content↔injected page-fetch bridge.

Rejected alternatives:
- *B — split into per-pipeline modules.* Cleaner long-term but bigger blast radius for a 2-flow extension. Revisit if a third flow lands.
- *C — push orchestration into the popup.* Loses the resilience that keeps Step 1 running when the popup closes mid-export. Regression.

## File-by-file change map

| File | Change | Notes |
|---|---|---|
| `popup.html` | Full rewrite | Two tabs (Export / Help); two modular step cards; help accordion |
| `popup.css` | Full rewrite | Dark header, line-icon styling, module cards, pills, page-aware states |
| `popup.js` | Significant rewrite | Drives two-step UI, page detection per step, two extract triggers, polls per-step state, builds two CSVs, hidden Debug tab via triple-click on footer |
| `background.js` | Extend (~+250 LOC) | New `state.followerStep` branch + handlers; per-step phases; remove `single-video-fetch` plumbing |
| `content.js` | Tiny extend | Add `is-followers-page` message handler |
| `injected.js` | No change | Existing intercept patterns already cover the follower-history call |
| `manifest.json` | Bump `version` to `0.2.0` | `host_permissions: *://*.tiktok.com/*` already covers the Followers path |

Removed:
- Single video tab (HTML + popup.js + `single-video-fetch` in background.js)
- Visible Debug tab (HTML remains rendered but hidden; plumbing preserved as a dev escape hatch)

## State machine

Split the single linear `phase` into two independent step branches. Each step has its own lifecycle and can be in any state independently of the other.

```js
state = {
  // shared
  insightTemplate: string | null,
  profile: { follower_count, account_created_time, creator_handle, creator_uid } | null,
  interceptCounts: { videoList, insight, profile },
  recentURLs: [],            // debug only
  lastVideoListSample: null, // debug only

  videoStep: {
    phase: 'idle' | 'collecting' | 'fetching-insights' | 'fetching-profile'
         | 'done'  | 'cancelled' | 'error',
    activeTabId: number | null,
    dateRange: { start: 'YYYY-MM-DD', end: 'YYYY-MM-DD' } | null,
    videos: { [aweme_id]: VideoMeta },
    rows: Row[],
    skipped: { aweme_id, reason }[],
    progress: { current, total, message },
    startedAt, finishedAt, error
  },

  followerStep: {
    phase: 'idle' | 'fetching' | 'done' | 'cancelled' | 'error',
    activeTabId: number | null,
    rows: FollowerRow[],
    progress: { message },
    startedAt, finishedAt, error
  }
}
```

Each step is independently cancellable, resettable, and downloadable. Cross-step interaction is limited to read-only sharing of `state.profile` (so the follower CSV can carry `creator_handle` / `creator_uid` for joining downstream).

## Data flow

### Step 1 — Video performance

Functionally identical to today, moved under `videoStep`:

1. Popup calls `chrome.tabs.query` to find the active TikTok tab, then sends `is-content-page` to `content.js`. Regex: `/(creator-center|tiktokstudio)\/content/i`.
2. Popup sends `start-video-export { dateRange, tabId }` to background.
3. Background: `videoStep.phase = 'collecting'`. Sends `scroll-to-bottom` to content.js. `injected.js` intercepts `item_list/v1` responses; background ingests video metadata into `videoStep.videos`.
4. Background filters `videoStep.videos` by `dateRange`, capped at 2000 videos.
5. Background: `videoStep.phase = 'fetching-insights'`. Loops with 2 s ± 0.5 s jitter, calling `buildInsightURL(state.insightTemplate, aweme_id)` per video via `page-fetch`. Parses with existing `parseInsightResponse`. Retries once after 3 s on failure; skips on second failure with reason.
6. Background: `videoStep.phase = 'fetching-profile'`. One `page-fetch` to the profile URL. Parses; attaches `follower_count` and `account_created_date` to every row. Also writes to shared `state.profile`.
7. Background: `videoStep.phase = 'done'`. Popup polls, shows the Save button. CSV is built in popup, downloaded via `chrome.downloads.download`.

### Step 2 — Follower history (NEW)

One GET, one parse, one CSV. No loop, no rate-limit, no scroll.

1. Popup sends `is-followers-page` to content.js. Regex: `/(creator-center|tiktokstudio)\/analytics\/followers/i`.
2. Popup sends `start-follower-export { tabId }` to background.
3. Background: `followerStep.phase = 'fetching'`. Calls:
   ```js
   buildFollowerHistoryURL(state.insightTemplate || DEFAULT_INSIGHT_BASE)
   ```
   which appends the URL-encoded `type_requests` payload:
   ```json
   [
     {"insigh_type":"follower_num_history","days":732,"end_days":1},
     {"insigh_type":"follower_num",         "days":732,"end_days":1},
     {"insigh_type":"net_follower_history", "days":732,"end_days":1}
   ]
   ```
   (Note the misspelled key `insigh_type` — that is TikTok's, not ours.)
4. Background sends `page-fetch` to content.js → injected.js performs credentialed fetch → response body returned.
5. Background calls `parseFollowerHistoryResponse(json, now, days=732, endDays=1)`:
   - Validates `status_code === 0`.
   - Reads `follower_num_history` and `net_follower_history` arrays (same length L).
   - **Date anchoring (handles `end_days` ambiguity).** `days`/`end_days` semantics aren't documented; the captured response ended with a `status: 2` entry that could be either today (unfinalized) or yesterday (missing). The parser resolves this empirically: find the last index `k` where `status === 0`, compare `follower_num_history[k].value` to the just-returned `follower_num.value`. If they match, `k → (today − 1)` (yesterday); otherwise `k → today`. Walk backwards from there one day per index.
   - For each index `i ∈ [0, L)`:
     - `date = mapIndexToDate(i, L, anchorIndex, anchorDate)`.
     - `follower_count = entry.status === 0 ? entry.value : ''`.
     - `daily_net = net_follower_history[i].status === 0 ? net_follower_history[i].value : ''`.
     - `data_quality = entry.status === 2 ? 'no_data' : ''`.
   - Trims to the **most recent 365 days**.
   - Attaches `creator_handle` and `creator_uid` from `state.profile` (blank if not yet captured).
6. Background: `followerStep.phase = 'done'`. Popup polls, shows Save. CSV built in popup, downloaded.

### "Open my … page" buttons

When the active tab is on `tiktok.com` but not the target page: `chrome.tabs.update(tabId, { url })`.
When no TikTok tab exists: `chrome.tabs.create({ url })`.

Target URLs:
- Step 1: `https://www.tiktok.com/tiktokstudio/content`
- Step 2: `https://www.tiktok.com/tiktokstudio/analytics/followers?dateRange=%7B%22type%22%3A%22fixed%22%2C%22pastDay%22%3A365%7D` (pre-applies the 365-day filter so the API call fires automatically).

## CSV schemas

### `tiktok_videos_{handle}_{YYYY-MM-DD}.csv`

Schema unchanged from current `tiktok_analytics_*.csv`. **Only the filename changes** (from `tiktok_analytics_` to `tiktok_videos_`) to disambiguate from the new follower file.

Columns (23):
```
video_id, post_date, post_time, caption, duration_ms, views, likes, comments, shares,
ECR, avg_watch_time_s, NAWP, watched_full_pct,
traffic_foryou_pct, traffic_follow_pct, traffic_profile_pct, traffic_search_pct,
new_followers, creator_uid, creator_handle, follower_count, account_created_date, data_quality
```

### `tiktok_followers_{handle}_{YYYY-MM-DD}.csv` (NEW)

Columns (6):
```
date, follower_count, daily_net, creator_handle, creator_uid, data_quality
```

- `date` — `YYYY-MM-DD`, derived from index (most-recent row = yesterday).
- `follower_count` — cumulative count that day; blank when `status: 2`.
- `daily_net` — signed integer net change from `net_follower_history`; blank when `status: 2`.
- `creator_handle`, `creator_uid` — joining keys, sourced from shared `state.profile`. Blank if no profile call has been intercepted yet.
- `data_quality` — `""` on good rows, `"no_data"` on `status: 2` rows.

Row count: exactly 365 (last 365 days, ending yesterday). Pre-account rows are present with blank counts and `data_quality=no_data` — preserves date continuity, makes account-creation visible.

## Help tab content

Five `<details>`-based accordion sections in `popup.html` with hard-coded copy and `[TBD]` placeholders where real values are missing:

| Section | Body | TBDs |
|---|---|---|
| About this extension | Purpose, two-file output, version `v0.2.0` | None |
| Your data & privacy | Runs locally, nothing uploaded, creator chooses what to share | None |
| Consent & withdrawal | Voluntary; can withdraw anytime; *View consent form* link | `[TBD: consent form path/URL]` |
| Ethics & approval | Reviewed by REC; data anonymized | `[TBD: REC name]`, `[TBD: REC ref]`, `[TBD: ethics email]` |
| Questions or help | Researcher contact | `[TBD: researcher name]`, `[TBD: researcher email]` |

`[TBD]` markers are intentionally visible so reviewers and creators can tell at a glance what's still missing. They get swapped for real values in a follow-up edit, not silently filled with placeholder-looking strings.

## Error handling

| Failure | Behaviour |
|---|---|
| No active TikTok tab | "Open TikTok Studio first"; both Extract buttons disabled |
| Wrong page for step | Module shows "Open my … page" button only; Extract hidden |
| Step 1: scroll never settles | Cap at 400 cycles (existing); proceeds with whatever loaded |
| Step 1: insight fetch fails for one video | Retry once after 3 s; if still fails, mark `skipped` with reason, continue |
| Step 1: profile fetch fails | Log; `follower_count` / `account_created_date` left blank |
| Step 2: insight fetch fails | Retry once after 3 s; surface *"Couldn't fetch follower history. Refresh and try again."* |
| Step 2: response missing `follower_num_history` | Error: *"TikTok response didn't include the expected data — open Debug (triple-click footer) and share with the researcher."* |
| Step 2: all rows `status: 2` | Still produces CSV (dates + blanks + `data_quality=no_data`); soft warning *"Your account may be too new for follower history."* |
| 401 / signed out | *"It looks like you're signed out of TikTok. Sign back in and try again."* |
| Popup closed mid-run | Background keeps running (state in `chrome.storage.session`); popup reattaches via polling on reopen |

## Testing

No test infrastructure today. Plan:

**1. Pure-function unit tests** (`tests/run.js`, zero npm deps, runs under Node).

Extract into a testable module (e.g. `lib/parsers.js`):
- `parseInsightResponse(json, video)` (existing, currently inline)
- `parseFollowerHistoryResponse(json, now, days, endDays)` (new)
- `buildFollowerHistoryURL(template)` (new)
- `mapIndexToDate(index, length, anchorIndex, anchorDate)` (new)

Fixtures in `tests/fixtures/`:
- `follower-history-response.json` — anonymized capture from this design session.
- `insight-response.json` — anonymized real insight payload.
- `item-list-response.json` — anonymized real video-list payload.

Anonymization: replace `creator_uid` with `TEST_UID`, `creator_handle` with `test_user`, `desc` text with `[caption redacted]`.

**2. Manual smoke checklist** in `tiktok-analytics-exporter/SMOKE.md`:

- [ ] Popup opens; Export tab visible with two cards; Help tab clickable.
- [ ] Open `https://www.tiktok.com/` (not Studio). Both modules show "Open my … page" buttons; Extract hidden.
- [ ] Click Step 1's open button → navigates to `/tiktokstudio/content` → reopen popup → Step 1 green "on page" → Extract → progress → done → Save → CSV lands with ≥1 row and 23 columns.
- [ ] Click Step 2's open button → navigates to `/tiktokstudio/analytics/followers?dateRange=...` → reopen popup → Step 2 green "on page" → Extract → progress → done → Save → CSV lands with 365 rows.
- [ ] Triple-click `v0.2.0` footer → Debug panel reveals; counters increment.
- [ ] Help tab: 5 accordion sections, `[TBD]` markers visible where unfilled.
- [ ] Cancel mid-Step-1 leaves popup in recoverable state.

**3. Skipping for now:** Puppeteer/E2E. Live TikTok session, flaky, overkill for a research tool.

## Out of scope (deferred)

- Submitting CSVs directly to a server (privacy — files stay on creator's device).
- Firefox / Edge support.
- Per-day follower data older than 365 days (the API returns 732; we trim).
- Re-applying the design to videos older than the visible TikTok Studio window (existing constraint).
- Real values for the Help tab `[TBD]` placeholders (follow-up edit once ethics docs are final).

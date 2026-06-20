# Smoke Test Checklist — TikTok Analytics Exporter

Run before shipping a new build to a participant.

## Setup
- [ ] `chrome://extensions` → Developer mode on → Load unpacked → select `tiktok-analytics-exporter/`.
- [ ] Pin extension to toolbar.
- [ ] Log into a real (test) TikTok account.

## UI shell
- [ ] Popup opens with **Export** and **Help** tabs only (no Single video, no visible Debug).
- [ ] Help tab: all 5 accordion sections render; first one is open by default.
- [ ] Triple-click `v0.2.0` footer → **Debug** tab reveals; clicking it shows the URL log.

## Step 1 — Video performance
- [ ] On `https://www.tiktok.com/` (not Studio), module 1 shows **Open my Content page** button only.
- [ ] Click that button → tab navigates to `/tiktokstudio/content` → reopen popup → module 1 shows green **You're on the Content page**.
- [ ] Click **Extract video stats** → progress bar fills → reaches **Done**.
- [ ] Click **Save** → CSV downloads as `tiktok_videos_{handle}_{YYYY-MM-DD}.csv`.
- [ ] Open the CSV: header has all 23 columns; at least one data row exists; ECR and NAWP are populated.

## Step 2 — Follower history
- [ ] Click **Open my Followers analytics** → tab navigates to `/tiktokstudio/analytics/followers?dateRange=...` → reopen popup → module 2 shows green **You're on the Followers page**.
- [ ] Click **Extract follower history** → progress shows → reaches **Done**.
- [ ] Click **Save** → CSV downloads as `tiktok_followers_{handle}_{YYYY-MM-DD}.csv`.
- [ ] Open the CSV: 365 rows; last row date = yesterday; pre-account rows have `data_quality=no_data` and blank counts; `creator_handle` populated.

## End-of-flow
- [ ] After both steps done, strip counter shows **2 of 2 ready**; dark **Both files are ready** banner visible.
- [ ] Reopen the popup later — state persists; both **Save** buttons still work.

## Recovery
- [ ] Mid-Step-1: click **Cancel** → module 1 returns to **ready** state.
- [ ] Mid-Step-1: close popup → reopen 10 s later → progress resumes.
- [ ] Debug → **Reset state** → both modules return to **idle** / **ready** (depending on page).

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

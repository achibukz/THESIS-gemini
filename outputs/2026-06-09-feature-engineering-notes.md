# Feature Engineering Notes — Context & Creator Features

**Date:** 2026-06-09
**Scope:** Pre-publication feature validity, temporal leakage, and data availability for the SFV engagement prediction model.

---

## The Core Constraint

The model is **pre-publication**: every feature used at inference time must be knowable *before* the video is posted. This rules out any feature derived from post-publication signals (views, shares, algorithmic traffic). It also creates a temporal leakage risk for creator-level features collected retrospectively.

---

## Post Time

### Validity
Post time is a valid pre-publication feature. The creator decides when to post, and that decision causally influences engagement outcomes through the following chain:

```
post_time → initial audience reach → early engagement signal → algorithmic amplification → final ECR/NAWP
```

Even though ECR and NAWP are measured after the fact (cumulative totals at export time), post time is still one of the inputs that determined those outcomes. No leakage.

### Encoding
Encode cyclically to preserve continuity (e.g., 11 PM and midnight are adjacent):

```python
import numpy as np

df['hour_sin'] = np.sin(2 * np.pi * df['post_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['post_hour'] / 24)
df['dow_sin']  = np.sin(2 * np.pi * df['post_dow'] / 7)
df['dow_cos']  = np.cos(2 * np.pi * df['post_dow'] / 7)
```

Do not use raw integer hour or day-of-week — linear encoding breaks the circular relationship.

---

## Follower Count

### Problem
The Chrome extension exports `follower_count` as a field, but it is currently **empty** in all exports — the scrape is not implemented yet.

**Extension TODO:** Update the Chrome extension to scrape `follower_count` from the TikTok creator profile or TikTok Studio. The scraped value will be the count at export time.

### Temporal Leakage
The scraped `follower_count` reflects the creator's current audience at export time, not at the time each video was posted. For a creator whose video history spans several months, this introduces leakage — the model would see a higher follower count than existed when older videos were posted.

### Mitigation: Historical Reconstruction
The export already contains `new_followers` per video — the number of followers each video contributed. Use this to roll back the follower count to each video's post date:

```python
# Sort newest-first (as TikTok exports)
df = df.sort_values('post_date', ascending=False).reset_index(drop=True)

# current_follower_count = scraped from profile at export time
running = current_follower_count
follower_at_post = []

for _, row in df.iterrows():
    follower_at_post.append(running)
    running -= row['new_followers']

df['follower_count_at_post'] = follower_at_post
```

**Limitation:** This only accounts for followers gained from the exported videos. Followers from profile visits, other sources, or videos outside the export window are not captured. Document this as an approximation in §4.1.

---

## Creator Age

### Problem
TikTok Studio does not expose account creation date through any interface accessible to the creator or via scraping.

### Computation from Export
Creator age at post time is computed from the creator's own export data. The earliest `post_date` in the export is treated as the creator's known start date on the platform. Creator age for each video is then the number of days between that first post and the current video's post date:

```python
import pandas as pd

df = df.sort_values('post_date')
df['first_post_date'] = df.groupby('creator_uid')['post_date'].transform('min')
df['creator_age_at_post_days'] = (
    pd.to_datetime(df['post_date']) - pd.to_datetime(df['first_post_date'])
).dt.days
```

This gives `creator_age_at_post_days = 0` for the creator's first exported video and grows with each subsequent post. It measures **time as an active poster**, not time since account registration.

**Limitation:** If TikTok Studio truncates the export to the most recent N videos, the computed `first_post_date` is later than the true account start — so creator age is systematically underestimated for creators with long histories. Document this in §4.1.

### Videos Posted Before
Retain `videos_posted_before` as a separate feature alongside creator age. The two capture different dimensions: one measures elapsed time, the other measures output volume.

```python
df['videos_posted_before'] = df.groupby('creator_uid').cumcount()
```

---

## Summary: Creator Feature Set

| Feature | Source | Status | Notes |
|---|---|---|---|
| `post_hour`, `post_dow` | export (`post_time` column) | Ready | Encode cyclically |
| `follower_count_at_post` | scraped current count + `new_followers` rollback | Pending extension update | Approximate; document limitation |
| `videos_posted_before` | computed from export sort order | Ready | Lower bound if export is truncated |
| `creator_age_at_post_days` | computed: `post_date - min(post_date)` per creator | Ready | Underestimated if export is truncated; document in §4.1 |
| `videos_posted_before` | computed from export sort order | Ready | Lower bound if export is truncated |

---

## Label Note

ECR and NAWP in the CSV are cumulative totals as of export time, not point-in-time values at posting. This is expected and does not invalidate the feature set — the model learns `[content, creator, post_time] → eventual ECR/NAWP`. The only risk is if videos were exported at very different points in their lifecycle (e.g., some at 7 days, some at 1 year), which would add noise to the labels. Consider filtering the dataset to videos at least N days old at export time to reduce this variance.

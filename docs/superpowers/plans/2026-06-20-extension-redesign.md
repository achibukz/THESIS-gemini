# Chrome Extension Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the TikTok Analytics Exporter popup to mockup `option-b-modular-v2.html` and add a follower-history CSV export, while keeping the existing video-performance export working.

**Architecture:** Extend the existing `background.js` orchestrator with a parallel `followerStep` state branch that reuses the captured insight-API template. Split `state.phase` into independent `videoStep.phase` / `followerStep.phase` branches so the two flows can finish in any order. Extract pure-function parsers into `lib/parsers.js` for unit testing.

**Tech Stack:** Manifest V3 Chrome extension, vanilla JS (ES modules in background.js, classic scripts in content.js/injected.js), `chrome.storage.session` for state, `chrome.downloads` for CSV delivery, Node + `node:assert` for parser unit tests (zero npm deps).

**Source spec:** `docs/superpowers/specs/2026-06-20-extension-redesign-design.md`
**Visual reference:** `tiktok-analytics-exporter/mockups/option-b-modular-v2.html`

---

## Phase 1 — Test scaffolding + pure-function parsers (TDD)

### Task 1: Set up Node test runner

**Files:**
- Create: `tiktok-analytics-exporter/tests/run.js`
- Create: `tiktok-analytics-exporter/tests/lib/assert-helpers.js`

- [ ] **Step 1: Create the test runner**

Create `tiktok-analytics-exporter/tests/run.js`:

```js
import { readdir } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

let passed = 0, failed = 0;
const failures = [];

globalThis.test = async (name, fn) => {
  try {
    await fn();
    passed++;
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
  } catch (err) {
    failed++;
    failures.push({ name, err });
    console.log(`  \x1b[31m✗\x1b[0m ${name}`);
    console.log(`      ${err.message}`);
  }
};

globalThis.describe = async (name, fn) => {
  console.log(`\n${name}`);
  await fn();
};

const files = (await readdir(here, { recursive: true }))
  .filter((f) => f.endsWith('.test.js'))
  .sort();

for (const f of files) {
  await import(pathToFileURL(join(here, f)).href);
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log('\nFailures:');
  for (const { name, err } of failures) {
    console.log(`  - ${name}: ${err.stack || err.message}`);
  }
  process.exit(1);
}
```

- [ ] **Step 2: Create the assert helpers**

Create `tiktok-analytics-exporter/tests/lib/assert-helpers.js`:

```js
import { strict as assert } from 'node:assert';

export function assertRowEquals(actual, expected, msg) {
  for (const key of Object.keys(expected)) {
    assert.equal(actual[key], expected[key], `${msg || ''} field "${key}"`);
  }
}

export function assertArrayLength(actual, expected, msg) {
  assert.equal(actual.length, expected, `${msg || 'array length'}: got ${actual.length}, expected ${expected}`);
}

export { assert };
```

- [ ] **Step 3: Verify the runner runs (with zero tests)**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `0 passed, 0 failed` (exit 0).

- [ ] **Step 4: Commit**

```bash
git add tiktok-analytics-exporter/tests/run.js tiktok-analytics-exporter/tests/lib/assert-helpers.js
git commit -m "add: zero-dep node test runner for extension parsers"
```

---

### Task 2: Extract `parseInsightResponse` into `lib/parsers.js` (TDD)

**Goal:** Move the existing parsing logic out of `background.js` into a testable ES module. No behaviour change.

**Files:**
- Create: `tiktok-analytics-exporter/lib/parsers.js`
- Create: `tiktok-analytics-exporter/tests/fixtures/insight-minimal.json`
- Create: `tiktok-analytics-exporter/tests/parsers/parseInsightResponse.test.js`
- Modify: `tiktok-analytics-exporter/background.js` (import from lib)

- [ ] **Step 1: Create a minimal insight fixture**

Create `tiktok-analytics-exporter/tests/fixtures/insight-minimal.json`. This shape mirrors a real per-video insight response trimmed to the fields the parser uses:

```json
{
  "data": {
    "status": 0,
    "video_info": {
      "aweme_id": "7000000000000000001",
      "create_time": 1750000000,
      "desc": "test caption",
      "duration": 29000,
      "statistics": {
        "digg_count": 12,
        "comment_count": 2,
        "share_count": 5
      },
      "author": {
        "uid": "TEST_UID",
        "unique_id": "test_user"
      }
    },
    "video_retention_rate_realtime": {
      "value": { "list": [{ "timestamp": "5000", "value": 0.46 }] }
    },
    "video_per_duration_realtime": { "value": { "value": 12.1 } },
    "video_finish_rate_realtime": { "value": { "value": 0.1273 } },
    "video_traffic_source_percent_realtime": {
      "value": { "value": [
        { "key": "For You", "value": 0.927 },
        { "key": "Follow", "value": 0.004 },
        { "key": "Personal Profile", "value": 0.034 },
        { "key": "Search", "value": 0.003 }
      ]}
    },
    "video_new_followers": { "value": { "value": 2 } },
    "realtime_total_video_views": { "value": { "value": 744 } }
  }
}
```

- [ ] **Step 2: Write the failing test**

Create `tiktok-analytics-exporter/tests/parsers/parseInsightResponse.test.js`:

```js
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseInsightResponse } from '../../lib/parsers.js';
import { assertRowEquals, assert } from '../lib/assert-helpers.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, '../fixtures/insight-minimal.json'), 'utf8'));

describe('parseInsightResponse', () => {
  test('parses ECR, NAWP, traffic, and core stats from a complete payload', () => {
    const result = parseInsightResponse(fx, { aweme_id: '7000000000000000001', create_time: 1750000000 });
    assert.equal(result.ok, true);
    assertRowEquals(result.row, {
      video_id: '7000000000000000001',
      duration_ms: 29000,
      views: 744,
      likes: 12,
      comments: 2,
      shares: 5,
      ECR: 0.46,
      avg_watch_time_s: 12.1,
      watched_full_pct: 0.1273,
      traffic_foryou_pct: 0.927,
      traffic_follow_pct: 0.004,
      traffic_profile_pct: 0.034,
      traffic_search_pct: 0.003,
      new_followers: 2,
      creator_uid: 'TEST_UID',
      creator_handle: 'test_user',
      data_quality: ''
    });
    assert.equal(result.row.NAWP, 0.417241);
  });

  test('flags status:2 as insufficient_data', () => {
    const bad = JSON.parse(JSON.stringify(fx));
    bad.data.status = 2;
    const result = parseInsightResponse(bad, { aweme_id: '7000000000000000001' });
    assert.ok(result.row.data_quality.includes('insufficient_data'));
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: FAIL with `Cannot find module ../../lib/parsers.js`.

- [ ] **Step 4: Create `lib/parsers.js` and move the parser**

Create `tiktok-analytics-exporter/lib/parsers.js`. Copy the existing functions from `background.js` lines ~551–696 (`parseInsightResponse`, `findInsight`, `findFirstByKey`, `readNumericValue`, `readRetentionAt`, `readTrafficSources`, `formatUnixDate`, `formatUnixTime`, `roundTo`) into this file. Add `export` to the public functions:

```js
export function parseInsightResponse(json, video) {
  const data = json?.data || json;
  if (!data) return { ok: false, reason: 'empty response' };

  const statusFlag = data?.status ?? json?.status;
  const dataQualityIssues = [];
  if (statusFlag === 2) dataQualityIssues.push('insufficient_data');

  const videoInfo =
    data?.video_info || data?.aweme_info || findFirstByKey(data, 'video_info') || {};
  const stats = videoInfo?.statistics || {};

  const retention = findInsight(data, 'video_retention_rate_realtime');
  const perDuration = findInsight(data, 'video_per_duration_realtime');
  const finishRate = findInsight(data, 'video_finish_rate_realtime');
  const trafficSource = findInsight(data, 'video_traffic_source_percent_realtime');
  const newFollowers = findInsight(data, 'video_new_followers');
  const totalViews = findInsight(data, 'realtime_total_video_views');

  const ecr = readRetentionAt(retention, '5000');
  if (ecr == null && statusFlag !== 2) dataQualityIssues.push('missing_ecr');

  const avgWatchTimeS = readNumericValue(perDuration);
  const durationMs = videoInfo?.video?.duration ?? videoInfo?.duration ?? video.duration_ms ?? null;
  const nawp = avgWatchTimeS != null && durationMs ? avgWatchTimeS / (durationMs / 1000) : null;

  const traffic = readTrafficSources(trafficSource);
  const createTs = videoInfo?.create_time ?? video.create_time;

  const row = {
    video_id: video.aweme_id,
    post_date: formatUnixDate(createTs),
    post_time: formatUnixTime(createTs),
    caption: videoInfo?.desc ?? video.desc ?? '',
    duration_ms: durationMs ?? video.duration_ms ?? '',
    views: readNumericValue(totalViews) ?? stats.play_count ?? '',
    likes: stats.digg_count ?? video.digg_count ?? '',
    comments: stats.comment_count ?? video.comment_count ?? '',
    shares: stats.share_count ?? video.share_count ?? '',
    ECR: ecr ?? '',
    avg_watch_time_s: avgWatchTimeS ?? '',
    NAWP: nawp != null ? roundTo(nawp, 6) : '',
    watched_full_pct: readNumericValue(finishRate) ?? '',
    traffic_foryou_pct: traffic.foryou ?? '',
    traffic_follow_pct: traffic.follow ?? '',
    traffic_profile_pct: traffic.profile ?? '',
    traffic_search_pct: traffic.search ?? '',
    new_followers: readNumericValue(newFollowers) ?? '',
    creator_uid: videoInfo?.author?.uid ?? '',
    creator_handle: videoInfo?.author?.unique_id ?? '',
    follower_count: '',
    account_created_date: '',
    data_quality: dataQualityIssues.join('|')
  };
  return { ok: true, row };
}

function findInsight(data, insighType) {
  if (!data || typeof data !== 'object') return null;
  if (Object.prototype.hasOwnProperty.call(data, insighType)) return data[insighType];
  const stack = [data];
  while (stack.length) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;
    if (Array.isArray(node)) {
      for (const item of node) {
        if (item && typeof item === 'object') {
          if (item.insigh_type === insighType || item.insight_type === insighType) return item;
          stack.push(item);
        }
      }
    } else {
      if (Object.prototype.hasOwnProperty.call(node, insighType)) return node[insighType];
      for (const key of Object.keys(node)) {
        const v = node[key];
        if (v && typeof v === 'object') stack.push(v);
      }
    }
  }
  return null;
}

function findFirstByKey(obj, key) {
  if (!obj || typeof obj !== 'object') return null;
  const stack = [obj];
  while (stack.length) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;
    if (Object.prototype.hasOwnProperty.call(node, key)) return node[key];
    if (Array.isArray(node)) for (const v of node) stack.push(v);
    else for (const k of Object.keys(node)) stack.push(node[k]);
  }
  return null;
}

function readNumericValue(node) {
  if (node == null) return null;
  const v = node?.value;
  if (v == null) return null;
  if (typeof v === 'object' && 'value' in v) {
    const n = Number(v.value);
    return Number.isFinite(n) ? n : null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function readRetentionAt(node, timestamp) {
  const list = node?.value?.list ?? node?.value?.value?.list ?? node?.list;
  if (!Array.isArray(list)) return null;
  const target = String(timestamp);
  const entry = list.find((e) => String(e?.timestamp) === target);
  if (!entry) return null;
  const v = entry.value;
  const n = Number(typeof v === 'object' ? v?.value : v);
  return Number.isFinite(n) ? n : null;
}

function readTrafficSources(node) {
  const out = { foryou: null, follow: null, profile: null, search: null };
  const list = node?.value?.value ?? node?.value?.list ?? node?.value;
  if (!Array.isArray(list)) return out;
  for (const entry of list) {
    const key = (entry?.key ?? entry?.name ?? '').toString();
    const val = Number(entry?.value ?? entry?.percent ?? 0);
    if (!Number.isFinite(val)) continue;
    const norm = key.toLowerCase();
    if (norm === 'for you') out.foryou = val;
    else if (norm === 'follow') out.follow = val;
    else if (norm === 'personal profile') out.profile = val;
    else if (norm === 'search') out.search = val;
  }
  return out;
}

export function formatUnixDate(unixSeconds) {
  if (!unixSeconds) return '';
  const ms = unixSeconds > 1e12 ? unixSeconds : unixSeconds * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

export function formatUnixTime(unixSeconds) {
  if (!unixSeconds) return '';
  const ms = unixSeconds > 1e12 ? unixSeconds : unixSeconds * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function roundTo(n, decimals) {
  const k = 10 ** decimals;
  return Math.round(n * k) / k;
}
```

- [ ] **Step 5: Update `background.js` to import from the module**

In `tiktok-analytics-exporter/background.js`:
- Add at the top of the file: `import { parseInsightResponse, formatUnixDate, formatUnixTime } from './lib/parsers.js';`
- Delete the local definitions of `parseInsightResponse`, `findInsight`, `findFirstByKey`, `readNumericValue`, `readRetentionAt`, `readTrafficSources`, `formatUnixDate`, `formatUnixTime`, `roundTo` (now lives in `lib/parsers.js`).

- [ ] **Step 6: Run tests + smoke-load the extension**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `2 passed, 0 failed`.

Then in Chrome: `chrome://extensions` → reload the extension. Open the popup. Confirm there are no console errors in the service-worker DevTools. (Functional test is deferred to Phase 4.)

- [ ] **Step 7: Commit**

```bash
git add tiktok-analytics-exporter/lib/parsers.js tiktok-analytics-exporter/tests/fixtures/insight-minimal.json tiktok-analytics-exporter/tests/parsers/parseInsightResponse.test.js tiktok-analytics-exporter/background.js
git commit -m "refactor: extract insight parser into lib/parsers.js with tests"
```

---

### Task 3: `buildFollowerHistoryURL` (TDD)

**Files:**
- Modify: `tiktok-analytics-exporter/lib/parsers.js` (add `buildFollowerHistoryURL`)
- Create: `tiktok-analytics-exporter/tests/parsers/buildFollowerHistoryURL.test.js`

- [ ] **Step 1: Write the failing test**

Create `tiktok-analytics-exporter/tests/parsers/buildFollowerHistoryURL.test.js`:

```js
import { buildFollowerHistoryURL } from '../../lib/parsers.js';
import { assert } from '../lib/assert-helpers.js';

describe('buildFollowerHistoryURL', () => {
  test('appends the three follower type_requests with days=732 end_days=1', () => {
    const template = 'https://www.tiktok.com/aweme/v2/data/insight/?aid=1988&locale=en';
    const url = buildFollowerHistoryURL(template);
    assert.ok(url.startsWith('https://www.tiktok.com/aweme/v2/data/insight/'));
    assert.ok(url.includes('aid=1988'));
    const typeReq = decodeURIComponent(new URL(url).searchParams.get('type_requests'));
    const parsed = JSON.parse(typeReq);
    assert.equal(parsed.length, 3);
    assert.deepEqual(parsed[0], { insigh_type: 'follower_num_history', days: 732, end_days: 1 });
    assert.deepEqual(parsed[1], { insigh_type: 'follower_num',         days: 732, end_days: 1 });
    assert.deepEqual(parsed[2], { insigh_type: 'net_follower_history', days: 732, end_days: 1 });
  });

  test('falls back to default base when template is empty', () => {
    const url = buildFollowerHistoryURL(null);
    assert.ok(url.startsWith('https://www.tiktok.com/aweme/v2/data/insight/'));
    assert.ok(url.includes('aid=1988'));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: FAIL with `buildFollowerHistoryURL is not a function`.

- [ ] **Step 3: Implement in `lib/parsers.js`**

Append to `tiktok-analytics-exporter/lib/parsers.js`:

```js
const DEFAULT_INSIGHT_BASE_FOR_FOLLOWERS =
  'https://www.tiktok.com/aweme/v2/data/insight/?aid=1988&app_language=en&app_name=tiktok_creator_center&device_platform=web_pc&locale=en&channel=tiktok_web&os=mac';

const FOLLOWER_TYPE_REQUESTS = [
  { insigh_type: 'follower_num_history', days: 732, end_days: 1 },
  { insigh_type: 'follower_num',         days: 732, end_days: 1 },
  { insigh_type: 'net_follower_history', days: 732, end_days: 1 }
];

export function buildFollowerHistoryURL(template) {
  const base = template || DEFAULT_INSIGHT_BASE_FOR_FOLLOWERS;
  const sep = base.includes('?') ? '&' : '?';
  return `${base}${sep}type_requests=${encodeURIComponent(JSON.stringify(FOLLOWER_TYPE_REQUESTS))}`;
}
```

- [ ] **Step 4: Run tests**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `4 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add tiktok-analytics-exporter/lib/parsers.js tiktok-analytics-exporter/tests/parsers/buildFollowerHistoryURL.test.js
git commit -m "add: buildFollowerHistoryURL parser + tests"
```

---

### Task 4: `mapIndexToDate` (TDD)

**Files:**
- Modify: `tiktok-analytics-exporter/lib/parsers.js`
- Create: `tiktok-analytics-exporter/tests/parsers/mapIndexToDate.test.js`

- [ ] **Step 1: Write the failing test**

Create `tiktok-analytics-exporter/tests/parsers/mapIndexToDate.test.js`:

```js
import { mapIndexToDate } from '../../lib/parsers.js';
import { assert } from '../lib/assert-helpers.js';

describe('mapIndexToDate', () => {
  test('maps the anchor index to the anchor date', () => {
    // anchor: index 100 → 2026-06-19
    const anchor = new Date(Date.UTC(2026, 5, 19));
    assert.equal(mapIndexToDate(100, 200, 100, anchor), '2026-06-19');
  });

  test('walks backward one day per index decrement', () => {
    const anchor = new Date(Date.UTC(2026, 5, 19)); // June 19
    assert.equal(mapIndexToDate(99, 200, 100, anchor), '2026-06-18');
    assert.equal(mapIndexToDate(98, 200, 100, anchor), '2026-06-17');
  });

  test('walks forward one day per index increment', () => {
    const anchor = new Date(Date.UTC(2026, 5, 19));
    assert.equal(mapIndexToDate(101, 200, 100, anchor), '2026-06-20');
  });

  test('handles month boundaries', () => {
    const anchor = new Date(Date.UTC(2026, 5, 1)); // June 1
    assert.equal(mapIndexToDate(99, 200, 100, anchor), '2026-05-31');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: FAIL with `mapIndexToDate is not a function`.

- [ ] **Step 3: Implement**

Append to `tiktok-analytics-exporter/lib/parsers.js`:

```js
export function mapIndexToDate(i, _length, anchorIndex, anchorDate) {
  const delta = i - anchorIndex;
  const d = new Date(Date.UTC(
    anchorDate.getUTCFullYear(),
    anchorDate.getUTCMonth(),
    anchorDate.getUTCDate() + delta
  ));
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`;
}
```

- [ ] **Step 4: Run tests**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `8 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add tiktok-analytics-exporter/lib/parsers.js tiktok-analytics-exporter/tests/parsers/mapIndexToDate.test.js
git commit -m "add: mapIndexToDate parser + tests"
```

---

### Task 5: `parseFollowerHistoryResponse` (TDD)

**Files:**
- Modify: `tiktok-analytics-exporter/lib/parsers.js`
- Create: `tiktok-analytics-exporter/tests/fixtures/follower-history-synth.json`
- Create: `tiktok-analytics-exporter/tests/parsers/parseFollowerHistoryResponse.test.js`

- [ ] **Step 1: Create a synthetic follower-history fixture**

Create `tiktok-analytics-exporter/tests/fixtures/follower-history-synth.json` — 10-entry slim version that exercises all cases (pre-account `status:2`, growing follower counts, signed daily nets):

```json
{
  "status_code": 0,
  "status_msg": "",
  "follower_num": { "status": 0, "value": 18 },
  "follower_num_history": [
    { "status": 2 },
    { "status": 2 },
    { "status": 0, "value": 0 },
    { "status": 0, "value": 1 },
    { "status": 0, "value": 3 },
    { "status": 0, "value": 7 },
    { "status": 0, "value": 12 },
    { "status": 0, "value": 17 },
    { "status": 0, "value": 18 },
    { "status": 2 }
  ],
  "net_follower_history": [
    { "status": 2 },
    { "status": 2 },
    { "status": 0, "value": 0 },
    { "status": 0, "value": 1 },
    { "status": 0, "value": 2 },
    { "status": 0, "value": 4 },
    { "status": 0, "value": 5 },
    { "status": 0, "value": 5 },
    { "status": 0, "value": 1 },
    { "status": 2 }
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `tiktok-analytics-exporter/tests/parsers/parseFollowerHistoryResponse.test.js`:

```js
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseFollowerHistoryResponse } from '../../lib/parsers.js';
import { assert, assertArrayLength } from '../lib/assert-helpers.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, '../fixtures/follower-history-synth.json'), 'utf8'));

describe('parseFollowerHistoryResponse', () => {
  test('anchors the last status:0 row to yesterday when its value matches follower_num', () => {
    const now = new Date(Date.UTC(2026, 5, 20)); // June 20
    const result = parseFollowerHistoryResponse(fx, now, { profile: null, limitDays: 365 });
    assert.equal(result.ok, true);
    // last status:0 index is 8 (value 18), matches follower_num.value=18 → date = June 19
    const yesterday = result.rows.find((r) => r.date === '2026-06-19');
    assert.ok(yesterday, 'expected a row dated 2026-06-19');
    assert.equal(yesterday.follower_count, 18);
    assert.equal(yesterday.daily_net, 1);
    assert.equal(yesterday.data_quality, '');
  });

  test('falls back to today anchoring when last status:0 value does not match follower_num', () => {
    const tweaked = JSON.parse(JSON.stringify(fx));
    tweaked.follower_num.value = 99; // mismatch with last status:0 value (18)
    const now = new Date(Date.UTC(2026, 5, 20));
    const result = parseFollowerHistoryResponse(tweaked, now, { profile: null, limitDays: 365 });
    assert.equal(result.ok, true);
    // last status:0 row (index 8) is mapped to today → 2026-06-20
    const today = result.rows.find((r) => r.date === '2026-06-20');
    assert.ok(today, 'expected a row dated 2026-06-20');
    assert.equal(today.follower_count, 18);
  });

  test('emits empty count and data_quality=no_data on status:2 rows', () => {
    const now = new Date(Date.UTC(2026, 5, 20));
    const result = parseFollowerHistoryResponse(fx, now, { profile: null, limitDays: 365 });
    const noData = result.rows.filter((r) => r.data_quality === 'no_data');
    assert.equal(noData.length, 3); // indices 0, 1, 9
    for (const r of noData) {
      assert.equal(r.follower_count, '');
      assert.equal(r.daily_net, '');
    }
  });

  test('attaches creator_handle and creator_uid from profile', () => {
    const now = new Date(Date.UTC(2026, 5, 20));
    const result = parseFollowerHistoryResponse(fx, now, {
      profile: { creator_handle: 'test_user', creator_uid: 'TEST_UID' },
      limitDays: 365
    });
    for (const r of result.rows) {
      assert.equal(r.creator_handle, 'test_user');
      assert.equal(r.creator_uid, 'TEST_UID');
    }
  });

  test('trims to limitDays rows', () => {
    const now = new Date(Date.UTC(2026, 5, 20));
    const result = parseFollowerHistoryResponse(fx, now, { profile: null, limitDays: 5 });
    assertArrayLength(result.rows, 5, 'rows trimmed to limitDays');
  });

  test('returns ok:false when follower_num_history is missing', () => {
    const result = parseFollowerHistoryResponse({ status_code: 0 }, new Date(), { profile: null, limitDays: 365 });
    assert.equal(result.ok, false);
    assert.match(result.reason, /follower_num_history/);
  });

  test('returns ok:false when status_code is nonzero', () => {
    const result = parseFollowerHistoryResponse({ status_code: 7 }, new Date(), { profile: null, limitDays: 365 });
    assert.equal(result.ok, false);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: FAIL with `parseFollowerHistoryResponse is not a function`.

- [ ] **Step 4: Implement**

Append to `tiktok-analytics-exporter/lib/parsers.js`:

```js
export function parseFollowerHistoryResponse(json, now, opts) {
  const { profile, limitDays } = opts;
  if (!json) return { ok: false, reason: 'empty response' };
  if (json.status_code !== 0 && json.status_code !== undefined) {
    return { ok: false, reason: `status_code=${json.status_code}` };
  }
  const hist = json.follower_num_history;
  if (!Array.isArray(hist) || hist.length === 0) {
    return { ok: false, reason: 'missing follower_num_history' };
  }
  const netHist = Array.isArray(json.net_follower_history) ? json.net_follower_history : [];
  const currentFollowerNum = json.follower_num?.value;

  // Anchor: find last index where status:0
  let anchorIndex = -1;
  for (let i = hist.length - 1; i >= 0; i--) {
    if (hist[i]?.status === 0) { anchorIndex = i; break; }
  }
  // If no status:0 anywhere, anchor last index to yesterday (degenerate but stable)
  if (anchorIndex === -1) anchorIndex = hist.length - 1;

  // Anchor date: yesterday if last status:0 value matches current follower_num, else today
  const yesterday = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() - 1));
  const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const anchorMatches = hist[anchorIndex]?.value === currentFollowerNum;
  const anchorDate = anchorMatches ? yesterday : today;

  const allRows = [];
  for (let i = 0; i < hist.length; i++) {
    const date = mapIndexToDate(i, hist.length, anchorIndex, anchorDate);
    const entry = hist[i] || {};
    const netEntry = netHist[i] || {};
    const isNoData = entry.status === 2;
    allRows.push({
      date,
      follower_count: isNoData ? '' : (entry.value ?? ''),
      daily_net: netEntry.status === 2 ? '' : (netEntry.value ?? ''),
      creator_handle: profile?.creator_handle ?? '',
      creator_uid: profile?.creator_uid ?? '',
      data_quality: isNoData ? 'no_data' : ''
    });
  }

  // Sort by date ascending so the most recent `limitDays` are the tail
  allRows.sort((a, b) => (a.date < b.date ? -1 : 1));
  const rows = limitDays > 0 ? allRows.slice(-limitDays) : allRows;
  return { ok: true, rows };
}
```

- [ ] **Step 5: Run tests**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `15 passed, 0 failed`.

- [ ] **Step 6: Commit**

```bash
git add tiktok-analytics-exporter/lib/parsers.js tiktok-analytics-exporter/tests/fixtures/follower-history-synth.json tiktok-analytics-exporter/tests/parsers/parseFollowerHistoryResponse.test.js
git commit -m "add: parseFollowerHistoryResponse parser + tests"
```

---

## Phase 2 — Background.js refactor (state machine)

### Task 6: Split `state.phase` into per-step branches

**Goal:** Refactor `state` so `phase`, `dateRange`, `videos`, `rows`, `skipped`, `progress` live under `state.videoStep`. Keep behaviour identical for the existing video flow.

**Files:**
- Modify: `tiktok-analytics-exporter/background.js`

- [ ] **Step 1: Refactor `defaultState()`**

Replace the existing `defaultState()` in `background.js` with:

```js
function defaultState() {
  return {
    // shared
    insightTemplate: null,
    profileTemplate: null,
    profile: null,
    recentURLs: [],
    interceptCounts: { videoList: 0, insight: 0, profile: 0 },
    lastVideoListSample: null,

    videoStep: {
      phase: 'idle',
      activeTabId: null,
      dateRange: null,
      videos: {},
      rows: [],
      skipped: [],
      progress: { current: 0, total: 0, message: '' },
      startedAt: null,
      finishedAt: null,
      error: null
    },

    followerStep: {
      phase: 'idle',
      activeTabId: null,
      rows: [],
      progress: { message: '' },
      startedAt: null,
      finishedAt: null,
      error: null
    }
  };
}
```

- [ ] **Step 2: Update every reference to old top-level fields**

Throughout `background.js`, replace these references:
- `s.phase` → `s.videoStep.phase`
- `s.dateRange` → `s.videoStep.dateRange`
- `s.videos` / `state.videos` → `s.videoStep.videos` / `state.videoStep.videos`
- `s.rows` / `state.rows` → `s.videoStep.rows` / `state.videoStep.rows`
- `s.skipped` / `state.skipped` → `s.videoStep.skipped` / `state.videoStep.skipped`
- `s.progress` / `state.progress` → `s.videoStep.progress` / `state.videoStep.progress`
- `s.activeTabId` → `s.videoStep.activeTabId`
- `s.startedAt`, `s.finishedAt`, `s.error` → `s.videoStep.startedAt`, `.finishedAt`, `.error`

Specifically: `ingestVideoList`, `startExport`, `runExport`, `filterVideosByDate`, `handleMessage` for `cancel-export`.

- [ ] **Step 3: Rename `startExport` / `runExport` and the message type**

- Rename function `startExport` → `startVideoExport`.
- Rename function `runExport` → `runVideoExport`.
- In `handleMessage`, change `case 'start-export'` → `case 'start-video-export'` (delegating to `startVideoExport`).
- In `handleMessage`, change `case 'cancel-export'` → `case 'cancel-video-export'`, mutating `s.videoStep.phase = 'cancelled'`.

- [ ] **Step 4: Smoke-load**

Open `chrome://extensions` → reload the extension. Open the popup. Open service-worker DevTools → console. Confirm no errors on load. (The popup will be broken since it still uses old message names — that's expected, fixed in Phase 4.)

- [ ] **Step 5: Run parser tests (should still pass — they don't touch state)**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `15 passed, 0 failed`.

- [ ] **Step 6: Commit**

```bash
git add tiktok-analytics-exporter/background.js
git commit -m "refactor: split state into videoStep/followerStep branches"
```

---

### Task 7: Remove `single-video-fetch` plumbing

**Files:**
- Modify: `tiktok-analytics-exporter/background.js`

- [ ] **Step 1: Delete the function and its case**

In `background.js`:
- Delete the `case 'single-video-fetch':` line and the `return singleVideoFetch(...)` line from `handleMessage`.
- Delete the entire `singleVideoFetch(tabId, awemeId)` function (lines ~474–513).
- Delete the helper `extractAwemeId(input)` (lines ~515–522) — only used by `singleVideoFetch`.
- Delete the import or local `buildInsightURL` if it's no longer used. (It IS still used by `fetchInsightRow`. Keep it.)

- [ ] **Step 2: Smoke-load**

Reload the extension. Service-worker DevTools → no errors.

- [ ] **Step 3: Run parser tests**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `15 passed, 0 failed`.

- [ ] **Step 4: Commit**

```bash
git add tiktok-analytics-exporter/background.js
git commit -m "remove: unused single-video-fetch plumbing from background"
```

---

### Task 8: Add `followerStep` message handlers + orchestrator

**Files:**
- Modify: `tiktok-analytics-exporter/background.js`

- [ ] **Step 1: Add the new message cases**

In `background.js`'s `handleMessage`, add these cases above the `default`:

```js
case 'start-follower-export':
  return startFollowerExport(msg.tabId);
case 'cancel-follower-export':
  await mutateState((s) => {
    s.followerStep.phase = 'cancelled';
    s.followerStep.progress.message = 'Cancelled by user';
  });
  return { ok: true };
case 'reset-follower-step':
  await mutateState((s) => {
    s.followerStep = defaultState().followerStep;
  });
  return { ok: true };
case 'reset-video-step':
  await mutateState((s) => {
    s.videoStep = defaultState().videoStep;
  });
  return { ok: true };
```

- [ ] **Step 2: Import the new parsers**

At the top of `background.js`, update the existing import to:

```js
import {
  parseInsightResponse,
  formatUnixDate,
  formatUnixTime,
  buildFollowerHistoryURL,
  parseFollowerHistoryResponse
} from './lib/parsers.js';
```

- [ ] **Step 3: Implement `startFollowerExport` and `runFollowerExport`**

Append to `background.js` (before the `sendToTab` helper):

```js
async function startFollowerExport(tabId) {
  if (!tabId) return { ok: false, error: 'Missing tabId' };

  await mutateState((s) => {
    s.followerStep.phase = 'fetching';
    s.followerStep.activeTabId = tabId;
    s.followerStep.rows = [];
    s.followerStep.progress = { message: 'Fetching follower history…' };
    s.followerStep.startedAt = Date.now();
    s.followerStep.finishedAt = null;
    s.followerStep.error = null;
  });

  runFollowerExport(tabId).catch(async (err) => {
    console.error('[tt-exporter] follower export failed', err);
    await mutateState((s) => {
      s.followerStep.phase = 'error';
      s.followerStep.error = String(err?.message || err);
    });
  });
  return { ok: true };
}

async function runFollowerExport(tabId) {
  let state = await getState();
  if (state.followerStep.phase === 'cancelled') return;

  const url = buildFollowerHistoryURL(state.insightTemplate);
  let res = await sendToTab(tabId, { type: 'page-fetch', url }).catch(
    (err) => ({ ok: false, error: String(err) })
  );
  if (!res?.ok) {
    await sleep(3000);
    res = await sendToTab(tabId, { type: 'page-fetch', url }).catch(
      (err) => ({ ok: false, error: String(err) })
    );
  }
  if (!res?.ok || !res.body) {
    await mutateState((s) => {
      s.followerStep.phase = 'error';
      s.followerStep.error = res?.error || 'fetch failed';
    });
    return;
  }

  let json;
  try { json = JSON.parse(res.body); }
  catch {
    await mutateState((s) => {
      s.followerStep.phase = 'error';
      s.followerStep.error = 'invalid JSON in response';
    });
    return;
  }

  state = await getState();
  const parsed = parseFollowerHistoryResponse(json, new Date(), {
    profile: state.profile,
    limitDays: 365
  });
  if (!parsed.ok) {
    await mutateState((s) => {
      s.followerStep.phase = 'error';
      s.followerStep.error = parsed.reason;
    });
    return;
  }

  const allNoData = parsed.rows.every((r) => r.data_quality === 'no_data');

  await mutateState((s) => {
    s.followerStep.phase = 'done';
    s.followerStep.rows = parsed.rows;
    s.followerStep.progress.message = allNoData
      ? 'Done. Your account may be too new for follower history.'
      : `Done. ${parsed.rows.length} days of follower data.`;
    s.followerStep.finishedAt = Date.now();
  });
}
```

- [ ] **Step 4: Smoke-load**

Reload the extension. Open service-worker DevTools → console. No errors. (No way to drive `start-follower-export` without the new popup — verified in Phase 4.)

- [ ] **Step 5: Run parser tests**

Run: `cd tiktok-analytics-exporter && node tests/run.js`
Expected: `15 passed, 0 failed`.

- [ ] **Step 6: Commit**

```bash
git add tiktok-analytics-exporter/background.js
git commit -m "add: followerStep orchestrator (startFollowerExport, runFollowerExport)"
```

---

## Phase 3 — Content script extension

### Task 9: Add `is-followers-page` check

**Files:**
- Modify: `tiktok-analytics-exporter/content.js`

- [ ] **Step 1: Add the URL check and message handler**

In `tiktok-analytics-exporter/content.js`:

Below the existing `isStudioContentPage()` function, add:

```js
function isFollowersAnalyticsPage() {
  return /\/(creator-center|tiktokstudio)\/analytics\/followers/i.test(window.location.pathname);
}
```

Inside `chrome.runtime.onMessage.addListener`, add a new branch alongside `is-studio-page`:

```js
if (msg?.type === 'is-followers-page') {
  sendResponse({ ok: true, isFollowers: isFollowersAnalyticsPage() });
  return false;
}
```

- [ ] **Step 2: Smoke-load**

Reload the extension. Navigate to `https://www.tiktok.com/tiktokstudio/analytics/followers`. Open the content-script console (right-click → Inspect → Console). No errors.

Optional sanity check: in the service-worker console, run:

```js
const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
await chrome.tabs.sendMessage(tab.id, { type: 'is-followers-page' });
```

Expected response: `{ ok: true, isFollowers: true }`.

- [ ] **Step 3: Commit**

```bash
git add tiktok-analytics-exporter/content.js
git commit -m "add: is-followers-page check in content script"
```

---

## Phase 4 — Popup UI rebuild

### Task 10: Rewrite `popup.html`

**Files:**
- Modify (full rewrite): `tiktok-analytics-exporter/popup.html`

- [ ] **Step 1: Replace the file with the production popup HTML**

Replace the contents of `tiktok-analytics-exporter/popup.html` with the structure below. This is the same structure as the approved mockup `mockups/option-b-modular-v2.html`, with:
- The mockup's inline `<style>` removed (replaced by `<link rel="stylesheet" href="popup.css">`).
- The mockup's inline `<script>` removed (replaced by `<script src="popup.js"></script>`).
- IDs that `popup.js` will hook into kept stable.
- A version footer with hidden Debug-tab triple-click handle.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Analytics Backup</title>
<link rel="stylesheet" href="popup.css" />
</head>
<body>

<div class="popup">
  <div class="head">
    <div class="brand">
      <div class="logo">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </div>
      <div>
        <h1>Analytics Backup</h1>
        <p>A companion for TikTok Studio</p>
      </div>
    </div>
    <div class="rights">
      <svg class="sh" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
      Exports only the stats TikTok already lets you, the creator, access.
    </div>
  </div>

  <div class="tabs">
    <button class="tab on" data-tab="export">Export</button>
    <button class="tab" data-tab="help">Help</button>
    <button class="tab hidden" data-tab="debug" id="tab-debug">Debug</button>
  </div>

  <!-- ============ EXPORT PANEL ============ -->
  <div class="panel on" id="p-export">
    <div class="strip">
      <div class="dots">
        <div class="d" id="dot1"><span class="pip">1</span><span class="t">Video stats</span></div>
        <span class="seg"></span>
        <div class="d" id="dot2"><span class="pip">2</span><span class="t">Followers</span></div>
      </div>
      <div class="count"><b id="cnt">0</b> of 2 ready</div>
    </div>

    <div class="body">

      <!-- MODULE 1: VIDEO PERFORMANCE -->
      <div class="mod video" id="mod1">
        <div class="mod-h">
          <div class="em">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="21" x2="21" y2="21"/><line x1="7" y1="21" x2="7" y2="12"/><line x1="12" y1="21" x2="12" y2="5"/><line x1="17" y1="21" x2="17" y2="9"/></svg>
          </div>
          <div class="tx"><b>Video performance</b><small>Per-video views, watch time &amp; retention</small></div>
          <span class="pill todo" id="pill1">Step 1</span>
        </div>
        <div class="mod-b">
          <div id="m1-idle">
            <div class="guide">
              <span class="ic"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></span>
              <p>This data lives on your <b>Content</b> page. Open it first, then come back here.</p>
            </div>
            <button class="btn open" id="m1-open">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
              Open my Content page
            </button>
            <div class="hintline">…or switch to that tab yourself, then reopen this popup</div>
          </div>
          <div id="m1-ready" class="hide">
            <div class="onpage"><span class="ic"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg></span> You're on the Content page — ready to extract.</div>
            <div class="range">
              <div class="rl">Date range</div>
              <div class="dates">
                <input type="date" id="m1-start" />
                <input type="date" id="m1-end" />
              </div>
              <div class="presets">
                <button data-days="30">30d</button>
                <button data-days="90" class="on">3 mo</button>
                <button data-days="365">1 yr</button>
                <button data-days="">All</button>
              </div>
            </div>
            <button class="btn extract" id="m1-extract">Extract video stats</button>
          </div>
          <div id="m1-run" class="hide">
            <div class="pbar"><i id="bar1"></i></div>
            <div class="pmeta" id="meta1">Starting…</div>
            <button class="btn ghost" id="m1-cancel">Cancel</button>
          </div>
          <div id="m1-done" class="hide">
            <div class="filerow">
              <span class="l"><span class="em2"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></span><span><b id="m1-file">video_performance.csv</b><small id="m1-summary"></small></span></span>
              <button class="dl" id="m1-save"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Save</button>
            </div>
            <details><summary>Skipped videos (<span id="m1-skipped-n">0</span>)</summary><ul id="m1-skipped"></ul></details>
          </div>
          <div id="m1-error" class="errbox hide"></div>
        </div>
      </div>

      <!-- MODULE 2: FOLLOWER HISTORY -->
      <div class="mod fol" id="mod2">
        <div class="mod-h">
          <div class="em">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          </div>
          <div class="tx"><b>Follower history</b><small>Daily follower count · last 365 days</small></div>
          <span class="pill todo" id="pill2">Step 2</span>
        </div>
        <div class="mod-b">
          <div id="m2-idle">
            <div class="guide">
              <span class="ic"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></span>
              <p>Open <b>Analytics → Followers</b>. We'll set the 365-day filter for you.</p>
            </div>
            <button class="btn open" id="m2-open">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
              Open my Followers analytics
            </button>
            <div class="hintline">…or switch to that tab yourself, then reopen this popup</div>
          </div>
          <div id="m2-ready" class="hide">
            <div class="onpage"><span class="ic"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg></span> You're on the Followers page — ready to extract.</div>
            <button class="btn extract" id="m2-extract">Extract follower history</button>
          </div>
          <div id="m2-run" class="hide">
            <div class="pbar"><i id="bar2"></i></div>
            <div class="pmeta" id="meta2">Fetching…</div>
          </div>
          <div id="m2-done" class="hide">
            <div class="filerow">
              <span class="l"><span class="em2"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></span><span><b id="m2-file">follower_history.csv</b><small id="m2-summary"></small></span></span>
              <button class="dl" id="m2-save"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Save</button>
            </div>
          </div>
          <div id="m2-error" class="errbox hide"></div>
        </div>
      </div>

      <div class="finish" id="finish">
        <div class="ic"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div>
        <b>Both files are ready</b>
        <p>Saved on your computer — nothing was uploaded. Send <b>both</b> to your researcher whenever you choose.</p>
      </div>
    </div>
  </div>

  <!-- ============ HELP PANEL ============ -->
  <div class="panel" id="p-help">
    <div class="body help">

      <details class="acc" open>
        <summary>
          <span class="si"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></span>
          <span class="st">About this extension</span>
          <span class="chev"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></span>
        </summary>
        <div class="ac-body">
          <p>This tool helps Filipino creators in our thesis study keep a copy of <b>their own</b> TikTok Studio analytics as CSV files. It reads the same numbers you can already see in TikTok Studio and saves them locally — nothing more.</p>
          <p>It produces two files: <b>video performance</b> and <b>follower history</b>.</p>
        </div>
      </details>

      <details class="acc">
        <summary>
          <span class="si"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg></span>
          <span class="st">Your data &amp; privacy</span>
          <span class="chev"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></span>
        </summary>
        <div class="ac-body">
          <p><b>Runs entirely on your computer.</b> The files are saved to your Downloads folder. The extension never sends your data to us or anyone else.</p>
          <p><b>You choose what to share.</b> Only the files you decide to send your researcher ever leave your device.</p>
        </div>
      </details>

      <details class="acc">
        <summary>
          <span class="si"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="15" x2="15" y2="15"/></svg></span>
          <span class="st">Consent &amp; withdrawal</span>
          <span class="chev"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></span>
        </summary>
        <div class="ac-body">
          <p>You agreed to take part in this study by signing the informed consent form. Using this tool is voluntary.</p>
          <p>You may <b>withdraw at any time</b> and ask that your shared files be deleted — no reason needed.</p>
          <a class="lnk" href="#">View consent form [TBD: consent form path/URL]</a>
        </div>
      </details>

      <details class="acc">
        <summary>
          <span class="si"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></span>
          <span class="st">Ethics &amp; approval</span>
          <span class="chev"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></span>
        </summary>
        <div class="ac-body">
          <p>This study was reviewed and approved by <b>[TBD: REC name]</b>. Your data is used only for academic research and is <b>anonymized</b> before analysis.</p>
          <p class="kv">Ethics reference: <b>[TBD: REC ref]</b></p>
          <p class="kv">Ethics contact: <b>[TBD: ethics email]</b></p>
        </div>
      </details>

      <details class="acc">
        <summary>
          <span class="si"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M22 6l-10 7L2 6"/></svg></span>
          <span class="st">Questions or help</span>
          <span class="chev"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></span>
        </summary>
        <div class="ac-body">
          <p>Reach the research team anytime:</p>
          <p class="kv">Researcher: <b>[TBD: researcher name]</b></p>
          <a class="lnk cyan" href="#">[TBD: researcher email]</a>
        </div>
      </details>

      <div class="help-foot"><span id="ver-foot">v0.2.0</span> · creator-side backup tool · for thesis research use</div>
    </div>
  </div>

  <!-- ============ DEBUG PANEL (hidden) ============ -->
  <div class="panel" id="p-debug">
    <div class="body">
      <div class="debug-meta">
        <span>video_list: <strong id="dbg-vl">0</strong></span>
        <span>insight: <strong id="dbg-ins">0</strong></span>
        <span>profile: <strong id="dbg-prof">0</strong></span>
      </div>
      <div class="debug-actions">
        <input id="dbg-filter" type="text" placeholder="filter URLs (e.g. follower, insight)" />
        <button id="dbg-copy" class="ghost">Copy all</button>
        <button id="dbg-clear" class="ghost">Reset state</button>
      </div>
      <ul id="dbg-urls"></ul>
      <details id="dbg-sample-wrap" class="hide">
        <summary>Last unparsed video-list payload shape</summary>
        <pre id="dbg-sample"></pre>
      </details>
    </div>
  </div>

</div>

<script src="popup.js"></script>
</body>
</html>
```

- [ ] **Step 2: Open the popup standalone in a browser**

Open `tiktok-analytics-exporter/popup.html` directly in Chrome (drag the file in, or `file://`). It will render unstyled (no `popup.css` yet) and inert (no `popup.js` yet). Confirm:
- All five Help accordion sections are present.
- Both Export-tab module cards render their idle state.
- No HTML errors in DevTools console.

- [ ] **Step 3: Commit**

```bash
git add tiktok-analytics-exporter/popup.html
git commit -m "feat: rewrite popup.html to mockup B v2 (Export + Help + hidden Debug)"
```

---

### Task 11: Rewrite `popup.css`

**Files:**
- Modify (full rewrite): `tiktok-analytics-exporter/popup.css`

- [ ] **Step 1: Replace the file**

Replace `tiktok-analytics-exporter/popup.css` with the styles below — identical to the CSS in `mockups/option-b-modular-v2.html`, plus a few production additions (`.errbox`, `.ghost` button variant, `.debug-meta`, `.debug-actions` for the hidden Debug panel, body `width:372px` for the popup chrome):

```css
:root{
  --card:#ffffff; --ink:#121417; --muted:#7a8089; --line:#ececef; --line2:#f3f4f6;
  --red:#fe2c55; --cyan:#20d5ec; --dark:#16171b; --ok:#10a554; --ok-soft:#e8f8ef;
  --amber:#b8791b; --amber-soft:#fff6e6; --amber-line:#f1ddb0;
  --err:#b00020; --err-soft:#fdecef; --err-line:#f3a3a3;
}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:var(--ink);width:372px;background:var(--card)}
svg{display:block}
.popup{width:372px;background:var(--card);overflow:hidden}
.head{background:var(--dark);padding:16px 18px}
.brand{display:flex;align-items:center;gap:10px}
.logo{width:32px;height:32px;border-radius:9px;position:relative;background:#000;display:grid;place-items:center;color:#fff}
.logo:after{content:"";position:absolute;inset:0;border-radius:9px;box-shadow:1.5px 1.5px 0 var(--cyan),-1.5px -1.5px 0 var(--red);opacity:.9}
.brand h1{margin:0;font-size:15.5px;font-weight:700;color:#fff;letter-spacing:-.2px}
.brand p{margin:1px 0 0;font-size:11px;color:#9aa0aa}
.rights{margin:11px 0 0;font-size:10.5px;color:#aeb3bd;display:flex;gap:7px;align-items:center}
.rights .sh{color:var(--cyan);flex:none}
.tabs{display:flex;background:#fff;border-bottom:1px solid var(--line);padding:0 8px}
.tab{flex:1;border:none;background:transparent;font:inherit;font-size:12px;color:var(--muted);
  padding:11px 4px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;font-weight:600}
.tab.on{color:var(--ink);border-bottom-color:var(--red)}
.tab.hidden{display:none}
.panel{display:none}.panel.on{display:block}
.strip{display:flex;align-items:center;justify-content:space-between;padding:12px 18px 4px}
.strip .dots{display:flex;align-items:center;gap:7px}
.strip .d{display:flex;align-items:center;gap:6px}
.strip .pip{width:20px;height:20px;border-radius:50%;background:#eceef1;color:var(--muted);
  display:grid;place-items:center;font-size:10.5px;font-weight:700}
.strip .d.done .pip{background:var(--ok);color:#fff}
.strip .d .t{font-size:10.5px;color:var(--muted);font-weight:600}
.strip .d.done .t{color:var(--ink)}
.strip .seg{width:18px;height:2px;background:#eceef1;border-radius:2px}
.strip .count{font-size:10.5px;color:var(--muted)}
.strip .count b{color:var(--ink)}
.body{padding:8px 16px 16px}
.mod{border:1px solid var(--line);border-radius:12px;margin-bottom:12px;overflow:hidden}
.mod-h{display:flex;align-items:center;gap:10px;padding:12px 13px}
.mod-h .em{width:30px;height:30px;border-radius:8px;display:grid;place-items:center;flex:none}
.mod.video .em{background:#ffe9ee;color:var(--red)}
.mod.fol .em{background:#e6fbfe;color:#0a9cb8}
.mod-h .tx{flex:1}
.mod-h b{font-size:12.5px;font-weight:650;display:block}
.mod-h small{font-size:10.5px;color:var(--muted)}
.pill{font-size:9.5px;font-weight:700;padding:3px 8px;border-radius:999px;white-space:nowrap;display:flex;align-items:center;gap:3px}
.pill.todo{background:#f2f3f5;color:var(--muted)}
.pill.ready{background:#e6fbfe;color:#0a8aa3}
.pill.done{background:var(--ok-soft);color:var(--ok)}
.mod-b{padding:0 13px 13px;border-top:1px solid var(--line)}
.guide{display:flex;gap:8px;align-items:flex-start;background:var(--amber-soft);border:1px solid var(--amber-line);
  border-radius:9px;padding:9px 10px;margin:12px 0 10px}
.guide .ic{color:var(--amber);flex:none;margin-top:1px}
.guide p{margin:0;font-size:11px;color:#7c5410;line-height:1.5}
.guide code{background:#fff;border:1px solid var(--amber-line);border-radius:4px;padding:0 4px;font-size:10px}
.onpage{display:flex;gap:7px;align-items:center;background:var(--ok-soft);border:1px solid #c2ebd3;
  border-radius:9px;padding:8px 10px;margin:12px 0 10px;font-size:11px;color:#106b3a}
.onpage .ic{color:var(--ok);flex:none}
.btn{width:100%;border:none;font:inherit;font-weight:700;font-size:12.5px;padding:11px;border-radius:9px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:7px}
.btn.open{background:#111;color:#fff}
.btn.open:hover{background:#000}
.btn.extract{background:var(--red);color:#fff}
.btn.extract:hover{background:#e0254a}
.btn.ghost{background:#fff;border:1px solid var(--line);color:var(--muted);margin-top:8px}
.btn.ghost:hover{border-color:#bbb;color:var(--ink)}
.btn:disabled{opacity:.55;cursor:not-allowed}
.hintline{font-size:10px;color:var(--muted);text-align:center;margin-top:7px}
.range{margin:0 0 10px}
.rl{font-size:10.5px;color:var(--muted);margin-bottom:6px;font-weight:600}
.dates{display:flex;gap:7px;margin-bottom:7px}
.dates input{flex:1;font:inherit;font-size:11px;padding:7px;border:1px solid var(--line);border-radius:7px}
.presets{display:flex;gap:5px;flex-wrap:wrap}
.presets button{font:inherit;font-size:10px;padding:4px 9px;border:1px solid var(--line);background:#fff;border-radius:7px;color:var(--muted);cursor:pointer}
.presets button.on{background:#111;border-color:#111;color:#fff}
.pbar{height:6px;background:#f0f1f3;border-radius:3px;overflow:hidden;margin:10px 0 6px}
.pbar > i{display:block;height:100%;width:0;background:var(--red);transition:width .25s}
.pmeta{font-size:10.5px;color:var(--muted);text-align:center}
.filerow{display:flex;align-items:center;justify-content:space-between;border:1px solid #c2ebd3;
  background:var(--ok-soft);border-radius:9px;padding:10px 12px;margin-top:11px}
.filerow .l{display:flex;align-items:center;gap:9px}
.filerow .em2{color:var(--ok);flex:none}
.filerow b{font-size:11.5px}
.filerow small{display:block;font-size:10px;color:#3f8c5f}
.filerow .dl{font-size:11px;color:var(--ok);font-weight:700;cursor:pointer;background:none;border:none;display:flex;align-items:center;gap:4px}
.errbox{background:var(--err-soft);border:1px solid var(--err-line);color:var(--err);
  border-radius:9px;padding:9px 11px;margin-top:10px;font-size:11px;line-height:1.45}
details{margin-top:9px;font-size:11px}
summary{cursor:pointer;color:var(--muted)}
#m1-skipped{margin:6px 0 0;padding-left:18px;max-height:100px;overflow:auto}
.finish{display:none;background:var(--dark);border-radius:12px;padding:15px 14px;color:#fff;text-align:center}
.finish.show{display:block}
.finish .ic{color:var(--cyan);margin:0 auto 6px}
.finish b{font-size:13px;display:block;margin:0 0 3px}
.finish p{margin:0;font-size:11px;color:#b9bcc4;line-height:1.5}
.hide{display:none !important}

/* Help tab */
.help{padding:6px 0 2px}
.acc{border:1px solid var(--line);border-radius:11px;margin-bottom:9px;overflow:hidden}
.acc summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:10px;padding:12px 13px}
.acc summary::-webkit-details-marker{display:none}
.acc .si{width:28px;height:28px;border-radius:8px;background:var(--line2);display:grid;place-items:center;color:#4b515a;flex:none}
.acc .st{flex:1;font-size:12.5px;font-weight:650;color:var(--ink)}
.acc .chev{color:var(--muted);transition:transform .18s}
.acc[open] .chev{transform:rotate(90deg)}
.acc .ac-body{padding:0 13px 13px 51px}
.acc .ac-body p{margin:0 0 8px;font-size:11.5px;color:#4d535b;line-height:1.6}
.acc .ac-body p:last-child{margin-bottom:0}
.acc .ac-body b{color:var(--ink)}
.lnk{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:650;color:var(--red);
  text-decoration:none;margin-top:2px}
.lnk.cyan{color:#0a9cb8}
.kv{font-size:11px;color:#4d535b;margin:2px 0}
.kv b{color:var(--ink)}
.help-foot{text-align:center;font-size:10px;color:var(--muted);margin-top:6px;cursor:default;user-select:none}

/* Debug tab */
.debug-meta{display:flex;gap:10px;margin:0 0 8px;color:var(--muted);font-size:11px}
.debug-meta strong{color:var(--ink)}
.debug-actions{display:flex;gap:6px;margin-bottom:8px}
.debug-actions input{flex:1;font:inherit;font-size:11px;padding:5px 7px;border:1px solid var(--line);border-radius:6px}
.debug-actions .ghost{font-size:11px;padding:5px 9px;background:#fff;border:1px solid var(--line);color:var(--muted);
  border-radius:6px;cursor:pointer;flex:none}
#dbg-urls{list-style:none;margin:0;padding:0;max-height:200px;overflow-y:auto;border:1px solid var(--line);border-radius:7px;font-size:10px}
#dbg-urls li{padding:4px 7px;border-bottom:1px solid var(--line);word-break:break-all;font-family:ui-monospace,Menlo,monospace}
#dbg-urls li:last-child{border-bottom:none}
#dbg-urls li.match-list{background:var(--ok-soft)}
#dbg-urls .badge{margin-right:4px;padding:0 4px;background:var(--line2);border-radius:2px;color:var(--muted)}
#dbg-sample{max-height:220px;overflow:auto;font-size:10px;font-family:ui-monospace,Menlo,monospace;background:var(--line2);padding:6px;border-radius:6px;white-space:pre-wrap}
```

- [ ] **Step 2: Visual check**

Open `tiktok-analytics-exporter/popup.html` directly in Chrome. Confirm:
- Dark header renders.
- Both module cards appear with the right line icons.
- Help tab content has 5 accordion sections (need to click the Help tab manually — JS not wired yet, but the tab button is visible).
- The Debug tab button is invisible (correct: it has class `hidden`).

- [ ] **Step 3: Commit**

```bash
git add tiktok-analytics-exporter/popup.css
git commit -m "feat: rewrite popup.css for mockup B v2 styling"
```

---

### Task 12: Rewrite `popup.js` — tab switching + state polling skeleton

**Goal:** Get tabs switching and the polled state plumbed in. Module-state rendering and downloads come in the next task.

**Files:**
- Modify (full rewrite): `tiktok-analytics-exporter/popup.js`

- [ ] **Step 1: Replace the file with the skeleton**

Replace `tiktok-analytics-exporter/popup.js`:

```js
const VIDEO_CSV_COLUMNS = [
  'video_id','post_date','post_time','caption','duration_ms','views','likes','comments','shares',
  'ECR','avg_watch_time_s','NAWP','watched_full_pct',
  'traffic_foryou_pct','traffic_follow_pct','traffic_profile_pct','traffic_search_pct',
  'new_followers','creator_uid','creator_handle','follower_count','account_created_date','data_quality'
];
const FOLLOWER_CSV_COLUMNS = ['date','follower_count','daily_net','creator_handle','creator_uid','data_quality'];

const STUDIO_CONTENT_URL   = 'https://www.tiktok.com/tiktokstudio/content';
const STUDIO_FOLLOWERS_URL = 'https://www.tiktok.com/tiktokstudio/analytics/followers?dateRange=%7B%22type%22%3A%22fixed%22%2C%22pastDay%22%3A365%7D';

let activeTabId = null;
let pollHandle = null;
let footerClickCount = 0;
let footerClickTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  init().catch((err) => console.error('[popup] init failed', err));
});

async function init() {
  const tab = await getActiveTab();
  activeTabId = tab?.id ?? null;
  wireTabs();
  wireDebugReveal();
  setDefaultDates();
  await refresh();
  pollHandle = setInterval(refresh, 750);
  window.addEventListener('unload', () => clearInterval(pollHandle));
}

function wireTabs() {
  for (const btn of document.querySelectorAll('.tab')) {
    btn.addEventListener('click', () => {
      const name = btn.dataset.tab;
      for (const t of document.querySelectorAll('.tab')) t.classList.toggle('on', t.dataset.tab === name);
      for (const p of document.querySelectorAll('.panel')) p.classList.toggle('on', p.id === `p-${name}`);
    });
  }
}

function wireDebugReveal() {
  const foot = document.getElementById('ver-foot');
  if (!foot) return;
  foot.addEventListener('click', () => {
    footerClickCount += 1;
    clearTimeout(footerClickTimer);
    footerClickTimer = setTimeout(() => (footerClickCount = 0), 600);
    if (footerClickCount >= 3) {
      footerClickCount = 0;
      document.getElementById('tab-debug').classList.remove('hidden');
    }
  });
}

function setDefaultDates() {
  const today = new Date();
  const ninety = new Date();
  ninety.setDate(today.getDate() - 90);
  const startEl = document.getElementById('m1-start');
  const endEl   = document.getElementById('m1-end');
  if (startEl) startEl.value = isoDate(ninety);
  if (endEl)   endEl.value   = isoDate(today);
}

function isoDate(d) { return d.toISOString().slice(0,10); }

async function refresh() {
  const res = await sendBg({ type: 'get-state' }).catch(() => null);
  const state = res?.state;
  if (!state) return;
  renderDebugCounters(state);
}

function renderDebugCounters(state) {
  const c = state.interceptCounts || {};
  setText('dbg-vl',   c.videoList ?? 0);
  setText('dbg-ins',  c.insight   ?? 0);
  setText('dbg-prof', c.profile   ?? 0);
}

function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = String(v); }

async function getActiveTab() {
  return new Promise((resolve) => chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => resolve(tabs[0])));
}
function sendBg(msg) {
  return new Promise((resolve) => chrome.runtime.sendMessage(msg, (response) => {
    void chrome.runtime.lastError;
    resolve(response);
  }));
}
```

- [ ] **Step 2: Reload extension and smoke-test**

`chrome://extensions` → reload. Open the popup. Confirm:
- Clicking the **Help** tab shows the 5 accordion sections.
- Clicking back to **Export** shows the two module cards.
- Triple-clicking the `v0.2.0` footer reveals the **Debug** tab; clicking it shows the debug panel.
- Service-worker DevTools: no errors.

- [ ] **Step 3: Commit**

```bash
git add tiktok-analytics-exporter/popup.js
git commit -m "feat: popup.js skeleton — tab switching + debug reveal + state poll"
```

---

### Task 13: Wire Step 1 (video) — page detect → extract → save

**Files:**
- Modify: `tiktok-analytics-exporter/popup.js`

- [ ] **Step 1: Update `init()` to call `wireModules`, then append Step 1 functions**

In `tiktok-analytics-exporter/popup.js`, edit the existing `init()` function to add a `wireModules();` call right after `wireDebugReveal();`:

```js
async function init() {
  const tab = await getActiveTab();
  activeTabId = tab?.id ?? null;
  wireTabs();
  wireDebugReveal();
  wireModules();                 // <-- ADD THIS LINE
  setDefaultDates();
  await refresh();
  pollHandle = setInterval(refresh, 750);
  window.addEventListener('unload', () => clearInterval(pollHandle));
}
```

Then append the following to the bottom of `popup.js`:

```js
let m1OnPage = false;
let m2OnPage = false;

function wireModules() {
  document.getElementById('m1-open').addEventListener('click', () => openTab(STUDIO_CONTENT_URL));
  document.getElementById('m1-extract').addEventListener('click', startVideoExtract);
  document.getElementById('m1-save').addEventListener('click', saveVideoCSV);
  document.getElementById('m1-cancel').addEventListener('click', () => sendBg({ type: 'cancel-video-export' }));
  for (const btn of document.querySelectorAll('#m1-ready .presets button')) {
    btn.addEventListener('click', () => applyPreset(btn));
  }
  document.getElementById('m2-open').addEventListener('click', () => openTab(STUDIO_FOLLOWERS_URL));
  document.getElementById('m2-extract').addEventListener('click', startFollowerExtract);
  document.getElementById('m2-save').addEventListener('click', saveFollowerCSV);
}

async function openTab(url) {
  if (activeTabId) {
    try { await chrome.tabs.update(activeTabId, { url, active: true }); return; }
    catch (_e) { /* fall through to create */ }
  }
  await chrome.tabs.create({ url });
}

function applyPreset(btn) {
  for (const b of document.querySelectorAll('#m1-ready .presets button')) b.classList.remove('on');
  btn.classList.add('on');
  const days = btn.dataset.days;
  const today = new Date();
  document.getElementById('m1-end').value = isoDate(today);
  if (!days) { document.getElementById('m1-start').value = ''; return; }
  const start = new Date();
  start.setDate(today.getDate() - Number(days));
  document.getElementById('m1-start').value = isoDate(start);
}

async function startVideoExtract() {
  hideErr('m1-error');
  if (!activeTabId) { showErr('m1-error', 'Open TikTok Studio first.'); return; }
  const dateRange = {
    start: document.getElementById('m1-start').value || null,
    end:   document.getElementById('m1-end').value   || null
  };
  await sendBg({ type: 'reset-video-step' });
  const res = await sendBg({ type: 'start-video-export', dateRange, tabId: activeTabId });
  if (!res?.ok) showErr('m1-error', res?.error || 'Failed to start');
  await refresh();
}

async function saveVideoCSV() {
  const res = await sendBg({ type: 'get-state' });
  const rows = res?.state?.videoStep?.rows || [];
  if (!rows.length) return;
  const handle = res.state.profile?.creator_handle || 'unknown';
  const today = isoDate(new Date());
  const filename = `tiktok_videos_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, VIDEO_CSV_COLUMNS));
}

async function checkPageStates() {
  if (!activeTabId) { m1OnPage = false; m2OnPage = false; return; }
  const [a, b] = await Promise.all([
    chrome.tabs.sendMessage(activeTabId, { type: 'is-studio-page' }).catch(() => null),
    chrome.tabs.sendMessage(activeTabId, { type: 'is-followers-page' }).catch(() => null)
  ]);
  m1OnPage = !!a?.isStudio;
  m2OnPage = !!b?.isFollowers;
}
```

Then replace the existing `refresh()` function with the version below (drives module rendering based on state):

```js
async function refresh() {
  await checkPageStates();
  const res = await sendBg({ type: 'get-state' }).catch(() => null);
  const state = res?.state;
  if (!state) return;
  renderModule1(state.videoStep);
  renderModule2(state.followerStep);
  renderFooter(state);
  renderDebugCounters(state);
  renderDebugUrls(state);
}

function renderModule1(vs) {
  const phase = vs.phase;
  showOnly('mod1', phaseToPaneM1(phase, m1OnPage));
  setPill('pill1', phase, m1OnPage, 1);
  setDot('dot1', phase === 'done');
  if (phase === 'fetching-insights' || phase === 'collecting' || phase === 'fetching-profile') {
    const total = vs.progress?.total || 0;
    const cur = vs.progress?.current || 0;
    const pct = total > 0 ? Math.min(100, (cur/total)*100) : phaseFallbackPct(phase);
    setBar('bar1', pct);
    setText('meta1', vs.progress?.message || `${cur}/${total}`);
  }
  if (phase === 'done') {
    setText('m1-summary', `${vs.rows.length} videos${vs.skipped.length ? ` · ${vs.skipped.length} skipped` : ''}`);
    renderSkipped(vs.skipped || []);
  }
  if (phase === 'error') showErr('m1-error', vs.error || 'Unknown error');
  else hideErr('m1-error');
}

function phaseToPaneM1(phase, onPage) {
  if (phase === 'idle' || phase === 'cancelled') return onPage ? 'ready' : 'idle';
  if (phase === 'collecting' || phase === 'fetching-insights' || phase === 'fetching-profile') return 'run';
  if (phase === 'done')  return 'done';
  if (phase === 'error') return onPage ? 'ready' : 'idle';
  return 'idle';
}

function phaseFallbackPct(phase) {
  if (phase === 'collecting') return 10;
  if (phase === 'fetching-profile') return 95;
  return 50;
}

function showOnly(modId, paneName) {
  for (const p of ['idle','ready','run','done']) {
    const el = document.getElementById(`${modId === 'mod1' ? 'm1' : 'm2'}-${p}`);
    if (el) el.classList.toggle('hide', p !== paneName);
  }
}

function setPill(id, phase, onPage, n) {
  const el = document.getElementById(id);
  if (!el) return;
  if (phase === 'done') { el.className = 'pill done'; el.textContent = 'Done'; return; }
  if (phase === 'collecting' || phase === 'fetching-insights' || phase === 'fetching-profile' || phase === 'fetching') {
    el.className = 'pill ready'; el.textContent = 'Running'; return;
  }
  if (onPage) { el.className = 'pill ready'; el.textContent = 'On page'; return; }
  el.className = 'pill todo'; el.textContent = `Step ${n}`;
}

function setDot(id, done) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('done', done);
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${pct}%`;
}

function renderSkipped(list) {
  setText('m1-skipped-n', list.length);
  const ul = document.getElementById('m1-skipped');
  if (!ul) return;
  ul.innerHTML = '';
  for (const s of list) {
    const li = document.createElement('li');
    li.textContent = `${s.aweme_id} — ${s.reason}`;
    ul.appendChild(li);
  }
}

function renderFooter(state) {
  const done1 = state.videoStep.phase === 'done' ? 1 : 0;
  const done2 = state.followerStep.phase === 'done' ? 1 : 0;
  setText('cnt', done1 + done2);
  document.getElementById('finish').classList.toggle('show', done1 + done2 === 2);
}

function buildCSV(rows, columns) {
  const lines = [columns.join(',')];
  for (const row of rows) lines.push(columns.map((c) => escapeCSV(row[c])).join(','));
  return lines.join('\n');
}
function escapeCSV(v) {
  if (v == null) return '';
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}
function sanitize(s) { return String(s).replace(/[^a-z0-9_-]/gi, '_').slice(0, 64); }
async function downloadCSV(filename, csv) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  await chrome.downloads.download({ url, filename, saveAs: true });
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
function showErr(id, msg) { const el = document.getElementById(id); if (el) { el.textContent = msg; el.classList.remove('hide'); } }
function hideErr(id) { const el = document.getElementById(id); if (el) { el.textContent = ''; el.classList.add('hide'); } }

// Placeholder for Step 2 — implemented in Task 14.
function renderModule2() {}
function startFollowerExtract() {}
function saveFollowerCSV() {}
function renderDebugUrls() {}
```

- [ ] **Step 2: Smoke-test Step 1**

Reload extension. In Chrome, log into TikTok and navigate to `https://www.tiktok.com/tiktokstudio/content`. Open the popup.
- Step 1 should switch from idle to **ready** ("You're on the Content page").
- Click **Extract video stats** → progress bar fills → reaches **done** → click **Save** → CSV downloads as `tiktok_videos_*.csv` with the right columns.

If at any point the popup gets stuck or the progress bar doesn't move, check the service-worker console for errors and fix before committing.

- [ ] **Step 3: Commit**

```bash
git add tiktok-analytics-exporter/popup.js
git commit -m "feat: popup.js Step 1 wiring (video extract + save)"
```

---

### Task 14: Wire Step 2 (follower) — page detect → extract → save

**Files:**
- Modify: `tiktok-analytics-exporter/popup.js`

- [ ] **Step 1: Replace the four placeholder functions**

At the bottom of `tiktok-analytics-exporter/popup.js`, **delete** the four placeholders:

```js
function renderModule2() {}
function startFollowerExtract() {}
function saveFollowerCSV() {}
function renderDebugUrls() {}
```

…and replace them with the real implementations:

```js
function renderModule2(fs) {
  const phase = fs.phase;
  showOnly('mod2', phaseToPaneM2(phase, m2OnPage));
  setPill('pill2', phase, m2OnPage, 2);
  setDot('dot2', phase === 'done');
  if (phase === 'fetching') {
    setBar('bar2', 50);
    setText('meta2', fs.progress?.message || 'Fetching…');
  }
  if (phase === 'done') {
    const allBlank = fs.rows.every((r) => r.data_quality === 'no_data');
    setText('m2-summary', allBlank
      ? `${fs.rows.length} days · account may be too new`
      : `${fs.rows.length} days`);
  }
  if (phase === 'error') showErr('m2-error', fs.error || 'Unknown error');
  else hideErr('m2-error');
}

function phaseToPaneM2(phase, onPage) {
  if (phase === 'idle' || phase === 'cancelled') return onPage ? 'ready' : 'idle';
  if (phase === 'fetching') return 'run';
  if (phase === 'done') return 'done';
  if (phase === 'error') return onPage ? 'ready' : 'idle';
  return 'idle';
}

async function startFollowerExtract() {
  hideErr('m2-error');
  if (!activeTabId) { showErr('m2-error', 'Open TikTok Studio first.'); return; }
  await sendBg({ type: 'reset-follower-step' });
  const res = await sendBg({ type: 'start-follower-export', tabId: activeTabId });
  if (!res?.ok) showErr('m2-error', res?.error || 'Failed to start');
  await refresh();
}

async function saveFollowerCSV() {
  const res = await sendBg({ type: 'get-state' });
  const rows = res?.state?.followerStep?.rows || [];
  if (!rows.length) return;
  const handle = res.state.profile?.creator_handle || 'unknown';
  const today = isoDate(new Date());
  const filename = `tiktok_followers_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, FOLLOWER_CSV_COLUMNS));
}

function renderDebugUrls(state) {
  const filterEl = document.getElementById('dbg-filter');
  const filter = (filterEl?.value || '').trim().toLowerCase();
  const urls = (state.recentURLs || []).slice().reverse();
  const ul = document.getElementById('dbg-urls');
  if (!ul) return;
  ul.innerHTML = '';
  for (const entry of urls) {
    if (filter && !entry.url.toLowerCase().includes(filter)) continue;
    const li = document.createElement('li');
    const isList = /item_list|aweme\/post|post\/list|follower/i.test(entry.url);
    if (isList) li.classList.add('match-list');
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = `${entry.method || 'GET'} ×${entry.count || 1}`;
    li.appendChild(badge);
    li.appendChild(document.createTextNode(entry.url));
    ul.appendChild(li);
  }
  const sample = state.lastVideoListSample;
  const wrap = document.getElementById('dbg-sample-wrap');
  const pre = document.getElementById('dbg-sample');
  if (sample) {
    wrap.classList.remove('hide');
    pre.textContent = JSON.stringify(sample, null, 2);
  } else {
    wrap.classList.add('hide');
    pre.textContent = '';
  }
}
```

Also extend the existing `wireModules()` function (added in Task 13) by inserting these lines at the end of its body — they wire up the Debug-tab actions now that Debug is reachable:

```js
  const dbgFilter = document.getElementById('dbg-filter');
  if (dbgFilter) dbgFilter.addEventListener('input', refresh);
  document.getElementById('dbg-copy').addEventListener('click', copyDebugURLs);
  document.getElementById('dbg-clear').addEventListener('click', async () => {
    await sendBg({ type: 'reset-state' });
    await refresh();
  });
```

And append the copy helper:

```js
function copyDebugURLs() {
  const urls = Array.from(document.querySelectorAll('#dbg-urls li'))
    .map((li) => li.textContent.trim()).join('\n');
  navigator.clipboard.writeText(urls).catch(() => {});
}
```

- [ ] **Step 2: Smoke-test Step 2**

Reload extension. Navigate to `https://www.tiktok.com/tiktokstudio/analytics/followers?dateRange=%7B%22type%22%3A%22fixed%22%2C%22pastDay%22%3A365%7D`. Open the popup.
- Step 2 should switch to **ready** ("You're on the Followers page").
- Click **Extract follower history** → progress shows → reaches **done** → click **Save** → CSV downloads as `tiktok_followers_*.csv` with 365 rows.
- Confirm the date column ends at *yesterday* and rows from before account creation have `data_quality=no_data` and blank counts.

- [ ] **Step 3: Smoke-test the "both done" state**

After both Step 1 and Step 2 have completed in the same session, confirm the **"Both files are ready"** dark banner appears at the bottom of the Export panel and the strip counter shows `2 of 2 ready`.

- [ ] **Step 4: Commit**

```bash
git add tiktok-analytics-exporter/popup.js
git commit -m "feat: popup.js Step 2 wiring (follower extract + save) + Debug actions"
```

---

## Phase 5 — Ship

### Task 15: Bump manifest version and add SMOKE.md

**Files:**
- Modify: `tiktok-analytics-exporter/manifest.json`
- Create: `tiktok-analytics-exporter/SMOKE.md`

- [ ] **Step 1: Bump version**

In `tiktok-analytics-exporter/manifest.json`, change `"version": "0.1.0"` → `"version": "0.2.0"`.

- [ ] **Step 2: Create SMOKE.md**

Create `tiktok-analytics-exporter/SMOKE.md`:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add tiktok-analytics-exporter/manifest.json tiktok-analytics-exporter/SMOKE.md
git commit -m "ship: bump to v0.2.0 + smoke checklist"
```

---

### Task 16: Run the full smoke checklist

- [ ] **Step 1: Walk through every box in `tiktok-analytics-exporter/SMOKE.md`** against the loaded unpacked extension.

- [ ] **Step 2: If any box fails**, file the failure as a follow-up commit (don't skip — if it fails it's a real bug to fix before the participant build).

- [ ] **Step 3: Final commit (only if any fixes landed during smoke)**

```bash
git add -p
git commit -m "fix: <specific smoke failure>"
```

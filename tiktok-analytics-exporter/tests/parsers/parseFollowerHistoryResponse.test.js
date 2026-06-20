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

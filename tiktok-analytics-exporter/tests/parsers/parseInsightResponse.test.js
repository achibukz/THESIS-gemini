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

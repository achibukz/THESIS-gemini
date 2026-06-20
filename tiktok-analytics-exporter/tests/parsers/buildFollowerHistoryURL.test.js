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

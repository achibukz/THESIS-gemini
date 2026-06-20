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

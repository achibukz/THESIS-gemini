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

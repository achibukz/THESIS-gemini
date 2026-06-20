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

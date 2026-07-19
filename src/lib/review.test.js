import test from 'node:test';
import assert from 'node:assert/strict';
import { parseGeminiJson } from './review.js';

test('parses JSON from a fenced response', () => {
  const result = parseGeminiJson('```json\n{"title":"Example"}\n```');
  assert.equal(result.title, 'Example');
});

test('returns an empty object when the response is not valid JSON', () => {
  const result = parseGeminiJson('No JSON here');
  assert.deepEqual(result, {});
});

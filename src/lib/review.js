export function sanitizeText(text, maxChars = 12000) {
  if (!text) return '';
  return text.replace(/\s+/g, ' ').trim().slice(0, maxChars);
}

export function parseGeminiJson(content) {
  if (!content) return {};

  const trimmed = content.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = fenced ? fenced[1] : trimmed;

  try {
    return JSON.parse(candidate);
  } catch {
    const fallback = candidate.match(/\{[\s\S]*\}/);
    if (!fallback) return {};
    try {
      return JSON.parse(fallback[0]);
    } catch {
      return {};
    }
  }
}

export function buildPrompt({ criteria, schema, text }) {
  const fieldList = schema
    .split(',')
    .map((field) => field.trim())
    .filter(Boolean)
    .join(', ');

  return `You are assisting with a systematic review screening workflow.

Return valid JSON only. Use this structure:
{
  "title": "...",
  "abstract": "...",
  "titleAbstractScreening": {
    "decision": "include|exclude|unclear",
    "reason": "..."
  },
  "fullTextScreening": {
    "decision": "include|exclude|unclear",
    "reason": "..."
  },
  "extractedFields": {
    "field1": "value",
    "field2": "value"
  }
}

Inclusion criteria:
${criteria || 'Use the review question and general methodological relevance.'}

Custom extraction fields:
${fieldList || 'none'}

Paper text:
${text}`;
}

import { NextResponse } from 'next/server';
import pdfParse from 'pdf-parse';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { buildPrompt, parseGeminiJson, sanitizeText } from '../../../src/lib/review.js';

export const runtime = 'nodejs';

export async function POST(request) {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files');
    const criteria = formData.get('criteria')?.toString() || '';
    const schema = formData.get('schema')?.toString() || '';
    const apiKey = formData.get('apiKey')?.toString() || process.env.GEMINI_API_KEY || '';

    if (!apiKey) {
      return NextResponse.json({ error: 'Missing Gemini API key. Provide it in the form or set GEMINI_API_KEY.' }, { status: 400 });
    }

    if (!files.length) {
      return NextResponse.json({ error: 'No PDF files uploaded.' }, { status: 400 });
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

    const results = [];

    for (const file of files) {
      if (typeof file === 'string' || file.name?.toLowerCase().endsWith('.pdf') === false) {
        continue;
      }

      const buffer = Buffer.from(await file.arrayBuffer());
      const pdfData = await pdfParse(buffer);
      const text = sanitizeText(pdfData.text || '');
      const prompt = buildPrompt({ criteria, schema, text });

      const response = await model.generateContent(prompt);
      const rawText = response.response.text();
      const parsed = parseGeminiJson(rawText);

      results.push({
        title: parsed.title || '',
        abstract: parsed.abstract || '',
        titleAbstractScreening: parsed.titleAbstractScreening || {
          decision: 'unclear',
          reason: 'No structured output returned.'
        },
        fullTextScreening: parsed.fullTextScreening || {
          decision: 'unclear',
          reason: 'No structured output returned.'
        },
        extractedFields: parsed.extractedFields || {},
        sourceFile: file.name
      });
    }

    return NextResponse.json({ results });
  } catch (error) {
    return NextResponse.json({ error: error.message || 'Unexpected error' }, { status: 500 });
  }
}

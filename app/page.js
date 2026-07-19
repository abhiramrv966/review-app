'use client';

import { useEffect, useMemo, useState } from 'react';

const defaultCriteria = `Include studies that evaluate an intervention or exposure relevant to the review question and report outcomes in human participants. Exclude editorials, commentaries, protocols, and studies without primary data.`;
const defaultSchema = 'population, intervention, comparator, outcome, study_design';

export default function HomePage() {
  const [apiKey, setApiKey] = useState('');
  const [criteria, setCriteria] = useState(defaultCriteria);
  const [schema, setSchema] = useState(defaultSchema);
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem('review-mvp-results');
      if (saved) {
        setResults(JSON.parse(saved));
      }
    } catch {
      // Ignore storage errors
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem('review-mvp-results', JSON.stringify(results));
  }, [results]);

  const summary = useMemo(() => {
    const includeCount = results.filter((item) => item.titleAbstractScreening?.decision === 'include').length;
    const excludeCount = results.filter((item) => item.titleAbstractScreening?.decision === 'exclude').length;
    return { includeCount, excludeCount, total: results.length };
  }, [results]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!files.length) {
      setMessage('Please upload at least one PDF.');
      return;
    }

    setLoading(true);
    setMessage('Processing PDFs with Gemini...');

    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append('files', file));
    formData.append('criteria', criteria);
    formData.append('schema', schema);
    if (apiKey) {
      formData.append('apiKey', apiKey);
    }

    try {
      const response = await fetch('/api/review', {
        method: 'POST',
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Review processing failed');
      }

      setResults((current) => [...payload.results, ...current]);
      setMessage(`Processed ${payload.results.length} paper${payload.results.length === 1 ? '' : 's'}.`);
    } catch (error) {
      setMessage(error.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'review-results.json';
    link.click();
    URL.revokeObjectURL(url);
  }

  function exportCsv() {
    const rows = results.map((item) => ({
      title: item.title || '',
      decision: item.titleAbstractScreening?.decision || '',
      fullTextDecision: item.fullTextScreening?.decision || '',
      reason: item.titleAbstractScreening?.reason || item.fullTextScreening?.reason || '',
      extractedFields: JSON.stringify(item.extractedFields || {})
    }));

    const header = ['title', 'decision', 'fullTextDecision', 'reason', 'extractedFields'];
    const csv = [header.join(','), ...rows.map((row) => header.map((key) => `"${String(row[key]).replaceAll('"', '""')}"`).join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'review-results.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main>
      <section className="panel" style={{ marginBottom: 20 }}>
        <h1>Systematic Review Screening MVP</h1>
        <p className="small" style={{ marginTop: -8 }}>
          Upload PDFs, let Gemini summarize each study, score title/abstract and full-text screening, and extract custom fields in one pass.
        </p>
      </section>

      <section className="panel" style={{ marginBottom: 20 }}>
        <form onSubmit={handleSubmit} className="grid">
          <div className="form-grid">
            <label>
              Gemini API key
              <input value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Optional if set as GEMINI_API_KEY" />
            </label>
            <label>
              Inclusion criteria
              <textarea value={criteria} onChange={(event) => setCriteria(event.target.value)} />
            </label>
            <label>
              Custom extraction fields
              <textarea value={schema} onChange={(event) => setSchema(event.target.value)} placeholder="population, intervention, outcome" />
            </label>
          </div>

          <label>
            Upload PDFs
            <input type="file" accept=".pdf" multiple onChange={(event) => setFiles(Array.from(event.target.files || []))} />
          </label>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <button type="submit" disabled={loading}>{loading ? 'Processing…' : 'Run screening'}</button>
            <button type="button" className="secondary" onClick={exportJson} disabled={!results.length}>Export JSON</button>
            <button type="button" className="secondary" onClick={exportCsv} disabled={!results.length}>Export CSV</button>
          </div>

          {message ? <p className="small">{message}</p> : null}
        </form>
      </section>

      <section className="panel" style={{ marginBottom: 20 }}>
        <h2>Summary</h2>
        <p className="small">{summary.total} processed • {summary.includeCount} include • {summary.excludeCount} exclude</p>
      </section>

      <section className="grid">
        {results.map((item, index) => (
          <article key={`${item.title || 'paper'}-${index}`} className="result-card">
            <div className={`badge ${item.titleAbstractScreening?.decision || 'unclear'}`}>
              {item.titleAbstractScreening?.decision || 'unclear'}
            </div>
            <h3>{item.title || `Paper ${index + 1}`}</h3>
            <p className="small">{item.abstract || 'Abstract unavailable'}</p>
            <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
              <div>
                <strong>Title/Abstract screening:</strong> {item.titleAbstractScreening?.decision || 'unclear'}
                <br />
                <span className="small">{item.titleAbstractScreening?.reason || 'No reason provided.'}</span>
              </div>
              <div>
                <strong>Full-text screening:</strong> {item.fullTextScreening?.decision || 'unclear'}
                <br />
                <span className="small">{item.fullTextScreening?.reason || 'No reason provided.'}</span>
              </div>
              <div>
                <strong>Extracted fields:</strong>
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{JSON.stringify(item.extractedFields || {}, null, 2)}</pre>
              </div>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}

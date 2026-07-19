# Systematic Review Screening Tool (Streamlit)

An AI-assisted screening tool for a systematic review, built with Streamlit + Google Gemini.

## What it does
- **Title/Abstract screening** — ingest citation exports (`.ris`, `.nbib`, `.csv`) *or* PDFs
- **Full-text screening** — ingest PDFs; each exclusion gets one structured reason (for PRISMA)
- **Custom data extraction** — typed field schema → evidence table, with a verbatim source quote per value
- **Spot-check workflow** — the AI decides, but every excluded and low-confidence record is queued for your review (a wrong *exclude* silently drops a study, so those are checked by default)
- **PRISMA counts + audit log** — auto-tallied flow numbers and a downloadable log of every AI decision and human override for your methods section
- **Exports** — screening results as CSV, evidence table as XLSX, audit log as JSON

> Methodological note: fully automated "AI decides" screening is the fastest option but the hardest to defend to peer reviewers. Keep the audit log and review the flagged excludes — that lets you honestly report "AI screening with human verification of excluded records."

## Files
- `streamlit_app.py` — the app (UI + stage orchestration)
- `review_core.py` — pure logic: RIS/CSV parsing, prompt building, schema handling, Gemini client
- `test_review_core.py` — unit tests for the pure logic (no API key needed)
- `requirements.txt` — Python dependencies
- `.streamlit/secrets.toml.example` — template for your API key

> The original Next.js MVP (`app/`, `src/`, `package.json`) is left in the repo but is superseded by the Streamlit app.

## Run locally
1. Create a virtual environment and install deps:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```
2. Add your Gemini API key (from https://aistudio.google.com/apikey):
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key, **or**
   - just type the key into the Setup tab at runtime.
3. Launch:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deploy on Streamlit Community Cloud (free)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app**, pick the repo/branch, set the main file to `streamlit_app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy. Uploaded files and results live in the session only (no database) — download your CSV/XLSX/audit log before closing the tab.

## Test
```bash
python -m unittest test_review_core.py
```

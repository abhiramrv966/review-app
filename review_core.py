"""Core logic for the systematic-review screening tool.

Pure, side-effect-free helpers (prompt building, parsing, schema handling) live
here so they can be unit-tested without a Gemini API key or Streamlit runtime.
The thin Gemini client wrapper is at the bottom.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DECISIONS = ("include", "exclude", "unclear")

# PRISMA needs excluded-with-reason counts, so full-text exclusions are drawn
# from a fixed list the user can edit in the UI.
DEFAULT_EXCLUSION_REASONS = [
    "Wrong population",
    "Wrong intervention/exposure",
    "Wrong comparator",
    "Wrong outcome",
    "Wrong study design",
    "Not primary data (review/editorial/protocol)",
    "Full text unavailable",
    "Duplicate",
    "Other",
]

DEFAULT_CRITERIA = (
    "Include studies that evaluate an intervention or exposure relevant to the "
    "review question and report outcomes in human participants. Exclude "
    "editorials, commentaries, protocols, and studies without primary data."
)

# Inclusion and exclusion criteria are kept separate (per stage) to mirror the
# standard systematic-review workflow.
DEFAULT_INCLUSION = (
    "Studies evaluating an intervention or exposure relevant to the review "
    "question, reporting outcomes in human participants, based on primary data."
)
DEFAULT_EXCLUSION = (
    "Editorials, commentaries, letters, protocols, reviews without primary "
    "data, animal or in-vitro studies, wrong population, or wrong outcomes."
)


def format_criteria(inclusion: str, exclusion: str) -> str:
    """Render separate inclusion/exclusion criteria into a prompt block."""
    inc = (inclusion or "").strip() or "(none specified — use general methodological relevance)"
    exc = (exclusion or "").strip() or "(none specified)"
    return f"INCLUSION criteria:\n{inc}\n\nEXCLUSION criteria:\n{exc}"

DEFAULT_SCHEMA_TEXT = """\
population | text | The study population / participants
sample_size | number | Total number of participants analysed
intervention | text | Intervention or exposure studied
comparator | text | Comparison / control group
outcome | text | Primary outcome(s) reported
study_design | enum(RCT, cohort, case-control, cross-sectional, other) |
"""


# --------------------------------------------------------------------------- #
# Text + JSON helpers
# --------------------------------------------------------------------------- #

def sanitize_text(text: str | None, max_chars: int = 20000) -> str:
    """Collapse whitespace and cap length (used for citation-record text)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def parse_json(content: str | None) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response."""
    if not content:
        return {}
    trimmed = content.strip()

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, re.IGNORECASE)
    candidate = fenced.group(1) if fenced else trimmed

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    brace = re.search(r"\{[\s\S]*\}", candidate)
    if not brace:
        return {}
    try:
        parsed = json.loads(brace.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_json_array(content: str | None) -> list[dict[str, Any]]:
    """Best-effort extraction of a JSON array of objects from a model response."""
    if not content:
        return []
    trimmed = content.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, re.IGNORECASE)
    candidate = fenced.group(1) if fenced else trimmed

    for text in (candidate, (re.search(r"\[[\s\S]*\]", candidate) or [None])[0]):
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            # a single object, or {"results": [...]}
            for v in parsed.values():
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
            return [parsed]
    return []


def normalize_decision(value: Any) -> str:
    """Coerce an arbitrary model value to one of DECISIONS."""
    if not isinstance(value, str):
        return "unclear"
    v = value.strip().lower()
    for d in DECISIONS:
        if d in v:
            return d
    return "unclear"


def normalize_confidence(value: Any) -> float:
    """Coerce a confidence value to a float in [0, 1]."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0:  # model may return a percentage
        f = f / 100.0
    return max(0.0, min(1.0, f))


# --------------------------------------------------------------------------- #
# Citation records
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    """A single citation to be screened at the title/abstract stage."""
    id: str
    title: str = ""
    abstract: str = ""
    authors: str = ""
    year: str = ""
    source: str = ""  # originating file name

    def as_text(self) -> str:
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.authors:
            parts.append(f"Authors: {self.authors}")
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        return sanitize_text("\n".join(parts))


# RIS tag -> field. RIS is a simple line-based "TAG  - value" format.
_RIS_TITLE_TAGS = ("TI", "T1", "TT")
_RIS_ABSTRACT_TAGS = ("AB", "N2")
_RIS_AUTHOR_TAGS = ("AU", "A1")
_RIS_YEAR_TAGS = ("PY", "Y1")


def parse_ris(content: str, source: str = "") -> list[Record]:
    """Parse a RIS / NBIB-style citation export into Record objects."""
    records: list[Record] = []
    cur: dict[str, Any] | None = None
    idx = 0

    def flush():
        nonlocal cur, idx
        if cur is None:
            return
        idx += 1
        year = str(cur.get("year", ""))[:4]
        records.append(
            Record(
                id=f"{source or 'ris'}-{idx}",
                title=sanitize_text(cur.get("title", ""), 1000),
                abstract=sanitize_text(cur.get("abstract", ""), 8000),
                authors="; ".join(cur.get("authors", [])),
                year=year,
                source=source,
            )
        )
        cur = None

    for raw in content.splitlines():
        line = raw.rstrip("\n")
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            # continuation of a wrapped multi-line value
            if cur is not None and line.strip() and cur.get("_last"):
                cur[cur["_last"]] = (cur.get(cur["_last"], "") + " " + line.strip()).strip()
            continue
        tag, value = m.group(1), m.group(2).strip()

        if tag == "TY":  # start of a new record
            flush()
            cur = {"authors": []}
        if cur is None:
            cur = {"authors": []}

        if tag == "ER":
            flush()
        elif tag in _RIS_TITLE_TAGS:
            cur["title"] = value
            cur["_last"] = "title"
        elif tag in _RIS_ABSTRACT_TAGS:
            # abstracts are often split across several AB/N2 lines -> accumulate
            cur["abstract"] = (cur.get("abstract", "") + " " + value).strip()
            cur["_last"] = "abstract"
        elif tag in _RIS_AUTHOR_TAGS:
            cur["authors"].append(value)
            cur["_last"] = None
        elif tag in _RIS_YEAR_TAGS:
            cur["year"] = value
            cur["_last"] = None
        else:
            cur["_last"] = None

    flush()
    return records


def detect_citation_columns(fieldnames) -> dict[str, str | None]:
    """Map a CSV's headers (case-insensitive) to our title/abstract/author/year fields.

    Works for PubMed, Scopus, Web of Science, Zotero, Rayyan and similar exports.
    """
    lower = {str(h).lower().strip(): h for h in (fieldnames or []) if h}

    def pick(*cands: str) -> str | None:
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    return {
        "title": pick("title", "article title", "document title", "primary title"),
        "abstract": pick("abstract", "abstract note", "abstract text"),
        "authors": pick("authors", "author", "author full names", "author names"),
        "year": pick("year", "publication year", "pub year", "py"),
    }


def parse_citation_csv(content: str, source: str = "") -> list[Record]:
    """Parse a CSV citation export (PubMed/Scopus/Zotero style headers)."""
    records: list[Record] = []
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return records

    cols = detect_citation_columns(reader.fieldnames)
    title_h, abstract_h = cols["title"], cols["abstract"]
    author_h, year_h = cols["authors"], cols["year"]

    for i, row in enumerate(reader, start=1):
        records.append(
            Record(
                id=f"{source or 'csv'}-{i}",
                title=sanitize_text(row.get(title_h, "") if title_h else "", 1000),
                abstract=sanitize_text(row.get(abstract_h, "") if abstract_h else "", 8000),
                authors=sanitize_text(row.get(author_h, "") if author_h else "", 1000),
                year=str(row.get(year_h, "") if year_h else "")[:4],
                source=source,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Extraction schema
# --------------------------------------------------------------------------- #

@dataclass
class SchemaField:
    name: str
    type: str = "text"          # text | number | enum | boolean
    hint: str = ""
    values: list[str] = field(default_factory=list)


def parse_schema(text: str) -> list[SchemaField]:
    """Parse a typed extraction schema.

    One field per line: ``name | type | hint``
    ``type`` may be ``enum(a, b, c)`` to constrain allowed values.
    A leading ``#`` or blank line is ignored.
    """
    fields: list[SchemaField] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        name = re.sub(r"\s+", "_", parts[0].strip()) if parts else ""
        if not name:
            continue
        raw_type = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "text"
        hint = parts[2].strip() if len(parts) > 2 else ""

        values: list[str] = []
        enum_m = re.match(r"enum\s*\((.*)\)", raw_type, re.IGNORECASE)
        if enum_m:
            ftype = "enum"
            # keep original case for the allowed values (e.g. "RCT")
            values = [v.strip() for v in enum_m.group(1).split(",") if v.strip()]
        else:
            ftype = raw_type.lower()
            if ftype not in ("text", "number", "boolean"):
                ftype = "text"

        fields.append(SchemaField(name=name, type=ftype, hint=hint, values=values))
    return fields


def schema_to_prompt_block(fields: list[SchemaField]) -> str:
    if not fields:
        return "none"
    lines = []
    for f in fields:
        desc = f.hint or ""
        if f.type == "enum" and f.values:
            desc = (desc + f" (one of: {', '.join(f.values)})").strip()
        elif f.type != "text":
            desc = (desc + f" ({f.type})").strip()
        lines.append(f'- "{f.name}": {desc or f.type}')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #

def build_ta_prompt(record_text: str, inclusion: str, exclusion: str) -> str:
    return f"""You are screening a citation for a systematic review at the TITLE/ABSTRACT stage.
Decide whether it should be INCLUDED for full-text review or EXCLUDED. When the
abstract is insufficient to decide, prefer "include" (do not exclude on missing
information at this stage).

Return ONLY valid JSON:
{{
  "decision": "include|exclude|unclear",
  "reason": "one concise sentence",
  "confidence": 0.0
}}
"confidence" is your certainty in the decision, from 0.0 to 1.0.

{format_criteria(inclusion, exclusion)}

Citation:
{record_text}"""


def build_ta_batch_prompt(items: list[dict[str, str]], inclusion: str, exclusion: str) -> str:
    """Screen several citations in one call (fewer API calls -> free-tier friendly)."""
    payload = json.dumps([{"id": it["id"], "text": it["text"]} for it in items], ensure_ascii=False)
    return f"""You are screening citations for a systematic review at the TITLE/ABSTRACT stage.
For EACH item below decide INCLUDE or EXCLUDE for full-text review. When an
abstract is insufficient to decide, prefer "include".

Return ONLY a JSON array with one object per input item, preserving the same "id":
[
  {{"id": "<id>", "decision": "include|exclude|unclear", "reason": "one concise sentence", "confidence": 0.0}}
]

{format_criteria(inclusion, exclusion)}

Items (JSON array of {{id, text}}):
{payload}"""


def build_fulltext_prompt(inclusion: str, exclusion: str, exclusion_reasons: list[str]) -> str:
    reasons = exclusion_reasons or DEFAULT_EXCLUSION_REASONS
    reason_list = "\n".join(f"- {r}" for r in reasons)
    return f"""You are screening the FULL TEXT of a study for a systematic review.
Apply the criteria in detail using the attached PDF. If excluding, choose exactly
one reason from the allowed list (PRISMA requires a single primary reason per
excluded study).

Return ONLY valid JSON:
{{
  "decision": "include|exclude|unclear",
  "exclusionReason": "<one of the allowed reasons, or empty if included>",
  "reason": "one concise sentence citing the relevant detail",
  "confidence": 0.0
}}

{format_criteria(inclusion, exclusion)}

Allowed exclusion reasons:
{reason_list}"""


def build_extraction_prompt(fields: list[SchemaField]) -> str:
    return f"""You are extracting structured data from the attached study PDF for a
systematic review evidence table. Extract each field below. For every field,
also give the exact quote (verbatim) from the paper that supports the value so
it can be verified. If a field is not reported, use "" for the value and note
"not reported".

Return ONLY valid JSON of this shape:
{{
  "fields": {{
    "<field_name>": {{ "value": "...", "source_quote": "..." }}
  }}
}}

Fields to extract:
{schema_to_prompt_block(fields)}"""


# --------------------------------------------------------------------------- #
# Result normalization
# --------------------------------------------------------------------------- #

def normalize_screen_result(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": normalize_decision(parsed.get("decision")),
        "reason": str(parsed.get("reason", "") or ""),
        "exclusionReason": str(parsed.get("exclusionReason", "") or ""),
        "confidence": normalize_confidence(parsed.get("confidence")),
    }


def normalize_extraction(parsed: dict[str, Any], fields: list[SchemaField]) -> dict[str, dict]:
    raw = parsed.get("fields", parsed) or {}
    out: dict[str, dict] = {}
    for f in fields:
        cell = raw.get(f.name, {})
        if isinstance(cell, dict):
            value = cell.get("value", "")
            quote = cell.get("source_quote", cell.get("quote", ""))
        else:
            value, quote = cell, ""
        out[f.name] = {"value": "" if value is None else str(value),
                       "source_quote": "" if quote is None else str(quote)}
    return out


def needs_review(result: dict[str, Any], confidence_threshold: float = 0.7,
                 review_excludes: bool = True) -> bool:
    """Spot-check policy: flag low-confidence items and (optionally) all excludes.

    Excludes are asymmetrically risky in a systematic review (a wrong exclude
    silently drops a study), so by default every exclude is queued for a human.
    """
    if result.get("confidence", 0.0) < confidence_threshold:
        return True
    if review_excludes and result.get("decision") == "exclude":
        return True
    return False


# --------------------------------------------------------------------------- #
# Gemini client wrapper
# --------------------------------------------------------------------------- #

class QuotaError(RuntimeError):
    """Raised when Gemini returns a 429 / quota-exceeded after retries."""


class AuthError(RuntimeError):
    """Raised when the API key is missing, malformed, or rejected (401/403)."""


def classify_api_error(exc: Exception) -> Exception:
    """Map a raw google-generativeai error to a friendly, typed exception."""
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "quota" in low or "resource" in low or "rate limit" in low:
        return QuotaError(
            "Gemini quota/rate limit hit (429). On the free tier this resets daily "
            "(~midnight Pacific). Lower the batch size, wait, or enable billing.")
    if any(t in msg for t in ("401", "403")) or "api key" in low or "permission" in low \
            or "unauthenticated" in low or "invalid" in low:
        return AuthError(
            "Gemini rejected the API key. Make sure it is a Google AI Studio key "
            "(it starts with 'AIza') from https://aistudio.google.com/apikey.")
    return exc


class GeminiClient:
    """Thin wrapper around google-generativeai with JSON-mode responses."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai  # imported lazily so tests don't need it
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(model_name)

    def _generate(self, parts: list[Any], max_retries: int = 3) -> str:
        import time
        delay = 5.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.model.generate_content(
                    parts,
                    generation_config={"response_mime_type": "application/json"},
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                friendly = classify_api_error(exc)
                last_exc = friendly
                # Only transient rate limits are worth retrying with backoff.
                if isinstance(friendly, QuotaError) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise friendly from exc
        raise last_exc  # pragma: no cover

    def screen_title_abstract(self, record: Record, inclusion: str, exclusion: str) -> dict[str, Any]:
        prompt = build_ta_prompt(record.as_text(), inclusion, exclusion)
        return normalize_screen_result(parse_json(self._generate([prompt])))

    def screen_title_abstract_batch(self, records: list[Record], inclusion: str,
                                    exclusion: str) -> dict[str, dict[str, Any]]:
        """Screen a batch of records in one API call; returns {record_id: result}."""
        items = [{"id": r.id, "text": r.as_text()} for r in records]
        prompt = build_ta_batch_prompt(items, inclusion, exclusion)
        arr = parse_json_array(self._generate([prompt]))
        by_id = {str(o.get("id")): o for o in arr}
        out: dict[str, dict[str, Any]] = {}
        for r in records:
            match = by_id.get(str(r.id))
            if match is None:
                out[r.id] = {"decision": "unclear", "reason": "No result returned for this item.",
                             "exclusionReason": "", "confidence": 0.0}
            else:
                out[r.id] = normalize_screen_result(match)
        return out

    def screen_full_text(self, pdf_bytes: bytes, inclusion: str, exclusion: str,
                         exclusion_reasons: list[str]) -> dict[str, Any]:
        prompt = build_fulltext_prompt(inclusion, exclusion, exclusion_reasons)
        parts = [{"mime_type": "application/pdf", "data": pdf_bytes}, prompt]
        return normalize_screen_result(parse_json(self._generate(parts)))

    def extract_data(self, pdf_bytes: bytes, fields: list[SchemaField]) -> dict[str, dict]:
        prompt = build_extraction_prompt(fields)
        parts = [{"mime_type": "application/pdf", "data": pdf_bytes}, prompt]
        return normalize_extraction(parse_json(self._generate(parts)), fields)

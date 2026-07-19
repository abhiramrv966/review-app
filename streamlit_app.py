"""Systematic Review Screening Tool — Streamlit app.

Stages:
  1. Setup            — API key, criteria, exclusion reasons, extraction schema
  2. T&A screening    — ingest RIS/CSV citations OR PDFs; AI decides, human spot-checks
  3. Full-text screen — ingest PDFs of records that passed T&A; structured exclusion reasons
  4. Data extraction  — ingest included PDFs; typed schema -> evidence table
  5. Export & PRISMA  — counts, audit log, downloads

Single-user, no database: everything lives in st.session_state and downloads.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import review_core as rc

st.set_page_config(page_title="SR Screening Tool", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def _init_state():
    ss = st.session_state
    ss.setdefault("api_key", "")
    ss.setdefault("model_name", "gemini-2.0-flash")
    ss.setdefault("criteria", rc.DEFAULT_CRITERIA)
    ss.setdefault("exclusion_reasons", list(rc.DEFAULT_EXCLUSION_REASONS))
    ss.setdefault("schema_text", rc.DEFAULT_SCHEMA_TEXT)
    ss.setdefault("confidence_threshold", 0.7)
    ss.setdefault("review_excludes", True)
    ss.setdefault("ta_results", [])
    ss.setdefault("ft_results", [])
    ss.setdefault("extractions", [])
    ss.setdefault("audit_log", [])


_init_state()

MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]


# --------------------------------------------------------------------------- #
# Helpers (defined before the tab blocks that use them)
# --------------------------------------------------------------------------- #

def log(event: str, **detail):
    st.session_state.audit_log.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **detail,
    })


def get_client() -> "rc.GeminiClient | None":
    key = st.session_state.api_key or st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        st.error("No Gemini API key. Add it in **Setup** or in Streamlit secrets as GEMINI_API_KEY.")
        return None
    try:
        return rc.GeminiClient(key, st.session_state.model_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not initialise Gemini client: {exc}")
        return None


def read_records(files) -> list[rc.Record]:
    records: list[rc.Record] = []
    for f in files:
        text = f.getvalue().decode("utf-8", errors="replace")
        name = f.name.lower()
        if name.endswith((".ris", ".nbib", ".txt")):
            records.extend(rc.parse_ris(text, source=f.name))
        elif name.endswith(".csv"):
            records.extend(rc.parse_citation_csv(text, source=f.name))
    return records


def badge(decision: str) -> str:
    return {"include": "🟢 include", "exclude": "🔴 exclude"}.get(decision, "🟡 unclear")


def extractions_to_df(extractions: list[dict], include_quotes: bool = False) -> pd.DataFrame:
    rows = []
    for ex in extractions:
        row = {"study": ex["study"]}
        for fname, cell in ex["fields"].items():
            row[fname] = cell.get("value", "")
            if include_quotes:
                row[f"{fname}__quote"] = cell.get("source_quote", "")
        rows.append(row)
    return pd.DataFrame(rows)


def render_screen_results(key: str):
    results = st.session_state.get(key, [])
    if not results:
        return
    inc = sum(r["decision"] == "include" for r in results)
    exc = sum(r["decision"] == "exclude" for r in results)
    unc = sum(r["decision"] == "unclear" for r in results)
    flagged = sum(r["needs_review"] and not r["confirmed"] for r in results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Include", inc)
    m2.metric("Exclude", exc)
    m3.metric("Unclear", unc)
    m4.metric("Needs review", flagged)

    only_flagged = st.checkbox("Show only records needing review", key=f"{key}_flag")
    for i, r in enumerate(results):
        if only_flagged and (not r["needs_review"] or r["confirmed"]):
            continue
        flag_txt = " ⚠️ **review**" if r["needs_review"] and not r["confirmed"] else ""
        header = f"{badge(r['decision'])} · conf {r['confidence']:.2f}{flag_txt} — {(r['title'] or r['id'])[:110]}"
        with st.expander(header):
            st.write(r["reason"] or "_no reason_")
            if r.get("exclusionReason"):
                st.write(f"**Exclusion reason:** {r['exclusionReason']}")
            if r.get("abstract"):
                st.caption(r["abstract"][:600])
            cols = st.columns(3)
            if cols[0].button("✅ Set include", key=f"{key}_inc_{i}"):
                r["decision"], r["confirmed"] = "include", True
                log("override", id=r["id"], to="include")
                st.rerun()
            if cols[1].button("❌ Set exclude", key=f"{key}_exc_{i}"):
                r["decision"], r["confirmed"] = "exclude", True
                log("override", id=r["id"], to="exclude")
                st.rerun()
            if cols[2].button("✔️ Confirm AI", key=f"{key}_ok_{i}"):
                r["confirmed"] = True
                log("confirm", id=r["id"], decision=r["decision"])
                st.rerun()


def screen_df(results) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": r["id"], "title": r["title"], "year": r.get("year", ""),
        "decision": r["decision"], "confidence": r["confidence"],
        "exclusion_reason": r.get("exclusionReason", ""),
        "reason": r["reason"], "needs_review": r["needs_review"], "confirmed": r["confirmed"],
    } for r in results])


# --------------------------------------------------------------------------- #
# Header + tabs
# --------------------------------------------------------------------------- #

st.title("🔎 Systematic Review Screening Tool")
st.caption(
    "AI-assisted title/abstract & full-text screening and custom data extraction. "
    "The AI decides; **excluded and low-confidence records are queued for your spot-check** "
    "(a wrong exclude silently drops a study, so those are reviewed by default)."
)

tab_setup, tab_ta, tab_ft, tab_extract, tab_export = st.tabs(
    ["⚙️ Setup", "1️⃣ Title/Abstract", "2️⃣ Full-text", "3️⃣ Data extraction", "📤 Export & PRISMA"]
)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

with tab_setup:
    st.subheader("Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.api_key = st.text_input(
            "Gemini API key", value=st.session_state.api_key, type="password",
            help="Or set GEMINI_API_KEY in Streamlit secrets and leave this blank.",
        )
        idx = MODELS.index(st.session_state.model_name) if st.session_state.model_name in MODELS else 0
        st.session_state.model_name = st.selectbox("Model", MODELS, index=idx)
    with c2:
        st.session_state.confidence_threshold = st.slider(
            "Spot-check confidence threshold", 0.0, 1.0,
            st.session_state.confidence_threshold, 0.05,
            help="Decisions below this confidence are flagged for human review.",
        )
        st.session_state.review_excludes = st.checkbox(
            "Always spot-check excluded records", value=st.session_state.review_excludes)

    st.session_state.criteria = st.text_area(
        "Inclusion / exclusion criteria", value=st.session_state.criteria, height=120)

    st.markdown("**Full-text exclusion reasons** (one per line — used for PRISMA counts)")
    reasons_text = st.text_area(
        "Exclusion reasons", value="\n".join(st.session_state.exclusion_reasons),
        height=160, label_visibility="collapsed")
    st.session_state.exclusion_reasons = [r.strip() for r in reasons_text.splitlines() if r.strip()]

    st.markdown("**Extraction schema** — one field per line: `name | type | hint`. "
                "Type may be `number`, `boolean`, or `enum(a, b, c)`.")
    st.session_state.schema_text = st.text_area(
        "Extraction schema", value=st.session_state.schema_text, height=160,
        label_visibility="collapsed")
    parsed_fields = rc.parse_schema(st.session_state.schema_text)
    if parsed_fields:
        st.caption("Parsed fields: " + ", ".join(f"`{f.name}` ({f.type})" for f in parsed_fields))


# --------------------------------------------------------------------------- #
# Stage 1 — Title / Abstract
# --------------------------------------------------------------------------- #

with tab_ta:
    st.subheader("Title & Abstract screening")
    st.caption("Upload citation exports (RIS/NBIB/CSV) — the standard way to screen many "
               "abstracts. PDFs also work but are unusual at this stage.")

    up = st.file_uploader(
        "Citation files (.ris, .nbib, .csv, .txt) or PDFs",
        type=["ris", "nbib", "csv", "txt", "pdf"], accept_multiple_files=True, key="ta_upload")

    pdf_files = [f for f in (up or []) if f.name.lower().endswith(".pdf")]
    cite_files = [f for f in (up or []) if not f.name.lower().endswith(".pdf")]

    records = read_records(cite_files) if cite_files else []
    if records:
        st.info(f"Parsed **{len(records)}** citation records.")
    if pdf_files:
        st.info(f"{len(pdf_files)} PDF(s) will be screened directly.")

    if st.button("▶️ Run title/abstract screening", type="primary",
                 disabled=not (records or pdf_files)):
        client = get_client()
        if client:
            results = []
            total = len(records) + len(pdf_files)
            prog = st.progress(0.0)
            done = 0

            for rec in records:
                try:
                    res = client.screen_title_abstract(rec, st.session_state.criteria)
                except Exception as exc:  # noqa: BLE001
                    res = {"decision": "unclear", "reason": f"Error: {exc}",
                           "exclusionReason": "", "confidence": 0.0}
                flag = rc.needs_review(res, st.session_state.confidence_threshold,
                                       st.session_state.review_excludes)
                results.append({
                    "id": rec.id, "title": rec.title, "abstract": rec.abstract,
                    "authors": rec.authors, "year": rec.year, "source": rec.source,
                    "stage": "title_abstract", **res, "needs_review": flag, "confirmed": False,
                })
                done += 1
                prog.progress(done / total)

            for f in pdf_files:
                try:
                    res = client.screen_full_text(f.getvalue(), st.session_state.criteria,
                                                  st.session_state.exclusion_reasons)
                    res["exclusionReason"] = ""  # no formal PRISMA reason at T&A stage
                except Exception as exc:  # noqa: BLE001
                    res = {"decision": "unclear", "reason": f"Error: {exc}",
                           "exclusionReason": "", "confidence": 0.0}
                flag = rc.needs_review(res, st.session_state.confidence_threshold,
                                       st.session_state.review_excludes)
                results.append({
                    "id": f.name, "title": f.name, "abstract": "(from PDF)",
                    "authors": "", "year": "", "source": f.name,
                    "stage": "title_abstract", **res, "needs_review": flag, "confirmed": False,
                })
                done += 1
                prog.progress(done / total)

            st.session_state.ta_results = results
            log("ta_screening", count=len(results),
                include=sum(r["decision"] == "include" for r in results),
                exclude=sum(r["decision"] == "exclude" for r in results))
            st.success(f"Screened {len(results)} records.")

    render_screen_results("ta_results")


# --------------------------------------------------------------------------- #
# Stage 2 — Full text
# --------------------------------------------------------------------------- #

with tab_ft:
    st.subheader("Full-text screening")
    st.caption("Upload the PDFs of records that passed title/abstract. Each exclusion "
               "gets one structured reason for the PRISMA diagram.")
    ft_up = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, key="ft_upload")

    if st.button("▶️ Run full-text screening", type="primary", disabled=not ft_up):
        client = get_client()
        if client:
            results = []
            prog = st.progress(0.0)
            for j, f in enumerate(ft_up, start=1):
                try:
                    res = client.screen_full_text(f.getvalue(), st.session_state.criteria,
                                                  st.session_state.exclusion_reasons)
                except Exception as exc:  # noqa: BLE001
                    res = {"decision": "unclear", "reason": f"Error: {exc}",
                           "exclusionReason": "", "confidence": 0.0}
                flag = rc.needs_review(res, st.session_state.confidence_threshold,
                                       st.session_state.review_excludes)
                results.append({
                    "id": f.name, "title": f.name, "abstract": "",
                    "authors": "", "year": "", "source": f.name,
                    "stage": "full_text", **res, "needs_review": flag, "confirmed": False,
                })
                prog.progress(j / len(ft_up))
            st.session_state.ft_results = results
            log("fulltext_screening", count=len(results),
                include=sum(r["decision"] == "include" for r in results),
                exclude=sum(r["decision"] == "exclude" for r in results))
            st.success(f"Screened {len(results)} full texts.")

    render_screen_results("ft_results")


# --------------------------------------------------------------------------- #
# Stage 3 — Data extraction
# --------------------------------------------------------------------------- #

with tab_extract:
    st.subheader("Custom data extraction")
    st.caption("Upload included study PDFs. Each field is extracted with a verbatim "
               "source quote so values are verifiable.")
    fields = rc.parse_schema(st.session_state.schema_text)
    if not fields:
        st.warning("Define an extraction schema in the Setup tab first.")
    ex_up = st.file_uploader("Included PDFs", type=["pdf"], accept_multiple_files=True, key="ex_upload")

    if st.button("▶️ Run extraction", type="primary", disabled=not (ex_up and fields)):
        client = get_client()
        if client:
            extractions = []
            prog = st.progress(0.0)
            for j, f in enumerate(ex_up, start=1):
                try:
                    data = client.extract_data(f.getvalue(), fields)
                except Exception as exc:  # noqa: BLE001
                    data = {fld.name: {"value": f"Error: {exc}", "source_quote": ""} for fld in fields}
                extractions.append({"study": f.name, "fields": data})
                prog.progress(j / len(ex_up))
            st.session_state.extractions = extractions
            log("extraction", count=len(extractions), fields=[f.name for f in fields])
            st.success(f"Extracted from {len(extractions)} studies.")

    if st.session_state.extractions:
        st.dataframe(extractions_to_df(st.session_state.extractions), use_container_width=True)
        with st.expander("Show source quotes"):
            st.dataframe(extractions_to_df(st.session_state.extractions, include_quotes=True),
                         use_container_width=True)


# --------------------------------------------------------------------------- #
# Stage 4 — Export & PRISMA
# --------------------------------------------------------------------------- #

with tab_export:
    st.subheader("Export & PRISMA counts")

    ta = st.session_state.ta_results
    ft = st.session_state.ft_results

    ta_inc = sum(r["decision"] == "include" for r in ta)
    ta_exc = sum(r["decision"] == "exclude" for r in ta)
    ft_inc = sum(r["decision"] == "include" for r in ft)
    ft_exc = sum(r["decision"] == "exclude" for r in ft)

    st.markdown(f"""
| PRISMA stage | Count |
|---|---|
| Records screened (title/abstract) | {len(ta)} |
| Excluded at title/abstract | {ta_exc} |
| Assessed for eligibility (full text) | {len(ft)} |
| Excluded at full text | {ft_exc} |
| **Included in synthesis** | **{ft_inc if ft else ta_inc}** |
""")

    if ft_exc:
        reasons: dict[str, int] = {}
        for r in ft:
            if r["decision"] == "exclude":
                k = r.get("exclusionReason") or "Unspecified"
                reasons[k] = reasons.get(k, 0) + 1
        st.markdown("**Full-text exclusions by reason**")
        st.table(pd.DataFrame([{"reason": k, "n": v} for k, v in reasons.items()]))

    st.divider()
    st.markdown("### Downloads")
    c1, c2, c3 = st.columns(3)
    if ta:
        c1.download_button("T&A results (CSV)", screen_df(ta).to_csv(index=False),
                           "ta_results.csv", "text/csv")
    if ft:
        c2.download_button("Full-text results (CSV)", screen_df(ft).to_csv(index=False),
                           "fulltext_results.csv", "text/csv")
    if st.session_state.extractions:
        edf = extractions_to_df(st.session_state.extractions, include_quotes=True)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            edf.to_excel(writer, index=False, sheet_name="evidence_table")
        c3.download_button("Evidence table (XLSX)", buf.getvalue(), "evidence_table.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.markdown("### Audit log")
    st.caption("Every AI decision, human override, and run — download this for your methods section.")
    if st.session_state.audit_log:
        st.download_button("Audit log (JSON)", json.dumps(st.session_state.audit_log, indent=2),
                           "audit_log.json", "application/json")
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)
    else:
        st.caption("No activity logged yet.")

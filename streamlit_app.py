"""Universal Systematic Review Assistant — Streamlit app.

Steps:
  1. Title & Abstract screening — upload a Search Results CSV (or RIS/PDF);
     screen each record against inclusion + exclusion criteria; download the
     SAME csv annotated with AI decision/confidence/reason (CSV in -> CSV out).
  2. Full-text screening — upload PDFs; structured single exclusion reason
     per excluded study (for PRISMA).
  3. Custom data extraction — list the fields you need; get an evidence table
     (CSV + XLSX) with a verbatim source quote per value.

Cross-cutting: AI decides, but excluded and low-confidence records are queued
for a human spot-check; batch processing for free-tier stability; PRISMA counts;
audit log; and a reset button. Single-user, no database — state lives in the
session and downloads.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import review_core as rc

st.set_page_config(page_title="Universal Systematic Review Assistant", page_icon="🔬", layout="wide")

MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]
BATCH_SIZES = [1, 5, 10, 20]

RESET_KEYS = ["ta_results", "ft_results", "extractions", "audit_log", "ta_csv_df", "ta_csv_ids"]


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def _init_state():
    ss = st.session_state
    ss.setdefault("api_key", "")
    ss.setdefault("model_name", "gemini-2.0-flash")
    ss.setdefault("batch_size", 5)
    ss.setdefault("throttle", True)
    ss.setdefault("confidence_threshold", 0.7)
    ss.setdefault("review_excludes", True)
    ss.setdefault("exclusion_reasons", list(rc.DEFAULT_EXCLUSION_REASONS))
    ss.setdefault("schema_text", rc.DEFAULT_SCHEMA_TEXT)
    # split inclusion / exclusion criteria per stage
    ss.setdefault("ta_inclusion", rc.DEFAULT_INCLUSION)
    ss.setdefault("ta_exclusion", rc.DEFAULT_EXCLUSION)
    ss.setdefault("ft_inclusion", rc.DEFAULT_INCLUSION)
    ss.setdefault("ft_exclusion", rc.DEFAULT_EXCLUSION)
    # results
    ss.setdefault("ta_results", [])
    ss.setdefault("ft_results", [])
    ss.setdefault("extractions", [])
    ss.setdefault("audit_log", [])
    ss.setdefault("ta_csv_df", None)   # original uploaded CSV (for CSV-out)
    ss.setdefault("ta_csv_ids", [])    # record id per CSV row (aligned to ta_csv_df)


_init_state()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log(event: str, **detail):
    st.session_state.audit_log.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event, **detail,
    })


def get_client() -> "rc.GeminiClient | None":
    key = (st.session_state.api_key or st.secrets.get("GEMINI_API_KEY", "")).strip()
    if not key:
        st.error("No Gemini API key. Add it in **Setup** or as GEMINI_API_KEY in Streamlit secrets.")
        return None
    if not key.startswith("AIza"):
        st.warning("⚠️ This doesn't look like a Google AI Studio API key (those start with `AIza`). "
                   "Get one at https://aistudio.google.com/apikey — a wrong credential type will be rejected.")
    try:
        return rc.GeminiClient(key, st.session_state.model_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not initialise Gemini client: {rc.classify_api_error(exc)}")
        return None


def badge(decision: str) -> str:
    return {"include": "🟢 include", "exclude": "🔴 exclude"}.get(decision, "🟡 unclear")


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str, keep_default_na=False)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", dtype=str,
                       keep_default_na=False, engine="python", on_bad_lines="skip")


def records_from_df(df: pd.DataFrame, source: str):
    """Build screening records from a citation DataFrame, keeping row alignment.

    Returns (records, ids, detected_columns).
    """
    cols = rc.detect_citation_columns(list(df.columns))
    records, ids = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        rid = f"{source}-{i}"
        records.append(rc.Record(
            id=rid,
            title=rc.sanitize_text(str(row[cols["title"]]) if cols["title"] else "", 1000),
            abstract=rc.sanitize_text(str(row[cols["abstract"]]) if cols["abstract"] else "", 8000),
            authors=rc.sanitize_text(str(row[cols["authors"]]) if cols["authors"] else "", 1000),
            year=str(row[cols["year"]])[:4] if cols["year"] else "",
            source=source,
        ))
        ids.append(rid)
    return records, ids, cols


def run_ta_batches(client, records: list[rc.Record], inclusion: str, exclusion: str):
    """Screen records, batching per the configured batch size.

    Quota/auth errors abort the whole run (raised to the caller) so we don't
    write the same error into every row. Other per-batch errors are recorded
    against that batch only. Returns {record_id: result}.
    """
    import time
    batch = max(1, int(st.session_state.batch_size))
    n_batches = (len(records) + batch - 1) // batch
    results: dict[str, dict] = {}
    prog = st.progress(0.0)
    for bi, start in enumerate(range(0, len(records), batch)):
        chunk = records[start:start + batch]
        try:
            if batch == 1:
                res = {chunk[0].id: client.screen_title_abstract(chunk[0], inclusion, exclusion)}
            else:
                res = client.screen_title_abstract_batch(chunk, inclusion, exclusion)
        except (rc.QuotaError, rc.AuthError):
            raise  # stop the run; caller shows one clear message
        except Exception as exc:  # noqa: BLE001
            res = {r.id: {"decision": "unclear", "reason": f"Error: {exc}",
                          "exclusionReason": "", "confidence": 0.0} for r in chunk}
        results.update(res)
        prog.progress(min(1.0, (start + len(chunk)) / len(records)))
        if st.session_state.throttle and bi < n_batches - 1:
            time.sleep(4)  # stay under free-tier requests-per-minute
    return results


def make_row(rec: rc.Record, res: dict, stage: str) -> dict:
    flag = rc.needs_review(res, st.session_state.confidence_threshold, st.session_state.review_excludes)
    return {"id": rec.id, "title": rec.title, "abstract": rec.abstract,
            "authors": rec.authors, "year": rec.year, "source": rec.source,
            "stage": stage, **res, "needs_review": flag, "confirmed": False}


def results_by_id(results: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in results}


def screened_csv(df: pd.DataFrame, ids: list[str], by_id: dict[str, dict]) -> str:
    """Original CSV + appended AI columns, reflecting any human overrides."""
    out = df.copy()
    dec, conf, reason, flag = [], [], [], []
    for rid in ids:
        r = by_id.get(rid, {})
        dec.append(r.get("decision", ""))
        conf.append(r.get("confidence", ""))
        reason.append(r.get("reason", ""))
        flag.append("yes" if (r.get("needs_review") and not r.get("confirmed")) else "")
    out["AI_Decision"] = dec
    out["AI_Confidence"] = conf
    out["AI_Reason"] = reason
    out["Needs_Review"] = flag
    return out.to_csv(index=False)


def render_screen_results(key: str):
    results = st.session_state.get(key, [])
    if not results:
        return
    inc = sum(r["decision"] == "include" for r in results)
    exc = sum(r["decision"] == "exclude" for r in results)
    unc = sum(r["decision"] == "unclear" for r in results)
    flagged = sum(r["needs_review"] and not r["confirmed"] for r in results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Include", inc); m2.metric("Exclude", exc)
    m3.metric("Unclear", unc); m4.metric("Needs review", flagged)

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
                log("override", id=r["id"], to="include"); st.rerun()
            if cols[1].button("❌ Set exclude", key=f"{key}_exc_{i}"):
                r["decision"], r["confirmed"] = "exclude", True
                log("override", id=r["id"], to="exclude"); st.rerun()
            if cols[2].button("✔️ Confirm AI", key=f"{key}_ok_{i}"):
                r["confirmed"] = True
                log("confirm", id=r["id"], decision=r["decision"]); st.rerun()


def screen_df(results) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": r["id"], "title": r["title"], "year": r.get("year", ""),
        "decision": r["decision"], "confidence": r["confidence"],
        "exclusion_reason": r.get("exclusionReason", ""),
        "reason": r["reason"], "needs_review": r["needs_review"], "confirmed": r["confirmed"],
    } for r in results])


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


# --------------------------------------------------------------------------- #
# Header + tabs
# --------------------------------------------------------------------------- #

st.title("🔬 Universal Systematic Review Assistant")
st.caption("Upload a CSV of search results, screen against your inclusion/exclusion criteria, and "
           "download an annotated CSV — plus full-text screening and custom data extraction. "
           "The AI decides; excluded & low-confidence records are queued for your spot-check.")

tab_setup, tab_ta, tab_ft, tab_extract, tab_export = st.tabs(
    ["⚙️ Setup", "📊 Step 1: Title & Abstract", "📄 Step 2: Full-Text",
     "🧬 Step 3: Data Extraction", "📤 Export & PRISMA"])


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

with tab_setup:
    st.subheader("Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.api_key = st.text_input(
            "Gemini API key", value=st.session_state.api_key, type="password",
            help="Or set GEMINI_API_KEY in Streamlit secrets and leave this blank.")
        idx = MODELS.index(st.session_state.model_name) if st.session_state.model_name in MODELS else 0
        st.session_state.model_name = st.selectbox("Model", MODELS, index=idx)
        st.session_state.batch_size = st.selectbox(
            "Select Batch Size", BATCH_SIZES,
            index=BATCH_SIZES.index(st.session_state.batch_size) if st.session_state.batch_size in BATCH_SIZES else 1,
            help="Records screened per API call at the title/abstract stage. "
                 "Larger = fewer calls (kinder to the free tier); smaller = more robust.")
        st.session_state.throttle = st.checkbox(
            "Free-tier safe mode (throttle requests)", value=st.session_state.throttle,
            help="Pause ~4s between batches to stay under the free-tier requests-per-minute limit.")
    with c2:
        st.session_state.confidence_threshold = st.slider(
            "Spot-check confidence threshold", 0.0, 1.0, st.session_state.confidence_threshold, 0.05,
            help="Decisions below this confidence are flagged for human review.")
        st.session_state.review_excludes = st.checkbox(
            "Always spot-check excluded records", value=st.session_state.review_excludes)
        st.markdown("&nbsp;")
        if st.button("🗑️ Reset Matrix Memory", help="Clear all screening results, extractions and the audit log."):
            for k in RESET_KEYS:
                st.session_state[k] = [] if k != "ta_csv_df" else None
            st.session_state.ta_csv_ids = []
            st.success("Cleared all results and logs.")

    st.markdown("**Full-text exclusion reasons** (one per line — used for PRISMA counts)")
    reasons_text = st.text_area("Exclusion reasons", value="\n".join(st.session_state.exclusion_reasons),
                                height=150, label_visibility="collapsed")
    st.session_state.exclusion_reasons = [r.strip() for r in reasons_text.splitlines() if r.strip()]

    st.markdown("**Extraction schema** — one field per line. Name only, or `name | type | hint`. "
                "Type may be `number`, `boolean`, or `enum(a, b, c)`.")
    st.session_state.schema_text = st.text_area("Extraction schema", value=st.session_state.schema_text,
                                                height=160, label_visibility="collapsed")
    parsed_fields = rc.parse_schema(st.session_state.schema_text)
    if parsed_fields:
        st.caption("Parsed fields: " + ", ".join(f"`{f.name}` ({f.type})" for f in parsed_fields))


# --------------------------------------------------------------------------- #
# Step 1 — Title & Abstract  (CSV in -> CSV out)
# --------------------------------------------------------------------------- #

with tab_ta:
    st.subheader("Step 1: Title & Abstract Screening")
    c1, c2 = st.columns(2)
    st.session_state.ta_inclusion = c1.text_area(
        "Abstract Inclusion Criteria", value=st.session_state.ta_inclusion, height=140)
    st.session_state.ta_exclusion = c2.text_area(
        "Abstract Exclusion Criteria", value=st.session_state.ta_exclusion, height=140)

    up = st.file_uploader(
        "Upload Search Results CSV (also accepts .ris / .nbib / PDF)",
        type=["csv", "ris", "nbib", "txt", "pdf"], accept_multiple_files=True, key="ta_upload")

    csv_files = [f for f in (up or []) if f.name.lower().endswith(".csv")]
    ris_files = [f for f in (up or []) if f.name.lower().endswith((".ris", ".nbib", ".txt"))]
    pdf_files = [f for f in (up or []) if f.name.lower().endswith(".pdf")]

    # Build the master CSV DataFrame (all uploaded CSVs stacked) + records.
    master_df = None
    csv_records: list[rc.Record] = []
    csv_ids: list[str] = []
    if csv_files:
        frames = []
        for f in csv_files:
            df = read_csv_bytes(f.getvalue())
            df.insert(0, "__source_file", f.name)
            frames.append(df)
        master_df = pd.concat(frames, ignore_index=True)
        csv_records, csv_ids, cols = records_from_df(master_df, "csv")
        st.info(f"Loaded **{len(master_df)}** rows from {len(csv_files)} CSV(s). "
                f"Detected columns → title: `{cols['title']}`, abstract: `{cols['abstract']}`.")
        if not cols["title"] and not cols["abstract"]:
            st.warning("No title/abstract column detected. Rename a column to 'Title' / 'Abstract' "
                       "or check the file — screening quality will be poor without them.")
    ris_records: list[rc.Record] = []
    for f in ris_files:
        ris_records.extend(rc.parse_ris(f.getvalue().decode("utf-8", errors="replace"), source=f.name))
    if ris_records:
        st.info(f"Parsed **{len(ris_records)}** RIS records.")
    if pdf_files:
        st.info(f"{len(pdf_files)} PDF(s) will be screened directly.")

    total_records = len(csv_records) + len(ris_records)
    if st.button("▶️ Run title/abstract screening", type="primary",
                 disabled=not (total_records or pdf_files)):
        client = get_client()
        if client:
            inc, exc = st.session_state.ta_inclusion, st.session_state.ta_exclusion
            results = []
            try:
                all_records = csv_records + ris_records
                if all_records:
                    by_id = run_ta_batches(client, all_records, inc, exc)
                    for rec in all_records:
                        results.append(make_row(rec, by_id[rec.id], "title_abstract"))

                for f in pdf_files:
                    res = client.screen_full_text(f.getvalue(), inc, exc, st.session_state.exclusion_reasons)
                    res["exclusionReason"] = ""  # no formal PRISMA reason at T&A stage
                    rec = rc.Record(id=f.name, title=f.name, abstract="(from PDF)", source=f.name)
                    results.append(make_row(rec, res, "title_abstract"))
            except (rc.QuotaError, rc.AuthError) as api_exc:
                st.error(f"⛔ {api_exc}")
                log("api_error", stage="title_abstract", error=str(api_exc))
                st.stop()

            st.session_state.ta_results = results
            st.session_state.ta_csv_df = master_df
            st.session_state.ta_csv_ids = csv_ids
            log("ta_screening", count=len(results), batch_size=st.session_state.batch_size,
                include=sum(r["decision"] == "include" for r in results),
                exclude=sum(r["decision"] == "exclude" for r in results))
            st.success(f"Screened {len(results)} records.")

    # CSV in -> CSV out download
    if st.session_state.ta_csv_df is not None and st.session_state.ta_results:
        out_csv = screened_csv(st.session_state.ta_csv_df, st.session_state.ta_csv_ids,
                               results_by_id(st.session_state.ta_results))
        st.download_button("⬇️ Download screened CSV (original + AI columns)", out_csv,
                           "screened_results.csv", "text/csv", type="primary")

    render_screen_results("ta_results")


# --------------------------------------------------------------------------- #
# Step 2 — Full text
# --------------------------------------------------------------------------- #

with tab_ft:
    st.subheader("Step 2: Full-Text Screening")
    c1, c2 = st.columns(2)
    st.session_state.ft_inclusion = c1.text_area(
        "Full-Text Inclusion Criteria", value=st.session_state.ft_inclusion, height=140)
    st.session_state.ft_exclusion = c2.text_area(
        "Full-Text Exclusion Criteria", value=st.session_state.ft_exclusion, height=140)

    ft_up = st.file_uploader("Upload Full-Text PDFs", type=["pdf"], accept_multiple_files=True, key="ft_upload")

    if st.button("▶️ Run full-text screening", type="primary", disabled=not ft_up):
        client = get_client()
        if client:
            import time
            inc, exc = st.session_state.ft_inclusion, st.session_state.ft_exclusion
            results, prog = [], st.progress(0.0)
            try:
                for j, f in enumerate(ft_up, start=1):
                    res = client.screen_full_text(f.getvalue(), inc, exc, st.session_state.exclusion_reasons)
                    rec = rc.Record(id=f.name, title=f.name, source=f.name)
                    results.append(make_row(rec, res, "full_text"))
                    prog.progress(j / len(ft_up))
                    if st.session_state.throttle and j < len(ft_up):
                        time.sleep(4)
            except (rc.QuotaError, rc.AuthError) as api_exc:
                st.error(f"⛔ {api_exc}")
                log("api_error", stage="full_text", error=str(api_exc))
                st.stop()
            st.session_state.ft_results = results
            log("fulltext_screening", count=len(results),
                include=sum(r["decision"] == "include" for r in results),
                exclude=sum(r["decision"] == "exclude" for r in results))
            st.success(f"Screened {len(results)} full texts.")

    render_screen_results("ft_results")


# --------------------------------------------------------------------------- #
# Step 3 — Data extraction
# --------------------------------------------------------------------------- #

with tab_extract:
    st.subheader("Step 3: Custom Data Extraction")
    st.caption("Define fields in the Setup tab. Each field is extracted with a verbatim source "
               "quote so values are verifiable.")
    fields = rc.parse_schema(st.session_state.schema_text)
    if not fields:
        st.warning("Define an extraction schema in the Setup tab first.")
    ex_up = st.file_uploader("Upload included study PDFs", type=["pdf"],
                             accept_multiple_files=True, key="ex_upload")

    if st.button("▶️ Run extraction", type="primary", disabled=not (ex_up and fields)):
        client = get_client()
        if client:
            import time
            extractions, prog = [], st.progress(0.0)
            try:
                for j, f in enumerate(ex_up, start=1):
                    data = client.extract_data(f.getvalue(), fields)
                    extractions.append({"study": f.name, "fields": data})
                    prog.progress(j / len(ex_up))
                    if st.session_state.throttle and j < len(ex_up):
                        time.sleep(4)
            except (rc.QuotaError, rc.AuthError) as api_exc:
                st.error(f"⛔ {api_exc}")
                log("api_error", stage="extraction", error=str(api_exc))
                st.stop()
            st.session_state.extractions = extractions
            log("extraction", count=len(extractions), fields=[f.name for f in fields])
            st.success(f"Extracted from {len(extractions)} studies.")

    if st.session_state.extractions:
        st.dataframe(extractions_to_df(st.session_state.extractions), use_container_width=True)
        with st.expander("Show source quotes"):
            st.dataframe(extractions_to_df(st.session_state.extractions, include_quotes=True),
                         use_container_width=True)
        edf = extractions_to_df(st.session_state.extractions, include_quotes=True)
        st.download_button("⬇️ Evidence table (CSV)", edf.to_csv(index=False),
                           "evidence_table.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Export & PRISMA
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
    st.caption("Every AI decision, human override, and run — download for your methods section.")
    if st.session_state.audit_log:
        st.download_button("Audit log (JSON)", json.dumps(st.session_state.audit_log, indent=2),
                           "audit_log.json", "application/json")
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)
    else:
        st.caption("No activity logged yet.")

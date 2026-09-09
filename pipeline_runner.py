"""
Runs the real app.py cascade (Step 1 -> Step 2 -> Step 3 -> Ideal export) end
to end, no Streamlit server and no manual clicks — by loading app.py's own
functions via `runpy` (same trick already proven in debug_pipeline.py) and
driving it with the SAME auto-run flags the app's own "All-in-One Input"
flow already uses.

Why this exists instead of a fresh reimplementation of the cascade: app.py's
Step 1/2/3 logic has been debugged extensively this project (role hierarchy,
Sr. Accountant/MRR/Record ID resolution, POD case handling...). Reusing it
directly guarantees this automated report matches the Streamlit app exactly,
with zero drift.

Key facts this relies on (verified empirically against streamlit==1.46.1):
  - Outside a real `streamlit run`, every widget (file_uploader, button,
    selectbox, radio) returns its default value regardless of what's in
    st.session_state — so files/choices must be injected at the session_state
    keys the app reads directly, not "simulated" through the widgets.
  - st.stop() is a no-op without a ScriptRunContext — it does NOT halt
    execution, so the Step 2 gate never blocks the script from reaching
    Step 3 in the same pass.
  - @st.fragment function bodies never run at all (return None immediately)
    — so the "HubSpot Sync" fragment is simply skipped; hs_parsed must be
    injected directly instead.
"""
from __future__ import annotations

import io
import os
import runpy
from datetime import datetime
from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")


class _FakeUpload(io.BytesIO):
    """Minimal stand-in for a Streamlit UploadedFile — io.BytesIO already
    implements a fully correct .seek()/.read(); we only add .name, which is
    all app.py's parser functions use beyond the file protocol."""
    def __init__(self, path_or_bytes, name: str):
        if isinstance(path_or_bytes, (bytes, bytearray)):
            data = bytes(path_or_bytes)
        else:
            with open(path_or_bytes, "rb") as f:
                data = f.read()
        super().__init__(data)
        self.name = name


def run_pipeline(volume_path: str, hc_path: str, hubspot_token: str, out_dir: str, log=print) -> str:
    """Run the full Ideal cascade and write the export Excel into `out_dir`.
    Returns the path of the written file."""
    import hubspot_client

    log("=" * 70)
    log("Step 0 — loading app.py's functions (no Streamlit server involved)...")
    ns = runpy.run_path(str(APP_PATH), run_name="_auto_capacity_report")
    st = ns["st"]

    # ── 1. HC Weekly Report ────────────────────────────────────────────────
    log("Step 1 — parsing HC Weekly Report...")
    with open(hc_path, "rb") as f:
        hc_bytes = f.read()
    hc_data = ns["_process_hc_report"](hc_bytes)
    st.session_state["hc_data"] = hc_data
    st.session_state["_hc_file_bytes"] = hc_bytes
    log(f"  HC parsed: total={hc_data.get('total', 0)} active staff, "
        f"mgr_total={hc_data.get('mgr_total', 0)} managers.")

    # ── 2. HubSpot — fetched live via API instead of a manual export ───────
    log("Step 2 — fetching client data from HubSpot API...")
    hs_df = hubspot_client.fetch_companies_dataframe(hubspot_token, log=log)
    hs_upload = _FakeUpload(_dataframe_to_xlsx_bytes(hs_df), name="hubspot_live.xlsx")
    hs_parsed = ns["_parse_hubspot_file"](hs_upload)
    st.session_state["hs_parsed"] = hs_parsed
    log(f"  HubSpot parsed: {len(hs_parsed)} eligible companies.")

    # ── 3. Volume/AHT file — via _load_volume_aht (NOT the button's direct-
    # read path, which bypasses the srs-sheet Sr. Accountant resolution) ───
    log("Step 3 — parsing Volume/AHT file (srs sheet included)...")
    vol_upload = _FakeUpload(volume_path, name=Path(volume_path).name)
    vol_log_lines = []
    df_vol = ns["_load_volume_aht"](vol_upload, lambda m: vol_log_lines.append(m))
    for line in vol_log_lines:
        log(f"  {line}")
    st.session_state["pipeline_vol_merged"] = df_vol
    log(f"  Volume parsed: {len(df_vol)} rows.")

    # These two gates are never actually blocking outside a real Streamlit
    # run (st.stop() is a no-op there), but set them anyway for robustness
    # against a future Streamlit version that changes that behavior.
    st.session_state["hs_sync_choice"] = "skip"
    st.session_state["s2_efficiency_choice"] = "skip"
    st.session_state["_s2_proceed"] = True
    st.session_state["_show_step1"] = True

    # ── PHASE A — Step 1 baseline ONLY (no cascade yet). This is what
    # populates calc_data (dict_hrs_per_fte/dict_workable_days) and the
    # runtime cost/utilization constants that AI Prediction needs below —
    # exactly like the manual Streamlit flow, where AI Prediction always
    # runs against an already-computed Step 1, then Step 1+3 re-run once
    # more with the synthetic rows mixed in (app.py:944-964). ─────────────
    st.session_state["_auto_run_baseline"] = True
    st.session_state.pop("_auto_run_cascade", None)
    log("Step 4 — running Step 1 (baseline) to prime calc_data for AI Prediction...")
    runpy.run_path(str(APP_PATH), run_name="_auto_capacity_report")

    # ── PHASE B — AI Prediction for HubSpot Onboarding clients not yet in
    # the volume file (no human review — confirmed with the user). ───────
    log("Step 5 — AI Prediction for new (Onboarding) clients...")
    try:
        import pandas as pd
        import ai_prediction_engine as aipe

        df_new_zero = _find_new_clients(ns, hs_parsed, st.session_state.df_clean)
        df_new_partial = _find_partial_month_clients(ns, st.session_state.df_clean, hs_parsed, log=log)

        if not df_new_partial.empty and not df_new_zero.empty:
            _dup = df_new_zero["company_name"].astype(str).str.lower().str.strip().isin(
                df_new_partial["company_name"].astype(str).str.lower().str.strip()
            )
            df_new_zero = df_new_zero[~_dup]

        if df_new_partial.empty:
            df_new = df_new_zero
        elif df_new_zero.empty:
            df_new = df_new_partial
        else:
            df_new = pd.concat([df_new_zero, df_new_partial], ignore_index=True)

        if df_new.empty:
            log("  AI Prediction: no HubSpot Onboarding clients and no partial-cycle "
                "clients found — skipping.")
        else:
            log(f"  AI Prediction: {len(df_new)} candidate client(s) total "
                f"({len(df_new_zero)} new/zero-volume, {len(df_new_partial)} existing-but-partial-cycle).")
            df_pred = aipe.train_and_predict(ns, st.session_state.df_clean, df_new, ai_month_idx=0, log=log)
            if df_pred is not None:
                st.session_state.df_clean = aipe.build_synthetic_rows(df_pred, st.session_state.df_clean)
                # Step 1's baseline handler always re-reads pipeline_vol_merged
                # (never df_clean) when no file was directly uploaded — without
                # updating this too, Phase C's "re-run Step 1" would silently
                # overwrite df_clean back to the original volume data, discarding
                # every synthetic row we just added.
                st.session_state.pipeline_vol_merged = st.session_state.df_clean.copy()
                ns["_build_client_master_map"]()
    except Exception as e:
        log(f"  AI Prediction failed ({e}) — continuing without new-client hours.")

    # ── PHASE C — Step 1 (rebuilt with synthetic rows) -> Step 3 -> export ──
    st.session_state["_auto_run_baseline"] = True
    st.session_state["_auto_run_cascade"] = True
    log("Step 6 — re-running Step 1 (baseline) -> Step 3 (cascade) -> Ideal export...")
    runpy.run_path(str(APP_PATH), run_name="_auto_capacity_report")

    xlsx_bytes = st.session_state.get("_cascade_export_buf_ideal")
    if not xlsx_bytes:
        raise RuntimeError(
            "Cascade did not produce an Ideal export buffer "
            "(st.session_state['_cascade_export_buf_ideal'] is empty). "
            "Check the log above for where the pipeline stopped."
        )

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"Capacity_Projection_Cascade_ROI_{datetime.now().strftime('%Y%m%d_%H%M')}_ideal.xlsx"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "wb") as f:
        f.write(xlsx_bytes)
    log(f"Done — wrote {out_path} ({len(xlsx_bytes):,} bytes).")
    return out_path


def _dataframe_to_xlsx_bytes(df) -> bytes:
    import pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Companies")
    return buf.getvalue()


def _find_new_clients(ns: dict, hs_parsed, df_clean):
    """HubSpot companies not yet represented in the volume file — matched by
    record_id (name as fallback), excluding Churn/blank lifecycle. Mirrors
    app.py:6692-6708 (the "Onboarding New Clients" detection used by the
    manual "Queue for AI Prediction" flow). `hs_parsed` is already filtered
    to POD-assigned companies upstream (hubspot_client.py), so no extra POD
    check is needed here.

    Returns a DataFrame shaped for ai_prediction_engine.train_and_predict():
    company_name, record_id, pod, go_live_date, mrr, pms, res_doors,
    res_prop, commercial_doors, commercial_properties, sqft_commercial,
    corp_books (the last 6 default to 0/Unknown — hs_parsed doesn't carry
    door/sqft counts, so these new clients predict at Tier 0/1, MRR(+PMS)
    only — still far better than the 0 hours they'd get otherwise)."""
    import pandas as pd
    _clean_record_id = ns["_clean_record_id"]
    _norm_name = ns["_norm_name"]

    if hs_parsed is None or hs_parsed.empty:
        return pd.DataFrame()

    baseline_names = (
        set(df_clean["client_name"].dropna().astype(str).apply(_norm_name))
        if "client_name" in df_clean.columns else set()
    )
    vol_rids = (
        set(_clean_record_id(df_clean["record_id"]).replace("", pd.NA).dropna())
        if "record_id" in df_clean.columns else set()
    )

    hs = hs_parsed.copy()
    lifecycle_norm = hs["_lifecycle"].astype(str).str.lower().str.strip()
    hs_rids = _clean_record_id(hs["record_id"]) if "record_id" in hs.columns else pd.Series("", index=hs.index)
    hs_name_in_vol = hs["client_name"].astype(str).apply(_norm_name).isin(baseline_names)
    in_vol = hs_rids.isin(vol_rids) | ((hs_rids == "") & hs_name_in_vol)

    lc_blank = {"—", "", "none", "nan"}
    candidates = hs[
        ~in_vol
        & ~lifecycle_norm.str.startswith("churn")
        & ~lifecycle_norm.isin(lc_blank)
    ].copy()
    if candidates.empty:
        return candidates

    return pd.DataFrame({
        "company_name": candidates["client_name"],
        "record_id": _clean_record_id(candidates["record_id"]) if "record_id" in candidates.columns else "",
        "pod": candidates.get("_pod", ""),
        "go_live_date": candidates.get("_start_date", ""),
        "mrr": candidates.get("_mrr", 0.0),
        "pms": candidates.get("_pms", "Unknown"),
        "res_doors": 0.0, "res_prop": 0.0,
        "commercial_doors": 0.0, "commercial_properties": 0.0,
        "sqft_commercial": 0.0, "corp_books": "Unknown",
    })


def _find_partial_month_clients(ns: dict, df_clean, hs_parsed, log=print):
    """Existing df_clean clients whose last month was NOT a full activity
    cycle — Status/Lifecycle reads Onboarding-like, or Go-Live falls within
    +/-20 days of today. Mirrors app.py:5567-5613 (the manual "New/Onboarding
    Clients" panel) and app.py:5692-5693 (the "Queue for AI Prediction"
    button's replace-set) — the exact same detection the manual Streamlit
    flow uses to warn "N of these already have hours in the input file...
    their existing hours will be replaced".

    Unlike `_find_new_clients` (which only catches clients with ZERO rows in
    df_clean), this catches clients that DO already have rows — their
    partial-cycle real hours are not representative of steady-state volume,
    so they get REPLACED by the AI-predicted projection instead.
    `build_synthetic_rows()` already purges-then-inserts by client_name, so
    returning these names as candidates is enough to get "replace" semantics
    for free — no separate merge logic needed.

    Returns a DataFrame shaped like `_find_new_clients()`'s output (ready to
    concat with it before calling `ai_prediction_engine.train_and_predict`),
    except feature columns here are pulled from the client's OWN existing
    df_clean data (POD/PMS/MRR/doors/sqft/corp_books) when available, giving
    better-tier predictions than the HubSpot-only Tier 0/1 fallback."""
    import pandas as pd

    if df_clean is None or df_clean.empty or "client_name" not in df_clean.columns:
        return pd.DataFrame()

    ci = {str(c).strip().lower(): c for c in df_clean.columns if pd.notna(c)}

    def _col(*substrings, exact=()):
        for e in exact:
            if e in ci:
                return ci[e]
        for sub in substrings:
            hit = next((v for k, v in ci.items() if sub in k), None)
            if hit:
                return hit
        return None

    st_col   = _col(exact=("status",))
    gl_col   = _col("go live", "go_live")
    pod_col  = _col(exact=("pod",))
    pms_col  = _col(exact=("pms",))
    mrr_col  = _col(exact=("mrr",))
    rid_col  = _col("record id", "record_id")
    resd_col = _col("res doors", "res_doors")
    resp_col = _col("res prop")
    cd_col   = _col("commercial doors", "comm doors")
    cp_col   = _col("commercial propert", "comm propert")
    sqft_col = _col("sqft")
    cb_col   = _col("corp books", "corp_books")

    agg = {}
    for c in (st_col, gl_col, pod_col, pms_col, mrr_col, rid_col,
              resd_col, resp_col, cd_col, cp_col, sqft_col, cb_col):
        if c and c not in agg:
            agg[c] = "first"
    for hc in ("Capacity Processing Hours", "Capacity reviewing hours"):
        if hc in df_clean.columns:
            agg[hc] = "sum"

    grp = df_clean.groupby("client_name", as_index=False).agg(agg) if agg else (
        df_clean[["client_name"]].drop_duplicates()
    )

    today_ts = pd.Timestamp.today().normalize()
    mask = pd.Series(False, index=grp.index)
    if st_col:
        mask |= grp[st_col].astype(str).str.lower().str.strip().isin(
            ["onboarding", "new client", "subscriber"]
        )
    if gl_col:
        gl_ser = pd.to_datetime(grp[gl_col], errors="coerce")
        gl_diff = (gl_ser - today_ts).dt.days
        mask |= gl_ser.notna() & (gl_diff >= -20) & (gl_diff <= 20)

    # Fallback when the volume file carries no Status column at all: use
    # HubSpot's own lifecycle stage instead (same fallback app.py uses at
    # the row level, app.py:5623-5630).
    if not st_col and hs_parsed is not None and not hs_parsed.empty and "_lifecycle" in hs_parsed.columns:
        _norm_name = ns["_norm_name"]
        hs_lc = hs_parsed.set_index(hs_parsed["client_name"].astype(str).apply(_norm_name))["_lifecycle"]
        hs_lc = hs_lc[~hs_lc.index.duplicated(keep="first")]
        grp_names = grp["client_name"].astype(str).apply(_norm_name)
        lc_vals = grp_names.map(hs_lc).astype(str).str.lower().str.strip()
        mask |= lc_vals.isin(["onboarding", "new client", "subscriber"])

    sel = grp[mask].copy()
    if sel.empty:
        return sel

    log(f"  Partial-cycle detection: {len(sel)} existing client(s) flagged "
        f"(Onboarding-like status and/or Go-Live within +/-20 days) — their "
        f"real hours will be REPLACED by the AI-predicted projection.")

    def _num(col):
        return pd.to_numeric(sel[col], errors="coerce").fillna(0.0) if col else 0.0

    def _txt(col, default=""):
        return sel[col].astype(str).replace({"nan": default, "None": default}).fillna(default) if col else default

    return pd.DataFrame({
        "company_name": sel["client_name"],
        "record_id": sel[rid_col] if rid_col else "",
        "pod": sel[pod_col] if pod_col else "",
        "go_live_date": sel[gl_col] if gl_col else "",
        "mrr": _num(mrr_col),
        "pms": _txt(pms_col, "Unknown"),
        "res_doors": _num(resd_col),
        "res_prop": _num(resp_col),
        "commercial_doors": _num(cd_col),
        "commercial_properties": _num(cp_col),
        "sqft_commercial": _num(sqft_col),
        "corp_books": _txt(cb_col, "Unknown"),
    })

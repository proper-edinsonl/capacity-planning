"""
Standalone pipeline debugger for Capacity Online V3.

Runs the REAL functions from app.py (imported via runpy, no Streamlit server
needed) against real input files, step by step. Each step prints:
  - what it's supposed to do
  - whether it ran (PASS/FAIL)
  - key validation metrics (row counts, non-null %, samples)
and saves a checkpoint (pickle) after each step so you can resume from any
point instead of re-running the whole pipeline.

Usage:
    python debug_pipeline.py                 # run all steps from scratch
    python debug_pipeline.py --from step5     # resume from a checkpoint
    python debug_pipeline.py --list           # list available checkpoints

Edit FILES below to point at your input files.
"""
import argparse
import io
import os
import pickle
import runpy
import sys
import traceback

CKPT_DIR = os.path.join(os.path.dirname(__file__), "_debug_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

FILES = {
    "volume":   r"D:\Gdrive\Unified Capacity Planning\2026\7-27\01 Volume_AHT_June EOM base input Removing New Clients.xlsx",
    "hc":       r"D:\Gdrive\Unified Capacity Planning\2026\7-27\02 Actual HC 7-27.xlsx",
    "hubspot":  r"D:\Gdrive\Unified Capacity Planning\2026\7-27\hubspot-crm-exports-capacity-clients-edy-do-not-del-2026-07-29.xlsx",
    "doorcount": r"D:\Gdrive\Unified Capacity Planning\2026\7-27\06 DoorCount_Variation_Template 7-27.xlsx",
    "remove_hours": r"D:\Gdrive\Unified Capacity Planning\2026\7-27\05 Remove_Hours_Template May Base.xlsx",
}

STEPS = []  # populated by @step decorator, in order


class _FakeUpload(io.BytesIO):
    """Minimal stand-in for a Streamlit UploadedFile — io.BytesIO already
    provides a fully correct .seek()/.read(); we only need to add .name,
    which is all app.py's parser functions use beyond the file protocol."""
    def __init__(self, path):
        with open(path, "rb") as f:
            super().__init__(f.read())
        self.name = os.path.basename(path)


def step(name, expected):
    def deco(fn):
        STEPS.append((name, expected, fn))
        return fn
    return deco


_UNPICKLEABLE_KEYS = {"ns", "st"}  # module refs — rebuilt fresh by the load_app step on resume


def _save_ckpt(name, state):
    picklable = {k: v for k, v in state.items() if k not in _UNPICKLEABLE_KEYS}
    with open(os.path.join(CKPT_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(picklable, f)


def _load_ckpt(name):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _pct(n, d):
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/0"


# ── Steps ────────────────────────────────────────────────────────────────

@step("load_app", "Import app.py's functions without launching the Streamlit server.")
def s_load_app(ctx):
    ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "app.py"), run_name="_debug_pipeline")
    ctx["ns"] = ns
    ctx["st"] = ns["st"]
    return {"ok": True, "detail": "app.py executed in bare mode, functions available"}


@step("load_volume", "Parse the Volume/AHT file → df with 'record_id' and 'MRR' populated for (almost) every row.")
def s_load_volume(ctx):
    ns, st = ctx["ns"], ctx["st"]
    log_events = []
    def log(msg):
        log_events.append(msg)
    up = _FakeUpload(FILES["volume"])
    df = ns["_load_volume_aht"](up, log)
    ctx["df_vol"] = df
    ctx["vol_log"] = log_events
    n = len(df)
    rid_nonblank = int((df.get("record_id", "").astype(str).str.strip() != "").sum()) if "record_id" in df.columns else 0
    mrr_col = "MRR" if "MRR" in df.columns else None
    import pandas as pd
    mrr_nonzero = int((pd.to_numeric(df[mrr_col], errors="coerce").fillna(0) != 0).sum()) if mrr_col else 0
    return {
        "ok": n > 0 and rid_nonblank > 0,
        "detail": (
            f"{n} rows · record_id non-blank: {_pct(rid_nonblank, n)} · "
            f"MRR column: {mrr_col!r} · MRR non-zero: {_pct(mrr_nonzero, n)}\n"
            f"log: {log_events[-5:]}"
        ),
    }


@step("srs_map", "Build record_id → email map from the 'srs' sheet.")
def s_srs_map(ctx):
    st = ctx["st"]
    srs_map = st.session_state.get("_srs_rid_email_map", {}) or {}
    ctx["srs_map"] = srs_map
    sample = dict(list(srs_map.items())[:5])
    return {
        "ok": len(srs_map) > 0,
        "detail": f"{len(srs_map)} record_id -> email mappings. Sample: {sample}",
    }


@step("set_df_clean", "Simulate app.py setting st.session_state.df_clean = the parsed volume df.")
def s_set_df_clean(ctx):
    st = ctx["st"]
    st.session_state.df_clean = ctx["df_vol"].copy()
    dfc = st.session_state.df_clean
    has_rid = "record_id" in dfc.columns
    has_mrr = "MRR" in dfc.columns
    return {
        "ok": has_rid and has_mrr,
        "detail": f"df_clean set, {len(dfc)} rows. has record_id={has_rid}, has MRR={has_mrr}",
    }


@step("load_hubspot", "Parse the HubSpot export → df with record_id, _mrr, client_name, _lifecycle, _is_terminating.")
def s_load_hubspot(ctx):
    ns, st = ctx["ns"], ctx["st"]
    up = _FakeUpload(FILES["hubspot"])
    hs = ns["_parse_hubspot_file"](up)
    st.session_state["hs_parsed"] = hs
    ctx["hs_parsed"] = hs
    n = len(hs)
    cols_present = [c for c in ["record_id", "_mrr", "client_name", "_lifecycle", "_is_terminating"] if c in hs.columns]
    rid_nonblank = int((hs["record_id"].astype(str).str.strip() != "").sum()) if "record_id" in hs.columns else 0
    lifecycle_vals = hs["_lifecycle"].value_counts().to_dict() if "_lifecycle" in hs.columns else {}
    term_count = int(hs["_is_terminating"].sum()) if "_is_terminating" in hs.columns else 0
    return {
        "ok": n > 0 and rid_nonblank > 0,
        "detail": (
            f"{n} rows · cols present: {cols_present} · record_id non-blank: {_pct(rid_nonblank, n)}\n"
            f"lifecycle values: {lifecycle_vals} · is_terminating count: {term_count}"
        ),
    }


@step("load_hc_report", "Parse the HC Weekly Report → role buckets, no employee falling into 'Other'.")
def s_load_hc(ctx):
    ns, st = ctx["ns"], ctx["st"]
    with open(FILES["hc"], "rb") as f:
        file_bytes = f.read()
    hc_data = ns["_process_hc_report"](file_bytes)
    st.session_state["hc_data"] = hc_data
    ctx["hc_data"] = hc_data
    detail_df = hc_data.get("detail")
    other_count = int((detail_df["Capacity Role"] == "Other").sum()) if detail_df is not None and "Capacity Role" in detail_df.columns else -1
    return {
        "ok": other_count == 0,
        "detail": (
            f"total={hc_data.get('total')} · by_role={hc_data.get('by_role')} · "
            f"acct_managers={hc_data.get('acct_managers')} · principal_accountants={hc_data.get('principal_accountants')} · "
            f"sr_acct_managers={hc_data.get('sr_acct_managers')} · mgr_total={hc_data.get('mgr_total')}\n"
            f"'Other' (unrecognized job titles) count: {other_count}"
            + (f" — TITLES: {detail_df[detail_df['Capacity Role']=='Other']['Job title'].unique().tolist()}" if other_count > 0 else "")
        ),
    }


@step("build_client_master_map", "Build _rid_map (record_id -> pod/sr/mrr/client_name) from df_clean.")
def s_build_cmm(ctx):
    ns, st = ctx["ns"], ctx["st"]
    ns["_build_client_master_map"]()
    rid_map = st.session_state.get("_rid_map", {}) or {}
    ctx["rid_map"] = rid_map
    sample_key = next(iter(rid_map), None)
    return {
        "ok": len(rid_map) > 0,
        "detail": f"{len(rid_map)} record_ids in _rid_map. Sample entry [{sample_key}]: {rid_map.get(sample_key)}",
    }


@step("person_client_extras", "Build MRR/Client Names/Client Status per email (the columns that were empty).")
def s_person_extras(ctx):
    ns, st = ctx["ns"], ctx["st"]
    srs_map = ctx["srs_map"]
    extras = ns["_build_person_client_extras"](srs_map)
    ctx["extras"] = extras
    n_with_mrr = sum(1 for v in extras.values() if v.get("mrr", 0))
    n_with_names = sum(1 for v in extras.values() if v.get("client_names"))
    n_with_status = sum(1 for v in extras.values() if v.get("status") not in ("—", "", None))
    sample_email = next(iter(extras), None)
    return {
        "ok": n_with_mrr > 0 and n_with_names > 0,
        "detail": (
            f"{len(extras)} emails resolved. MRR>0: {n_with_mrr} · has names: {n_with_names} · "
            f"has status: {n_with_status}\n"
            f"Sample [{sample_email}]: {extras.get(sample_email)}"
        ),
    }


@step("person_client_extras_no_srs_map", "REGRESSION CHECK: same as above but simulating the "
      "'Generate Baseline' button path — _srs_rid_email_map is EMPTY (that path never runs "
      "_load_volume_aht), so this must still work via df_clean's own 'Sr. Accountant' column, "
      "AND Onboarding clients (no volume rows yet) must still show up via the srs-sheet fallback.")
def s_person_extras_no_srs(ctx):
    ns, st = ctx["ns"], ctx["st"]
    # Also blank out the other srs-derived session keys _get_srs_rid_email_map()
    # would otherwise use, to truly simulate "nothing but the raw upload widget".
    st.session_state.pop('_srs_rid_email_map', None)
    st.session_state.pop('_srs_sheet_raw', None)
    st.session_state.pop('_vol_file_bytes', None)
    st.session_state['main_data_upload'] = _FakeUpload(FILES["volume"])

    extras = ns["_build_person_client_extras"]({})   # empty rid_email_map on purpose
    n_with_mrr = sum(1 for v in extras.values() if v.get("mrr", 0))
    n_with_names = sum(1 for v in extras.values() if v.get("client_names"))
    n_onboarding = sum(1 for v in extras.values() if "Onboarding" in v.get("status", ""))
    sample_email = "mariana.patino@proper.ai"  # assigned an Onboarding client per the srs sheet
    return {
        "ok": n_with_mrr > 0 and n_with_names > 0 and n_onboarding > 0,
        "detail": (
            f"{len(extras)} emails resolved with EVERY srs-derived session key cleared "
            f"(only the raw file_uploader widget available). "
            f"MRR>0: {n_with_mrr} · has names: {n_with_names} · people with an Onboarding client: {n_onboarding}\n"
            f"Sample [{sample_email}]: {extras.get(sample_email)}"
        ),
    }


# ── Runner ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_step", default=None, help="resume from this step name (uses saved checkpoint of the PRIOR step)")
    ap.add_argument("--list", action="store_true", help="list step names and saved checkpoints")
    args = ap.parse_args()

    if args.list:
        for name, expected, _ in STEPS:
            has_ckpt = os.path.exists(os.path.join(CKPT_DIR, f"{name}.pkl"))
            print(f"  [{'x' if has_ckpt else ' '}] {name:<24} — {expected}")
        return

    ctx = {}
    start_idx = 0
    if args.from_step:
        names = [n for n, _, _ in STEPS]
        if args.from_step not in names:
            print(f"Unknown step {args.from_step!r}. Known: {names}")
            sys.exit(1)
        start_idx = names.index(args.from_step)
        # load the last checkpoint before start_idx
        for i in range(start_idx - 1, -1, -1):
            prior_name = STEPS[i][0]
            loaded = _load_ckpt(prior_name)
            if loaded is not None:
                ctx = loaded
                print(f"Resumed context from checkpoint '{prior_name}'")
                break
        # `ns`/`st` (module refs) are never pickled — always rebuild them by
        # re-running the (fast) load_app step, then replay whatever this
        # session's data implies about st.session_state so later steps that
        # read session_state directly still see it.
        s_load_app(ctx)
        st = ctx["st"]
        if "df_vol" in ctx:
            st.session_state.df_clean = ctx["df_vol"].copy()
        if "hs_parsed" in ctx:
            st.session_state["hs_parsed"] = ctx["hs_parsed"]
        if "hc_data" in ctx:
            st.session_state["hc_data"] = ctx["hc_data"]
        if "srs_map" in ctx:
            st.session_state["_srs_rid_email_map"] = ctx["srs_map"]
        if "rid_map" in ctx:
            st.session_state["_rid_map"] = ctx["rid_map"]

    print("=" * 90)
    for i, (name, expected, fn) in enumerate(STEPS):
        if i < start_idx:
            continue
        print(f"\n[{i+1}/{len(STEPS)}] STEP: {name}")
        print(f"   Expected: {expected}")
        try:
            result = fn(ctx)
            ok = result.get("ok", False)
            detail = result.get("detail", "")
            status = "PASS ✅" if ok else "FAIL ❌"
            print(f"   Result:   {status}")
            for line in str(detail).splitlines():
                print(f"             {line}")
            _save_ckpt(name, ctx)
            if not ok:
                print(f"\n   ⚠️  Step '{name}' did not meet its expectation — stopping here.")
                print(f"   Fix the issue, then resume with: python debug_pipeline.py --from {name}")
                break
        except Exception:
            print("   Result:   CRASH 💥")
            traceback.print_exc()
            print(f"\n   Fix the issue, then resume with: python debug_pipeline.py --from {name}")
            break
    print("=" * 90)


if __name__ == "__main__":
    main()

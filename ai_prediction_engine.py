"""
Standalone reimplementation of app.py's "AI Prediction" logic (train a
RandomForest per task on existing clients, predict hours for new ones, build
synthetic df_clean rows) — usable outside Streamlit.

Why a reimplementation instead of calling into app.py directly: the real
logic lives inside `_make_ai_prediction_fragment`'s `@st.fragment` body
(app.py:235-1034), and fragment bodies never execute outside a real
`streamlit run` (see pipeline_runner.py's module docstring) — there is no
function to call. Rather than refactor that fragment (it's shared by 3 UI
instances: the early-access panel, Step 4 scenario planning, and a
prediction-only mode — touching it risks the manual tool the team uses
daily), this module mirrors ONLY the orchestration (train -> predict ->
build synthetic rows), reusing app.py's own constants/classes/functions via
`ns` (the module namespace `runpy.run_path` returns) wherever possible so
there's a single source of truth for feature-tier definitions, the AHT
learning curve, cost/utilization maps, and role-reviewer mapping.

Mirrors, line-range references into app.py as of this session:
  - Training + prediction: app.py:437-798
  - Synthetic df_clean row construction + merge: app.py:874-923
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def train_and_predict(ns: dict, df_clean_existing: pd.DataFrame, df_new: pd.DataFrame,
                       ai_month_idx: int = 0, log=print) -> pd.DataFrame | None:
    """Train one RandomForest per task on `df_clean_existing`, predict hours
    for every client in `df_new`, apply the 45-55% MRR cost band, and return
    a row-per-(client, task) DataFrame (`df_pred`) — the same shape
    `build_synthetic_rows()` below consumes. Returns None if there isn't
    enough training data or nothing to predict (logged, not raised, so the
    caller can just skip AI Prediction and continue the pipeline)."""
    st = ns["st"]
    ALL_NUM_COLS = ns["ALL_NUM_COLS"]
    ALL_CAT_COLS = ns["ALL_CAT_COLS"]
    COLS_T0_NUM = ns["COLS_T0_NUM"]; COLS_T1_NUM = ns["COLS_T1_NUM"]
    COLS_T2_NUM = ns["COLS_T2_NUM"]; COLS_T3_NUM = ns["COLS_T3_NUM"]; COLS_T4_NUM = ns["COLS_T4_NUM"]
    RandomForestRegressor = ns["RandomForestRegressor"]
    OneHotEncoder = ns["OneHotEncoder"]
    _reviewer_role = ns["_reviewer_role"]
    _clean_cols = ns["_clean_cols"]
    BASELINE_NETWORK_DAYS = ns["BASELINE_NETWORK_DAYS"]
    relativedelta = ns["relativedelta"]
    today = ns["today"]
    utilization_map = ns["utilization_map"]; cost_map = ns["cost_map"]
    util_acc1 = ns["util_acc1"]; util_sr = ns["util_sr"]
    cost_acc1 = ns["cost_acc1"]; cost_sr = ns["cost_sr"]
    absenteeism = ns["absenteeism"]; attrition = ns["attrition"]

    # ── 1. PREPARE TRAINING DATA ─────────────────────────────────────────
    df_vol = _clean_cols(df_clean_existing.copy())
    if "status" in df_vol.columns:
        df_vol = df_vol[df_vol["status"].astype(str).str.strip().str.lower() == "client"]
    if "go_live" in df_vol.columns:
        df_vol["go_live"] = pd.to_datetime(df_vol["go_live"], errors="coerce")
        df_vol = df_vol[df_vol["go_live"] < (pd.Timestamp.today() - pd.DateOffset(months=3))]
    if df_vol.empty:
        log("  AI Prediction: training data is empty after filters — skipping.")
        return None

    for col in ALL_NUM_COLS:
        df_vol[col] = pd.to_numeric(df_vol[col], errors="coerce").fillna(0) if col in df_vol.columns else 0.0

    if "type" in df_vol.columns and "subtype" in df_vol.columns:
        df_vol["task_name"] = df_vol["type"].astype(str) + " - " + df_vol["subtype"].astype(str)
    else:
        df_vol["task_name"] = "General Task"

    vol_target_candidates = [c for c in df_vol.columns if "closed_tickets" in c and "proc" in c]
    vol_target = vol_target_candidates[0] if vol_target_candidates else None
    if vol_target is None:
        log("  AI Prediction: could not find 'Closed tickets with Proc time' column — skipping.")
        return None
    df_vol[vol_target] = pd.to_numeric(df_vol[vol_target], errors="coerce").fillna(0)

    for col in ALL_CAT_COLS:
        if col not in df_vol.columns:
            df_vol[col] = "Unknown"
        df_vol[col] = df_vol[col].astype(str).replace(["nan", "None", ""], "Unknown").fillna("Unknown")

    client_features = (
        df_vol[["client_name"] + ALL_NUM_COLS + ALL_CAT_COLS]
        .drop_duplicates(subset=["client_name"])
        .set_index("client_name")
    )

    pms_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
    pms_encoded = pms_enc.fit_transform(client_features[["pms"]])
    pms_cols = pms_enc.get_feature_names_out(["pms"])

    cb_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
    cb_encoded = cb_enc.fit_transform(client_features[["corp_books"]])
    cb_cols = cb_enc.get_feature_names_out(["corp_books"])

    df_pms = pd.DataFrame(pms_encoded, columns=pms_cols, index=client_features.index)
    df_cb = pd.DataFrame(cb_encoded, columns=cb_cols, index=client_features.index)

    feat_t0 = COLS_T0_NUM
    feat_t1 = COLS_T1_NUM + list(pms_cols)
    feat_t2 = COLS_T2_NUM + list(pms_cols)
    feat_t3 = COLS_T3_NUM + list(pms_cols)
    feat_t4 = COLS_T4_NUM + list(pms_cols) + list(cb_cols)

    # ── 2. TRAIN MODELS PER TASK ──────────────────────────────────────────
    ideal_proc_col = [c for c in df_vol.columns if "ideal_proc" in c]
    models = {}
    for task in df_vol["task_name"].unique():
        df_task = df_vol[df_vol["task_name"] == task]
        train_data = (
            client_features.join(df_pms).join(df_cb)
            .join(df_task.groupby("client_name")[vol_target].sum())
            .fillna(0)
        )
        if len(train_data) <= 3:
            continue
        p_aht_col = [c for c in df_task.columns if "proc_aht" in c or ("final" in c and "proc" in c)]
        r_aht_col = [c for c in df_task.columns if "rev_aht" in c or ("final" in c and "rev" in c)]
        if ideal_proc_col and not df_task[ideal_proc_col[0]].dropna().empty:
            proc_role = df_task[ideal_proc_col[0]].mode().iloc[0]
        else:
            proc_role = "Accountant I"
        models[task] = {
            "model_t0": RandomForestRegressor(100, random_state=42).fit(train_data[feat_t0], train_data[vol_target]),
            "model_t1": RandomForestRegressor(100, random_state=42).fit(train_data[feat_t1], train_data[vol_target]),
            "model_t2": RandomForestRegressor(100, random_state=42).fit(train_data[feat_t2], train_data[vol_target]),
            "model_t3": RandomForestRegressor(100, random_state=42).fit(train_data[feat_t3], train_data[vol_target]),
            "model_t4": RandomForestRegressor(100, random_state=42).fit(train_data[feat_t4], train_data[vol_target]),
            "proc_aht": df_task[p_aht_col[0]].mean() if p_aht_col else 15.0,
            "rev_aht": df_task[r_aht_col[0]].mean() if r_aht_col else 5.0,
            "proc_role": str(proc_role).strip(),
            "rev_role": _reviewer_role(proc_role),
        }
    if not models:
        log("  AI Prediction: no tasks had enough training data (need > 3 clients per task) — skipping.")
        return None
    log(f"  AI Prediction: trained {len(models)} task models from {len(client_features)} existing clients.")

    # ── 3. PREPARE NEW CLIENTS ────────────────────────────────────────────
    df_new = df_new.copy()
    default_gl = (pd.Timestamp.today() + pd.DateOffset(days=30)).strftime("%Y-%m-%d")
    if "go_live_date" not in df_new.columns:
        df_new["go_live_date"] = default_gl
    else:
        df_new["go_live_date"] = (
            pd.to_datetime(df_new["go_live_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna(default_gl)
        )
    if "pod" not in df_new.columns:
        df_new["pod"] = ""
    for col in ALL_NUM_COLS:
        if col not in df_new.columns:
            df_new[col] = 0.0
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0)
    for col in ALL_CAT_COLS:
        if col not in df_new.columns:
            df_new[col] = "Unknown"
        df_new[col] = df_new[col].astype(str).replace(["nan", "None", ""], "Unknown").fillna("Unknown")

    new_pms = pd.DataFrame(pms_enc.transform(df_new[["pms"]]), columns=pms_cols, index=df_new.index)
    new_cb = pd.DataFrame(cb_enc.transform(df_new[["corp_books"]]), columns=cb_cols, index=df_new.index)
    df_new_enc = df_new.join(new_pms).join(new_cb)

    # ── 4. PREDICT ────────────────────────────────────────────────────────
    hrs_fte_month = st.session_state.calc_data["dict_hrs_per_fte"][ai_month_idx]
    network_days_m = st.session_state.calc_data["dict_workable_days"][ai_month_idx]
    vol_scale = network_days_m / BASELINE_NETWORK_DAYS

    rows = []
    for idx, row in df_new.iterrows():
        client_name = row.get("company_name", f"Client_{idx}")
        record_id = row.get("record_id", "")
        go_live = row["go_live_date"]
        mrr_val = row.get("mrr", 0)
        pod_val = row.get("pod", "")

        pms_val = str(row.get("pms", "Unknown")).strip()
        has_pms = pms_val.lower() not in ["unknown", "", "nan", "none"]
        corp_val = str(row.get("corp_books", "")).lower()
        has_t4 = row.get("sqft_commercial", 0) > 0 or corp_val not in ["unknown", "0", "", "nan", "no"]
        has_t3 = row.get("commercial_properties", 0) > 0 or row.get("commercial_doors", 0) > 0
        has_t2 = row.get("res_doors", 0) > 0 or row.get("res_prop", 0) > 0

        if has_t4 and has_pms:   mkey, fcols = "model_t4", feat_t4
        elif has_t3 and has_pms: mkey, fcols = "model_t3", feat_t3
        elif has_t2 and has_pms: mkey, fcols = "model_t2", feat_t2
        elif has_pms:            mkey, fcols = "model_t1", feat_t1
        else:                    mkey, fcols = "model_t0", feat_t0

        feats = df_new_enc.loc[idx, fcols].values.reshape(1, -1)

        _gl_parsed = pd.to_datetime(go_live, errors="coerce")
        _mes_start = (today + relativedelta(months=ai_month_idx)).replace(day=1)
        if pd.notna(_gl_parsed):
            _m_diff = (_mes_start.year - _gl_parsed.year) * 12 + (_mes_start.month - _gl_parsed.month)
            _gl_eom_p = (_gl_parsed + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).date()
            _wd_m0_p = int(np.busday_count(str(_gl_parsed.date()), str(_gl_eom_p)))
            _sm0_p = _wd_m0_p < 15
            if _m_diff == 0: _lc = 1.17
            elif _m_diff == 1: _lc = 1.17 if _sm0_p else 1.03
            elif _m_diff == 2: _lc = 1.03 if _sm0_p else 1.02
            elif _m_diff == 3: _lc = 1.02 if _sm0_p else 1.0
            else: _lc = 1.0
        else:
            _lc = 1.17

        for task, md in models.items():
            pred_vol = md[mkey].predict(feats)[0] * vol_scale
            if pred_vol <= 0.1:
                continue
            p_hrs = (pred_vol * md["proc_aht"] * _lc) / 60
            r_hrs = (pred_vol * md["rev_aht"] * _lc) / 60
            t_parts = str(task).split(" - ", 1) if " - " in str(task) else [str(task), ""]
            v_type, v_subtype = t_parts[0].strip(), t_parts[1].strip()
            util_p = utilization_map.get(md["proc_role"], util_acc1)
            util_r = utilization_map.get(md["rev_role"], util_sr)
            full_p_hrs = p_hrs * (1 + (1 - util_p) + absenteeism + attrition)
            full_r_hrs = r_hrs * (1 + (1 - util_r) + absenteeism + attrition)
            hourly_p = cost_map.get(md["proc_role"], cost_acc1) / hrs_fte_month if hrs_fte_month > 0 else 0
            hourly_r = cost_map.get(md["rev_role"], cost_sr) / hrs_fte_month if hrs_fte_month > 0 else 0
            rows.append({
                "_client": client_name, "_record_id": record_id, "_pod": pod_val,
                "_go_live": go_live, "_mrr": mrr_val, "_task": task,
                "_type": v_type, "_subtype": v_subtype,
                "_proc_role": md["proc_role"], "_rev_role": md["rev_role"],
                "_pred_vol": pred_vol, "_proc_aht": md["proc_aht"], "_rev_aht": md["rev_aht"],
                "_prod_p_hrs": p_hrs, "_prod_r_hrs": r_hrs,
                "_full_p_hrs": full_p_hrs, "_full_r_hrs": full_r_hrs,
                "_full_cost_p": full_p_hrs * hourly_p, "_full_cost_r": full_r_hrs * hourly_r,
            })

    if not rows:
        log("  AI Prediction: all predicted volumes were 0 — skipping.")
        return None
    df_pred = pd.DataFrame(rows)

    # ── 5. APPLY 45-55% MARGIN BAND (AHT scaled to nearest band edge) ────
    client_costs = df_pred.groupby("_client").agg(
        total_full_cost=("_full_cost_p", "sum"),
        total_full_cost_r=("_full_cost_r", "sum"),
        mrr=("_mrr", "first"),
    )
    client_costs["total_cost"] = client_costs["total_full_cost"] + client_costs["total_full_cost_r"]
    for c_name, cdata in client_costs.iterrows():
        c_mrr, m_cost = cdata["mrr"], cdata["total_cost"]
        if c_mrr <= 0 or m_cost <= 0:
            continue
        max_t, min_t = c_mrr * 0.55, c_mrr * 0.45
        adj = 1.0
        if m_cost > max_t: adj = max_t / m_cost
        elif m_cost < min_t: adj = min_t / m_cost
        if adj != 1.0:
            mask = df_pred["_client"] == c_name
            for col in ["_proc_aht", "_rev_aht"]:
                df_pred.loc[mask, col] *= adj
            df_pred.loc[mask, "_prod_p_hrs"] = (df_pred.loc[mask, "_pred_vol"] * df_pred.loc[mask, "_proc_aht"]) / 60
            df_pred.loc[mask, "_prod_r_hrs"] = (df_pred.loc[mask, "_pred_vol"] * df_pred.loc[mask, "_rev_aht"]) / 60
            util_p_vals = df_pred.loc[mask, "_proc_role"].map(lambda r: utilization_map.get(r, util_acc1))
            util_r_vals = df_pred.loc[mask, "_rev_role"].map(lambda r: utilization_map.get(r, util_sr))
            shrink_p = util_p_vals.map(lambda u: 1 + (1 - u) + absenteeism + attrition)
            shrink_r = util_r_vals.map(lambda u: 1 + (1 - u) + absenteeism + attrition)
            df_pred.loc[mask, "_full_p_hrs"] = df_pred.loc[mask, "_prod_p_hrs"] * shrink_p.values
            df_pred.loc[mask, "_full_r_hrs"] = df_pred.loc[mask, "_prod_r_hrs"] * shrink_r.values
            hp = df_pred.loc[mask, "_proc_role"].map(lambda r: cost_map.get(r, cost_acc1) / hrs_fte_month if hrs_fte_month > 0 else 0)
            hr = df_pred.loc[mask, "_rev_role"].map(lambda r: cost_map.get(r, cost_sr) / hrs_fte_month if hrs_fte_month > 0 else 0)
            df_pred.loc[mask, "_full_cost_p"] = df_pred.loc[mask, "_full_p_hrs"] * hp.values
            df_pred.loc[mask, "_full_cost_r"] = df_pred.loc[mask, "_full_r_hrs"] * hr.values

    log(f"  AI Prediction: {df_pred['_client'].nunique()} new client(s), "
        f"{len(df_pred)} task rows, "
        f"{(df_pred['_prod_p_hrs'].sum() + df_pred['_prod_r_hrs'].sum()):.1f} total productive hrs.")
    return df_pred


def build_synthetic_rows(df_pred: pd.DataFrame, existing_df_clean: pd.DataFrame) -> pd.DataFrame:
    """Turn `df_pred` (one row per client x task) into synthetic df_clean
    rows and merge them in, replacing any previous rows for the same client
    names (mirrors app.py:874-923)."""
    sr_lookup = {}
    if "Sr. Accountant" in existing_df_clean.columns and "client_name" in existing_df_clean.columns:
        sr_lookup = (
            existing_df_clean.dropna(subset=["client_name"])
            .groupby(existing_df_clean["client_name"].str.lower().str.strip())["Sr. Accountant"]
            .first().to_dict()
        )

    synth_rows = []
    for _, rp in df_pred.iterrows():
        cn = str(rp.get("_client", "")).strip()
        sr_val = sr_lookup.get(cn.lower().strip(), "")
        pod_synth = str(rp.get("_pod", "") or "").strip()
        if pod_synth.lower() in ("nan", "none", ""):
            pod_synth = ""
        synth_rows.append({
            "client_name": cn,
            "record_id": str(rp.get("_record_id", "") or ""),
            "POD": pod_synth,
            "Go Live": pd.to_datetime(rp.get("_go_live"), errors="coerce"),
            "Final Service Date": pd.NaT,
            "MRR": float(rp.get("_mrr", 0) or 0),
            "type": str(rp.get("_type", "") or ""),
            "subtype": str(rp.get("_subtype", "") or ""),
            "Closed tickets with Proc time": float(rp.get("_pred_vol", 0) or 0),
            "Closed tickets with rev time": float(rp.get("_pred_vol", 0) or 0),
            ">>> FINAL Capacity Proc AHT": float(rp.get("_proc_aht", 0) or 0),
            ">>> FINAL Capacity Rev AHT": float(rp.get("_rev_aht", 0) or 0),
            "Ideal Proc": str(rp.get("_proc_role", "Accountant I") or "Accountant I"),
            "Ideal Rev": str(rp.get("_rev_role", "Sr. Accountant") or "Sr. Accountant"),
            "Sr. Accountant": sr_val,
            "status": "client",
        })
    synth_df = pd.DataFrame(synth_rows)
    new_names_lower = set(synth_df["client_name"].str.lower().str.strip())

    clean_ex = existing_df_clean.copy()
    keep = ~clean_ex["client_name"].astype(str).str.lower().str.strip().isin(new_names_lower)
    return pd.concat([clean_ex[keep], synth_df], ignore_index=True)

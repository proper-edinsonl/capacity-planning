"""
HubSpot companies fetcher for the automated capacity report.

Pulls Companies from the HubSpot API and shapes them into a DataFrame with
the exact column names `_parse_hubspot_file()` (in app.py) already knows how
to read — so the rest of the capacity pipeline (which was built and tested
around a manual HubSpot export) needs zero changes.

Also applies the business rules the user specified for this automated run:
  - Only companies with a POD assigned are kept.
  - Go Live Date, if blank, is filled with (today + 30 days).
  - Final Service Date is only kept for Lifecycle Stage in {Churn, Pending
    Termination} — for every other stage (Client, Onboarding, On Notice,
    Retention, or anything else) it's blanked out, so a stale/incorrect FSD
    doesn't make an active client look like it's about to churn.
  - Two specific companies are always excluded (test/internal accounts).

Token storage: a local JSON file next to this script (`.hubspot_config.json`,
gitignored) — see `load_token()`/`save_token()`. Never committed, never sent
anywhere except HubSpot's own API.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

CONFIG_PATH = Path(__file__).with_name(".hubspot_config.json")

HUBSPOT_API_BASE = "https://api.hubapi.com"

# Internal HubSpot property name -> the column label _parse_hubspot_file expects.
PROPERTY_TO_COLUMN = {
    "name": "Company name",
    "pod": "POD",
    "lifecyclestage": "Lifecycle Stage",
    "retention_status": "Retention Status",
    "last_billed_cmrr": "Last Billed MRR",
    "original_contract_mrr": "Original CMRR",
    "go_live_date": "Go Live Date",
    "delivery_confirmed_go_live_date": "Delivery Confirmed Go-Live Date",
    "target_go_live_date": "Target Go-Live Date",
    "final_service_date": "Final Service Date",
    "property_management_software": "PMS",
}

# Company names to always exclude (test/internal accounts, not real clients).
EXCLUDED_COMPANY_NAMES = {
    "eng test client",
    "proper technologies inc",
    "proper technologies inc.",
    "proper technologies, inc.",
}

# Lifecycle Stage labels for which the Final Service Date should be honored.
FSD_HONORED_LIFECYCLE_STAGES = {"churn", "pending termination"}

# Placeholder/dummy company records that shadow a POD bucket instead of a
# real client (e.g. a company literally named "POD 6") — confirmed with the
# user as garbage records to always exclude, regardless of lifecycle stage
# or POD assignment.
_POD_PLACEHOLDER_RE = re.compile(r"^pod\s*\d+$", re.IGNORECASE)


def load_token() -> str | None:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            tok = str(data.get("token", "")).strip()
            return tok or None
        except Exception:
            return None
    return None


def save_token(token: str) -> None:
    CONFIG_PATH.write_text(json.dumps({"token": token.strip()}, indent=2), encoding="utf-8")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _fetch_property_options(token: str, log=print) -> dict:
    """Return {internal_property_name: {raw_value: label}} for enum-type
    properties whose stored value differs from its display label — most
    importantly 'lifecyclestage', where the API returns opaque numeric IDs
    for custom stages (e.g. '117973646') instead of 'Onboarding'."""
    resp = requests.get(
        f"{HUBSPOT_API_BASE}/crm/v3/properties/companies",
        headers=_headers(token), timeout=30,
    )
    resp.raise_for_status()
    props = {p["name"]: p for p in resp.json().get("results", [])}
    option_maps = {}
    for pname in ("lifecyclestage", "retention_status", "pod", "property_management_software"):
        p = props.get(pname)
        if not p:
            continue
        option_maps[pname] = {opt["value"]: opt.get("label", opt["value"]) for opt in p.get("options", [])}
    return option_maps


def _fetch_all_companies(token: str, log=print) -> list[dict]:
    """Server-side filtered fetch via the Search API — only companies that
    actually HAVE a POD assigned, instead of pulling the entire portal
    (leads/prospects included) and dropping ~99% of it client-side. This is
    the single biggest speed win: the portal has ~85k companies total but
    only a few hundred with a POD.

    Also excludes the two known test/internal accounts server-side (by exact
    name) as a first pass — `fetch_companies_dataframe()` still re-checks
    both rules client-side afterward as a safety net, in case a name filter
    doesn't match exactly (extra spacing, a period, etc.)."""
    properties = ["hs_object_id"] + list(PROPERTY_TO_COLUMN.keys())
    results: list[dict] = []
    after = None
    page = 0
    body_base = {
        "filterGroups": [{
            "filters": [
                {"propertyName": "pod", "operator": "HAS_PROPERTY"},
                {"propertyName": "name", "operator": "NEQ", "value": "Eng Test Client"},
                {"propertyName": "name", "operator": "NEQ", "value": "Proper Technologies Inc"},
            ]
        }],
        "properties": properties,
        "limit": 100,
    }
    while True:
        page += 1
        body = dict(body_base)
        if after:
            body["after"] = after
        resp = requests.post(
            f"{HUBSPOT_API_BASE}/crm/v3/objects/companies/search",
            headers={**_headers(token), "Content-Type": "application/json"},
            json=body, timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("results", [])
        results.extend(batch)
        log(f"  HubSpot: page {page} -> {len(batch)} companies (total so far: {len(results)})")
        after = payload.get("paging", {}).get("next", {}).get("after")
        if not after:
            break
        time.sleep(0.15)  # gentle on rate limits
    return results


def fetch_companies_dataframe(token: str, log=print) -> pd.DataFrame:
    """Fetch all Companies from HubSpot and return a DataFrame shaped exactly
    like what `_parse_hubspot_file()` expects from a manual export, with the
    user's business rules already applied (POD required, Go Live fallback,
    conditional Final Service Date, excluded test accounts)."""
    log("HubSpot: resolving property option labels (Lifecycle Stage, etc.)...")
    option_maps = _fetch_property_options(token, log)

    log("HubSpot: fetching companies...")
    raw_companies = _fetch_all_companies(token, log)
    log(f"HubSpot: {len(raw_companies)} companies fetched total.")

    rows = []
    for c in raw_companies:
        props = c.get("properties", {}) or {}
        row = {"Record ID": c.get("id", "")}
        for prop_name, column in PROPERTY_TO_COLUMN.items():
            val = props.get(prop_name)
            if val is not None and prop_name in option_maps:
                val = option_maps[prop_name].get(val, val)
            row[column] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        log("HubSpot: no companies returned — nothing to process.")
        return df

    n_total = len(df)

    # ── Exclude test/internal accounts ──────────────────────────────────────
    name_norm = df["Company name"].astype(str).str.strip().str.lower()
    excl_mask = name_norm.isin(EXCLUDED_COMPANY_NAMES)
    if excl_mask.any():
        log(f"HubSpot: excluding {excl_mask.sum()} test/internal account(s): "
            f"{df.loc[excl_mask, 'Company name'].tolist()}")
    df = df[~excl_mask].copy()

    # ── Exclude placeholder/dummy records (company name IS a POD label,
    # e.g. "POD 6") — these are not real clients. ───────────────────────────
    placeholder_mask = df["Company name"].astype(str).str.strip().str.match(_POD_PLACEHOLDER_RE)
    if placeholder_mask.any():
        log(f"HubSpot: excluding {placeholder_mask.sum()} placeholder/dummy record(s) "
            f"(company name is a POD label): {df.loc[placeholder_mask, 'Company name'].tolist()}")
    df = df[~placeholder_mask].copy()

    # ── POD required — drop companies with no POD assigned ─────────────────
    pod_blank = df["POD"].isna() | (df["POD"].astype(str).str.strip() == "")
    if pod_blank.any():
        log(f"HubSpot: dropping {pod_blank.sum()} companies with no POD assigned.")
    df = df[~pod_blank].copy()

    # ── Go Live resolution: HubSpot carries THREE go-live-ish dates —
    # Go Live Date, Delivery Confirmed Go-Live Date, and Target Go-Live
    # Date. A company can have a real date sitting in any one of them while
    # the other two are blank or stale — checking only "Go Live Date" (the
    # old behavior) missed real dates and fell through to the +30-day
    # synthetic fallback even when the company clearly already has a known
    # go-live. Per the user: take the LATEST (max) of whichever of the 3
    # are populated, not a fixed priority order. ────────────────────────────
    gl_main    = pd.to_datetime(df["Go Live Date"], errors="coerce")
    gl_confirm = pd.to_datetime(df["Delivery Confirmed Go-Live Date"], errors="coerce")
    gl_target  = pd.to_datetime(df["Target Go-Live Date"], errors="coerce")
    gl_resolved = pd.concat([gl_main, gl_confirm, gl_target], axis=1).max(axis=1, skipna=True)

    today_ts = pd.Timestamp(datetime.now().date())
    fallback_gl = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    n_gl_filled = int(gl_resolved.isna().sum())
    if n_gl_filled:
        log(f"HubSpot: {n_gl_filled} companies have no date in any of the 3 Go-Live "
            f"fields (Go Live / Delivery Confirmed / Target) — filling with today+30 "
            f"days ({fallback_gl}).")

    # Companies whose date came from a REAL HubSpot field (not the synthetic
    # +30 fallback), and how many days old that real date is — a client
    # whose real go-live is already months in the past should have actual
    # Volume/AHT hours already; if the pipeline still treats them as
    # brand-new, that's a record-matching problem to flag, not a genuinely
    # new client to run through the day-1 ramp-up learning curve.
    df["_go_live_is_real"] = gl_resolved.notna()
    df["_go_live_age_days"] = (today_ts - gl_resolved).dt.days

    df["Go Live Date"] = gl_resolved.fillna(pd.Timestamp(fallback_gl)).dt.strftime("%Y-%m-%d")
    df["Delivery Confirmed Go-Live Date"] = gl_resolved.dt.strftime("%Y-%m-%d")

    # ── Last Billed MRR: HubSpot stores a literal "0" (not blank/null) for
    # clients that haven't been billed yet — e.g. Onboarding or newly-live
    # clients — which is NOT the same as "no last billed value". Treat 0 as
    # missing so downstream MRR resolution (_parse_hubspot_file's own
    # "Last Billed MRR" -> "Original CMRR" fallback) actually fires the way
    # the user's rule intends ("si no tiene last billed, usar Original/
    # Contracted CMRR"). Without this, a real .fillna(NaN-only) fallback
    # never triggers because 0 is a present, non-null value.
    lb_num = pd.to_numeric(df["Last Billed MRR"], errors="coerce")
    n_lb_zeroed = int((lb_num == 0).sum())
    if n_lb_zeroed:
        log(f"HubSpot: {n_lb_zeroed} companies show Last Billed MRR = 0 (not yet "
            f"billed) — falling back to Original CMRR for these.")
    df["Last Billed MRR"] = lb_num.replace(0, pd.NA)

    # ── Final Service Date: only honored for Churn / Pending Termination ───
    lifecycle_norm = df["Lifecycle Stage"].astype(str).str.strip().str.lower()
    fsd_keep_mask = lifecycle_norm.isin(FSD_HONORED_LIFECYCLE_STAGES)
    n_fsd_before = df["Final Service Date"].notna().sum()
    df.loc[~fsd_keep_mask, "Final Service Date"] = None
    n_fsd_after = df["Final Service Date"].notna().sum()
    log(f"HubSpot: Final Service Date kept for {n_fsd_after} companies "
        f"(Churn/Pending Termination) — blanked for {n_fsd_before - n_fsd_after} others.")

    log(f"HubSpot: {len(df)}/{n_total} companies eligible after all rules "
        f"(excluded test accounts + no-POD).")
    return df.reset_index(drop=True)

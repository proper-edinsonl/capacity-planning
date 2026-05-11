# Error Log — Capacity Online V3

## Prevention Rules
> Distilled lessons — read these first before every session.

1. **Email vs. name format mismatch:** Before joining a DataFrame column against any email-keyed dict, check the column's actual format. `df_clean['Sr. Accountant']` stores full names from the spreadsheet. Always build a `name → email` normalization dict from `hc_data['by_sr']` and apply it before `groupby` or `.get()` lookups. Never assume emails and names are interchangeable.

2. **Productive hours only — never use shrinkage columns:** Employee busy hours must come from `Prod Hrs Proc (M1)` / `Prod Hrs Rev (M1)`. Never use `Total Hrs Proc w/ Shrinkage` / `Total Hrs Rev w/ Shrinkage` for capacity utilization — shrinkage inflates hours and overstates workload.

3. **Don't wrap computation inside display guards:** If a flag (like `_no_active`) is meant to control *display* (e.g., Hrs Left = 0), do not let it also skip the underlying *computation* (busy hours). Compute first, then gate the display separately.

4. **Session state keys disappear — always guard and invalidate:** Never use `st.session_state['key']` with bare brackets in export/download blocks. Use `.get()` with fallbacks. When an upstream input changes (HubSpot upload, Step 1 re-run, scenario load), explicitly pop all derived keys (`final_dashboards`, `_s2_proceed`, etc.) so stale data doesn't survive the change.

5. **Active clients = FSD null or >= today:** When counting clients per Sr. or any per-Sr. metric, always filter `df_clean` by `Final Service Date is null OR >= today`. Churned clients inflate ratios. Never count from `final_dashboards['cliente']` alone — it's scoped to the cascade window, not current reality.

6. **Productive HC = explicit role list:** Use `isin({'Accountant I', 'Accountant II', 'General Accountant', 'Sr. Accountant'})` for headcount counts — never `.ne('Other')`. Managers map to 'Other' but that doesn't mean every non-'Other' role is productive capacity.

7. **Scope summary sections to active filters:** If a POD filter is active, don't render "Overall" aggregate summaries — they show data the user didn't ask for and contradict the filtered view. Always gate summary sections on filter state.

---

## Error Entries

### [2026-05-11] — Busy hours = 0 for all Alert Relocate employees

**Error:** In the Employee Level tab, every employee flagged as "Alert Relocate" showed 0 busy hours across all months, even when they had active client assignments.

**Root Cause:** The busy-hour accumulation loop was wrapped inside `if not _no_active:`. Employees with `_no_active = True` (in volume assignments but not in `_hc_active_set`) were skipped entirely — the loop never ran, so `_busy` stayed 0.0 for all their months.

**Fix Applied:** Removed the `if not _no_active:` guard from around the busy-hour loop in `app.py` ~line 7759. All employees now compute busy hours. The `_no_active` flag is still used *after* the loop to set `Hrs Left = 0` and `Util% = None` — it only gates display, not computation.

**Prevention Rule:** Don't wrap data computation inside display guards. Compute first, gate display separately.

---

### [2026-05-11] — Employee Level busy hours inflated (using shrinkage columns)

**Error:** Employee Level busy hours were higher than expected — matching "required" hours inflated for absenteeism/attrition rather than actual productive hours.

**Root Cause:** `app.py` ~line 7587 was reading `Total Hrs Proc w/ Shrinkage` and `Total Hrs Rev w/ Shrinkage`. These columns multiply productive hours by a shrinkage factor (> 1.0), overstating actual work time.

**Fix Applied:** Changed to `Prod Hrs Proc (M1)` and `Prod Hrs Rev (M1)` — pure productive hours with no shrinkage applied.

**Prevention Rule:** Busy hours = productive hours only. Never use shrinkage columns for utilization or capacity calculations.

---

### [2026-05-11] — Sr. Ratios sheet missing from Excel export

**Error:** "Download — Ideal Pairs" Excel export did not include the `Sr_Ratios` sheet, even after the tab was set up correctly.

**Root Cause:** The export read from `st.session_state['_s3_sr_ratios_df']`, which was only written when the user visited the Sr. Ratios tab during that session. If the user went straight to download, the key didn't exist and the sheet was silently skipped.

**Fix Applied:** Moved Sr. Ratios computation inline inside the export builder in `app.py`. The sheet is now always computed fresh at export time — no tab visit required.

**Prevention Rule:** Never let an export sheet depend on a session state key that requires a prior UI interaction. Compute export data inline or guarantee it's always populated before the export runs.

---

### [2026-05-11] — Sr. Ratios counting churned clients in "Clients Assigned"

**Error:** "Clients Assigned" in the Sr. Ratios tab included clients that had already ended service, inflating each Sr.'s client count.

**Root Cause:** The three places computing client count per Sr. (`app.py` ~lines 5820, 8167, 8447) read from `df_clean` with no filter (all clients including churned) or from `final_dashboards['cliente']` (cascade window only, not current churn state).

**Fix Applied:** Added FSD filter in all three places:
```python
_fsd = pd.to_datetime(_df.get('Final Service Date', ...), errors='coerce')
_active = _df[_fsd.isna() | (_fsd >= pd.Timestamp.today().normalize())]
```

**Prevention Rule:** Active clients = `Final Service Date` is null OR >= today. Apply this filter every time you count clients per Sr.

---

### [2026-05-11] — Sr. Ratios "Clients Assigned" = 0 for most Sr. Accountants

**Error:** "Clients Assigned" showed 0 for most Srs (e.g., Kevin Salazar with 9 DRs, Lucia Castillo with 13 DRs). A few Srs showed non-zero counts at random.

**Root Cause:** `df_clean['Sr. Accountant']` stores full names (e.g., `"Kevin Salazar"`). After `.lower().strip()`, the groupby produced a name-keyed dict. But the HC loop iterates with `by_sr_email` keys — lowercase emails (e.g., `"kevin.salazar@proper.ai"`). Every `.get(email_key, 0)` returned 0. Only Srs whose spreadsheet column already stored an email showed non-zero results.

**Fix Applied:** Built a `name → email` normalization dict from `hc_data['by_sr']` at all three locations (~lines 5820, 8171, 8450):
```python
_n2e = {str(n).lower().strip(): str(d.get('email','')).lower().strip()
        for n, d in (hc_data.get('by_sr', {}) or {}).items()}
_norm = sr_col.apply(lambda v: _n2e.get(v, v) if '@' not in v else v)
```
Applied before `groupby` so the resulting dict is always email-keyed.

**Prevention Rule:** When joining a DataFrame column against an email-keyed dict, normalize first using `hc_data['by_sr']` to build a name→email map. Never assume the column format matches the dict key format.

---

### [2026-05-11] — Actual HC in waterfall counting managers

**Error:** "Actual HC" in waterfall Steps 3 and 4 showed inflated headcount — Accounting Managers and Assistant Managers were included.

**Root Cause:** `app.py` ~line 1873 used `active_pods['Capacity Role'].ne('Other')` to count HC. Any role that didn't map to `'Other'` in `_HC_ROLE_MAP` was counted, including manager roles.

**Fix Applied:**
```python
total = int(active_pods['Capacity Role'].isin(_productive_roles).sum())
# _productive_roles = {'Accountant I', 'Accountant II', 'General Accountant', 'Sr. Accountant'}
```

**Prevention Rule:** Use `isin(_productive_roles)` for all HC counts — never `.ne('Other')`. Managers are not productive capacity.

---

### [2026-05-11] — Stale cascade data shown after HubSpot re-upload

**Error:** After uploading a new HubSpot file, the Step 3 cascade dashboard still showed results from the previous run.

**Root Cause:** `final_dashboards` and `_s2_proceed` were not invalidated on HubSpot upload. The step-by-step flow found these keys present and skipped recomputation.

**Fix Applied:** Added in the HubSpot upload handler at `app.py` ~line 3788:
```python
st.session_state.pop('final_dashboards', None)
st.session_state.pop('_s2_proceed', None)
```

**Prevention Rule:** When any upstream input changes, explicitly pop all derived session state keys. Don't rely on the user to re-run steps manually.

---

### [2026-05-11] — Overall Role Summary visible when POD filter is active

**Error:** In the Employee Level tab, the "Overall Role Summary" table appeared even when a specific POD filter was selected, showing aggregate data that contradicted the filtered view.

**Root Cause:** The section rendered unconditionally — no check for active POD filter (`_ov_cascade_pods`).

**Fix Applied:** Added guard at `app.py` ~line 7522:
```python
if not _el_gen.empty and not _ov_cascade_pods:
    st.markdown(f"#### Overall Role Summary — {_ov_scope_label}")
```

**Prevention Rule:** Gate "Overall" summary sections on filter state. If any scoping filter is active, hide summaries that aggregate beyond that scope.

---

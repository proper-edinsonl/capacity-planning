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

8. **One FTE convention per report:** All HC/FTE figures must use `capacity hours ÷ (7.5h × working days)` with the role's utilization/absenteeism uplift applied to task hours. Per-person capacities (AC1 134, GA 126, Sr = 7.5h×wd − ops rhythm) only cap individual assignments — never use them as the denominator for required HC.

9. **Hours ≠ headcount in shift models:** Invoice hours are the same whether people do 6h/day or 4h/day; only the number of dedicated people changes. Report hours and dedicated people in separate columns and state the scope (which roles are in/out) on every headline number.

10. **Streamlit cache keys:** Never name a cache-busting argument with a leading `_` — `@st.cache_data` excludes those from the key. Use `cache_stamp=(path, mtime, size)`.

11. **Scale volume by dates, not by hours:** Ticket volume per month comes from the client's active fraction (go-live / final service date). Hour ratios embed the learning curve and produce absurd growth.

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
### [2026-07-27] — Hybrid model: two conventions for the same "FTE" number

**Error:** The manager report showed "358.3 FTEs required today" in the header cards but "353.5 required" in the POD balance, and "384 on payroll" vs "372 on payroll" — with no explanation. Managers could not tell which number to quote.

**Root Cause:** Two different conventions coexisted. (a) The role tables divided *uplifted demand hours* by a nominal FTE (7.5h × working days = 157.5). (b) The POD/client tables divided *productive hours* by each role's real monthly capacity (AC1/AC2 134, GA 126). Since the actual uplift is ~1.37 and 157.5/134 = 1.175, the two lenses differed by ~50 FTEs. On top of that, the POD balance only counted pod × role pairs that hold ticket work, so the 12 above-Sr managers dropped out of the payroll side (384 → 372).

**Fix Applied:** One convention everywhere — `capacity hours ÷ (7.5h × working days)`, with the per-row `uplift` column (role/activity utilization × absenteeism) added in `build_core` and helpers `fte_div_of` / `capacity_hours` / `required_hc` in `hybrid_transition.py`. Per-person capacities now only cap individual assignments. Scope labels added to every headline card, and the net-balance card was scoped to AC1 → Sr (supervision excluded).

**Prevention Rule:** One definition of FTE per report. If a second lens is genuinely needed, label both explicitly and show the bridge between them — never let two conventions sit in the same page unlabeled.

---

### [2026-07-27] — Invoice block: counting people as if they were hours

**Error:** The role table said the AC1 invoice block needed 82 FTEs while the task detail computed 36.7 for the same invoices.

**Root Cause:** `role_scenarios` added `inv_fte` (people dedicated to the invoice shift — a Mixed-AP person occupies a whole slot with only 4h/day of invoices) directly into an FTE total built from hours. The task detail counted only the hours consumed.

**Fix Applied:** The FTE total is now pure hours (`remaining + new_inv_hrs) / fte_div`); the number of dedicated people is reported separately as `inv_people_{scenario}` along with `spare_inv_hrs_{scenario}`. Role table and task detail reconcile within 0.3%.

**Prevention Rule:** Hours and headcount are different units. A shift structure (6h vs 4h per day) changes *people*, not *hours* — report them in separate columns and never sum them.

---

### [2026-07-27] — Streamlit cache served stale data after overwriting an input file

**Error:** Re-running the model after saving a new version of a workbook at the same path returned the previous results.

**Root Cause:** `@st.cache_data` keys on the argument values (the path strings). A cache-busting stamp was added but named `_stamp_key` — Streamlit **excludes** any argument whose name starts with `_` from the cache key, so it had no effect.

**Fix Applied:** Renamed to `cache_stamp=_stamp(export, vol, hc)` (size + mtime of each file) in `load_data` and `build_core`. Verified by overwriting the same path twice and getting different results.

**Prevention Rule:** In Streamlit, never prefix a cache-key argument with `_`. Use `_` only for objects you deliberately want excluded (unhashable handles).

---

### [2026-07-27] — Volume scaled with the month-over-month hours ratio

**Error:** Projected invoice volume exploded (+58% company-wide); HOA West alone jumped from 1,718 to ~28,000 tickets.

**Root Cause:** Monthly scaling used `Client_FTEs_by_Month` hours ratios, which embed the learning curve — a client live from July 13 shows tiny June hours, so the July/June ratio was 16×.

**Fix Applied:** Volume now scales by the client's active fraction of each calendar month, computed from Go Live / Final Service Date (`active_fraction`).

**Prevention Rule:** Hours ratios are not volume ratios. Scale ticket volume by dates (ramp/churn), never by hour columns that include learning-curve or utilization effects.

---

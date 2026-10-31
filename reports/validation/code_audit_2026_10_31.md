# Code audit — second verifier, 2026-10-31

An independent second verifier (Codex) audited `main` after the 2026-10 package refactor (PRs #259–#263) and then re-checked the fixes. **All six findings (F1–F6) and the three follow-up gaps are fixed and merged** (#264, #265, #266). Every finding was reproduced before fixing; none was introduced by the refactor (all were present at `b5bac03`, before phase 1).

Reproduction: `reports/validation/code_audit_2026_10_31_checks.py` (results in the sibling `.json`) now asserts the **corrected** behaviour for every finding; the follow-up scenarios live in `tests/test_audit_2026_10_31.py` (23 tests, each of the follow-up tests fails on the pre-fix code).

| ID | Priority | Area | Fixed in | Numbers move? |
|:---|:---|:---|:---|:---|
| F1 | P1 | MR tier-2 H3→H6 reconstruction | #265 + follow-up #266 | MR period only (see impact) |
| F2 | P2 | Training-row completeness | #265 | No (on the validation data) |
| F3 | P2 | HRI tuning weights | #264 | HRI mode only |
| F4 | P2 | HRI + RI optimizer | #264 | — (combination refused) |
| F5 | P2 | Resimulation identity | #264 + follow-up #266 | No |
| F6 | P2 | `ci_upper` selection fallback | #264 + follow-up #266 | No |

## F1 — P1: MR tier-2 H3→H6 reconstruction did not reproduce the bin forecast

> **FIXED, 2026-10-31 (#265, completed in #266).** For tier-2 accounts (H3-mature, H6-immature) the curve is evaluated once per bin on the cohort's Σnum/Σden H3, `t = max(curve(H3_bin), H3_bin)`, and allocated as `H3_account × t / (oa-weighted mean account H3)`. Because the accounts' H6 exposure is pro-rated by `oa_amt`, the reconstructed bin rate equals `t` for any exposure mix.

Location: `src/analysis/mr_pipeline.py`, `_assign_tiered_risk` (tier 2).

**Trigger.** Hybrid MR with `power` curvature ≠ 1 (selected by `mr_extrapolation_method = "auto"`), or — for the follow-up — any method when the defaulting accounts' `oa_amt` is not proportional to their H3 exposure.

**Reproduction.** 100 equal-exposure loans, one H3 default, bin H3 0.10 %, main H6/H3 ratio 2: the bin curve gives 0.20 % but the per-account reconstruction gave 0.10 % (α = 0.5) and 20 % (α = 2). After #265 the nonlinear part was fixed, but with the defaulter's `oa_amt` ×10 its H3 exposure the reconstructed rate was still ×9.17 the forecast (×0.10 with ×0.1) — **for `linear` too**, i.e. this part predates the audit. The #265 test had used `oa_amt` proportional to the H3 pile, the very assumption that failed.

**Cause.** The curve was fitted on bin aggregates but applied to each account's sparse H3 (mostly 0 or a single default); averaging nonlinear transforms does not give the transform of the average. Separately, the per-account ratio was normalised on the H3 exposure while the H6 numerator is rebuilt on `oa_amt`-pro-rated exposure.

**Real-data impact** (`demanda_direct_out`, pre-ecom config, 5 segments; `main` vs fix): main period unchanged; MR "Actual" risk no_premium_ab +0.021 pp, no_premium_cd +0.008 pp, premium +0.002 pp, precon +0.004 pp, no_premium_ef unchanged; the re-optimised MR mask flips 15 cells in no_premium_cd (MR optimum production −0.32 %) and 6 in premium (−0.03 %). The direction is conservative. ecom/inftel and Apple extracts (more early defaults) have not been measured.

## F2 — P2: An auxiliary indicator's missingness changed the training population

> **FIXED, 2026-10-31 (#265).** Training requires only the model features and the selected target's outcome pair (`todu_30ever_h6` / `todu_amt_pile_h6` for b2; nothing extra for the sparse-by-design HRI pair). Other indicators are summed NaN-skipping.

Location: `src/model/inference_optimized.py`, `_prepare_pipeline_data`.

**Reproduction.** Three loans with complete H6, one default: adding the H3 pair to `indicators` dropped the defaulted loan (rows 3 → 2, H6 default numerator 10 → 0). **Real data:** 0 of 27,022 H6-valid booked loans lack H3, so no impact on the validation snapshot.

## F3 — P2: HRI hyperparameter tuning used b2 exposure weights

> **FIXED, 2026-10-31 (#264).** `tune_tree_models` / `tune_linear_models` pass `target_var` to `_get_regression_weights`, matching feature selection and the final fit (`h_den_h6` in HRI mode). Inactive for b2 runs.

## F4 — P2: HRI reject-inference trials differed from the final application

> **FIXED, 2026-10-31 (#264), by refusal.** `risk_indicator = "hri_h6"` with `run_ri_optimizer = true` fails config validation: the optimizer's trials solve and calibrate on the b2 basis without the HRI multiplier the final run applies (trial vs runtime rejected HRI numerator 50 vs 75 in the reproduction). Re-enable only once trials share the HRI path.

## F5 — P2: Resimulation accepted material config changes

> **FIXED, 2026-10-31 (#264, completed in #266).** `resimulation_meta.json` stores a fingerprint and snapshot of the **effective** validated config — CLI overrides such as `--baseline` applied, optional-column stripping not yet — and resimulation raises `ResimulationError`, naming the changed fields, for any difference outside the risk targets, `log_level`, `base_scenario_only` and `dq_allow_warnings`. Runs saved before the fix fall back to the old four-field hash, which now also refuses instead of warning.

**Reproduction.** Changing `date_fin_book_obs`, `reject_uplift_factor` or `directions` left the old hash unchanged. Follow-up: the first fix snapshotted the config before `--baseline` was applied, so a baseline run's artifacts matched an ordinary config.

## F6 — P2: An unattainable `ci_upper` selection still read as feasible

> **FIXED, 2026-10-31 (#264, completed in #266).** `_resolve_selection` returns the applied basis; the "Optimum selected" row persists `selection_basis_requested`, `selection_basis_applied` and `ci_upper_feasible`. An unmet margin raises a segment-report banner, appears in the consolidated comparison and in the workbook's "Recommendation & key risks" narrative; the three columns are shown in HTML whenever `ci_upper` was requested.

**Reproduction.** Two candidates with CI upper 1.10 % / 1.30 % against a 1.00 % target: neither qualifies, the point rule picks the 0.95 % / CI 1.30 % point and `Feasible = True` was the only persisted signal. **Known limitation (not changed):** the selection-aware diagnostics still describe point-rule re-selection even when `ci_upper` was requested.

## Verification

- `uv run python reports/validation/code_audit_2026_10_31_checks.py` — passes with corrected assertions for F1–F6.
- `uv run pytest tests/test_audit_2026_10_31.py` — 23 tests; the follow-up tests fail on the pre-fix code.
- Full suite at merge of #266: 1760 passed, 1 skipped; 0 import cycles (lazy imports included).
- Real-data comparisons: F3–F6 do not move numbers under the default configuration; F1/F2 as above.

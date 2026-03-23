# Enhancement Roadmap (Unresolved)

## High
### Bug
- `src/mr_pipeline.py` — `_assign_tiered_risk()` uses hard-coded maturity thresholds (Tier 1: `>=6`, Tier 2: `>=3`) rather than `PreprocessingSettings.mr_maturity_months`. Mitigation: wire `mr_maturity_months` into tiering logic and align it with bin-level maturity filtering (including `mr_maturity_months=0` behavior).
- `src/mr_pipeline.py` — MR hybrid “H3 floor enforcement” (`H6 must be >= H3`) may override the intended reliability hierarchy when `b2_mr_h3` exists but extrapolation reliability thresholds are not met. Mitigation: gate floor enforcement by the same reliability criterion used for H3 extrapolation.
- `src/mr_pipeline.py` — `process_mr_period()` recalibrates repesca risk after `run_optimization_pipeline()` by scaling `todu_30ever_h6_rep`. This can desynchronize “optimized MR decisions” from “reported MR risk surface.” Mitigation: either re-run optimization after recalibration or explicitly document that recalibration is report-only.
- `src/mr_pipeline.py` — repesca recalibration multiplies by `cal_factor.values` after merges; this is fragile if merge keys are non-unique or if row order changes. Mitigation: apply recalibration by re-merging `cal_factor` onto `data_summary_desagregado_mr` using `merge_keys` (no `.values` alignment assumptions).
- `src/mr_pipeline.py` — MR hybrid merge uses outer merges that can expand rows if merge keys are non-unique. Mitigation: add uniqueness checks and row-count assertions after merges.
- `src/reject_inference.py` — reject inference may apply the H6/H3 uplift multiplier to both numerators by default; if the observed H6/H3 ratio differs for rejected vs booked populations, this can propagate bias into H3→H6 extrapolation. Mitigation: validate the ratio stability assumption and/or consider `reject_apply_h3_multiplier=false` for sensitivity analyses.
- `src/optimization_utils.py` — legacy 2-variable monotonic cutoff enumeration is unbounded (`combinations_with_replacement`), risking memory exhaustion. Mitigation: enforce a hard cap and fallback to MILP sweep.
- `src/models.py` — 2-var feature construction differs from N-D (feature engineering changes with number of optimization variables), risking inconsistent model behavior. Mitigation: unify feature engineering across dimensionalities.
- `src/mr_pipeline.py` — H6/H3 ratio clipping to `[0.5, 5.0]` can still bias extrapolated risk. Mitigation: consider robust/percentile-based clipping or alternative calibration.

### Statistical & Methodological
- `src/preprocess_improved.py:_run_data_transformations` — bin-edge learning uses a pre-update booked mask (`status_name == BOOKED`) and then applies `update_status_and_reject_reason(...)`. This can create inconsistency between learned bin edges/cutoffs and downstream booked definition. Mitigation: learn edges using the same booked definition used later (recompute booked mask after relabeling).
- `src/pipeline/preprocessing.py` + `src/plots.py:_prepare_transformation_data` — `tasa_fin`/stress computed from `data_clean` using a last-`n_months` filter relative to max date in the passed dataframe, which may include months outside `[date_ini_book_obs, date_fin_book_obs]`. Mitigation: compute `tasa_fin`/stress on frames filtered to the configured observation window.

## Medium
### Bug / Correctness
- `src/optimization_utils.py` + `src/utils.py` — MILP risk budgeting guards global denominators in some paths, but NaN risk can still arise when the selected mask yields zero exposure denominators; Pareto dominance/scenario selection may not explicitly filter NaN-risk solutions. Mitigation: enforce an eps constraint on accepted exposure denominators and/or filter NaN-risk solutions before dominance + selection.
- `src/reject_inference.py` — Bayesian smoothing Beta prior for acceptance rates may be policy-outcome driven rather than Beta-distributed counts; `prior_strength` is arbitrary and can be stale across time. Mitigation: validate prior strength via backtesting or empirical Bayes.

### Statistical & Methodological
- `src/reject_inference.py` — acceptance rates are not time-weighted/decay-aware; underwriting policy drift can cause historical rates to mis-estimate selection bias for current cohorts. Mitigation: compute rates on recent windows or apply time-decay weights.
- `src/optimization_utils.py` — monotonicity constraints are hard rules on noisy point estimates from sparse/noisy bins, with no uncertainty bands. Mitigation: explore relaxed constraints with confidence bands/smoothing.
- `src/utils.py` — H3→H6 extrapolation curve fitting does not incorporate heteroscedasticity/variance structure in denominators; fitting is ratio-driven rather than variance-aware. Mitigation: consider inverse-variance style weighting.

### Testing Gaps
- `src/mr_pipeline.py` — add focused tests for MR hybrid reconstruction edge cases:
  - `mr_maturity_months != 6` and `mr_maturity_months=0`
  - bins where `b2_mr_h3` exists but `n_obs_mr_h3 < h3_min_obs`
  - repesca recalibration when merge keys are non-unique (row-order safety)
  - MR-only bins where hybrid selection leads to `model_fallback` and ensure NaN-risk handling in downstream metrics.

## Low
### Bug / Robustness
- `src/models.py` — `plt.show()` can block in headless/CI; guard or remove in favor of `savefig()`.


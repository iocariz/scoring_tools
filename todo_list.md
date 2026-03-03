# Enhancement Roadmap

## Recently Fixed (audit findings)

- ~~Settings mutation in batch mode~~ — `data_manager.py` now uses `model_copy()` instead of mutating `settings` in-place; `run_batch.py` uses `copy.deepcopy` instead of shallow `.copy()`
- ~~Hardcoded multiplier=7 in legacy/MR paths~~ — `kpi_of_fact_sol` and `calculate_metrics_from_cuts` now accept `multiplier` parameter, threaded from `settings.multiplier`
- ~~Isotonic monotonicity direction in reject inference~~ — `_enforce_multiplier_monotonicity` now accepts `inv_vars` and sets `increasing=False` for inverted variables; threaded through `apply_parceling_adjustment` → `apply_reject_inference` → `run_optimization_pipeline`

## Audit Findings — High Priority

### Sensitivity analysis: hardcoded risk_step, no n_bootstraps threading
- `src/pipeline/optimization.py` sensitivity phase uses `risk_step` from settings but the sensitivity bootstrap still uses a local default
- `compute_scenario_bootstrap_ci` may not respect `settings.n_bootstraps` in all code paths
- Verify that all bootstrap call sites use `settings.n_bootstraps`

### MILP monotonicity constraints may reject valid solutions for inverted variables
- `milp_solve_cutoffs` in `src/optimization_utils.py` enforces monotonicity via `cutoff[i] <= cutoff[i+1]` on var0
- If var0 is in `inv_vars`, the constraint direction should be reversed (`cutoff[i] >= cutoff[i+1]`)
- Currently, `inv_vars` only controls the acceptance filter (`>=` vs `<=`), not the MILP constraint direction
- This could cause the solver to produce suboptimal or infeasible results when var0 is inverted

### Bootstrap CI uses `replace=True` without stratification
- `bootstrap_confidence_interval` in `src/metrics.py` resamples uniformly with replacement
- For skewed Pareto frontiers, this can produce unstable CIs for tail solutions
- Consider stratified bootstrap or BCa (bias-corrected accelerated) intervals

### MR H6 extrapolation assumes linear scaling
- `calculate_todu_30ever_from_b2` in `src/utils.py` back-calculates `todu_30ever_h6` from `b2_ever_h3` using `multiplier_h6 / multiplier_h3` ratio
- This linear extrapolation may understate H6 risk when loss emergence is non-linear (convex)
- Consider adding a configurable extrapolation curve (linear/exponential/logistic)

### Reject inference parceling per-variable pass overwrites grid structure
- `_enforce_multiplier_monotonicity` (now direction-aware) still applies isotonic per variable independently
- Second pass overwrites multipliers set by the first pass (last-variable-wins)
- For 2D grids, should fit a bivariate isotonic regression or iterate until convergence

### Stressor applied as multiplicative factor without bounds
- `src/inference_optimized.py` applies `stressor` directly: `risk *= stressor`
- No validation that stressor is within a reasonable range (e.g., 0.5–3.0)
- Extreme values silently produce nonsensical risk predictions
- Add bounds validation in `PreprocessingSettings`

### `tasa_fin` applied after reject inference but not validated
- Financing rate `tasa_fin` is multiplied into indicators after RI adjustment
- No check that `tasa_fin > 0`, no warning if it exceeds typical ranges
- Could silently zero out all risk metrics if set to 0

## Audit Findings — Medium Priority

### No feature importance logging outside SHAP
- Model training logs loss metrics but not which features drive predictions
- For non-SHAP runs, add permutation importance or coefficient logging
- Useful for quick model interpretability without the SHAP overhead

### Global optimizer does not validate segment frontier consistency
- `src/global_optimizer.py` accepts segment frontiers as-is
- If one segment's frontier is dominated by another, the allocator may silently give it zero weight
- Add a diagnostic log when a segment's entire frontier falls outside the feasible region

### PSI/CSI thresholds are hardcoded constants
- `PSI_UNSTABLE_THRESHOLD`, `PSI_SHIFT_THRESHOLD`, `CSI_THRESHOLD` in `src/constants.py`
- Should be configurable via `config.toml` for domain-specific calibration
- Different portfolios may have different drift sensitivity requirements

### `_auto_populate_bins` silently promotes legacy config
- `src/config.py` auto-generates `BinConfig` entries from legacy `octroi_bins`/`efx_bins` fields
- No deprecation warning is emitted — users don't know they're using the legacy path
- Add `logger.warning` when auto-promotion occurs

### Optuna timeout not configurable
- `src/optuna_tuning.py` uses a hardcoded study timeout
- Should be exposed as `optuna_timeout` in `PreprocessingSettings`

### Report HTML lacks accessibility attributes
- `src/reporting.py` generates tables without `scope`, `aria-label`, or alt text
- Not critical but matters for regulatory audit compliance in some jurisdictions

### Dashboard error handling is UI-level only
- `dashboard.py` catches exceptions and displays `html.Div("Error: ...")`
- No logging of the actual exception — debugging requires reproducing in CLI
- Add `logger.exception` before returning the error div

### Trend analysis SPC uses fixed window=6
- `src/trends.py` rolling median/MAD uses `window=6` (months)
- Should be configurable; short-run portfolios may need `window=3`

### No data lineage tracking
- Pipeline outputs don't record which input files / config versions produced them
- Add a metadata sidecar (JSON) with input hashes, config snapshot, git SHA

## Tier 2 — High value, moderate effort

### Temporal cross-validation
- Replace random k-fold with walk-forward or expanding-window splits in `_select_model_type_cv` / `_select_feature_set_cv` (`src/inference_optimized.py`)
- Prevents temporal leakage and produces time-aware performance estimates
- The CV infrastructure already exists — main work is swapping the splitter and adding a config flag

### Rolling-window PSI
- Extend `src/stability.py` to compute PSI across monthly vintages (not just Main vs. MR)
- `compute_monthly_metrics` in `src/trends.py` already aggregates by month; wire it to `calculate_psi`
- Store the PSI time series and plot it in trends
- Turns drift detection from a binary flag into a trajectory for early warning

### Alert delivery (Slack/webhook)
- `src/alerts.py` writes JSON files only — alerts reach nobody
- Add a pluggable `AlertSink` interface with implementations: `JsonSink` (existing), `WebhookSink`, `SlackSink`, `EmailSink`
- Configure via `config.toml` `[alerts]` section
- Integrate SPC anomalies from `src/trends.py` into the `DriftAlert` system (currently only PSI alerts)

### Alert deduplication & escalation
- Track alert history across runs (append to a persistent JSON/SQLite store)
- Suppress re-alerting on the same drift within a configurable window
- Escalate moderate → critical if drift persists for N consecutive runs

## Tier 3 — High value, higher effort

### Adaptive Pareto sweep + warm-starting
- Replace the linear 50-point sweep in `trace_pareto_frontier` with adaptive bisection that concentrates points near frontier "knees"
- Pass previous MILP solution as warm-start hint to HiGHS for adjacent risk targets
- Expected to produce a better frontier in less total solve time

### Model-based reject inference
- Only parceling (acceptance-rate uplift) is implemented in `src/reject_inference.py`
- Add propensity-score reweighting as a second method (simplest principled approach)
- Optionally add augmentation (fit model on booked, predict rejects)
- Build a validation framework: if any "through-the-door" data exists, benchmark RI methods against it

### Champion/challenger framework
- Run two configs in parallel and generate a comparison report
- The pipeline is already parameterized by config — main work is the comparison output
- Highlight where frontiers diverge, which segments are affected, net production/risk delta
- High value for model governance and regulatory discussions

### Prediction intervals / uncertainty quantification
- Models produce point predictions — sparse bins get no explicit uncertainty
- Add conformal prediction intervals or quantile regression estimators
- Surface uncertainty in optimization (e.g., flag Pareto solutions that depend on unreliable bin predictions)

### Robust allocation under uncertainty
- Bootstrap CIs exist at segment level but are ignored during global allocation in `src/global_optimizer.py`
- Implement robust optimization: optimize worst-case production at the 5th-percentile frontier
- Alternatively: stochastic programming over the bootstrap samples

### Cross-period Pareto comparison
- After applying main-period cutoffs to MR data, check whether the main-period frontier is still Pareto-optimal
- If the MR data would produce a strictly dominating frontier, flag model drift

### Change-point detection in trends
- SPC with rolling median/MAD is the only method in `src/trends.py`
- Add CUSUM, EWMA, or Bayesian change-point detection (e.g., `ruptures` library)
- Add seasonal decomposition (STL) to reduce false SPC anomalies from calendar effects

### ~~Full N>2 variable support~~ (Done)
- ~~`fixed_cutoffs` is hard-blocked for 3+ variables~~ — N>2 uses `create_fixed_cutoff_mask` (per-variable accepted bin lists)
- ~~`mask_to_cutoffs` uses per-dimension projections (lossy) for N>2~~ — now returns `_cells` dict, `_marginal_*`, and conditional cutoffs for last dim
- ~~Legacy enumeration fallback is 2-var only~~ — N>2 uses GA fallback via `_ga_pareto_fallback`
- ~~`plot_3d_surface` is 2-var only~~ — gate relaxed to `len(variables) >= 2`, uses first 2 vars for surface
- ~~Complete the N-D generalization across all code paths~~

### Per-segment production floor in MILP
- `min_production` exists in global allocator's `SegmentConstraints` but not in the per-segment MILP
- Adding it would let a segment guarantee minimum lending volume independently of global allocation

### PDF report export
- Reports are HTML only (`src/reporting.py`)
- Add `weasyprint` or `playwright` PDF renderer for audit packages

### Dashboard write-back
- The Dash dashboard (`dashboard.py`) is read-only
- Allow users to adjust cutoffs in the cutoff explorer and write back to `config.toml` / `segments.toml`

### Vintage-based cohort analysis
- Pipeline processes a date range as a single block
- Add vintage-level analysis: cohort by origination month, track risk evolution per cohort

### Integration test with fixture dataset
- Tests use synthetic DataFrames — no end-to-end coverage
- Add a small versioned fixture dataset and an integration test running the full pipeline from `config.toml` → reports

### Benchmark suite
- No performance regression detection
- Add benchmarks for the MILP solver and full pipeline on a fixed synthetic dataset
- Track across commits via `pytest-benchmark` or `asv`

### Schema evolution / config migration
- `config.toml` has legacy fields auto-promoted via `_auto_populate_bins`
- Consider a versioned schema (`config_version = 2`) with explicit migration

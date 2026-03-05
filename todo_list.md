# Enhancement Roadmap

## Recently Fixed (audit findings)

- ~~Settings mutation in batch mode~~ — `data_manager.py` uses `model_copy()`; `run_batch.py` uses `copy.deepcopy`
- ~~Hardcoded multiplier=7 in legacy/MR paths~~ — `kpi_of_fact_sol` and `calculate_metrics_from_cuts` now accept `multiplier` parameter
- ~~Isotonic monotonicity direction in reject inference~~ — `_enforce_multiplier_monotonicity` now accepts `inv_vars` and sets correct direction
- ~~Sensitivity analysis defaults centralized~~ — `DEFAULT_N_BOOTSTRAPS` / `DEFAULT_SENSITIVITY_LEVELS` in `src/constants.py`; all call sites reference these
- ~~MILP monotonicity + inv_vars~~ — was already correct (`src/optimization_utils.py:110`); invalid finding
- ~~Bootstrap CI stratification~~ — was already correct; neither bootstrap resamples the frontier; invalid finding
- ~~Full N>2 variable support~~ — N-D fixed cutoffs, GA fallback, conditional reporting all implemented
- ~~MR H6 extrapolation assumes linear scaling~~ — `extrapolate_h3_to_h6()` in `src/utils.py` supports linear/power/logistic curves; configured via `mr_extrapolation_method` / `mr_extrapolation_curvature` in `PreprocessingSettings`
- ~~Auto-calibrate H3→H6 curvature~~ — `fit_h3_extrapolation_curve()` in `src/utils.py` fits curvature via weighted log-log regression; `mr_extrapolation_method = "auto"` resolves to linear or power based on main-period data; diagnostics in `mr_risk_comparison_*.csv`

## Audit Findings — High Priority

### Reject inference parceling per-variable overwrites grid structure
- `_enforce_multiplier_monotonicity` at `src/reject_inference.py:182-191` applies isotonic per variable independently
- Second pass overwrites multipliers set by the first pass (last-variable-wins)
- For 2D grids, should fit a bivariate isotonic regression or iterate until convergence

### Stressor applied as multiplicative factor without bounds
- `src/models.py:269` applies `stressor` directly: `risk *= stressor`
- Also: `calculate_stress_factor` (`src/utils.py:160`) returns 0.0 on empty data, which zeroes all predictions
- No validation that stressor is within a reasonable range (e.g., 0.5–3.0)
- Add bounds validation in `PreprocessingSettings`

### `tasa_fin` applied after reject inference but not validated
- Financing rate `tasa_fin` is multiplied into indicators at `src/inference_optimized.py:1669`
- No check that `tasa_fin > 0`, no warning if it exceeds typical ranges
- Could silently zero out all risk metrics if set to 0

### MR pipeline unchecked `.iloc[0]`
- `src/mr_pipeline.py:70` assumes `optimal_solution_df` is non-empty
- `IndexError` if optimization produced no solutions (empty Pareto frontier)
- Add a guard or raise a descriptive error before the `.iloc[0]` access

## Audit Findings — Medium Priority

### No feature importance logging outside SHAP
- Model training logs loss metrics but not which features drive predictions
- For non-SHAP runs, add permutation importance or coefficient logging at `src/inference_optimized.py`

### Global optimizer does not validate segment frontier consistency
- `src/global_optimizer.py` accepts segment frontiers as-is
- If one segment's frontier is dominated, the allocator may silently give it zero weight
- Add a diagnostic log when a segment's entire frontier falls outside the feasible region

### PSI/CSI thresholds hardcoded in plotting code
- Constants `PSI_UNSTABLE_THRESHOLD`, `PSI_SHIFT_THRESHOLD`, `CSI_THRESHOLD` in `src/constants.py` are used correctly in logic
- However, **plotting code** at `src/stability.py:383,405-406` duplicates them as hardcoded literals instead of referencing the constants
- Should also be configurable via `config.toml` for domain-specific calibration

### `_auto_populate_bins` silently promotes legacy config
- `src/config.py` auto-generates `BinConfig` entries from legacy `octroi_bins`/`efx_bins` fields
- No deprecation warning is emitted — users don't know they're using the legacy path
- Add `logger.warning` when auto-promotion occurs

### Optuna timeout not configurable
- `src/optuna_tuning.py` uses a hardcoded study timeout
- Should be exposed as `optuna_timeout` in `PreprocessingSettings`

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

### No `test_data_manager.py`
- `src/data_manager.py` has only `DataValidationError` tested via imports; no dedicated tests
- Add tests for SAS file loading, column standardization, and H3 column handling

### MR `calculate_todu_30ever_from_b2` ignores `multiplier_h3`
- `src/mr_pipeline.py:664` calls with default `multiplier=7` even when input was extrapolated from H3 using `multiplier_h3=4`
- Risk metric back-calculation uses wrong scaling factor for H3-extrapolated outcomes

## Tier 2 — High value, moderate effort

### Temporal cross-validation
- Replace random k-fold with walk-forward or expanding-window splits in `_select_model_type_cv` / `_select_feature_set_cv` (`src/inference_optimized.py`)
- Prevents temporal leakage and produces time-aware performance estimates

### Rolling-window PSI
- Extend `src/stability.py` to compute PSI across monthly vintages (not just Main vs. MR)
- `compute_monthly_metrics` in `src/trends.py` already aggregates by month; wire it to `calculate_psi`
- Turns drift detection from a binary flag into a trajectory for early warning

### Alert delivery (Slack/webhook)
- `src/alerts.py` writes JSON files only — alerts reach nobody
- Add a pluggable `AlertSink` interface: `JsonSink`, `WebhookSink`, `SlackSink`, `EmailSink`
- Integrate SPC anomalies from `src/trends.py` into the `DriftAlert` system

### Alert deduplication & escalation
- Track alert history across runs (append to a persistent JSON/SQLite store)
- Suppress re-alerting on the same drift within a configurable window
- Escalate moderate → critical if drift persists for N consecutive runs

### Fairness / disparate impact monitoring
- Optional demographic parity and equalized odds metrics per cutoff solution
- Supports fair lending compliance (ECOA / HMDA reporting requirements)
- Add as opt-in module gated by `[fairness]` config section

### Cutoff robustness analysis
- Jack-knife leave-one-fold-out retraining to measure cutoff stability across data perturbations
- Flag Pareto solutions whose cutoffs shift significantly under resampling
- Integrates with existing bootstrap infrastructure in `src/utils.py`

## Tier 3 — High value, higher effort

### Adaptive Pareto sweep + warm-starting
- Replace the linear 50-point sweep in `trace_pareto_frontier` with adaptive bisection near frontier "knees"
- Pass previous MILP solution as warm-start hint to HiGHS for adjacent risk targets

### Model-based reject inference
- Add propensity-score reweighting as a second method alongside parceling in `src/reject_inference.py`
- Optionally add augmentation (fit model on booked, predict rejects)
- Build a validation framework: benchmark RI methods against "through-the-door" data if available

### Champion/challenger framework
- Run two configs in parallel and generate a comparison report
- Highlight where frontiers diverge, which segments are affected, net production/risk delta

### Prediction intervals / uncertainty quantification
- Add conformal prediction intervals or quantile regression estimators
- Surface uncertainty in optimization (flag Pareto solutions that depend on unreliable bin predictions)

### Robust allocation under uncertainty
- Bootstrap CIs exist at segment level but are ignored during global allocation in `src/global_optimizer.py`
- Implement robust optimization: optimize worst-case production at the 5th-percentile frontier

### Cross-period Pareto comparison
- After applying main-period cutoffs to MR data, check whether the main-period frontier is still Pareto-optimal
- If the MR data would produce a strictly dominating frontier, flag model drift

### Change-point detection in trends
- Add CUSUM, EWMA, or Bayesian change-point detection (e.g., `ruptures` library) to `src/trends.py`
- Add seasonal decomposition (STL) to reduce false SPC anomalies from calendar effects

### Per-segment production floor in MILP
- `min_production` exists in global allocator's `SegmentConstraints` but not in the per-segment MILP
- Adding it would let a segment guarantee minimum lending volume independently of global allocation

### PDF report export
- Reports are HTML only (`src/reporting.py`)
- Add `weasyprint` or `playwright` PDF renderer for audit packages

### Dashboard write-back
- The Dash dashboard (`dashboard.py`) is read-only
- Allow users to adjust cutoffs and write back to `config.toml` / `segments.toml`

### Vintage-based cohort analysis
- Add vintage-level analysis: cohort by origination month, track risk evolution per cohort

### Integration test with fixture dataset
- Add a small versioned fixture dataset and an integration test running the full pipeline from `config.toml` → reports

### Benchmark suite
- Add benchmarks for the MILP solver and full pipeline on a fixed synthetic dataset
- Track across commits via `pytest-benchmark` or `asv`

### Schema evolution / config migration
- `config.toml` has legacy fields auto-promoted via `_auto_populate_bins`
- Consider a versioned schema (`config_version = 2`) with explicit migration

### Report HTML accessibility
- `src/reporting.py` generates tables without `scope`, `aria-label`, or alt text
- Semantic HTML already present; issue is minor ARIA gaps
- Low priority but relevant for regulatory audit compliance in some jurisdictions

### Model ensemble with Bayesian averaging
- Ensemble top-K models via Bayesian Model Averaging instead of single-best selection in `src/inference_optimized.py`
- Reduces model selection variance, especially with small training sets

### Cohort-level (vintage) model validation
- Backtest at origination-month level for early warning on recent cohorts
- Complements vintage-based cohort analysis with model performance tracking per vintage

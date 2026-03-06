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
- ~~StratifiedKFold for imbalanced CV~~ — replaced `KFold` with `StratifiedKFold` (binarized risk indicator) in `optuna_tuning.py` and `inference_optimized.py`; falls back to `KFold` when stratification not possible
- ~~PSI epsilon handling distorts divergence~~ — epsilon now applied only inside `log(p/q)`, not by re-normalizing distributions; fixed in `stability.py`, `metrics.py` (PSI + IV); also replaced hardcoded thresholds in plotting with `PSI_STABLE_THRESHOLD`/`PSI_UNSTABLE_THRESHOLD` constants
- ~~Silent data loss from unlogged `dropna()`~~ — added row-count logging before each `dropna()` in `preprocess_improved.py` (5 locations: `learn_quantile_bins`, `learn_income_bins`, `learn_optimization_bins`, `assess_binning_gini`, `_apply_binning_from_config`)
- ~~`calculate_stress_factor` returns 0.0 on empty data~~ — changed to return neutral `1.0` to prevent zeroing all risk predictions; also handles zero `overall_bad_rate` case
- ~~Reject inference denominator warning~~ — enhanced warning in `reject_inference.py:compute_acceptance_rates` to log non-score rejection share of all rejections; escalates to `WARNING` when >5% of demand, adds actionable guidance when >10%

---

## Audit Findings — High Priority

### Reject inference parceling per-variable overwrites grid structure
- `_enforce_multiplier_monotonicity` at `src/reject_inference.py:182-191` applies isotonic per variable independently
- Second pass overwrites multipliers set by the first pass (last-variable-wins)
- For 2D grids, should fit a bivariate isotonic regression or iterate until convergence

### Stressor applied as multiplicative factor without bounds
- `src/models.py:269` applies `stressor` directly: `risk *= stressor`
- ~~`calculate_stress_factor` returns `0.0` on empty data~~ — fixed: now returns `1.0`
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

### Missing column existence checks in optimization
- `src/optimization_utils.py:429-431` accesses `todu_30ever_h6`, `todu_amt_pile_h6` without checking existence — will raise cryptic `KeyError`
- `src/optimization_utils.py:735-737` same unchecked access in GA fallback path
- Add explicit guards or raise descriptive errors before column access

---

## Audit Findings — Medium Priority

### No feature importance logging outside SHAP
- Model training logs loss metrics but not which features drive predictions
- For non-SHAP runs, add permutation importance or coefficient logging at `src/inference_optimized.py`

### Global optimizer does not validate segment frontier consistency
- `src/global_optimizer.py` accepts segment frontiers as-is
- If one segment's frontier is dominated, the allocator may silently give it zero weight
- Add a diagnostic log when a segment's entire frontier falls outside the feasible region

### PSI/CSI thresholds should be configurable
- ~~Hardcoded thresholds in plotting code~~ — fixed: `stability.py` now uses `PSI_STABLE_THRESHOLD`/`PSI_UNSTABLE_THRESHOLD` constants
- Should also be configurable via `config.toml` for domain-specific calibration

### `_auto_populate_bins` silently promotes legacy config
- `src/config.py:368-393` auto-generates `BinConfig` from legacy `octroi_bins`/`efx_bins` fields
- No deprecation warning is emitted — users don't know they're using the legacy path
- Add `logger.warning` when auto-promotion occurs

### Optuna timeout not configurable
- `src/optuna_tuning.py` uses a hardcoded study timeout
- Should be exposed as `optuna_timeout` in `PreprocessingSettings`

### Dashboard error handling is UI-level only
- `dashboard.py` catches exceptions and displays `html.Div("Error: ...")`
- No logging of the actual exception — debugging requires reproducing in CLI
- Add `logger.exception` before returning the error div
- Line 226: `except Exception` swallows tomllib errors without logging

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

### GA fallback missing division-by-zero guard
- `src/optimization_utils.py:751` computes risk coefficients but doesn't guard against all-zero `todu_amt`
- Main MILP path (line 220-222) has this guard; GA fallback does not
- Risk constraint becomes vacuous when denominator is zero

### `mr_extrapolation_curvature` not validated against method
- `src/config.py:253` — `mr_extrapolation_curvature` has bounds `gt=0, le=5.0`, but when `mr_extrapolation_method = "auto"` the value is ignored
- Add a model validator to warn when both `method="auto"` and a non-default curvature are set

### Empty except blocks swallow errors silently
- `src/consolidation.py:1574,1581` — `except (TypeError, ValueError): pass` when sorting pivot indices
- Should log at DEBUG level for visibility: `logger.debug(f"Could not sort pivot: {e}")`

### Broad `except Exception` in Optuna trial evaluation
- `src/optuna_tuning.py:248-250,255-257` — catches all exceptions at DEBUG level
- Could mask configuration errors or OOM; use more specific exception types

---

## Statistical & Methodological — Critical

### Non-nested cross-validation for hyperparameter tuning (optimistic bias)
- `src/optuna_tuning.py:120-124,155-156` — Optuna tunes hyperparameters on full dataset, then CV evaluates on same dataset with different seed
- Code acknowledges this: "R² may be slightly optimistic due to hyperparameter selection on the same data"
- Violates cardinal principle that hyperparameter selection must use different data than evaluation
- Implement proper nested CV: outer loop for performance estimation, inner loop for Optuna tuning

### Reject inference denominator excludes non-score rejections
- `src/reject_inference.py:60-108` — acceptance rate = `n_booked / (n_booked + n_score_rejected)`
- Non-score rejections (manual review, fraud, missing docs) excluded from denominator
- Inflates acceptance rate, leading to understated reject multipliers
- Bins with high non-score rejection rates will underestimate reject population risk
- If non-score rejections are >5-10% of total, bias becomes material

### Bin edges learned on full training data without holdout
- `src/preprocess_improved.py:330-420` — `learn_optimization_bins()` fits DecisionTreeRegressor on full booked population
- Same data used for bin learning and downstream model training/evaluation
- Biases bin edges toward training distribution; no validation-fold assessment of bin stability
- Use stratified train/validation split (80/20); learn edges on train fold, validate stability on holdout

---

## Statistical & Methodological — High Priority

### H3→H6 extrapolation ratio assumes stability across cohorts
- `src/mr_pipeline.py:344-360` — `h6_h3_ratio = b2_main / b2_main_h3`, clipped to [0.5, 5.0]
- Assumes ratio is stable between main and MR cohorts — violated when:
  - MR period has different risk composition (newer vintages, policy changes)
  - Economic conditions shifted (unemployment, interest rates)
  - Selection changed (different cutoffs in MR vs main period)
- Hard clipping masks the bias but doesn't correct it
- Add cohort-level ratio stability test; flag when MR ratio diverges >20% from main-period ratio

### No autocorrelation adjustment in SPC trend detection
- `src/trends.py:228-241` — SPC bounds computed from rolling statistics without accounting for temporal autocorrelation
- Financial metrics (approval rates, risk) are typically autocorrelated (rho ~ 0.5-0.7)
- If rho=0.7, effective sample size in 6-month window is ~2, not 6 — CIs are too narrow
- Estimate autocorrelation and inflate SE: `SE_adj = SE / sqrt(1 - rho)`
- Without correction, anomaly detection has high false-positive rate

### Bootstrap CI uses simple percentile without bias correction
- `src/utils.py:425-433` and `src/metrics.py:101-102` — uses `np.percentile(boot_vals, alpha/2)`
- No BCa (bias-corrected and accelerated) correction
- For skewed distributions (Gini, KS on imbalanced data), percentile CIs are anticonservative
- Reported 95% CIs may have actual coverage <90%
- Implement BCa: compute bias `z0 = norm.ppf(mean(boot < point_est))`, acceleration via jackknife

### TweedieGLM uses log(exposure) as feature instead of proper offset
- `src/estimators.py:252-334` — adds `log(exposure)` as a column in `X_aug`
- In proper GLM, offset is NOT subject to regularization; as a feature it gets penalized by `alpha`
- Biases exposure coefficient away from 1.0, causing systematic rate prediction errors at extrapolated exposures
- Use `sample_weight` parameter to implement offset, or exclude from regularization

### Vintage imbalance unweighted in bin learning
- `run_batch.py:99-154` and `src/preprocess_improved.py:330-420` — data pooled across observation period without time-stratification
- Newer cohorts with less performance history treated equally as mature vintages
- No vintage-specific weights or cohort-age adjustments
- Implement cohort-age weights in `learn_optimization_bins()`: scale by months-since-origination

---

## Statistical & Methodological — Medium Priority

### One-SE rule doesn't account for CV fold correlation
- `src/inference_optimized.py:70-106` — threshold = `best_mean + best_se` assumes independent folds
- CV folds from same split are correlated; effective degrees of freedom < k
- Threshold is overly generous, selecting overly complex models
- Ref: Breiman et al. (1984); adjust margin for fold correlation

### Bayesian smoothing Beta prior not justified for acceptance rates
- `src/reject_inference.py:76-85` — uses Beta-Binomial with `prior_strength=10`
- Acceptance rates are policy outcomes (cutoff-based), not Beta-distributed phenomena
- Prior strength is arbitrary; no theoretical justification for pseudo-observation count
- Global rate prior may be stale if risk distribution changed over time
- Consider empirical Bayes or cross-validated prior strength

### Acceptance rates not time-weighted for temporal drift
- `src/reject_inference.py` — rates computed on full historical period without time-weighting
- If lending standards changed (tighter cutoffs), historical rates understate current selection bias
- Leads to insufficient risk adjustment for reject population
- Add time-decay weighting or compute rates on recent window only

### Monotonicity constraints may over-constrain via noisy risk estimates
- `src/optimization_utils.py:110-174` — enforces `x[riskier_cell] <= x[safer_cell]`
- Risk estimates in sparse bins are noisy; a "riskier" cell may have upward-biased estimate
- No confidence interval used — all estimates treated as point estimates
- Pareto frontier may be too conservative; consider relaxed constraints with confidence bands

### Sensitivity perturbation doesn't maintain variable correlation
- `src/sensitivity.py:19-45` — perturbs risk independently from production
- Risk and production are correlated (larger exposures have more defaults in absolute terms)
- Creates artificial scenarios that violate historical correlation structure
- Implement multivariate perturbations maintaining historical risk-production co-movement

### H3→H6 log-log regression not weighted by heteroscedasticity
- `src/utils.py:120-179` — `np.polyfit(log_h3, log_h6, 1, w=w)` weights by observation count
- Does not account for heteroscedasticity: cells with more defaults have different variance
- Ratio H6/H3 has error proportional to `1/denominator`, not captured by count weights
- Use inverse-variance weighting based on `1 / (todu_amt_pile_h3 + epsilon)`

### No multiple-comparison correction in sensitivity analysis
- `src/sensitivity.py:48-134` — tests 6 perturbation levels (`[-20,-10,-5,5,10,20]`) independently
- No Bonferroni or FDR adjustment for multiple comparisons
- Type I error inflated when testing many cutoff-level combinations
- Also applies to PSI across multiple variables (`src/stability.py:108-114`)

### No MCAR/MAR/MNAR assessment for missing data
- `src/preprocess_improved.py:230+` — `dropna()` applied with implicit MCAR assumption
- Credit data missingness is likely MNAR: missing scores indicate incomplete applications, missing income indicates business customers
- Complete-case analysis is biased under MNAR
- Document missing data assumption explicitly; consider sensitivity analysis under MAR

### No VIF or collinearity diagnostics despite acknowledged multicollinearity
- `src/models.py:150-156` — code comment acknowledges polynomial features on bin indices create high VIF
- No VIF computation; relies entirely on Ridge/Lasso regularization as mitigant
- PolynomialFeatures applied without standardization — raw bin indices (1..N)
- Add VIF computation before model training; set threshold (VIF < 10); standardize features

### Quantile binning handles tied values poorly
- `src/preprocess_improved.py:236` — `pd.Series.quantile(quantiles).unique()` collapses ties
- When many records fall at same value (common in credit scores), quantile bins become degenerate
- Example: if median=100 and 40% of data has value 100, bin boundary is meaningless
- No warning about minimum bin population; add minimum-population check after binning

### MR `min_obs` threshold has no statistical justification
- `src/mr_pipeline.py:338,348,355` — default threshold of 30 observations for MR inclusion
- Threshold is arbitrary; doesn't account for event count, relative bin size, or required precision
- A bin with 30 obs but 0 events gives zero risk with high uncertainty
- Use effective sample size: consider event count and required SE precision (e.g., SE < 0.5%)

### Parceling calibration gamma assumes linear selection model
- `src/reject_inference_optimizer.py` — formula: `true_risk = observed_risk / acceptance_rate^gamma`
- Derived from threshold selection model assuming gamma=1 (linear)
- Credit scoring is nonlinear (log-odds, tree-based); relationship between acceptance rate and risk extrapolation is not linear
- Gamma may systematically over/under-correct depending on true model curvature
- Add diagnostic: compare gamma-adjusted risk to through-the-door data when available

### Marginal impact ignores risk constraint binding
- `src/sensitivity.py:204-277` — computes `delta_risk = new_risk - base_risk` per cell flip
- Doesn't check whether `new_risk <= target_risk`; if current solution is at constraint boundary, cell flip may be infeasible
- Reported marginal impact may be unachievable within the risk budget

### b2_ever_h6 clipping at zero may mask data quality issues
- `src/utils.py:24-62` — `np.clip(result, 0, None)` silently converts negative risk to zero
- Negative risk arises from data entry errors, misaligned definitions, or selection bias
- Silently treating as zero-risk cells leads to overly aggressive acceptance in those cells
- Add logging when negative values are clipped

---

## Performance — High Impact

### O(n^2) Pareto dominance check
- `src/optimization_utils.py:498-510` — nested loop checking dominance between all solution pairs
- For 1000 solutions: ~500K comparisons
- Replace with vectorized numpy dominance filter or sort-and-scan algorithm

### Double nested loop in Excel export
- `src/consolidation.py:1484-1509` — iterates rows x columns TWICE (values then formatting)
- Merge into single pass; use openpyxl bulk operations instead of cell-by-cell

### Sequential DataFrame merges in MR pipeline
- `src/mr_pipeline.py:247-304` — 10+ sequential `.merge()` calls creating intermediate DataFrames
- Consolidate into single groupby aggregation or use `pd.concat()` with keys

### `iterrows()` in hot paths
- `src/consolidation.py:494,1484,1929,1942` — metric extraction and export loops
- `src/inference_optimized.py:725` — cell predictions loop
- `src/stability.py:382` — PSI color mapping
- Replace with `.itertuples()`, vectorized operations, or `to_numpy()` iteration

---

## Performance — Medium Impact

### Unnecessary DataFrame `.copy()` calls
- `src/inference_optimized.py:287,309,346,557,696,771` — multiple copies during preprocessing
- `src/plots.py:685,1179,1189,1358,1472,1775` — plot preparation copies
- `src/trends.py:36,67,205` — trend analysis copies
- Use `.copy(deep=False)` where mutations are column-level only, or eliminate copies via method chaining

### `apply()` with lambda functions
- `src/trends.py:226` — rolling MAD computation
- `src/metrics.py:351-352` — tuple formatting
- `src/plots.py:494,1811,1816` — text column generation
- Replace with vectorized numpy/pandas operations or list comprehensions

### Chained string operations in data loading
- `src/data_manager.py:89-91` — converts to "string" dtype (high memory), applies 2 string methods, then converts to "category"
- Do replacements on original dtype before converting; use direct numpy operations

### Repeated `groupby().size()` in reject inference
- `src/reject_inference.py:66-67` — two separate groupby + merge operations
- Consolidate into single aggregation with `.agg()` for one pass through data

### Nested range loops in 2D grid processing
- `src/plots.py:744-779,1681-1710` — nested `range(len(...))` for acceptance boundary detection
- Use numpy `argmax`/`argmin` on masked arrays for vectorized boundary search

---

## Testing Gaps — Critical

### No tests for pipeline orchestration layers
- `src/pipeline/preprocessing.py` — `run_preprocessing_phase()` untested
- `src/pipeline/inference.py` — `run_inference_phase()` untested
- `src/pipeline/optimization.py` — `run_optimization_phase()` untested
- `src/pipeline/reporting.py` — `generate_segment_report()`, `generate_batch_reports()` untested
- These are the main entry points that wire modules together; failures here break the whole pipeline

### Massive untested surface in inference_optimized.py
- ~20 functions with zero direct test coverage including:
  - `_get_model_complexity()`, `_apply_one_se_rule()`, `calculate_target_metric()`
  - `_select_model_type_cv()`, `_select_feature_set_cv()` (CV model selection)
  - `compute_cell_level_ci()`, `todu_average_inference()`
  - `compute_pre_reject_inference_data()`, `_compute_shap_values()`
- Most complex module in the codebase (1713 LOC) with least proportional coverage

### Untested reject inference orchestration
- `apply_reject_inference()` — the main orchestration function — has no direct tests
- `_enforce_multiplier_monotonicity()` — the isotonic enforcement — untested
- Edge cases: zero acceptance rate bins, unknown method names, empty demand data

### Missing error path tests across modules
- `data_manager.py:load_and_prepare_data()` — FileNotFoundError, encoding errors untested
- `global_optimizer.py:optimize_exact()` — MILP solver timeout/infeasibility untested
- `mr_pipeline.py:process_mr_period()` — insufficient H3 data, division by zero untested
- `optimization_utils.py:CellGrid.from_summary()` — duplicate bin edges, NaN in grid untested

---

## Testing Gaps — High Priority

### Optuna tuning minimal coverage
- `src/optuna_tuning.py` — only 1 smoke test (`test_tune_tree_models_runs_without_crashing`)
- Missing: invalid data, direction constraints, monotonic constraints, trial failure handling

### Trends and sensitivity analysis untested functions
- `src/trends.py:detect_trend_changes()` — untested; SPC anomaly detection logic uncovered
- `src/sensitivity.py:compute_cell_marginal_impact()`, `sensitivity_cell_detail()` — untested
- N>2 variable edge cases missing in sensitivity analysis

### Visualization functions untested
- `src/plots.py` — `plot_roc_curve()`, `visualize_metrics()`, `RiskProductionVisualizer` class, `plot_group_statistics()`, `plot_bin_threshold_diagnostic()` all untested
- Missing: error handling for empty/invalid score arrays

### Consolidation Excel export untested
- `src/consolidation.py:export_consolidated_excel()` — writes to disk but no verification tests
- Missing: H3 metrics consolidation, multi-period (main + MR), CSV/JSON edge cases

### No tests for batch mode or web UIs
- `run_batch.py` — multi-segment loading, global bin learning, supersegment aggregation untested
- `dashboard.py` / `gradio_dashboard.py` — no test coverage at all
- Allocation across segments (`run_allocation.py`) untested end-to-end

### Missing boundary and edge case tests
- Empty DataFrames in `compute_acceptance_rates()`
- Single bin in `fit_h3_extrapolation_curve()`
- Division by zero / `log(0)` in extrapolation
- Risk exactly at threshold boundaries (0.15% for PSI)
- Conflicting date ranges in MR period config
- `SegmentConstraints` with infeasible `min_risk > max_risk`

---

## Architecture — High Impact, Moderate Effort

### Circular dependency: inference_optimized <-> optuna_tuning
- `src/inference_optimized.py:42` imports from `optuna_tuning.py`
- `src/optuna_tuning.py:15-16` imports `process_dataset` from `inference_optimized.py`
- Changes to either require coordinated updates; hard to test in isolation
- Extract shared interface or move `process_dataset` to a shared module

### File I/O scattered across business logic (39 instances)
- `consolidation.py` — `.to_csv()`, `pd.read_csv()`, `.write_html()` mixed with aggregation logic
- `inference_optimized.py` — `joblib.dump()`, `.write_html()`, `.to_csv()` in model training functions
- `mr_pipeline.py` — `.to_csv()`, `.write_html()` in risk analysis functions
- `plots.py` — `.write_html()` in 25+ locations
- Extract to I/O abstraction layer; decouple persistence from business logic

### Inconsistent error handling strategy across pipeline
- `main.py:66-95` — custom exception hierarchy with chaining (good)
- `consolidation.py:395-400` — silent exception handling with `logger.error`
- `mr_pipeline.py:120-130` — `traceback.print_exc()` in try/except (anti-pattern)
- Create custom exception hierarchy: `PipelineError`, `DataError`, `ModelError`, `OptimizationError`

### Visualization code in non-visualization modules
- `inference_optimized.py` creates 3+ Plotly figures directly instead of delegating to `plots.py`
- `mr_pipeline.py` builds risk comparison figures directly
- Move all chart creation to `plots.py` factory methods

---

## Architecture — High Impact, High Effort

### Large modules should be split
- `consolidation.py` (1949 LOC) → loader / aggregation / export submodules
- `plots.py` (1902 LOC) → static plots / interactive visualizer / 3D plots / utilities
- `inference_optimized.py` (1713 LOC) → data prep / model training / feature engineering / orchestration
- `optimization_utils.py` (1439 LOC) → grid operations / MILP solver / solution evaluation

### Strategy pattern for pluggable algorithms
- Reject inference: hardcoded `if bayesian_smoothing` → `RejectInferenceStrategy` interface
- Binning: hardcoded "quantile"/"optimization" → `BinningStrategy` with registry
- Extrapolation: hardcoded method dispatch → `ExtrapolationStrategy` interface
- Optimization solvers: scipy.milp + GA fallback → `OptimizationSolver` interface (SCIP, CBC, Gurobi)

### Public APIs expose internal implementation details
- `optimization_utils.py` exports 14 functions, many internal (`classify_by_mask`, `mask_to_cutoffs`, `decode_mask`, `_ga_pareto_fallback`)
- `pipeline/optimization.py` imports 13 functions from `optimization_utils`
- Expose only high-level APIs: `grid, frontier = optimization.solve_frontier(summary_df, settings)`

### OutputPaths dependency injection
- 30+ functions accept `output: OutputPaths | None = None` with inline default construction
- Fragile: forgotten parameter defaults to `.` directory
- Consider context manager or DI container pattern

---

## Robustness — Medium Priority

### `os.chdir()` in batch processing not protected against ThreadPoolExecutor
- `run_batch.py:430-441` — `_working_directory()` uses process-global `os.chdir()`
- Comment warns about ThreadPoolExecutor incompatibility but no runtime guard
- Add runtime check to prevent usage with `ThreadPoolExecutor`

### Joblib deserialization security
- `src/persistence.py:183` — `joblib.load()` can execute arbitrary code from untrusted pickle
- Safe for internal pipeline; document security requirements if models are loaded from external sources

### Silent NaN propagation with hardcoded fallback
- `src/optimization_utils.py:431-433` — if `max_risk` is NaN, silently falls back to `20.0`
- Add logging when NaN fallback occurs so users know underlying data is problematic

### Type annotations missing on public APIs
- `src/metrics.py:39,47,53,69-71` — `ks_statistic()`, `compute_metrics()`, `train_logistic_regression()` lack param/return types
- `src/models.py:29,45,72` — `extract_splits_from_tree()`, `optimal_splits_using_tree()`, `calculate_financing_rates()` lack return types
- Reduces IDE support and type safety; add explicit hints per Python 3.12 target

---

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
- Escalate moderate -> critical if drift persists for N consecutive runs

### Fairness / disparate impact monitoring
- Optional demographic parity and equalized odds metrics per cutoff solution
- Supports fair lending compliance (ECOA / HMDA reporting requirements)
- Add as opt-in module gated by `[fairness]` config section

### Cutoff robustness analysis
- Jack-knife leave-one-fold-out retraining to measure cutoff stability across data perturbations
- Flag Pareto solutions whose cutoffs shift significantly under resampling
- Integrates with existing bootstrap infrastructure in `src/utils.py`

---

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
- Add a small versioned fixture dataset and an integration test running the full pipeline from `config.toml` -> reports

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

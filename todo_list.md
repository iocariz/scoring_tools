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
- ~~PSI epsilon handling distorts divergence~~ — PSI/CSI in `src/stability.py` now use epsilon-adjusted percentages consistently in both the difference and `log(p/q)` terms without re-normalizing distributions; plotting thresholds remain centralized via `PSI_STABLE_THRESHOLD`/`PSI_UNSTABLE_THRESHOLD`
- ~~Silent data loss from unlogged `dropna()`~~ — added row-count logging before each `dropna()` in `preprocess_improved.py` (5 locations: `learn_quantile_bins`, `learn_income_bins`, `learn_optimization_bins`, `assess_binning_gini`, `_apply_binning_from_config`)
- ~~`calculate_stress_factor` returns 0.0 on empty data~~ — changed to return neutral `1.0` to prevent zeroing all risk predictions; also handles zero `overall_bad_rate` case
- ~~Reject inference denominator warning~~ — enhanced warning in `reject_inference.py:compute_acceptance_rates` to log non-score rejection share of all rejections; escalates to `WARNING` when >5% of demand, adds actionable guidance when >10%
- ~~Reject inference parceling per-variable overwrites grid structure~~ — `_enforce_multiplier_monotonicity` now iterates isotonic passes across all variables until convergence (max 10 iterations, tol=1e-6) instead of single-pass per variable
- ~~Stressor bounds validation~~ — `calculate_B2` in `src/models.py` now clamps non-positive stressor to 0.01 and warns when >5.0
- ~~`tasa_fin` validation~~ — `src/pipeline/preprocessing.py` now validates `tasa_fin > 0` (defaults to 1.0 if non-positive) and warns if >5.0
- ~~MR pipeline / audit unchecked `.iloc[0]`~~ — `mr_pipeline.py:70` was already guarded; added missing guard in `src/audit.py:generate_audit_table` which had no empty-check before `.iloc[0]`
- ~~Missing column existence checks in optimization~~ — added explicit `required_cols` validation in both `trace_pareto_frontier` and `_pareto_ga_fallback` in `src/optimization_utils.py`; returns empty result with descriptive error instead of `KeyError`
- ~~TweedieGLM ignored `sample_weight` with exposure~~ — `src/estimators.py` now forwards `sample_weight` in exposure-aware fits
- ~~TweedieGLM divide-by-zero on zero exposure~~ — `src/estimators.py` now clips exposure to a minimum positive value during prediction
- ~~HurdleRegressor negative predictions~~ — `src/estimators.py` now clips outputs to non-negative values
- ~~Degenerate Beta prior in reject inference smoothing~~ — `src/reject_inference.py` now floors `alpha` / `beta` at `0.5`
- ~~`bins_tuple` ordering followed dict insertion order~~ — `src/pipeline/inference.py` now orders bin edges explicitly by `inference_variables`
- ~~`classify_by_mask` assumed reset-index column named `index`~~ — `src/optimization_utils.py` now maps masks via variable tuples and preserves the original index
- ~~Global optimizer MILP fallback used `argmax(rounded)`~~ — `src/global_optimizer.py` now decodes near-integral solutions via `argmax(seg_x)`
- ~~MR `calculate_todu_30ever_from_b2` omitted configured multiplier~~ — `src/mr_pipeline.py` now passes `settings.multiplier` explicitly when reconstructing `todu_30ever_h6`
- ~~Consolidation ratio SE treated as additive~~ — `src/consolidation.py` now combines optimum risk CI uncertainty via an exposure-weighted delta-method instead of adding ratio SEs directly
- ~~Consolidation production CI used midpoint-derived intervals~~ — `src/consolidation.py` now aggregates production CI by summing segment lower/upper bounds
- ~~Scenario auto-detection stopped at first segment~~ — `consolidate_segments()` now scans all segment folders and keeps named-suffix deduplication for base scenarios
- ~~Pipeline orchestration layer coverage~~ — added focused wrapper tests in `tests/test_pipeline_orchestration.py` for `run_preprocessing_phase()`, `run_inference_phase()`, `run_optimization_phase()`, `generate_segment_report()`, and `generate_batch_reports()`
- ~~Inconsistent `dayfirst` parsing in `filter_by_date()`~~ — `src/preprocess_improved.py` now parses source values and filter bounds with explicit `dayfirst=False`; ambiguous-date regression added in `tests/test_preprocessing.py`
- ~~Greedy global production floor enforcement~~ — `src/global_optimizer.py` now prioritizes production growth until `global_production_floor` is met and raises if infeasible; regression coverage added in `tests/test_global_optimizer.py`

### Recently Fixed (deep review audit, 2026-03-13)

**Critical (round 1):**
- ~~PSI/CSI difference term used epsilon-adjusted values~~ — now uses original proportions for the difference `(Actual% - Expected%)`, epsilon only inside `log()`; both PSI and CSI consistent
- ~~Isotonic monotonicity collapsed N-D to 1-D means~~ — `_enforce_multiplier_monotonicity` rewritten with per-slice isotonic regression and alternating-pass convergence
- ~~`only_mr_sparse` evaluated before `can_extrapolate`~~ — reordered so H3 extrapolation is not blocked for MR-only bins that have H3 data

**Medium (round 1):**
- ~~Power extrapolation formula didn't match fitted model~~ — `extrapolate_h3_to_h6` power method now uses deviation `b2_h3/b2_h3_main` with the fitted alpha, matching the log-log regression model
- ~~`fit_h3_extrapolation_curve` SE inconsistent with polyfit~~ — SE now uses `eff_w = w**2` to match `np.polyfit`'s weighting convention
- ~~`calculate_annual_coef` missing date validation~~ — validates `date_fin >= date_ini`
- ~~Bin edges sort validation~~ — both `validate_bins_length` and `BinConfig.__post_init__` now enforce sorted, monotonically increasing edges
- ~~SPC n_sigma too permissive~~ — default changed from 2.0 to 3.0 (standard 3-sigma rule)
- ~~Production CI aggregation used sum-of-bounds~~ — now uses variance addition (independence assumption) for statistically correct interval widths
- ~~GA fallback missing post-hoc feasibility check~~ — now verifies monotonicity + risk constraints after GA optimization
- ~~Outlier detection used non-robust z-score~~ — `inference_optimized.py` now uses MAD-based detection instead of `scipy.stats.zscore`
- ~~MR b2 values filled NaN with 0.0~~ — removed `.fillna(0.0)` on `b2_main`, `b2_mr`, `b2_main_h3`; NaN now correctly triggers fallback logic
- ~~RI optimizer missing `enforce_monotonicity`~~ — `OptimizerInputs` dataclass now includes `enforce_monotonicity` field, passed through to `apply_parceling_adjustment`
- ~~RI optimizer missing `per_bin_tasa_fin`~~ — `OptimizerInputs` dataclass now includes `per_bin_tasa_fin`, applied in `evaluate_ri_params`
- ~~Greedy allocator `min_production` override could violate `max_risk`~~ — now checks `max_risk` before accepting production floor override
- ~~`LinearRegression` intercept caused bias in `reg_todu_amt_pile`~~ — set `fit_intercept=False` since zero production should predict zero exposure
- ~~Bayesian smoothing/prior_strength not propagated to `compute_acceptance_rates`~~ — optimization pipeline now passes both parameters through

**High (round 2):**
- ~~Power extrapolation NaN for MR-only bins~~ — `extrapolate_h3_to_h6` now falls back to linear per-element when `b2_h3_main` is NaN/0 instead of producing NaN
- ~~NaN in bootstrap breaks `np.percentile`~~ — changed to `np.nanpercentile` so CIs are computed correctly even with NaN replicates
- ~~Rounding to 2dp in intermediate b2 calculations~~ — `calculate_b2_ever_h6` calls for `b2_main`, `b2_mr`, `b2_main_h3`, `b2_mr_h3` now use `decimals=6` to preserve precision for ratio computations
- ~~main.py re-run after RI optimizer missing per_bin_stress/per_bin_tasa_fin~~ — second `run_scenario_analysis` loop now passes both kwargs

**Medium (round 2):**
- ~~Dead code: ratio_of_sums reconciliation~~ — `main_agg` dropped raw columns before reconciliation check; rewritten to recompute from `data_booked` directly
- ~~Empty DataFrame crash in `_enforce_multiplier_monotonicity`~~ — added early return guard for empty DataFrames
- ~~Calibration error used raw acceptance rate while RI used smoothed~~ — `_compute_calibration_error` now uses `smoothed_acceptance_rate` when available, consistent with `apply_parceling_adjustment`
- ~~Greedy allocator fallback ignores `max_risk`~~ — when `min_production` is unachievable, fallback now respects `max_risk` constraint
- ~~CV with single fold produces NaN~~ — `np.std(ddof=1)` guarded with `len > 1` check

**Low (round 2):**
- ~~PSI/CSI epsilon application inconsistent~~ — CSI changed from `.clip()` to `.where()` matching PSI's approach
- ~~Variables without bins pass validation~~ — `_auto_populate_bins` now validates every variable has a corresponding `bins` entry after auto-population

---

## Code Audit Findings (2026-03-10)

New findings from comprehensive 4-way audit (statistics, MILP/optimization, data pipeline, architecture).
Deduplicated against existing items. Items already tracked above are not repeated.

### BUG — CRITICAL

#### ~~C1-new: Float equality in Pareto deduplication~~ ✓ FIXED
- `optimization_utils.py:1428` — round `b2_ever_h6` to 4dp before `drop_duplicates`
- `global_optimizer.py:149` — tolerance-based cummax comparison (1e-4 threshold)

#### C2-new: Sparse grid zero-fill creates phantom cells in CellGrid
- `optimization_utils.py:59` — `CellGrid.from_summary()` fills missing bin combos with 0 production/risk
- MILP treats phantom cells as "free" rejections (0 cost, 0 risk contribution)
- In 3D grids with many empty cells, frontier includes solutions accepting impossible bin combos
- Fix: add `observed_in_data` flag; optionally exclude or penalize unobserved cells

#### ~~C3-new: GA fallback Pareto filtering incomplete~~ ✓ FIXED
- `optimization_utils.py:_ga_pareto_fallback` — added cummax stage-1 + full pairwise dominance stage-2 (mirrors `trace_pareto_frontier` logic)

### BUG — HIGH

#### H1-new: H3 reject-inference propagation locks in historical ratio
- `reject_inference.py:356-359` — multiplier applied identically to H6 and H3 numerators
- Preserves observed H6/H3 ratio exactly; if biased, bias propagates to H3→H6 extrapolation
- Fix: add flag to optionally apply multiplier only to H6 (keep H3 original)

#### H2-new: Outer merge in MR pipeline risks Cartesian product
- `mr_pipeline.py:271` — `mr_agg` (H6-valid subset) merged with `mr_prod` (all booked) via `how="outer"`
- Different row counts can produce duplicates if merge keys aren't unique
- Fix: add row-count assertion after merge

#### H3-new: No enumeration limit for legacy 2-var combinatorics
- `optimization_utils.py:1288-1320` — `combinations_with_replacement` unbounded
- 15 bins × 30 cutoffs → 3.5M solutions → potential memory exhaustion
- Fix: add hard limit (e.g., 500K) with automatic fallback to MILP sweep

#### H4-new: 2D vs N-D feature paths produce different feature sets
- `models.py:126-147` — 2-var hardcoded polynomial vs N-D `PolynomialFeatures(degree=3)`
- Switching between 2 and 3 variables changes model features unpredictably
- Fix: unify on `PolynomialFeatures` for all dimensions

#### H5-new: H6/H3 ratio clipped to [0.5, 5.0] without logging
- `mr_pipeline.py:354-356` — clips silently, masking high-growth bins
- Fix: log count of clipped bins; consider percentile-based clip

### BUG — MEDIUM

#### ~~M1-new: BinConfig doesn't validate edge monotonicity~~ ✓ FIXED
- `config.py:185-189` — both `validate_bins_length` and `BinConfig.__post_init__` now enforce sorted, monotonically increasing edges
- Also added validation that every variable has a corresponding bins entry after auto-population

#### ~~M2-new: PSI/CSI epsilon application inconsistent~~ ✓ FIXED
- CSI changed from `.clip()` to `.where()` matching PSI's approach — both now use `.where(pct > 0, min_pct)`

#### M3-new: b2_ever_h6 calculated in 4 different places
- `utils.py`, `consolidation.py`, `inference_optimized.py`, `mr_pipeline.py`
- `consolidation.py` doesn't use `DEFAULT_RISK_MULTIPLIER` constant
- Fix: consolidate all to use `calculate_b2_ever_h6()` from utils.py

#### M4-new: Column names as magic strings despite Columns enum
- `Columns` enum exists in `constants.py` but many modules hardcode `"oa_amt_h0"` etc.
- Fix: audit and migrate to enum references

#### M5-new: Bare exception handlers swallow traceback context
- `inference_optimized.py:1058`, `run_batch.py:94-96`, `global_optimizer.py` (~300)
- Fix: use specific exception types; add `logger.exception()` for unexpected errors

#### M6-new: DQ checks don't validate pipeline invariants
- `data_quality.py:430-477` — no checks for merge key uniqueness, post-binning bin counts, cross-column consistency
- Fix: add merge-key and post-binning validation rules

#### M7-new: Segment names unsanitized in filesystem paths
- `run_batch.py:239` — `../../malicious` in segments.toml could escape output directory
- Fix: validate segment names against `^[a-zA-Z0-9_-]+$`

#### M8-new: Settings object mutated in-place during pipeline **(partially fixed)**
- `preprocess_improved.py:873-904, 1066-1076` — learned bin edges and directions written back to settings
- data_manager.py uses `model_copy()` (correct), but MR-period processing may inherit first-run mutations
- Fix: deep-copy settings before each pipeline phase, or return new settings from each phase

### TESTING GAPS (new)

#### T1-new: 1D mode not tested end-to-end
- No test runs `variables=['single_var']` through config → MILP → visualization → report
- Fix: add integration test with 1D config

#### T2-new: N>2 optimization integration untested
- No test for 3-variable CellGrid → MILP → Pareto → cutoff summary
- Fix: add 3D integration test

#### T3-new: Edge cases untested
- All-rejected data, empty frontier, single-row input, all-NaN risk columns
- Fix: add edge-case test suite

### DASHBOARD IMPROVEMENTS (2026-03-10) ✓ IMPLEMENTED

#### Phase 1: `load_cutoff_data()` fix for all dimensions
- Removed `len(variables) == 2` cap in fallback pattern matching (dashboard.py + gradio_dashboard.py)
- Mask decode now fires for 1D (`len(variables) != 2` instead of `> 2`)

#### Phase 5: `is_1d` flag and store data
- Added `is_1d` flag to `cutoff-data-store`; 1D routes through mask-based path (`is_nd=True`)
- Updated layout: 1D-specific labels, guide text, acceptance strip header
- DataTable always visible (was hidden for N==3); slice grids shown for N>=3

#### Phase 3: Generalize slice grids beyond N==3
- `_build_nd_slice_grid_figure()` now accepts `fixed_vars` dict + `grid_x_var`/`grid_y_var` — works for any N>=3
- `_build_nd_slice_grid_panel()` iterates all unique combos of variables[2:] as slices (capped at 12)
- All `len(variables) != 3` guards relaxed to `< 3`

#### Phase 4: Per-variable marginal impact summary
- New `variable-marginal-impact` chart (small multiples, one subplot per variable)
- Shows avg production delta per bin value, grouped by variable
- Visible only for N>2; hidden for 1D and 2D

#### 1D visualization in callback
- Bar chart with green/red bins (production + accept/reject coloring) in heatmap slot
- Per-bin marginal impact bar chart
- Gradio: 1D bar chart with accept/reject colors

---

### ARCHITECTURAL

#### A1-new: inference_optimized.py is too monolithic (1811 lines)
- Mixes model training, optimization orchestration, SHAP, and visualization
- Fix: split into inference.py, optimization.py, visualization.py, explainability.py

#### A2-new: Global state via os.chdir in ProcessPoolExecutor
- `run_batch.py:430-441` — `os.chdir()` is process-global
- Safe with ProcessPoolExecutor but breaks with ThreadPoolExecutor
- Fix: pass explicit paths instead of changing cwd

#### A3-new: Global bin learning doesn't validate coverage
- `run_batch.py:99-154` — learned edges may not span all segments' score ranges
- Fix: validate per-segment min/max falls within learned edges

---

## Code Audit Findings (2026-03-06)

Items below are new findings from a comprehensive codebase audit. Duplicates against existing
todo items have been removed. Items marked **(partially fixed)** overlap with "Recently Fixed"
entries above but have residual issues.

### BUG — CRITICAL

#### ~~C1: Optuna `tune_linear_models` reuses unfitted model across folds~~
- `src/optuna_tuning.py:324-328` — model object is not cloned between CV folds
- Each fold trains on the already-fitted model from the prior fold, leaking information
- Fix: `clone(model)` before each fold's `.fit()`

#### ~~C2: Inconsistent sample weights between `inference_optimized` and `optuna_tuning`~~
- `src/inference_optimized.py` passes `sample_weight` to model `.fit()` during CV
- `src/optuna_tuning.py` may not propagate the same weighting scheme
- Models selected via Optuna are evaluated under different conditions than final training

#### ~~C3: MR pipeline missing `multiplier` in Optimum row risk calculation~~
- `src/mr_pipeline.py:189` — `calculate_b2_ever_h6(opt_rn, opt_rd, as_percentage=True)` omits `multiplier=multiplier`
- Optimum row always uses default multiplier (7) regardless of config
- Causes incorrect risk display in MR comparison tables

### BUG — HIGH

#### ~~H1: H3 extrapolation prioritized over direct H6 observations in hybrid risk~~
- `src/mr_pipeline.py:362-368` — condition ordering: `[only_mr_sparse, can_extrapolate, use_mr]`
- `can_extrapolate` (H3→H6 extrapolation) is checked before `use_mr` (direct H6 observations)
- When both H3 and H6 data exist, the extrapolated value is used instead of the direct observation
- Fix: swap order so `use_mr` is checked before `can_extrapolate`

#### ~~H2: Weighted R² and SE computation uses unweighted residuals~~
- `src/utils.py:155-170` — `np.polyfit` does WLS but `ss_res = np.sum(residuals**2)` is unweighted
- R² is inconsistent (weighted fit, unweighted goodness-of-fit)
- SE of slope is wrong, affecting confidence in extrapolation parameters
- Fix: `ss_res = np.sum(w * residuals**2)`

#### H3: Settings H3 field cleanup not propagated in data_manager **(partially fixed)**
- `src/data_manager.py:105-115` — settings object may be mutated for H3 column cleanup
- "Recently Fixed" entry says `model_copy()` is used, but the H3-specific cleanup path may still mutate the original
- Verify that all H3 column name standardization uses the copied settings

#### ~~H4: Inconsistent `dayfirst` parsing in date columns~~
- `src/preprocess_improved.py` — `filter_by_date()` now parses both source values and boundary dates with explicit `dayfirst=False`
- Regression coverage added in `tests/test_preprocessing.py` for ambiguous string dates

#### ~~H5: TweedieGLM drops `sample_weight` when exposure column is present~~
- `src/estimators.py:280-290` — when exposure is used, `sample_weight` passed to `.fit()` is ignored
- Exposure handling replaces weighting rather than composing with it
- Weighted training (e.g., for vintage imbalance) silently has no effect

#### ~~H7: TweedieGLM `predict` division by zero on zero exposure~~
- `src/estimators.py:310-320` — `predictions / exposure` when exposure contains zeros
- No guard against zero exposure values
- Fix: clip exposure to minimum positive value before division

#### ~~H8: HurdleRegressor can produce negative predictions~~
- `src/estimators.py:180-200` — combines P(event) × E[amount|event]
- Continuous sub-model can predict negative amounts; product can be negative
- Credit risk predictions must be non-negative
- Fix: `np.clip(predictions, 0, None)` at output

#### ~~H10: Degenerate Beta prior in reject inference Bayesian smoothing~~
- `src/reject_inference.py:76-85` — Beta prior with `alpha = global_rate * prior_strength`, `beta = (1 - global_rate) * prior_strength`
- When `global_rate` ≈ 0 or ≈ 1, one parameter approaches 0, creating a degenerate prior
- Fix: add floor `max(alpha, 0.5)`, `max(beta, 0.5)` (Jeffreys-like minimum)

#### H9: Isotonic monotonicity collapses N-D structure to 1-D marginals **(partially fixed)**
- `src/reject_inference.py` — `_enforce_multiplier_monotonicity` iterates isotonic passes per variable
- "Recently Fixed" entry says convergence loop added, but isotonic regression on marginals doesn't guarantee joint monotonicity in N-D
- A cell can satisfy all marginal monotonicity constraints while violating the partial-order constraint
- Consider lattice-based isotonic regression for true N-D monotonicity

#### ~~H11: Global optimizer MILP fallback uses `argmax` on rounded values~~
- `src/global_optimizer.py` — when MILP returns fractional solution, `argmax(rounded)` may select dominated point
- Should use proper rounding heuristic that respects constraints

#### ~~H12: Greedy global production floor was not enforced~~
- `src/global_optimizer.py` — per-segment `min_production` was already handled; the remaining gap was `global_production_floor` in the greedy path
- `optimize_greedy()` now prioritizes production growth until the floor is met and raises `RuntimeError` when infeasible under the risk target
- Regression coverage added in `tests/test_global_optimizer.py`

#### ~~H13: Consolidation ratio SE treated as additive~~
- `src/consolidation.py` — optimum risk CIs now use an exposure-weighted delta-method combination of per-segment uncertainty
- Ratio statistics (risk = defaults/exposure) are no longer combined by adding segment ratio SEs directly
- Regression coverage added in `tests/test_consolidation.py`

#### ~~H14: `bins_tuple` ordering follows dict insertion order~~
- `src/pipeline/inference.py:91-93` — `bins_tuple` constructed from dict keys
- If dict order doesn't match `inference_vars` order, bin assignments misalign with model expectations
- Fix: explicitly order by `inference_vars`

### BUG — MEDIUM

#### ~~M5: `classify_by_mask` assumes column named "index"~~
- `src/optimization_utils.py` — references hardcoded `"index"` column name
- Fails if DataFrame index was not reset or has different name
- Fix: use `.reset_index()` or parametrize column name

#### M6: GA fallback rounds continuous variables to integers
- `src/optimization_utils.py` — GA treats bin indices as continuous, rounds at evaluation
- Rounding creates flat fitness landscape regions where gradient is zero
- GA may fail to converge near optimal integer solutions

#### M7: Non-strict Pareto dominance filter
- `src/optimization_utils.py` — Pareto filter uses `<=` instead of `<` for at least one objective
- Weakly dominated solutions remain on frontier, inflating solution count
- Fix: require strict improvement in at least one objective

#### M8: `_transform_variables_2d` mutates input DataFrame
- `src/models.py` — modifies input DataFrame in-place via column assignment
- Callers may not expect side effects
- Fix: operate on `.copy()` at function entry

#### M9: HurdleRegressor `get_params` breaks sklearn `clone()`
- `src/estimators.py` — `get_params()` returns nested estimator objects
- `sklearn.base.clone()` fails when sub-estimators aren't properly clonable
- Affects Optuna tuning and CV where `clone(model)` is called

#### M11: Cell-level CI uses normal distribution instead of t-distribution
- `src/inference_optimized.py` — `z * se` for confidence intervals
- With small cell counts (n < 30), normal approximation is anti-conservative
- Fix: use `t.ppf(alpha/2, df=n-1)` instead of `norm.ppf`

#### M12: SPC numpy operations lose DataFrame index
- `src/trends.py` — numpy operations on rolling statistics drop the datetime index
- Anomaly detection results can't be joined back to source data by date
- Fix: preserve index through numpy operations

#### M13: SPC trend detection has look-ahead bias
- `src/trends.py` — rolling statistics computed on the full series including future data
- In production, SPC bounds should only use data available at each point in time
- Fix: use expanding or strictly backward-looking windows

#### M14: Outlier removal in inference can remove high-risk bins
- `src/inference_optimized.py` — outlier detection based on statistical criteria
- High-risk bins are natural outliers in credit data; removing them biases risk estimates downward
- Fix: only remove outliers on features, not on the target variable

#### M16: `str.match` partial matching in preprocessing
- `src/preprocess_improved.py` — `str.match()` matches from start of string but doesn't require full match
- Can match unintended column names that share a prefix
- Fix: use `str.fullmatch()` or anchor with `$`

#### ~~M17: Consolidation production CI uses midpoints instead of bounds~~ ✓ FIXED (then improved)
- `src/consolidation.py` — aggregate production CIs now use variance addition (independence assumption) for statistically correct interval widths
- Supersedes the initial sum-of-bounds fix with proper SE aggregation: `combined_SE = sqrt(Σ SE_i²)`

#### ~~M18: Scenario auto-detection stops at first segment~~
- `src/consolidation.py` — scenario detection now scans all segment folders instead of stopping after the first populated one
- Preserves base-scenario deduplication and prefers named `_base` over the empty suffix when both exist

### BUG — LOW

#### L1: `plt.show()` blocks execution in non-interactive environments
- `src/models.py` — calls `plt.show()` which blocks in headless/CI environments
- Fix: guard with `if plt.isinteractive()` or remove in favor of `savefig()`

#### L4: Rolling mean column actually contains median
- `src/trends.py` — column named `rolling_mean` is computed with `.rolling().median()`
- Misleading column name
- Fix: rename to `rolling_median`

#### L5: Annualization with `n_months=0` not guarded
- `src/audit.py` — annualization formula divides by `n_months`
- If date range produces 0 months, division by zero occurs
- Fix: `max(n_months, 1)` or return un-annualized value

### STATISTICAL — HIGH

#### ~~S1: PSI epsilon-protected log but unprotected difference breaks non-negativity~~
- `src/stability.py` now uses epsilon-adjusted percentages consistently in both the difference and `log(p/q)` terms for PSI
- Mirrored the same formula fix in categorical CSI calculations
- Regression coverage added in `tests/test_stability.py`

### STATISTICAL — MEDIUM

#### S2: Bootstrap only resamples booked population
- `src/utils.py:425-433` — bootstrap CIs computed only on booked/accepted loans
- Reject inference adjustments not propagated through bootstrap
- CIs reflect sampling uncertainty of booked data only, not total portfolio uncertainty
- Underestimates true CI width when reject inference adjustment is significant

### METHODOLOGICAL — MEDIUM

#### M10: Acceptance rate `fillna(0)` creates extreme reject inference multipliers
- `src/reject_inference.py` — missing acceptance rates filled with 0
- `1 / acceptance_rate` with rate = 0 produces infinity; clipped but extreme
- Fix: fill with minimum observed positive rate or use Bayesian prior

### NUMERICAL — MEDIUM

#### ~~N1: `nan_to_num` silently converts NaN risk to 0 in MR pipeline~~ ✓ FIXED
- `src/mr_pipeline.py` — `np.nan_to_num` and `.fillna(0.0)` removed from b2 computations
- NaN risk now correctly triggers fallback logic (main_imputed or model_fallback) rather than being treated as zero

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

### `_auto_populate_bins` silently promotes legacy config **(partially fixed)**
- `src/config.py:368-393` auto-generates `BinConfig` from legacy `octroi_bins`/`efx_bins` fields
- No deprecation warning is emitted — users don't know they're using the legacy path
- Add `logger.warning` when auto-promotion occurs
- **(Fixed)** Validation now ensures every variable has a corresponding `bins` entry after auto-population

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

### ~~MR `calculate_todu_30ever_from_b2` ignored the configured multiplier~~
- `src/mr_pipeline.py:708-711` now passes `settings.multiplier` explicitly instead of relying on the default `multiplier=7`
- H6 risk back-calculation now uses the configured H6 scaling factor consistently

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

- ~~Non-nested cross-validation for hyperparameter tuning~~ — `tune_tree_models` and `tune_linear_models` now use 80/20 stratified holdout split: Optuna tunes on tuning set, final evaluation on held-out data for unbiased RMSE. Falls back to fresh-seed CV when dataset < 200 rows or holdout produces too few bins.
- ~~Reject inference denominator excludes non-score rejections~~ — added `reject_include_all_rejections` config field (default False for backward compat). When True, all rejections count in denominator, producing lower acceptance rates and higher multipliers. Wired through config → `pipeline/optimization.py` → `inference_optimized.py` → `reject_inference.py:compute_acceptance_rates`.
- ~~Bin edges learned on full training data without holdout~~ — `learn_optimization_bins` now runs a holdout stability check: applies learned edges to 80/20 train/holdout split, logs per-bin risk on both folds, warns if risk ordering reverses on holdout (overfit signal). Bin edges still learned on full data for maximum signal.

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

### Bin-edge learning uses pre-relabel “booked” population
- `src/preprocess_improved.py:_run_data_transformations` learns bin edges using `status_name == BOOKED` from `data_clean`
- `update_status_and_reject_reason(...)` runs after bin-edge learning and can relabel some records from booked->rejected based on `m_ct_direct*`
- TODO: ensure bin-edge learning and downstream “booked” label definition are consistent (use the post-update booked mask or recompute booked subset after relabeling)

### `tasa_fin`/stress computed on unfiltered `data_clean` time range
- `src/pipeline/preprocessing.py` computes transformation rate from `data_clean`
- `src/plots.py:_prepare_transformation_data` applies the “last n months” cutoff relative to `df[date_col].max()` of the passed dataframe
- This may include months outside `[date_ini_book_obs, date_fin_book_obs]`
- TODO: compute `tasa_fin`/stress using date-filtered frames aligned to the configured observation window (main book period)

---

## Statistical & Methodological — Medium Priority

### One-SE rule doesn't account for CV fold correlation
- `src/inference_optimized.py:70-106` — threshold = `best_mean + best_se` assumes independent folds
- CV folds from same split are correlated; effective degrees of freedom < k
- Threshold is overly generous, selecting overly complex models
- Ref: Breiman et al. (1984); adjust margin for fold correlation

### MILP→Pareto can propagate NaN risk from zero exposure denominators
- `src/optimization_utils.py:milp_solve_cutoffs` guards global `Σ(todu_amt_pile_h6) == 0`, but not the denominator for the *selected mask*
- `src/utils.py:calculate_b2_ever_h6` returns NaN when denominator==0; `trace_pareto_frontier` does not explicitly drop NaN-risk solutions
- TODO: enforce a minimum accepted exposure denominator in MILP (e.g., `Σ(todu_amt_pile_h6*x) >= eps`) and/or filter NaN-risk masks before Pareto dominance + scenario selection

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

### ~~No tests for pipeline orchestration layers~~
- Added focused wrapper coverage in `tests/test_pipeline_orchestration.py` for `src/pipeline/preprocessing.py`, `src/pipeline/inference.py`, `src/pipeline/optimization.py`, and `src/pipeline/reporting.py`
- The main entry points used by `main.py` and `run_batch.py` now have direct regression coverage

### Massive untested surface in inference_optimized.py
- ~20 functions with zero direct test coverage including:
  - `_get_model_complexity()`, `_apply_one_se_rule()`, `calculate_target_metric()`
  - `_select_model_type_cv()`, `_select_feature_set_cv()` (CV model selection)
  - `compute_cell_level_ci()`, `todu_average_inference()`
  - `compute_pre_reject_inference_data()`, `_compute_shap_values()`
- Most complex module in the codebase (1713 LOC) with least proportional coverage

### ~~Untested reject inference orchestration~~
- `tests/test_reject_inference.py` already has direct coverage for `apply_reject_inference()` and `_enforce_multiplier_monotonicity()`
- Remaining useful gaps are narrower edge cases such as empty demand data and more zero-acceptance-rate scenarios

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

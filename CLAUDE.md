# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk scoring and portfolio optimization pipeline. Processes loan application data (SAS `.sas7bdat` files), trains risk models on an N-dimensional score grid (e.g., internal "octroi" bins × external "EFX" bins, optionally × income bins), and finds optimal acceptance cutoffs via MILP with monotonicity constraints over a Pareto frontier of risk (`b2_ever_h6`) vs. production (`oa_amt_h0`). Supports H3 (3-month) early risk metrics alongside H6, hybrid MR extrapolation with auto-calibrated H3→H6 curvature, advanced reject inference with Bayesian smoothing, and multi-segment batch runs with global bin learning and per-segment constraints.

## Commands

```bash
# Install
uv pip install -e .

# Run pipeline (single segment)
uv run python main.py
uv run python main.py --config path/to/config.toml
uv run python main.py --training-only          # preprocessing + model training only
uv run python main.py --model-path models/dir  # use pre-trained model
uv run python main.py --skip-dq-checks         # skip data quality checks entirely
uv run python main.py --allow-dq-warnings      # proceed past non-critical DQ warnings (DQ is fail-closed by default)
uv run python main.py --baseline               # baseline: show current portfolio, no optimization
uv run python main.py --log-file run.log       # capture logs to file

# Run batch (multi-segment, reads config.toml + segments.toml)
uv run python run_batch.py
uv run python run_batch.py --parallel --workers 4 -s segment1 segment2
uv run python run_batch.py --list               # list available segments
uv run python run_batch.py --reuse-models        # reuse existing supersegment models
uv run python run_batch.py --consolidate-only    # only generate consolidated report
uv run python run_batch.py --clean               # clean output dirs before running
uv run python run_batch.py --no-report           # skip HTML reports
uv run python run_batch.py --no-backtest         # skip the out-of-time backtest step (M4) that feeds the consolidated report
uv run python run_batch.py --training-only       # only run DQ + training
uv run python run_batch.py --allow-dq-warnings   # proceed past non-critical DQ warnings (fail-closed by default)
uv run python run_batch.py --log-file batch.log  # capture all logs to file
uv run python run_batch.py --baseline           # baseline mode for all segments
uv run python run_batch.py --cutoff-ordering-mode bottom_up  # sequential cutoff ordering

# Resimulation (re-run scenarios with different risk targets, no re-training/optimization)
uv run python main.py --config output/segment/config_segment.toml --resimulate 0.8 1.2 1.6
uv run python run_batch.py --resimulate 0.8 1.2 1.6           # all segments, same targets
uv run python run_batch.py --resimulate 1.0 -s no_premium_cd  # single segment
uv run python run_batch.py --resimulate scenarios.toml         # per-segment targets from TOML
# scenarios.toml format:  no_premium_cd = [0.8, 1.2, 1.6]  or  no_premium_cd = 1.4

# Analyze logs and suggest config improvements
uv run python analyze_logs.py batch.log                      # analyze a log file
uv run python analyze_logs.py output/*/logs/*.log --verbose  # analyze all segment logs
uv run python analyze_logs.py batch.log -o report.txt        # save report to file

# Run allocation (global MILP across segments)
uv run python run_allocation.py --target 2.5
uv run python run_allocation.py --what-if 2.0,2.5,3.0
uv run python run_allocation.py --target 2.5 --method greedy
uv run python run_allocation.py --target 2.5 --production-floor 1000000
uv run python run_allocation.py --target 2.5 --lock segment1:3
# Writes beside `--output`: `<stem>_policy_cutoff_table.csv`, `<stem>_allocation_narrative.md` (single target); multi-target adds `<stem>_what_if.csv`, `<stem>_allocation_narratives.md`

# Run selection bias analysis
uv run python run_selection_bias_analysis.py
uv run python run_selection_bias_analysis.py --config config.toml --segments-config segments.toml

# Run out-of-time backtest of frozen cutoffs (M4)
uv run python run_backtest.py                                       # all frozen segments under output/
uv run python run_backtest.py -s no_premium_cd premium              # specific segments
uv run python run_backtest.py --data-dir output --output output/backtest --scenario base
uv run python run_backtest.py -s no_premium_cd --holdout-start 2025-06-01 --holdout-end 2025-08-01  # override auto-window
# Writes under `--output`: `backtest_<segment><suffix>.csv` (+ `_calibration.csv`), `backtest_consolidated<suffix>.csv`, `backtest_summary<suffix>.md`

# Reproducibility check — golden headline numbers (M5)
uv run python run_reproducibility.py -s no_premium_cd                       # re-derive + compare to committed reference
uv run python run_reproducibility.py -s no_premium_cd --update-reference    # (re)establish the committed reference
uv run python run_reproducibility.py -s no_premium_cd --model-path <dir>    # reuse a trained model (faster)
uv run python run_reproducibility.py -s no_premium_cd --risk-tol-pp 0.02 --prod-tol-pct 0.5
# PASS iff headline (risk/production/cells/accepted-set) reproduces within tolerance AND data SHA-256 matches the reference

# Policy registry + champion/challenger comparison (phase 1)
uv run python run_policy_registry.py --register                       # freeze each segment's base policy into the registry
uv run python run_policy_registry.py --register -s no_premium_cd --make-champion  # promote to champion (else first is auto-champion)
uv run python run_policy_registry.py --list -s no_premium_cd          # champion + policy history
uv run python run_policy_registry.py --compare                        # challenger (current base policy) vs champion on the matured holdout
uv run python run_policy_registry.py --compare -s no_premium_cd --holdout-start 2025-06-01 --holdout-end 2025-08-01  # pinned eval window
# Committed registry: reports/policy_registry/<segment>.json (one champion/segment, base scenario). --compare writes under --output (default output/policy_registry):
#   policy_compare_<segment>_base.csv (+ _celldiff.csv), policy_compare_consolidated_base.csv, policy_compare_summary_base.md
# Verdict is risk-only + noise-aware (BETTER/WORSE only with >=10 realized defaults in both policies AND fully separated risk CIs; else INCONCLUSIVE); reuses the M4 backtest + M5 headline/provenance, read-only w.r.t. the pipeline.

# 3D risk surfaces (b2_ever_h6 over the two score bins, one coloured surface per audit category)
uv run python plot_risk_surface.py                          # all segments + reporting supersegments (main period)
uv run python plot_risk_surface.py --period mr              # the proposed cutoff applied to the out-of-time (MR) cohort
uv run python plot_risk_surface.py -s no_premium_cd premium # specific segments
uv run python plot_risk_surface.py --scenario base --no-supersegments
# Reads a completed run's data_summary_desagregado_*.csv + accepted_cells_*.csv. ONE continuous b2_ever_h6 surface per income_bin
# (3rd grid var, faceted side-by-side), coloured per cell by audit category = (booked-before x accepted-now): keep/swap_out/swap_in/rejected.
# "booked before" = cell had booked exposure. One category per cell (segments are unambiguous; supersegments resolve by exposure-weighted
# majority). Writes HTML under --output (default output/risk_surfaces) + index.html.

# Generate methodology presentation (.pptx / .pdf) — explains HOW the pipeline works
uv run python generate_presentation.py
uv run python generate_presentation.py --pdf     # also convert to PDF (requires LibreOffice)

# Generate management RESULTS presentation (.pptx) — the actual numbers for management
uv run python generate_results_presentation.py                       # reads output/consolidated_risk_production.csv + backtest + risk surfaces
uv run python generate_results_presentation.py --surface-segment premium --pdf  # restrict surfaces to one segment (default: all)
# Exec deck: title/recommendation, exec-summary KPIs, methodology-in-brief slide, production/risk/acceptance by segment,
# swap-in/out bridge, a 2D+3D risk-surface slide PER segment AND reporting supersegment, out-of-time validation, next steps.
# Charts via src/presentation_charts.py (matplotlib + kaleido for 3D). Writes output/results_deck/Credit_Risk_Results_<scenario>.pptx
# (+ images/). 3D snapshot needs kaleido==0.2.1 (degrades to 2D-only if absent).

# Tests
uv run pytest tests/                                    # all tests
uv run pytest tests/test_models.py -v                   # single file
uv run pytest tests/test_models.py::test_func -v        # single test
uv run pytest --cov=src tests/ --cov-report=term-missing # with coverage

# Lint & format (ruff)
uv run ruff check .          # lint check
uv run ruff check --fix .    # lint auto-fix
uv run ruff format .         # format

# Docker
docker build -t scoring-tools .
```

Makefile shortcuts: `make run`, `make run-batch`, `make test`, `make lint`, `make format`, `make clean`, `make setup`, `make docker-build`, `make docker-run`.

## Architecture

### Pipeline Phases (main.py)

1. **Config** — `PreprocessingSettings` (Pydantic) loaded from `config.toml` via `from_toml()`
2. **Data Loading** — `src/data_manager.py` reads SAS files, standardizes columns (H3 columns optional)
3. **Preprocessing** — `src/pipeline/preprocessing.py` orchestrates DQ checks (`src/data_quality.py`), filtering/binning (`src/preprocess_improved.py`). Bin edge learning uses `"quantile"` (unsupervised equal-count splits) computed on the date-filtered **demand** population (all statuses) that the bins are applied to. The legacy `"optimization"` method (supervised `DecisionTreeRegressor` split) is deprecated — it leaked the risk target the optimizer maximizes — and now falls back to quantile with a warning.
4. **Inference** — `src/pipeline/inference.py` orchestrates model training with CV across feature sets; custom sklearn estimators in `src/estimators.py` (`HurdleRegressor`, `TweedieGLM`). Trains on `inference_variables` (subset of `variables`), decoupled from the optimization grid. Optuna hyperparameter tuning available for tree-based models.
5. **Optimization** — `src/pipeline/optimization.py` generates all monotonic cutoff combinations, computes KPIs per solution, applies optional reject inference (parceling with linear/power/sigmoid methods, optional Bayesian smoothing), filters to Pareto frontier. Supports swap-in production/risk constraints, fixed cutoffs, GA fallback for N>2 variables, baseline mode (no optimization), and sequential cutoff ordering across segments (nested mask constraints).
6. **Scenario Analysis** — selects optimal Pareto points at pessimistic/base/optimistic risk thresholds; bootstrap CI, MR validation (with H3→H6 extrapolation), PSI/CSI stability, audit tables
7. **Sensitivity Analysis** (optional) — cutoff sensitivity analysis
8. **RI Optimizer** (optional) — automated reject inference parameter tuning via grid/Optuna search; re-runs optimization if better params found
9. **Trend Analysis** — monthly metrics with SPC anomaly detection
10. **HTML Report** — self-contained segment report via Jinja2 templates with embedded Plotly charts

### Module Layout

- `src/pipeline/` — thin orchestration wrappers (`config_loader.py`, `preprocessing.py`, `inference.py`, `optimization.py`, `reporting.py`) that coordinate the core modules
- `src/` — core logic:
  - Config & constants: `config.py`, `constants.py`, `schema.py`
  - Data: `data_manager.py`, `data_quality.py`, `preprocess_improved.py`
  - Models: `estimators.py`, `models.py`, `inference_optimized.py`, `optuna_tuning.py`, `persistence.py`
  - Optimization: `optimization_utils.py`, `global_optimizer.py`
  - Reject inference: `reject_inference.py`, `reject_inference_optimizer.py`
  - Analysis: `mr_pipeline.py`, `stability.py`, `sensitivity.py`, `trends.py`, `alerts.py`, `audit.py`, `selection_bias.py`, `selection_bias_plots.py`
  - Validation & governance: `lineage.py` (M2 data lineage), `backtest.py` (M4 out-of-time backtest), `reproducibility.py` (M5 golden-numbers), `policy_registry.py` (policy registry + champion/challenger)
  - Output: `consolidation.py`, `reporting.py`, `plots.py`, `styles.py`, `metrics.py`, `utils.py`, `portfolio_owner.py`
- Entry points: `main.py` (single segment), `run_batch.py` (multi-segment with global bin learning + supersegment model sharing), `run_allocation.py` (MILP allocation with segment constraints/locking), `run_backtest.py` (out-of-time backtest of frozen cutoffs, M4), `run_reproducibility.py` (golden-numbers reproducibility check, M5), `run_policy_registry.py` (policy registry + champion/challenger comparison), `generate_presentation.py` (PowerPoint/PDF output), `run_score_metrics.py` (score discriminance), `run_selection_bias_analysis.py` (selection bias diagnostics), `dashboard.py` / `interactive_allocator.py` (Dash web UIs), `gradio_dashboard.py` (Gradio web UI)

### Key Design Patterns

- **`OutputPaths` dataclass** — centralized path management; every pipeline phase receives an instance
- **`PreprocessingSettings` Pydantic model** — all config flows through this with field/model validators (≥ 2 variables, bin edges ≥ 2, date parsing, MR period pairing, range constraints)
- **`BinConfig` dataclass** — N-variable binning config (`source_col`, `output_col`, `bin_edges`/`max_bins`, `method`) replacing legacy `octroi_bins`/`efx_bins`
- **`SegmentConstraints` dataclass** — per-segment risk bounds (`min_risk`/`max_risk`), production floors (`min_production`), frontier locking (`locked_sol_fac`)
- **`StatusName` / `RejectReason` / `Columns` enums** in `src/constants.py` — centralized string constants and numeric defaults (`DEFAULT_N_BOOTSTRAPS`, `DEFAULT_SENSITIVITY_LEVELS`) across the codebase
- **Custom sklearn estimators** — `HurdleRegressor` and `TweedieGLM` implement full `BaseEstimator`/`RegressorMixin` interface
- **Chunked processing** in optimization — feasible solutions processed in memory-efficient chunks for the combinatorial N-D grid search

## Configuration

Two-tier config: `config.toml` (global defaults) overridden per-segment by `segments.toml`.

**Core fields:** `variables` (≥ 2 score names), `inference_variables` (subset for model training), date ranges, `optimum_risk`, `risk_step`, `multiplier`, `directions` (monotonicity hints per variable).

**Model training:** `cv_folds`. `model_hurdle_per_loan` (bool, default `false`, Expert/default-off, audit #6) — when `true`, a two-part `HurdleRegressor` is offered as a model candidate, trained on **per-loan** data (real zero mass in the default indicator) with exposure-weighted severity, and scored on the same bin-level CV RMSE as the other candidates; it is skipped automatically when the per-loan zero mass is degenerate (∉ [2%, 99.9%]). When `false` the hurdle is not offered at all — on the bin-aggregated target it degenerates to plain Ridge/Lasso. Enabling it can change the selected risk model and therefore the cutoffs (validate on real data first).

**Data quality (M2, fail-closed):** `dq_allow_warnings` (bool, default `false`). DQ is a hard gate by default — both FAILED-severity checks (negative counts/amounts, unparseable dates, booked-ratio < 0.01) **and** WARNING-severity checks (date-coverage gaps, numeric outliers, small segments, booked-ratio 0.01–0.05) halt the run. The `--allow-dq-warnings` CLI flag (or `dq_allow_warnings=true` in config) is the analyst escape hatch: it relaxes only the WARNING tier (FAILED still halts). `--skip-dq-checks` skips DQ entirely. The flag can only *relax* DQ; absent, the resolved config value applies.

**Data lineage (M2):** every run writes `output/<segment>/data/run_lineage.json` (`src/lineage.py`) capturing the source data file (path, size, mtime as the snapshot/extraction proxy, SHA-256), loaded row count, run timestamp + run-id, git commit (+dirty), config path/hash, and headline assumptions (multiplier, stress_mode, optimum_risk, MR/RI toggles). A run banner is logged at entry; batch mode generates one canonical run-id shared by all segments. The segment HTML report shows a "Data Lineage / Provenance" section; the consolidated Excel exec summary carries a provenance line. Capture is best-effort — a lineage failure never aborts the pipeline; git/file stats degrade gracefully (Docker/non-repo/missing file).

**Binning:** `octroi_bins`/`efx_bins` (legacy) or `[preprocessing.bins.*]` (N-variable `BinConfig` with `source_col`, `output_col`, `bin_edges`/`max_bins`, `method`). Batch mode supports global bin learning across all segments.

**Reject inference:** `reject_inference_method` ("none"/"parceling"), `reject_parceling_method` ("linear"/"power"/"sigmoid"), `reject_uplift_factor`, `reject_max_risk_multiplier`, `reject_bayesian_smoothing`, `reject_bayesian_prior_strength`, `reject_enforce_monotonicity`, `reject_include_all_rejections`. **No/low-demand handling:** `reject_no_demand_anchor_percentile` (default 0.10) and `reject_confidence_scale` (default 10.0) — repesca bins with little/no demand have their acceptance rate shrunk toward a conservative *low* anchor (the given percentile of observed rates) by confidence `1 − exp(−n/scale)`, instead of the old anti-conservative median fallback; no-demand bins (conf 0) get the anchor, well-observed bins (conf ≈ 1) are ~unchanged. Smaller `scale` shrinks only genuinely sparse bins. RI optimizer: `run_ri_optimizer`, `ri_validation_split`, `ri_optimizer_methods` ("grid"/"optuna"), `ri_optuna_n_trials`, `ri_calibration_gamma`, `ri_uplift_range`, `ri_max_mult_range`.

**H3 metrics:** `multiplier_h3` (scaling factor for 3-month risk), `use_mr_outcomes` (enable H3→H6 extrapolation), `mr_min_obs_per_bin`, `mr_extrapolation_method` (`"linear"` / `"power"` / `"logistic"` / `"auto"`), `mr_extrapolation_curvature` (power exponent; ignored when method is `"auto"` — curvature is fitted from main-period data via weighted log-log regression in `fit_h3_extrapolation_curve`), `mr_maturity_months` (min months since booking for H6 maturity filter), `mr_extrapolation_risk_multiplier` (relative ceiling for MR risk), `mr_extrapolation_hard_cap` (absolute ceiling for risk %).

**Out-of-time backtest (M4):** `run_backtest.py` + `src/backtest.py` apply a completed run's **frozen** accepted-cell set (`accepted_cells*.csv`) **as-is** to a held-out out-of-time cohort and compare realized vs predicted risk/production. Distinct from the inline MR check (`mr_pipeline.process_mr_period`), which **re-optimizes** the mask and is risk-only — the backtest never re-fits bins/model/mask (read-only; writes only under `--output`). The held-out window is auto-derived as `(date_fin_book_obs, max(mis_date) − maturity_months]` (the post-training cohorts old enough to have a realized H6; trades recency for maturity, no H3 extrapolation). Risk is compared directly (a rate); the clean same-basis drift signal is **in-sample realized → OOT realized** (both booked-only under the identical cutoff) — `predicted` (from `optimal_solution*.csv`) carries reject-inference + stress conservatism, so it is a different basis. Production is reported as rates (acceptance %, realized €), since absolute € is not comparable across periods of different size. Per-cell calibration (predicted vs realized per accepted cell, with booked counts) flags thin cells. The OOT risk carries a **bootstrap CI** and the per-segment flag is **noise-aware**: `DRIFT` only when there are ≥`MIN_DEFAULTS_FOR_DRIFT` (10) OOT defaults AND the OOT CI sits entirely above the in-sample CI; `INCONCLUSIVE` when too few defaults; `OK` when CIs overlap (added after a 2026-06 investigation found the no_premium_cd/ef drift was within sampling noise on a ~2-cohort window — not a maturity artifact). `run_batch` runs the backtest **automatically** after segments (gated by `--no-backtest`, reusing the loaded data) so the consolidated report's out-of-time sheet is always populated; `backtest_all` (in `run_backtest.py`) is the shared orchestration.

**Consolidated report — trust layer.** `consolidated_risk_production.xlsx` (`export_consolidated_excel`, `src/consolidation.py`) surfaces the validation evidence for credit-risk/policy consumers: Exec-Summary KPI cards show **bootstrap CI bands** + a plain-language "Recommendation & key risks" narrative; a **"Validation & Governance"** sheet (data snapshot SHA/git/config from lineage, key assumptions + governance tier incl. `multiplier`-FIXED, per-segment reproducibility + PSI/stability status, sign-off pointer); and an **"Out-of-time Validation"** sheet (M4 backtest: predicted vs realized + CIs + the noise-aware flag). All trust-layer readers degrade gracefully (a missing artifact shows a note, never crashes the workbook).

**Validation & reproducibility (M5):** the validation pack lives in `reports/validation/` — `assumptions_register.md` (every load-bearing assumption with value/source/governance tier; `multiplier`/`multiplier_h3` are FIXED constants), `reproduction_runbook.md` (independent-reviewer steps), `model_validation_report.md` (reproduction + adversarial "break the numbers" findings), `mrm_signoff.md` (sign-off + residual risks + conditions for live/automated cutoffs), and `reference/<segment>_headline.json` (committed golden numbers + the data SHA-256/git/config they pin to). `run_reproducibility.py` + `src/reproducibility.py` re-derive a segment's headline (risk/production/accepted-cells) and PASS only if it reproduces within tolerance **and** the data SHA-256 still matches the reference (a changed snapshot fails loudly → re-validate). It reproduces under `allow_dq_warnings=True` (the same DQ posture the headline was set with). Standalone single-segment; production pooled-`total` numbers are validated via the M4 backtest + the #7 multi-segment validation.

**Stress & transformation:** `stress_mode` ("global"/"per_bin"/"disabled"), `per_bin_tasa_fin` (compute transformation rate per grid cell).

**Swap-in constraints:** `max_swapin_production_pct`, `max_swapin_risk`.

**Scenario selection basis (audit #28, Expert/default-off):** `selection_risk_basis` (`"point"` default | `"ci_upper"`). `"point"` is the classic rule (max production whose point-estimate risk ≤ target). `"ci_upper"` requires the candidate's bootstrap risk **CI-upper** ≤ target (per-candidate CI from the selection-aware bootstrap, `src/selection_uncertainty.py`) — a noise-margin rule that is materially more conservative by construction (on `no_premium_cd` it cut production −61% at the same 1.1% target); enabling it changes cutoffs and needs M5-style sign-off. Independent of the flag, every MILP-path scenario logs + persists selection-aware diagnostics on the summary row: `risk_ci_sel_lower/upper` (CI of the optimized headline including re-selection), `selection_optimism_pp` (estimated winner's-curse understatement), `selection_reselect_pct`. The fixed-mask `risk_ci_*` is computed on the blended booked+RI basis matching the headline (`risk_ci_basis` column; booked-only realized CI in `risk_booked_ci_*`).

**Segment constraints** (in `segments.toml`): `min_risk`, `max_risk`, `min_production` (production floor), `locked_sol_fac` (lock to specific frontier point). Supersegments (`[supersegments.*]`) define named groups of segments. Each segment can independently reference a **modelling supersegment** (shared model training) and a **reporting supersegment** (consolidated report grouping) via `modelling_supersegment` and `reporting_supersegment` fields. Legacy `supersegment` field sets both. Resolution: `modelling_supersegment > supersegment > None`, `reporting_supersegment > supersegment > None`.

**Fixed cutoffs:** `fixed_cutoffs` to skip MILP and use predefined cutoff combinations. For 2-var: paired bins/cutoffs lists. For N>2: per-variable lists of accepted bin values (cell accepted iff all coordinates are in their respective accepted lists). Sub-keys inside `fixed_cutoffs`:
- `strict_validation` (bool, default `false`) — fail loudly when the predefined cutoffs don't match the bin grid; otherwise drop silently with a warning.
- `run_all_scenarios` (bool, default `false`) — when true, also generate pessimistic/optimistic scenarios instead of just base.

**Per-variable minimum accepted bin:** `min_accepted_bin_by_variable` (dict) forces cells with `var_bin < threshold` to be rejected in optimization. Value is either a scalar (applies to all rows) or an `income_bin`-keyed map for per-income thresholds. Conflicts with must-accept `floor_cells` are detected and logged. Legacy alternative to `fixed_cutoffs` for simple monotonic pre-commits; prefer `fixed_cutoffs` for new segments.

**Baseline mode:** `baseline_mode` (bool, default false) — show current booked portfolio as-is with no cutoff optimization (Optimum = Actual, zero swap-in/swap-out). MR inference still runs to predict risk for immature loans. Only the base scenario is generated; sensitivity and RI optimizer are skipped. Available via config or `--baseline` CLI flag.

**Base scenario only:** `base_scenario_only` (bool, default false) — generate only the base scenario; skip pessimistic and optimistic. Config-only flag; no CLI equivalent (unlike `--baseline`). Distinct from `baseline_mode`: base-scenario-only still runs optimization, just with one risk target.

**Sequential cutoff ordering:** `cutoff_floor_segment` (per-segment, in `segments.toml`) names the segment whose accepted cells constrain this segment, enforcing nested acceptance masks across segments (e.g., `mask_ef ⊆ mask_cd ⊆ mask_ab`). `cutoff_ordering_mode` (`"bottom_up"` / `"top_down"`, default `"bottom_up"`) controls the optimization direction: bottom-up optimizes the tightest segment first and propagates floor constraints (must-accept); top-down optimizes the least restrictive first and propagates ceiling constraints (must-reject). Segments are automatically topologically sorted by dependency. In parallel mode, constrained segments run sequentially after unconstrained ones complete.

## Testing

Tests use pytest with synthetic DataFrames (no real data needed). Fixtures use `numpy.random.RandomState(42)` for reproducibility. Optional deps (`shap`, `pandera`) are conditionally imported with skip guards. Coverage target: 80% on patch (codecov). CI runs lint → test → Docker build on push/PR to `main`.

## Ruff Configuration

Line length: 120. Target: Python 3.12. Rules enabled: E, F, W, I, UP, B, SIM, C4. Per-file ignores for line length and unused vars. Format uses double quotes and Unix line endings. isort configured with `src` as first-party.

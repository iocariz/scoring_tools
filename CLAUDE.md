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
uv run python main.py --skip-dq-checks         # skip data quality checks
uv run python main.py --baseline               # baseline: show current portfolio, no optimization

# Run batch (multi-segment, reads config.toml + segments.toml)
uv run python run_batch.py
uv run python run_batch.py --parallel --workers 4 -s segment1 segment2
uv run python run_batch.py --list               # list available segments
uv run python run_batch.py --reuse-models        # reuse existing supersegment models
uv run python run_batch.py --consolidate-only    # only generate consolidated report
uv run python run_batch.py --clean               # clean output dirs before running
uv run python run_batch.py --no-report           # skip HTML reports
uv run python run_batch.py --training-only       # only run DQ + training
uv run python run_batch.py --log-file batch.log  # capture all logs to file
uv run python run_batch.py --baseline           # baseline mode for all segments
uv run python run_batch.py --cutoff-ordering-mode bottom_up  # sequential cutoff ordering

# Analyze logs and suggest config improvements
uv run python analyze_logs.py batch.log                      # analyze a log file
uv run python analyze_logs.py output/*/logs/*.log --verbose  # analyze all segment logs
uv run python analyze_logs.py batch.log -o report.txt        # save report to file

# Run allocation (global MILP across segments)
uv run python run_allocation.py --target 2.5
uv run python run_allocation.py --target 2.5 --method greedy
uv run python run_allocation.py --target 2.5 --production-floor 1000000
uv run python run_allocation.py --target 2.5 --lock segment1:3

# Generate presentation (.pptx / .pdf)
uv run python generate_presentation.py
uv run python generate_presentation.py --pdf     # also convert to PDF (requires LibreOffice)

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
3. **Preprocessing** — `src/pipeline/preprocessing.py` orchestrates DQ checks (`src/data_quality.py`), filtering/binning (`src/preprocess_improved.py`). Bin edge learning supports `"quantile"` (equal-count) and `"optimization"` (production-weighted risk split via `DecisionTreeRegressor`) methods, dispatched by `BinConfig.method`.
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
  - Analysis: `mr_pipeline.py`, `stability.py`, `sensitivity.py`, `trends.py`, `alerts.py`, `audit.py`
  - Output: `consolidation.py`, `reporting.py`, `plots.py`, `styles.py`, `metrics.py`, `utils.py`
- Entry points: `main.py` (single segment), `run_batch.py` (multi-segment with global bin learning + supersegment model sharing), `run_allocation.py` (MILP allocation with segment constraints/locking), `generate_presentation.py` (PowerPoint/PDF output), `run_score_metrics.py` (score discriminance), `dashboard.py` / `interactive_allocator.py` (Dash web UIs), `gradio_dashboard.py` (Gradio web UI)

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

**Binning:** `octroi_bins`/`efx_bins` (legacy) or `[preprocessing.bins.*]` (N-variable `BinConfig` with `source_col`, `output_col`, `bin_edges`/`max_bins`, `method`). Batch mode supports global bin learning across all segments.

**Reject inference:** `reject_inference_method` ("none"/"parceling"), `reject_parceling_method` ("linear"/"power"/"sigmoid"), `reject_uplift_factor`, `reject_max_risk_multiplier`, `reject_bayesian_smoothing`, `reject_bayesian_prior_strength`, `reject_enforce_monotonicity`, `reject_include_all_rejections`. RI optimizer: `run_ri_optimizer`, `ri_validation_split`, `ri_optimizer_methods` ("grid"/"optuna"), `ri_optuna_n_trials`, `ri_calibration_gamma`, `ri_uplift_range`, `ri_max_mult_range`.

**H3 metrics:** `multiplier_h3` (scaling factor for 3-month risk), `use_mr_outcomes` (enable H3→H6 extrapolation), `mr_min_obs_per_bin`, `mr_extrapolation_method` (`"linear"` / `"power"` / `"logistic"` / `"auto"`), `mr_extrapolation_curvature` (power exponent; ignored when method is `"auto"` — curvature is fitted from main-period data via weighted log-log regression in `fit_h3_extrapolation_curve`), `mr_maturity_months` (min months since booking for H6 maturity filter), `mr_extrapolation_risk_multiplier` (relative ceiling for MR risk), `mr_extrapolation_hard_cap` (absolute ceiling for risk %).

**Stress & transformation:** `stress_mode` ("global"/"per_bin"/"disabled"), `per_bin_tasa_fin` (compute transformation rate per grid cell).

**Swap-in constraints:** `max_swapin_production_pct`, `max_swapin_risk`.

**Segment constraints** (in `segments.toml`): `min_risk`, `max_risk`, `min_production` (production floor), `locked_sol_fac` (lock to specific frontier point). Supersegments (`[supersegments.*]`) define named groups of segments. Each segment can independently reference a **modelling supersegment** (shared model training) and a **reporting supersegment** (consolidated report grouping) via `modelling_supersegment` and `reporting_supersegment` fields. Legacy `supersegment` field sets both. Resolution: `modelling_supersegment > supersegment > None`, `reporting_supersegment > supersegment > None`.

**Fixed cutoffs:** `fixed_cutoffs` to skip MILP and use predefined cutoff combinations. For 2-var: paired bins/cutoffs lists. For N>2: per-variable lists of accepted bin values (cell accepted iff all coordinates are in their respective accepted lists).

**Baseline mode:** `baseline_mode` (bool, default false) — show current booked portfolio as-is with no cutoff optimization (Optimum = Actual, zero swap-in/swap-out). MR inference still runs to predict risk for immature loans. Only the base scenario is generated; sensitivity and RI optimizer are skipped. Available via config or `--baseline` CLI flag.

**Sequential cutoff ordering:** `cutoff_floor_segment` (per-segment, in `segments.toml`) names the segment whose accepted cells constrain this segment, enforcing nested acceptance masks across segments (e.g., `mask_ef ⊆ mask_cd ⊆ mask_ab`). `cutoff_ordering_mode` (`"bottom_up"` / `"top_down"`, default `"bottom_up"`) controls the optimization direction: bottom-up optimizes the tightest segment first and propagates floor constraints (must-accept); top-down optimizes the least restrictive first and propagates ceiling constraints (must-reject). Segments are automatically topologically sorted by dependency. In parallel mode, constrained segments run sequentially after unconstrained ones complete.

## Testing

Tests use pytest with synthetic DataFrames (no real data needed). Fixtures use `numpy.random.RandomState(42)` for reproducibility. Optional deps (`shap`, `pandera`) are conditionally imported with skip guards. Coverage target: 80% on patch (codecov). CI runs lint → test → Docker build on push/PR to `main`.

## Ruff Configuration

Line length: 120. Target: Python 3.12. Rules enabled: E, F, W, I, UP, B, SIM, C4. Per-file ignores for line length and unused vars. Format uses double quotes and Unix line endings. isort configured with `src` as first-party.

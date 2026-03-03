# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk scoring and portfolio optimization pipeline. Processes loan application data (SAS `.sas7bdat` files), trains risk models on an N-dimensional score grid (e.g., internal "octroi" bins × external "EFX" bins, optionally × income bins), and finds optimal acceptance cutoffs via MILP with monotonicity constraints over a Pareto frontier of risk (`b2_ever_h6`) vs. production (`oa_amt_h0`). Supports H3 (3-month) early risk metrics alongside H6, hybrid MR extrapolation, advanced reject inference with Bayesian smoothing, and multi-segment batch runs with global bin learning and per-segment constraints.

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

# Run batch (multi-segment, reads config.toml + segments.toml)
uv run python run_batch.py
uv run python run_batch.py --parallel --workers 4 -s segment1 segment2
uv run python run_batch.py --list               # list available segments

# Run allocation (global MILP across segments)
uv run python run_allocation.py
uv run python run_allocation.py --production-floor 1000000
uv run python run_allocation.py --lock segment1:3

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

Makefile shortcuts: `make run`, `make run-batch`, `make test`, `make lint`, `make format`, `make docker-build`, `make docker-run`.

## Architecture

### Pipeline Phases (main.py)

1. **Config** — `PreprocessingSettings` (Pydantic) loaded from `config.toml` via `from_toml()`
2. **Data Loading** — `src/data_manager.py` reads SAS files, standardizes columns (H3 columns optional)
3. **Preprocessing** — `src/pipeline/preprocessing.py` orchestrates DQ checks (`src/data_quality.py`), filtering/binning (`src/preprocess_improved.py`). Bin edge learning supports `"quantile"` (equal-count) and `"optimization"` (production-weighted risk split via `DecisionTreeRegressor`) methods, dispatched by `BinConfig.method`.
4. **Inference** — `src/pipeline/inference.py` orchestrates model training with CV across feature sets; custom sklearn estimators in `src/estimators.py` (`HurdleRegressor`, `TweedieGLM`). Trains on `inference_variables` (subset of `variables`), decoupled from the optimization grid. Optuna hyperparameter tuning available for tree-based models.
5. **Optimization** — `src/pipeline/optimization.py` generates all monotonic cutoff combinations, computes KPIs per solution, applies optional reject inference (parceling with linear/power/sigmoid methods, optional Bayesian smoothing), filters to Pareto frontier. Supports swap-in production/risk constraints. Fixed cutoffs and GA fallback are fully supported for N>2 variables.
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
- Entry points: `main.py` (single segment), `run_batch.py` (multi-segment with global bin learning), `run_allocation.py` (MILP allocation with segment constraints/locking), `run_score_metrics.py` (score discriminance), `dashboard.py` / `interactive_allocator.py` (Dash web UIs)

### Key Design Patterns

- **`OutputPaths` dataclass** — centralized path management; every pipeline phase receives an instance
- **`PreprocessingSettings` Pydantic model** — all config flows through this with field/model validators (≥ 2 variables, bin edges ≥ 2, date parsing, MR period pairing, range constraints)
- **`BinConfig` dataclass** — N-variable binning config (`source_col`, `output_col`, `bin_edges`/`max_bins`, `method`) replacing legacy `octroi_bins`/`efx_bins`
- **`SegmentConstraints` dataclass** — per-segment risk bounds (`min_risk`/`max_risk`), production floors (`min_production`), frontier locking (`locked_sol_fac`)
- **`StatusName` / `RejectReason` / `Columns` enums** in `src/constants.py` — centralized string constants across the codebase
- **Custom sklearn estimators** — `HurdleRegressor` and `TweedieGLM` implement full `BaseEstimator`/`RegressorMixin` interface
- **Chunked processing** in optimization — feasible solutions processed in memory-efficient chunks for the combinatorial N-D grid search

## Configuration

Two-tier config: `config.toml` (global defaults) overridden per-segment by `segments.toml`.

**Core fields:** `variables` (≥ 2 score names), `inference_variables` (subset for model training), date ranges, `optimum_risk`, `risk_step`, `multiplier`, `directions` (monotonicity hints per variable).

**Binning:** `octroi_bins`/`efx_bins` (legacy) or `[preprocessing.bins.*]` (N-variable `BinConfig` with `source_col`, `output_col`, `bin_edges`/`max_bins`, `method`). Batch mode supports global bin learning across all segments.

**Reject inference:** `reject_inference_method` ("none"/"parceling"), `reject_parceling_method` ("linear"/"power"/"sigmoid"), `reject_bayesian_smoothing`, `reject_bayesian_prior_strength`, `reject_enforce_monotonicity`, `ri_calibration_gamma`, `ri_optimizer_method` ("grid"/"optuna"), `ri_optuna_n_trials`.

**H3 metrics:** `multiplier_h3` (scaling factor for 3-month risk), `use_mr_outcomes` (enable H3→H6 extrapolation), `mr_min_obs_per_bin`.

**Swap-in constraints:** `max_swapin_production_pct`, `max_swapin_risk`.

**Segment constraints** (in `segments.toml`): `min_risk`, `max_risk`, `min_production` (production floor), `locked_sol_fac` (lock to specific frontier point).

**Fixed cutoffs:** `fixed_cutoffs` to skip MILP and use predefined cutoff combinations. For 2-var: paired bins/cutoffs lists. For N>2: per-variable lists of accepted bin values (cell accepted iff all coordinates are in their respective accepted lists).

## Testing

Tests use pytest with synthetic DataFrames (no real data needed). Fixtures use `numpy.random.RandomState(42)` for reproducibility. Optional deps (`shap`, `pandera`) are conditionally imported with skip guards. Coverage target: 80% on patch (codecov). CI runs lint → test → Docker build on push/PR to `main`.

## Ruff Configuration

Line length: 120. Target: Python 3.12. Rules enabled: E, F, W, I, UP, B, SIM, C4. Per-file ignores for line length and unused vars. Format uses double quotes and Unix line endings. isort configured with `src` as first-party.

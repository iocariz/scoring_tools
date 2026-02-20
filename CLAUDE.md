# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk scoring and portfolio optimization pipeline. Processes loan application data (SAS `.sas7bdat` files), trains risk models on a 2D score grid (e.g., internal "octroi" bins × external "EFX" bins), and finds optimal acceptance cutoffs via exhaustive monotonically-constrained search over a Pareto frontier of risk (`b2_ever_h6`) vs. production (`oa_amt_h0`).

## Commands

```bash
# Install
uv pip install -e .

# Run pipeline (single segment)
uv run python main.py

# Run batch (multi-segment, reads config.toml + segments.toml)
uv run python run_batch.py
uv run python run_batch.py --parallel --workers 4 -s segment1 segment2

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
2. **Data Loading** — `src/data_manager.py` reads SAS files, standardizes columns
3. **Preprocessing** — `src/pipeline/preprocessing.py` orchestrates DQ checks (`src/data_quality.py`), filtering/binning (`src/preprocess_improved.py`)
4. **Inference** — `src/pipeline/inference.py` orchestrates model training with 5-fold CV across feature sets; custom sklearn estimators in `src/estimators.py` (`HurdleRegressor`, `TweedieGLM`)
5. **Optimization** — `src/pipeline/optimization.py` generates all monotonic cutoff combinations, computes KPIs per solution, applies optional reject inference (parceling), filters to Pareto frontier
6. **Scenario Analysis** — selects optimal Pareto points at pessimistic/base/optimistic risk thresholds; bootstrap CI, MR validation, PSI/CSI stability, audit tables
7. **Trend Analysis** — monthly metrics with SPC anomaly detection

### Module Layout

- `src/pipeline/` — thin orchestration wrappers (`config_loader.py`, `preprocessing.py`, `inference.py`, `optimization.py`) that coordinate the core modules
- `src/` — core logic: `config.py`, `constants.py`, `data_manager.py`, `data_quality.py`, `preprocess_improved.py`, `inference_optimized.py`, `optimization_utils.py`, `reject_inference.py`, `mr_pipeline.py`, `stability.py`, `trends.py`, `audit.py`, `consolidation.py`, `global_optimizer.py`, `metrics.py`, `plots.py`, `utils.py`
- Entry points: `main.py` (single segment), `run_batch.py` (multi-segment), `run_allocation.py` (MILP allocation), `run_score_metrics.py` (score discriminance), `dashboard.py` / `interactive_allocator.py` (Dash web UIs)

### Key Design Patterns

- **`OutputPaths` dataclass** — centralized path management; every pipeline phase receives an instance
- **`PreprocessingSettings` Pydantic model** — all config flows through this with field/model validators (exact 2 variables, bin edges ≥ 2, date parsing, MR period pairing, range constraints)
- **`StatusName` / `RejectReason` / `Columns` enums** in `src/constants.py` — centralized string constants across the codebase
- **Custom sklearn estimators** — `HurdleRegressor` and `TweedieGLM` implement full `BaseEstimator`/`RegressorMixin` interface
- **Chunked processing** in optimization — feasible solutions processed in memory-efficient chunks for the combinatorial 2D grid search

## Configuration

Two-tier config: `config.toml` (global defaults) overridden per-segment by `segments.toml`. Key fields: `variables` (exactly 2 score names), `octroi_bins`/`efx_bins` (bin edges), date ranges, `optimum_risk`, `risk_step`, `multiplier`, `reject_inference_method` ("none"/"parceling"), `fixed_cutoffs`, `inv_var1`, `cz_config`.

## Testing

Tests use pytest with synthetic DataFrames (no real data needed). Fixtures use `numpy.random.RandomState(42)` for reproducibility. Optional deps (`shap`, `pandera`) are conditionally imported with skip guards. Coverage target: 80% on patch (codecov). CI runs lint → test → Docker build on push/PR to `main`.

## Ruff Configuration

Line length: 120. Target: Python 3.12. Rules enabled: E, F, W, I, UP, B, SIM, N. Per-file ignores exist for `tests/` (E501, F841) and `__init__.py` (F401). Format uses double quotes and Unix line endings.

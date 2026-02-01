# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Credit risk scoring and portfolio optimization system for loan applications. Processes loan application data (demand/booked), calculates risk indicators, determines optimal cutoffs to maximize production while minimizing risk, and includes scenario analysis with MR (Recent Monitoring) period validation.

## Development Commands

### Installation & Setup
```bash
# Install dependencies (using uv package manager)
uv pip install -e .
```

### Running the Pipeline
```bash
# Single segment (uses config.toml)
uv run python main.py

# Multiple segments in batch (uses segments.toml)
uv run python run_batch.py

# Run specific segments
uv run python run_batch.py -s no_premium_ab premium

# Parallel execution
uv run python run_batch.py --parallel --workers 4

# Reuse existing supersegment models (skip retraining)
uv run python run_batch.py --reuse-models

# Clean output directories before running
uv run python run_batch.py --clean

# Skip data quality checks (not recommended)
uv run python run_batch.py --skip-dq-checks

# Generate only consolidated report (segments already processed)
uv run python run_batch.py --consolidate-only
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file with verbose output
uv run pytest tests/test_mr_pipeline.py -v

# Run specific test
uv run pytest tests/test_utils.py::test_calculate_b2_ever_h6 -v
```

### Interactive Dashboard
```bash
# Launch interactive dashboard for exploring results
uv run python dashboard.py
```

## Core Architecture

### Pipeline Flow (main.py)
The main pipeline executes in this order:
1. **Data Quality Checks** (`src/data_quality.py`): Validates data integrity before processing (fails fast on bad data)
2. **Data Loading & Preprocessing** (`src/preprocess_improved.py`): Loads SAS data, filters by segment/date, applies score binning to create cluster variables `sc_octroi_new_clus` and `new_efx_clus`
3. **Risk Inference** (`src/inference_optimized.py`): Trains `HurdleRegressor` models for zero-inflated data to predict `todu_30ever_h6` risk indicator
4. **Todu Average Inference**: Fits linear model for `todu_amt_pile_h6` prediction
5. **Optimization Pipeline** (`src/utils.py`): Generates all feasible cutoff solutions (with monotonicity constraints), calculates KPIs using joblib parallelization, finds Pareto-optimal solutions
6. **Scenario Analysis**: Runs multiple risk thresholds (`optimum_risk` ± `scenario_step`) with corresponding MR period validation
7. **MR Period Processing** (`src/mr_pipeline.py`): Validates strategy on recent data, generates swap-in/swap-out analysis
8. **Stability Analysis** (`src/stability.py`): Calculates PSI/CSI metrics comparing Main vs MR periods

### Key Modules

- **`src/preprocess_improved.py`**: `PreprocessingConfig` dataclass, `complete_preprocessing_pipeline()` returns `(data_clean, data_booked, data_demand)`. Supports regex patterns in `segment_filter` for supersegments (using `|` as OR operator)
- **`src/inference_optimized.py`**: `HurdleRegressor` class for zero-inflated regression, `inference_pipeline()` for model training, `run_optimization_pipeline()` for aggregation
- **`src/utils.py`**: `get_fact_sol()` generates feasible solutions with monotonicity constraints using `itertools.combinations_with_replacement`, `kpi_of_fact_sol()` and `get_optimal_solutions()` use joblib parallelization for performance
- **`src/mr_pipeline.py`**: `process_mr_period()` applies Main Period cutoffs to MR data, `calculate_metrics_from_cuts()` computes swap-in/swap-out analysis
- **`src/data_quality.py`**: `run_data_quality_checks()` validates data before processing (required columns, missing values, segment existence, date ranges, numeric outliers, etc.)
- **`src/stability.py`**: `compare_main_vs_mr()` generates PSI/CSI stability reports, `calculate_psi()` computes Population Stability Index
- **`src/consolidation.py`**: `consolidate_segments()` aggregates metrics across segments, supersegments, and total portfolio
- **`src/plots.py`**: `RiskProductionVisualizer` class for interactive Pareto frontier dashboards, `plot_risk_vs_production()` for time series
- **`src/models.py`**: `calculate_risk_values()`, tree-based and clustering optimal splits
- **`src/metrics.py`**: Gini, KS, PSI calculations, bootstrap confidence intervals
- **`src/constants.py`**: Centralized enums and constants (`StatusName.BOOKED`, `Columns.TODU_30EVER_H6`, etc.)
- **`src/styles.py`**: Unified color constants, `apply_plotly_style()` and `apply_matplotlib_style()` helpers
- **`src/persistence.py`**: Model saving/loading with metadata (features, hyperparameters, training date)
- **`src/estimators.py`**: Custom estimator implementations like `HurdleRegressor`

### Configuration System

All pipeline parameters are defined in TOML configuration files:

#### `config.toml` (Base Configuration)
- **Date ranges**: `date_ini_book_obs`, `date_fin_book_obs` (Main period), `date_ini_book_obs_mr`, `date_fin_book_obs_mr` (MR period)
- **Score binning**: `octroi_bins`, `efx_bins` (bin edges for creating cluster variables)
- **Variables**: `variables = ['sc_octroi_new_clus', 'new_efx_clus']` (2D grid for optimization)
- **Indicators**: Columns for aggregation (e.g., `["acct_booked_h0", "oa_amt", "todu_30ever_h6"]`)
- **Scenario config**: `optimum_risk` (base risk threshold), `scenario_step` (step size for scenario analysis)
- **CZ config**: `[preprocessing.cz_config]` contains comfort zone limits by year

#### `segments.toml` (Segment & Supersegment Configuration)
Defines individual segments and supersegments (shared model training):
- **Supersegments**: `[supersegments.NAME]` with `segment_filters` list combines multiple segments for model training
- **Segments**: `[segments.NAME]` with `segment_filter` string and optional overrides (`optimum_risk`, `scenario_step`)
- **Shared models**: Segments with `supersegment = "NAME"` use the shared model trained on combined data

### Key Concepts

- **b2_ever_h6**: Primary risk metric = `7 * todu_30ever_h6 / todu_amt_pile_h6`
- **Cluster variables**: `sc_octroi_new_clus` and `new_efx_clus` created from score binning, used as 2D grid for optimization
- **Feasible Solutions**: All valid cutoff combinations across cluster bins with monotonicity constraint (cuts must be non-decreasing or non-increasing depending on direction)
- **Pareto Optimal**: Solutions on efficient frontier (best production for given risk level)
- **Swap-in**: Rejected applications that would pass optimal cuts (opportunity)
- **Swap-out**: Booked applications that would fail optimal cuts (risk avoidance)
- **Supersegments**: Combine multiple segments for shared model training, then run optimization individually (useful when segments share similar risk characteristics)
- **Scenario Analysis**: Tests strategy robustness by running optimization at multiple risk thresholds (`optimum_risk` ± `scenario_step`)
- **MR Period**: Recent Monitoring period used to validate strategy on newer data
- **PSI/CSI**: Population/Characteristic Stability Index for detecting distribution drift (PSI < 0.1 = stable, 0.1-0.25 = moderate, ≥0.25 = unstable)

## Output Artifacts

### Per-Segment Outputs (in `output/SEGMENT_NAME/`)
- `data/optimal_solution_{scenario}.csv`: Selected optimal cutoffs per scenario
- `data/risk_production_summary_table_{scenario}.csv`: Impact metrics for Main Period
- `data/risk_production_summary_table_mr_{scenario}.csv`: Impact metrics for MR Period
- `data/stability_psi_{scenario}.csv`: PSI/CSI stability metrics (Main vs MR)
- `images/risk_production_visualizer_{scenario}.html`: Interactive Pareto frontier dashboard
- `images/stability_report_{scenario}.html`: Interactive stability dashboard with PSI charts
- `images/b2_ever_h6_vs_octroi_and_risk_score_{scenario}.html`: 3D risk surface visualization
- `models/model_YYYYMMDD_HHMMSS/`: Saved model artifacts with metadata.json and features.txt

### Consolidated Outputs (in `output/`)
- `consolidated_risk_production.csv`: Aggregated metrics by segment, supersegment, and total portfolio
- `consolidated_risk_production.html`: Interactive consolidated dashboard

### Supersegment Outputs (in `output/_supersegment_NAME/`)
When segments share a supersegment, the shared model is saved here and reused by all member segments.

## Important Implementation Details

### Parallel Processing
- `kpi_of_fact_sol()` and `get_optimal_solutions()` use joblib for parallel chunk processing
- Functions called by joblib must be picklable (top-level functions, not lambdas or nested functions)
- Batch processing (`run_batch.py`) can run segments in parallel with `--parallel` flag

### Score Binning & Variables
- Score binning creates cluster variables from continuous scores using `pd.cut()` with bin edges from config
- The system currently assumes 2D grid optimization (`sc_octroi_new_clus`, `new_efx_clus`) - generalizing to N-dimensions would require architectural changes
- Bin indices (1-9) are used as column names in feasible solutions DataFrame

### Data Quality Validation
- Data quality checks run automatically before processing (can be skipped with `--skip-dq-checks`)
- Failures stop the pipeline immediately; warnings allow continuation but are logged
- Checks include: required columns, missing values, segment existence, segment size, date ranges, numeric outliers, indicator values, booked ratio, duplicates

### Supersegment Model Sharing
- When multiple segments belong to a supersegment, the model is trained once on combined data and saved in `output/_supersegment_NAME/`
- Individual segments load the shared model for inference but run optimization independently
- Use `--reuse-models` flag to skip retraining existing supersegment models

### Segment Filter Patterns
- Single segment: exact match (e.g., `"personal_loans,_fusion_and_tj_direct-no_premium-a-b"`)
- Supersegment: regex pattern with `|` as OR operator (e.g., `"personal_loans.*no_premium-a-b|personal_loans.*no_premium-c-d"`)

### HurdleRegressor
- Custom estimator for zero-inflated data (common in credit risk where many applications have zero risk)
- Two-stage model: (1) logistic classifier for zero/non-zero, (2) regressor for non-zero values
- Supports any scikit-learn compatible base estimators

### Memory Optimization
- `optimize_dtypes()` reduces memory usage by downcasting int64/float64 to smaller types
- Chunked processing in KPI calculation to handle large solution spaces
- Garbage collection (`gc.collect()`) after memory-intensive operations

## Common Development Patterns

### Adding a New Segment
1. Add segment definition to `segments.toml` under `[segments.NAME]`
2. Specify `segment_filter` and optional overrides (`optimum_risk`, `scenario_step`)
3. Optionally assign to supersegment with `supersegment = "supersegment_name"`
4. Run with `uv run python run_batch.py -s NAME`

### Adding a New Scenario
Scenarios are automatically generated from `optimum_risk` ± `scenario_step` in config. To change:
1. Modify `optimum_risk` (base value) in config.toml or segments.toml
2. Modify `scenario_step` (increment size) to control number of scenarios

### Modifying Score Binning
1. Update `octroi_bins` and/or `efx_bins` in config.toml
2. Bin edges should include `-inf` and `inf` as first/last values
3. Bins create cluster variables used for optimization grid

### Debugging Failed Optimizations
1. Check `output/SEGMENT_NAME/logs/` for error logs
2. Verify data quality passed all checks (or rerun without `--skip-dq-checks`)
3. Check that segment_filter matches actual data in `segment_cut_off` column
4. Ensure sufficient data exists for both Main and MR periods

### Testing Changes
- Add unit tests in `tests/` following existing patterns
- Use pytest fixtures for common test data
- Test modules: `test_preprocessing.py`, `test_mr_pipeline.py`, `test_utils.py`, `test_validation.py`, `test_consolidation.py`

## Dependencies

Python 3.12+ required. Key dependencies:
- **Data**: pandas, numpy, scipy
- **ML**: scikit-learn, joblib (parallel processing)
- **Visualization**: plotly, matplotlib, seaborn, dash, ipywidgets
- **Logging**: loguru, colorama
- **Config**: tomllib (builtin), tomli-w
- **Testing**: pytest
- **Interactive**: ipython, tqdm

Package manager: `uv` (modern Python package installer)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a credit risk scoring and portfolio optimization system. It processes loan application data (demand/booked), calculates risk indicators (e.g., `b2_ever_h6`), and determines optimal cutoffs to maximize production while minimizing risk. The system includes scenario analysis and MR (Recent Monitoring) period validation.

## Commands

### Run the main pipeline
```bash
uv run python main.py
```

### Run tests
```bash
uv run pytest tests/
uv run pytest tests/test_mr_pipeline.py -v  # Single test file
```

### Install dependencies
```bash
uv pip install -e .
```

## Architecture

### Core Pipeline Flow (main.py)
1. **Data Loading & Preprocessing** (`src/preprocess_improved.py`): Loads SAS data, filters by segment/date, applies score binning to create `sc_octroi_new_clus` and `new_efx_clus` clusters
2. **Risk Inference** (`src/inference_optimized.py`): Trains models (including Hurdle regression for zero-inflated data) to predict `todu_30ever_h6` risk
3. **Todu Average Inference**: Fits linear model for `todu_amt_pile_h6` prediction
4. **Optimization Pipeline**: Generates all feasible cutoff solutions, calculates KPIs, finds Pareto-optimal solutions
5. **Scenario Analysis**: Runs multiple risk thresholds (base +/- 0.1) with corresponding MR period validation
6. **MR Period Processing** (`src/mr_pipeline.py`): Validates strategy on recent data, generates swap-in/swap-out analysis

### Key Modules

- **`src/preprocess_improved.py`**: `PreprocessingConfig` dataclass, `complete_preprocessing_pipeline()` returns `(data_clean, data_booked, data_demand)`
- **`src/inference_optimized.py`**: `HurdleRegressor` class for zero-inflated regression, `inference_pipeline()` for model training, `run_optimization_pipeline()` for aggregation
- **`src/utils.py`**: `get_fact_sol()` generates feasible solutions, `kpi_of_fact_sol()` and `get_optimal_solutions()` use joblib parallelization
- **`src/mr_pipeline.py`**: `process_mr_period()` applies Main Period cutoffs to MR data, `calculate_metrics_from_cuts()` computes swap analysis
- **`src/plots.py`**: `RiskProductionVisualizer` class for interactive optimization dashboards, `plot_risk_vs_production()` for time series
- **`src/models.py`**: `calculate_risk_values()`, tree-based and clustering optimal splits
- **`src/metrics.py`**: Gini, KS, PSI calculations, bootstrap confidence intervals
- **`src/styles.py`**: Unified color constants and `apply_plotly_style()`/`apply_matplotlib_style()` helpers

### Configuration

All pipeline parameters are in `config.toml`:
- Date ranges (`date_ini_book_obs`, `date_fin_book_obs`, `date_ini_book_obs_mr`, `date_fin_book_obs_mr`)
- Score binning (`octroi_bins`, `efx_bins`)
- Variables (`variables`: typically `['sc_octroi_new_clus', 'new_efx_clus']`)
- Indicators (`indicators`: columns for aggregation)
- CZ config (`cz_config`: comfort zone limits by year)

### Key Concepts

- **b2_ever_h6**: Risk metric = `7 * todu_30ever_h6 / todu_amt_pile_h6`
- **Feasible Solutions**: All valid cutoff combinations across cluster bins (monotonic constraint)
- **Pareto Optimal**: Solutions on the efficient frontier (best production for given risk level)
- **Swap-in/Swap-out**: Analysis of rejected apps that would pass optimal cuts vs booked apps that would fail

## Output Artifacts

- `data/optimal_solution_{scenario}.csv`: Selected cutoffs per scenario
- `data/risk_production_summary_table_{scenario}.csv`: Impact metrics
- `images/risk_production_visualizer_{scenario}.html`: Interactive Pareto frontier dashboard
- `images/b2_ever_h6_vs_octroi_and_risk_score.html`: 3D risk surface
- `models/`: Saved model artifacts with metadata.json and features.txt

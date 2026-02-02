# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk scoring and portfolio optimization system. Processes loan application data (demand/booked), calculates risk indicators (e.g., `b2_ever_h6`), and determines optimal cutoffs to maximize production while minimizing risk. Includes scenario analysis, MR (Recent Monitoring) period validation, and multi-segment batch processing with supersegment support.

## Commands

```bash
# Install dependencies
uv pip install -e .

# Run single segment (uses config.toml)
uv run python main.py

# Run batch processing (uses segments.toml)
uv run python run_batch.py
uv run python run_batch.py -s no_premium_ab premium  # Specific segments
uv run python run_batch.py --parallel --workers 4     # Parallel execution
uv run python run_batch.py --list                     # List available segments

# Run tests
uv run pytest tests/
uv run pytest tests/test_mr_pipeline.py -v  # Single test file
```

## Architecture

### Core Pipeline Flow (main.py)
1. **Data Quality Checks** (`src/data_quality.py`): Validates data integrity before processing
2. **Preprocessing** (`src/preprocess_improved.py`): Loads SAS data, filters by segment/date, creates `sc_octroi_new_clus` and `new_efx_clus` clusters via score binning
3. **Risk Inference** (`src/inference_optimized.py`): Trains Hurdle regression for zero-inflated `todu_30ever_h6` prediction
4. **Optimization** (`src/utils.py`): Generates feasible cutoff solutions, calculates KPIs, finds Pareto-optimal solutions using joblib parallelization
5. **Scenario Analysis**: Runs multiple risk thresholds (base +/- 0.1) with MR period validation
6. **MR Processing** (`src/mr_pipeline.py`): Validates strategy on recent data, generates swap-in/swap-out analysis
7. **Stability Analysis** (`src/stability.py`): Calculates PSI/CSI comparing Main vs MR periods

### Batch Processing (run_batch.py)
- **Supersegments**: Train a single model on combined data from multiple segments, then run optimization individually. Defined in `segments.toml`.
- **Consolidation** (`src/consolidation.py`): Aggregates metrics across segments, supersegments, and total portfolio.

### Key Modules

| Module | Key Components |
|--------|----------------|
| `src/preprocess_improved.py` | `PreprocessingConfig` dataclass, `complete_preprocessing_pipeline()` |
| `src/inference_optimized.py` | `HurdleRegressor`, `inference_pipeline()`, `run_optimization_pipeline()` |
| `src/utils.py` | `get_fact_sol()`, `kpi_of_fact_sol()`, `get_optimal_solutions()` (joblib parallel) |
| `src/mr_pipeline.py` | `process_mr_period()`, `calculate_metrics_from_cuts()` |
| `src/stability.py` | `compare_main_vs_mr()`, `calculate_psi()` |
| `src/data_quality.py` | Data validation checks (missing values, segment size, outliers) |
| `src/plots.py` | `RiskProductionVisualizer` class for interactive dashboards |
| `src/metrics.py` | Gini, KS, PSI calculations, bootstrap confidence intervals |

### Configuration

**`config.toml`** - Base configuration:
- Date ranges (`date_ini_book_obs`, `date_fin_book_obs`, `date_ini_book_obs_mr`, `date_fin_book_obs_mr`)
- Score binning (`octroi_bins`, `efx_bins`)
- Variables (`variables`: typically `['sc_octroi_new_clus', 'new_efx_clus']`)
- Risk threshold (`optimum_risk`, `scenario_step`)
- Optional `fixed_cutoffs` section to skip optimization

**`segments.toml`** - Segment definitions for batch processing:
- Individual segment configs with optional overrides
- Supersegment definitions for shared model training

### Key Concepts

- **b2_ever_h6**: Risk metric = `7 * todu_30ever_h6 / todu_amt_pile_h6`
- **Feasible Solutions**: All valid cutoff combinations across cluster bins (monotonic constraint)
- **Pareto Optimal**: Solutions on the efficient frontier (best production for given risk level)
- **Swap-in/Swap-out**: Rejected apps that would pass optimal cuts vs booked apps that would fail
- **PSI (Population Stability Index)**: Distribution drift detection (<0.1 stable, 0.1-0.25 moderate, ≥0.25 unstable)

## Output Structure

```
output/
├── _supersegment_no_premium/    # Shared model for supersegment
│   └── models/
├── no_premium_ab/               # Per-segment outputs
│   ├── models/
│   ├── images/
│   │   ├── risk_production_visualizer_{scenario}.html
│   │   └── stability_report_{scenario}.html
│   └── data/
│       ├── optimal_solution_{scenario}.csv
│       ├── risk_production_summary_table_{scenario}.csv
│       └── stability_psi_{scenario}.csv
├── consolidated_risk_production.csv   # Portfolio-level aggregation
└── consolidated_risk_production.html  # Interactive dashboard
```

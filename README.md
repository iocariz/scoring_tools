# Scoring Optimization Tools

## Overview

This project implements a comprehensive risk scoring and portfolio optimization pipeline for credit portfolio management. It processes loan application data (demand and booked), calculates key risk indicators (e.g., `b2_ever_h6`), and determines optimal cutoffs to maximize production while minimizing risk.

Key features include:

- **Optimization**: Finds Pareto-optimal cutoffs on a 2D grid (`sc_octroi`, `efx`).
- **Scenario Analysis**: Tests sensitivity to risk thresholds (base ± step).
- **Recent Monitoring (MR)**: Validates strategy performance on recent data.
- **Stability Analysis**: Automatically calculates PSI/CSI to detect distribution drift.
- **Supersegments**: Supports training shared models across multiple segments.
- **Robust Configuration**: Type-safe configuration using Pydantic.

## Quick Start

### Installation

```bash
# Install dependencies using uv
uv pip install -e .
```

### Running the Pipeline

```bash
# Run single segment (uses config.toml)
uv run python main.py

# Run multiple segments in batch (uses segments.toml)
uv run python run_batch.py

# Run specific segments
uv run python run_batch.py -s no_premium_ab premium

# Run in parallel
uv run python run_batch.py --parallel --workers 4
```

## Architecture

### Core Pipeline Flow (`main.py`)

1. **Configuration**: Loads and validates settings using `src/config.py` (Pydantic).
2. **Data Loading**: Loads SAS data and standardizes columns via `src/data_manager.py`.
3. **Data Quality**: Validates data integrity (missing values, outliers, etc.) via `src/data_quality.py`.
4. **Preprocessing**: Filters by segment/date, creates cluster variables via `src/preprocess_improved.py`.
5. **Inference**:
    - **Risk**: Trains/loads `HurdleRegressor` for `todu_30ever_h6`.
    - **Amount**: Trains linear model for `todu_amt_pile_h6`.
6. **Optimization**: Generates feasible solutions (monotonic constraints) and finds Pareto frontier via `src/utils.py`.
7. **Scenario Analysis**: Iterates through risk thresholds (`optimum_risk` ± `scenario_step`).
8. **Validation**:
    - **MR Period**: Applies cutoffs to recent data (`src/mr_pipeline.py`).
    - **Stability**: Calculates PSI/CSI (`src/stability.py`).
9. **Reporting**: Generates HTML dashboards and CSV summaries.

### Key Modules

| Module | Description |
| :--- | :--- |
| `src/config.py` | Pydantic settings definition and validation. |
| `src/data_manager.py` | Centralized data loading and column validation. |
| `src/inference_optimized.py` | Training/inference logic for risk models. |
| `src/utils.py` | Optimization algorithms (Pareto frontier, solution generation). |
| `src/mr_pipeline.py` | Recent Monitoring (MR) validation logic. |
| `src/stability.py` | PSI/CSI stability metrics calculation. |
| `src/plots.py` | Interactive Plotly dashboards. |

## Configuration

The system uses **TOML** files for configuration, validated by Pydantic.

### `config.toml` (Base Settings)

Global settings for the pipeline.

```toml
[preprocessing]
# Segment to process
segment_filter = "personal_loans"

# Input Data
data_path = "data/demanda_direct_out.sas7bdat"

# Date Ranges (Main Analysis Period)
date_ini_book_obs = "2024-07-01"
date_fin_book_obs = "2025-06-01"

# Date Ranges (Recent Monitoring - MR)
date_ini_book_obs_mr = "2025-06-01"
date_fin_book_obs_mr = "2025-08-01"

# Optimization Grid Variables (must be 2)
variables = ["sc_octroi_new_clus", "new_efx_clus"]

# Columns to Keep & Indicators
keep_vars = ["mis_date", "status_name", ...]
indicators = ["acct_booked_h0", "oa_amt", ...]

# Binning Configuration (Edges for clustering)
# Note: Supports -inf/inf. Used to create the grid variables.
octroi_bins = [-inf, 364.0, 370.0, ..., inf]
efx_bins = [-inf, 2.0, 7.0, ..., inf]

# Economic Parameters
multiplier = 7.0         # Risk multiplier
z_threshold = 3.0       # Outlier threshold
optimum_risk = 1.1      # Target risk appetite (%)
scenario_step = 0.1     # Variance for scenario analysis (+/- step)

# Comfort Zone (Yearly Limits)
[preprocessing.cz_config]
2024 = 1.3
2025 = 1.2
```

### `segments.toml` (Batch Processing)

Defines multiple segments and overrides base configuration.

```toml
# Supersegment: Shared model training
[supersegments.no_premium]
segment_filters = [
    "personal_loans...no_premium-a",
    "personal_loans...no_premium-b"
]

# Individual Segment
[segments.no_premium_a]
segment_filter = "personal_loans...no_premium-a"
supersegment = "no_premium"  # Use shared model
optimum_risk = 1.2           # Override base setting
```

## Logic & Concepts

### Risk Metric: `b2_ever_h6`

The primary risk indicator is calculated as:
$$ \text{Risk} = \frac{7 \times \text{todu\_30ever\_h6}}{\text{todu\_amt\_pile\_h6}} $$

### Feasible Solutions

The optimizer generates all possible combinations of cutoffs on the 2D grid defined by `octroi_bins` and `efx_bins`. It enforces **monotonicity**: stricter scores must have strictly lower or equal acceptance.

### Scenario Analysis

To test robustness, the pipeline runs optimizations for multiple risk levels:

1. **Base**: `optimum_risk`
2. **Pessimistic**: `optimum_risk - scenario_step`
3. **Optimistic**: `optimum_risk + scenario_step`

For each scenario, it produces a full set of outputs (optimal solution, impact analysis, MR validation).

### Stability (PSI/CSI)

Automatically compares the distribution of key variables between the **Main** period and **MR** period.

- **PSI < 0.1**: Stable
- **PSI 0.1 - 0.25**: Moderate drift
- **PSI > 0.25**: Significant drift (Unstable)

## Outputs

Artifacts are saved in `output/SEGMENT_NAME/`:

| Path | Description |
| :--- | :--- |
| `data/optimal_solution_{scenario}.csv` | Selected cutoffs. |
| `data/risk_production_summary_{scenario}.csv` | Impact metrics (Production vs Risk). |
| `data/stability_psi_{scenario}.csv` | PSI stability report. |
| `images/risk_production_visualizer_{scenario}.html` | Interactive Pareto dashboard. |
| `images/stability_report_{scenario}.html` | Visual stability report. |
| `models/` | Saved models (if training occurred). |

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific validation tests
uv run pytest tests/test_validation.py -v
```

### Adding a New Segment

1. Add the segment definition to `segments.toml`.
2. Run `uv run python run_batch.py -s NEW_SEGMENT_NAME`.

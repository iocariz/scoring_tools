# Scoring Optimization Tools

## Overview

This project implements a risk scoring and optimization pipeline for credit portfolio management. It processes loan application data (`demand` and `booked`), calculates key risk indicators (e.g., `b2_ever_h6`), and determines optimal cutoffs to maximize production while minimizing risk.

Recent work has focused on integrating a **Recent Monitoring (MR)** period to validate model performance and strategy stability over a newer time window, as well as implementing **Scenario Analysis** to test sensitivity to risk thresholds.

## Quick Start

```bash
# Install dependencies
uv pip install -e .

# Run single segment (uses config.toml)
uv run python main.py

# Run multiple segments (uses segments.toml)
uv run python run_batch.py

# Run tests
uv run pytest tests/
```

## Batch Processing

The `run_batch.py` script enables processing multiple segments with different configurations, including support for **supersegments** (shared model training across related segments).

### Basic Usage

```bash
# List available segments and supersegments
uv run python run_batch.py --list

# Run all segments
uv run python run_batch.py

# Run specific segments
uv run python run_batch.py -s no_premium_ab no_premium_cd premium

# Run in parallel mode
uv run python run_batch.py --parallel --workers 4
```

### Supersegments (Shared Model Training)

Supersegments allow training a single inference model on combined data from multiple segments, then running optimization individually for each sub-segment. This is useful when segments share similar risk characteristics.

```bash
# Segments in a supersegment automatically use the shared model
uv run python run_batch.py -s no_premium_ab no_premium_cd no_premium_ef
# Result: 1 model trained on combined data, 3 separate optimizations
```

### Additional Options

```bash
# Reuse existing supersegment models (skip retraining)
uv run python run_batch.py --reuse-models

# Clean output directories before running
uv run python run_batch.py --clean

# Clean only (don't run pipeline)
uv run python run_batch.py --clean-only

# Clean specific segments
uv run python run_batch.py -s premium --clean-only

# Skip data quality checks (not recommended for production)
uv run python run_batch.py --skip-dq-checks
```

### Configuration Files

#### `config.toml` - Base Configuration

Contains default settings for all segments:

```toml
[preprocessing]
keep_vars = ["mis_date", "status_name", "risk_score_rf", ...]
indicators = ["acct_booked_h0", "oa_amt", "todu_30ever_h6", ...]
segment_filter = "personal_loans,_fusion_and_tj_direct-no_premium-a-b"
octroi_bins = [-inf, 364.0, 370.0, ...]
efx_bins = [-inf, 2.0, 7.0, ...]
date_ini_book_obs = "2024-07-01"
date_fin_book_obs = "2025-06-01"
optimum_risk = 1.1
```

#### `segments.toml` - Segment Definitions

Defines segments and supersegments with optional overrides:

```toml
# Supersegment: combines multiple segments for shared model training
[supersegments.no_premium]
segment_filters = [
    "personal_loans,_fusion_and_tj_direct-no_premium-a-b",
    "personal_loans,_fusion_and_tj_direct-no_premium-c-d",
    "personal_loans,_fusion_and_tj_direct-no_premium-e-f"
]

# Individual segments
[segments.no_premium_ab]
segment_filter = "personal_loans,_fusion_and_tj_direct-no_premium-a-b"
optimum_risk = 1.1
supersegment = "no_premium"  # Uses shared model

[segments.premium]
segment_filter = "personal_loans,_fusion_and_tj_direct-premium-"
optimum_risk = 0.9
# No supersegment = trains its own model
```

### Output Structure

```
output/
├── _supersegment_no_premium/    # Shared model for supersegment
│   ├── models/
│   │   └── model_YYYYMMDD_HHMMSS/
│   ├── images/
│   ├── data/
│   └── logs/
├── no_premium_ab/               # Uses shared model, runs optimization
│   ├── models/
│   ├── images/
│   ├── data/
│   └── logs/
├── no_premium_cd/
├── no_premium_ef/
└── premium/                     # Individual model + optimization
```

## Consolidated Reports

After processing all segments, the pipeline automatically generates a **consolidated risk production report** that aggregates metrics across:

- Individual segments
- Supersegments (combined segments)
- Total portfolio
- Main and MR periods
- All scenarios

### Generated Files

| File | Description |
|------|-------------|
| `output/consolidated_risk_production.csv` | Full data with all metrics |
| `output/consolidated_risk_production.html` | Interactive dashboard |

### Sample Console Output

```
================================================================================
CONSOLIDATED RISK PRODUCTION SUMMARY
================================================================================

────────────────────────────────────────
SCENARIO: 1.1
────────────────────────────────────────

  MAIN Period:
    Actual Production:  €45,234,567
    Optimum Production: €48,765,432
    Delta:              €3,530,865 (+7.8%)
    Risk:               1.12% → 1.08%

  MR Period:
    Actual Production:  €12,456,789
    Optimum Production: €13,234,567
    Delta:              €777,778 (+6.2%)
    Risk:               1.15% → 1.10%

  By Supersegment (Main Period):
    no_premium: €32,123,456 → €34,567,890 (+7.6%)
    premium:    €13,111,111 → €14,197,542 (+8.3%)

================================================================================
```

### Usage

```bash
# Run segments and generate consolidated report (default)
uv run python run_batch.py

# Skip consolidation
uv run python run_batch.py --no-consolidation

# Only generate consolidated report (segments already processed)
uv run python run_batch.py --consolidate-only
```

### Metrics Included

| Metric | Description |
|--------|-------------|
| `actual_production` | Current portfolio production |
| `actual_risk` | Current portfolio risk |
| `optimum_production` | Production with optimal cutoffs |
| `optimum_risk` | Risk with optimal cutoffs |
| `production_delta` | Optimum - Actual production |
| `production_delta_pct` | Percentage change |
| `swap_in_*` | Rejected applications that pass optimal cuts |
| `swap_out_*` | Booked applications that fail optimal cuts |

## Stability Metrics (PSI/CSI)

The pipeline automatically calculates **Population Stability Index (PSI)** and **Characteristic Stability Index (CSI)** when comparing Main vs MR periods. This detects distribution drift that could indicate model degradation.

### PSI Interpretation

| PSI Value | Status | Action |
|-----------|--------|--------|
| < 0.1 | Stable (green) | No action needed |
| 0.1 - 0.25 | Moderate (yellow) | Investigation recommended |
| ≥ 0.25 | Unstable (red) | Action required |

### Formula

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Where:
- **Expected%** = proportion in each bin for baseline (Main period)
- **Actual%** = proportion in each bin for comparison (MR period)

### Sample Report Output

```
======================================================================
STABILITY REPORT (PSI/CSI)
======================================================================

Baseline:   Main Period (n=45,234)
Comparison: MR Period (n=12,456)

Overall PSI: 0.0823 ✓ (stable)

UNSTABLE (PSI ≥ 0.25):
  ✗ risk_score_rf: PSI=0.3124 (unstable)

MODERATE (0.1 ≤ PSI < 0.25):
  ⚠ sc_octroi: PSI=0.1456 (moderate)
  ⚠ new_efx: PSI=0.1203 (moderate)

STABLE (PSI < 0.1):
  ✓ oa_amt: PSI=0.0234 (stable)
  ✓ todu_amt_pile_h6: PSI=0.0156 (stable)
  ... (8 more)

----------------------------------------------------------------------
Stability: 8/12 stable, 3 moderate, 1 unstable
======================================================================
```

### Output Files

| File | Description |
|------|-------------|
| `images/stability_report_{scenario}.html` | Interactive dashboard with PSI charts |
| `data/stability_psi_{scenario}.csv` | PSI values for all variables |

### Programmatic Usage

```python
from src.stability import compare_main_vs_mr, calculate_psi

# Compare two DataFrames
report = compare_main_vs_mr(
    main_df=data_main,
    mr_df=data_mr,
    variables=['sc_octroi', 'new_efx', 'oa_amt'],
    score_variable='sc_octroi',
    output_path='stability.html'
)

# Check stability
if not report.is_stable:
    print(f"Warning: {len(report.unstable_vars)} unstable variables")

# Calculate PSI for a single variable
psi_value, breakdown = calculate_psi(
    baseline=data_main['sc_octroi'],
    comparison=data_mr['sc_octroi'],
    bins=10
)
```

## Data Quality Checks

The pipeline includes automatic data quality validation that runs before processing. This prevents wasted compute time on bad data and catches common issues early.

### Checks Performed

| Check | Description | Failure Threshold |
|-------|-------------|-------------------|
| **Required Columns** | Verifies all needed columns exist | Any column missing |
| **Missing Values** | Checks for nulls in key columns | >20% missing (fail), >5% (warn) |
| **Segment Exists** | Validates segment_filter matches data | No matching rows |
| **Segment Size** | Ensures enough data for modeling | <100 rows (fail), <1000 (warn) |
| **Date Range** | Verifies data covers expected period | Gaps in date coverage |
| **Numeric Outliers** | Detects extreme values (z-score > 5) | >1% outliers |
| **Indicator Values** | Checks for negative counts/amounts | Any negative values |
| **Booked Ratio** | Validates booked vs total applications | <1% booked (fail), <5% (warn) |
| **Duplicate Rows** | Checks for duplicate records | >1% duplicates |

### Sample Report Output

```
============================================================
DATA QUALITY REPORT
============================================================

FAILURES:
  ✗ Segment Filter: No data matches segment filter 'invalid_segment'
    - available_segments (top 10): {'premium': 15000, 'no_premium': 25000}

WARNINGS:
  ⚠ Missing Values: Some missing values detected (>5%)
    - risk_score_rf: 7.2% (1,234 rows)
  ⚠ Date Range: Data starts 15 days after expected start
    - expected_range: 2024-07-01 to 2025-06-01
    - actual_range: 2024-07-16 to 2025-06-01

PASSED:
  ✓ Required Columns: All 25 required columns present
  ✓ Segment Size: Segment has 15,234 rows
  ✓ Numeric Outliers: No severe outliers in 8 numeric columns
  ✓ Indicator Values: All 12 indicators have valid values
  ✓ Booked Ratio: Booked ratio: 45.2% (6,875 of 15,234)
  ✓ Duplicate Rows: No significant duplicates across all columns

------------------------------------------------------------
Data Quality: 6/9 passed, 2 warnings, 1 failures
============================================================
```

### Handling Failures

- **Failures** stop the pipeline immediately. Fix the underlying data issue before re-running.
- **Warnings** allow the pipeline to continue but are logged for review.
- Use `--skip-dq-checks` to bypass validation (not recommended for production).

## Workflow & Steps Followed

### 1. Data Loading & Validation

* **Data Loading**: SAS datasets are loaded and standardized (column names, types).
* **Data Quality Checks**: Automatic validation of data integrity (see [Data Quality Checks](#data-quality-checks) section).
* **Filtering**: Data is filtered by date ranges (`date_ini_book_obs`, `date_fin_book_obs`) and specific segments.

### 2. Data Preprocessing & Initial Analysis

* **Transformation Rate**: Calculated the conversion from eligible applications to booked loans.
* **Feature Engineering**: Created risk indicators like `todu_amt_pile_h6` and `todu_30ever_h6`.
* **Score Binning**: Creates cluster variables (`sc_octroi_new_clus`, `new_efx_clus`) from score bins.

### 3. Risk Model Inference

* **Modeling**: A Hurdle-Ridge regression model is trained to predict `todu_30ever_h6` (risk) based on loan characteristics (`oa_amt`, etc.).
* **Inference**: The model fills in missing risk values for rejected applications (repesca), allowing for a theoretical "what-if" analysis on the total population.
* **Supersegment Support**: Models can be trained on combined data from multiple segments for better generalization.

### 4. Optimization Pipeline (Main Period)

* **Aggregation**: Data is aggregated by clusters (e.g., `sc_octroi_new_clus`, `new_efx_clus`).
* **Feasible Solutions**: The system iterates through all possible cutoff combinations across the cluster variables.
* **Parallel Processing**: Computationally intensive steps (`kpi_of_fact_sol`, `get_optimal_solutions`) are parallelized using `joblib` to maximize performance.
* **KPI Calculation**: For each solution, it calculates resulting Production, Risk (%), and Efficiency.
* **Optimal Solution Selection**: Identifies the Pareto frontier of solutions that offer the best trade-off between risk and production.

### 5. Scenario Analysis

To test the robustness of the strategy against varying risk appetites, the pipeline implements scenario analysis:

* **Variable**: `optimum_risk` (formerly `cz2024`).
* **Scenarios**: Automatically runs for the base configuration value +/- 0.1 (e.g., 1.0, 1.1, 1.2).
* **Outputs**: Generates distinct summary tables and optimal solution files for each scenario (e.g., `data/optimal_solution_1.1.csv`).

### 6. MR Period Implementation (Recent Monitoring)

A specific pipeline was built to validate the strategy on the MR ("Mois Récent") dataset. This pipeline is now integrated into the **Scenario Analysis** loop:

* **Process**: For *each* scenario (e.g., 1.0, 1.1, 1.2), the MR analysis is executed using that scenario's specific optimal solution.
* **Outputs**:
  * `data/risk_production_summary_table_mr_{scenario}.csv`: Impact analysis for that specific risk scenario on recent data.
  * `images/b2_ever_h6_vs_octroi_and_risk_score_mr_{scenario}.html`: Visualization of the risk surface.

### 7. Risk Production Summary Table (MR)

We generated a summary table to quantify the impact of the Main Period's optimal strategy on the MR data:

1. **Cut Extraction**: Loaded the specific cutoffs (e.g., `Bin 1 <= 4.0`) from the optimal solution found in the Main Period.
2. **Simulation**: Applied these cuts to the MR dataset.
3. **Metrics**:
    * **Actual**: The baseline risk/production of the portfolio as it was booked.
    * **Swap-in**: Rejected applicants in MR that *passed* the optimal cuts (potential opportunity).
    * **Swap-out**: Booked applicants in MR that *failed* the optimal cuts (risk avoidance).
    * **Optimum**: The theoretical portfolio (Actual - Swap-out + Swap-in).

## Key Artifacts

### Per-Segment Outputs

| Artifact | Description |
| :--- | :--- |
| `data/optimal_solution_{scenario}.csv` | The selected optimal strategy (cutoffs) for a specific risk scenario. |
| `data/risk_production_summary_table_{scenario}.csv` | Impact analysis for the Main Period (per scenario). |
| `data/risk_production_summary_table_mr_{scenario}.csv` | Impact analysis applying the strategy to the MR Period (per scenario). |
| `data/stability_psi_{scenario}.csv` | PSI/CSI stability metrics comparing Main vs MR periods. |
| `images/risk_production_visualizer_{scenario}.html` | Interactive dashboard for exploring the efficient frontier. |
| `images/stability_report_{scenario}.html` | Interactive stability dashboard with PSI charts. |

### Consolidated Outputs (Portfolio-Level)

| Artifact | Description |
| :--- | :--- |
| `output/consolidated_risk_production.csv` | Aggregated metrics by supersegment, segment, and total. |
| `output/consolidated_risk_production.html` | Interactive consolidated dashboard. |

## CLI Reference

### `main.py` - Single Segment Pipeline

```bash
uv run python main.py [--config CONFIG_PATH]
```

### `run_batch.py` - Multi-Segment Batch Processing

```bash
uv run python run_batch.py [OPTIONS]

Options:
  -s, --segments NAME [NAME ...]  Specific segments to run (default: all)
  -l, --list                      List available segments and exit
  -p, --parallel                  Run segments in parallel
  -w, --workers N                 Number of parallel workers (default: CPU count)
  -o, --output DIR                Base output directory (default: output)
  -c, --config FILE               Path to base config file (default: config.toml)
  --segments-config FILE          Path to segments config (default: segments.toml)
  --reuse-models                  Reuse existing supersegment models if available
  --clean                         Remove output directories before running
  --clean-only                    Only clean output directories (don't run)
  --skip-dq-checks                Skip data quality checks (not recommended)
  --no-consolidation              Skip generating consolidated report
  --consolidate-only              Only generate consolidated report (skip segments)
```

## Suggested Improvements

### Technical Improvements

1. **Unit Testing**: Continue expanding `pytest` coverage. While `src/mr_pipeline.py` has tests, adding tests for the parallelized `src/utils.py` functions (mocking `joblib`) would ensure stability.
2. **CI/CD Integration**: Implement a GitHub Actions workflow to run tests and linters (ruff/mypy) on push.
3. **Containerization**: Create a `Dockerfile` to standardize the execution environment (Python version, system dependencies for `sas7bdat`).
4. **Error Handling**: Enhance exception handling in the parallel workers to ensure robust logging and graceful failure if a single chunk fails.

### Functional Improvements

1. **Dynamic Variable Support**: The current logic heavily assumes a 2-variable grid (`sc_octroi_new_clus`, `new_efx_clus`). Generalizing this to N-dimensions would allow for more complex strategies.
2. **Stability Metrics**: Add specific metrics to compare Main vs. MR distribution stability (PSI - Population Stability Index) automatically in the pipeline.
3. **Interactive Reporting**: Consolidate the multiple HTML outputs into a single Streamlit or Dash app for easier navigation between scenarios.
4. **Resume Failed Runs**: Add `--resume` flag to only re-run failed segments from previous batch.
5. **Run by Supersegment**: Add `--supersegment NAME` flag to run all segments belonging to a supersegment.

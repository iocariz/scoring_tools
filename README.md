# Scoring Optimization Tools

## Overview

A credit risk scoring and portfolio optimization pipeline that processes loan application data, trains risk models, and determines optimal acceptance cutoffs to maximize production while controlling risk. The system operates on a two-dimensional grid of score variables (e.g., internal score bins and external score bins), evaluating all feasible cutoff combinations under monotonicity constraints to identify Pareto-optimal strategies.

### Key Capabilities

- **Pareto Optimization**: Exhaustive search of monotonic cutoff combinations on a 2D score grid, identifying the efficient frontier of risk vs. production.
- **Scenario Analysis**: Evaluates strategy robustness across pessimistic, base, and optimistic risk appetites.
- **Recent Monitoring (MR)**: Validates proposed cutoffs against a holdout recent period.
- **Stability Analysis**: PSI/CSI drift detection between main and MR periods.
- **Supersegments**: Trains shared models across related segments, then optimizes individually.
- **Reject Inference**: Corrects selection bias for score-rejected applications using acceptance-rate-based parceling.
- **Fixed Cutoffs**: Bypasses optimization to evaluate predefined cutoff configurations.
- **Global Allocation**: Distributes a portfolio-wide risk budget across segments using MILP or greedy solvers.
- **Score Discriminance**: Gini, lift, precision-recall, ROC analysis, and DeLong pairwise model comparison.
- **Trend Monitoring**: Monthly metric aggregation with SPC-based anomaly detection.
- **Bootstrap Confidence Intervals**: Quantifies uncertainty on production and risk estimates.
- **Interactive Dashboards**: Plotly/Dash web applications for exploring results.

---

## Quick Start

### Installation

```bash
uv pip install -e .
```

### Basic Execution

```bash
# Single segment (uses config.toml)
uv run python main.py

# All segments in batch (uses config.toml + segments.toml)
uv run python run_batch.py

# Specific segments only
uv run python run_batch.py -s no_premium_ab premium

# Parallel execution
uv run python run_batch.py --parallel --workers 4
```

---

## Execution Options

### 1. `main.py` -- Single Segment Pipeline

Runs the full pipeline for one segment: config loading, data preparation, model training, optimization, scenario analysis, MR validation, stability, and trend analysis.

```bash
uv run python main.py [OPTIONS]
```

| Flag | Short | Description |
|:-----|:------|:------------|
| `--config PATH` | `-c` | Configuration TOML file (default: `config.toml`) |
| `--model-path DIR` | `-m` | Pre-trained model directory (skips training) |
| `--training-only` | `-t` | Run only preprocessing and model training (skip optimization) |
| `--skip-dq-checks` | | Skip data quality checks |

**Examples:**

```bash
# Default run
uv run python main.py

# Use custom config
uv run python main.py --config configs/segment_a.toml

# Training only (for supersegment model creation)
uv run python main.py --training-only

# Use a pre-trained model, skip directly to optimization
uv run python main.py --model-path output/_supersegment_no_premium/models/model_20250101_120000
```

### 2. `run_batch.py` -- Multi-Segment Batch Processing

Orchestrates the pipeline across all segments defined in `segments.toml`. Handles supersegment model training, per-segment optimization, and consolidated reporting.

```bash
uv run python run_batch.py [OPTIONS]
```

| Flag | Short | Description |
|:-----|:------|:------------|
| `--segments NAME [NAME ...]` | `-s` | Run specific segments only (default: all) |
| `--list` | `-l` | List available segments and exit |
| `--parallel` | `-p` | Run segments in parallel |
| `--workers N` | `-w` | Number of parallel workers (default: CPU count) |
| `--output DIR` | `-o` | Base output directory (default: `output`) |
| `--config PATH` | `-c` | Base config file (default: `config.toml`) |
| `--segments-config PATH` | | Segments config file (default: `segments.toml`) |
| `--reuse-models` | | Reuse existing supersegment models (skip retraining) |
| `--clean` | | Remove output directories before running |
| `--clean-only` | | Only clean output directories, don't run pipeline |
| `--skip-dq-checks` | | Skip data quality checks |
| `--no-consolidation` | | Skip consolidated report generation |
| `--consolidate-only` | | Only generate consolidated report (skip segments) |

**Examples:**

```bash
# Run all segments, clean first
uv run python run_batch.py --clean

# Run two segments in parallel
uv run python run_batch.py -s no_premium_cd premium --parallel

# Reuse existing supersegment models
uv run python run_batch.py --reuse-models

# Just regenerate consolidated report from existing outputs
uv run python run_batch.py --consolidate-only

# List configured segments
uv run python run_batch.py --list
```

### 3. `run_allocation.py` -- Global Portfolio Allocation

After batch processing, allocates a global risk target across segments by selecting one point from each segment's efficient frontier to maximize total production.

```bash
uv run python run_allocation.py --target TARGET [OPTIONS]
```

| Flag | Description |
|:-----|:------------|
| `--target FLOAT` | **(Required)** Global risk target in % (e.g., `1.0`) |
| `--data-dir DIR` | Directory containing frontier CSVs (default: `data`) |
| `--output PATH` | Output CSV file (default: `allocation_results.csv`) |
| `--scenario NAME` | Scenario to use (default: `base`) |
| `--method {exact,greedy}` | Optimization method (default: `exact`) |
| `--segments-config PATH` | Segments config file for min/max risk constraints (default: `segments.toml`) |

**Methods:**

- **`exact`** (MILP via `scipy.optimize.milp`): Globally optimal allocation. Falls back to greedy if infeasible.
- **`greedy`**: Hill-climbing heuristic. Faster but may find local optima.

**Examples:**

```bash
# Optimal allocation at 1.0% global risk
uv run python run_allocation.py --target 1.0

# Greedy solver with custom scenario
uv run python run_allocation.py --target 1.2 --method greedy --scenario optimistic
```

### 4. `run_score_metrics.py` -- Score Discriminance Analysis

Evaluates score performance: Gini coefficient, lift tables, precision-recall curves, ROC curves, and DeLong pairwise comparison between score models.

```bash
uv run python run_score_metrics.py [OPTIONS]
```

| Flag | Short | Description |
|:-----|:------|:------------|
| `--segments NAME [NAME ...]` | `-s` | Specific segments to evaluate (default: all) |
| `--output DIR` | `-o` | Output directory (default: `output`) |
| `--config PATH` | `-c` | Base config file (default: `config.toml`) |
| `--segments-config PATH` | | Segments config file (default: `segments.toml`) |

### 5. `dashboard.py` -- Interactive Results Dashboard

Web-based Dash application for exploring pipeline results. Supports scenario comparison, main/MR period visualization, and interactive cutoff exploration.

```bash
uv run python dashboard.py [OPTIONS]
```

| Flag | Short | Description |
|:-----|:------|:------------|
| `--output DIR` | | Output location (auto-detects structure) |
| `--segment NAME` | `-s` | Initial segment to display |
| `--port PORT` | `-p` | Port (default: `8050`) |
| `--debug` | | Debug mode |

### 6. `interactive_allocator.py` -- Global Allocation Dashboard

Interactive web application for real-time global portfolio optimization. Configure risk targets per segment and visualize the allocation interactively.

```bash
uv run python interactive_allocator.py [OPTIONS]
```

| Flag | Short | Description |
|:-----|:------|:------------|
| `--port PORT` | `-p` | Port (default: `8051`) |
| `--debug` | | Debug mode |

---

## Pipeline Phases

When `main.py` runs, the pipeline executes these phases sequentially:

### Phase 1: Configuration

Loads and validates the TOML configuration file using Pydantic. Computes the annual coefficient for period normalization based on the observation window length.

### Phase 2: Data Loading

Loads SAS data (`.sas7bdat`), standardizes column names (lowercase, underscores), and validates required columns. In batch mode, data is loaded once and shared across segments.

### Phase 3: Preprocessing

1. **Data Quality Checks** -- Schema validation, missing values, outlier detection (Z-score), date range validation, categorical consistency.
2. **Filtering** -- By segment (regex), date range, and application status (booked / score-rejected / other).
3. **Feature Engineering** -- Bins continuous scores into cluster variables using configured bin edges.
4. **Stress Factor** -- Calculated from the worst 5% of booked applications as a risk correction.
5. **Transformation Rate** -- Monthly financing rate over a rolling window (`n_months`).

### Phase 4: Inference (Model Training)

Trains a polynomial surface model on the 2D score grid to predict risk (`b2_ever_h6`).

- **Feature sets tested**: simple (2 features), base (3: with interaction), polynomial (squared/cubic), full.
- **Estimators evaluated**: `HurdleRegressor`, `TweedieGLM`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`.
- **Selection**: 5-fold cross-validation on mean R².
- **TODU amount model**: Linear regression `oa_amt` -> `todu_amt_pile_h6`, saved separately.
- **SHAP analysis**: Feature importance computed and saved with model metadata.
- **Pre-trained models**: Can load supersegment models via `--model-path` to skip training.

### Phase 5: Optimization

1. **Feasible Solutions**: Generates all cutoff combinations on the 2D grid enforcing monotonicity (better scores permit more lenient cutoffs). Processed in chunks for memory efficiency.
2. **KPI Calculation**: For each solution computes production (`oa_amt_h0`), risk (`b2_ever_h6`), swap-in, and swap-out metrics.
3. **Reject Inference** (optional): Adjusts predicted risk for score-rejected bins based on per-bin acceptance rates.
4. **Pareto Frontier**: Identifies non-dominated solutions (maximum production for each risk level).

### Phase 6: Scenario Analysis

For each scenario (pessimistic / base / optimistic):

1. Selects the optimal solution from the Pareto frontier at the scenario's risk threshold.
2. Computes bootstrap confidence intervals (100 resamples).
3. Generates interactive Pareto dashboard (HTML).
4. Runs MR period validation with the selected cutoffs.
5. Calculates PSI/CSI stability metrics between main and MR periods.
6. Generates audit tables (record-level classification: keep / swap-in / swap-out / rejected).
7. Saves all outputs (CSVs, HTML visualizations).

### Phase 7: Trend Analysis

Computes monthly aggregated metrics (approval rate, production, risk) and detects anomalies using Statistical Process Control (rolling mean +/- n-sigma bounds).

---

## Configuration

### `config.toml` -- Base Settings

Global pipeline parameters. Per-segment overrides go in `segments.toml`.

```toml
[preprocessing]
# --- Required ---

# Columns to retain from source data
keep_vars = ["authorization_id", "mis_date", "status_name", "risk_score_rf",
             "se_decision_id", "reject_reason", "score_rf", "segment_cut_off", "early_bad"]

# Indicator columns (targets and amounts)
indicators = ["acct_booked_h0", "oa_amt", "todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]

# Grid variables (exactly 2)
variables = ["sc_octroi_new_clus", "new_efx_clus"]

# Bin edges for clustering (supports -inf / inf)
octroi_bins = [-inf, 364.0, 370.0, 379.0, 382.0, 389.0, 397.0, 404.0, 416.0, 431.0, inf]
efx_bins = [-inf, 2.0, 7.0, 11.0, 16.0, 22.0, 27.0, 33.0, 38.0, 43.0, 47.0, 51.0,
            56.0, 62.0, 68.0, 73.0, 78.0, 83.0, 89.0, 94.0, inf]

# Main observation period
date_ini_book_obs = "2024-07-01"
date_fin_book_obs = "2025-06-01"

# --- Optional ---

# Recent Monitoring period (both required if either is set)
date_ini_book_obs_mr = "2025-07-01"
date_fin_book_obs_mr = "2025-12-01"

# Data source
data_path = "data/demanda_direct_out.sas7bdat"   # default

# Segment filter (usually overridden per-segment in segments.toml)
segment_filter = "unknown"                         # default

# Score measures for discriminance analysis
score_measures = ["m_ct_direct_sc_nov23", "m_ct_direct_sc_jan23", ...]

# Economic parameters
multiplier = 7.0          # Risk formula multiplier (default: 7.0)
z_threshold = 3.0         # Outlier detection Z-score threshold (default: 3.0)
optimum_risk = 1.1        # Target risk appetite in % (default: 1.1)
risk_step = 0.1           # Scenario step: optimum_risk +/- risk_step (default: 0.1)
n_months = 24             # Rolling window for transformation rate (default: 12)
inv_var1 = false          # Invert var1 comparison: >= instead of <= (default: false)

# Logging
log_level = "INFO"        # default: "INFO"

# Comfort zone yearly limits
[preprocessing.cz_config]
2022 = 4.5
2023 = 4.2
2024 = 3.8
2025 = 3.5
```

#### Reject Inference (Optional)

Adjusts predicted risk for score-rejected bins to correct selection bias. Only the risk indicator (`todu_30ever_h6`) is adjusted; revenue (`oa_amt`) is observable for all records.

```toml
reject_inference_method = "none"       # "none" (default) or "parceling"
reject_uplift_factor = 1.5            # Scaling coefficient for reject ratio (0.0-10.0, default: 1.5)
reject_max_risk_multiplier = 3.0      # Upper cap for per-bin multiplier (1.0-10.0, default: 3.0)
```

**Parceling formula** (per bin):
```
acceptance_rate = n_booked / (n_booked + n_score_rejected)
reject_ratio = 1 - acceptance_rate
risk_multiplier = clamp(1 + uplift_factor * reject_ratio, 1.0, max_multiplier)
adjusted_todu_30ever_h6 = todu_30ever_h6 * risk_multiplier
```

#### Fixed Cutoffs (Optional)

Skip optimization and apply predefined cutoffs. Useful for regulatory scenarios or validating approved strategies.

```toml
[preprocessing.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # var0 bins
new_efx_clus = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]                            # var1 cutoffs per bin
strict_validation = false   # true: raise errors; false: warnings (default: false)
run_all_scenarios = false   # true: pessimistic/base/optimistic; false: base only (default: false)
```

### `segments.toml` -- Batch Segment Configuration

Defines segments and supersegments for `run_batch.py`. Per-segment settings override `config.toml` values.

```toml
# --- Supersegments ---
# Train a single model on combined data from multiple segments.
[supersegments.no_premium]
segment_filters = [
    "personal_loans,_fusion_and_tj_direct-no_premium-a-b",
    "personal_loans,_fusion_and_tj_direct-no_premium-c-d",
    "personal_loans,_fusion_and_tj_direct-no_premium-e-f"
]

# --- Segments ---
[segments.no_premium_ab]
segment_filter = "personal_loans,_fusion_and_tj_direct-no_premium-a-b"
optimum_risk = 1.1          # Per-segment risk appetite
risk_step = 0.1             # Per-segment scenario step
supersegment = "no_premium" # Use shared model
min_risk = 0.8              # Optional: floor for global allocation
max_risk = 1.4              # Optional: ceiling for global allocation

# Optional: fixed cutoffs for this segment
[segments.no_premium_ab.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
new_efx_clus = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
strict_validation = true
run_all_scenarios = true

[segments.no_premium_cd]
segment_filter = "personal_loans,_fusion_and_tj_direct-no_premium-c-d"
optimum_risk = 1.3
risk_step = 0.1
supersegment = "no_premium"

[segments.premium]
segment_filter = "personal_loans,_fusion_and_tj_direct-premium-"
optimum_risk = 0.9
risk_step = 0.1
# No supersegment = individual model training
```

**Per-segment overridable fields:** `segment_filter`, `optimum_risk`, `risk_step`, `inv_var1`, `reject_inference_method`, `reject_uplift_factor`, `reject_max_risk_multiplier`, and any `[preprocessing]` key from `config.toml`.

---

## Features

### Risk Metric: `b2_ever_h6`

The primary risk indicator, calculated as:

```
b2_ever_h6 = multiplier * todu_30ever_h6 / todu_amt_pile_h6
```

Where `multiplier` defaults to 7, `todu_30ever_h6` is the sum of 30+ day delinquency events over a 6-month horizon, and `todu_amt_pile_h6` is the total outstanding amount over the same horizon.

### Feasible Solutions and Monotonicity

The optimizer generates all valid cutoff combinations on the 2D grid (`var0` bins x `var1` cutoff levels). A **monotonicity constraint** is enforced: for each solution, bins with better scores (lower risk) must have cutoffs that are at least as lenient as bins with worse scores. This ensures the strategy is economically coherent.

### Pareto Frontier

From all feasible solutions, the pipeline identifies non-dominated solutions: those where no other solution offers both higher production and lower risk. The result is the **efficient frontier**, saved as `pareto_optimal_solutions.csv`.

### Scenario Analysis

Three scenarios are generated per segment based on `optimum_risk` and `risk_step`:

| Scenario | Risk Threshold |
|:---------|:---------------|
| Pessimistic | `optimum_risk - risk_step` |
| Base | `optimum_risk` |
| Optimistic | `optimum_risk + risk_step` |

Each scenario selects the Pareto-optimal solution with maximum production at or below its risk threshold, then produces a full set of outputs.

### Supersegments

When multiple segments share similar populations, a **supersegment** trains a single model on their combined data:

1. `run_batch.py` detects segments referencing the same `supersegment` name.
2. Trains the model once on the union of all segment populations.
3. Each segment loads the shared model and runs optimization independently with its own `optimum_risk`.

This produces more stable models and avoids redundant training.

### MR Period (Recent Monitoring)

When `date_ini_book_obs_mr` and `date_fin_book_obs_mr` are configured, the pipeline applies the selected cutoffs to a recent holdout period. This validates that the proposed strategy performs as expected on data not used during optimization. The MR results include risk, production, swap-in/swap-out metrics, and stability analysis.

### Stability Analysis (PSI/CSI)

Compares distributions between the main and MR periods using the Population Stability Index:

```
PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
```

| PSI Range | Interpretation |
|:----------|:---------------|
| < 0.10 | Stable |
| 0.10 -- 0.25 | Moderate drift |
| > 0.25 | Significant drift |

Generates per-variable PSI values, overall score PSI, and color-coded HTML reports.

### Audit Tables

Record-level classification for each application:

| Classification | Description |
|:---------------|:------------|
| `keep` | Booked and passes the proposed cutoff |
| `swap_out` | Booked but fails the proposed cutoff |
| `swap_in` | Score-rejected (`09-score`) but passes the proposed cutoff |
| `rejected` | Score-rejected and still fails |
| `rejected_other` | Non-score rejections (`08-other`) -- not candidates for cutoff changes |

### Bootstrap Confidence Intervals

For the selected optimal solution, the pipeline runs 100 bootstrap resamples of the booked population, recalculating production and risk for each sample. The 2.5th and 97.5th percentiles provide 95% confidence intervals.

### Reject Inference (Parceling)

The model trains exclusively on booked (approved) applications, creating selection bias. The parceling method corrects this by computing the acceptance rate per (var0, var1) bin and applying a risk multiplier to score-rejected records. Bins with lower acceptance rates receive larger uplifts, capped at `reject_max_risk_multiplier`.

### Global Portfolio Allocation

After running all segments, `run_allocation.py` solves a portfolio-level optimization: select one point from each segment's efficient frontier to maximize total production subject to a weighted-average global risk constraint. The MILP formulation uses binary decision variables and linear constraints. Per-segment risk bounds (`min_risk`, `max_risk` in `segments.toml`) are respected.

### Score Discriminance

`run_score_metrics.py` evaluates all score variables defined in `score_measures`:

- Gini coefficient and KS statistic per score
- Lift tables (decile-based)
- Precision-recall and ROC curves
- DeLong test for pairwise statistical comparison between models
- Per-segment and per-supersegment analysis
- Main period and MR period comparison

### Trend Analysis

Monthly aggregation of approval rate, production volume, mean production, and risk metrics. Anomalies are detected using Statistical Process Control: months where the metric falls outside rolling mean +/- 2 standard deviations are flagged.

---

## Architecture

### Module Reference

#### Entry Points

| Module | Purpose |
|:-------|:--------|
| `main.py` | Single-segment pipeline runner |
| `run_batch.py` | Multi-segment batch orchestrator with supersegment support |
| `run_allocation.py` | Global portfolio risk allocation |
| `run_score_metrics.py` | Score discriminance analysis |
| `dashboard.py` | Interactive Dash results dashboard |
| `interactive_allocator.py` | Interactive allocation dashboard |

#### Pipeline Orchestration (`src/pipeline/`)

| Module | Purpose |
|:-------|:--------|
| `config_loader.py` | Config loading, validation, and annual coefficient computation |
| `preprocessing.py` | Orchestrates DQ checks, filtering, binning, stress factor |
| `inference.py` | Orchestrates model training or loading |
| `optimization.py` | Orchestrates optimization, scenario analysis, MR, stability |

#### Core Modules (`src/`)

| Module | Purpose |
|:-------|:--------|
| `config.py` | `PreprocessingSettings` (Pydantic) and `OutputPaths` definitions |
| `data_manager.py` | SAS data loading and column standardization |
| `data_quality.py` | Schema validation, outlier detection, quality checks |
| `preprocess_improved.py` | Date/segment filtering, feature engineering, binning |
| `inference_optimized.py` | Model training pipeline with feature selection and CV |
| `models.py` | Variable transformations and risk calculations |
| `estimators.py` | Custom estimators: `HurdleRegressor`, `TweedieGLM` |
| `persistence.py` | Model serialization with metadata (save/load) |
| `optimization_utils.py` | Feasible solution generation, KPI calculation, Pareto filtering |
| `reject_inference.py` | Acceptance rate computation and parceling adjustment |
| `mr_pipeline.py` | MR period validation and metrics |
| `stability.py` | PSI/CSI drift detection |
| `trends.py` | Monthly metrics aggregation and anomaly detection |
| `audit.py` | Record-level classification (keep/swap-in/swap-out/rejected) |
| `consolidation.py` | Multi-segment aggregation and consolidated reporting |
| `global_optimizer.py` | MILP and greedy global portfolio allocation |
| `metrics.py` | Gini, lift, precision-recall, ROC, DeLong test |
| `plots.py` | `RiskProductionVisualizer` and Plotly chart generation |
| `styles.py` | Consistent plot styling and color palette |
| `utils.py` | `calculate_b2_ever_h6`, bootstrap CI, cutoff summary generation |
| `constants.py` | Enums (`StatusName`, `RejectReason`, `Columns`) and defaults |
| `schema.py` | Pandera data schema validators |
| `alerts.py` | Alert generation for drift anomalies |

---

## Output Structure

### Per-Segment Outputs (`output/{segment}/`)

#### Data (`data/`)

| File | Description |
|:-----|:------------|
| `pareto_optimal_solutions.csv` | All Pareto-optimal solutions on the efficient frontier |
| `optimal_solution_{scenario}.csv` | Selected cutoffs for the scenario |
| `risk_production_summary_{scenario}.csv` | Actual vs Optimum risk and production metrics |
| `data_summary_desagregado_{scenario}.csv` | Bin-level disaggregated data |
| `efficient_frontier_{scenario}.csv` | Frontier data for global allocation |
| `cutoff_summary_by_segment.csv` | Cutoff summary (long format, all scenarios) |
| `cutoff_summary_wide.csv` | Cutoff summary (wide format, all scenarios) |
| `risk_production_summary_mr_{scenario}.csv` | MR period metrics |
| `data_summary_desagregado_mr_{scenario}.csv` | MR period bin-level data |
| `stability_psi_{scenario}.csv` | Per-variable PSI values |
| `drift_alerts_{scenario}.json` | Drift alert details |
| `monthly_metrics_{segment}.csv` | Monthly aggregated metrics |
| `trend_anomalies_{segment}.csv` | Detected trend anomalies |

#### Images (`images/`)

| File | Description |
|:-----|:------------|
| `risk_vs_production.html` | Risk vs production scatter (preprocessing) |
| `transformation_rate.html` | Monthly financing rate over time |
| `b2_ever_h6_vs_octroi_and_risk_score.html` | Main period risk distribution |
| `risk_production_visualizer_{scenario}.html` | Interactive Pareto dashboard per scenario |
| `b2_ever_h6_vs_octroi_and_risk_score_mr_{scenario}.html` | MR period risk distribution |
| `stability_report_{scenario}.html` | PSI/CSI stability dashboard |
| `metric_trends_{segment}.html` | Monthly metric trend charts |

#### Models (`models/`)

| File | Description |
|:-----|:------------|
| `model_{timestamp}/model.joblib` | Trained risk model |
| `model_{timestamp}/metadata.json` | CV scores, features, hyperparameters |
| `model_{timestamp}/shap_summary.png` | SHAP feature importance |
| `todu_model.joblib` | TODU amount regression model |
| `todu_avg_inference.html` | TODU inference visualization |

### Consolidated Outputs (`output/`)

| File | Description |
|:-----|:------------|
| `consolidated_risk_production.csv` | Aggregated metrics across all segments |
| `consolidated_risk_production.html` | Portfolio-level interactive dashboard |
| `score_discriminance.csv` | Gini and discriminance metrics per score |
| `score_discriminance_*.png` | Score discrimination plots |
| `allocation_results.csv` | Global allocation results (if `run_allocation.py` was run) |

### Supersegment Outputs (`output/_supersegment_{name}/`)

Contains the shared model artifacts and training config. Same structure as per-segment `models/` directory.

---

## Workflows

### Standard Batch Workflow

```bash
# 1. Configure base settings
vim config.toml

# 2. Define segments and supersegments
vim segments.toml

# 3. Run the full batch pipeline
uv run python run_batch.py --clean --parallel

# 4. Review consolidated report
open output/consolidated_risk_production.html

# 5. (Optional) Run score discriminance analysis
uv run python run_score_metrics.py

# 6. (Optional) Global risk allocation
uv run python run_allocation.py --target 1.0

# 7. (Optional) Launch interactive dashboard
uv run python dashboard.py
```

### Fixed Cutoffs Workflow

When cutoffs are predetermined (e.g., approved by a committee):

```toml
# In segments.toml
[segments.my_segment.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
new_efx_clus = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
strict_validation = true
run_all_scenarios = true    # Generate all 3 scenarios with fixed cutoffs
```

```bash
uv run python run_batch.py -s my_segment
```

### Supersegment Workflow

```toml
# In segments.toml

# 1. Define the supersegment with all member segment filters
[supersegments.no_premium]
segment_filters = ["segment_a_filter", "segment_b_filter", "segment_c_filter"]

# 2. Each segment references the supersegment
[segments.segment_a]
segment_filter = "segment_a_filter"
supersegment = "no_premium"
optimum_risk = 1.1

[segments.segment_b]
segment_filter = "segment_b_filter"
supersegment = "no_premium"
optimum_risk = 1.3
```

```bash
# run_batch.py automatically:
# 1. Trains model on combined (segment_a + segment_b + segment_c) data
# 2. Runs optimization per segment with the shared model
uv run python run_batch.py
```

---

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_optimization_utils.py -v

# Run with coverage report
uv run pytest --cov=src tests/
```

### Test Files

| Test File | Coverage |
|:----------|:---------|
| `test_integration.py` | End-to-end pipeline |
| `test_preprocessing.py` | Data filtering, binning, feature engineering |
| `test_data_quality.py` | Schema validation, outlier detection |
| `test_validation.py` | Config validation |
| `test_models.py` | Model training and transformations |
| `test_estimators.py` | `HurdleRegressor`, `TweedieGLM` |
| `test_optimization_utils.py` | Solution generation, KPI calculation, Pareto |
| `test_global_optimizer.py` | MILP and greedy allocation |
| `test_mr_pipeline.py` | MR period processing |
| `test_stability.py` | PSI/CSI calculations |
| `test_trends.py` | Monthly metrics, anomaly detection |
| `test_audit.py` | Audit table generation |
| `test_consolidation.py` | Multi-segment aggregation |
| `test_reject_inference.py` | Reject inference adjustments |
| `test_metrics.py` | Score performance metrics |
| `test_plots.py` | Visualization functions |
| `test_utils.py` | Utility functions |
| `test_persistence.py` | Model save/load |
| `test_shap.py` | SHAP analysis |

### Adding a New Segment

1. Add the segment definition to `segments.toml` with at minimum `segment_filter`.
2. Optionally set `optimum_risk`, `risk_step`, `supersegment`, and/or `fixed_cutoffs`.
3. Run: `uv run python run_batch.py -s NEW_SEGMENT_NAME`

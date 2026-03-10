# Scoring Optimization Tools

## Overview

A credit risk scoring and portfolio optimization pipeline that processes loan application data, trains risk models, and determines optimal acceptance cutoffs to maximize production while controlling risk. The system operates on an N-dimensional grid of score variables (e.g., internal score bins × external score bins, optionally × income bins), evaluating all feasible cutoff combinations under monotonicity constraints to identify Pareto-optimal strategies.

### Key Capabilities

- **Pareto Optimization**: MILP-based search of monotonic cutoff combinations on an N-dimensional score grid (2D and beyond), identifying the efficient frontier of risk vs. production.
- **Scenario Analysis**: Evaluates strategy robustness across pessimistic, base, and optimistic risk appetites.
- **Recent Monitoring (MR)**: Validates proposed cutoffs against a holdout recent period.
- **Stability Analysis**: PSI/CSI drift detection between main and MR periods.
- **Supersegments**: Trains shared models across related segments, then optimizes individually.
- **Reject Inference**: Corrects selection bias for score-rejected applications using acceptance-rate-based parceling with three functional forms (linear, power, sigmoid), optional Bayesian smoothing, monotonicity enforcement, per-bin confidence scores, and automated parameter tuning via grid search or Optuna.
- **Sensitivity Analysis**: Measures cutoff stability under risk perturbations, identifying per-cell flip thresholds.
- **Marginal Impact**: Analytical O(N) computation of the production and risk impact of flipping each cell's accept/reject status.
- **Cell-Level Confidence Intervals**: K-fold CV prediction intervals per grid cell, quantifying model uncertainty.
- **Optimization-Aware Binning**: Supervised bin splitting using production-weighted risk differentiation, giving the optimizer maximal leverage from additional dimensions (e.g., income).
- **Fixed Cutoffs**: Bypasses optimization to evaluate predefined cutoff configurations. Supports both 2-variable (paired bins/cutoffs) and N>2 (per-variable accepted bin lists).
- **Swap-In Constraints**: Optional MILP constraints that cap the swap-in (repesca) population's production share and/or risk directly inside the solver, so the Pareto frontier only contains solutions with controlled swap-in exposure.
- **Fixed-Cell Constraints**: Pin individual cells as forced-accept or forced-reject before re-optimizing.
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
| `--production-floor FLOAT` | Optional global minimum production target enforced during allocation |
| `--lock SEGMENT:SOL_FAC` | Lock a segment to a specific frontier point (repeatable) |

**Methods:**

- **`exact`** (MILP via `scipy.optimize.milp`): Globally optimal allocation. Falls back to greedy if infeasible.
- **`greedy`**: Hill-climbing heuristic. Faster but may find local optima; when `--production-floor` is supplied, it still grows production until the floor is met or raises if the floor is infeasible under the target.

**Examples:**

```bash
# Optimal allocation at 1.0% global risk
uv run python run_allocation.py --target 1.0

# Enforce a minimum global production floor
uv run python run_allocation.py --target 1.0 --production-floor 50000

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

Web-based Dash application for exploring pipeline results. Supports scenario comparison, main/MR period visualization, interactive cutoff exploration with marginal impact heatmaps, uncertainty overlays, and pin mode for fixed-cell re-optimization.

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
3. **Feature Engineering** -- Bins continuous scores into cluster variables using configured bin edges. When edges are learned automatically (`max_bins` without `bin_edges`), two methods are available: `"quantile"` (equal-count splits, default) and `"optimization"` (production-weighted risk split via `DecisionTreeRegressor`).
4. **Stress Factor** -- Risk correction for the rejected population. Three modes: `"global"` (single scalar from the worst 5% of booked, default), `"per_bin"` (separate factor per grid cell), or `"disabled"` (no stress adjustment, recommended when parceling is active).
5. **Transformation Rate** -- Monthly financing rate over a rolling window (`n_months`). When `per_bin_tasa_fin = true`, computed per grid cell instead of as a global scalar.

### Phase 4: Inference (Model Training)

Trains a polynomial surface model on the score grid (uses the first 2 variables for 3D visualization, supports N variables for model training) to predict risk (`b2_ever_h6`).

- **Feature sets tested**: simple (2 features), base (3: with interaction), polynomial (squared/cubic), full.
- **Estimators evaluated**: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `HurdleRegressor`, `TweedieGLM`, `XGBoost`, `LightGBM` (via Optuna hyperparameter tuning).
- **Cross-Validation Strategy**:
  - **Feature Selection & Tuning**: Uses 5-Fold Cross-Validation. *Crucially, validation folds are physically split on the raw unaggregated data first*. Aggregation and outlier filtering are then performed independently on the train and validation sets to prevent target leakage.
  - **Selection Metric**: Models and feature sets are ranked by weighted CV RMSE. The **one-standard-error rule** is applied relative to the lowest mean RMSE, selecting the simplest eligible candidate. CV R² is reported for diagnostics where available, but it is not the primary selector.
  - **Outer Validation**: After Optuna selects hyperparameters, candidates are re-evaluated on fresh folds; when enough raw data is available, a nested holdout split is used for a less biased estimate.
  - **Weighting**: Tuning, feature selection, cell-level CI fitting, and final training share the same sample-weight precedence: `todu_amt_pile_h6`, then `oa_amt_h0`, then `n_observations`.
- **Continuous Impact Sanity Check**: Runs a secondary `_estimate_continuous_impact` analysis to verify the linear relationship between the continuous production volume and risk metrics. Because this runs on heavily compressed, post-aggregated groups (where `N` is tiny), it uses **Leave-One-Out (LOO) CV** paired with `cross_val_predict` to ensure an unbiased, mathematically stable global R² trend estimate.
- **Monotonic constraints**: Tree models (XGBoost, LightGBM) use monotonic constraints derived from the configured `directions` dictionary, ensuring each variable's constraint matches its actual risk relationship (ascending or descending).
- **TODU amount model**: Linear regression `oa_amt` -> `todu_amt_pile_h6`, saved separately.
- **SHAP analysis**: Feature importance computed and saved with model metadata.
- **Pre-trained models**: Can load supersegment models via `--model-path` to skip training.

### Phase 5: Optimization

1. **Feasible Solutions**: Generates all cutoff combinations on the 2D grid enforcing monotonicity (better scores permit more lenient cutoffs). Processed in chunks for memory efficiency.
2. **KPI Calculation**: For each solution computes production (`oa_amt_h0`), risk (`b2_ever_h6`), swap-in, and swap-out metrics.
3. **Reject Inference** (optional): Adjusts predicted risk for score-rejected bins based on per-bin acceptance rates, with configurable parceling method (linear/power/sigmoid), optional Bayesian smoothing, and monotonicity enforcement.
4. **Pareto Frontier**: Identifies non-dominated solutions (maximum production for each risk level).

### Phase 6: Scenario Analysis

For each scenario (pessimistic / base / optimistic):

1. Selects the optimal solution from the Pareto frontier at the scenario's risk threshold.
2. Computes bootstrap confidence intervals (1,000 resamples).
3. Generates interactive Pareto dashboard (HTML).
4. Runs MR period validation with the selected cutoffs.
5. Calculates PSI/CSI stability metrics between main and MR periods.
6. Generates audit tables (record-level classification: keep / swap-in / swap-out / rejected).
7. Saves all outputs (CSVs, HTML visualizations).

### Phase 7: Sensitivity Analysis (Optional)

When `run_sensitivity = true` in configuration, runs after optimization:

1. **Risk perturbation**: Scales the `todu_30ever_h6_rep` column by +/-5%, 10%, 20% and re-solves the MILP at each level.
2. **Cell flip thresholds**: For each cell, finds the minimum perturbation that would flip its accept/reject status.
3. **Marginal impact**: Analytically computes the production and risk change from flipping each individual cell.

Outputs are saved to `sensitivity_analysis.csv`, `sensitivity_cell_detail.csv`, and `cell_marginal_impact.csv`.

### Phase 8: Trend Analysis

Computes monthly aggregated metrics (approval rate, production, risk) and detects anomalies using robust Statistical Process Control: a one-period-lagged rolling median with MAD-based scale, plus a t-distribution adjustment for small windows.

---

## Configuration

### `config.toml` -- Base Settings

Global pipeline parameters. Per-segment overrides go in `segments.toml`. All settings live under the `[preprocessing]` section.

#### Complete Parameter Reference

##### Core (Required)

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `keep_vars` | list[str] | *(required)* | Columns to retain from source data |
| `indicators` | list[str] | *(required)* | Target and amount indicator columns |
| `variables` | list[str] | *(required, >= 2, unique)* | Grid variables for optimization |
| `date_ini_book_obs` | str | *(required)* | Main observation period start date (YYYY-MM-DD) |
| `date_fin_book_obs` | str | *(required)* | Main observation period end date (YYYY-MM-DD) |

##### Data and Segment

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `data_path` | str | `"data/demanda_direct_out.sas7bdat"` | Path to SAS source data file |
| `segment_filter` | str | `"unknown"` | Segment regex filter (usually overridden per-segment) |
| `inference_variables` | list[str] | `= variables` | Subset of `variables` used for model training (>= 2, must be subset of `variables`) |
| `score_measures` | list[str] | `None` | Score columns for discriminance analysis (`run_score_metrics.py`) |
| `log_level` | str | `"INFO"` | Logging level |

##### Binning

Legacy 2-variable binning:

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `octroi_bins` | list[float] | `[]` | Bin edges for var0 (>= 2 values, supports `-inf`/`inf`) |
| `efx_bins` | list[float] | `[]` | Bin edges for var1 (>= 2 values, supports `-inf`/`inf`) |

N-variable binning (under `[preprocessing.bins.VAR_NAME]`):

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `source_col` | str | *(required)* | Raw column name in the data |
| `output_col` | str | *(required)* | Name of the binned column to create |
| `bin_edges` | list[float] | `[]` | Fixed bin edges (>= 2 values). When empty, edges are learned via `max_bins` |
| `max_bins` | int | `None` | Max bins for supervised edge learning. Required if `bin_edges` is empty |
| `method` | str | `"quantile"` | `"quantile"` (equal-count) or `"optimization"` (production-weighted risk split via DecisionTreeRegressor) |

Monotonicity directions (under `[preprocessing.directions]`):

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `<variable_name>` | int | *(auto-inferred)* | `1` = ascending risk (higher bin = riskier), `-1` = descending risk (higher bin = safer). Auto-inferred from data if not set |

##### Economic Parameters

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `multiplier` | float | `7.0` | > 0 | Risk formula multiplier for H6 metric |
| `multiplier_h3` | float | `4.0` | > 0 | Risk formula multiplier for H3 metric |
| `optimum_risk` | float | `1.1` | | Target risk appetite in % |
| `risk_step` | float | `0.1` | | Scenario step: creates pessimistic (`optimum_risk - risk_step`) and optimistic (`optimum_risk + risk_step`) |
| `n_months` | int | `12` | | Rolling window (months) for transformation rate computation |
| `z_threshold` | float | `3.0` | > 0 | Outlier detection Z-score threshold |

Comfort zone yearly limits (under `[preprocessing.cz_config]`):

```toml
[preprocessing.cz_config]
2022 = 4.5
2023 = 4.2
2024 = 3.8
```

##### MR (Recent Monitoring) Period

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `date_ini_book_obs_mr` | str | `None` | | MR period start date. Both MR dates required if either is set |
| `date_fin_book_obs_mr` | str | `None` | | MR period end date |
| `use_mr_outcomes` | bool | `false` | | Enable hybrid MR risk inference (use observed MR risk where sufficient) |
| `mr_min_obs_per_bin` | int | `30` | >= 1 | Min observations for an MR bin to qualify as `mr_observed` |
| `mr_extrapolation_method` | str | `"linear"` | | H3→H6 extrapolation: `"linear"`, `"power"`, `"logistic"`, or `"auto"` (fits curvature from data) |
| `mr_extrapolation_curvature` | float | `1.0` | 0-5 | Power exponent for `"power"` method. Ignored when `"auto"` |

##### Stress Factor

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `stress_mode` | str | `"global"` | `"global"` (single scalar from worst 5% of booked), `"per_bin"` (per grid cell, fallback to global for bins < 20 obs), or `"disabled"` (`stress_factor = 1.0`) |

When `stress_mode = "global"` and parceling is active, a log warning suggests `"disabled"` to avoid double-counting selection bias.

##### Transformation Rate

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `per_bin_tasa_fin` | bool | `false` | Compute `tasa_fin` per grid cell instead of a single global scalar. Bins with < 10 eligible records or invalid rates fall back to global |

##### Reject Inference

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `reject_inference_method` | str | `"none"` | | `"none"` or `"parceling"` |
| `reject_parceling_method` | str | `"linear"` | | `"linear"`, `"power"`, or `"sigmoid"` |
| `reject_uplift_factor` | float | `1.5` | 0-10 | Scaling coefficient for parceling |
| `reject_max_risk_multiplier` | float | `3.0` | 1-10 | Upper cap for per-bin risk multiplier |
| `reject_bayesian_smoothing` | bool | `false` | | Beta-Binomial smoothing of acceptance rates |
| `reject_bayesian_prior_strength` | float | `10.0` | 0-1000 | Bayesian prior strength (higher = more shrinkage toward global rate) |
| `reject_enforce_monotonicity` | bool | `false` | | Isotonic regression on multipliers per variable axis |
| `reject_include_all_rejections` | bool | `false` | | Include policy rejections (`08-other`) in acceptance rate denominator |

**Parceling formulas** (per bin):

- **Linear** (default): `multiplier = 1 + uplift_factor * (1 - rate)`
- **Power**: `multiplier = (1 / rate) ^ uplift_factor` — grows faster at low acceptance rates
- **Sigmoid**: `multiplier = 1 + uplift_factor / (1 + exp(10 * (rate - 0.5)))` — smooth S-curve, steep transition around 50% acceptance

All multipliers are floored at 1.0 and capped at `reject_max_risk_multiplier`.

##### Reject Inference Optimizer

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `run_ri_optimizer` | bool | `false` | | Enable automated RI parameter search |
| `ri_optimizer_method` | str | `"grid"` | | `"grid"` (exhaustive) or `"optuna"` (TPE, seed=42) |
| `ri_calibration_gamma` | float | `1.0` | 0-1 | Selection-bias exponent. Target = `booked_risk / acceptance_rate^gamma`. Lower = less aggressive |
| `ri_uplift_range` | list[float] | `[0.0, 5.0]` | | Search range [min, max] for `reject_uplift_factor` |
| `ri_max_mult_range` | list[float] | `[1.0, 5.0]` | | Search range [min, max] for `reject_max_risk_multiplier` |
| `ri_uplift_steps` | int | `11` | | Grid divisions for uplift (grid method) |
| `ri_max_mult_steps` | int | `9` | | Grid divisions for max multiplier (grid method) |
| `ri_optuna_n_trials` | int | `100` | 10-10000 | Number of Optuna trials |

Selection rule: minimum calibration error among feasible solutions, with ties within 5% broken by maximizing production. When MR data is available, reports `degradation_ratio = mr_error / main_error` for temporal stability validation.

##### MILP and Pareto Tuning

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `milp_time_limit` | float | `30.0` | > 0 | MILP solver timeout in seconds |
| `pareto_n_points` | int | `50` | 5-500 | Number of risk targets in Pareto sweep |
| `n_bootstraps` | int | `1000` | 100-50000 | Bootstrap replicates for confidence intervals |
| `cv_folds` | int | `4` | 2-10 | Cross-validation folds for model training |

##### Swap-In Constraints

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `max_swapin_production_pct` | float | `None` | 0-100 | Max % of total accepted production from swap-in. `None` = no limit |
| `max_swapin_risk` | float | `None` | 0-100 | Max `b2_ever_h6` (%) for swap-in population only. `None` = no limit |

##### Sensitivity Analysis

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `run_sensitivity` | bool | `false` | Run sensitivity analysis after optimization |
| `sensitivity_levels` | list[float] | `[-20, -10, -5, 5, 10, 20]` | Perturbation percentages to evaluate |

##### Fixed Cutoffs

Skip MILP optimization and apply predefined cutoffs. Set under `[preprocessing.fixed_cutoffs]`.

**2-variable** (paired bins/cutoffs — lists must have equal length):

```toml
[preprocessing.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0, 4.0]  # var0 bin values
new_efx_clus = [3, 4, 5, 6]                 # var1 max cutoff per var0 bin
strict_validation = false                     # Raise errors instead of warnings (default: false)
run_all_scenarios = false                     # Generate all 3 scenarios (default: false, base only)
```

**N>2 variables** (per-variable accepted bin lists):

```toml
[preprocessing.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0]
new_efx_clus = [1.0, 2.0, 3.0, 4.0]
income_bin = [1.0, 2.0]
strict_validation = false
run_all_scenarios = false
```

---

### `segments.toml` -- Per-Segment Overrides

Defines segments for batch processing. Each segment can override **any** `config.toml` parameter — the segment config is recursively merged on top of the base config.

#### Supersegment Definition

```toml
[supersegments.NAME]
segment_filters = ["segment_a", "segment_b"]  # Segment filters to combine for shared model training
```

#### Segment Definition

```toml
[segments.NAME]
segment_filter = "segment_a"   # (required) Segment regex filter
```

##### Common Per-Segment Overrides

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `segment_filter` | str | *(required)* | Segment regex filter value |
| `optimum_risk` | float | *(from config.toml)* | Per-segment risk appetite |
| `risk_step` | float | *(from config.toml)* | Per-segment scenario step |
| `supersegment` | str | `None` | Name of shared model supersegment |
| `variables` | list[str] | *(from config.toml)* | Override grid variables |
| `inference_variables` | list[str] | *(from config.toml)* | Override model training variables |

##### Allocation Constraints (used by `run_allocation.py`)

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `min_risk` | float | `None` | Minimum risk bound for global allocation |
| `max_risk` | float | `None` | Maximum risk bound for global allocation |
| `min_production` | float | `None` | Production floor for global allocation |
| `locked_sol_fac` | float | `None` | Lock segment to a specific frontier point (`sol_fac` value) |

##### Per-Segment Pipeline Overrides

Any `config.toml` field can be overridden per segment. Common examples:

```toml
[segments.conservative_segment]
segment_filter = "conservative"
optimum_risk = 0.8
risk_step = 0.05
reject_inference_method = "parceling"
reject_parceling_method = "power"
stress_mode = "disabled"
per_bin_tasa_fin = true
n_months = 12
max_swapin_production_pct = 10.0

[segments.conservative_segment.directions]
sc_octroi_new_clus = -1
new_efx_clus = -1

[segments.conservative_segment.bins.income_bin]
source_col = "income_t1t2_m"
output_col = "income_bin"
max_bins = 3
method = "optimization"

[segments.conservative_segment.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0]
new_efx_clus = [5, 6, 8]
```

### Global Portfolio Allocation

After running all segments, `run_allocation.py` solves a portfolio-level optimization: select one point from each segment's efficient frontier to maximize total production subject to a weighted-average global risk constraint. The MILP formulation uses binary decision variables and linear constraints. Per-segment bounds (`min_risk`, `max_risk`, `min_production` in `segments.toml`) are respected, and an optional portfolio-wide production floor can be enforced via `--production-floor`.

### Score Discriminance

`run_score_metrics.py` evaluates all score variables defined in `score_measures`:

- Gini coefficient and KS statistic per score
- Lift tables (decile-based)
- Precision-recall and ROC curves
- DeLong test for pairwise statistical comparison between models
- Per-segment and per-supersegment analysis
- Main period and MR period comparison

### Sensitivity Analysis

Measures how stable the optimized cutoffs are under model risk uncertainty. The pipeline perturbs the risk indicator (`todu_30ever_h6_rep`) by configurable percentages, re-solves the MILP at each level, and compares the resulting masks:

- **Aggregate summary**: Number of cells that flip, accept-to-reject vs reject-to-accept transitions, new production and risk at each perturbation level.
- **Per-cell detail**: Minimum perturbation percentage at which each cell changes status, with flip direction.

### Marginal Impact

For each cell in the grid, analytically computes the effect of flipping its status (accept → reject or vice versa) on portfolio production and risk. Uses baseline sums computed once, then adjusts numerator/denominator per cell — O(N) total, not O(N²).

Output columns: `delta_production` (EUR change), `delta_risk_pct` (percentage point change in `b2_ever_h6`), `cell_production`, `cell_risk`.

### Cell-Level Confidence Intervals

During model training, K-fold cross-validation produces per-cell prediction intervals:

1. For each fold, train on the train split, aggregate the validation split by grid bins, and predict the target variable per cell.
2. Across folds, compute mean, standard deviation, and confidence interval bounds per cell.
3. For small sample counts, interval half-widths use a Student-t critical value; otherwise a normal approximation is used.

Cells observed in fewer than 2 folds receive NaN intervals. Results are saved to `cell_level_ci.csv` and optionally displayed as uncertainty annotations on the dashboard heatmap.

### Fixed-Cell Constraints

Individual cells can be pinned as forced-accept or forced-reject before re-optimization. The MILP solver enforces these by setting `lb = ub = value` for pinned cells, while monotonicity and risk constraints remain active. If the constraints are contradictory (same cell pinned as both accept and reject, or pins make the risk target infeasible), the solver returns `None`.

Available through:
- **`solve_with_fixed_cells()`** in `src/optimization_utils.py` for programmatic use.
- **Pin Mode** in the dashboard Cutoff Explorer: click cells to cycle through unpinned → accept → reject → unpinned, then re-optimize.

### Swap-In Constraints

The MILP solver accepts two optional constraints that limit the swap-in (repesca) population — score-rejected applicants that would be accepted under the new optimized cutoffs:

1. **Production share cap** (`max_swapin_production_pct`): Ensures the fraction of total accepted production coming from swap-in does not exceed a given percentage. Linearized as:

   ```
   sum((oa_amt_h0_rep[i] - pct/100 * oa_amt_h0[i]) * x[i]) <= 0
   ```

2. **Risk cap** (`max_swapin_risk`): Ensures the `b2_ever_h6` of the swap-in sub-population stays below a given percentage. Linearized as:

   ```
   sum((multiplier * todu_30ever_h6_rep[i] - max_risk/100 * todu_amt_pile_h6_rep[i]) * x[i]) <= 0
   ```

Both are added as inequality rows alongside the existing risk budget and monotonicity constraints. When `None` (default), the MILP behaves exactly as before. If the constraints are too tight, the solver returns `None` (infeasible).

### Trend Analysis

Monthly aggregation of approval rate, production volume, mean production, and risk metrics. Anomalies are detected using robust Statistical Process Control: a lagged rolling median is used as the center line, MAD is converted to a standard-deviation-equivalent scale, and short windows use a Student-t critical value instead of a fixed z-score.

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

When `date_ini_book_obs_mr` and `date_fin_book_obs_mr` are configured, the pipeline applies the selected cutoffs to a recent holdout period (typically the most recent 6 months). This validates that the proposed strategy performs as expected on data not used during optimization. The MR results include risk, production, swap-in/swap-out metrics, and stability analysis.

#### Hybrid Risk Estimation

In the main period, all booked applications have a full 6-month performance window, so `b2_ever_h6` is directly observable. In the MR period this is not the case: only the earliest cohorts have matured enough for reliable outcome data. The pipeline uses a **hybrid per-bin risk estimation** controlled by `use_mr_outcomes` and `mr_min_obs_per_bin`:

| Priority | Source | Condition | Method |
|:---------|:-------|:----------|:-------|
| 1 | `mr_observed` | Enough **valid** MR H6 observations in the bin | Direct MR-period `b2_ever_h6` |
| 2 | `h3_extrapolated` | H3 configured, direct MR H6 is unavailable or insufficient, valid main-period H6/H3 ratio, and enough mature H3 observations in MR | MR-observed H3 scaled by the main-period ratio (see below) |
| 3 | `main_imputed` | Main-period bin exists but MR H6/H3 evidence is insufficient | Main-period `b2_ever_h6` |
| 4 | `model_fallback` | Bin exists only in MR and lacks enough valid MR H6/H3 outcomes | Inferred via the trained risk model |

#### H3-Based H6 Extrapolation

The 3-month horizon indicator (`b2_ever_h3`) matures in half the time of H6. In a 6-month MR window, mature H3 is usually available for more cohorts than mature H6, so the pipeline can extrapolate from H3 when direct H6 evidence is too sparse. It does this by using the **main-period H6/H3 ratio** as a scaling factor:

```
b2_h6_estimated = b2_mr_h3 × f(b2_main_h6 / b2_main_h3)
```

Where, for each score bin:
- `b2_main_h6` and `b2_main_h3` are computed from the main period (all applications fully matured)
- `b2_mr_h3` is computed from **only** the MR accounts with mature H3 data (accounts from months with at least 3 months of performance history; immature accounts with NaN H3 are excluded from the aggregation)
- `f(ratio)` is the extrapolation curve controlled by `mr_extrapolation_method`

**Extrapolation methods** (`mr_extrapolation_method`):

| Method | Formula | Use case |
|:-------|:--------|:---------|
| `linear` (default) | `b2_mr_h3 × ratio` | Proportional H6/H3 relationship |
| `power` | `b2_mr_h3 × ratio^curvature` | Convex/concave H6/H3 relationship |
| `logistic` | `b2_mr_h3 × (1 + 2·tanh(k·(ratio-1)/2)/k)` | Caps extreme ratios |
| `auto` | Fits curvature from main-period data | No domain expertise needed |

**Auto-calibration** (`mr_extrapolation_method = "auto"`): Performs a weighted log-log regression `log(b2_h6) = c + α·log(b2_h3)` across main-period bins to determine the H3→H6 curvature. If α's 95% confidence interval includes 1.0, linear is selected; otherwise power with the fitted α (clipped to [0.3, 3.0]) is used. Weights are per-bin observation counts (`n_obs_main`), downweighting noisy thin bins. The resolved method, curvature, and diagnostics (α, SE, R², n_bins) are logged and saved as `fitted_method` / `fitted_curvature` columns in `mr_risk_comparison_*.csv`.

**Safeguards:**
- Direct MR H6 sufficiency (`n_obs_mr`) counts only rows with non-null `todu_30ever_h6` **and** `todu_amt_pile_h6`; sparse or all-null H6 bins do not qualify for `mr_observed`.
- Bins where `b2_main_h3 ≈ 0` skip extrapolation and fall back to direct MR H6 if enough valid H6 observations exist, otherwise `main_imputed`.
- The number of MR accounts with mature H3 (`n_obs_mr_h3`) must meet the `mr_min_obs_per_bin` threshold; otherwise the bin falls back to `mr_observed` or `main_imputed`.
- The per-bin `h6_h3_ratio` and `n_obs_mr_h3` are included in the comparison CSV (`mr_risk_comparison_*.csv`) for auditing.
- Auto-calibration requires at least 4 valid bins with positive H3 and H6 and a non-degenerate log-H3 design; otherwise it falls back to linear.

This feature activates automatically when `use_mr_outcomes = true` and `multiplier_h3` is configured. No additional configuration is required beyond the optional `mr_extrapolation_method`.

### Stability Analysis (PSI/CSI)

Compares distributions between the main and MR periods using the Population Stability Index. A unified epsilon constant (`PSI_EPSILON = 0.0001`) is applied consistently to zero-percentage bins/categories in both the difference and `log(p/q)` terms used by PSI/CSI, preventing `log(0)` and avoiding asymmetric smoothing artifacts.

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

For the selected optimal solution, the pipeline runs 1,000 bootstrap resamples of the booked population, recalculating production and risk for each sample. The 2.5th and 97.5th percentiles provide 95% confidence intervals.

### Reject Inference (Parceling)

The model trains exclusively on booked (approved) applications, creating selection bias. The parceling method corrects this by computing the acceptance rate per (var0, var1) bin and applying a risk multiplier to score-rejected records. Bins with lower acceptance rates receive larger uplifts, capped at `reject_max_risk_multiplier`.

Three functional forms are available:

| Method | Formula | Best For |
|:-------|:--------|:---------|
| `linear` | `1 + factor * (1 - rate)` | General use, interpretable |
| `power` | `(1 / rate) ^ factor` | Heavy tail risk, aggressive at low acceptance |
| `sigmoid` | `1 + factor / (1 + exp(10 * (rate - 0.5)))` | Smooth S-curve, gentle at extremes |

**Bayesian smoothing** stabilizes noisy acceptance rates in small bins using a Beta-Binomial posterior with the global acceptance rate as prior. The `prior_strength` parameter controls the degree of shrinkage toward the global rate.

**Monotonicity enforcement** uses isotonic regression to ensure multipliers are non-decreasing along each variable axis (marginal monotonicity), preventing lower-risk bins from receiving higher adjustments than higher-risk bins.

**Per-bin confidence scores** (`1 - exp(-n_total / 50)`) quantify the reliability of each bin's reject inference adjustment based on sample size. Included in diagnostic output.

#### RI Parameter Optimizer

When `run_ri_optimizer = true`, the pipeline searches over `(reject_uplift_factor, reject_max_risk_multiplier)` to find parameters that minimize **calibration error** against the selection-bias model.

The calibration target for each cell is `booked_risk / acceptance_rate^gamma`, based on the standard selection-bias model: if a bin accepts the least-risky fraction *a* of applicants, the true population risk is approximately `observed / a^gamma`. The `gamma` parameter (default 1.0) controls target aggressiveness — lower values reduce the assumed separation between booked and rejected risk.

Two optimization methods are available:

- **Grid search** (`ri_optimizer_method = "grid"`): Exhaustive evaluation over a regular grid of parameter combinations. Deterministic and transparent.
- **Optuna TPE** (`ri_optimizer_method = "optuna"`): Tree-structured Parzen Estimator with `seed=42` for reproducibility. More sample-efficient for large search spaces.

Both methods use the same selection rule: minimum calibration error among feasible solutions, with ties within 5% of the minimum broken by maximizing production.

**Out-of-time MR validation**: When MR-period data is available, the optimizer automatically applies the best parameters to the MR period and reports `degradation_ratio = mr_error / main_error`. This validates temporal stability of the calibration — a ratio near 1.0 indicates the RI correction generalizes well to the holdout period.

### Global Portfolio Allocation

After running all segments, `run_allocation.py` solves a portfolio-level optimization: select one point from each segment's efficient frontier to maximize total production subject to a weighted-average global risk constraint. The MILP formulation uses binary decision variables and linear constraints. Per-segment bounds (`min_risk`, `max_risk`, `min_production` in `segments.toml`) are respected, and an optional portfolio-wide production floor can be enforced via `--production-floor`.

### Score Discriminance

`run_score_metrics.py` evaluates all score variables defined in `score_measures`:

- Gini coefficient and KS statistic per score
- Lift tables (decile-based)
- Precision-recall and ROC curves
- DeLong test for pairwise statistical comparison between models
- Per-segment and per-supersegment analysis
- Main period and MR period comparison

### Sensitivity Analysis

Measures how stable the optimized cutoffs are under model risk uncertainty. The pipeline perturbs the risk indicator (`todu_30ever_h6_rep`) by configurable percentages, re-solves the MILP at each level, and compares the resulting masks:

- **Aggregate summary**: Number of cells that flip, accept-to-reject vs reject-to-accept transitions, new production and risk at each perturbation level.
- **Per-cell detail**: Minimum perturbation percentage at which each cell changes status, with flip direction.

### Marginal Impact

For each cell in the grid, analytically computes the effect of flipping its status (accept → reject or vice versa) on portfolio production and risk. Uses baseline sums computed once, then adjusts numerator/denominator per cell — O(N) total, not O(N²).

Output columns: `delta_production` (EUR change), `delta_risk_pct` (percentage point change in `b2_ever_h6`), `cell_production`, `cell_risk`.

### Cell-Level Confidence Intervals

During model training, K-fold cross-validation produces per-cell prediction intervals:

1. For each fold, train on the train split, aggregate the validation split by grid bins, and predict the target variable per cell.
2. Across folds, compute mean, standard deviation, and confidence interval bounds per cell.
3. For small sample counts, interval half-widths use a Student-t critical value; otherwise a normal approximation is used.

Cells observed in fewer than 2 folds receive NaN intervals. Results are saved to `cell_level_ci.csv` and optionally displayed as uncertainty annotations on the dashboard heatmap.

### Fixed-Cell Constraints

Individual cells can be pinned as forced-accept or forced-reject before re-optimization. The MILP solver enforces these by setting `lb = ub = value` for pinned cells, while monotonicity and risk constraints remain active. If the constraints are contradictory (same cell pinned as both accept and reject, or pins make the risk target infeasible), the solver returns `None`.

Available through:
- **`solve_with_fixed_cells()`** in `src/optimization_utils.py` for programmatic use.
- **Pin Mode** in the dashboard Cutoff Explorer: click cells to cycle through unpinned → accept → reject → unpinned, then re-optimize.

### Swap-In Constraints

The MILP solver accepts two optional constraints that limit the swap-in (repesca) population — score-rejected applicants that would be accepted under the new optimized cutoffs:

1. **Production share cap** (`max_swapin_production_pct`): Ensures the fraction of total accepted production coming from swap-in does not exceed a given percentage. Linearized as:

   ```
   sum((oa_amt_h0_rep[i] - pct/100 * oa_amt_h0[i]) * x[i]) <= 0
   ```

2. **Risk cap** (`max_swapin_risk`): Ensures the `b2_ever_h6` of the swap-in sub-population stays below a given percentage. Linearized as:

   ```
   sum((multiplier * todu_30ever_h6_rep[i] - max_risk/100 * todu_amt_pile_h6_rep[i]) * x[i]) <= 0
   ```

Both are added as inequality rows alongside the existing risk budget and monotonicity constraints. When `None` (default), the MILP behaves exactly as before. If the constraints are too tight, the solver returns `None` (infeasible).

### Trend Analysis

Monthly aggregation of approval rate, production volume, mean production, and risk metrics. Anomalies are detected using robust Statistical Process Control: a lagged rolling median is used as the center line, MAD is converted to a standard-deviation-equivalent scale, and short windows use a Student-t critical value instead of a fixed z-score.

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
| `gradio_dashboard.py` | Gradio web UI for results exploration |

#### Pipeline Orchestration (`src/pipeline/`)

| Module | Purpose |
|:-------|:--------|
| `config_loader.py` | Config loading, validation, and annual coefficient computation |
| `preprocessing.py` | Orchestrates DQ checks, filtering, binning, stress factor (global/per-bin/disabled), transformation rate (global/per-bin) |
| `inference.py` | Orchestrates model training or loading |
| `optimization.py` | Orchestrates optimization, scenario analysis, MR, stability |

#### Core Modules (`src/`)

| Module | Purpose |
|:-------|:--------|
| `config.py` | `PreprocessingSettings` (Pydantic) and `OutputPaths` definitions |
| `data_manager.py` | SAS data loading and column standardization |
| `data_quality.py` | Schema validation, outlier detection, quality checks |
| `preprocess_improved.py` | Date/segment filtering, feature engineering, binning (quantile and optimization-aware) |
| `inference_optimized.py` | Model training pipeline with feature selection and CV |
| `models.py` | Variable transformations and risk calculations |
| `estimators.py` | Custom estimators: `HurdleRegressor`, `TweedieGLM` |
| `optuna_tuning.py` | Optuna hyperparameter tuning for tree and linear models |
| `persistence.py` | Model serialization with metadata (save/load) |
| `optimization_utils.py` | Feasible solution generation, KPI calculation, Pareto filtering, fixed-cell MILP |
| `sensitivity.py` | Sensitivity analysis, risk perturbation, cell flip thresholds, marginal impact |
| `reject_inference.py` | Acceptance rate computation, parceling (linear/power/sigmoid), Bayesian smoothing, monotonicity enforcement, per-bin confidence |
| `reject_inference_optimizer.py` | Grid search and Optuna TPE optimization over RI parameters, power-corrected calibration, MR out-of-time validation |
| `mr_pipeline.py` | MR period validation and metrics |
| `stability.py` | PSI/CSI drift detection |
| `trends.py` | Monthly metrics aggregation and anomaly detection |
| `audit.py` | Record-level classification (keep/swap-in/swap-out/rejected) |
| `consolidation.py` | Multi-segment aggregation and consolidated reporting |
| `global_optimizer.py` | MILP and greedy global portfolio allocation |
| `metrics.py` | Gini, lift, precision-recall, ROC, DeLong test |
| `plots.py` | `RiskProductionVisualizer` and Plotly chart generation |
| `styles.py` | Consistent plot styling and color palette |
| `utils.py` | `calculate_b2_ever_h6`, `extrapolate_h3_to_h6`, `fit_h3_extrapolation_curve`, bootstrap CI, cutoff summary generation |
| `constants.py` | Enums (`StatusName`, `RejectReason`, `Columns`), numeric defaults (`DEFAULT_N_BOOTSTRAPS`, `DEFAULT_SENSITIVITY_LEVELS`) |
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
| `sensitivity_analysis.csv` | Sensitivity analysis results (if `run_sensitivity = true`) |
| `sensitivity_cell_detail.csv` | Per-cell flip thresholds |
| `cell_marginal_impact.csv` | Per-cell marginal production/risk impact |
| `ri_optimizer_results.csv` | RI parameter optimization results (if `run_ri_optimizer = true`) |
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
| `model_{timestamp}/cell_level_ci.csv` | Per-cell prediction confidence intervals |
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

**2-variable** (paired bins/cutoffs):

```toml
# In segments.toml
[segments.my_segment.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
new_efx_clus = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
strict_validation = true
run_all_scenarios = true    # Generate all 3 scenarios with fixed cutoffs
```

**N>2 variables** (per-variable accepted bins):

```toml
# In segments.toml — cell accepted iff all coordinates in accepted lists
[segments.my_segment.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0]
new_efx_clus = [1.0, 2.0, 3.0, 4.0]
income_bin = [1.0, 2.0]
strict_validation = true
run_all_scenarios = true
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

# Focused validation for the core modeling / MR / optimization stack
uv run pytest tests/test_estimators.py tests/test_optuna_tuning.py tests/test_mr_pipeline.py tests/test_optimization_utils.py tests/test_global_optimizer.py tests/test_models.py tests/test_reject_inference.py tests/test_utils.py -q

# Focused validation for H3→H6 extrapolation and MR fallback logic
uv run pytest tests/test_utils.py tests/test_mr_pipeline.py -q
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
| `test_reject_inference_optimizer.py` | RI parameter optimizer and calibration error |
| `test_metrics.py` | Score performance metrics |
| `test_plots.py` | Visualization functions |
| `test_utils.py` | Utility functions |
| `test_persistence.py` | Model save/load |
| `test_optuna_tuning.py` | Optuna hyperparameter tuning |
| `test_sensitivity.py` | Sensitivity analysis, marginal impact, fixed-cell constraints |
| `test_shap.py` | SHAP analysis |

### Adding a New Segment

1. Add the segment definition to `segments.toml` with at minimum `segment_filter`.
2. Optionally set `optimum_risk`, `risk_step`, `supersegment`, and/or `fixed_cutoffs`.
3. Run: `uv run python run_batch.py -s NEW_SEGMENT_NAME`

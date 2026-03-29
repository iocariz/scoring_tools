# Scoring Optimization Tools

## Overview

A credit risk scoring and portfolio optimization pipeline that processes loan application data, trains risk models, and determines optimal acceptance cutoffs to maximize production while controlling risk. The system operates on an N-dimensional grid of score variables (e.g., internal score bins × external score bins, optionally × income bins), evaluating all feasible cutoff combinations under monotonicity constraints to identify Pareto-optimal strategies.

### Key Capabilities

- **Pareto Optimization**: MILP-based search of monotonic cutoff combinations on an N-dimensional score grid (2D and beyond), identifying the efficient frontier of risk vs. production.
- **Scenario Analysis**: Evaluates strategy robustness across pessimistic, base, and optimistic risk appetites.
- **Recent Monitoring (MR)**: Validates proposed cutoffs against a holdout recent period.
- **Stability Analysis**: PSI/CSI drift detection between main and MR periods.
- **Supersegments**: Trains shared models across related segments, then optimizes individually.
- **Reject Inference**: Corrects selection bias for score-rejected applications using acceptance-rate-based parceling with three functional forms (linear, power, sigmoid), optional Bayesian smoothing, monotonicity enforcement, temporal weighting/windowing, per-bin confidence diagnostics (raw and effective counts), and automated parameter tuning via grid search or Optuna.
- **Sensitivity Analysis**: Measures cutoff stability under risk perturbations, identifying per-cell flip thresholds.
- **Marginal Impact**: Analytical O(N) computation of the production and risk impact of flipping each cell's accept/reject status.
- **Cell-Level Confidence Intervals**: K-fold CV prediction intervals per grid cell, quantifying model uncertainty.
- **Optimization-Aware Binning**: Supervised bin splitting using production-weighted risk differentiation, giving the optimizer maximal leverage from additional dimensions (e.g., income).
- **Fixed Cutoffs**: Bypasses optimization to evaluate predefined cutoff configurations. Supports both 2-variable (paired bins/cutoffs) and N>2 (per-variable accepted bin lists).
- **Swap-In Constraints**: Optional MILP constraints that cap the swap-in (repesca) population's production share and/or risk directly inside the solver, so the Pareto frontier only contains solutions with controlled swap-in exposure.
- **Fixed-Cell Constraints**: Pin individual cells as forced-accept or forced-reject before re-optimizing.
- **Baseline Mode**: Show the current booked portfolio as-is (no cutoff optimization). MR inference still runs to predict risk for immature loans. Useful for benchmarking the existing policy before proposing new cutoffs.
- **Sequential Cutoff Ordering**: Enforce nested acceptance masks across segments (e.g., `mask_ef ⊆ mask_cd ⊆ mask_ab`) via configurable bottom-up (floor) or top-down (ceiling) constraints. Segments are automatically ordered by dependency.
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
| `--baseline` | | Baseline mode: show current portfolio as-is (no optimization) |
| `--skip-dq-checks` | | Skip data quality checks |

**Examples:**

```bash
# Default run
uv run python main.py

# Baseline mode: current portfolio metrics only
uv run python main.py --baseline

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
| `--training-only` | | Only run data quality + model training (skip optimization/reporting steps) |
| `--baseline` | | Baseline mode: show current portfolio as-is (no optimization) |
| `--cutoff-ordering-mode` | | Cutoff ordering direction: `bottom_up` (default) or `top_down` |
| `--no-report` | | Skip generating HTML reports |

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

You must pass **at least one** risk target via `--target` and/or `--what-if` (see below).

```bash
uv run python run_allocation.py --target TARGET [OPTIONS]
uv run python run_allocation.py --what-if 2.0,2.5,3.0 [OPTIONS]
```

| Flag | Description |
|:-----|:------------|
| `--target FLOAT` | Global risk target in % (e.g., `1.0`). Optional if `--what-if` is provided. |
| `--what-if LIST` | Comma-separated extra risk targets in % (e.g., `2.0,2.5,3.0`). Runs one optimization per target and builds a comparison pack. Combine with `--target` to include that target first without duplicating. |
| `--data-dir DIR` | Base directory for frontier discovery (default: `output`). Supports both `<data-dir>/<segment>/data/efficient_frontier_{scenario}.csv` and single-segment `<data-dir>/data/efficient_frontier_{scenario}.csv` |
| `--output PATH` | Primary output CSV (default: `allocation_results.csv`). The stem controls companion filenames (see **Portfolio-owner outputs**). |
| `--scenario NAME` | Scenario to use (default: `base`) |
| `--method {exact,greedy}` | Optimization method (default: `exact`) |
| `--segments-config PATH` | Segments config file for min/max risk constraints (default: `segments.toml`) |
| `--production-floor FLOAT` | Optional global minimum production target enforced during allocation |
| `--lock SEGMENT:SOL_FAC` | Lock a segment to a specific frontier point (repeatable) |

**Methods:**

- **`exact`** (MILP via `scipy.optimize.milp`): Globally optimal allocation. Falls back to greedy if infeasible.
- **`greedy`**: Hill-climbing heuristic. Faster but may find local optima; when `--production-floor` is supplied, it still grows production until the floor is met or raises if the floor is infeasible under the target.

**Portfolio-owner outputs** (written next to `--output`, using its stem, e.g. `allocation_results`):

| File | When | Contents |
|:-----|:-----|:-----------|
| `<stem>_policy_cutoff_table.csv` | Always | Long-form policy table: per-segment cutoffs (2-var bin → max accepted bin), or mask summary for N-var, plus optional swap-in/out columns from the frontier row. |
| `<stem>_allocation_narrative.md` | Single target only | Plain-language summary: risk target vs achieved, configured segment limits, and which constraints bind (MILP slack; greedy uses tolerance-based hints). |
| `<stem>_what_if.csv` | Multiple targets | One row per target: achieved global risk, production, method, binding constraint labels, segment count. |
| `<stem>_allocation_narratives.md` | Multiple targets | Same narrative style as above, with one section per target. |

The primary `--output` CSV is always the **first** target’s full per-segment frontier row export (`to_full_dataframe()`).

**Examples:**

```bash
# Optimal allocation at 1.0% global risk (writes narrative + policy table)
uv run python run_allocation.py --target 1.0

# Compare several global risk caps without repeating --target
uv run python run_allocation.py --what-if 2.0,2.5,3.0 --output allocation_base.csv

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

1. **MILP Solve**: For each risk target, solves a binary integer program maximizing production subject to a risk budget and monotonicity constraints (see [Methodology > MILP Optimization](#milp-optimization)).
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

1. **Risk perturbation**: Scales the `todu_30ever_h6_rep` column by +/-5%, 10%, 20% and re-solves the MILP at each level (respecting active swap-in constraints and MILP time limit).
2. **Cell flip thresholds**: For each cell, finds the minimum perturbation that would flip its accept/reject status.
3. **Marginal impact**: Analytically computes the production and risk change from flipping each individual cell.

Outputs are saved to `sensitivity_analysis.csv`, `sensitivity_cell_detail.csv`, and `cell_marginal_impact.csv`.

### Phase 8: Trend Analysis

Computes monthly aggregated metrics (approval rate, production, risk) and detects anomalies using robust Statistical Process Control: a one-period-lagged rolling median with MAD-based scale, plus a t-distribution adjustment for small windows.

---

## Methodology

### Data Populations

The system distinguishes three application populations based on their origination outcome:

| Population | Definition | Role |
|---|---|---|
| **Booked** | Applications that were approved and disbursed | Observed outcomes (risk + production) |
| **Score-rejected (repesca)** | Applications rejected by the credit score cutoff (`reject_reason = "09-score"`) | Candidates for cutoff optimization; outcomes unobserved, must be inferred |
| **Policy-rejected** | Applications rejected for non-score reasons (fraud, documentation, policy rules; `reject_reason = "08-other"`) | Excluded from optimization — rejection is unrelated to creditworthiness |

Only score-rejected applications are considered for potential acceptance under new cutoffs. Policy rejections are excluded because they would be rejected regardless of cutoff changes.

### Risk Metric

The primary risk metric is **b2_ever_h6** (6-month vintage delinquency rate):

```
b2_ever_h6 = multiplier × todu_30ever_h6 / todu_amt_pile_h6
```

where:
- `todu_30ever_h6`: Count of applications with 30+ day delinquency within a 6-month observation horizon
- `todu_amt_pile_h6`: Total cumulative exposure amount within the same horizon
- `multiplier`: Annualization constant (default: 7)

A complementary 3-month metric (**b2_ever_h3**) uses `multiplier_h3 = 4` and the corresponding H3 columns. This is used for early-warning validation when the out-of-time period lacks a full 6-month maturation window (see [MR Validation](#out-of-time-validation-mr-period)).

Risk is always non-negative (clipped at 0). Division by zero exposure yields NaN, indicating cells with no volume.

### N-Dimensional Binning

Each scoring variable is discretized into ordered bins. Two methods are available:

| Method | Description | When to use |
|---|---|---|
| **Quantile** | Equal-count bins from the empirical distribution | Basic 2D models where you want statistical stability in every cell |
| **Optimization** | Production-weighted risk splits via `DecisionTreeRegressor`, finding cut points that maximize separation of the risk metric | Critical for N>2 variables (e.g., adding income) — finds the bin boundaries where risk/production diverges most |

Bin configuration is specified per variable via `BinConfig`, which includes `source_col`, `output_col`, `bin_edges` or `max_bins`, and `method`.

Each variable has an associated **direction** that encodes the risk ordering of its bins:

- **Inverted** (direction = -1, variable in `inv_vars`): Higher bin index = **safer**. Common for credit scores (higher score = better creditworthiness).
- **Default** (direction = 1, variable not in `inv_vars`): Higher bin index = **riskier**.

Directions control the monotonicity constraints in the MILP.

### Annualization

All indicator columns (`todu_30ever_h6`, `todu_amt_pile_h6`, `oa_amt_h0`) are scaled by an **annual coefficient** before aggregation:

```
annual_coef = 12 / n_months_in_observation_period
```

This normalizes risk and production to a 12-month equivalent, making results comparable across observation periods of different lengths.

### Risk Adjustment Pipeline

The repesca population has no observed outcomes. Their risk and production must be estimated through a sequential correction pipeline.

#### Risk Model Prediction

A regression model is trained on booked applications to predict `b2_ever_h6` as a function of the score variables (see Phase 4). For repesca records, the trained model predicts `b2_ever_h6`, from which `todu_30ever_h6` is back-calculated:

```
todu_30ever_h6 = b2_ever_h6 × todu_amt_pile_h6 / multiplier
```

#### Stress Factor

The stress factor adjusts repesca risk predictions upward based on the observed risk concentration in the riskiest tail of the booked population.

**Global mode** (default): A single scalar computed from the entire booked population:

```
stress_factor = worst_bad_rate / overall_bad_rate
```

where `worst_bad_rate` is the risk rate of the bottom 5% of booked applications (by score), and `overall_bad_rate` is the risk rate of all booked applications. Applied multiplicatively:

```
b2_ever_h6_repesca = stress_factor × model.predict(X)
```

**Per-bin mode** (`stress_mode = "per_bin"`): Computes a separate stress factor for each unique combination of binning variables:

```
stress_factor_bin = worst_bad_rate_bin / overall_bad_rate_bin
```

This captures the fact that tail risk concentration varies across the score grid. Bins with fewer than 20 booked records fall back to the global stress factor. Recommended when score variables span very different risk profiles.

**Disabled mode** (`stress_mode = "disabled"`): Sets `stress_factor = 1.0`. Appropriate when reject inference parceling is active (to avoid double-counting selection bias) or when the model is trained on through-the-door data.

#### Reject Inference (Parceling)

After stress adjustment, an optional reject inference correction further uplifts repesca risk based on per-bin acceptance rates.

**Acceptance Rate Computation.** For each unique combination of binning variables:

```
acceptance_rate = n_booked / (n_booked + n_score_rejected)
```

Only score rejections (`09-score`) count in the denominator. Policy rejections (`08-other`) are excluded because their rejection is unrelated to the credit score cutoff. When `reject_include_all_rejections = true`, all rejections are included instead.

**Bayesian Smoothing** (optional, `reject_bayesian_smoothing = true`): Stabilizes noisy rates using a Beta-Binomial posterior:

```
global_rate = total_booked / total_demand
alpha = prior_strength × global_rate
beta  = prior_strength × (1 - global_rate)
smoothed_rate = (n_booked + alpha) / (n_total + alpha + beta)
```

The `reject_bayesian_prior_strength` parameter (default 10.0) controls shrinkage: low (1-5) gives minimal shrinkage, medium (10-50) stabilizes bins with < 50 observations, high (100+) pulls all bins strongly toward the global rate.

**Parceling Methods.** A risk multiplier is applied to `todu_30ever_h6` only — production amounts (`oa_amt_h0`) are not adjusted since production is fully observable. Three functional forms are available:

**Linear** (default): `multiplier = 1 + uplift_factor × (1 - acceptance_rate)`

| Acceptance rate | Factor = 1.5 | Interpretation |
|---|---|---|
| 1.0 (all accepted) | 1.0× | No uplift needed |
| 0.7 | 1.45× | Moderate uplift |
| 0.3 | 2.05× | Substantial uplift |
| 0.0 (all rejected) | 2.50× | Maximum linear uplift |

Steady, interpretable penalty. Best for general use.

**Power**: `multiplier = (1 / clip(acceptance_rate, 0.01, 1.0)) ^ uplift_factor`

| Acceptance rate | Factor = 1.0 | Factor = 1.5 |
|---|---|---|
| 1.0 | 1.0× | 1.0× |
| 0.5 | 2.0× | 2.83× |
| 0.2 | 5.0× | 11.2× |
| 0.1 | 10.0× | 31.6× (capped) |

Grows super-linearly at low acceptance rates. Best for heavy-tail risk when risk concentrates in the rejection tail.

**Sigmoid**: `multiplier = 1 + uplift_factor / (1 + exp(10 × (acceptance_rate - 0.5)))`

| Acceptance rate | Factor = 1.5 |
|---|---|
| 1.0 | ~1.00× |
| 0.7 | ~1.04× |
| 0.5 | 1.75× |
| 0.3 | ~2.46× |
| 0.0 | ~2.50× |

Smooth S-curve: gentle at the extremes, steep transition around 50%. Best when the risk-selectivity relationship saturates — very selective bins are not infinitely riskier, and fully accepting bins still have some baseline uncertainty.

All multipliers are floored at 1.0 and capped at `reject_max_risk_multiplier` (default 3.0).

**Monotonicity Enforcement** (optional, `reject_enforce_monotonicity = true`): Post-processes multipliers using `sklearn.isotonic.IsotonicRegression` to ensure they are non-decreasing along each variable axis (marginal monotonicity). The direction of monotonicity respects each variable's risk ordering. Enforcement operates via alternating projections: for each variable, multipliers are averaged across the other axes, fit with isotonic regression along that variable's sorted bins, and mapped back. Iterates until convergence (max change < 1e-6) or 10 iterations.

**Per-Bin Confidence Scores**: Each bin receives a confidence score: `confidence = 1 - exp(-n_total_effective / 50)`.

| Total observations | Confidence |
|---|---|
| 10 | 0.18 |
| 25 | 0.39 |
| 50 | 0.63 |
| 100 | 0.86 |
| 200 | 0.98 |

In outputs, confidence diagnostics include:
- `ri_bin_count`: raw integer count (booked + rejected) for backward compatibility
- `ri_bin_count_raw`: explicit raw integer count
- `ri_bin_count_effective`: time-decay effective sample size (float)

Low-confidence bins (< 0.5) have sparse effective evidence. Confidence diagnostics are auxiliary metadata and are not used as MILP coefficients or constraints.

**Guardrails:**

| Guardrail | Mechanism |
|---|---|
| **Unseen bins** | Repesca bins with no matching demand data receive the median observed acceptance rate (or 0.5 if unavailable). When Bayesian smoothing is active, the smoothed median is used. |
| **Floor** | Multiplier is floored at 1.0 — risk is never adjusted downward. |
| **Cap** | Multiplier is capped at `reject_max_risk_multiplier` (default 3.0). |
| **Monotonicity** | When enabled, isotonic regression enforces non-decreasing multipliers along each variable axis. |

#### Transformation Rate (tasa_fin)

The transformation rate converts demand-level indicators to expected booked-equivalent values:

```
tasa_fin = total_booked_amount / total_eligible_amount
```

Applied to repesca only after reject inference — all repesca indicator columns are scaled by `tasa_fin`:

```
repesca_indicators *= tasa_fin
```

This adjusts for the fact that not all accepted applications will ultimately be disbursed.

**Per-bin** (`per_bin_tasa_fin = true`): Computes a separate rate per grid cell. Bins with fewer than 10 eligible records or with invalid rates (non-positive or > 5.0) fall back to the global rate. Recommended when segments have markedly different conversion patterns.

#### Integration Sequence

The full adjustment sequence for repesca, applied to the per-bin aggregated summary:

```
1. Aggregate repesca by N-dimensional grid (sum indicators × annual_coef)
2. Predict risk via trained model → b2_ever_h6 → todu_30ever_h6
   (stress factor is embedded in the prediction step — global or per-bin)
3. Apply reject inference parceling multiplier to todu_30ever_h6
4. Scale ALL indicators by tasa_fin (global or per-bin)
5. Merge with booked summary → combined grid for MILP
```

For booked data, no model prediction or adjustment is needed — observed values are used directly (aggregated and annualized).

### MILP Optimization

The core optimization finds the acceptance policy that maximizes production subject to a risk budget, enforcing monotonicity across the score grid.

#### Problem Formulation

The `CellGrid` class normalizes the aggregated data into a problem space where each unique combination of bins is a "cell" with a binary decision variable.

**Decision variables:**
```
x[i] ∈ {0, 1}  for each cell i in the N-dimensional grid
```

where `x[i] = 1` means all applications in cell `i` are accepted. Each cell contains aggregated Production (`oa_amt_h0`), Risk Numerator (`todu_30ever_h6`), and Risk Denominator (`todu_amt_pile_h6`).

**Objective** (maximize production):
```
maximize  Σ oa_amt_h0[i] × x[i]
```

**Risk constraint** (linearized): The true constraint is a ratio `multiplier × Σ(todu_30ever × x) / Σ(todu_amt_pile × x) ≤ target/100`, which is non-linear. Since the multiplier is a constant scalar, it can be rearranged into linear form:

```
Σ (multiplier × todu_30ever_h6[i] - (target_risk / 100) × todu_amt_pile_h6[i]) × x[i] ≤ 0
```

Each cell contributes a coefficient that is positive when the cell's risk exceeds the target and negative when below.

#### Monotonicity Constraints

For each pair of adjacent cells along each dimension:
```
x[riskier_cell] - x[safer_cell] ≤ 0
```

This ensures: if a safer cell is rejected, all riskier cells along that dimension must also be rejected. The result is a "staircase" acceptance pattern. Monotonicity is enforced **marginally** (per dimension independently), not jointly.

#### Swap-In Constraints

Two optional constraints limit the contribution of repesca to the optimized portfolio:

1. **Production fraction cap** (`max_swapin_production_pct`):
   ```
   Σ (oa_amt_h0_rep[i] - (pct/100) × oa_amt_h0[i]) × x[i] ≤ 0
   ```

2. **Swap-in risk cap** (`max_swapin_risk`):
   ```
   Σ (multiplier × todu_30ever_h6_rep[i] - (max_risk/100) × todu_amt_pile_h6_rep[i]) × x[i] ≤ 0
   ```

Both are added as inequality rows alongside risk and monotonicity constraints. When `None`, the MILP behaves as before. If too tight, the solver returns infeasible.

#### Solver and Fallbacks

The MILP is solved using `scipy.optimize.milp` with a configurable time limit (default 30 seconds). If infeasible, the system falls back to:
- **Genetic algorithm** (N > 2 variables) via `pymoo` — near-optimal Pareto frontiers
- **Legacy enumeration** (2 variables) — brute-force evaluation of all monotonic combinations

**Output Translation**: `mask_to_cutoffs()` translates the binary vector `x` back into actionable business rules. For 2D grids, it determines the maximum accepted external score bin per internal score bin. For N>2 grids, it returns: `_cells` (lossless cell-level dict preserving the full mask), `_marginal_{var}` (per-dimension acceptance), and conditional cutoffs for the last dimension.

### Pareto Frontier Construction

#### Risk Sweep

1. Compute `max_risk` = `b2_ever_h6` when all cells are accepted.
2. Create `n_points` (default 50) evenly spaced risk targets from 0.01% to `max_risk × 1.1`.
3. Solve the MILP at each target. Deduplicate solutions by acceptance mask.
4. Evaluate KPIs for each unique solution.

#### Pareto Filtering

Solutions are sorted by ascending risk and filtered for Pareto optimality:

1. **Monotone production filter**: Keep solution `i` only if its production exceeds all solutions with lower risk.
2. **Full dominance filter**: Remove any solution dominated by another (lower or equal risk AND higher or equal production, with at least one strict inequality).

The resulting frontier has strictly increasing production as risk increases.

#### Scenario Selection

Three operating points are selected from the Pareto frontier:

| Scenario | Risk target |
|---|---|
| Pessimistic | `optimum_risk - risk_step` |
| Base | `optimum_risk` |
| Optimistic | `optimum_risk + risk_step` |

For each scenario, the system generates cutoff summaries, bootstrap confidence intervals, MR validation, and stability metrics.

### Out-of-Time Validation (MR Period)

When a separate validation period is configured, the system validates the selected cutoffs on data not used during optimization.

#### Risk Source Priority

For each bin, the pipeline selects a risk estimate following a strict priority:

| Priority | Source | Condition | Method |
|:---------|:-------|:----------|:-------|
| 1 | `h3_extrapolated` | H3 data exists, `n_obs_mr_h3 ≥ min_obs`, and `b2_main_h3` is non-zero | MR-observed H3 scaled by main-period H6/H3 ratio |
| 2 | `mr_observed` | `n_obs_mr ≥ min_obs` and H3 extrapolation was not triggered | Direct MR-period `b2_ever_h6` |
| 3 | `main_imputed` | Main-period bin exists but MR evidence is insufficient | Main-period `b2_ever_h6` |
| 4 | `model_fallback` | Bin absent from main period and `n_obs_mr < min_obs` | Inferred via the trained risk model |

Each bin's selected source is logged in the `risk_source` column for auditability.

#### H3 → H6 Extrapolation

The 3-month horizon indicator (`b2_ever_h3`) matures in half the time of H6. In a 6-month MR window, mature H3 is usually available for more cohorts, so the pipeline can extrapolate:

```
h6_h3_ratio = b2_main / b2_main_h3
b2_ever_h6 = extrapolate_h3_to_h6(b2_mr_h3, h6_h3_ratio, method, curvature)
```

| Method | Formula | When to use |
|---|---|---|
| `linear` (default) | `b2_mr_h3 × ratio` | Proportional H6/H3 relationship |
| `power` | `b2_mr_h3 × ratio × (b2_mr_h3/b2_main_h3)^(α-1)` | Convex (α > 1) or concave (< 1); falls back to linear per bin when `b2_main_h3` unavailable |
| `logistic` | `b2_mr_h3 × (1 + 2·tanh(k·(ratio-1)/2)/k)` | Caps extreme ratios smoothly |
| `auto` | Fits curvature from main-period data | Recommended — no manual tuning needed |

**Auto-calibration** (`method = "auto"`): Performs a weighted log-log regression on main-period bins:

```
log(b2_h6) = c + α × log(b2_h3)
```

If α's 95% CI includes 1.0, selects `linear`; otherwise selects `power` with fitted α (clipped to [0.3, 3.0]). Requires ≥ 4 valid bins; falls back to linear if insufficient. The fitted curvature indicates the risk maturation pattern:

- `α ≈ 1.0`: Linear maturation
- `α > 1.0`: Convex (risk accelerates with time)
- `α < 1.0`: Concave (risk saturates)

**Safeguards:** Bins where `b2_main_h3 ≈ 0` skip extrapolation. For the `power` method, bins where `b2_main_h3` is NaN (e.g., MR-only bins with no main-period data) automatically fall back to `linear` extrapolation for those specific bins rather than producing NaN. The number of MR accounts with mature H3 must meet `mr_min_obs_per_bin`. Auto-calibration requires non-degenerate log-H3 design.

#### Worked Example

Consider bin (octroi=2, efx=5):

**Main period (fully mature):**
- `todu_30ever_h6` = 100, `todu_amt_pile_h6` = 1000 → `b2_main` = 7 × 100/1000 = 70%
- `todu_30ever_h3` = 45, `todu_amt_pile_h3` = 900 → `b2_main_h3` = 4 × 45/900 = 20%
- **H6/H3 ratio:** 70% / 20% = 3.5

**MR period (H3 mature, H6 immature):**
- `todu_30ever_h3` = 8, `todu_amt_pile_h3` = 160 → `b2_mr_h3` = 4 × 8/160 = 20%
- **Extrapolated H6 risk (linear):** 20% × 3.5 = **70%**
- **Extrapolated H6 risk (power, α=1.3):** 20% × 3.5^1.3 = **98.8%**

With `mr_extrapolation_method = "auto"`, the curvature α is fitted from data — if the H6/H3 relationship is convex (α > 1), the estimate adjusts accordingly.

#### Revenue Prediction

For the MR period, `todu_amt_pile_h6` (exposure) is predicted using a regression model trained on the main period:

```
todu_amt_pile_h6_bin = reg_todu_amt_pile.predict(oa_amt_bin)
```

This is then pro-rated back to account level:

```
todu_amt_pile_h6_account = (todu_amt_pile_h6_bin / oa_amt_bin) × oa_amt_account
```

The risk numerator is derived via inverse formula:

```
todu_30ever_h6 = b2_ever_h6 × todu_amt_pile_h6 / multiplier_h6
```

#### Risk Production Summary

The function `calculate_metrics_from_cuts` applies the optimal cutoff solution to MR data and produces:

| Row | Risk (%) | Production (€) | Interpretation |
|---|---|---|---|
| **Actual** (booked) | Observed booked risk | Observed booked production | Current policy baseline |
| **Swap-in** (repesca accepted) | Repesca risk passing cutoff | Repesca production passing cutoff | Upside from opening cutoffs |
| **Swap-out** (booked rejected) | Booked risk failing cutoff | Booked production failing cutoff | Downside from tightening cutoffs |
| **Optimum** | Net after applying cutoff changes | Net production | Expected outcome |

When H3 data is available, each row also includes **Risk H3 (%)** — the observed 3-month risk metric. This provides an early-risk view alongside the standard H6 metric, and appears in both the HTML consolidated report (Segment Comparison — MR Period) and the Excel workbook (RP MR sheets).

#### Comparison Diagnostics

For every bin, drift metrics are computed and saved as `mr_risk_comparison{suffix}.csv`:

```
b2_delta     = b2_mr − b2_main
b2_delta_pct = (b2_delta / b2_main) × 100
```

Output columns include: `b2_main`, `b2_mr`, `n_obs_main`, `n_obs_mr`, `b2_ever_h6_tmp`, `risk_source`, `b2_delta`, `b2_delta_pct`, `mr_production`, `fitted_method`, `fitted_curvature`, and optional H3 columns (`b2_delta_h3`, `b2_delta_pct_h3`, `h6_h3_ratio`).

### Reject Inference Parameter Optimization

When `run_ri_optimizer = true`, the system automatically finds optimal `reject_uplift_factor` and `reject_max_risk_multiplier` values.

#### Calibration Target

Under the standard selection-bias model, if a bin accepted fraction `a` of its applicants, the true full-population risk is:

```
target_risk = booked_risk / a^γ
```

where:
- `booked_risk = multiplier × todu_30ever_h6_boo / todu_amt_pile_h6_boo`
- `a` = `acceptance_rate`, soft-clipped to a minimum of 0.05
- `γ` = `ri_calibration_gamma` (default 1.0)

| γ value | Behavior |
|---|---|
| 1.0 | Standard 1/a model: full correction for selection bias |
| 0.7 | Moderate correction, assumes partial sorting within bins |
| 0.5 | Conservative correction, appropriate when bins are coarse |
| → 0 | Minimal correction, target approaches booked risk |

Lower γ when bins are coarse, acceptance decisions involve factors beyond the binning variables, or the standard model produces implausibly high targets.

#### Evaluation Metric

Exposure-weighted mean squared relative error:

```
Error = Σ(wᵢ × ((predicted_riskᵢ - target_riskᵢ) / target_riskᵢ)²) / Σ(wᵢ)
```

where `wᵢ = todu_amt_pile_h6` (exposure weight). Cells with undefined or zero target risk are excluded.

**Invariant pre-computation**: Steps that do not depend on RI parameters (aggregated booked and repesca summaries) are computed once. The inner loop only re-runs: parceling → `tasa_fin` → merge → CellGrid → MILP → evaluate.

#### Selection Criteria

1. **Feasibility** — the parameter pair must produce a valid MILP solution at `optimum_risk`.
2. **Minimum calibration error** — among feasible solutions, select the lowest error.
3. **Tie-breaking** — if multiple pairs have error within 5% of the minimum, break ties by maximizing production. This favors parameters that are equally well-calibrated but less restrictive.

#### Optimization Methods

| Method | Description |
|---|---|
| **Grid search** (default) | Exhaustive evaluation over a regular grid. Ranges: `ri_uplift_range` [0.0, 5.0], `ri_max_mult_range` [1.0, 5.0]. Steps: `ri_uplift_steps` (11) × `ri_max_mult_steps` (9) = 99 combinations. Deterministic and transparent. |
| **Optuna TPE** | Tree-structured Parzen Estimator using `seed=42`. Continuous parameter ranges with `ri_optuna_n_trials` (default 100). More sample-efficient for large search spaces. |

#### Out-of-Time Validation

When MR-period data is available, the optimizer validates the best parameters on the holdout period and reports the **degradation ratio**:

```
degradation_ratio = mr_calibration_error / main_calibration_error
```

| Degradation ratio | Interpretation |
|---|---|
| ~1.0 | Stable: RI correction generalizes well |
| 1.0 – 2.0 | Moderate degradation: some temporal drift |
| > 2.0 | Significant: RI parameters may be overfit — consider lowering γ or using coarser bins |

### Stability Analysis (PSI/CSI)

Measures distribution drift between main and MR periods:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

A unified epsilon constant (`PSI_EPSILON = 0.0001`) is applied only inside the `log()` term for zero-percentage bins, while the difference term `(Actual% - Expected%)` uses the original (unmodified) proportions. This preserves the standard PSI scale and ensures non-negativity. Both PSI and CSI use the same epsilon application method (`.where()`).

| PSI range | Interpretation |
|---|---|
| < 0.1 | Stable — no significant change |
| 0.1 – 0.25 | Moderate — investigation recommended |
| ≥ 0.25 | Unstable — action required |

PSI is computed per binned variable. CSI (Characteristic Stability Index) uses the same formula applied to categorical distributions.

### Sensitivity Analysis

When enabled, the system perturbs cutoffs at configurable levels (default: ±5%, ±10%, ±20%) and re-evaluates production and risk at each perturbation. Outputs include aggregate summaries (cell flips, transitions) and per-cell flip thresholds.

### Marginal Impact

For each cell, analytically computes the effect of flipping its status (accept ↔ reject) on portfolio production and risk. Uses baseline sums computed once, then adjusts per cell — O(N) total. Output: `delta_production`, `delta_risk_pct`, `cell_production`, `cell_risk`.

### Trend Analysis and Monitoring

Monthly metrics are tracked over time with **Statistical Process Control (SPC)** anomaly detection: a lagged rolling median as center line, MAD-based scale, and t-distribution adjustment for small windows.

### Data Flow Diagram

```
Raw SAS Data
    │
    ├── Data quality checks
    ├── Fraud/out-of-norm filtering
    └── N-variable binning
    │
    ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Booked Population  │    │   Demand Population  │
│  (observed outcomes) │    │ (booked + rejected)  │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
   ┌───────┴───────┐          ┌───────┴───────┐
   │ Risk model    │          │ Extract score │
   │ training (CV, │          │ rejections    │
   │ 1SE rule)     │          │ (repesca)     │
   └───────┬───────┘          └───────┬───────┘
           │                          │
           │               ┌──────────┴──────────┐
           │               │ Predict repesca risk │
           │               │ (model + stress)     │
           │               └──────────┬──────────┘
           │                          │
           │               ┌──────────┴──────────┐
           │               │ Reject inference     │
           │               │ (parceling uplift)   │
           │               └──────────┬──────────┘
           │                          │
           │               ┌──────────┴──────────┐
           │               │ Transformation rate  │
           │               │ (tasa_fin scaling)   │
           │               └──────────┬──────────┘
           │                          │
   ┌───────┴───────┐                  │
   │ Aggregate by  │                  │
   │ grid (_boo)   │                  │
   └───────┬───────┘                  │
           │                          │
           └──────────┬───────────────┘
                      │
              ┌───────┴───────┐
              │  Merge into   │
              │  CellGrid     │
              │  (boo + rep)  │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │ MILP solve ×  │
              │ 50 risk       │
              │ targets       │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │ Pareto filter │
              │ (dominance)   │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │ Scenario      │
              │ selection     │
              │ (3 points)    │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │ Validation    │
              │ (MR, PSI,     │
              │  bootstrap)   │
              └───────────────┘
```

### Assumptions and Limitations

1. **Selection bias correction is approximate**: Reject inference assumes the acceptance rate within each bin is a sufficient statistic for the degree of selection bias. In reality, within-bin heterogeneity means the correction is an approximation.

2. **Marginal monotonicity**: The MILP enforces monotonicity per dimension independently, not jointly across dimensions. This is computationally efficient but can theoretically allow acceptance patterns that are not jointly monotone (though grid connectivity tends to prevent this in practice).

3. **Linearized risk constraint**: The risk budget constraint is linearized by treating the multiplier as a constant. This is exact when the multiplier is truly constant across all cells, which holds by construction.

4. **Stress factor assumes tail representativeness**: The stress factor extrapolates from the riskiest 5% of booked applications to the rejected population. In `per_bin` mode, the assumption is localized per grid cell. When parceling is active, consider `stress_mode = "disabled"` to avoid double-counting.

5. **Transformation rate uniformity**: By default, `tasa_fin` is a single rate applied uniformly. When `per_bin_tasa_fin = true`, per-cell rates are used instead, though bins with sparse data fall back to the global rate.

6. **Model predictions for rejected population**: The risk model is trained on booked applications and applied to rejected applications (extrapolation). The model assumes the risk-score relationship is stable across both populations (up to the stress and RI corrections).

---

## Additional Features

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

For the selected optimal solution, the pipeline runs 1,000 bootstrap resamples of the booked population, recalculating production and risk for each sample. The 2.5th and 97.5th percentiles (via `np.nanpercentile`) provide 95% confidence intervals. NaN-valued replicates (from empty-denominator bins) are excluded from percentile computation rather than corrupting the CI.

### Cell-Level Confidence Intervals

During model training, K-fold cross-validation produces per-cell prediction intervals:

1. For each fold, train on the train split, aggregate the validation split by grid bins, and predict the target variable per cell.
2. Across folds, compute mean, standard deviation, and confidence interval bounds per cell.
3. For small sample counts, interval half-widths use a Student-t critical value; otherwise a normal approximation is used.

Cells observed in fewer than 2 folds receive NaN intervals. Results are saved to `cell_level_ci.csv` and optionally displayed as uncertainty annotations on the dashboard heatmap.

### Fixed-Cell Constraints

Individual cells can be pinned as forced-accept or forced-reject before re-optimization. The MILP solver enforces these by setting `lb = ub = value` for pinned cells, while monotonicity and risk constraints remain active. If the constraints are contradictory, the solver returns `None`.

Available through:
- **`solve_with_fixed_cells()`** in `src/optimization_utils.py` for programmatic use.
- **Pin Mode** in the dashboard Cutoff Explorer: click cells to cycle through unpinned → accept → reject → unpinned, then re-optimize.

### Baseline Mode

Shows the current booked portfolio as-is with no cutoff optimization (Optimum = Actual, zero swap-in/swap-out). MR inference still runs to predict risk for immature loans. Only the base scenario is generated; sensitivity analysis and RI optimizer are skipped. Stale scenario files from previous runs are automatically cleaned up.

```bash
# Single segment
uv run python main.py --baseline

# Batch
uv run python run_batch.py --baseline -s precon

# Or via config.toml / segments.toml
# baseline_mode = true
```

### Sequential Cutoff Ordering

Enforces nested acceptance masks across segments so that less restrictive segments always accept a superset of more restrictive segments' cells (e.g., `mask_ef ⊆ mask_cd ⊆ mask_ab`).

**Configuration** in `segments.toml`:

```toml
[segments.no_premium_ef]
# No cutoff_floor_segment — tightest cutoff, optimized first (bottom-up)

[segments.no_premium_cd]
cutoff_floor_segment = "no_premium_ef"  # cd must be less restrictive than ef

[segments.no_premium_ab]
cutoff_floor_segment = "no_premium_cd"  # ab must be less restrictive than cd
```

**Ordering modes** (set in `config.toml` or via `--cutoff-ordering-mode`):

| Mode | Order | Constraint | Best when |
|:-----|:------|:-----------|:----------|
| `bottom_up` (default) | Tightest first (ef → cd → ab) | Floor: must accept previous segment's cells | Tightest segment has lowest/comparable risk target |
| `top_down` | Least restrictive first (ab → cd → ef) | Ceiling: can only accept previous segment's cells | Least restrictive segment has highest risk target |

Segments are automatically topologically sorted by dependency. Circular dependencies are detected and rejected. In parallel mode (`--parallel`), constrained segments run sequentially after unconstrained ones complete.

### Supersegments

When multiple segments share similar populations, a **supersegment** trains a single inference model on their combined data. Each segment then loads the shared model and runs optimization independently with its own `optimum_risk`. This produces more stable models for sparse segments.

When using supersegments, the risk surface tends to be flatter since the segment represents a narrow slice of the combined population. Pair `supersegment` with `run_ri_optimizer = false` and aggressive manual reject inference factors (`reject_uplift_factor = 3.0+`) to capture the specific rejection bias.

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
| `variables` | list[str] | *(required, >= 2, unique)* | Grid variables for optimization. Start with 2 core scores; if adding a 3rd (e.g., income), use `"optimization"` binning |
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

**Practical guidance**: Use explicit `bin_edges` when you have established legacy tiers. Use `"quantile"` for basic 2D models where you want equal-count distributions. Use `"optimization"` when adding N>2 variables — it trains decision trees to split boundaries where risk/production diverges most.

##### Economic Parameters

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `multiplier` | float | `7.0` | > 0 | Risk formula multiplier for H6 metric |
| `multiplier_h3` | float | `4.0` | > 0 | Risk formula multiplier for H3 metric |
| `optimum_risk` | float | `1.1` | | Target risk appetite in %. Tune per segment according to business risk limits |
| `risk_step` | float | `0.1` | | Scenario step: creates pessimistic (`optimum_risk - risk_step`) and optimistic (`optimum_risk + risk_step`). Use `0.05` for sharp frontiers, `0.2+` for starkly different strategies |
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
| `use_mr_outcomes` | bool | `false` | | Enable hybrid MR risk inference. Recommended `true` to use real H3 metrics instead of pure model imputation |
| `mr_min_obs_per_bin` | int | `30` | >= 1 | Min observations for an MR bin to qualify. Push to `50` for high volume, drop to `10` for tiny segments |
| `mr_extrapolation_method` | str | `"linear"` | | H3→H6 extrapolation: `"linear"`, `"power"`, `"logistic"`, or `"auto"`. Recommended: `"auto"` (fits curvature from data) |
| `mr_extrapolation_curvature` | float | `1.0` | 0-5 | Power exponent for `"power"` method. Ignored when `"auto"` |
| `mr_extrapolation_risk_multiplier` | float | `3.0` | > 0, ≤ 10 | Safety cap on extrapolated bin risk relative to main-period risk |
| `mr_extrapolation_hard_cap` | float | `15.0` | > 0, ≤ 100 | Hard percentage cap on extrapolated risk |

##### Stress Factor

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `stress_mode` | str | `"global"` | `"global"` (single scalar from worst 5% of booked), `"per_bin"` (per grid cell, fallback to global for bins < 20 obs), or `"disabled"` (`stress_factor = 1.0`). When parceling is active, use `"disabled"` to avoid double-counting |

##### Transformation Rate

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `per_bin_tasa_fin` | bool | `false` | Compute `tasa_fin` per grid cell instead of a single global scalar. Use when segments have markedly different conversion rates. Bins with < 10 eligible records or invalid rates fall back to global |

##### Reject Inference

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `reject_inference_method` | str | `"none"` | | `"none"` or `"parceling"`. Always use `"parceling"` to correct selection bias; `"none"` only for experimental baselines |
| `reject_parceling_method` | str | `"linear"` | | `"linear"` (steady, interpretable), `"power"` (aggressive at low acceptance), or `"sigmoid"` (smooth S-curve) |
| `reject_uplift_factor` | float | `1.5` | 0-10 | Scaling coefficient. `1.0–1.5` for data-rich segments, `2.0–4.0` for sparse segments on supersegments |
| `reject_max_risk_multiplier` | float | `3.0` | 1-10 | Upper cap for per-bin risk multiplier. Standard range `3.0–5.0` |
| `reject_bayesian_smoothing` | bool | `false` | | Beta-Binomial smoothing of acceptance rates. Enable for sparse segments with < 30 observations per bin |
| `reject_bayesian_prior_strength` | float | `10.0` | 0-1000 | Bayesian prior strength (higher = more shrinkage toward global rate). Increase to `50+` for extremely noisy data |
| `reject_enforce_monotonicity` | bool | `false` | | Isotonic regression on multipliers per variable axis. Enable for noisy segments with non-monotone raw rates |
| `reject_include_all_rejections` | bool | `false` | | Include policy rejections (`08-other`) in acceptance rate denominator. Usually `false` — only penalize score-based rejections |
| `reject_acceptance_recent_months` | int | `None` | >= 1 | If set, compute acceptance rates using only the most recent N months |
| `reject_acceptance_decay_half_life_months` | float | `None` | > 0 | If set, apply exponential time-decay to acceptance-rate counts (takes precedence over recent window) |
| `reject_acceptance_date_col` | str | `"mis_date"` | non-empty | Date column used for RI temporal windowing/decay weighting |
| `reject_apply_h3_multiplier` | bool | `false` | | If `true`, applies RI multiplier to H3 numerator as well as H6; default `false` preserves original H3 for extrapolation stability |

**Parceling formulas** (per bin):

- **Linear** (default): `multiplier = 1 + uplift_factor * (1 - rate)`
- **Power**: `multiplier = (1 / rate) ^ uplift_factor` — grows faster at low acceptance rates
- **Sigmoid**: `multiplier = 1 + uplift_factor / (1 + exp(10 * (rate - 0.5)))` — smooth S-curve, steep transition around 50% acceptance

All multipliers are floored at 1.0 and capped at `reject_max_risk_multiplier`.

##### Reject Inference Optimizer

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `run_ri_optimizer` | bool | `false` | | Enable automated RI parameter search. Enable for mature segments with enough data |
| `ri_optimizer_method` | str | `"grid"` | | `"grid"` (exhaustive, deterministic) or `"optuna"` (TPE, seed=42, more sample-efficient) |
| `ri_calibration_gamma` | float | `1.0` | 0-1 | Selection-bias exponent. Target = `booked_risk / acceptance_rate^gamma`. Lower = less aggressive correction |
| `ri_uplift_range` | list[float] | `[0.0, 5.0]` | | Search range [min, max] for `reject_uplift_factor` |
| `ri_max_mult_range` | list[float] | `[1.0, 5.0]` | | Search range [min, max] for `reject_max_risk_multiplier` |
| `ri_uplift_steps` | int | `11` | | Grid divisions for uplift (grid method) |
| `ri_max_mult_steps` | int | `9` | | Grid divisions for max multiplier (grid method) |
| `ri_optuna_n_trials` | int | `100` | 10-10000 | Number of Optuna trials |

Selection rule: minimum calibration error among feasible solutions, with ties within 5% broken by maximizing production. When MR data is available, reports `degradation_ratio = mr_error / main_error` for temporal stability validation.

**Practical guidance**: If a segment uses a supersegment (flat risk surface), keep `run_ri_optimizer = false` and set manual values. Start with `γ = 1.0` and reduce if the optimizer selects very aggressive parameters or MR validation shows degradation > 2.0.

##### MILP and Pareto Tuning

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `milp_time_limit` | float | `30.0` | > 0 | MILP solver timeout in seconds |
| `pareto_n_points` | int | `50` | 5-500 | Number of risk targets in Pareto sweep |
| `n_bootstraps` | int | `1000` | 100-50000 | Bootstrap replicates for confidence intervals |
| `cv_folds` | int | `4` | 2-10 | Cross-validation folds for model training |
| `monotonicity_relaxation_enabled` | bool | `false` | | Enable uncertainty-aware relaxation of local monotonicity constraints in sparse/ambiguous cell adjacencies |
| `monotonicity_uncertainty_min_exposure` | float | `0.0` | >= 0 | Minimum exposure threshold used by monotonicity relaxation gating |
| `monotonicity_uncertainty_z_threshold` | float | `1.0` | >= 0 | Z-score ambiguity threshold used by monotonicity relaxation gating |

##### Swap-In Constraints

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `max_swapin_production_pct` | float | `None` | 0-100 | Max % of total accepted production from swap-in. Use when risk committee limits portfolio growth from untested loans |
| `max_swapin_risk` | float | `None` | 0-100 | Max `b2_ever_h6` (%) for swap-in population only. Use as a hard stop when RI metrics might be flawed |

##### Sensitivity Analysis

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `run_sensitivity` | bool | `false` | Run sensitivity analysis after optimization. Enable before major strategy deployments |
| `sensitivity_levels` | list[float] | `[-20, -10, -5, 5, 10, 20]` | Perturbation percentages to evaluate |

##### Baseline Mode & Cutoff Ordering

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `baseline_mode` | bool | `false` | Show current portfolio as-is (no optimization, Optimum = Actual) |
| `cutoff_floor_segment` | str | `null` | Segment whose accepted cells constrain this segment (sequential ordering) |
| `cutoff_ordering_mode` | str | `"bottom_up"` | `"bottom_up"` (floor constraints) or `"top_down"` (ceiling constraints) |

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

Use fixed cutoffs when evaluating legacy business configurations or running backtests. Set `strict_validation = true` to crash on non-contiguous boundaries. `run_all_scenarios` only applies when fixed cutoffs are set.

##### Example Reject Inference Configuration

```toml
reject_inference_method = "parceling"
reject_parceling_method = "sigmoid"
reject_uplift_factor = 1.5
reject_max_risk_multiplier = 3.0

# Bayesian smoothing for small bins
reject_bayesian_smoothing = true
reject_bayesian_prior_strength = 20.0

# Enforce monotone multipliers
reject_enforce_monotonicity = true

# Auto-tune parameters with Optuna
run_ri_optimizer = true
ri_optimizer_method = "optuna"
ri_optuna_n_trials = 200
ri_calibration_gamma = 0.8
ri_uplift_range = [0.0, 5.0]
ri_max_mult_range = [1.0, 5.0]
```

---

### `segments.toml` -- Per-Segment Overrides

Defines segments for batch processing. Each segment can override **any** `config.toml` parameter — the segment config is recursively merged on top of the base config.

#### Supersegment Definition

Two supersegment kinds exist with independent semantics:

```toml
# Modelling supersegment — segments share a trained inference model
[modelling_supersegments.NAME]
segment_filters = ["segment_a", "segment_b"]

# Reporting supersegment — segments grouped in the consolidated report
[reporting_supersegments.NAME]
segment_filters = ["segment_a", "segment_b"]
```

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `segment_filters` | list[str] | *(required)* | Segment filter values belonging to this group |
| `learn_own_bin_edges` | bool | `false` | Learn bin edges from the supersegment's own population instead of using global edges |
| `bin_edges.<var_name>` | list[float] | `None` | Fixed bin edges for a variable, overriding both global and learned edges. Takes priority over `learn_own_bin_edges` |

**Fixed bin edges example** — force a specific income split for a reporting supersegment:

```toml
[reporting_supersegments.pl_new]
segment_filters = ["new_fintonic", "new_no_fintonic_ob"]
bin_edges.income_bin = [-inf, 1500.0, inf]   # [0, 1500) and [1500, max)
```

**Learned bin edges example** — learn edges from the supersegment's own population:

```toml
[reporting_supersegments.pl_new]
segment_filters = ["new_fintonic", "new_no_fintonic_ob"]
learn_own_bin_edges = true
```

For backward compatibility, a plain `[supersegments.NAME]` table is treated as both modelling and reporting.

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
| `modelling_supersegment` | str | `None` | Name of modelling supersegment (shared model training) |
| `reporting_supersegment` | str | `None` | Name of reporting supersegment (consolidated report grouping) |
| `supersegment` | str | `None` | Legacy: sets both modelling + reporting supersegment |
| `variables` | list[str] | *(from config.toml)* | Override grid variables |
| `inference_variables` | list[str] | *(from config.toml)* | Override model training variables |
| `baseline_mode` | bool | `false` | Baseline mode for this segment (no optimization) |
| `cutoff_floor_segment` | str | `None` | Segment whose accepted cells constrain this one (sequential ordering) |

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
| `optimization_utils.py` | `CellGrid`, MILP solver, Pareto filtering, `mask_to_cutoffs`, fixed-cell constraints |
| `sensitivity.py` | Sensitivity analysis, risk perturbation, cell flip thresholds, marginal impact |
| `reject_inference.py` | Acceptance rate computation, parceling (linear/power/sigmoid), Bayesian smoothing, monotonicity enforcement, per-bin confidence |
| `reject_inference_optimizer.py` | Grid search and Optuna TPE optimization over RI parameters, power-corrected calibration, MR out-of-time validation |
| `mr_pipeline.py` | MR period validation and metrics |
| `stability.py` | PSI/CSI drift detection |
| `trends.py` | Monthly metrics aggregation and anomaly detection |
| `audit.py` | Record-level classification (keep/swap-in/swap-out/rejected) |
| `consolidation.py` | Multi-segment aggregation and consolidated reporting |
| `global_optimizer.py` | MILP and greedy global portfolio allocation |
| `portfolio_owner.py` | Policy/cutoff tables and allocation constraint narratives for `run_allocation.py` |
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
| `mr_risk_comparison_{scenario}.csv` | Per-bin MR risk comparison with drift metrics and risk source |
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
| `consolidated_risk_production.html` | Portfolio-level interactive dashboard with segment comparison (main + MR periods) |
| `consolidated_risk_production.xlsx` | Management-ready Excel workbook (see below) |
| `score_discriminance.csv` | Gini and discriminance metrics per score |
| `score_discriminance_*.png` | Score discrimination plots |
| `allocation_results.csv` | Global allocation: per-segment chosen frontier row (if `run_allocation.py` was run) |
| `allocation_results_policy_cutoff_table.csv` | Policy/cutoff table for portfolio owners (same stem as `--output`) |
| `allocation_results_allocation_narrative.md` | Constraint narrative for a single `--target` |
| `allocation_results_what_if.csv` | Multi-target comparison (when `--what-if` lists several targets) |
| `allocation_results_allocation_narratives.md` | Multi-target constraint narratives |

#### Excel Workbook Sheets

| Sheet | Content |
|:------|:--------|
| **Executive Summary** | KPI cards (main + MR), base-scenario summary tables, top segment opportunities, acceptance grids |
| **Portfolio Summary** | TOTAL and supersegment rows across all scenarios and periods |
| **Segment Detail** | Per-segment rows with actual/optimum production, risk (H6 + H3), rejection rates |
| **Cutoff Comparison** | Concatenated cutoff data across all segments |
| **Grid {segment}** | Acceptance/rejection grid per segment with A/R colored cells |
| **RP {segment}** | Main-period risk production summary (Actual / Swap-in / Swap-out / Optimum / Summary) |
| **RP MR {segment}** | MR-period risk production summary with observed H6 and H3 risk |

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

# 6. (Optional) Global risk allocation (policy table + narrative written beside --output)
uv run python run_allocation.py --target 1.0
# uv run python run_allocation.py --what-if 1.0,1.5,2.0 --output allocation_results.csv

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

# 1. Define modelling supersegment (shared model training)
[modelling_supersegments.total]
segment_filters = ["segment_a_filter", "segment_b_filter", "segment_c_filter"]

# 2. Define reporting supersegments (consolidated report grouping)
[reporting_supersegments.group_new]
segment_filters = ["segment_a_filter", "segment_b_filter"]
bin_edges.income_bin = [-inf, 1500.0, inf]   # fixed bin edges for this group

[reporting_supersegments.group_known]
segment_filters = ["segment_c_filter"]
learn_own_bin_edges = true                    # learn from own population

# 3. Each segment references its supersegments independently
[segments.segment_a]
segment_filter = "segment_a_filter"
modelling_supersegment = "total"
reporting_supersegment = "group_new"
optimum_risk = 1.1

[segments.segment_b]
segment_filter = "segment_b_filter"
modelling_supersegment = "total"
reporting_supersegment = "group_new"
optimum_risk = 1.3

[segments.segment_c]
segment_filter = "segment_c_filter"
modelling_supersegment = "total"
reporting_supersegment = "group_known"
optimum_risk = 1.0
```

```bash
# run_batch.py automatically:
# 1. Trains model on combined (segment_a + segment_b + segment_c) data
# 2. Applies per-reporting-supersegment bin edges
# 3. Runs optimization per segment with the shared model
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
| `test_portfolio_owner.py` | Policy/cutoff tables and allocation narratives |
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

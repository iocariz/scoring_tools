# Reject Inference Methodology

## Problem Statement

A credit risk model trained exclusively on accepted (booked) applications suffers from **selection bias**: the model never observes outcomes for rejected applicants. When this model is used to predict risk for the score-rejected population (*repesca*), it systematically underestimates their true risk because the training data only represents the lower-risk tail that passed the original cutoff.

Reject inference corrects this bias by adjusting the predicted risk of repesca applicants upward, proportionally to how selective the original acceptance process was in each score bin.

### Modules

| Module | Role |
|---|---|
| `src/reject_inference.py` | Computes per-bin acceptance rates (with optional Bayesian smoothing), applies risk multipliers (linear/power/sigmoid), enforces monotonicity, and computes per-bin confidence scores |
| `src/reject_inference_optimizer.py` | Grid-search and Optuna TPE optimization of uplift parameters against a power-corrected calibration target, with out-of-time MR validation |

---

## 1. Supported Methods

Controlled by `reject_inference_method` in `config.toml` / `PreprocessingSettings`:

| Method | Behavior |
|---|---|
| `"none"` (default) | No adjustment. Repesca predictions used as-is from the base risk model. |
| `"parceling"` | Per-bin risk multiplier derived from the empirical acceptance rate. |

---

## 2. The Parceling Approach

The parceling method rests on a core assumption: the true risk of a subpopulation is inversely related to how aggressively the system filtered applicants in that bin. A bin that accepted only 30% of its demand saw the model reject the riskiest 70% — so the raw model prediction for the full population (including those rejected applicants) must be scaled up.

### 2.1 Computing Acceptance Rates

For each unique combination of binning variables (e.g., `["sc_octroi_new_clus", "new_efx_clus"]`), an acceptance rate is computed from the demand population:

```
acceptance_rate = n_booked / (n_booked + n_score_rejected)
```

**Key detail — only score rejections count.** The denominator includes only `09-score` rejections (applicants rejected by the credit-risk cutoff). Applicants rejected for other reasons (`08-other` — fraud, documentation, policy rules) are excluded entirely because:

1. Their rejection is unrelated to creditworthiness.
2. They are not candidates for cutoff-based optimization.
3. Including them would artificially deflate acceptance rates and over-correct risk.

The system logs a warning when non-score rejections are a significant fraction of demand, as their exclusion may overstate acceptance rates in bins with many policy rejections.

#### Bayesian Smoothing (Optional)

When `reject_bayesian_smoothing = true`, raw per-bin acceptance rates are stabilized using a Beta-Binomial posterior:

```
global_rate = total_booked / total_demand
alpha = prior_strength × global_rate
beta  = prior_strength × (1 - global_rate)
smoothed_rate = (n_booked + alpha) / (n_total + alpha + beta)
```

The `reject_bayesian_prior_strength` parameter (default 10.0) controls the degree of shrinkage:
- **Low strength (1-5)**: Minimal shrinkage, smoothed rates close to raw rates.
- **Medium strength (10-50)**: Moderate shrinkage, stabilizes bins with fewer than ~50 observations.
- **High strength (100+)**: Strong shrinkage, all bins pulled heavily toward the global rate.

**When to use Bayesian smoothing:**
- When the grid has many small bins (< 30 observations) producing noisy acceptance rates.
- When neighboring bins have wildly different raw rates despite similar risk profiles.
- When the parceling adjustment produces erratic multiplier patterns across the grid.

When smoothed rates are present, they are used in place of raw rates for the multiplier computation. The raw `acceptance_rate` column is preserved for diagnostics.

### 2.2 Applying Risk Uplift

A scaling multiplier is applied to the risk numerator `todu_30ever_h6`. Revenue indicators (`oa_amt_h0`) are **not** adjusted — production amounts are fully observable regardless of acceptance status.

Three functional forms are available, controlled by `reject_parceling_method`:

#### Linear (default)

```
multiplier = 1 + reject_uplift_factor × (1 - acceptance_rate)
```

| Acceptance rate | Factor = 1.5 | Interpretation |
|---|---|---|
| 1.0 (all accepted) | 1.0× | No uplift needed |
| 0.7 | 1.45× | Moderate uplift |
| 0.3 | 2.05× | Substantial uplift |
| 0.0 (all rejected) | 2.50× | Maximum linear uplift |

The relationship is purely linear: each percentage point of rejection adds a fixed increment to the multiplier.

#### Power

```
multiplier = (1 / acceptance_rate) ^ reject_uplift_factor
```

| Acceptance rate | Factor = 1.0 | Factor = 1.5 | Interpretation |
|---|---|---|---|
| 1.0 | 1.0× | 1.0× | No uplift |
| 0.5 | 2.0× | 2.83× | Moderate uplift |
| 0.2 | 5.0× | 11.2× | Heavy uplift (capped) |
| 0.1 | 10.0× | 31.6× | Extreme (capped) |

The power method grows super-linearly at low acceptance rates. It is grounded in the assumption that rejected applicants are drawn from the riskier tail of the distribution, so risk scales as a power law of the inverse acceptance rate. The acceptance rate is internally clamped to a minimum of 0.01 to avoid division by zero.

#### Sigmoid

```
multiplier = 1 + reject_uplift_factor / (1 + exp(steepness × (acceptance_rate - midpoint)))
```

With `steepness = 10` and `midpoint = 0.5`.

| Acceptance rate | Factor = 1.5 | Interpretation |
|---|---|---|
| 1.0 | ~1.00× | Virtually no uplift |
| 0.7 | ~1.04× | Gentle uplift |
| 0.5 | 1.75× | Midpoint: half of max uplift |
| 0.3 | ~2.46× | Near-full uplift |
| 0.0 | ~2.50× | Full uplift (asymptotic) |

The sigmoid produces a smooth S-curve: gentle at the extremes (rates near 0% or 100%), with a steep transition around the midpoint (50%). This avoids the abrupt behavior of the linear method at high acceptance rates and the explosive growth of the power method at low rates.

**When to use sigmoid:** When the relationship between selectivity and risk is expected to saturate at the extremes — very selective bins are not infinitely riskier, and fully accepting bins still have some baseline model uncertainty.

### 2.3 Monotonicity Enforcement (Optional)

When `reject_enforce_monotonicity = true`, the raw multipliers are post-processed using `sklearn.isotonic.IsotonicRegression(increasing=True)` to ensure they are non-decreasing along each variable axis (marginal monotonicity).

**Why enforce monotonicity:** Raw multipliers can be non-monotone when acceptance rates fluctuate across bins due to sampling noise or policy artifacts. A lower-risk bin receiving a higher multiplier than a higher-risk bin is economically incoherent and can distort the optimization. Isotonic regression resolves this by fitting the smallest non-decreasing function that minimizes squared error against the raw multipliers.

The enforcement operates per-variable axis: for each binning variable, the multipliers are averaged across the other axes, fit with isotonic regression along that variable's sorted bins, and the isotonic values are mapped back. After isotonic adjustment, multipliers are re-clipped to `[1.0, reject_max_risk_multiplier]`.

### 2.4 Per-Bin Confidence Scores

Each bin receives a confidence score based on the total number of observations (booked + score-rejected):

```
confidence = 1 - exp(-n_total / scale)
```

where `scale = 50`. This produces:

| Total observations | Confidence |
|---|---|
| 10 | 0.18 |
| 25 | 0.39 |
| 50 | 0.63 |
| 100 | 0.86 |
| 200 | 0.98 |

The confidence score and bin count (`ri_confidence`, `ri_bin_count`) are included in the parceling output for diagnostics. They are dropped before downstream optimization (MILP) to avoid introducing non-indicator columns into the grid.

**Interpretation:** Low-confidence bins (< 0.5) have sparse data and should be interpreted cautiously. When combined with Bayesian smoothing, these are precisely the bins whose rates are most aggressively shrunk toward the global rate.

### 2.5 Guardrails

| Guardrail | Mechanism |
|---|---|
| **Unseen bins** | Repesca bins with no matching demand data receive the median observed acceptance rate as a fallback (or 0.5 if the median is unavailable). When Bayesian smoothing is active, the smoothed median rate is used. |
| **Floor** | The multiplier is floored at 1.0 — risk is never adjusted downward. |
| **Cap** | The multiplier is capped at `reject_max_risk_multiplier` (default 3.0) to prevent extreme estimates in near-zero acceptance bins. |
| **Monotonicity** | When enabled, isotonic regression enforces non-decreasing multipliers along each variable axis. |

### 2.6 Pipeline Integration

Reject inference is applied **after** the stress factor and **before** the financing rate (`tasa_fin`). The sequence within the optimization pipeline is:

```
1. Aggregate booked summary  (per-bin todu + production)
2. Aggregate repesca summary (per-bin todu + production)
3. Apply risk model predictions to repesca
4. Apply stress factor
5. ► Apply reject inference (multiplier on todu_30ever_h6)    ◄
     a. Compute acceptance rates (with optional Bayesian smoothing)
     b. Compute per-bin confidence scores
     c. Apply parceling adjustment (linear/power/sigmoid)
     d. Enforce monotonicity (if enabled)
6. Drop diagnostic columns (acceptance_rate, smoothed_acceptance_rate,
   reject_risk_multiplier, ri_confidence, ri_bin_count)
7. Apply financing rate (tasa_fin) to indicators
8. Merge booked + repesca
9. Build CellGrid → MILP → Pareto
```

---

## 3. Parameter Optimization

When `run_ri_optimizer = true` in config, the system automatically finds the best `reject_uplift_factor` and `reject_max_risk_multiplier` using either grid search or Optuna TPE.

### 3.1 Theoretical Calibration Target

Under the standard selection-bias model, if a bin accepts the least-risky fraction *a* of its applicants, the true full-population risk is:

```
target_risk = booked_risk / a^γ
```

where:
- `booked_risk = multiplier × todu_30ever_h6_boo / todu_amt_pile_h6_boo`
- *a* = `acceptance_rate`, soft-clipped to a minimum of 0.05 to avoid explosive targets
- *γ* = `ri_calibration_gamma` (default 1.0)

#### Power-Corrected Calibration (γ parameter)

The classic model uses `γ = 1.0`, which assumes that the accepted fraction represents a uniform draw from the risk distribution within each bin — i.e., all applicants in a bin have identical risk, and acceptance selects the least-risky ones perfectly.

In practice, applicants within a bin are only partially sorted by risk. Setting `γ < 1` relaxes the correction:

| γ value | Behavior |
|---|---|
| 1.0 | Standard 1/a model: full correction for selection bias |
| 0.7 | Moderate correction, assumes partial sorting within bins |
| 0.5 | Conservative correction, appropriate when bins are coarse |
| → 0 | Minimal correction, target approaches booked risk |

**When to adjust γ:** Lower γ when bins are coarse (few bins, wide ranges), when acceptance decisions involve factors beyond the binning variables (manual overrides), or when the standard model produces calibration targets that exceed reasonable risk levels.

### 3.2 Invariant Pre-computation

Steps that do not depend on RI parameters are computed once via `compute_pre_reject_inference_data`:

- Aggregated booked summary (with `_boo` suffix)
- Aggregated repesca summary before RI adjustment

The inner loop only re-runs: parceling adjustment → `tasa_fin` → merge → CellGrid → MILP solve → evaluate.

### 3.3 Evaluation Metric

For each candidate parameter pair, the optimizer computes the **exposure-weighted mean squared relative error**:

```
Error = Σ(wᵢ × ((predicted_riskᵢ - target_riskᵢ) / target_riskᵢ)²) / Σ(wᵢ)
```

where:
- `predicted_risk` = blended risk after RI (booked + RI-corrected repesca)
- `target_risk` = `booked_risk / acceptance_rate^γ` (the theoretical true risk)
- `wᵢ` = `todu_amt_pile_h6` (exposure weight)

Cells with undefined or zero target risk are excluded.

### 3.4 Selection Criteria

1. **Feasibility** — the parameter pair must produce a valid MILP solution at the configured `optimum_risk` target.
2. **Minimum calibration error** — among feasible solutions, select the lowest error.
3. **Tie-breaking** — if multiple pairs have calibration error within 5% of the minimum, break ties by maximizing production (`oa_amt_h0`). This favors parameters that are equally well-calibrated but less restrictive.

### 3.5 Optimization Methods

Two search strategies are available, controlled by `ri_optimizer_method`:

#### Grid Search (default)

Exhaustive evaluation over a regular grid of parameter combinations.

| Config key | Default | Description |
|---|---|---|
| `ri_uplift_range` | `[0.0, 5.0]` | Min/max for `reject_uplift_factor` |
| `ri_uplift_steps` | `11` | Number of grid points for uplift factor |
| `ri_max_mult_range` | `[1.0, 5.0]` | Min/max for `reject_max_risk_multiplier` |
| `ri_max_mult_steps` | `9` | Number of grid points for max multiplier |

Total combinations evaluated: `ri_uplift_steps × ri_max_mult_steps` (default: 99).

**Advantages:** Deterministic, complete coverage of the search space, results are directly comparable across runs.

#### Optuna TPE

Tree-structured Parzen Estimator using `optuna.samplers.TPESampler(seed=42)` for reproducibility.

| Config key | Default | Description |
|---|---|---|
| `ri_optuna_n_trials` | `100` | Number of trials (10–10,000) |
| `ri_uplift_range` | `[0.0, 5.0]` | Continuous range for `reject_uplift_factor` |
| `ri_max_mult_range` | `[1.0, 5.0]` | Continuous range for `reject_max_risk_multiplier` |

Uses `suggest_float` for both parameters (continuous search within the range).

**Advantages:** More sample-efficient for large search spaces, explores promising regions more densely, can find better solutions with fewer evaluations than grid search.

**When to use Optuna:** When the search space is large (many steps needed for fine resolution) or when you want to explore the parameter space more efficiently. The TPE sampler concentrates trials in promising regions after an initial exploration phase.

### 3.6 Out-of-Time MR Validation

When MR-period data (`data_booked_mr`, `data_demand_mr`, `annual_coef_mr`) is available, the optimizer automatically validates the best parameters on the holdout period:

1. Computes pre-reject-inference data for the MR period.
2. Computes MR-period acceptance rates.
3. Applies the best RI parameters from the main period to MR data via `evaluate_ri_params`.
4. Reports calibration error on both periods and the **degradation ratio**:

```
degradation_ratio = mr_calibration_error / main_calibration_error
```

| Degradation ratio | Interpretation |
|---|---|
| ~1.0 | Stable: RI correction generalizes well to the MR period |
| 1.0 – 2.0 | Moderate degradation: some temporal drift in the acceptance-rate/risk relationship |
| > 2.0 | Significant degradation: RI parameters may be overfit to the main period |

The MR validation results are appended to the optimizer CSV output for the best-parameter row.

---

## 4. Configuration Reference

All parameters live in `PreprocessingSettings` (loaded from `config.toml` or `segments.toml`):

### Core RI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reject_inference_method` | `"none"` \| `"parceling"` | `"none"` | Top-level method switch |
| `reject_parceling_method` | `"linear"` \| `"power"` \| `"sigmoid"` | `"linear"` | Functional form for the acceptance-rate-to-multiplier mapping |
| `reject_uplift_factor` | `float` [0, 10] | `1.5` | Scaling coefficient (slope for linear, exponent for power, max uplift for sigmoid) |
| `reject_max_risk_multiplier` | `float` [1, 10] | `3.0` | Hard cap on the per-bin multiplier |

### Bayesian Smoothing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reject_bayesian_smoothing` | `bool` | `false` | Enable Beta-Binomial posterior smoothing of acceptance rates |
| `reject_bayesian_prior_strength` | `float` (0, 1000] | `10.0` | Prior strength — higher values produce more shrinkage toward the global rate |

### Monotonicity Enforcement

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reject_enforce_monotonicity` | `bool` | `false` | Enforce non-decreasing multipliers along each variable axis via isotonic regression |

### Optimizer Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_ri_optimizer` | `bool` | `false` | Enable RI parameter optimization |
| `ri_optimizer_method` | `"grid"` \| `"optuna"` | `"grid"` | Optimization strategy |
| `ri_uplift_range` | `[float, float]` | `[0.0, 5.0]` | Search range for `reject_uplift_factor` |
| `ri_uplift_steps` | `int` | `11` | Grid points for uplift (grid method only) |
| `ri_max_mult_range` | `[float, float]` | `[1.0, 5.0]` | Search range for `reject_max_risk_multiplier` |
| `ri_max_mult_steps` | `int` | `9` | Grid points for max multiplier (grid method only) |
| `ri_optuna_n_trials` | `int` [10, 10000] | `100` | Number of Optuna TPE trials |
| `ri_calibration_gamma` | `float` (0, 1] | `1.0` | Power exponent for calibration target — lower values produce less aggressive targets |

**Example `config.toml`:**

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

## 5. Practical Guidance

**When to enable reject inference:**
- When the repesca (score-rejected) population is a significant fraction of demand and will be included in the optimization grid.
- When the current acceptance policy is selective (average acceptance rate well below 100%), making the selection bias material.

**When to leave it as `"none"`:**
- When the optimization does not consider repesca applicants.
- When acceptance rates are uniformly high (>90%) across all bins — the correction would be negligible.

**Choosing a parceling method:**
- **Linear** is the safest default. It produces moderate, predictable uplift and is less sensitive to extreme acceptance rates.
- **Power** is appropriate when there is strong prior belief that risk concentrates in the rejection tail (e.g., high-risk consumer lending). It produces larger adjustments for heavily filtered bins but can be aggressive — always pair it with a reasonable `reject_max_risk_multiplier` cap.
- **Sigmoid** is a good middle ground when you want smooth behavior at the extremes. It avoids the abrupt jump of linear at high rates and the explosive growth of power at low rates. Particularly suited when the risk-selectivity relationship is expected to saturate.

**When to enable Bayesian smoothing:**
- When many bins have fewer than 30–50 total observations.
- When raw acceptance rates produce erratic multiplier patterns across the grid.
- Start with `prior_strength = 10` and increase if small bins still show excessive volatility.

**When to enable monotonicity enforcement:**
- When the multiplier profile across bins is non-monotone and economically incoherent.
- When the resulting risk surface needs to satisfy regulatory monotonicity requirements.
- Always check the raw (pre-enforcement) multipliers to understand the degree of correction applied.

**Choosing an optimizer method:**
- **Grid search** for small parameter spaces (default 99 combinations) or when deterministic reproducibility is critical.
- **Optuna** for larger search spaces or when you want finer resolution without the quadratic cost of a dense grid.

**Interpreting optimizer results:**
- A low calibration error (< 0.01) indicates the RI parameters reproduce the theoretical selection-bias model well.
- If no feasible solution is found, the risk target may be too tight for the adjusted risk surface — consider relaxing `optimum_risk` or widening the parameter grid.
- The 5% tolerance band for tie-breaking means the optimizer slightly favors production over calibration precision when the two are nearly equivalent, avoiding over-conservative parameter choices.
- **MR degradation ratio near 1.0** confirms the RI calibration is temporally stable. A ratio > 2.0 suggests overfitting to the main period — consider lowering `ri_calibration_gamma` or using coarser bins.

**Adjusting calibration gamma:**
- Start with `γ = 1.0` (standard model) and reduce if the optimizer selects very aggressive parameters or if MR validation shows significant degradation.
- `γ = 0.7` is a reasonable conservative starting point when the standard model produces implausibly high risk targets for low-acceptance bins.

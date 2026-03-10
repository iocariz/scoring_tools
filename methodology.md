# Methodology: Optimal Cutoff Determination for a Credit Risk Decision Grid

## 1. Objective

This system determines **optimal acceptance cutoffs** across an N-dimensional score grid to maximize loan production while controlling portfolio risk. Each dimension of the grid represents a binned risk score (e.g., internal credit score, external bureau score, income band). The output is a monotonic acceptance policy — a set of binary accept/reject decisions per cell — that respects a configurable risk budget.

The system operates on a **Pareto frontier** of risk vs. production, where each point represents an acceptance policy that cannot improve production without accepting more risk, and vice versa.

---

## 2. Pipeline Overview

The methodology follows seven sequential phases:

```
Phase 1: Configuration
Phase 2: Data Loading
Phase 3: Preprocessing (binning, stress factor, transformation rate)
Phase 4: Risk Inference (model training, prediction for rejected population)
Phase 5: Optimization (MILP-based Pareto frontier construction)
Phase 6: Scenario Analysis (selection of operating points, validation)
Phase 7: Post-Analysis (sensitivity, RI optimization, trends, reporting)
```

Each phase is described in detail below.

---

## 3. Data Populations

The system distinguishes three application populations based on their origination outcome:

| Population | Definition | Role |
|---|---|---|
| **Booked** | Applications that were approved and disbursed | Observed outcomes (risk + production) |
| **Score-rejected (repesca)** | Applications rejected by the credit score cutoff (`reject_reason = "09-score"`) | Candidates for cutoff optimization; outcomes unobserved, must be inferred |
| **Policy-rejected** | Applications rejected for non-score reasons (fraud, documentation, policy rules; `reject_reason = "08-other"`) | Excluded from optimization — rejection is unrelated to creditworthiness |

Only score-rejected applications are considered for potential acceptance under new cutoffs. Policy rejections are excluded because they would be rejected regardless of cutoff changes.

---

## 4. Risk Metric Definition

The primary risk metric is **b2_ever_h6** (6-month vintage delinquency rate):

```
b2_ever_h6 = multiplier × todu_30ever_h6 / todu_amt_pile_h6
```

where:
- `todu_30ever_h6`: Count of applications with 30+ day delinquency within a 6-month observation horizon
- `todu_amt_pile_h6`: Total cumulative exposure amount within the same horizon
- `multiplier`: Annualization constant (default: 7)

A complementary 3-month metric (**b2_ever_h3**) uses `multiplier_h3 = 4` and the corresponding H3 columns. This is used for early-warning validation when the out-of-time period lacks a full 6-month maturation window (see Section 11).

Risk is always non-negative (clipped at 0). Division by zero exposure yields NaN, indicating cells with no volume.

---

## 5. N-Dimensional Binning

### 5.1 Bin Construction

Each scoring variable is discretized into ordered bins. Two methods are available:

| Method | Description |
|---|---|
| **Quantile** | Equal-count bins from the empirical distribution |
| **Optimization** | Production-weighted risk splits via `DecisionTreeRegressor`, finding cut points that maximize separation of the risk metric |

Bin configuration is specified per variable via `BinConfig`, which includes `source_col`, `output_col`, `bin_edges` or `max_bins`, and `method`.

### 5.2 Direction Convention

Each variable has an associated **direction** that encodes the risk ordering of its bins:

- **Inverted** (direction = -1, variable in `inv_vars`): Higher bin index = **safer**. Common for credit scores (higher score = better creditworthiness).
- **Default** (direction = 1, variable not in `inv_vars`): Higher bin index = **riskier**.

Directions control the monotonicity constraints in the MILP (Section 8.2).

---

## 6. Annualization and Observation Period

All indicator columns (`todu_30ever_h6`, `todu_amt_pile_h6`, `oa_amt_h0`) are scaled by an **annual coefficient** before aggregation:

```
annual_coef = 12 / n_months_in_observation_period
```

This normalizes risk and production to a 12-month equivalent, making results comparable across observation periods of different lengths.

---

## 7. Risk Adjustment Pipeline for Rejected Population

The repesca population has no observed outcomes. Their risk and production must be estimated. The adjustment pipeline applies three sequential corrections:

### 7.1 Risk Model Prediction (Phase 4)

A regression model is trained on booked applications to predict `b2_ever_h6` as a function of the score variables. The model selection process:

1. **Feature generation**: Polynomial features up to degree 3 (interactions, squared, cubic terms) are generated from the score variables.
2. **Model type selection**: Cross-validated comparison of Linear Regression, Ridge, Lasso, ElasticNet, TweedieGLM, HurdleRegressor, XGBoost, and LightGBM. The **one-standard-error rule** selects the simplest model within 1 SE of the best CV RMSE.
3. **Feature set selection**: The selected model type is evaluated across feature sets of increasing complexity. Again, the 1SE rule applies.
4. **Exposure prediction**: A separate `todu_amt_pile ~ oa_amt` regression provides per-bin exposure estimates.

For repesca records, the trained model predicts `b2_ever_h6`, from which `todu_30ever_h6` is back-calculated:

```
todu_30ever_h6 = b2_ever_h6 × todu_amt_pile_h6 / multiplier
```

### 7.2 Stress Factor

The stress factor adjusts repesca risk predictions upward based on the observed risk concentration in the riskiest tail of the booked population.

Three **stress modes** are available (configured via `stress_mode`):

#### Global mode (default)

A single scalar computed from the entire booked population:

```
stress_factor = worst_bad_rate / overall_bad_rate
```

where:
- `worst_bad_rate`: Risk rate of the bottom 5% of booked applications (by score)
- `overall_bad_rate`: Risk rate of all booked applications

Applied multiplicatively to the model's `b2_ever_h6` predictions for repesca:

```
b2_ever_h6_repesca = stress_factor × model.predict(X)
```

**Rationale**: The risk model is trained on the booked population, which has already been filtered. The stress factor captures the degree to which risk concentrates in the tail, providing a conservative upward adjustment for the unobserved rejected population.

#### Per-bin mode (`stress_mode = "per_bin"`)

Computes a separate stress factor for each unique combination of binning variables:

```
stress_factor_bin = worst_bad_rate_bin / overall_bad_rate_bin
```

This captures the fact that tail risk concentration varies across the score grid — bins with narrow score ranges or skewed distributions may have very different tail behavior from the global average. Each repesca cell is then stressed by its own bin-specific factor rather than the global scalar.

Bins with fewer than `min_obs_per_bin` (default 20) booked records fall back to the global stress factor for stability.

**When to use**: Per-bin stress is recommended when the score variables span very different risk profiles (e.g., external bureau score × income band) and a single global multiplier would over-stress low-risk bins while under-stressing high-risk ones.

**Interaction with parceling**: When `stress_mode = "per_bin"`, the per-bin stress factor captures some of the same selection-bias signal that parceling addresses. Using both simultaneously may result in double-counting. Consider using `stress_mode = "disabled"` when parceling is active.

#### Disabled mode (`stress_mode = "disabled"`)

Sets `stress_factor = 1.0` (no stress adjustment). Appropriate when:
- Reject inference via parceling is active and already corrects for selection bias
- The model is trained on a representative population (e.g., through-the-door data)

### 7.3 Reject Inference (Parceling)

After stress adjustment, an optional **reject inference** correction further uplifts repesca risk based on per-bin acceptance rates. See Section 9 for the full methodology.

### 7.4 Transformation Rate (tasa_fin)

The transformation rate converts demand-level indicators to expected booked-equivalent values:

```
tasa_fin = total_booked_amount / total_eligible_amount
```

This rate is computed from the `oa_amt` column across all eligible applications (booked + rejected with a valid decision), aggregated over the configured `n_months` window.

**Applied to repesca only**: After reject inference, all repesca indicator columns are scaled by `tasa_fin`:

```
repesca_indicators *= tasa_fin
```

This adjusts for the fact that not all accepted applications will ultimately be disbursed.

#### Per-bin transformation rate (`per_bin_tasa_fin = true`)

When enabled, a separate `tasa_fin` is computed for each unique combination of binning variables:

```
tasa_fin_bin = booked_amount_bin / eligible_amount_bin
```

This captures the fact that conversion rates vary across the score grid — high-risk bins may have lower transformation rates (more drop-offs, documentation failures) while low-risk, high-income bins may convert at higher rates.

Bins with fewer than `min_eligible_per_bin` (default 10) eligible records, or bins where the computed rate is non-positive or exceeds 5.0, fall back to the global `tasa_fin` for stability.

**When to use**: Per-bin tasa_fin is recommended when the population includes segments with markedly different conversion patterns (e.g., secured vs. unsecured products in the same grid, or income bands with very different documentation requirements).

### 7.5 Integration Sequence

The full adjustment sequence for repesca, applied to the per-bin aggregated summary:

```
1. Aggregate repesca by N-dimensional grid (sum indicators × annual_coef)
2. Predict risk via trained model → b2_ever_h6 → todu_30ever_h6
   (stress factor is embedded in the prediction step — global or per-bin)
3. Apply reject inference parceling multiplier to todu_30ever_h6
4. Scale ALL indicators by tasa_fin (global or per-bin)
5. Merge with booked summary → combined grid for MILP
```

When per-bin stress or per-bin tasa_fin is enabled, the corresponding DataFrame is merged onto the repesca summary by grid coordinates. Cells with no per-bin value fall back to the global scalar.

For booked data, no model prediction or adjustment is needed — observed values are used directly (aggregated and annualized).

---

## 8. MILP Optimization

The core optimization finds the acceptance policy that maximizes production subject to a risk budget, enforcing monotonicity across the score grid.

### 8.1 Problem Formulation

**Decision variables**:
```
x[i] ∈ {0, 1}  for each cell i in the N-dimensional grid
```
where `x[i] = 1` means all applications in cell `i` are accepted, and `x[i] = 0` means they are rejected.

**Objective** (maximize production):
```
maximize  Σ oa_amt_h0[i] × x[i]
```

**Risk constraint** (linearized):

The true constraint is a ratio:
```
multiplier × Σ(todu_30ever_h6[i] × x[i]) / Σ(todu_amt_pile_h6[i] × x[i]) ≤ target_risk / 100
```

This is non-linear but can be rearranged into a linear form because the multiplier is a constant scalar:
```
Σ (multiplier × todu_30ever_h6[i] - (target_risk / 100) × todu_amt_pile_h6[i]) × x[i] ≤ 0
```

Each cell contributes a coefficient `risk_coeff[i]` that is positive when the cell's risk exceeds the target and negative when below. The constraint says the exposure-weighted average risk of accepted cells must not exceed the target.

### 8.2 Monotonicity Constraints

For each pair of adjacent cells along each dimension:
```
x[riskier_cell] - x[safer_cell] ≤ 0
```

This ensures: if a safer cell is rejected, all riskier cells along that dimension must also be rejected. The result is a "staircase" acceptance pattern — moving from safer to riskier bins, there is a single cutoff point per dimension beyond which all cells are rejected.

The direction convention determines which cell is "riskier" (Section 5.2). Monotonicity is enforced **marginally** (per dimension independently), not jointly.

### 8.3 Optional Swap-In Constraints

Two additional constraints can limit the contribution of repesca to the optimized portfolio:

1. **Production fraction cap**: Repesca production cannot exceed a fraction of total production.
   ```
   Σ (oa_amt_h0_rep[i] - (pct/100) × oa_amt_h0[i]) × x[i] ≤ 0
   ```

2. **Swap-in risk cap**: The risk of the repesca component alone cannot exceed a threshold.
   ```
   Σ (multiplier × todu_30ever_h6_rep[i] - (max_risk/100) × todu_amt_pile_h6_rep[i]) × x[i] ≤ 0
   ```

### 8.4 Solver

The MILP is solved using `scipy.optimize.milp` with a configurable time limit (default 30 seconds). If infeasible, the system falls back to:
- **Genetic algorithm** (N > 2 variables) via `pymoo`
- **Legacy enumeration** (2 variables) — brute-force evaluation of all monotonic cutoff combinations

---

## 9. Reject Inference Methodology

### 9.1 Problem Statement

A risk model trained exclusively on booked applications suffers from **selection bias**: it never observes outcomes for rejected applicants. When this model predicts risk for the repesca population, it systematically underestimates their true risk because the training data only represents the lower-risk fraction that passed the original cutoff.

Reject inference corrects this bias by adjusting repesca risk upward, proportionally to how selective the original acceptance process was in each bin.

### 9.2 Acceptance Rate Computation

For each unique combination of binning variables, an acceptance rate is computed from the demand population:

```
acceptance_rate = n_booked / (n_booked + n_score_rejected)
```

Only score rejections (`09-score`) count in the denominator. Policy rejections (`08-other`) are excluded because their rejection is unrelated to the credit score cutoff.

#### Bayesian Smoothing (Optional)

When enabled (`reject_bayesian_smoothing = true`), raw rates are stabilized using a Beta-Binomial posterior:

```
global_rate = total_booked / total_demand
alpha = prior_strength × global_rate
beta  = prior_strength × (1 - global_rate)
smoothed_rate = (n_booked + alpha) / (n_total + alpha + beta)
```

The `prior_strength` parameter (default 10.0) controls shrinkage toward the global rate. Higher values produce more shrinkage, stabilizing bins with sparse data.

### 9.3 Parceling Multiplier

A risk multiplier is applied to the repesca risk numerator (`todu_30ever_h6`). Production amounts (`oa_amt_h0`) are not adjusted — production is fully observable regardless of acceptance status.

Three functional forms are available:

**Linear** (default):
```
multiplier = 1 + uplift_factor × (1 - acceptance_rate)
```

**Power**:
```
multiplier = (1 / clip(acceptance_rate, 0.01, 1.0)) ^ uplift_factor
```

**Sigmoid**:
```
multiplier = 1 + uplift_factor / (1 + exp(10 × (acceptance_rate - 0.5)))
```

All multipliers are floored at 1.0 (risk is never adjusted downward) and capped at `reject_max_risk_multiplier` (default 3.0).

### 9.4 Monotonicity Enforcement (Optional)

When enabled, isotonic regression (`sklearn.isotonic.IsotonicRegression`) ensures multipliers are non-decreasing along each variable axis. This prevents economically incoherent patterns where a lower-risk bin receives a higher multiplier than a higher-risk bin.

### 9.5 Per-Bin Confidence Scores

Each bin receives a confidence score based on observation count:

```
confidence = 1 - exp(-n_total / 50)
```

Low-confidence bins (< 0.5, corresponding to < 35 observations) have sparse data and should be interpreted cautiously. Confidence scores are diagnostic only and are dropped before the MILP.

### 9.6 Unseen Bins

Repesca bins with no matching demand data receive the median observed acceptance rate as a fallback (or the smoothed median when Bayesian smoothing is active). If the median is unavailable, 0.5 is used.

---

## 10. Pareto Frontier Construction

The Pareto frontier is built by solving the MILP at multiple risk targets and filtering for non-dominated solutions.

### 10.1 Risk Sweep

1. Compute `max_risk` = `b2_ever_h6` when all cells are accepted.
2. Create `n_points` (default 50) evenly spaced risk targets from 0.01% to `max_risk × 1.1`.
3. Solve the MILP at each target. Deduplicate solutions by acceptance mask.
4. Evaluate KPIs for each unique solution.

### 10.2 Pareto Filtering

After solving, solutions are sorted by ascending risk and filtered for Pareto optimality:

1. **Monotone production filter**: Keep solution `i` only if its production exceeds all solutions with lower risk.
2. **Full dominance filter**: Remove any solution dominated by another (lower or equal risk AND higher or equal production, with at least one strict inequality).

The resulting frontier has strictly increasing production as risk increases — each point represents a meaningful trade-off.

### 10.3 Scenario Selection

Three operating points are selected from the Pareto frontier:

| Scenario | Risk target |
|---|---|
| Pessimistic | `optimum_risk - risk_step` |
| Base | `optimum_risk` |
| Optimistic | `optimum_risk + risk_step` |

For each scenario, the system generates cutoff summaries, bootstrap confidence intervals, MR validation, and stability metrics.

---

## 11. Out-of-Time Validation (MR Period)

When a separate validation period is configured, the system validates the selected cutoffs on data not used during optimization.

### 11.1 MR Risk Assessment

1. Apply the optimized cutoffs to MR-period demand data.
2. Classify each MR application as accepted or rejected under the new policy.
3. Compute realized risk and production for both populations.
4. Generate the **Risk Production Summary Table**:

| Row | Risk | Production | Interpretation |
|---|---|---|---|
| Actual (booked) | Observed booked risk | Observed booked production | Current policy baseline |
| Swap-in (repesca accepted) | Repesca risk passing cutoff | Repesca production passing cutoff | Upside from opening cutoffs |
| Swap-out (booked rejected) | Booked risk failing cutoff | Booked production failing cutoff | Downside from tightening cutoffs |
| Optimum | Net after applying cutoff changes | Net production | Expected outcome |

### 11.2 H3 → H6 Extrapolation

When the MR period is too recent for full 6-month outcomes, the 3-month risk metric is extrapolated to 6-month equivalent:

1. **Fit**: Weighted log-log regression on main-period data:
   ```
   log(b2_h6) = c + α × log(b2_h3)
   ```
2. **Apply**: Use the fitted relationship to convert MR-period `b2_h3` to estimated `b2_h6`.

The fitted curvature `α` indicates the risk maturation pattern:
- `α ≈ 1.0`: Linear maturation
- `α > 1.0`: Convex (risk accelerates with time)
- `α < 1.0`: Concave (risk saturates)

When `mr_extrapolation_method = "auto"`, the system automatically fits the curvature from main-period data.

---

## 12. Reject Inference Parameter Optimization

When enabled (`run_ri_optimizer = true`), the system automatically finds optimal `reject_uplift_factor` and `reject_max_risk_multiplier` values.

### 12.1 Calibration Target

Under the standard selection-bias model, if a bin accepted fraction `a` of its applicants, the true full-population risk is:

```
target_risk = booked_risk / a^γ
```

where `γ = ri_calibration_gamma` (default 1.0). Lower `γ` relaxes the correction (appropriate when bins are coarse or acceptance decisions involve factors beyond the binning variables).

### 12.2 Evaluation Metric

Exposure-weighted mean squared relative error:

```
Error = Σ(wᵢ × ((predicted_riskᵢ - target_riskᵢ) / target_riskᵢ)²) / Σ(wᵢ)
```

where `wᵢ = todu_amt_pile_h6` (exposure weight).

### 12.3 Selection Criteria

1. **Feasibility**: The parameter pair must produce a valid MILP solution at the configured `optimum_risk` target.
2. **Minimum calibration error**: Among feasible solutions, select the lowest error.
3. **Tie-breaking**: If multiple pairs have error within 5% of the minimum, prefer the one that maximizes production.

### 12.4 Optimization Methods

| Method | Description |
|---|---|
| **Grid search** (default) | Exhaustive evaluation over a regular grid (default 11 × 9 = 99 combinations) |
| **Optuna TPE** | Tree-structured Parzen Estimator with continuous parameter ranges, more sample-efficient for large search spaces |

### 12.5 Out-of-Time Validation

When MR-period data is available, the optimizer validates the best parameters on the holdout period and reports the **degradation ratio**:

```
degradation_ratio = mr_calibration_error / main_calibration_error
```

A ratio near 1.0 indicates temporal stability; a ratio > 2.0 suggests overfitting.

---

## 13. Stability Analysis

### 13.1 Population Stability Index (PSI)

Measures distribution drift between two populations (e.g., main period vs. MR period):

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

| PSI range | Interpretation |
|---|---|
| < 0.1 | Stable — no significant change |
| 0.1 – 0.25 | Moderate — investigation recommended |
| ≥ 0.25 | Unstable — action required |

PSI is computed for each binned variable to detect score distribution shifts that could degrade model performance.

### 13.2 Bootstrap Confidence Intervals

Production and risk confidence intervals are computed via record-level bootstrap resampling (default 1000 iterations) on the booked portfolio, respecting the selected cutoff at each resample.

---

## 14. Sensitivity Analysis

When enabled, the system perturbs cutoffs at configurable levels (default: ±5%, ±10%, ±20%) and re-evaluates production and risk at each perturbation. This quantifies how robust the selected operating point is to small cutoff changes.

---

## 15. Trend Analysis and Monitoring

Monthly metrics are tracked over time with **Statistical Process Control (SPC)** anomaly detection. This enables ongoing monitoring of risk, production, and acceptance rate trends after deployment.

---

## 16. End-to-End Data Flow

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

---

## 17. Key Assumptions and Limitations

1. **Selection bias correction is approximate**: Reject inference assumes the acceptance rate within each bin is a sufficient statistic for the degree of selection bias. In reality, within-bin heterogeneity means the correction is an approximation.

2. **Marginal monotonicity**: The MILP enforces monotonicity per dimension independently, not jointly across dimensions. This is computationally efficient but can theoretically allow acceptance patterns that are not jointly monotone (though the grid connectivity tends to prevent this in practice).

3. **Linearized risk constraint**: The risk budget constraint is linearized by treating the multiplier as a constant. This is exact when the multiplier is truly constant across all cells, which holds by construction.

4. **Stress factor assumes tail representativeness**: The stress factor extrapolates from the riskiest 5% of booked applications to the rejected population. This assumes the risk-score relationship in the booked tail is informative about the rejected population. In `per_bin` mode, the assumption is localized per grid cell, reducing its impact. When parceling is active, consider using `stress_mode = "disabled"` to avoid double-counting selection bias corrections.

5. **Transformation rate uniformity**: By default, `tasa_fin` is a single rate applied uniformly to all repesca cells. When `per_bin_tasa_fin = true`, per-cell rates are used instead, though bins with sparse data fall back to the global rate.

6. **Model predictions for rejected population**: The risk model is trained on booked applications and applied to rejected applications (extrapolation). The model assumes the risk-score relationship is stable across both populations (up to the stress and RI corrections).

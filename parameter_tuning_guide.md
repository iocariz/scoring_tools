# Comprehensive Scoring Tools Parameter Tuning Guide

This guide explains **all possible parameters** in the scoring pipeline, grouped by their function, with practical advice on **when to use which settings**. It serves as both a reference for [config.toml](file:///Users/inigo_ocariz/src/scoring_tools/config.toml) (global defaults) and [segments.toml](file:///Users/inigo_ocariz/src/scoring_tools/segments.toml) (per-segment overrides).

---

## 1. Core Optimization & Economic Parameters

These parameters define the primary business targets and data constraints for the MILP optimizer.

### Primary Risk Targets
* **`optimum_risk`**: The target maximum portfolio risk percentage (e.g., `1.5` means 1.5% target `b2_ever_h6`). The optimizer attempts to maximize production without exceeding this overall rate. Tune this per segment according to your business risk limits.
* **`risk_step`**: The increment used to generate alternative scenarios (e.g., `0.1`). If `optimum_risk = 1.5`, the pipeline outputs "base" (1.5%), "optimistic" (1.6%), and "pessimistic" (1.4%).
  * **When to use smaller steps (`0.05`)**: When the segment's efficient frontier is very sharp and you want to evaluate highly granular cutoff changes.
  * **When to use larger steps (`0.2+`)**: When you want to present starkly different business strategies (conservative vs aggressive) to stakeholders.
* **[multiplier](file:///Users/inigo_ocariz/src/scoring_tools/src/reject_inference.py#179-237) / `multiplier_h3`**: The constants used in the risk formula (`multiplier * todu_30ever / todu_amt_pile`). Default is `7.0` for H6 and `4.0` for H3.

### Grid Variables & Features
* **[variables](file:///Users/inigo_ocariz/src/scoring_tools/src/models.py#126-148)**: The list of grid variables (e.g., internal/external score bins) used for cutoff optimization. Must be >= 2 variables.
  * **Recommendation**: Start with 2 core scores (e.g., internal custom score + external bureau score). If you add a 3rd variable (e.g., `income_bin`), ensure you run the `"optimization"` binning method so it provides maximal lift to the optimizer.
* **[inference_variables](file:///Users/inigo_ocariz/src/scoring_tools/src/config.py#414-427)**: Subset of [variables](file:///Users/inigo_ocariz/src/scoring_tools/src/models.py#126-148) used strictly for training the probability surface model.
* **`keep_vars` / `indicators`**: Required structural parameters defining columns to retain and the core target metrics to predict.
* **`date_ini_book_obs` / `date_fin_book_obs`**: Defines the main observation window for training and optimization.

---

## 2. Reject Inference (Selection Bias Correction)

Score-rejected applications inherently lack performance outcomes. The model must infer their risk. Since models are trained on (safer) accepted populations, treating rejections at face value underestimates true risk.

* **`reject_inference_method`**: `"none"` or `"parceling"`.
  * **Recommendation**: **Always** use `"parceling"` to correct selection bias. Only use `"none"` for experimental baselines.
* **`reject_parceling_method`**: The mathematical shape of the penalty.
  * **When to use `"linear"` (Default)**: For general use. It provides a steady penalty and is highly transparent/interpretable.
  * **When to use `"power"`**: When dealing with heavy-tail risk. It gets extremely aggressive at very low acceptance rates, heavily penalizing bins that are almost never funded.
  * **When to use `"sigmoid"`**: When you want a smooth S-curve penalty that transitions steeply around 50% acceptance but remains gentle at the extreme 1% and 99% edges to prevent mathematical blowups.
* **`reject_bayesian_smoothing`**: `true` / `false`.
  * **When to use (`true`)**: When your segment is sparse with low volume, where empirical acceptance rates for single grid cells might be 0/1 (0%) or 1/1 (100%), creating extreme multiplier spikes.
* **`reject_bayesian_prior_strength`**: Controls the shrinkage force (default `10.0`). Increase to `50.0` or higher if the dataset is extremely noisy.
* **`reject_enforce_monotonicity`**: `true` / `false`.
  * **When to use (`true`)**: When dealing with very noisy or small segments where the raw historical acceptance rates jump randomly back and forth across logical score bands.
* **`reject_include_all_rejections`**: `true` / `false`. Whether to include `08-other` (policy rejections) in the acceptance rate denominator.
  * **Recommendation**: Usually `false`. You only want to penalize based on people who were strictly rejected *by the score*, not by hard policy knockouts.

### Manual vs Automated Reject Tuning
* **[run_ri_optimizer](file:///Users/inigo_ocariz/src/scoring_tools/src/pipeline/optimization.py#719-892)**: When `true` (default), the pipeline uses Optuna to sweep and find the mathematically optimal `reject_uplift_factor` and `reject_max_risk_multiplier`.
  * **When to use (`true`)**: As the default for mature segments where the optimizer can safely find a stable calibration point based on the data.
* **Important Override**: If a segment's risk surface is too flat (e.g., due to supersegments pooling data), set `run_ri_optimizer = false` in [segments.toml](file:///Users/inigo_ocariz/src/scoring_tools/segments.toml) to force manual values.
* **`reject_uplift_factor` (Manual)**: Scale of the penalty (e.g., `1.5` to `4.0`). Higher means steeper risk penalization for low acceptance bins.
  * **Recommendation**: `1.0 - 1.5` for gentle extrapolation on data-rich, stable segments. `2.0 - 4.0` for aggressive extrapolation on sparse segments relying on shared [supersegments](file:///Users/inigo_ocariz/src/scoring_tools/run_batch.py#171-176).
* **`reject_max_risk_multiplier` (Manual)**: Hard cap on how many times the base predicted risk can be multiplied.
  * **Recommendation**: Standard range is `3.0 - 5.0` to prevent risk predictions from exploding to chemically unreasonable levels (like assigning a 50% default rate randomly for a single bin).

---

## 3. MR Period Extrapolation (Recent Cohorts)

The "Most Recent" (MR) period handles recently booked loans that haven't reached their mature H6 evaluation.

* **`date_ini_book_obs_mr` / `date_fin_book_obs_mr`**: Defines the recent holdout evaluation window.
* **`use_mr_outcomes`**: `true` / `false`.
  * **Recommendation**: Set to `true` to enable hybrid risk calculation. It forces the system to rely on sparse real H3 metrics instead of pure model imputation.
* **`mr_min_obs_per_bin`**: Minimum valid records required (e.g., `30`) before trusting observed MR risk.
  * **When to tune**: If you have high volume, push this up to `50`. If you have a very tiny segment, drop it to `10` to force real observations to govern, accepting the variance.
* **`mr_extrapolation_method`**: How to map immature H3 data to H6. Options: `"linear"`, `"power"`, `"logistic"`, or `"auto"`.
  * **When to use `"auto"` (Recommended)**: Always. It fits a weighted log-log regression to automatically determine the actual mathematical trajectory tailored to that exact segment based on its historical Main period data.
  * **When to use `"linear"`**: Only if the dataset is so noisy the `"auto"` regression fails and you simply want to fallback to a simple proportional extrapolation.
* **`mr_extrapolation_risk_multiplier` & `mr_extrapolation_hard_cap`**: Safety constraints capping how high an empty extrapolated bin can spike. Typical defaults are `3.0` (multiplier limit) and `15.0` (hard % cap).

---

## 4. Supersegments & Shared Models

* **[supersegment](file:///Users/inigo_ocariz/src/scoring_tools/run_batch.py#171-176)**: Tells the pipeline to skip training a bespoke model for sparse segment data and borrow a robust master model. 
  * Defined under `[segments.my_segment]` block.
  * **When to use**: Set to `"total"` for highly sparse/immature sectors (e.g., a brand new product or pilot program) where there are simply too few booked loans to train a stable mathematical surface.
  * **When NOT to use**: Do not use it for mature, high-volume segments. Let them natively train their own perfectly tailored risk curves.
  * **Tuning Note**: Borrowing a broad model generates "flatter" risk predictions since the segment represents a narrow slice of features. Always pair `supersegment="total"` with `run_ri_optimizer = false` and aggressive manual reject inference factors (`reject_uplift_factor=3.0+`) to accurately capture the specific rejection bias curve for that slice.

---

## 5. Additional Modifiers & Constraints

### Swap-In (Repesca) Limits
Caps the impact of newly accepted (previously rejected) applicants purely inside the MILP optimizer.
* **`max_swapin_production_pct`**: Hard limit on the percentage (0-100) of overall optimal production that is allowed to originate from previously rejected applicants.
  * **When to use**: When your risk-committee dictates that you cannot grow the portfolio by swapping out high quality loans for risky untested loans beyond a certain limit (e.g., max 15% swap-in growth).
* **`max_swapin_risk`**: A separate risk threshold exclusively analyzing the sub-population of swap-ins.
  * **When to use**: When you suspect your reject inference metrics might be flawed and you want a hard mathematical stop limiting the risk profile of newly integrated loans.

### Sensitivity Analysis
Evaluates the robustness of the optimized cutoffs.
* **[run_sensitivity](file:///Users/inigo_ocariz/src/scoring_tools/src/pipeline/optimization.py#641-717)**: Prints flip-threshold analysis to gauge if the model hinges on highly unstable grid boundaries.
  * **When to use**: Before major strategy deployments. It runs a flip-threshold analysis measuring how many risk-basis-points a grid cell must shift before the optimizer would have rejected it. High sensitivity means your strategy is unstable.
* **`sensitivity_levels`**: Array of perturbation shocks applied (e.g., `[-20, -10, -5, 5, 10, 20]`).

### Stress Factor & Transformation Rate
* **`stress_mode`**: How to penalize rejected applicants' underlying performance. Options: `"global"` (top 5% worst), `"per_bin"` (granular), or `"disabled"`.
  * **Recommendation**: If using powerful Reject Inference Parceling, it is generally recommended to use `"disabled"` to avoid double-penalizing the swap-in population. Use `"global"` only if parceling is off.
* **`per_bin_tasa_fin`**: If `true`, Computes the monthly financing rate fraction uniquely for every grid cell instead of a scalar assumption.
  * **When to use**: Set to `true` if you know that different credit segments finance at wildly different transformation rates (e.g., your tier 1 internal customers accept offers immediately, but bottom-tier customers drag it out). Otherwise `false` (scalar assumption).

### Binning Controls
Overrides how variables construct the grid.
* **[bin_edges](file:///Users/inigo_ocariz/src/scoring_tools/run_batch.py#99-155)**: Specify explicit numeric thresholds.
  * **When to use**: When you have established legacy tiers.
* **`max_bins` & `method`**: If edges aren't fixed, pipeline computes them securely via `"quantile"` (balanced) or `"optimization"` (decision trees optimizing risk/production separation).
  * **When to use `"quantile"`**: For basic 2D models where you want equal-count distributions ensuring statistical stability in every cell.
  * **When to use `"optimization"`**: Absolutely critical when adding N>2 variables (e.g., adding `income`). It trains small decision trees to split the bin boundaries exactly where the risk/production ratio diverges the most, feeding the MILP optimizer the most potent possible leverage points.
* **`directions`**: Forces monotonicity expectations (`1` for direct ascending risk, `-1` for descending).

---

## 6. Fixed Cutoffs (Bypass MILP)

Sometimes business rules dictate static rules explicitly bypassing algorithmic optimization. Ensure to set under `[preprocessing.fixed_cutoffs]`.
* **2-Variable Array**: Define a specific set of threshold arrays (e.g., `sc_octroi_new_clus = [1, 2, 3]`, `new_efx_clus = [4, 5, 6]`).
  * **When to use**: When evaluating legacy business configurations, or running backtests on the exact cutoff lines in production yesterday.
* **`strict_validation`**: Errors out rather than warning if constraints aren't physically contiguous.
  * **Recommendation**: Set `true` to force a hard crash if someone specifies a physically impossible or non-contiguous step boundary in the TOML file.
* **`run_all_scenarios`**: normally fixed rules only emit a "base" line. Toggle `true` to force pessimistic/optimistic generation around the fixed points.

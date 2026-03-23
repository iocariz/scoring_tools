# Enhancement Roadmap (Unresolved)

## High
### Bug
## Medium
### Bug / Correctness
- `src/optimization_utils.py` + `src/utils.py` — MILP risk budgeting guards global denominators in some paths, but NaN risk can still arise when the selected mask yields zero exposure denominators; Pareto dominance/scenario selection may not explicitly filter NaN-risk solutions. Mitigation: enforce an eps constraint on accepted exposure denominators and/or filter NaN-risk solutions before dominance + selection.
- `src/reject_inference.py` — Bayesian smoothing Beta prior for acceptance rates may be policy-outcome driven rather than Beta-distributed counts; `prior_strength` is arbitrary and can be stale across time. Mitigation: validate prior strength via backtesting or empirical Bayes.

### Statistical & Methodological
- `src/reject_inference.py` — acceptance rates are not time-weighted/decay-aware; underwriting policy drift can cause historical rates to mis-estimate selection bias for current cohorts. Mitigation: compute rates on recent windows or apply time-decay weights.
- `src/optimization_utils.py` — monotonicity constraints are hard rules on noisy point estimates from sparse/noisy bins, with no uncertainty bands. Mitigation: explore relaxed constraints with confidence bands/smoothing.
- `src/utils.py` — H3→H6 extrapolation curve fitting does not incorporate heteroscedasticity/variance structure in denominators; fitting is ratio-driven rather than variance-aware. Mitigation: consider inverse-variance style weighting.

### Testing Gaps
- `src/mr_pipeline.py` — add focused tests for MR hybrid reconstruction edge cases:
  - `mr_maturity_months != 6` and `mr_maturity_months=0`
  - bins where `b2_mr_h3` exists but `n_obs_mr_h3 < h3_min_obs`
  - repesca recalibration when merge keys are non-unique (row-order safety)
  - MR-only bins where hybrid selection leads to `model_fallback` and ensure NaN-risk handling in downstream metrics.

## Low
### Bug / Robustness
- `src/models.py` — `plt.show()` can block in headless/CI; guard or remove in favor of `savefig()`.


# Audit Findings — Remaining To-Do

Findings from the codebase audit (errors & methodological mistakes). Severities and
`file:line` references are from the audit; a few are design choices flagged for
confirmation rather than definite bugs. Verify before changing behavior — several
items shift headline risk/production numbers.

**Status legend:** `[ ]` open · `[x]` done

---

## Done
- [x] **#1 — Bin-edge target leakage + wrong population.** `learn_optimization_bins` deprecated (forced quantile); edges now learned on the date-filtered demand population in all 3 call sites (`preprocess_improved.py`, `run_batch.py`). Verified: income edge shifted −20.8% (booked→demand) on real data.
- [x] **#2 — MR maturity anchored to `max(mis_date)`.** H3 path (`mr_pipeline.py:410`) and `_assign_tiered_risk` (`:939`) now use the observation-horizon anchor (`date_fin_book_obs_mr`), consistent with the H6 path. (H6 path was already correct given the end-of-month-next-month data convention.)
- [x] **#3 — RI calibration / acceptance-rate population mismatch.** Root cause was `reject_include_all_rejections=true`: the all-rejects acceptance rate fed both the parceling uplift and the calibration target, but the blend is only score-rejected. Deprecated the flag and forced **score-only** acceptance rates everywhere (`reject_inference.py:compute_acceptance_rates`); fixed the misleading warning that recommended the flag. Original "wrong estimand" framing was overstated — downgraded from HIGH; the objective's intent (blend ≈ selection-model population risk) is sound.

---

## Tier 1 — Methodological flaws that bias results

- [ ] **#4 — Bayesian/empirical-Bayes smoothing mixes scales under time decay.** `reject_inference.py:202, 228-239`. Float effective counts mixed with count-scale Beta prior → over-shrinks to global rate; EB moment estimator (`between_var = sample_var - within_var_mean`) is unweighted and uses a global `p`, so the auto-tuned prior strength is unsound. **HIGH.**
- [ ] **#5 — No-demand cells get the *smallest* RI uplift.** `reject_inference.py:553-555`. Bins absent from the acceptance-rate table (zero demand = highest selection bias) are filled with the *median* acceptance rate → near-1.0 multiplier. Anti-conservative; should use a low rate. **HIGH.**
- [ ] **#6 — HurdleRegressor / zero-inflation rationale is inert.** `estimators.py:95`, `inference_optimized.py:295-300, 1269`. Two-part model is fit on the bin-aggregated continuous ratio `b2_ever_h6` (≈ never exactly zero), so `P(y>0)≈1` and the hurdle degenerates to plain Ridge/Lasso while still selected as a distinct model. The "zero proportion" diagnostic is also computed post-aggregation (misleading). **HIGH.**
- [ ] **#7 — Model selection reuses one holdout; 1-SE rule degenerates.** `inference_optimized.py:944`, `optuna_tuning.py:187,232`. Same 0.2 holdout picks the winner among all candidates → selection-biased "unbiased" RMSE. Tree/holdout models report `CV Std RMSE = 0`, so the 1-SE band collapses to argmin and the SE column is inconsistent across rows. **HIGH.**
- [ ] **#8 — TweedieGLM uses a penalized free `log(exposure)` coefficient, not a fixed offset.** `estimators.py:292-296`. L2 shrinks it away from 1.0 (docstring expects ~1); when the weight col is `todu_amt_pile_h6`, exposure doubles as offset feature *and* regression weight. Also model selection scores Tweedie by weighted RMSE on the ratio instead of Tweedie deviance (`inference_optimized.py:687`). **HIGH.**
- [ ] **#9 — Cell-level CIs are too narrow.** `inference_optimized.py:832-848`. `std/sqrt(n)` over correlated cross-fold predictions underestimates SE (CV-variance pathology); ≤5 folds make the `t`-interval unstable. Over-confident uncertainty. **HIGH.**

## Tier 2 — Numerical / statistical bugs

- [ ] **#10 — SPC control limits not estimated from within-subgroup variation.** `trends.py:233-260`. Rolling MAD of *levels* inflates sigma during drift (masking anomalies); fallback to full-series global std; t-quantile multiplies a per-point spread, not a SE; `window=3` (`main.py:601`) builds limits on 3 points. SPC layer is essentially uncalibrated. **HIGH.**
- [ ] **#11 — PSI uses a non-standard epsilon-only-in-log variant.** `stability.py:201`, `metrics.py:227`. Difference term uses true proportions, only the log is epsilon-floored. **Debatable** — code comments defend it as deliberate; likely a defensible design choice, not a clear bug. Confirm asymmetric weighting of appearing/disappearing bins is acceptable vs. the 0.1/0.25 thresholds. **LOW/debatable.**
- [ ] **#12 — PSI silently returns ~0 for low-cardinality scores.** `stability.py:167-191`. `qcut(..., duplicates="drop")` can collapse to a single bin → reports "stable" for a genuinely shifted population; `pd.cut` fallback only triggers on `ValueError`, not on silent duplicate-dropping. **MED.**
- [ ] **#13 — global_optimizer Pareto pruning is fragile.** `global_optimizer.py:165`. Hard-coded `1e-4` jitter threshold + equal-production ties. *Severity lower than originally rated* — mostly keeps the right points. Recommend switching to the canonical `prod > prev_max` sweep used in `optimization_utils.trace_pareto_frontier` for consistency. **MED/LOW.**
- [ ] **#14 — Global risk is production-weighted, but segment risk is exposure-weighted.** `global_optimizer.py:342, 453`. Segment risk = `sum(bad)/sum(exposure)`; global MILP constraint + reported global risk weight segment risk% by *production* (`oa_amt_h0`) → a production-weighted average of ratios. Inconsistent unless production ∝ exposure. **Definitional — confirm intended convention with business before "fixing."** **MED.**
- [ ] **#15 — MILP decode falls back to `argmax` without re-checking feasibility.** `global_optimizer.py:442`. On degenerate/fractional per-segment solution (`sum != 1`), picks `argmax` and returns `success=True` without asserting the chosen rows satisfy the global risk target → infeasible reported as optimal. **MED.**
- [ ] **#16 — Monotonicity-direction inference uses a mis-calibrated permutation test.** `preprocess_improved.py:1060-1089`. Unweighted ranks + production-weighted covariance, permuting ranks with weights fixed → mis-calibrated p-value; non-significant defaults to ascending (`direction=1`), which can impose the wrong monotonicity constraint on the MILP. **MED.**
- [ ] **#17 — Partial-order monotonicity "fix" is not a valid isotonic projection.** `reject_inference.py:431-479`. Averaging the two offending cells can re-introduce violations elsewhere; `for _ in range(5)` may exit non-converged; final clip can re-break monotonicity. Valid PAVA over a poset pools whole level sets. **MED.**

## Tier 3 — Data quality & loading (fail-open / silent coercion)

- [ ] **#18 — DQ checks fail open.** `data_quality.py`. Date-coverage gaps & parse failures are WARNING only (`249-285`); negative counts/amounts WARNING only (`288-368`); `check_booked_ratio` divides by the *full unfiltered* DataFrame (`384-387`) → understated per-segment ratios. With `allow_warnings=True` (default), invalid data passes. **HIGH-value quick win.**
- [ ] **#19 — Indiscriminate string/categorical coercion can mangle dates & IDs.** `data_manager.py:92-93`. Every object/string col lowercased, spaces→underscores, cast to `category` — corrupts string `mis_date` (breaks date filtering) and case-sensitive IDs. Also `encoding="latin-1"` hard-coded (`:18`) → mojibake on non-latin-1 files, mis-routing filters. **MED.**
- [ ] **#20 — DQ column check is case-insensitive but `validate_data_columns` is case-sensitive.** `data_quality.py:114-133` vs `data_manager.py:37-45`. A column that "passes" DQ can still fail later — false assurance. **LOW.**
- [ ] **#21 — Out-of-range scores silently snapped into edge bins.** `preprocess_improved.py:603-633`. Up to 1% of records outside configured finite edges coerced into first/last bin rather than flagged → mixes out-of-distribution scores into boundary cells. **LOW.**

---

## Related sub-findings (lower priority / context)

- [ ] **#2b — Immature zero-H6 rows pass the `.notna()` gate.** `mr_pipeline.py:320`. Mitigated by the #2 anchor fix, but a correct anchor doesn't NaN-out H6=0 immature rows at source. Separate question: should immature H6 zeros be treated as NaN? **LOW.**
- [ ] **RI Gini imputation is self-fulfilling.** `selection_bias.py:660`. Imputed bad rate is a deterministic monotone function of the score decile, so Gini on the imputed-rejected portion is mechanically inflated (diagnostic-only). **MED (diagnostic).**
- [ ] **Thorndike `attenuation_pct` can be negative.** `selection_bias.py:167`. `corrected_gini` can come out ≤ `observed_gini` (independent clipping) → nonsensical negative "attenuation"; no clamp. **LOW.**

---

## Notes
- Confidence: items directly verified during the audit are higher-confidence; the rest come from careful reviews with quoted code at medium-high confidence. **#11 and #13 are likely overstated** (design choices). Confirm Tier-1 methodology items with a domain expert before changing behavior.
- Several fixes (#3–#9, #14) interact with the credit methodology — scope/confirm intent before implementing.
- Quick, high-value hardening: **#10 (SPC), #15 (feasibility assert), #18 (fail-open DQ).**

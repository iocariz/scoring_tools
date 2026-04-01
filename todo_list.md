# Audit findings (codebase review)

Unresolved items from the methodological / statistical review. Strikethrough when fixed.

## HIGH

1. ~~**`apply_reject_inference` docstring vs default** (`src/reject_inference.py`, ~668–671) — Docstring said `apply_h3_multiplier` defaulted to **True**; actual default is **`False`**.~~ **Fixed:** Docstring and `apply_parceling_adjustment` comments aligned with `False` default; debug log uses `apply_h3_multiplier`.

## MEDIUM

2. **Consolidated optimum risk CIs** (`src/consolidation.py`, `aggregate_metrics`) — Pooled risk point is correct from summed numerators/denominators; combined segment CIs use independence + exposure-weighted SE stacking. **Coverage** may differ from nominal 95% for the true pooled portfolio rate. **Fix:** Document as heuristic in reports or refine method (e.g. bootstrap at consolidated level).

3. **Bootstrap CIs for Optimum** (`src/utils.py`, `calculate_bootstrap_intervals`) — Resamples booked only; swap-in/reject model error not in interval. **Fix:** Ensure report text states scope (sampling uncertainty of booked path under fixed cut / fixed repesca production).

4. **`fillna(0)` on grid/KPI display** (`src/optimization_utils.py`, `src/plots.py`) — Missing cells can show as 0% risk in some views. **Fix:** Distinguish “no data” vs “zero risk” in UI where feasible; rely on `observed` for optimization.

5. ~~**RI optimizer outer merge** (`src/reject_inference_optimizer.py`, merged booked+repesca) — `fillna(0)` can mask misaligned bin keys. **Fix:** Optional validation log or assert on key coverage after merge.~~ **Fixed:** Merge-key column dtypes preserved after outer merge + fillna(0).

6. ~~**H3→H6 power fallback without `b2_h3_main`** (`src/utils.py`, `extrapolate_h3_to_h6`) — Legacy branch `b2_h3 * ratio^curvature` when main H3-by-bin missing may diverge from fitted log-log path. **Fix:** Document or warn when fallback path is used.~~ **Fixed:** Added `logger.warning()` when legacy fallback path is used.

## LOW

7. **PSI thresholds** (`src/stability.py`) — Epsilon-in-log formulation is intentional; standard 0.1/0.25 bands remain conventional.

8. **RI N-D monotonicity** (`src/reject_inference.py`) — Alternating isotonic + pairwise fix is not full lattice isotonic regression.

9. **Gini/KS bootstrap** (`src/metrics.py`) — Skipped degenerate resamples can yield `(0,0)` CIs on tiny/degenerate samples.

10. **Global frontier pruning** (`src/global_optimizer.py`) — Assumes risk-sorted frontier input; dominated-point drop is consistent with that.

11. **MILP weighted risk** (`src/global_optimizer.py`) — `Σ p(r−T) ≤ 0` matches production-weighted average risk ≤ target; no change unless definition of “global risk” changes.

## Audit 2025-03-31

### MEDIUM — Bugs

12. **H3→H6 ratio hardcoded threshold** (`src/mr_pipeline.py`, ~597) — Per-bin ratio excludes bins with H3 risk < 0.01 (1%). Low-risk portfolios lose significant bins, forcing fallback imputation. **Fix:** Make threshold relative to global H3 rate or configurable.

### MEDIUM — Statistical

13. **IV unstable with zero-event bins** (`src/metrics.py`, 465–468) — `WOE = ln(perc_good / epsilon)` with epsilon=0.0001 produces extreme values for bins with zero bad accounts, inflating IV. **Fix:** Use Laplace smoothing (add 0.5 to all counts) or collapse zero-event bins.

14. **Bootstrap CIs use simple percentile method** (`src/metrics.py`, 101–102) — For bounded statistics like Gini, percentile CIs can have <95% actual coverage. **Fix:** Consider BCa method, or document as approximate.

15. **Maturity calculation truncates to calendar months** (`src/mr_pipeline.py`, 359–361) — `(year_diff * 12 + month_diff)` creates discontinuities at month boundaries. **Fix:** Use `(reference_date - mis_date).dt.days / 30.437` for fractional months.

### MEDIUM — Performance

16. **O(n²) Pareto dominance check** (`src/optimization_utils.py`, 726–738, duplicated at 1094–1106) — Pairwise loop is redundant for 2D: Stage 1 sort-and-sweep already produces correct frontier. Stage 2 only needed for N>2 objectives. **Fix:** Skip Stage 2 for 2-objective case or replace with vectorized numpy.

### LOW-MEDIUM — Robustness

17. **Missing config cross-validation** (`src/config.py`) — Invalid combinations not caught early: `mr_maturity_months` exceeding data window, `reject_acceptance_decay_half_life_months` without valid date column, `per_bin_tasa_fin=True` with 1 variable. **Fix:** Add `@model_validator(mode=”after”)` checks.

18. **CellGrid constructed repeatedly** (`src/optimization_utils.py`) — `CellGrid.from_summary()` called multiple times with same data during sensitivity analysis. **Fix:** Construct once and pass through.

### LOW — Logging

19. **Warning messages lack actionable context** (multiple files) — Missing bins in parceling, zero-exposure cells in MILP, failed date parsing: all log generic messages without listing which bins/cells/values are affected. **Fix:** Include bin IDs and aggregate stats in warnings.

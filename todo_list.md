# Audit findings (codebase review)

Unresolved items from the methodological, statistical, and code review. Organized by **topic → severity**. Original item numbers preserved for traceability. Resolved items archived at the bottom.

---

## 1. Security

### CRITICAL

43. **Jinja2 `autoescape=False` — stored XSS in HTML reports** (`src/reporting.py`, 1077–1079) — `Environment(..., autoescape=False)`; segment names, scenario labels, and CSV cell values are interpolated into report HTML via helpers like `_build_cutoff_reference_table`. A malicious value in `segments.toml` or an upstream CSV becomes stored XSS when the report opens in a browser (and is served by the Dash `/static/` route). **Fix:** `autoescape=select_autoescape(["html"])`; wrap pre-built HTML fragments (Plotly JSON, etc.) in `jinja2.Markup` only where genuinely safe.

44. **`joblib.load` of pickle files = RCE** (`src/persistence.py`, 183; `src/pipeline/inference.py`, 68) — Both call sites load pickle-backed `.pkl` files from paths configurable via `--model-path` / `config.toml`. `joblib.load` is `pickle.load`; a crafted file executes arbitrary Python on load. **Fix:** SHA-256 sidecar (`model.pkl.sha256`) verified before load + `Path.resolve()` prefix-check against a trusted directory. Long-term: store coefficients via safetensors/JSON for the custom estimators.

### HIGH

46. **Root `/static/<path:filename>` route has no validation** (`dashboard.py`, 2727–2731) — The segment route is protected by `_is_allowed_static_segment`; the root route is not. `images_dir = get_images_dir(None)` can be relative (`Path("images")`), so the effective base depends on cwd. **Fix:** `images_dir.resolve()` + extension allowlist (`.html`, `.png`, `.csv`); add a regression test (current `tests/test_dashboard_security.py` only tests the segment route).

47. **`debug=True` passthrough via `--debug` CLI flag** (`dashboard.py`, 4397; `interactive_allocator.py`, 547) — `app.run(debug=args.debug, ...)` enables Werkzeug's interactive console, an RCE vector if the dashboard is network-accessible. **Fix:** Refuse `debug=True` unless `DASHBOARD_DEBUG_ALLOWED=1` is set.

48. **`gradio --share` tunnels portfolio data to gradio.live publicly** (`gradio_dashboard.py`, 1117) — `app.launch(server_port=args.port, share=args.share)`; `--share` creates a public, unauthenticated URL displaying loan KPIs. **Fix:** Remove the flag, or gate behind `GRADIO_SHARE_ALLOWED=1` with a hard warning.

49. **PII rows logged at DEBUG** (`src/preprocess_improved.py`, 52) — `logger.debug(f"{name} first rows:\n{df.head(3)}")` emits raw loan-application records into `--log-file`. **Fix:** Log structural summary only (shape, dtypes, numeric describe); never row-level data.

50. **No authentication on any web UI** (`dashboard.py`, `interactive_allocator.py`, `gradio_dashboard.py`) — Default `localhost` bind is good, but no authn layer. A misconfigured reverse proxy or accidental port exposure leaks portfolio risk KPIs. **Fix:** Flask `before_request` Basic Auth against env-var creds for any non-localhost deployment; document the requirement.

51. **No lockfile committed** (`pyproject.toml`) — original audit statement was stale: `uv.lock` was tracked since `01440b9`. R1 attempted to add `uv lock --check` to CI but had to revert it: the committed lockfile is platform-resolved (~86 packages, macOS-origin) and a Linux CI fresh resolution yields ~144 packages, so the check fails on every PR regardless of whether anyone touched pyproject. **Remaining fix (R2):** regenerate `uv.lock` on a Linux runner (or in a pinned Docker image) to produce a lockfile matching the CI environment, then re-enable the `uv lock --check` step. Alternative: adopt uv's universal / multi-platform lockfile mode once stable. Meanwhile the lockfile is still committed — CVE exposure is bounded to the lockfile's macOS-resolved set, not unbounded.

88. **HTML-building helpers interpolate user-influenced values without escaping** (`src/reporting.py`, `_build_cutoff_reference_table` 134–310, `_build_scenario_kpi_table` 341, `_build_acceptance_matrices` 408, `csv_to_html_table` 83) — Jinja2 `autoescape` is now enabled (#43 fixed), but these helpers emit pre-built HTML strings that are then marked `| safe` in the template. They interpolate scenario names, CSV column labels, and cell values directly via f-strings (e.g. `f'<span class="badge">{scenario.title()}</span>'`). A malicious value in `segments.toml` or an upstream CSV still becomes stored XSS through this path. **Fix:** escape every user-influenced interpolation inside these helpers using `html.escape()` (already imported); add unit tests with malicious inputs as regression guards.

### LOW

76. **Path-traversal test coverage gap** (`tests/test_dashboard_security.py`) — Tests literal `../etc`, `a/../../b`, `..\windows`; no URL-encoded (`..%2Fetc`), null-byte, or UNC variants. Flask normalizes most before routing, but the gap leaves regressions undetected.

79. **`subprocess.run(["soffice", ...])`** (`generate_presentation.py`, 1033–1037) — Currently safe (list form, hardcoded path), but `pptx_path` parameterization would make it injection-prone. Document the constraint inline.

---

## 2. Error Handling & Robustness

### CRITICAL

45. **9× `except Exception: pass` silently corrupts consolidated reports** (`src/consolidation.py`, 638, 2137, 2228, 2260, 2438, 2464, 3520, 3533, 3550) — Failures during config loading, Excel cell writes, and segment data assembly are swallowed without trace. The output is wrong silently rather than loudly. **Fix:** Replace each `pass` with `logger.warning(..., exc_info=True)`; re-raise where the result is downstream-load-bearing.

### LOW–MEDIUM

17. **Missing config cross-validation** (`src/config.py`) — Invalid combinations not caught early: `mr_maturity_months` exceeding data window, `reject_acceptance_decay_half_life_months` without valid date column, `per_bin_tasa_fin=True` with 1 variable. **Fix:** Add `@model_validator(mode="after")` checks.

83. **Tied quantile bin edges silently dropped** (`src/preprocess_improved.py`, 240) — `valid.quantile(quantiles).unique().tolist()` collapses duplicate thresholds without warning; a `max_bins=5` request on a skewed score can yield 2 bins and the optimization runs with fewer dimensions than configured. **Fix:** `logger.warning` when `len(thresholds) < max_bins − 1`; raise if below the monotonicity-relevant minimum.

84. **NaN source values silently dropped in `pd.cut` binning** (`src/preprocess_improved.py`, ~589–591, around the cut call) — `-inf`-bounded bins catch `−∞` but not NaN; NaN rows stay NaN after cut and are excised downstream without a drop-count log. Survivorship bias when missingness is non-random. **Fix:** explicit NaN-bin assignment or fail-loud count + threshold guard.

86. ~~**Audit validation silently returns `None` on missing reference rows** (`src/audit.py`, 511–512) — When expected Swap-in/Swap-out rows are missing from the summary table, `validate_audit_against_summary` returns `None`, which downstream callers treat as not-failed. A malformed summary passes validation. Methodological complement to type-safety item #54. **Fix:** raise on precondition failure or return `False`; add `pd.isna` check on `values[0]` before equality comparison.~~ **Fixed jointly with #54:** missing rows and NaN reference values now both log a warning and return `False`. Single path for "could not validate" vs "validation disagreed" is the log record — return type is now unambiguously `bool`.

### LOW

19. **Warning messages lack actionable context** (multiple files) — Missing bins in parceling, zero-exposure cells in MILP, failed date parsing: all log generic messages without listing which bins/cells/values are affected. **Fix:** Include bin IDs and aggregate stats in warnings.

85. **Category collisions via `str.lower().str.replace(" ", "_")`** (`src/data_manager.py`, 93) — `"SEGMENT AB"` and `"segment_ab"` both collapse to `segment_ab`; no before/after `nunique` check, so distinct segments get silently pooled. **Fix:** collision detection with warning; optional hard-fail when standardization reduces cardinality.

---

## 3. Type Safety & Correctness

### HIGH

52. **`Literal` types weakened to `str` at config boundary** (`src/reject_inference_optimizer.py`, 50, 133; `src/inference_optimized.py`, 1843, 1846) — `OptimizerInputs.parceling_method: str` then passed to `apply_parceling_adjustment(... Literal["linear","power","sigmoid"])`. Misspellings reach the function and silently fall through. **Fix:** Type the field as `Literal[...]` and validate at config load; or `cast()` after explicit membership check.

53. **`base_path: str` reassigned to `Path`** (`src/persistence.py`, 23, 37–44) — Signature lies; mypy reports `"str".mkdir`. **Fix:** `base_path: str | Path = "models"` and use a single local `path = Path(base_path)`.

54. ~~**`bool | None` return with missing return** (`src/audit.py`, 470–528) — `validate_audit_against_summary` declares `-> bool | None`, mypy flags missing return at 470, callers treat as `bool`. **Fix:** Return `False` (or raise) on missing-column precondition; tighten signature to `-> bool`.~~ **Fixed:** Signature tightened to `-> bool`. Every precondition failure (missing Metric column, empty Swap-in / Swap-out rows, NaN reference values) now logs a warning and returns `False` explicitly — no path falls through to an implicit `None`. Jointly fixes #86 (same function's silent-pass-on-missing-rows behaviour).

55. **`union-attr` runtime risk on `calculate_b2_ever_h6`** (`src/optimization_utils.py`, 1763–1798; `src/mr_pipeline.py`, 1044–1077; `src/preprocess_improved.py`, 1042–1048; `src/plots.py`, 1960–1961; `src/inference_optimized.py`, 142) — Returns `Series | float`; callers do `.isna()/.fillna()/.values` without narrowing → `AttributeError` on scalar inputs. **Fix:** Always return `pd.Series` (wrap scalars) or narrow at every call site.

56. **PEP 484 implicit-Optional violations** (`src/styles.py`, 62; `src/metrics.py`, 151–152, 273) — `def f(title: str = None)` etc. with `no_implicit_optional=True`. **Fix:** `str | None = None`.

---

## 4. Architecture & Code Organization

### HIGH

57. **Functions massively exceeding the <50-line rule** —
    - `export_consolidated_excel` 2004 lines (`src/consolidation.py`, 1636) — entire module inside one function, untestable.
    - `process_mr_period` 943 lines (`src/mr_pipeline.py`, 1132) — 7 distinct responsibilities.
    - `_compute_hybrid_mr_risk` 618 lines (`src/mr_pipeline.py`, 305).
    - `run_optimization_phase` 444 lines (`src/pipeline/optimization.py`, 41).
    - `main` 428 lines (`run_batch.py`, 1232).
    - `main` 334 lines (`main.py`, 254).
    **Fix:** Extract by phase into private helpers; for MR add `_extrapolate_h3`, `_apply_ri_adjustment`, `_build_mr_diagnostics`. For Excel: per-tab sheet builders + orchestrator + formatting pass.

### MEDIUM

63. **`src/pipeline/optimization.py` is not "thin orchestration"** (1,250 lines, 4 substantive phases) — Violates the contract in `CLAUDE.md`. `run_optimization_phase`, `run_scenario_analysis`, `run_sensitivity_phase`, `run_ri_optimizer_phase` each contain business logic. **Fix:** Split into `cutoff_optimization.py` / `scenarios.py` / `sensitivity.py`. Make `_build_scenario_list`, `_compute_mr_annual_coef`, `_save_cutoff_summaries` public (currently imported as privates from `main.py`, 17–22). Introduce `OptimizationResult` dataclass to replace the 7-tuple at `src/pipeline/optimization.py`, 55 — same pattern as `PreprocessingResult`.

64. **Three competing dashboards duplicate ~15–20 KB of code** (`dashboard.py` 175 KB Dash, `gradio_dashboard.py` 48 KB Gradio, `interactive_allocator.py` 22 KB Dash) — Each redefines `get_data_dir`, `get_images_dir`, `get_available_segments`, `get_scenarios`, `parse_coefficients`, `_resolve_supersegment`, `load_cutoff_data`. **Fix:** Extract `src/dashboard_data.py`; have all three depend on it. Reassess whether `gradio_dashboard.py` is still needed.

65. **2-D assumptions leak from N-D abstraction** (`src/optimization_utils.py`, 820, 825, 915, 958; `src/pipeline/optimization.py`, 179, 377, 1262; `src/utils.py`, 488, 662; `src/inference_optimized.py`, 429; `src/sensitivity.py`, 83, 133; `src/reject_inference.py`, 411; `src/plots.py`, 302) — `if len(variables) == 2:` branching everywhere; hardcoded `["new_efx_clus", "sc_octroi_new_clus", "income_bin"]` fallback at `src/consolidation.py`, 3536. The 2-var path returns `cut_map`, N>2 returns `_cells` + `_marginal_*` — downstream must know the shape. **Fix:** `CutoffSpec` value type with `.as_2d_cut_map()` adapter; remove hardcoded column lists.

66. ~~**Legacy `octroi_bins` / `efx_bins` still load-bearing** (`src/config.py`, 308–311, 458–471, 558–598) — `_auto_populate_bins` reverse-maps positional vars to hardcoded `score_rf`, `risk_score_rf`. Default `data_path = "data/demanda_direct_out.sas7bdat"` (line 317) cements the 2-var-flavored design. **Fix:** Emit `DeprecationWarning`; provide `scripts/migrate_legacy_bins.py`; generalize default `data_path`.~~ **Fixed (R2 part 1):** `_auto_populate_bins` now emits a `DeprecationWarning` listing each variable populated via the legacy path and pointing at `scripts/migrate_legacy_bins.py`. Migration script rewrites `octroi_bins` / `efx_bins` into `[preprocessing.bins.*]` blocks in place (with a `.bak`). 6 regression tests cover both directions (warning fires / does not fire) and the migration rewrite. `data_path` default kept — renaming is a separate operational concern with deploy-side impact.

67. ~~**`os.chdir` for output isolation** (`run_batch.py`, 594–605) — `_working_directory` chdir'ing each segment is the only piece of process-global state; safe with `ProcessPoolExecutor` but blocks any thread-pool migration. **Fix:** Replace with explicit `OutputPaths(base_dir=dirs["root"])` passed through `main.main(output=...)`. Make `OutputPaths.base_dir` non-optional. Removes the cwd-sniffing in `main.py`, 103–110.~~ **Fixed (R2 part 2):** `_working_directory` context manager removed. Both call sites in `run_batch.py` (per-segment and per-supersegment) now pass `output=OutputPaths(base_dir=dirs["root"].resolve())` directly to `main.main()`. No process-global cwd change; the pipeline's existing `output=` plumbing already threads through every phase. `import os` + `from contextlib import contextmanager` dropped from run_batch. Kept `base_dir` optional in `OutputPaths` (pipeline phases use `None` as "standalone default"); the audit's ask to make it non-optional would cascade into ~10 fallback sites and is deferred. Kept the `run_resimulation` cwd-sniffing in main.py (separate resimulation CLI heuristic). Unblocks future ThreadPoolExecutor migration.

---

## 5. MR / H3→H6 Inference

*Complements resolved items #12 (low-H3 ratio gate) and #24 below.*

### MEDIUM–HIGH

34. **Out-of-time calibration of MR hybrid risk** (`src/mr_pipeline.py`, optimization inputs) — After computing `b2_ever_h6_tmp`, fit a calibrator (e.g. isotonic or Platt) on a holdout slice where H6 is mature so predicted rank/probability matches realized defaults. **Fix:** Optional config flag; apply before aggregation into optimization grid.

35. **Reliability-weighted blend instead of hard source switching** (`src/mr_pipeline.py`, `_compute_hybrid_mr_risk`) — Replace pure `np.select` priority with weights `w_mr`, `w_h3`, `w_main` from counts, recency, and fit quality; blend risks to avoid discontinuities at `min_obs` boundaries. **Fix:** Feature-flagged path; keep legacy discrete ladder as default until validated.

36. **Uncertainty-aware cutoff optimization** (`src/optimization_utils.py`, `trace_pareto_frontier` / MILP) — Propagate per-bin uncertainty (e.g. SE from counts or bootstrap) and optimize on conservative risk (`mean + k·SE`) or document chance-style constraints. **Fix:** Extend `CellGrid` / constraint row; config `k` and uncertainty source.

### MEDIUM

24. **H3→H6 auto-calibration sample selection bias** (`src/mr_pipeline.py`, 480–502) — Only bins with both main H3 and H6 fit the log-log curve. MR-only or H6-only bins excluded, biasing curvature toward stable bins. **Fix:** Document limitation; consider including MR H3 data in the fit when available.

37. **Hierarchical shrinkage for sparse bins** (`src/mr_pipeline.py`) — Partial pooling (empirical Bayes or simple James–Stein) for bin-level MR rates or H6/H3 ratios so sparse cells borrow strength from segment/global instead of abrupt median fallback. **Fix:** Complements **#12**; reduces reliance on hard ratio floors alone.

38. **Time-varying H6/H3 mapping** (`src/mr_pipeline.py`) — Monthly H6/H3 trend is logged but ratio used for extrapolation is static; fit ratio as function of cohort month or macro covariates when trend warning fires. **Fix:** Optional model or rolling-window ratio per vintage.

39. **Beyond fixed calendar maturity** (`src/mr_pipeline.py`, `_compute_hybrid_mr_risk`) — Add effective-maturity weights or survival-style censoring instead of only hard drop before `mr_maturity_months` (and fixed 3mo for H3). **Fix:** Configurable strategy; document vs current binary filter.

40. **Model fallback: reduce booked-only bias** (`src/mr_pipeline.py`, `model_fallback` + `calculate_B2`) — Fallback uses model trained on booked; add separate calibration or training weights / reject-aware targets for MR-only bins. **Fix:** Document scope; optional retrain or post-hoc calibration on fallback subset.

41. **Adaptive risk caps for extrapolation / fallback** (`src/config.py`, `mr_pipeline.py`) — Replace or augment fixed `mr_extrapolation_risk_multiplier` and `mr_extrapolation_hard_cap` with data-driven limits (e.g. bin-level percentiles, posterior upper bounds). **Fix:** Config profiles; backward-compatible defaults.

### LOW–MEDIUM

42. **Run-quality gates for MR-heavy outputs** (`main.py` / `pipeline/optimization.py` / reports) — Fail or prominently warn when extrapolated share of MR production, `model_fallback` share, or H6/H3 instability exceeds thresholds (complements **#17** config validation). **Fix:** Thresholds in config; exit code or report banner.

---

## 6. Optimization & Cutoff Selection

### MEDIUM

28. **Hard monotonicity in bin space** (`src/optimization_utils.py`, `_build_monotonicity_constraints` + MILP) — Optimum is best **monotone rectangular** policy, not best subset of cells; empirical risk need not be monotone (noise, selection, sparse bins). **Fix:** Document constraint as policy choice; optional sensitivity without monotonicity on small grids.

29. **Objective is production, not economic value** (`src/optimization_utils.py`, `milp_solve_cutoffs`) — Maximizes `oa_amt_h0` subject to risk caps; margins/LGD/pricing by bin are ignored. **Fix:** Document; optional weighted objective if data exists.

82. **Uniform multiplicative sensitivity perturbation can violate monotonicity** (`src/sensitivity.py`, 39, `perturb_risk_summary`) — Applies a flat factor to all cells' `todu_30ever_h6_rep`; MILP is constrained by monotone rectangular policies, so perturbation reshuffles empirical monotonicity in some cells. The reported "sensitivity" then mixes (a) true response of the optimum and (b) constraint-violation rearrangement. **Fix:** per-cell perturbation respecting monotonicity, or report monotonicity-violation count alongside the sensitivity deltas.

### LOW

31. **MILP time limit** (`src/optimization_utils.py`, `milp_solve_cutoffs`) — `scipy.optimize.milp` `time_limit` can yield suboptimal or no solution for a target, weakening a frontier point. **Fix:** Log/time budget; retry with higher limit on failure.

32. **GA Pareto fallback is heuristic** (`src/optimization_utils.py`, `_ga_pareto_fallback`) — Does not match exact MILP quality on the discretized grid. **Fix:** Document; prefer MILP when tractable.

33. **N>2 heatmap is a 2D marginal** (`src/plots.py`, `RiskProductionVisualizer`) — Extra dimensions summed for display; overlay is not the full N-D acceptance set. **Fix:** Caption/report text so readers do not equate heatmap with the true policy.

74. **Magic numbers duplicated across MILP and GA paths** (`src/optimization_utils.py`, 570, 584, 1047) — `sweep_min = 0.01`, overshoot factor `1.1` repeated. `constants.py` already holds `DEFAULT_N_POINTS_3D` so precedent exists. **Fix:** `SWEEP_RISK_FLOOR_PCT = 0.01`, `SWEEP_RISK_CEILING_OVERSHOOT = 1.1` in `constants.py`.

---

## 7. Statistical & Methodological

### MEDIUM

2. **Consolidated optimum risk CIs** (`src/consolidation.py`, `aggregate_metrics`) — Pooled risk point is correct from summed numerators/denominators; combined segment CIs use independence + exposure-weighted SE stacking. **Coverage** may differ from nominal 95% for the true pooled portfolio rate. **Fix:** Document as heuristic in reports or refine method (e.g. bootstrap at consolidated level).

3. **Bootstrap CIs for Optimum** (`src/utils.py`, `calculate_bootstrap_intervals`) — Resamples booked only; swap-in/reject model error not in interval. **Fix:** Ensure report text states scope (sampling uncertainty of booked path under fixed cut / fixed repesca production).

14. **Bootstrap CIs use simple percentile method** (`src/metrics.py`, 101–102) — For bounded statistics like Gini, percentile CIs can have <95% actual coverage. **Fix:** Consider BCa method, or document as approximate.

21. **TweedieGLM log(exposure) treated as feature, not GLM offset** (`src/estimators.py`, 289–296) — Exposure is appended as a learned feature via `_add_log_exposure`. In a proper GLM, the offset is not regularized. sklearn's regularization penalizes the exposure coefficient toward zero, biasing exposure-adjusted predictions for small samples. **Fix:** Document as approximate, or use a custom linear predictor with true offset.

22. **Polynomial features on bin indices create multicollinearity** (`src/models.py`, 149–167) — Degree-3 polynomials on bin indices (1,2,3...) produce highly correlated features. Destabilizes Ridge coefficients and makes regularization hyperparameter selection sensitive. **Fix:** Use orthogonal polynomials or normalize indices to [-1, 1] before transformation.

23. **One-SE rule applied three times compounds conservatism** (`src/inference_optimized.py`, 587, 697) — Applied at model type, feature set, and combined selection. Each stage independently favors simpler models; compounding pushes the final model away from best-performing. **Fix:** Consider nested CV or single joint selection.

### LOW

7. **PSI thresholds** (`src/stability.py`) — Epsilon-in-log formulation is intentional; standard 0.1/0.25 bands remain conventional.

9. **Gini/KS bootstrap** (`src/metrics.py`) — Skipped degenerate resamples can yield `(0,0)` CIs on tiny/degenerate samples.

10. **Global frontier pruning** (`src/global_optimizer.py`) — Assumes risk-sorted frontier input; dominated-point drop is consistent with that.

11. **MILP weighted risk** (`src/global_optimizer.py`) — `Σ p(r−T) ≤ 0` matches production-weighted average risk ≤ target; no change unless definition of "global risk" changes.

87. **TweedieGLM predict-time fallback uses training-median exposure silently** (`src/estimators.py`, 329–332) — When the exposure column is absent at predict time, `_median_exposure` from training is substituted with no log. Safe for mesh-grid visualization, but if it fires on a real cohort with systematically different exposure (new product mix, portfolio shift), predictions are silently mis-scaled. **Fix:** `logger.warning` on fallback; require an explicit `fallback_exposure` parameter for non-visualization callers; consider raising for production prediction paths.

---

## 8. Reject Inference

### MEDIUM

25. **Parceling exogeneity assumption undocumented** (`src/reject_inference.py`, docstring) — Parceling assumes `Rejection ⊥ Risk | Score`. If manual overrides or policy rules influence rejection beyond the score, the multiplier is biased. **`safe_rate` floor, `max_risk_multiplier` cap, and method choice** (linear/power/sigmoid) further distort tail risk and can move the optimized cutoff. **Fix:** Document the assumption and sensitivity of optimal policy to RI params/caps in the docstring or report.

80. **Empirical-Bayes prior is circular and 50/50 blend is arbitrary** (`src/reject_inference.py`, 225–230) — `empirical_strength = (p·(1−p)/between_var) − 1` is derived from the same acceptance rates being shrunk; `effective_prior_strength = 0.5·configured + 0.5·empirical` hard-codes a weight with no justification. Standard EB, but undocumented as approximate and unstable when `between_var` is near zero or when the bin count is small. **Fix:** document as approximate; expose the blend weight as config; skip EB when `n_bins < 10` or `between_var` is non-positive/near-zero (beyond the existing warning).

81. **Missing repesca bins filled with unconditional median acceptance rate** (`src/reject_inference.py`, 546–554) — When a repesca bin has no demand data, `fallback_rate = median(acceptance_rate)` is applied and the risk multiplier is computed as if it were a true observation. No confidence penalty, no hierarchical shrinkage — off-distribution bins receive the same uplift as well-observed ones. **Fix:** use `compute_ri_confidence` (already exists in the module) to downweight or flag low-confidence bins; or shrink toward the global rate with a variance-dependent weight. Complements **#37** (hierarchical shrinkage).

### LOW

8. **RI N-D monotonicity** (`src/reject_inference.py`) — Alternating isotonic + pairwise fix is not full lattice isotonic regression.

---

## 9. Testing & Coverage

### HIGH

58. **Critical-path modules below 33% coverage** —
    - `src/pipeline/config_loader.py` 0%
    - `src/pipeline/optimization.py` 19%
    - `src/inference_optimized.py` 23%
    - `src/plots.py` 29%
    - `src/optuna_tuning.py` 33%
    **Fix:** Scenario tests for `run_optimization_phase` on small synthetic grids (mock MILP solver); parametrized TOML edge-case tests for `config_loader.py`.

59. **CI does not enforce coverage gate** (`.github/workflows/ci.yml`, 65, 74) — No `--cov-fail-under=80` on pytest; `fail_ci_if_error: false` suppresses Codecov failures. The 80% project target is not enforced. **Fix:** `pytest --cov=src --cov-fail-under=80`; flip Codecov flag. **Partially fixed in R1:** gate set to `--cov-fail-under=60` (baseline was 62%). Ratchet upward as R2 decomposition + R3 add tests. `fail_ci_if_error` still `false` with a comment noting when to flip (once Codecov upload is reliably green on main).

### MEDIUM

68. **Vacuous test assertions** (`tests/test_mr_pipeline.py` × 11; `tests/test_shap.py` × 4; `tests/test_reporting.py` × 1) — `assert result is not None` only verifies no exception was raised; would pass with all-zero or all-NaN outputs. `conftest.py` is 5 lines, so seeds and DataFrame shapes drift between files. **Fix:** Replace with shape/value assertions; centralize fixtures in `conftest.py`.

---

## 10. Performance

### MEDIUM

69. **`iterrows()` in performance-critical loops** (`src/consolidation.py`, 495, 673, 698, 1995; `src/mr_pipeline.py`, 133, 534, 662, 810, 1228) — 10–100× slower than vectorized ops. `mr_pipeline.py`, 810 builds a dict from sorted rows — replace with `dict(zip(col_a, col_b))` or `.to_dict()`. **Fix:** Replace each with `.apply()`, `np.vectorize`, or column arithmetic.

71. **Deferred imports inside frequently-called functions** (`src/estimators.py`, 79; `src/optimization_utils.py`, 1699; `src/inference_optimized.py`, 1079–1080) — `import pandas as pd`, `import gc`, `import traceback` mid-function obscure dependencies from static analysis. **Fix:** Move to module top; reserve in-function imports for genuinely optional features (`shap`, `optuna`).

72. **Logging volume produces 28 MB log files** — 382 `logger.info` + 103 `logger.debug` in `src/`, many inside loops (e.g., per-MILP-solve in `trace_pareto_frontier`). **Fix:** Demote loop-interior `logger.info` to `debug`; add `rotation="50 MB", retention=3` at the loguru `add` call (`run_batch.py`, 388).

### LOW–MEDIUM

18. **CellGrid constructed repeatedly** (`src/optimization_utils.py`) — `CellGrid.from_summary()` called multiple times with same data during sensitivity analysis. **Fix:** Construct once and pass through.

---

## 11. Hygiene & Tooling

### HIGH

60. **40 MB of artefacts tracked in git** — `bath_run.log` (28 MB, also a typo of `batch_run`), `batch.log` (12 MB), `batch_run.log`, `.DS_Store`, `coverage.xml`, `Credit_Risk_Scoring_Pipeline_Presentation.{pdf,pptx}`, `selection_bias.{docx,pptx}`, `allocation_results_*.csv`, `segments_suggested.toml`. Adds ~40 MB to every clone. **Fix:** `git rm --cached` for each; extend `.gitignore` to cover `*.log`, `*.pdf`, `*.pptx`, `*.docx`, `coverage.xml`, `*_suggested.toml`, `allocation_results*.csv`.

61. **`print()` calls bypass loguru** (`src/consolidation.py`, 3644–3681 — 15 calls; `src/data_quality.py`, 81–102 — 9 calls) — Cannot be silenced or routed to `--log-file`. **Fix:** Replace with `logger.info()` consistently. (Distinct from #19 which is about message *content*; this is about *channel*.)

### LOW

75. **Dead variables / unused imports** (`main.py`, 7 `import numpy as np`; `src/consolidation.py`, 2650 `total_rn`/`total_rd`; `src/consolidation.py`, 3506 `seg_vars`; `generate_presentation.py`, 283 `color`) — **Fix:** `ruff check --fix .` handles 3 of 4; manually verify `seg_vars`.

78. **Worktrees under `.claude/worktrees/`** hold near-complete copies of `main.py`/`run_batch.py`/`dashboard.py` — Confirm these are intentional review branches and not abandoned.

---

## 12. Documentation

### HIGH

62. **`main()` docstring documents 4 of 10 parameters** (`main.py`, 267–278) — Missing `output`, `floor_cells_path`, `floor_cells_mode`, `resimulate_risk`, `baseline_mode`, `base_scenario_only`. **Fix:** Sync docstring with signature; mention resimulation and sequential-cutoff entry points.

### MEDIUM

70. ~~**Undocumented config fields in CLAUDE.md** — `min_accepted_bin_by_variable` (`src/config.py`, 334; `src/pipeline/optimization.py`, 308), `base_scenario_only` (`src/config.py`, 329; no CLI flag unlike `--baseline`), `strict_validation` (`src/pipeline/optimization.py`, 177), `run_all_scenarios` (`src/pipeline/optimization.py`, 878). **Fix:** Document in CLAUDE.md; flag deprecated `min_accepted_bin_by_variable` vs `fixed_cutoffs`.~~ **Fixed:** All four documented in CLAUDE.md Configuration section. `strict_validation` and `run_all_scenarios` added as sub-bullets under "Fixed cutoffs"; `base_scenario_only` added as a distinct paragraph clarifying it differs from `baseline_mode` (still runs optimization, just one target); `min_accepted_bin_by_variable` documented with the scalar-vs-income-keyed-map distinction and flagged as the legacy alternative to `fixed_cutoffs`.

### LOW

77. **`analyze_logs.py` regex-parses loguru output** (62 KB) — Reverse-channel feedback for `segments.toml` suggestions; fragile and tightly coupled to log format. **Fix:** Have the pipeline emit JSON sidecars per segment; have `analyze_logs.py` consume those.

---

## 13. Miscellaneous quality

### LOW

73. **`HurdleRegressor` does not set `self.feature_names_in_`** (`src/estimators.py`, 69–81) — Local `feature_names_in_ = None` is never attached to `self`; sklearn ≥1.0 expects this attribute on fitted estimators for pipeline introspection. **Fix:** `self.feature_names_in_ = X.columns.to_numpy()` before `check_X_y`.

---

## Archive — Resolved / Not an issue

1. ~~**`apply_reject_inference` docstring vs default** (`src/reject_inference.py`, ~668–671) — Docstring said `apply_h3_multiplier` defaulted to **True**; actual default is **`False`**.~~ **Fixed:** Docstring and `apply_parceling_adjustment` comments aligned with `False` default; debug log uses `apply_h3_multiplier`.

4. ~~**`fillna(0)` on grid/KPI display** (`src/optimization_utils.py`, `src/plots.py`) — Missing cells can show as 0% risk in some views. **Also** (`kpi_of_fact_sol`, `src/optimization_utils.py`): NaN `b2_ever_h6` from zero exposure is filled with **0** for display/downstream — optimistic for ranking and `b2 <= optimum_risk` selection on the legacy 2-var path.~~ **Fixed:** `kpi_of_fact_sol` now preserves NaN in base `b2_ever_h6` (used for ranking/selection); only suffixed display columns (_cut/_rep/_boo) are fillna(0). `get_optimal_solutions` filters NaN/inf-risk solutions before Pareto ranking (matching the MILP path). `_get_selected_solution_row` excludes NaN-risk solutions and uses explicit `sort_values` + `tail(1)` for robust max-production selection (also fixes #30).

5. ~~**RI optimizer outer merge** (`src/reject_inference_optimizer.py`, merged booked+repesca) — `fillna(0)` can mask misaligned bin keys. **Fix:** Optional validation log or assert on key coverage after merge.~~ **Fixed:** Merge-key column dtypes preserved after outer merge + fillna(0).

6. ~~**H3→H6 power fallback without `b2_h3_main`** (`src/utils.py`, `extrapolate_h3_to_h6`) — Legacy branch `b2_h3 * ratio^curvature` when main H3-by-bin missing may diverge from fitted log-log path. **Fix:** Document or warn when fallback path is used.~~ **Fixed:** Added `logger.warning()` when legacy fallback path is used.

12. ~~**H3→H6 ratio hardcoded threshold** (`src/mr_pipeline.py`, ~597) — Per-bin ratio excludes bins with H3 risk < 0.01 (1%). Low-risk portfolios lose significant bins, forcing fallback imputation.~~ **Fixed:** Threshold is now relative to the segment's median H3 rate (10% of median, floored at 0.001). Low-risk portfolios retain most bins for ratio calibration instead of falling back to imputation.

13. ~~**IV unstable with zero-event bins** (`src/metrics.py`, 465–468) — `WOE = ln(perc_good / epsilon)` with epsilon=0.0001 produces extreme values for bins with zero bad accounts, inflating IV.~~ **Fixed:** Replaced epsilon substitution with Laplace smoothing (+0.5 to bad and good counts per bin). Produces stable WOE/IV for zero-event bins without inflating the metric.

15. ~~**Maturity calculation truncates to calendar months** (`src/mr_pipeline.py`, 359–361) — `(year_diff * 12 + month_diff)` creates discontinuities at month boundaries.~~ **Not an issue:** All input dates use 01/MM/YY format, so the formula produces exact integer months.

16. ~~**O(n²) Pareto dominance check** (`src/optimization_utils.py`, 726–738, duplicated at 1094–1106) — Pairwise loop is redundant for 2D: Stage 1 sort-and-sweep already produces correct frontier. Stage 2 only needed for N>2 objectives.~~ **Fixed:** Removed O(n²) Stage 2 at both locations (MILP and GA paths). Verified: sort-and-sweep alone produces 0 dominated points and strictly increasing production on a 259-solution frontier.

20. ~~**Regression weights not validated** (`src/inference_optimized.py`, 352–356) — `_get_regression_weights` returns raw column values (todu_amt_pile_h6, oa_amt_h0, or n_observations) without checking for non-negativity, zeros, or extreme outliers. Zero weights silently drop samples; no normalization means one high-exposure bin can dominate regularization.~~ **Fixed:** Added non-negativity clip, outlier cap at 99th percentile, and normalization to sum=N. Returns None if all weights are zero.

26. ~~**MILP Pareto sweep is a discrete approximation** (`src/optimization_utils.py`, `trace_pareto_frontier`) — Frontier is built from MILPs on a linear grid of risk targets (`pareto_n_points`) plus mask dedup. Many Pareto-optimal monotone masks may never appear as optima for any grid target, so `optimum_risk` can map to a **suboptimal** production point.~~ **Fixed:** Added iterative bisection refinement (up to 3 rounds) after the linear sweep. Between each pair of adjacent Pareto points with risk gap > 0.02 pp, a midpoint MILP solve discovers missed masks. On `no_premium_cd` this grew the frontier from 72→254 solutions and recovered +782K€ (+1.6%) production at the same risk cap.

27. ~~**Legacy 2-var Pareto dedup by rounded risk** (`src/optimization_utils.py`, `get_optimal_solutions`) — Drops duplicate `b2_ever_h6` after `round(4)`, which can merge distinct frontier points and remove the best production for a tight risk cap.~~ **Fixed:** Replaced `round(4)` dedup with exact `(b2_ever_h6, oa_amt_h0)` dedup. The Pareto cummax filter already removes dominated points, so rounding-based dedup was both redundant and destructive.

30. ~~**Scenario row selection assumes sort order** (`src/plots.py`, `_get_selected_solution_row`) — `b2 <= optimum_risk` then `tail(1)` is max production under cap only if `data_summary` is sorted by increasing `b2_ever_h6` (true for default MILP/legacy outputs, fragile if CSV reload/merge reorders rows).~~ **Fixed:** as part of #4 — explicit `sort_values(["b2_ever_h6", "oa_amt_h0"]).tail(1)` now used.

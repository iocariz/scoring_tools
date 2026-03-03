# Enhancement Roadmap

## Tier 2 — High value, moderate effort

### Temporal cross-validation
- Replace random k-fold with walk-forward or expanding-window splits in `_select_model_type_cv` / `_select_feature_set_cv` (`src/inference_optimized.py`)
- Prevents temporal leakage and produces time-aware performance estimates
- The CV infrastructure already exists — main work is swapping the splitter and adding a config flag

### Rolling-window PSI
- Extend `src/stability.py` to compute PSI across monthly vintages (not just Main vs. MR)
- `compute_monthly_metrics` in `src/trends.py` already aggregates by month; wire it to `calculate_psi`
- Store the PSI time series and plot it in trends
- Turns drift detection from a binary flag into a trajectory for early warning

### Alert delivery (Slack/webhook)
- `src/alerts.py` writes JSON files only — alerts reach nobody
- Add a pluggable `AlertSink` interface with implementations: `JsonSink` (existing), `WebhookSink`, `SlackSink`, `EmailSink`
- Configure via `config.toml` `[alerts]` section
- Integrate SPC anomalies from `src/trends.py` into the `DriftAlert` system (currently only PSI alerts)

### Alert deduplication & escalation
- Track alert history across runs (append to a persistent JSON/SQLite store)
- Suppress re-alerting on the same drift within a configurable window
- Escalate moderate → critical if drift persists for N consecutive runs

## Tier 3 — High value, higher effort

### Adaptive Pareto sweep + warm-starting
- Replace the linear 50-point sweep in `trace_pareto_frontier` with adaptive bisection that concentrates points near frontier "knees"
- Pass previous MILP solution as warm-start hint to HiGHS for adjacent risk targets
- Expected to produce a better frontier in less total solve time

### Model-based reject inference
- Only parceling (acceptance-rate uplift) is implemented in `src/reject_inference.py`
- Add propensity-score reweighting as a second method (simplest principled approach)
- Optionally add augmentation (fit model on booked, predict rejects)
- Build a validation framework: if any "through-the-door" data exists, benchmark RI methods against it

### Champion/challenger framework
- Run two configs in parallel and generate a comparison report
- The pipeline is already parameterized by config — main work is the comparison output
- Highlight where frontiers diverge, which segments are affected, net production/risk delta
- High value for model governance and regulatory discussions

### Prediction intervals / uncertainty quantification
- Models produce point predictions — sparse bins get no explicit uncertainty
- Add conformal prediction intervals or quantile regression estimators
- Surface uncertainty in optimization (e.g., flag Pareto solutions that depend on unreliable bin predictions)

### Robust allocation under uncertainty
- Bootstrap CIs exist at segment level but are ignored during global allocation in `src/global_optimizer.py`
- Implement robust optimization: optimize worst-case production at the 5th-percentile frontier
- Alternatively: stochastic programming over the bootstrap samples

### Cross-period Pareto comparison
- After applying main-period cutoffs to MR data, check whether the main-period frontier is still Pareto-optimal
- If the MR data would produce a strictly dominating frontier, flag model drift

### Change-point detection in trends
- SPC with rolling median/MAD is the only method in `src/trends.py`
- Add CUSUM, EWMA, or Bayesian change-point detection (e.g., `ruptures` library)
- Add seasonal decomposition (STL) to reduce false SPC anomalies from calendar effects

### Full N>2 variable support
- `fixed_cutoffs` is hard-blocked for 3+ variables
- `mask_to_cutoffs` uses per-dimension projections (lossy) for N>2
- Legacy enumeration fallback is 2-var only
- `plot_3d_surface` is 2-var only
- Complete the N-D generalization across all code paths

### Per-segment production floor in MILP
- `min_production` exists in global allocator's `SegmentConstraints` but not in the per-segment MILP
- Adding it would let a segment guarantee minimum lending volume independently of global allocation

### PDF report export
- Reports are HTML only (`src/reporting.py`)
- Add `weasyprint` or `playwright` PDF renderer for audit packages

### Dashboard write-back
- The Dash dashboard (`dashboard.py`) is read-only
- Allow users to adjust cutoffs in the cutoff explorer and write back to `config.toml` / `segments.toml`

### Vintage-based cohort analysis
- Pipeline processes a date range as a single block
- Add vintage-level analysis: cohort by origination month, track risk evolution per cohort

### Integration test with fixture dataset
- Tests use synthetic DataFrames — no end-to-end coverage
- Add a small versioned fixture dataset and an integration test running the full pipeline from `config.toml` → reports

### Benchmark suite
- No performance regression detection
- Add benchmarks for the MILP solver and full pipeline on a fixed synthetic dataset
- Track across commits via `pytest-benchmark` or `asv`

### Schema evolution / config migration
- `config.toml` has legacy fields auto-promoted via `_auto_populate_bins`
- Consider a versioned schema (`config_version = 2`) with explicit migration

# Feature Roadmap

Features that improve cutoff optimization quality — better inputs, better search, better validation.

## Better Inputs to the Optimizer

### 1. Score Calibration Report
If the risk model is miscalibrated (predicted risk ≠ observed risk by decile), the MILP optimizes against wrong risk estimates and the "optimal" cutoff is suboptimal in reality. A calibration curve with Hosmer-Lemeshow test would flag when model inputs to the optimizer are stale, triggering retraining before re-optimizing.

### 2. Champion-Challenger for Risk Models
The optimizer is only as good as the risk surface it sees. Running two models side-by-side (using existing `delong_test`, Gini/KS comparison) and feeding the better model's predictions into the MILP would improve the risk surface the optimizer works with. Leverages existing `inference_variables` decoupling.

### 3. Vintage-Aware Risk Surface
Current risk is computed from the full observation window. If recent vintages show different risk patterns (e.g., post-COVID cohorts behaving differently), the optimizer uses a blended risk surface that may not reflect current origination quality. Weighting recent vintages more heavily in the aggregation grid would give the optimizer a more accurate risk surface.

### 4. Reject Inference Refinement
The optimizer's repesca (swap-in) risk estimates depend entirely on reject inference quality. Improvements to the parceling model directly improve cutoff quality:
- **Bivariate parceling** — current method is univariate (acceptance rate per bin). Joint modeling of acceptance and risk across multiple score dimensions would reduce bias.
- **External data augmentation** — if external bureau data is available for rejected applicants, it can anchor the reject risk estimate instead of relying solely on the acceptance rate model.

### 13. Data-Quality Gate for the Optimizer
DQ checks already run in `src/data_quality.py` but only produce log output — they do not block optimization. A hard gate that refuses to run the MILP when key signals degrade (e.g., >5% missing on any `inference_variable` in the most recent month, score-distribution PSI > threshold vs training window, negative exposure counts in MR) prevents garbage-in optimization and forces a data refresh before cutoffs move. Output: a `dq_report.json` consumed as a precondition by `run_optimization_phase`.

## Better Search / Solver

### 5. Multi-Objective Pareto with Additional Dimensions
Current Pareto frontier is 2D (risk vs production). Adding a third objective (e.g., concentration risk, portfolio diversification, or expected loss volatility) would find cutoffs that are not just risk/production optimal but also resilient. Requires extending the MILP or using NSGA-III (pymoo already imported for GA fallback).

### 6. Robust Optimization Under Uncertainty
The MILP uses point estimates for cell-level risk. Incorporating uncertainty (e.g., confidence intervals from bootstrap) into the optimization would find cutoffs that are optimal in the worst case, not just the expected case. Formulation: `maximize production subject to risk ≤ target` becomes `maximize production subject to risk_upper_CI ≤ target`.

### 7. Dynamic / Rolling Cutoff Optimization
Current pipeline optimizes cutoffs at a point in time. A rolling mode that re-optimizes monthly on a sliding window and tracks how cutoffs evolve would reveal whether the current static cutoff is diverging from the rolling optimum, signaling when to refresh.

### 8. Automated Segment Discovery
Segments are manually defined in `segments.toml`. A data-driven step (decision tree or k-means on risk predictors) that discovers natural population clusters could reveal segment boundaries where separate cutoffs would outperform a single grid. The optimizer could then be run per discovered segment.

### 14. Economic-Value Objective
Current MILP maximizes `oa_amt_h0` (production) subject to a risk cap. Real P&L depends on margin, LGD, funding cost, and pricing per bin — a cell with twice the margin but slightly higher risk can still be the right accept. Replace the linear production objective with `Σ cell_value × x_cell` where `cell_value = production × (expected_margin − LGD × risk − funding_cost)`, driven from a per-bin economics table. Keep the current risk constraint; add a "production floor" constraint to avoid degenerate ultra-low-volume high-margin solutions. Closes todo #29.

## Better Validation of Cutoffs

### 9. Out-of-Time Backtesting
"If we had applied today's cutoffs N months ago, what would have happened?" Loop over historical windows calling the existing optimization pipeline. Measures cutoff stability — if optimal cutoffs change dramatically month to month, the current solution is fragile and shouldn't be trusted.

### 10. Cutoff Drift Detection
Monitor *applied cutoffs* vs *optimal cutoffs* over time, analogous to existing PSI/CSI for score distributions. When the gap exceeds a threshold, the portfolio is operating suboptimally and cutoffs should be refreshed. New section in the consolidated report.

### 11. Swap Impact Tracking
After cutoffs are deployed, track whether the actual swap-in population's risk matches the reject-inference prediction. If swap-in risk is consistently higher than predicted, the reject inference model is biased and cutoffs are too aggressive. Feeds back into feature #4.

### 12. What-If Scenario API
Lightweight REST API that accepts a cutoff configuration and returns projected risk/production/swap metrics instantly. Lets stakeholders explore "what if we tighten bin 3 by one notch?" without running the full pipeline. Builds on existing `interactive_allocator.py` and `gradio_dashboard.py`.

### 15. Fairness / Disparate-Impact Analysis
For credit decisions, differential approval rates across protected-class proxies (income band, geography, age tier) are a regulatory exposure, not a theoretical concern. Add a post-optimization diagnostic that, for each chosen cutoff, reports approval-rate deltas by proxy group and flags bins where the monotone rectangular policy produces disparate impact. Output: a `fairness_report.html` section alongside the existing scenario report; hard-warn when a proxy group's approval rate differs by more than a configurable threshold (default 20% of the majority rate — the 80% rule). Uses existing aggregation infrastructure in `src/consolidation.py`.

## Governance & Operations

### 16. Versioned Cutoff Registry
Every deployed cutoff should be reproducible from `(config_hash, model_hash, data_hash, cutoff_hash) → deployment_date`. Today this provenance is scattered across filenames and log lines. A durable registry (e.g., `registry/cutoffs.parquet`) recording each deployment with those four hashes, the segment, the operator, and the resulting Pareto position enables: drift forensics (#10 and #11 become tractable), regulator-facing reproducibility, and automated rollback ("reapply last-known-good cutoff"). Cheap to build on top of existing `OutputPaths` and `persistence.py` metadata sidecars; high leverage when something goes wrong in production.

## Reporting & Consumption

*The pipeline's sophistication is often invisible to readers of the output. These items improve how existing results get consumed, not what gets computed.*

### 17. Executive Summary Tab (Excel, front page)
`export_consolidated_excel` opens depth-first; committee-readable numbers are buried several tabs in. Add a front tab with 5 rows: recommended cutoff, production, optimum risk with CI, swap-in exposure, top DQ/MR caveats. Generated from the same KPI source as the rest of the workbook — not a separate computation path.

### 18. Policy-Diff Tab (Excel)
What changed vs the last deployed cutoff: cells moved Accept→Reject (and vice versa), production/risk delta, economic impact using the feature #14 cell value. This is what operations actually needs to open Excel for; currently they reconstruct it by hand. Depends on feature #16 (registry) to know what "last deployed" is.

### 19. Metadata-Driven Conditional Formatting with Provenance (Excel)
Bin-level cells colored by risk, production, and swap-in share, driven by a metadata layer rather than hardcoded rules. Every cell shows its `risk_source` (`mr_observed` / `h3_extrapolated` / `model_fallback` / `main_imputed`) and applied RI multiplier inline. Makes the audit trail visible in the artifact auditors actually read.

### 20. HTML Report TL;DR Panel
Top-of-page scenario-comparison block: pessimistic / base / optimistic on one row, with bootstrap CIs rendered as error bars (not buried in a downstream table). One-sentence recommendation. Today readers scroll to find the conclusion; this puts it where their eye lands first.

### 21. Inline Cell Provenance Annotations (HTML)
Hover or click on any cell in the heatmap reveals its risk source, confidence, MR extrapolation curvature (if applicable), and RI multiplier. The pipeline distinguishes these carefully — the report currently does not expose the distinction. Closes the gap between the methodology and what consumers see.

### 22. "Since Last Run" Diff Section (HTML)
KPI deltas, cutoff changes, new DQ warnings, new audit/trend anomalies. Transforms the report from a first-read document into a recurring monitoring artifact. Depends on feature #16 to know what "last run" is.

### 23. Downloadable Policy-Spec JSON (HTML)
Next to each scenario, a machine-readable cutoff specification (cell-level accept mask + bin edges + metadata hashes) that downstream systems can consume. Removes the current "copy numbers out of HTML into configuration" step that is an error magnet.

### 24. Deployed-vs-Optimal Overlay (Dashboard)
Main heatmap shows currently-deployed cutoffs overlaid on the optimal frontier in real time, with delta colored. Operationalizes features #10 (drift detection) and #11 (swap tracking) visually rather than as numbers in a report. Answers "are we still on the frontier?" at a glance.

### 25. Cell Drill-Down to Contributing Loans (Dashboard)
Click a cell → contributing loan count, risk-source breakdown, confidence, and a sample of the underlying applications. Today the dashboard stops at the cell level; analysts debugging a surprising risk number have to leave the tool and query SAS or Excel exports.

### 26. Single KPI Module
A shared `src/report_kpis.py` that Excel, HTML, and dashboard all import. Prevents the current drift where each venue recomputes "optimum risk" or "swap-in production" slightly differently and quietly disagrees. Also the natural place to centralize CI computation for #20.

### 27. Config-Driven Report Sections
`[report.sections]` in `config.toml` enables/disables sections (sensitivity, RI optimizer output, MR diagnostics, etc.) without editing templates. Currently skipping a section means forking the Jinja2 template or the Excel exporter. Lets stakeholders produce audience-specific reports from the same run.

### 28. Comparative Mode (Two Runs Side-by-Side)
Load two scenario outputs (pre/post retrain, two segment variants, champion/challenger model — feature #2) and render every chart and table with both overlaid, with deltas. Today comparing means opening two reports in separate browser tabs and eyeballing.

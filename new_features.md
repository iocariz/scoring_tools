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

## Better Search / Solver

### 5. Multi-Objective Pareto with Additional Dimensions
Current Pareto frontier is 2D (risk vs production). Adding a third objective (e.g., concentration risk, portfolio diversification, or expected loss volatility) would find cutoffs that are not just risk/production optimal but also resilient. Requires extending the MILP or using NSGA-III (pymoo already imported for GA fallback).

### 6. Robust Optimization Under Uncertainty
The MILP uses point estimates for cell-level risk. Incorporating uncertainty (e.g., confidence intervals from bootstrap) into the optimization would find cutoffs that are optimal in the worst case, not just the expected case. Formulation: `maximize production subject to risk ≤ target` becomes `maximize production subject to risk_upper_CI ≤ target`.

### 7. Dynamic / Rolling Cutoff Optimization
Current pipeline optimizes cutoffs at a point in time. A rolling mode that re-optimizes monthly on a sliding window and tracks how cutoffs evolve would reveal whether the current static cutoff is diverging from the rolling optimum, signaling when to refresh.

### 8. Automated Segment Discovery
Segments are manually defined in `segments.toml`. A data-driven step (decision tree or k-means on risk predictors) that discovers natural population clusters could reveal segment boundaries where separate cutoffs would outperform a single grid. The optimizer could then be run per discovered segment.

## Better Validation of Cutoffs

### 9. Out-of-Time Backtesting
"If we had applied today's cutoffs N months ago, what would have happened?" Loop over historical windows calling the existing optimization pipeline. Measures cutoff stability — if optimal cutoffs change dramatically month to month, the current solution is fragile and shouldn't be trusted.

### 10. Cutoff Drift Detection
Monitor *applied cutoffs* vs *optimal cutoffs* over time, analogous to existing PSI/CSI for score distributions. When the gap exceeds a threshold, the portfolio is operating suboptimally and cutoffs should be refreshed. New section in the consolidated report.

### 11. Swap Impact Tracking
After cutoffs are deployed, track whether the actual swap-in population's risk matches the reject-inference prediction. If swap-in risk is consistently higher than predicted, the reject inference model is biased and cutoffs are too aggressive. Feeds back into feature #4.

### 12. What-If Scenario API
Lightweight REST API that accepts a cutoff configuration and returns projected risk/production/swap metrics instantly. Lets stakeholders explore "what if we tighten bin 3 by one notch?" without running the full pipeline. Builds on existing `interactive_allocator.py` and `gradio_dashboard.py`.

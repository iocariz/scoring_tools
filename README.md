# Scoring Optimization Tools

## Overview

This project implements a risk scoring and optimization pipeline for credit portfolio management. It processes loan application data (`demand` and `booked`), calculates key risk indicators (e.g., `b2_ever_h6`), and determines optimal cutoffs to maximize production while minimizing risk.

Recent work has focused on integrating a **Recent Monitoring (MR)** period to validate model performance and strategy stability over a newer time window.

## Workflow & Steps Followed

### 1. Data Preprocessing & Initial Analysis

* **Data Loading**: SAS datasets are loaded and standardized (column names, types).
* **Filtering**: Data is filtered by date ranges (`date_ini_book_obs`, `date_fin_book_obs`) and specific segments.
* **Transformation Rate**: Calculated the conversion from eligible applications to booked loans.
* **Feature Engineering**: Created risk indicators like `todu_amt_pile_h6` and `todu_30ever_h6`.

### 2. Risk Model Inference

* **Modeling**: A Hurdle-Ridge regression model is trained to predict `todu_30ever_h6` (risk) based on loan characteristics (`oa_amt`, etc.).
* **Inference**: The model fills in missing risk values for rejected applications (repesca), allowing for a theoretical "what-if" analysis on the total population.

### 3. Optimization Pipeline (Main Period)

* **Aggregation**: Data is aggregated by clusters (e.g., `sc_octroi_new_clus`, `new_efx_clus`).
* **Feasible Solutions**: The system iterates through all possible cutoff combinations across the cluster variables.
* **KPI Calculation**: For each solution, it calculates resulting Production (€), Risk (%), and Efficiency.
* **Optimal Solution Selection**: Identifies the Pareto frontier of solutions that offer the best trade-off between risk and production.

### 4. MR Period Implementation (Recent Monitoring)

A specific pipeline was built to validate the strategy on the MR ("Mois Récent") dataset:

* **Data Preparation**: Extracted MR data using specific date configurations.
* **Inference Application**: Applied the *same* risk models trained on the main period to the MR inputs to ensure consistency.
* **Full Pipeline Processing**: Instead of simple aggregation, the full `run_optimization_pipeline` was applied to MR data. This ensures:
  * **Booked (`_boo`)**: Actual observed performance.
  * **Rejected (`_rep`)**: Inferred performance for rejections.
* **Metric Calculation**:
  * `b2_ever_h6` was recalculated for the MR period using the aggregated components: `7 * sum(todu_30ever) / sum(todu_amt_pile)`.

### 5. Risk Production Summary Table (MR)

We generated a summary table to quantify the impact of the Main Period's optimal strategy on the MR data:

1. **Cut Extraction**: Loaded the specific cutoffs (e.g., `Bin 1 <= 4.0`) from the optimal solution found in the Main Period.
2. **Simulation**: Applied these cuts to the MR dataset.
3. **Metrics**:
    * **Actual**: The baseline risk/production of the portfolio as it was booked.
    * **Swap-in**: Rejected applicants in MR that *passed* the optimal cuts (potential opportunity).
    * **Swap-out**: Booked applicants in MR that *failed* the optimal cuts (risk avoidance).
    * **Optimum**: The theoretical portfolio (Actual - Swap-out + Swap-in).

### 6. Code Refactoring (`src/mr_pipeline.py`)

To improve maintainability, the MR logic was extracted from `main.py` into a dedicated module `src/mr_pipeline.py`. This encapsulates:

* Data filtering and preparation.
* Inference model application.
* Aggregation and visualization logic.
* Summary table generation.

## Key Artifacts

| Artifact | Description |
| :--- | :--- |
| `data/optimal_solution.csv` | The selected optimal strategy (cutoffs) from the main period. |
| `data/risk_production_summary_table.csv` | Impact analysis for the Main Period. |
| `data/risk_production_summary_table_mr.csv` | **New**: Impact analysis applying the strategy to the MR Period. |
| `images/risk_production_visualizer.html` | Interactive dashboard for exploring the efficient frontier. |
| `images/b2_ever_h6_vs_octroi_and_risk_score_mr.html` | 3D surface plot of risk distribution in the MR period. |

## Suggested Improvements

### Technical Improvements

1. **Unit Testing**: Add `pytest` for the new `src/mr_pipeline.py` and `src/utils.py`. The current validation relies on running the full pipeline; unit tests would allow faster iteration on specific logic (e.g., ensuring the Swap-in/Swap-out math is invariant).
2. **Configuration Management**: Move hardcoded constants (like the multiplier `7`, Z-thresholds `3.0`, or column names like `sc_octroi_new_clus`) into `config.toml`. Currently, some variable names are effectively hardcoded in the script logic.
3. **Type Hinting**: Expand Python type hints (MyPy) across all functions to prevent data type errors (e.g., ensuring `annual_coef` is always a float).
4. **Parallel Processing**: The `get_fact_sol` and `kpi_of_fact_sol` steps can be computationally intensive. Using `joblib` or `multiprocessing` to parallelize the chunk processing could significantly speed up execution.

### Functional Improvements

1. **Dynamic Variable Support**: The current logic heavily assumes a 2-variable grid (`sc_octroi_new_clus`, `new_efx_clus`). Generalizing this to N-dimensions would allow for more complex strategies.
2. **Stability Metrics**: Add specific metrics to compare Main vs. MR distribution stability (PSI - Population Stability Index) automatically in the pipeline.
3. **Scenario Analysis**: Allow the user to input a custom set of cuts (manual override) via config to test specific business hypotheses without running the full optimization search.

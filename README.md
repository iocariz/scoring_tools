# Scoring Optimization Tools

## Overview

This project implements a risk scoring and optimization pipeline for credit portfolio management. It processes loan application data (`demand` and `booked`), calculates key risk indicators (e.g., `b2_ever_h6`), and determines optimal cutoffs to maximize production while minimizing risk.

Recent work has focused on integrating a **Recent Monitoring (MR)** period to validate model performance and strategy stability over a newer time window, as well as implementing **Scenario Analysis** to test sensitivity to risk thresholds.

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
* **Parallel Processing**: Computationally intensive steps (`kpi_of_fact_sol`, `get_optimal_solutions`) are parallelized using `joblib` to maximize performance.
* **KPI Calculation**: For each solution, it calculates resulting Production (€), Risk (%), and Efficiency.
* **Optimal Solution Selection**: Identifies the Pareto frontier of solutions that offer the best trade-off between risk and production.

### 4. Scenario Analysis

To test the robustness of the strategy against varying risk appetites, the pipeline implements scenario analysis:

* **Variable**: `optimum_risk` (formerly `cz2024`).
* **Scenarios**: Automatically runs for the base configuration value +/- 0.1 (e.g., 3.7, 3.8, 3.9).
* **Outputs**: Generates distinct summary tables and optimal solution files for each scenario (e.g., `data/optimal_solution_3.7.csv`).

### 5. MR Period Implementation (Recent Monitoring)

A specific pipeline was built to validate the strategy on the MR ("Mois Récent") dataset. This pipeline is now integrated into the **Scenario Analysis** loop:

* **Process**: For *each* scenario (e.g., 1.0, 1.1, 1.2), the MR analysis is executed using that scenario's specific optimal solution.
* **Outputs**:
  * `data/risk_production_summary_table_mr_{scenario}.csv`: Impact analysis for that specific risk scenario on recent data.
  * `images/b2_ever_h6_vs_octroi_and_risk_score_mr_{scenario}.html`: Visualization of the risk surface.

### 6. Risk Production Summary Table (MR)

We generated a summary table to quantify the impact of the Main Period's optimal strategy on the MR data:

1. **Cut Extraction**: Loaded the specific cutoffs (e.g., `Bin 1 <= 4.0`) from the optimal solution found in the Main Period.
2. **Simulation**: Applied these cuts to the MR dataset.
3. **Metrics**:
    * **Actual**: The baseline risk/production of the portfolio as it was booked.
    * **Swap-in**: Rejected applicants in MR that *passed* the optimal cuts (potential opportunity).
    * **Swap-out**: Booked applicants in MR that *failed* the optimal cuts (risk avoidance).
    * **Optimum**: The theoretical portfolio (Actual - Swap-out + Swap-in).

## Key Artifacts

| Artifact | Description |
| :--- | :--- |
| `data/optimal_solution_{scenario}.csv` | The selected optimal strategy (cutoffs) for a specific risk scenario. |
| `data/risk_production_summary_table_{scenario}.csv` | Impact analysis for the Main Period (per scenario). |
| `data/risk_production_summary_table_mr_{scenario}.csv` | **New**: Impact analysis applying the strategy to the MR Period (per scenario). |
| `images/risk_production_visualizer_{scenario}.html` | Interactive dashboard for exploring the efficient frontier. |
| `data/optimal_solution.csv` | *Base Case* (Backward Compatible) file. |

## Suggested Improvements

### Technical Improvements

1. **Unit Testing**: Continue expanding `pytest` coverage. While `src/mr_pipeline.py` has tests, adding tests for the parallelized `src/utils.py` functions (mocking `joblib`) would ensure stability.
2. **CI/CD Integration**: Implement a GitHub Actions workflow to run tests and linters (ruff/mypy) on push.
3. **Containerization**: Create a `Dockerfile` to standardize the execution environment (Python version, system dependencies for `sas7bdat`).
4. **Error Handling**: Enhance exception handling in the parallel workers to ensure robust logging and graceful failure if a single chunk fails.

### Functional Improvements

1. **Dynamic Variable Support**: The current logic heavily assumes a 2-variable grid (`sc_octroi_new_clus`, `new_efx_clus`). Generalizing this to N-dimensions would allow for more complex strategies.
2. **Stability Metrics**: Add specific metrics to compare Main vs. MR distribution stability (PSI - Population Stability Index) automatically in the pipeline.
3. **Interactive Reporting**: Consolidate the multiple HTML outputs into a single Streamlit or Dash app for easier navigation between scenarios.

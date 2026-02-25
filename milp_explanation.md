# MILP Optimization in Scoring Tools

Mixed-Integer Linear Programming (MILP) is used in this codebase to find the optimal credit risk policies (cutoffs) that maximize loan production while strictly adhering to a predefined risk budget (max bad rate constraint). This approach replaces legacy brute-force enumeration methods, allowing the system to efficiently handle multi-dimensional score grids (e.g., matching an internal score with an external bureau score or even 3+ variables).

The entire optimization process logic is primarily defined in [src/optimization_utils.py](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py). The process executes in three key stages: Problem Formulation, Monotonicity Constraints, and the Pareto Sweep.

## 1. Problem Formulation: The CellGrid

The optimization problem starts by taking the historically booked and "repesca" (reject-inference adjusted) loans, which have been aggregated into bins forming an N-dimensional grid of cells (e.g., Octroi Score Bin × EFX Score Bin).

The [CellGrid](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#28-72) class normalizes this data into a problem space:
- **Variables**: Each dimension of the grid (e.g., internal score, external score).
- **Cells**: Each unique combination of bins is a "cell" with a binary decision variable $x_i \in \{0, 1\}$.
  - $x_i = 1$: The policy accepts loans in this cell.
  - $x_i = 0$: The policy rejects loans in this cell.
- **KPIs**: Each cell contains the total aggregated Production (`oa_amt_h0`), Risk Numerator (`todu_30ever_h6`), and Risk Denominator (`todu_amt_pile_h6`).

The objective is to **Maximize Total Production**:
$$\text{Maximize} \sum (Production_i \cdot x_i)$$

Subject to a **Risk Budget Constraint** (e.g., maximum allowable default rate).
Because a raw threshold is non-linear (a ratio of sum(numerator)/sum(denominator)), the constraint is linearized using a fixed risk multiplier ($M \approx 7$):

$$\sum (M \cdot Numerator_i - \frac{TargetRisk}{100} \cdot Denominator_i) \cdot x_i \leq 0$$

## 2. Monotonicity Constraints

In credit risk, it does not make business sense to approve a high-risk borrower while rejecting a low-risk borrower. 

To enforce this, the [_build_monotonicity_constraints](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#79-144) function creates a sparse matrix ($A_{mono}$) representing the logical rule: **"If a safer cell is rejected, all riskier neighbors must also be rejected."**

Mathematically, for any two adjacent cells along any dimension where `riskier` borders `safer`:
$$x_{riskier} \leq x_{safer} \implies \mathbf{x_{riskier} - x_{safer} \leq 0}$$

The code dynamically infers which direction is "riskier" using the `inv_vars` tracking:
- **Normal Variables (e.g., Risk Scores)**: Higher bin index = riskier population.
- **Inverted Variables (e.g., Credit Scores)**: Higher bin index = safer population.

This matrix ensures that the final result operates as a cohesive "staircase" boundary of cutoffs rather than a random scatterplot of accepted patches.

## 3. The Solver and Pareto Sweep

The problem is structured as vectors and matrices ($c$, $A_{mono}$, $A_{risk}$, bounds) and passed into `scipy.optimize.milp`. 

### [trace_pareto_frontier()](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#333-417)

Rather than returning just one solution, the business needs an efficient frontier (a Pareto curve) to understand the trade-offs between risk and volume. The [trace_pareto_frontier](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#333-417) function orchestrates this:

1. **Calculate Maximum Risk**: Determines the maximum possible risk if *all* cells are accepted.
2. **Sweep Risk Targets**: Iterates over $N$ (e.g., 50) evenly spaced risk targets from $0.01\%$ up to the maximum risk.
3. **Solve & Collect**: For each target risk, it invokes the MILP solver via [milp_solve_cutoffs()](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#151-233) and saves the optimal mask ($x$ vector).
4. **Pareto Filter**: It processes the collected solutions, stripping out suboptimal ones (where higher risk didn't yield more production) or raw duplicates to produce the final `pareto_df`.

### Output and Translation

Finally, the function [mask_to_cutoffs()](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#424-486) translates the binary vector $x$ back into human-readable, actionable business rules. For example, in a 2D grid, it determines: "For Internal Score Bin 1, what is the maximum External Score Bin that was accepted?" This gives the exact `cutoff_value` seen in the summary reports.

> [!NOTE] 
> If `scipy.optimize.milp` returns infeasible solutions (or if an N-dimensional grid is extremely massive), the codebase contains a fallback Genetic Algorithm (GA) implementation ([_ga_pareto_fallback](file:///Users/inigo_ocariz/src/scoring_tools/src/optimization_utils.py#569-653)) using the `pymoo` library to find near-optimal frontiers instead.

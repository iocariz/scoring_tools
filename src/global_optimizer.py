"""
Global Portfolio Optimization Module

This module provides the logic to allocate risk targets across multiple segments
to maximize global production while adhering to a global risk constraint.

Two methods are available:
- "exact": MILP solver (scipy.optimize.milp) — globally optimal
- "greedy": Hill-climbing heuristic — fast but approximate
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array


@dataclass
class AllocationResult:
    global_risk: float
    global_production: float
    allocations: dict[str, int]  # segment_name -> solution_id (sol_fac)
    segment_metrics: dict[str, dict[str, float]] # segment_name -> {risk, production}
    method: str = ""
    target: float | None = None
    segment_details: dict[str, dict] = field(default_factory=dict)  # full frontier row per segment

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with allocation summary and swap details per segment."""
        detail_cols = [
            'acct_booked_h0_boo', 'acct_booked_h0_rep', 'acct_booked_h0_cut',
            'oa_amt_h0_boo', 'oa_amt_h0_rep', 'oa_amt_h0_cut',
            'b2_ever_h6_boo', 'b2_ever_h6_rep', 'b2_ever_h6_cut',
        ]
        rows = []
        for seg, metrics in self.segment_metrics.items():
            row = {
                'segment': seg,
                'risk': metrics['risk'],
                'production': metrics['production'],
                'sol_fac': self.allocations[seg],
            }
            details = self.segment_details.get(seg, {})
            for col in detail_cols:
                if col in details:
                    row[col] = details[col]
            rows.append(row)
        return pd.DataFrame(rows)

    def to_full_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with all frontier columns for each selected point."""
        rows = []
        for seg in sorted(self.segment_details):
            row = dict(self.segment_details[seg])
            row['segment'] = seg
            rows.append(row)
        if not rows:
            return self.to_dataframe()
        df = pd.DataFrame(rows)
        # Move segment to first column
        cols = ['segment'] + [c for c in df.columns if c != 'segment']
        return df[cols]

    def __str__(self) -> str:
        lines = []
        lines.append("=" * 90)
        lines.append("GLOBAL ALLOCATION RESULTS")
        lines.append("=" * 90)
        lines.append(f"Global Risk:       {self.global_risk:.4f}%")
        lines.append(f"Global Production: {self.global_production:,.0f}")
        if self.target is not None:
            lines.append(f"Risk Target:       {self.target:.4f}%")
        if self.method:
            lines.append(f"Method:            {self.method}")
        lines.append("-" * 90)
        lines.append(f"{'Segment':<25} {'Risk':>10} {'Production':>15} {'Sol ID':>8}")
        lines.append("-" * 90)
        for seg in sorted(self.segment_metrics):
            m = self.segment_metrics[seg]
            lines.append(
                f"{seg:<25} {m['risk']:>9.3f}% {m['production']:>15,.0f} {self.allocations[seg]:>8}"
            )

        # Swap details (only if segment_details are populated with swap columns)
        has_swap = any(
            'acct_booked_h0_rep' in d or 'acct_booked_h0_cut' in d
            for d in self.segment_details.values()
        )
        if has_swap:
            lines.append("")
            lines.append("-" * 90)
            lines.append(
                f"{'Segment':<25} {'Swap In':>10} {'Swap Out':>10} "
                f"{'Prod Swap In':>15} {'Prod Swap Out':>15}"
            )
            lines.append("-" * 90)
            for seg in sorted(self.segment_details):
                d = self.segment_details[seg]
                si = d.get('acct_booked_h0_rep', 0)
                so = d.get('acct_booked_h0_cut', 0)
                si_prod = d.get('oa_amt_h0_rep', 0)
                so_prod = d.get('oa_amt_h0_cut', 0)
                lines.append(
                    f"{seg:<25} {si:>10,.0f} {so:>10,.0f} "
                    f"{si_prod:>15,.0f} {so_prod:>15,.0f}"
                )

        lines.append("=" * 90)
        return "\n".join(lines)


class GlobalAllocator:
    def __init__(self):
        self.frontiers: dict[str, pd.DataFrame] = {}

    def load_frontier(self, segment_name: str, frontier_df: pd.DataFrame):
        """
        Load an efficient frontier for a segment.
        Expected columns: ['sol_fac', 'b2_ever_h6', 'oa_amt_h0']

        Dominated points (higher risk without higher production) are pruned.
        """
        required_cols = ['sol_fac', 'b2_ever_h6', 'oa_amt_h0']
        if not all(col in frontier_df.columns for col in required_cols):
            raise ValueError(f"Frontier for {segment_name} missing required columns: {required_cols}")

        # Sort by risk ascending
        sorted_df = frontier_df.sort_values('b2_ever_h6').reset_index(drop=True)

        # Prune dominated points: keep only strictly increasing production
        prod = sorted_df['oa_amt_h0']
        cummax = prod.cummax()
        pareto_mask = cummax != cummax.shift(1)
        pareto_mask.iloc[0] = True
        n_before = len(sorted_df)
        sorted_df = sorted_df.loc[pareto_mask].reset_index(drop=True)
        n_pruned = n_before - len(sorted_df)

        self.frontiers[segment_name] = sorted_df
        if n_pruned:
            logger.info(f"Loaded frontier for {segment_name}: {len(sorted_df)} points ({n_pruned} dominated pruned)")
        else:
            logger.info(f"Loaded frontier for {segment_name}: {len(sorted_df)} points")

    def _warn_unknown_constraints(self, risk_constraints: dict | None) -> None:
        if risk_constraints:
            unknown = set(risk_constraints) - set(self.frontiers)
            if unknown:
                logger.warning(f"risk_constraints for unknown segments ignored: {unknown}")

    def optimize(
        self,
        global_risk_target: float,
        risk_constraints: dict[str, tuple[float, float]] | None = None,
        method: str = "exact",
    ) -> AllocationResult:
        """
        Allocate risk to maximize global production subject to global_risk_target.

        Parameters
        ----------
        method : str
            "exact" uses MILP solver (globally optimal).
            "greedy" uses hill-climbing heuristic.
        """
        if not self.frontiers:
            raise ValueError("No frontiers loaded. Call load_frontier() first.")

        if method == "exact":
            try:
                return self.optimize_exact(global_risk_target, risk_constraints)
            except Exception as e:
                warnings.warn(
                    f"Exact solver failed ({e}), falling back to greedy.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.optimize_greedy(global_risk_target, risk_constraints)
        elif method == "greedy":
            return self.optimize_greedy(global_risk_target, risk_constraints)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'exact' or 'greedy'.")

    # ------------------------------------------------------------------
    # Exact MILP solver
    # ------------------------------------------------------------------
    def optimize_exact(
        self,
        global_risk_target: float,
        risk_constraints: dict[str, tuple[float, float]] | None = None,
    ) -> AllocationResult:
        """
        Solve allocation as a Mixed-Integer Linear Program.

        Binary variables x[s,j] = 1 iff segment s picks frontier point j.
        - Maximize  Σ p[s,j] x[s,j]
        - Subject to:
            Σ_j x[s,j] = 1               (one point per segment)
            Σ p[s,j](r[s,j] - T) x[s,j] ≤ 0  (global risk ≤ T)
            optional per-segment risk bounds
        """
        if not self.frontiers:
            raise ValueError("No frontiers loaded. Call load_frontier() first.")

        self._warn_unknown_constraints(risk_constraints)

        segments = sorted(self.frontiers.keys())
        # Build variable index map: var_offset[s] is the starting index for segment s
        var_offset: dict[str, int] = {}
        n_vars = 0
        for seg in segments:
            var_offset[seg] = n_vars
            n_vars += len(self.frontiers[seg])

        # Objective: maximize production  →  minimize -production
        c = np.zeros(n_vars)
        for seg in segments:
            df = self.frontiers[seg]
            off = var_offset[seg]
            c[off : off + len(df)] = -df['oa_amt_h0'].values

        # All variables are binary
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=0, ub=1)

        # --- Equality constraints (one point per segment) ---
        eq_rows, eq_cols, eq_data = [], [], []
        for i, seg in enumerate(segments):
            off = var_offset[seg]
            k = len(self.frontiers[seg])
            eq_rows.extend([i] * k)
            eq_cols.extend(range(off, off + k))
            eq_data.extend([1.0] * k)

        A_eq = csc_array((eq_data, (eq_rows, eq_cols)), shape=(len(segments), n_vars))
        b_eq = np.ones(len(segments))

        # --- Inequality constraints ---
        ub_rows, ub_cols, ub_data = [], [], []
        b_ub_list: list[float] = []

        # Row 0: global risk  Σ p[s,j]*(r[s,j] - T) * x[s,j] ≤ 0
        for seg in segments:
            df = self.frontiers[seg]
            off = var_offset[seg]
            k = len(df)
            coeffs = df['oa_amt_h0'].values * (df['b2_ever_h6'].values - global_risk_target)
            ub_rows.extend([0] * k)
            ub_cols.extend(range(off, off + k))
            ub_data.extend(coeffs.tolist())
        b_ub_list.append(0.0)

        # Per-segment risk bounds (optional)
        row_idx = 1
        if risk_constraints:
            for seg, (min_r, max_r) in risk_constraints.items():
                if seg not in self.frontiers:
                    continue
                df = self.frontiers[seg]
                off = var_offset[seg]
                k = len(df)
                risks = df['b2_ever_h6'].values

                # min_r ≤ Σ r[s,j]*x[s,j]  →  -Σ r[s,j]*x[s,j] ≤ -min_r
                ub_rows.extend([row_idx] * k)
                ub_cols.extend(range(off, off + k))
                ub_data.extend((-risks).tolist())
                b_ub_list.append(-min_r)
                row_idx += 1

                # Σ r[s,j]*x[s,j] ≤ max_r
                ub_rows.extend([row_idx] * k)
                ub_cols.extend(range(off, off + k))
                ub_data.extend(risks.tolist())
                b_ub_list.append(max_r)
                row_idx += 1

        n_ub = row_idx
        A_ub = csc_array((ub_data, (ub_rows, ub_cols)), shape=(n_ub, n_vars))
        b_ub = np.array(b_ub_list)

        constraints = [
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub),
        ]

        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if not result.success:
            raise RuntimeError(f"MILP solver failed: {result.message}")

        # Decode solution
        x = result.x
        allocations = {}
        segment_metrics = {}
        segment_details = {}
        total_prod = 0.0
        total_risk_num = 0.0

        for seg in segments:
            df = self.frontiers[seg]
            off = var_offset[seg]
            k = len(df)
            seg_x = x[off : off + k]
            chosen = int(np.argmax(seg_x))
            row = df.iloc[chosen]

            allocations[seg] = int(row['sol_fac'])
            prod = row['oa_amt_h0']
            risk = row['b2_ever_h6']
            segment_metrics[seg] = {'risk': risk, 'production': prod}
            segment_details[seg] = row.to_dict()
            total_prod += prod
            total_risk_num += risk * prod

        global_risk = total_risk_num / total_prod if total_prod > 0 else 0.0

        logger.info(f"MILP solved: global_risk={global_risk:.4f}%, production={total_prod:,.0f}")
        return AllocationResult(
            global_risk=global_risk,
            global_production=total_prod,
            allocations=allocations,
            segment_metrics=segment_metrics,
            method="exact",
            target=global_risk_target,
            segment_details=segment_details,
        )

    # ------------------------------------------------------------------
    # Greedy hill-climbing solver (original algorithm)
    # ------------------------------------------------------------------
    def optimize_greedy(
        self,
        global_risk_target: float,
        risk_constraints: dict[str, tuple[float, float]] | None = None,
        max_iterations: int = 10_000,
    ) -> AllocationResult:
        """
        Allocate risk to maximize global production using greedy hill climbing.
        """
        if not self.frontiers:
            raise ValueError("No frontiers loaded. Call load_frontier() first.")

        self._warn_unknown_constraints(risk_constraints)

        # 1. Initialize with minimum viable solution for each segment
        current_indices = dict.fromkeys(self.frontiers, 0)

        # Apply min_risk constraints
        if risk_constraints:
            for seg, (min_r, _max_r) in risk_constraints.items():
                if seg in self.frontiers:
                    # Find first solution >= min_r
                    df = self.frontiers[seg]
                    valid_idx = df[df['b2_ever_h6'] >= min_r].index
                    if not valid_idx.empty:
                        current_indices[seg] = valid_idx[0]
                    else:
                        logger.warning(f"Segment {seg} cannot meet min_risk {min_r}. Using max available.")
                        current_indices[seg] = df.index[-1]

        # 2. Greedy Hill Climbing
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            if iteration % 100 == 0:
                logger.debug(f"Greedy iteration {iteration}")

            # Calculate current state
            total_production = 0.0
            weighted_risk_num = 0.0

            for seg, idx in current_indices.items():
                row = self.frontiers[seg].iloc[idx]
                prod = row['oa_amt_h0']
                risk = row['b2_ever_h6']

                total_production += prod
                weighted_risk_num += risk * prod

            if total_production == 0:
                current_global_risk = 0.0
            else:
                current_global_risk = weighted_risk_num / total_production

            # Determine Mode: Recovery vs Growth
            is_recovery_mode = current_global_risk > global_risk_target

            # Find best next step
            best_score = -float('inf')
            best_segment_to_increment = None

            for seg in sorted(self.frontiers.keys()):
                idx = current_indices[seg]
                df = self.frontiers[seg]

                if idx >= len(df) - 1:
                    continue

                next_row = df.iloc[idx + 1]

                # Check max_risk constraint per segment
                if risk_constraints and seg in risk_constraints:
                    _, max_r = risk_constraints[seg]
                    if next_row['b2_ever_h6'] > max_r:
                        continue

                # Calculate deltas
                curr_row = df.iloc[idx]
                delta_p = next_row['oa_amt_h0'] - curr_row['oa_amt_h0']

                risk_mass_next = next_row['b2_ever_h6'] * next_row['oa_amt_h0']
                risk_mass_curr = curr_row['b2_ever_h6'] * curr_row['oa_amt_h0']
                delta_risk_mass = risk_mass_next - risk_mass_curr

                if delta_p > 1e-9:
                    marginal_risk = delta_risk_mass / delta_p
                else:
                    if delta_risk_mass < 0:
                         marginal_risk = -float('inf')
                    else:
                         marginal_risk = float('inf')

                if is_recovery_mode:
                    if marginal_risk < current_global_risk:
                        surplus = delta_p * (current_global_risk - marginal_risk)
                        if surplus > best_score:
                            best_score = surplus
                            best_segment_to_increment = seg

                else:
                    new_risk_num = weighted_risk_num + delta_risk_mass
                    new_prod = total_production + delta_p
                    new_global_risk = new_risk_num / new_prod if new_prod > 0 else 0

                    if new_global_risk <= global_risk_target:
                        if delta_risk_mass <= 1e-9:
                            score = float('inf')
                        else:
                            score = delta_p / delta_risk_mass

                        if score > best_score:
                            best_score = score
                            best_segment_to_increment = seg

            if best_segment_to_increment is None:
                if is_recovery_mode:
                    logger.warning(f"Optimization stopped: Cannot reduce risk below {current_global_risk:.4f}% (Target: {global_risk_target}%). Stuck at local minimum.")
                else:
                    logger.info("Optimization stopped: Target reached or no efficient moves left.")
                break

            current_indices[best_segment_to_increment] += 1

        if iteration >= max_iterations:
            logger.warning(f"Greedy optimization hit max_iterations ({max_iterations}).")

        # 3. Compile Results
        allocations = {}
        segment_metrics = {}
        segment_details = {}

        final_total_prod = 0.0
        final_risk_num = 0.0

        for seg, idx in current_indices.items():
            row = self.frontiers[seg].iloc[idx]
            allocations[seg] = int(row['sol_fac'])
            segment_metrics[seg] = {
                'risk': row['b2_ever_h6'],
                'production': row['oa_amt_h0']
            }
            segment_details[seg] = row.to_dict()
            final_total_prod += row['oa_amt_h0']
            final_risk_num += row['b2_ever_h6'] * row['oa_amt_h0']

        final_global_risk = final_risk_num / final_total_prod if final_total_prod > 0 else 0.0

        return AllocationResult(
            global_risk=final_global_risk,
            global_production=final_total_prod,
            allocations=allocations,
            segment_metrics=segment_metrics,
            method="greedy",
            target=global_risk_target,
            segment_details=segment_details,
        )

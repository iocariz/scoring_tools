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

from src.optimization_utils import pareto_keep_indices


@dataclass
class SegmentConstraints:
    """Per-segment constraints for the global allocator."""

    min_risk: float | None = None
    max_risk: float | None = None
    min_production: float | None = None  # production floor
    locked_sol_fac: int | None = None  # lock to specific frontier point

    def to_risk_tuple(self) -> tuple[float, float] | None:
        """Convert to legacy (min_risk, max_risk) tuple."""
        if self.min_risk is None and self.max_risk is None:
            return None
        return (self.min_risk or 0.0, self.max_risk or float("inf"))


@dataclass
class AllocationResult:
    global_risk: float
    global_production: float
    allocations: dict[str, int]  # segment_name -> solution_id (sol_fac)
    segment_metrics: dict[str, dict[str, float]]  # segment_name -> {risk, production}
    method: str = ""
    target: float | None = None
    segment_details: dict[str, dict] = field(default_factory=dict)  # full frontier row per segment
    binding_constraints: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with allocation summary and swap details per segment."""
        detail_cols = [
            "acct_booked_h0_boo",
            "acct_booked_h0_rep",
            "acct_booked_h0_cut",
            "oa_amt_h0_boo",
            "oa_amt_h0_rep",
            "oa_amt_h0_cut",
            "b2_ever_h6_boo",
            "b2_ever_h6_rep",
            "b2_ever_h6_cut",
        ]
        rows = []
        for seg, metrics in self.segment_metrics.items():
            row = {
                "segment": seg,
                "risk": metrics["risk"],
                "production": metrics["production"],
                "sol_fac": self.allocations[seg],
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
            row["segment"] = seg
            rows.append(row)
        if not rows:
            return self.to_dataframe()
        df = pd.DataFrame(rows)
        # Move segment to first column
        cols = ["segment"] + [c for c in df.columns if c != "segment"]
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
            lines.append(f"{seg:<25} {m['risk']:>9.3f}% {m['production']:>15,.0f} {self.allocations[seg]:>8}")

        # Swap details (only if segment_details are populated with swap columns)
        has_swap = any("acct_booked_h0_rep" in d or "acct_booked_h0_cut" in d for d in self.segment_details.values())
        if has_swap:
            lines.append("")
            lines.append("-" * 90)
            lines.append(f"{'Segment':<25} {'Swap In':>10} {'Swap Out':>10} {'Prod Swap In':>15} {'Prod Swap Out':>15}")
            lines.append("-" * 90)
            for seg in sorted(self.segment_details):
                d = self.segment_details[seg]
                si = d.get("acct_booked_h0_rep", 0)
                so = d.get("acct_booked_h0_cut", 0)
                si_prod = d.get("oa_amt_h0_rep", 0)
                so_prod = d.get("oa_amt_h0_cut", 0)
                lines.append(f"{seg:<25} {si:>10,.0f} {so:>10,.0f} {si_prod:>15,.0f} {so_prod:>15,.0f}")

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
        required_cols = ["sol_fac", "b2_ever_h6", "oa_amt_h0"]
        if not all(col in frontier_df.columns for col in required_cols):
            raise ValueError(f"Frontier for {segment_name} missing required columns: {required_cols}")

        finite_mask = np.isfinite(frontier_df["b2_ever_h6"]) & np.isfinite(frontier_df["oa_amt_h0"])
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            logger.warning(f"Frontier for {segment_name}: dropping {n_bad} row(s) with non-finite risk/production")
            frontier_df = frontier_df.loc[finite_mask].copy()
        if frontier_df.empty:
            raise ValueError(f"Frontier for {segment_name} has no finite points after filtering")

        if frontier_df["sol_fac"].duplicated().any():
            dup_count = int(frontier_df["sol_fac"].duplicated().sum())
            logger.warning(
                f"Frontier for {segment_name}: found {dup_count} duplicate sol_fac value(s); keeping lowest-risk rows"
            )
            frontier_df = frontier_df.sort_values(["sol_fac", "b2_ever_h6", "oa_amt_h0"], ascending=[True, True, False])
            frontier_df = frontier_df.drop_duplicates(subset=["sol_fac"], keep="first")

        # Sort by risk ascending
        sorted_df = frontier_df.sort_values("b2_ever_h6").reset_index(drop=True)

        # Prune dominated points: keep only strictly increasing production. Uses the canonical
        # Pareto sweep shared with optimization_utils.trace_pareto_frontier (audit #13) so the
        # per-segment frontier and the allocation candidate set prune by the same rule.
        n_before = len(sorted_df)
        keep = pareto_keep_indices(sorted_df["oa_amt_h0"].values)
        sorted_df = sorted_df.iloc[keep].reset_index(drop=True)
        n_pruned = n_before - len(sorted_df)

        self.frontiers[segment_name] = sorted_df
        if n_pruned:
            logger.info(f"Loaded frontier for {segment_name}: {len(sorted_df)} points ({n_pruned} dominated pruned)")
        else:
            logger.info(f"Loaded frontier for {segment_name}: {len(sorted_df)} points")

    def _warn_unknown_constraints(self, constraints: dict | None) -> None:
        if constraints:
            unknown = set(constraints) - set(self.frontiers)
            if unknown:
                logger.warning(f"risk_constraints for unknown segments ignored: {unknown}")

    @staticmethod
    def _convert_legacy_constraints(
        risk_constraints: dict[str, tuple[float, float]] | None,
    ) -> dict[str, SegmentConstraints]:
        """Convert old-style (min_risk, max_risk) tuples to SegmentConstraints."""
        if not risk_constraints:
            return {}
        return {seg: SegmentConstraints(min_risk=lo, max_risk=hi) for seg, (lo, hi) in risk_constraints.items()}

    def optimize(
        self,
        global_risk_target: float,
        risk_constraints: dict[str, tuple[float, float]] | None = None,
        constraints: dict[str, SegmentConstraints] | None = None,
        global_production_floor: float | None = None,
        method: str = "exact",
    ) -> AllocationResult:
        """
        Allocate risk to maximize global production subject to global_risk_target.

        Parameters
        ----------
        risk_constraints : dict, optional
            Legacy format: {segment: (min_risk, max_risk)}.
        constraints : dict, optional
            New format: {segment: SegmentConstraints(...)}.
            Takes precedence over risk_constraints.
        global_production_floor : float, optional
            Minimum total production across all segments.
        method : str
            "exact" uses MILP solver (globally optimal).
            "greedy" uses hill-climbing heuristic.
        """
        if not self.frontiers:
            raise ValueError("No frontiers loaded. Call load_frontier() first.")

        # Convert legacy risk_constraints to SegmentConstraints if needed
        if risk_constraints and not constraints:
            constraints = self._convert_legacy_constraints(risk_constraints)

        if method == "exact":
            try:
                return self.optimize_exact(
                    global_risk_target, constraints=constraints, global_production_floor=global_production_floor
                )
            except (ValueError, RuntimeError) as e:
                warnings.warn(
                    f"Exact solver failed ({e}), falling back to greedy.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.optimize_greedy(
                    global_risk_target, constraints=constraints, global_production_floor=global_production_floor
                )
        elif method == "greedy":
            return self.optimize_greedy(
                global_risk_target, constraints=constraints, global_production_floor=global_production_floor
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'exact' or 'greedy'.")

    # ------------------------------------------------------------------
    # Exact MILP solver
    # ------------------------------------------------------------------
    def optimize_exact(
        self,
        global_risk_target: float,
        risk_constraints: dict[str, tuple[float, float]] | None = None,
        constraints: dict[str, SegmentConstraints] | None = None,
        global_production_floor: float | None = None,
    ) -> AllocationResult:
        """
        Solve allocation as a Mixed-Integer Linear Program.

        Binary variables x[s,j] = 1 iff segment s picks frontier point j.
        - Maximize  Σ p[s,j] x[s,j]
        - Subject to:
            Σ_j x[s,j] = 1               (one point per segment)
            Σ p[s,j](r[s,j] - T) x[s,j] ≤ 0  (global risk ≤ T)
            optional per-segment risk/production bounds, segment locking
        """
        if not self.frontiers:
            raise ValueError("No frontiers loaded. Call load_frontier() first.")

        # Convert legacy risk_constraints if new-style not provided
        if risk_constraints and not constraints:
            constraints = self._convert_legacy_constraints(risk_constraints)

        self._warn_unknown_constraints(constraints)

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
            c[off : off + len(df)] = -df["oa_amt_h0"].values

        # All variables are binary
        integrality = np.ones(n_vars)
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)

        # --- Handle segment locking via variable bounds ---
        locked_segments: set[str] = set()
        if constraints:
            for seg, sc in constraints.items():
                if seg not in self.frontiers or sc.locked_sol_fac is None:
                    continue
                df = self.frontiers[seg]
                match = df[df["sol_fac"] == sc.locked_sol_fac]
                if match.empty:
                    raise ValueError(
                        f"Locked sol_fac={sc.locked_sol_fac} not found in frontier for segment '{seg}'. "
                        f"Available: {df['sol_fac'].tolist()}"
                    )
                locked_idx = match.index[0]
                off = var_offset[seg]
                k = len(df)
                # Fix all vars for this segment: locked one to 1, rest to 0
                for j in range(k):
                    if j == locked_idx:
                        lb[off + j] = 1.0
                    else:
                        ub[off + j] = 0.0
                locked_segments.add(seg)
                logger.info(f"Segment '{seg}' locked to sol_fac={sc.locked_sol_fac} (frontier index {locked_idx})")

        bounds = Bounds(lb=lb, ub=ub)

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
        ub_labels: list[str] = []  # for binding constraint detection

        # Row 0: global risk  Σ p[s,j]*(r[s,j] - T) * x[s,j] ≤ 0
        for seg in segments:
            df = self.frontiers[seg]
            off = var_offset[seg]
            k = len(df)
            coeffs = df["oa_amt_h0"].values * (df["b2_ever_h6"].values - global_risk_target)
            ub_rows.extend([0] * k)
            ub_cols.extend(range(off, off + k))
            ub_data.extend(coeffs.tolist())
        b_ub_list.append(0.0)
        ub_labels.append("global_risk")

        row_idx = 1

        # Per-segment constraints (risk bounds, production floor)
        if constraints:
            for seg, sc in constraints.items():
                if seg not in self.frontiers:
                    continue
                df = self.frontiers[seg]
                off = var_offset[seg]
                k = len(df)

                # Min risk: -Σ r[s,j]*x[s,j] ≤ -min_r
                if sc.min_risk is not None:
                    risks = df["b2_ever_h6"].values
                    ub_rows.extend([row_idx] * k)
                    ub_cols.extend(range(off, off + k))
                    ub_data.extend((-risks).tolist())
                    b_ub_list.append(-sc.min_risk)
                    ub_labels.append(f"{seg}_min_risk")
                    row_idx += 1

                # Max risk: Σ r[s,j]*x[s,j] ≤ max_r
                if sc.max_risk is not None:
                    risks = df["b2_ever_h6"].values
                    ub_rows.extend([row_idx] * k)
                    ub_cols.extend(range(off, off + k))
                    ub_data.extend(risks.tolist())
                    b_ub_list.append(sc.max_risk)
                    ub_labels.append(f"{seg}_max_risk")
                    row_idx += 1

                # Per-segment production floor: -Σ p[s,j]*x[s,j] ≤ -min_production
                if sc.min_production is not None:
                    prods = df["oa_amt_h0"].values
                    ub_rows.extend([row_idx] * k)
                    ub_cols.extend(range(off, off + k))
                    ub_data.extend((-prods).tolist())
                    b_ub_list.append(-sc.min_production)
                    ub_labels.append(f"{seg}_min_production")
                    row_idx += 1

        # Global production floor: -Σ_all p[s,j]*x[s,j] ≤ -global_production_floor
        if global_production_floor is not None:
            for seg in segments:
                df = self.frontiers[seg]
                off = var_offset[seg]
                k = len(df)
                prods = df["oa_amt_h0"].values
                ub_rows.extend([row_idx] * k)
                ub_cols.extend(range(off, off + k))
                ub_data.extend((-prods).tolist())
            b_ub_list.append(-global_production_floor)
            ub_labels.append("global_production_floor")
            row_idx += 1

        n_ub = row_idx
        A_ub = csc_array((ub_data, (ub_rows, ub_cols)), shape=(n_ub, n_vars))
        b_ub = np.array(b_ub_list)

        milp_constraints = [
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub),
        ]

        result = milp(
            c=c,
            constraints=milp_constraints,
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
            rounded = np.round(seg_x).astype(int)
            if rounded.sum() != 1:
                logger.warning(
                    f"MILP solution for {seg} has {rounded.sum()} selected points "
                    f"(expected 1). Using argmax as fallback."
                )
            chosen = int(np.argmax(seg_x))
            row = df.iloc[chosen]

            allocations[seg] = int(row["sol_fac"])
            prod = row["oa_amt_h0"]
            risk = row["b2_ever_h6"]
            segment_metrics[seg] = {"risk": risk, "production": prod}
            segment_details[seg] = row.to_dict()
            total_prod += prod
            total_risk_num += risk * prod

        global_risk = total_risk_num / total_prod if total_prod > 0 else 0.0

        # Detect binding constraints (slack < 1e-6)
        binding = []
        if n_ub > 0:
            # Keep diagnostics sparse-safe to avoid dense matrix blow-ups on large frontiers.
            ub_values = np.asarray(A_ub @ x, dtype=float).ravel()
            for i, label in enumerate(ub_labels):
                slack = b_ub[i] - ub_values[i]
                if abs(slack) < 1e-6:
                    binding.append(label)

        logger.info(f"MILP solved: global_risk={global_risk:.4f}%, production={total_prod:,.0f}")
        if binding:
            logger.info(f"Binding constraints: {binding}")
        return AllocationResult(
            global_risk=global_risk,
            global_production=total_prod,
            allocations=allocations,
            segment_metrics=segment_metrics,
            method="exact",
            target=global_risk_target,
            segment_details=segment_details,
            binding_constraints=binding,
        )

    # ------------------------------------------------------------------
    # Greedy hill-climbing solver (original algorithm)
    # ------------------------------------------------------------------
    def optimize_greedy(
        self,
        global_risk_target: float,
        risk_constraints: dict[str, tuple[float, float]] | None = None,
        constraints: dict[str, SegmentConstraints] | None = None,
        global_production_floor: float | None = None,
        max_iterations: int = 10_000,
    ) -> AllocationResult:
        """
        Allocate risk to maximize global production using greedy hill climbing.
        """
        if not self.frontiers:
            raise ValueError("No frontiers loaded. Call load_frontier() first.")

        # Convert legacy risk_constraints if new-style not provided
        if risk_constraints and not constraints:
            constraints = self._convert_legacy_constraints(risk_constraints)

        self._warn_unknown_constraints(constraints)

        # 1. Initialize with minimum viable solution for each segment
        current_indices = dict.fromkeys(self.frontiers, 0)
        locked_segments: set[str] = set()

        if constraints:
            for seg, sc in constraints.items():
                if seg not in self.frontiers:
                    continue

                # Segment locking: fix to the locked frontier point
                if sc.locked_sol_fac is not None:
                    df = self.frontiers[seg]
                    match = df[df["sol_fac"] == sc.locked_sol_fac]
                    if match.empty:
                        raise ValueError(
                            f"Locked sol_fac={sc.locked_sol_fac} not found in frontier for segment '{seg}'. "
                            f"Available: {df['sol_fac'].tolist()}"
                        )
                    current_indices[seg] = match.index[0]
                    locked_segments.add(seg)
                    logger.info(f"Greedy: segment '{seg}' locked to sol_fac={sc.locked_sol_fac}")
                    continue

                # Apply min_risk constraints
                if sc.min_risk is not None:
                    df = self.frontiers[seg]
                    valid_idx = df[df["b2_ever_h6"] >= sc.min_risk].index
                    if not valid_idx.empty:
                        current_indices[seg] = valid_idx[0]
                    else:
                        logger.warning(f"Segment {seg} cannot meet min_risk {sc.min_risk}. Using max available.")
                        current_indices[seg] = df.index[-1]

                # Apply min_production: start at first point meeting production floor
                if sc.min_production is not None:
                    df = self.frontiers[seg]
                    valid_idx = df[df["oa_amt_h0"] >= sc.min_production].index
                    if not valid_idx.empty:
                        candidate = max(current_indices[seg], valid_idx[0])
                        # Check that min_production doesn't force a max_risk violation
                        if sc.max_risk is not None:
                            candidate_risk = df.loc[candidate, "b2_ever_h6"]
                            if candidate_risk > sc.max_risk:
                                logger.warning(
                                    f"Segment {seg}: min_production={sc.min_production} "
                                    f"forces risk={candidate_risk:.2f}% > max_risk={sc.max_risk:.2f}%. "
                                    f"Keeping lower index to respect max_risk."
                                )
                            else:
                                current_indices[seg] = candidate
                        else:
                            current_indices[seg] = candidate
                    else:
                        logger.warning(
                            f"Segment {seg} cannot meet min_production {sc.min_production}. Using max available."
                        )
                        # Respect max_risk even in fallback
                        if sc.max_risk is not None:
                            valid_risk = df[df["b2_ever_h6"] <= sc.max_risk]
                            if not valid_risk.empty:
                                current_indices[seg] = valid_risk.index[-1]
                            else:
                                logger.warning(
                                    f"Segment {seg}: no frontier point meets max_risk={sc.max_risk}. Using safest point."
                                )
                                current_indices[seg] = df.index[0]
                        else:
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
                prod = row["oa_amt_h0"]
                risk = row["b2_ever_h6"]

                total_production += prod
                weighted_risk_num += risk * prod

            if total_production == 0:
                current_global_risk = 0.0
            else:
                current_global_risk = weighted_risk_num / total_production

            # Determine Mode: Recovery vs Growth
            is_recovery_mode = current_global_risk > global_risk_target

            # Find best next step
            best_score = -float("inf")
            best_segment_to_increment = None
            best_candidate_risk = float("inf")
            needs_more_production = global_production_floor is not None and total_production < global_production_floor

            for seg in sorted(self.frontiers.keys()):
                # Skip locked segments
                if seg in locked_segments:
                    continue

                idx = current_indices[seg]
                df = self.frontiers[seg]

                if idx >= len(df) - 1:
                    continue

                next_row = df.iloc[idx + 1]

                # Check max_risk constraint per segment
                if constraints and seg in constraints and constraints[seg].max_risk is not None:
                    if next_row["b2_ever_h6"] > constraints[seg].max_risk:
                        continue

                # Calculate deltas
                curr_row = df.iloc[idx]
                delta_p = next_row["oa_amt_h0"] - curr_row["oa_amt_h0"]

                risk_mass_next = next_row["b2_ever_h6"] * next_row["oa_amt_h0"]
                risk_mass_curr = curr_row["b2_ever_h6"] * curr_row["oa_amt_h0"]
                delta_risk_mass = risk_mass_next - risk_mass_curr

                if delta_p > 1e-9:
                    marginal_risk = delta_risk_mass / delta_p
                else:
                    if delta_risk_mass < 0:
                        marginal_risk = -float("inf")
                    else:
                        marginal_risk = float("inf")

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
                        if needs_more_production:
                            score = delta_p
                            if score > best_score or (score == best_score and new_global_risk < best_candidate_risk):
                                best_score = score
                                best_candidate_risk = new_global_risk
                                best_segment_to_increment = seg
                        else:
                            if delta_risk_mass <= 1e-9:
                                score = float("inf")
                            else:
                                score = delta_p / delta_risk_mass

                            if score > best_score:
                                best_score = score
                                best_segment_to_increment = seg

            if best_segment_to_increment is None:
                if is_recovery_mode:
                    logger.warning(
                        f"Optimization stopped: Cannot reduce risk below {current_global_risk:.4f}% (Target: {global_risk_target}%). Stuck at local minimum."
                    )
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
            allocations[seg] = int(row["sol_fac"])
            segment_metrics[seg] = {"risk": row["b2_ever_h6"], "production": row["oa_amt_h0"]}
            segment_details[seg] = row.to_dict()
            final_total_prod += row["oa_amt_h0"]
            final_risk_num += row["b2_ever_h6"] * row["oa_amt_h0"]

        final_global_risk = final_risk_num / final_total_prod if final_total_prod > 0 else 0.0

        # Heuristic binding flags (greedy has no dual slacks — tolerance-based only)
        binding_heuristic: list[str] = []
        tol = 1e-3
        if abs(final_global_risk - global_risk_target) < tol:
            binding_heuristic.append("global_risk")
        if (
            global_production_floor is not None
            and abs(final_total_prod - global_production_floor) / max(global_production_floor, 1.0) < 1e-4
        ):
            binding_heuristic.append("global_production_floor")
        if constraints:
            for seg, sc in constraints.items():
                if seg not in segment_metrics:
                    continue
                r = segment_metrics[seg]["risk"]
                p = segment_metrics[seg]["production"]
                if sc.min_risk is not None and abs(r - sc.min_risk) < tol:
                    binding_heuristic.append(f"{seg}_min_risk")
                if sc.max_risk is not None and abs(r - sc.max_risk) < tol:
                    binding_heuristic.append(f"{seg}_max_risk")
                if sc.min_production is not None and abs(p - sc.min_production) / max(sc.min_production, 1.0) < 1e-4:
                    binding_heuristic.append(f"{seg}_min_production")

        # Warn if global production floor is not met (greedy only grows, so this is informational)
        if global_production_floor is not None and final_total_prod < global_production_floor:
            logger.warning(
                f"Greedy result production {final_total_prod:,.0f} is below global floor {global_production_floor:,.0f}"
            )
            raise RuntimeError(
                f"Greedy solver could not satisfy global production floor {global_production_floor:,.0f} "
                f"(achieved {final_total_prod:,.0f}) under target {global_risk_target:.4f}%"
            )

        return AllocationResult(
            global_risk=final_global_risk,
            global_production=final_total_prod,
            allocations=allocations,
            segment_metrics=segment_metrics,
            method="greedy",
            target=global_risk_target,
            segment_details=segment_details,
            binding_constraints=binding_heuristic,
        )

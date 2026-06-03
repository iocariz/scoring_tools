"""
Portfolio-owner-facing summaries: policy/cutoff tables and allocation narratives.

These helpers format existing optimizer outputs into readable tables and prose
for committees and policy owners (no change to optimization math).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.global_optimizer import AllocationResult, SegmentConstraints

# Columns on efficient-frontier rows that are KPIs, not var0 bin indices
_FRONTIER_NON_BIN_KEYS = frozenset(
    {
        "sol_fac",
        "b2_ever_h6",
        "oa_amt_h0",
        "todu_30ever_h6",
        "todu_amt_pile_h6",
        "todu_30ever_h3",
        "todu_amt_pile_h3",
        "acceptance_mask",
        "text",
        "segment",
    }
)


def _is_bin_column_key(name: object) -> bool:
    """True if *name* looks like a var0 bin index column (numeric label)."""
    if isinstance(name, (int, np.integer)) and not isinstance(name, bool):
        return True
    if not isinstance(name, str):
        return False
    if name in _FRONTIER_NON_BIN_KEYS:
        return False
    if name.startswith(("acct_", "oa_amt_", "b2_ever", "todu_", "ri_")):
        return False
    try:
        float(name)
    except ValueError:
        return False
    return True


def policy_cutoff_table_from_allocation(
    segment_details: dict[str, dict],
    *,
    scenario: str = "base",
    target_risk_pct: float | None = None,
    var0_label: str = "score_bin_axis_0",
    var1_label: str = "max_accepted_bin_axis_1",
) -> pd.DataFrame:
    """
    Build a long-form policy/cutoff table from selected efficient-frontier rows.

    For classic 2-score grids, numeric columns on the frontier row are interpreted
    as var0-bin → var1 cutoff (max accepted bin on axis 1). For N>2 grids, includes
    a row with ``acceptance_mask`` when present.

    Parameters
    ----------
    segment_details:
        ``AllocationResult.segment_details`` — one dict per segment (frontier row).
    scenario:
        Scenario label (e.g. ``base``).
    target_risk_pct:
        Optional global risk target used for this allocation run (for traceability).
    var0_label, var1_label:
        Column names for the 2-var interpretation (portfolio-owner readable).

    Returns
    -------
    DataFrame with one row per segment×bin (2-var) or one row per segment (N-var mask).
    """
    rows: list[dict] = []
    for segment in sorted(segment_details):
        row = segment_details[segment]
        sol = int(row.get("sol_fac", -1))
        risk = float(row.get("b2_ever_h6", np.nan))
        prod = float(row.get("oa_amt_h0", np.nan))

        swap_in = row.get("acct_booked_h0_rep")
        swap_out = row.get("acct_booked_h0_cut")
        prod_si = row.get("oa_amt_h0_rep")
        prod_so = row.get("oa_amt_h0_cut")

        mask_str = row.get("acceptance_mask")
        if isinstance(mask_str, str) and mask_str:
            n_on = sum(1 for x in mask_str.split(",") if x.strip() == "1")
            rows.append(
                {
                    "segment": segment,
                    "scenario": scenario,
                    "target_risk_pct": target_risk_pct,
                    "sol_fac": sol,
                    "risk_pct": risk,
                    "production_eur": prod,
                    "policy_type": "N_var_mask",
                    var0_label: None,
                    var1_label: None,
                    "accepted_cells": n_on,
                    "acceptance_mask_preview": mask_str[:200] + ("…" if len(mask_str) > 200 else ""),
                    "swap_in_accounts": swap_in,
                    "swap_out_accounts": swap_out,
                    "swap_in_production_eur": prod_si,
                    "swap_out_production_eur": prod_so,
                }
            )
            continue

        bin_keys = sorted(
            ((float(k), k) for k in row if _is_bin_column_key(k)),
            key=lambda t: t[0],
        )
        if not bin_keys:
            rows.append(
                {
                    "segment": segment,
                    "scenario": scenario,
                    "target_risk_pct": target_risk_pct,
                    "sol_fac": sol,
                    "risk_pct": risk,
                    "production_eur": prod,
                    "policy_type": "summary_only",
                    var0_label: None,
                    var1_label: None,
                    "accepted_cells": None,
                    "acceptance_mask_preview": None,
                    "swap_in_accounts": swap_in,
                    "swap_out_accounts": swap_out,
                    "swap_in_production_eur": prod_si,
                    "swap_out_production_eur": prod_so,
                }
            )
            continue

        for _bin_f, k in bin_keys:
            cutoff = row[k]
            if pd.isna(cutoff):
                continue
            rows.append(
                {
                    "segment": segment,
                    "scenario": scenario,
                    "target_risk_pct": target_risk_pct,
                    "sol_fac": sol,
                    "risk_pct": risk,
                    "production_eur": prod,
                    "policy_type": "2_var_cutoff",
                    var0_label: float(k),
                    var1_label: float(cutoff) if np.isfinite(cutoff) else cutoff,
                    "accepted_cells": None,
                    "acceptance_mask_preview": None,
                    "swap_in_accounts": swap_in,
                    "swap_out_accounts": swap_out,
                    "swap_in_production_eur": prod_si,
                    "swap_out_production_eur": prod_so,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def policy_cutoff_table_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render a compact markdown table for email/committee packs."""
    if df.empty:
        return "_No policy rows generated (empty frontier details)._\n"
    show = df.head(max_rows)
    try:
        return show.to_markdown(index=False)
    except ImportError:
        return "```\n" + show.to_string(index=False) + "\n```\n"


def allocation_constraint_narrative(
    result: AllocationResult,
    constraints: dict[str, SegmentConstraints] | None,
    *,
    global_production_floor: float | None = None,
) -> str:
    """
    Plain-language explanation of limits and which ones bind (if reported by the solver).

    Parameters
    ----------
    result:
        Output of ``GlobalAllocator.optimize`` / ``optimize_exact`` / ``optimize_greedy``.
    constraints:
        Per-segment constraints passed into the allocator (may be empty).
    global_production_floor:
        Global minimum production, if any.
    """
    lines: list[str] = []
    lines.append("## Allocation constraint narrative")
    lines.append("")
    if not getattr(result, "feasible", True):
        note = result.message or "constraints not satisfied"
        lines.append(
            f"- ⚠ **INFEASIBLE:** {note}. No allocation meets the target under the current frontiers/constraints; "
            "the figures below are the closest feasible-effort result, **not** an optimum."
        )
        lines.append("")
    tgt = result.target
    if tgt is not None:
        lines.append(
            f"- **Risk target (input):** {tgt:.4f}% (production-weighted portfolio b2 ceiling used by the MILP/greedy solver)."
        )
    lines.append(
        f"- **Achieved global risk (production-weighted — the optimized ceiling):** {result.global_risk:.4f}% "
        f"| **Global production:** €{result.global_production:,.0f} | **Method:** {result.method or 'unknown'}."
    )
    pbr = getattr(result, "portfolio_bad_rate", None)
    if pbr is not None:
        lines.append(
            f"- **Portfolio bad-rate (exposure-weighted — reference only):** {pbr:.4f}% "
            "(Σbad/Σexposure across segments; the optimizer constrains the production-weighted figure above)."
        )
    if global_production_floor is not None:
        met = result.global_production + 1e-6 >= global_production_floor
        floor_note = "met" if met else "below configured floor — infeasible or check solver"
        lines.append(
            f"- **Global production floor (input):** €{global_production_floor:,.0f} "
            f"({floor_note}; achieved €{result.global_production:,.0f})."
        )

    if constraints:
        lines.append("")
        lines.append("**Configured segment limits** (from `segments.toml` / API):")
        for seg in sorted(constraints):
            sc = constraints[seg]
            parts = []
            if sc.locked_sol_fac is not None:
                parts.append(f"locked to frontier point sol_fac={sc.locked_sol_fac}")
            if sc.min_risk is not None:
                parts.append(f"min risk ≥ {sc.min_risk:.4f}%")
            if sc.max_risk is not None:
                parts.append(f"max risk ≤ {sc.max_risk:.4f}%")
            if sc.min_production is not None:
                parts.append(f"min production ≥ €{sc.min_production:,.0f}")
            if parts:
                lines.append(f"  - **{seg}:** " + "; ".join(parts))

    lines.append("")
    if result.binding_constraints:
        lines.append("**Binding at optimum** (zero slack within numerical tolerance):")
        for label in result.binding_constraints:
            lines.append(f"  - {_explain_binding_label(label, constraints, global_production_floor, result.target)}")
    else:
        lines.append("**Binding constraints:** None within tolerance (MILP slack; greedy uses heuristic flags only).")

    if tgt is not None and result.binding_constraints:
        if "global_risk" not in result.binding_constraints and abs(result.global_risk - tgt) > 0.02:
            lines.append("")
            lines.append(
                f"_Note: Achieved risk {result.global_risk:.4f}% differs from target {tgt:.4f}% — "
                "often because segment floors/ceilings or locks prevent hitting the global cap exactly._"
            )

    lines.append("")
    return "\n".join(lines)


def _explain_binding_label(
    label: str,
    constraints: dict[str, SegmentConstraints] | None,
    global_floor: float | None,
    target: float | None,
) -> str:
    if label == "global_risk" and target is not None:
        return (
            f"`{label}` — Portfolio production-weighted risk is at the limit for target **{target:.4f}%** "
            "(adding production would exceed the cap without changing segment mix)."
        )
    if label == "global_production_floor":
        if global_floor is not None:
            return f"`{label}` — Total production is at the configured global floor **€{global_floor:,.0f}**."
        return f"`{label}` — Total production is at the configured global production floor."
    if label.endswith("_min_risk"):
        seg = label[: -len("_min_risk")]
        v = constraints.get(seg).min_risk if constraints and seg in constraints else None  # type: ignore[union-attr]
        extra = f" (floor **{v:.4f}%**)" if v is not None else ""
        return f"`{label}` — Segment **{seg}** is at its minimum risk constraint{extra}."
    if label.endswith("_max_risk"):
        seg = label[: -len("_max_risk")]
        v = constraints.get(seg).max_risk if constraints and seg in constraints else None  # type: ignore[union-attr]
        extra = f" (ceiling **{v:.4f}%**)" if v is not None else ""
        return f"`{label}` — Segment **{seg}** is at its maximum risk constraint{extra}."
    if label.endswith("_min_production"):
        seg = label[: -len("_min_production")]
        v = constraints.get(seg).min_production if constraints and seg in constraints else None  # type: ignore[union-attr]
        extra = f" (floor **€{v:,.0f}**)" if v is not None else ""
        return f"`{label}` — Segment **{seg}** is at its minimum production constraint{extra}."
    return f"`{label}`"


def write_portfolio_owner_pack(
    result: AllocationResult,
    constraints: dict[str, SegmentConstraints] | None,
    *,
    output_csv: str | Path,
    scenario: str = "base",
    global_production_floor: float | None = None,
) -> tuple[Path, Path]:
    """
    Write ``policy_cutoff_table`` CSV and ``allocation_narrative.md`` beside *output_csv*.

    Returns
    -------
    (policy_table_path, narrative_path)
    """
    output_csv = Path(output_csv)
    out_dir = output_csv.parent
    stem = output_csv.stem

    policy_df = policy_cutoff_table_from_allocation(
        result.segment_details,
        scenario=scenario,
        target_risk_pct=result.target,
    )
    policy_path = out_dir / f"{stem}_policy_cutoff_table.csv"
    policy_df.to_csv(policy_path, index=False)

    narrative = allocation_constraint_narrative(
        result,
        constraints,
        global_production_floor=global_production_floor,
    )
    narrative_path = out_dir / f"{stem}_allocation_narrative.md"
    narrative_path.write_text(narrative + "\n", encoding="utf-8")

    return policy_path, narrative_path


def what_if_dataframe_row(
    result: AllocationResult,
    target_risk_pct: float,
    *,
    feasible: bool = True,
) -> dict:
    """One row for the what-if comparison table."""
    return {
        "target_risk_pct": target_risk_pct,
        "achieved_global_risk_pct": result.global_risk,
        "global_production_eur": result.global_production,
        "method": result.method,
        "feasible": feasible,
        "binding_constraints": "; ".join(result.binding_constraints) if result.binding_constraints else "",
        "n_segments": len(result.allocations),
    }

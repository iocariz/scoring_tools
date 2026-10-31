"""Shared data-loading helpers for the dashboards (R2 todo #64).

Before this module existed, ``dashboard.py``, ``gradio_dashboard.py``, and
``interactive_allocator.py`` each redefined near-identical versions of
``get_data_dir`` / ``get_images_dir`` / ``get_available_segments`` /
``get_scenarios`` / ``_resolve_supersegment`` / ``parse_coefficients``. The
audit flagged ~15–20 KB of drift-prone duplication.

Design notes:
  - Every function takes an explicit ``output_base: Path`` parameter rather
    than reading a module-level constant. Each dashboard keeps its own
    ``OUTPUT_BASE`` (mutated at startup from CLI args) and passes it in.
  - Functions are pure: no logging, no I/O other than filesystem reads,
    no side effects. Makes them trivial to unit-test.
  - Static-URL builders (``/static/...``) stay in dashboard.py because
    they're Flask-routing-specific; this module only deals with filesystem
    paths.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Scenario ordering used by get_scenarios() for deterministic sort.
# Named constant so both dashboards can share it.
SCENARIO_ORDER: dict[str, int] = {"pessimistic": 0, "base": 1, "optimistic": 2}


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_data_dir(output_base: Path, segment: str | None = None) -> Path:
    """Return the ``data/`` directory for *segment* (or the root in single-run mode)."""
    if segment:
        return output_base / segment / "data"
    if (output_base / "data").exists():
        return output_base / "data"
    return Path("data")


def get_images_dir(output_base: Path, segment: str | None = None) -> Path:
    """Return the ``images/`` directory for *segment* (or the root in single-run mode)."""
    if segment:
        return output_base / segment / "images"
    if (output_base / "images").exists():
        return output_base / "images"
    return Path("images")


def get_models_dir(output_base: Path, segment: str | None = None) -> Path:
    """Return the ``models/`` directory for *segment*, with supersegment fallback.

    If the segment has no local models dir but references a supersegment
    that does, the supersegment's models dir is returned instead. This
    matches how ``run_batch.py`` saves supersegment-shared models under
    ``output/_supersegment_<ss>/models/``.
    """
    if segment:
        direct = output_base / segment / "models"
        if direct.exists() and any(direct.iterdir()):
            return direct
        superseg = resolve_supersegment(segment, output_base=output_base)
        if superseg:
            ss_models = output_base / f"_supersegment_{superseg}" / "models"
            if ss_models.exists():
                return ss_models
        return direct
    if (output_base / "models").exists():
        return output_base / "models"
    return Path("models")


# ---------------------------------------------------------------------------
# Segment / scenario discovery
# ---------------------------------------------------------------------------


def get_available_segments(output_base: Path) -> list[str]:
    """Enumerate directories under *output_base* that look like valid segments.

    A directory qualifies if: it is not a dot/underscore-prefixed hidden
    folder, and it contains a ``data/`` subdir. Sorted alphabetically.
    """
    if not output_base.exists():
        return []
    return sorted(
        d.name
        for d in output_base.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and (d / "data").exists()
    )


def get_scenarios(output_base: Path, segment: str | None = None) -> list[str]:
    """Parse scenario names from the CSV files produced by the pipeline.

    Names are either canonical (``pessimistic`` / ``base`` / ``optimistic``)
    or the legacy numeric format (``1.2`` etc.). Returned list is sorted
    by :data:`SCENARIO_ORDER` (pessimistic, base, optimistic) with numeric
    scenarios appended alphabetically after.
    """
    data_dir = get_data_dir(output_base, segment)
    if not data_dir.exists():
        return []

    scenarios: list[str] = []
    for f in data_dir.glob("risk_production_summary_table_*.csv"):
        if "_mr" in f.name:
            # MR files are secondary; discover from the main files only.
            continue
        m = re.search(r"risk_production_summary_table_(pessimistic|base|optimistic)\.csv", f.name)
        if m:
            scenarios.append(m.group(1))
            continue
        m = re.search(r"risk_production_summary_table_(\d+\.\d+)\.csv", f.name)
        if m:
            scenarios.append(m.group(1))

    # Unsuffixed base file fallback.
    if (data_dir / "risk_production_summary_table.csv").exists() and "base" not in scenarios:
        scenarios.append("base")

    return sorted(scenarios, key=lambda x: (SCENARIO_ORDER.get(x, 99), x))


def get_scenario_paths(
    output_base: Path,
    scenario: str,
    segment: str | None = None,
) -> dict[str, Path]:
    """Return the filesystem paths for a scenario's four main CSV artifacts.

    Static-URL path resolution (``/static/<segment>/risk_production...html``)
    is dashboard-specific and NOT included here — that belongs in
    ``dashboard.py``. This module stays framework-agnostic.

    For the base scenario, falls back to the unsuffixed filename when the
    suffixed version is missing.
    """
    data_dir = get_data_dir(output_base, segment)
    suffix = f"_{scenario}"
    paths = {
        "main_csv": data_dir / f"risk_production_summary_table{suffix}.csv",
        "mr_csv": data_dir / f"risk_production_summary_table_mr{suffix}.csv",
        "cutoffs": data_dir / f"optimal_solution{suffix}.csv",
        "summary_data": data_dir / f"data_summary_desagregado{suffix}.csv",
    }
    if scenario == "base":
        fallbacks = {
            "main_csv": data_dir / "risk_production_summary_table.csv",
            "mr_csv": data_dir / "risk_production_summary_table_mr.csv",
            "cutoffs": data_dir / "optimal_solution.csv",
            "summary_data": data_dir / "data_summary_desagregado.csv",
        }
        for key in paths:
            if not paths[key].exists() and fallbacks[key].exists():
                paths[key] = fallbacks[key]
    return paths


# ---------------------------------------------------------------------------
# Supersegment resolution
# ---------------------------------------------------------------------------


def resolve_supersegment(segment: str | None, output_base: Path | None = None) -> str | None:
    """Look up the modelling (or plain) supersegment name for *segment*.

    Searches both the working-directory ``segments.toml`` and (if
    *output_base* is provided) ``<output_base>/segments.toml`` — the
    latter is useful when dashboards open a previously-generated output
    tree that has its own snapshot of segments.toml.

    Returns ``None`` if the segment is unknown or no supersegment is
    configured. Resolution order matches
    :func:`src.core.utils.resolve_modelling_supersegment`:
    ``modelling_supersegment`` > ``supersegment`` > ``None``.
    """
    if not segment:
        return None

    candidate_paths: list[Path] = [Path("segments.toml")]
    if output_base is not None:
        candidate_paths.append(output_base / "segments.toml")

    for toml_path in candidate_paths:
        if not toml_path.exists():
            continue
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            seg_config = data.get("segments", {})
            cfg = seg_config.get(segment)
            if cfg is not None:
                return cfg.get("modelling_supersegment") or cfg.get("supersegment")
        except (OSError, tomllib.TOMLDecodeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Model artefact parsing
# ---------------------------------------------------------------------------


def parse_coefficients(model_dir: str | Path) -> list[dict[str, Any]]:
    """Parse ``model_summary.txt`` under *model_dir* for feature coefficients.

    Returns a list of ``{"feature": str, "coefficient": float}`` dicts in
    the order they appear in the summary. An empty list is returned if
    the summary file is missing, unreadable, or has no coefficient section.

    The ``MODEL COEFFICIENTS`` section is delimited by
    ``---``-rules and terminated by the next all-caps header.
    """
    summary_path = Path(model_dir) / "model_summary.txt"
    if not summary_path.exists():
        return []

    coefficients: list[dict[str, Any]] = []
    in_section = False
    try:
        text = summary_path.read_text()
    except OSError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "MODEL COEFFICIENTS" in line:
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("---") or not line:
            continue
        if line.isupper() and ":" not in line:
            # Next section header — stop.
            break
        if ":" not in line:
            continue
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        try:
            value = float(parts[1].strip())
        except ValueError:
            continue
        coefficients.append({"feature": name, "coefficient": value})
    return coefficients


def calculate_metrics_from_custom_cuts(
    data: pd.DataFrame,
    cut_map: dict,
    var0_col: str,
    var1_col: str,
    inv_vars: list[str] | None = None,
    multiplier: float = 7.0,
) -> dict[str, Any]:
    """
    Calculate metrics for custom cutoff configuration.

    Args:
        data: Summary data DataFrame
        cut_map: Dictionary mapping var0 bins to var1 cutoff values
        var0_col: First variable column name
        var1_col: Second variable column name
        inv_vars: List of variables with inverted risk ordering
        multiplier: Risk multiplier (default 7.0)

    Returns:
        Dictionary with calculated metrics
    """
    if inv_vars is None:
        inv_vars = []

    df = data.copy()

    # Normalize cut_map keys to float to match data column types
    cut_map_norm = {float(k): v for k, v in cut_map.items()}

    # Apply cuts
    df["cut_limit"] = df[var0_col].astype(float).map(cut_map_norm)

    # Respect the optimization direction (RMSE/error is typically an inv_var where higher score = lower risk)
    if var1_col in inv_vars:
        df["passes_cut"] = df[var1_col] >= df["cut_limit"]
    else:
        df["passes_cut"] = df[var1_col] <= df["cut_limit"]

    def calc_metrics(subset, suffix):
        prod_col = f"oa_amt_h0{suffix}"
        risk_num_col = f"todu_30ever_h6{suffix}"
        risk_den_col = f"todu_amt_pile_h6{suffix}"

        if prod_col not in subset.columns:
            return 0, 0, 0, 0

        prod = subset[prod_col].sum()
        risk_num = subset[risk_num_col].sum() if risk_num_col in subset.columns else 0
        risk_den = subset[risk_den_col].sum() if risk_den_col in subset.columns else 0
        b2_ever = (risk_num / risk_den * multiplier) if risk_den > 0 else 0.0
        return prod, b2_ever, risk_num, risk_den

    # Actual (All Booked)
    actual_prod, actual_risk, actual_rn, actual_rd = calc_metrics(df, "_boo")

    # Swap-in (Repesca that passes)
    swap_in_df = df[df["passes_cut"]]
    si_prod, si_risk, si_rn, si_rd = calc_metrics(swap_in_df, "_rep")

    # Swap-out (Booked that fails)
    swap_out_df = df[~df["passes_cut"]]
    so_prod, so_risk, so_rn, so_rd = calc_metrics(swap_out_df, "_boo")

    # Optimum
    opt_prod = (actual_prod - so_prod) + si_prod
    opt_rn = (actual_rn - so_rn) + si_rn
    opt_rd = (actual_rd - so_rd) + si_rd
    opt_risk = (opt_rn / opt_rd * multiplier) if opt_rd > 0 else 0.0

    return {
        "actual_prod": actual_prod,
        "actual_risk": actual_risk,
        "swap_in_prod": si_prod,
        "swap_in_risk": si_risk,
        "swap_out_prod": so_prod,
        "swap_out_risk": so_risk,
        "opt_prod": opt_prod,
        "opt_risk": opt_risk,
        "opt_prod_pct": opt_prod / actual_prod if actual_prod > 0 else 0,
    }


def calculate_metrics_from_mask(
    data: pd.DataFrame,
    mask: list[int] | np.ndarray,
    variables: list[str],
    multiplier: float = 7.0,
) -> dict[str, Any]:
    """Calculate metrics from an N-dimensional acceptance mask.

    Thin wrapper around CellGrid + evaluate_solution, returning the same dict
    shape as calculate_metrics_from_custom_cuts().
    """
    from src.optimize.optimization_utils import CellGrid, evaluate_solution

    mask_arr = np.asarray(mask, dtype=int)
    grid = CellGrid.from_summary(data, variables)
    kpis = evaluate_solution(mask_arr, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier)

    actual_prod = kpis.get("oa_amt_h0_boo", 0.0) + kpis.get("oa_amt_h0_cut", 0.0)
    opt_prod = kpis.get("oa_amt_h0", 0.0)

    return {
        "actual_prod": actual_prod,
        "actual_risk": kpis.get("b2_ever_h6_boo", 0.0),
        "swap_in_prod": kpis.get("oa_amt_h0_rep", 0.0),
        "swap_in_risk": kpis.get("b2_ever_h6_rep", 0.0),
        "swap_out_prod": kpis.get("oa_amt_h0_cut", 0.0),
        "swap_out_risk": kpis.get("b2_ever_h6_cut", 0.0),
        "opt_prod": opt_prod,
        "opt_risk": kpis.get("b2_ever_h6", 0.0),
        "opt_prod_pct": opt_prod / actual_prod if actual_prod > 0 else 0,
    }

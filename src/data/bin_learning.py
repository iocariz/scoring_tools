"""Bin-edge learning shared across segments (batch mode): global edges on the full demand population and
per-supersegment edges, so every segment of a group bins on the same grid.
"""

from typing import Any

import pandas as pd
from loguru import logger


def learn_global_bin_edges(data: pd.DataFrame, base_config: dict[str, Any]) -> dict[str, list[float]]:
    """Learn bin edges on the full dataset (all segments) for consistency.

    When bins use ``max_bins`` instead of fixed ``bin_edges``, edges are learned
    (unsupervised quantiles) from data.  In batch mode we want a single set of
    edges shared across all segments, so we learn them here on the full demand
    population (all statuses, date-filtered to the observation window) before
    any segment run.  ``method = "optimization"`` is deprecated (target leakage)
    and falls back to quantile with a warning.

    Parameters
    ----------
    data : pd.DataFrame
        Full pre-loaded and standardized dataset (all segments).
    base_config : dict
        Base ``[preprocessing]`` configuration dictionary.

    Returns
    -------
    dict[str, list[float]]
        Mapping of variable name → learned bin edges.  Empty if no bins
        require learning.
    """
    from src.data.preprocess_improved import filter_by_date, learn_quantile_bins

    bins_config = base_config.get("bins", {})
    if not bins_config:
        return {}

    # Basic quality filters (NO segment filter).  Learn on the DEMAND population
    # (all statuses) the bins are applied to — not booked-only, which is a
    # selected, risk-truncated subset.  Date-filter to the observation window to
    # avoid look-ahead bias.
    mask = (data["fuera_norma"] == "n") & (data["fraud_flag"] == "n") & (data["nature_holder"] != "legal")
    data_demand = data[mask]
    date_ini = base_config.get("date_ini_book_obs")
    date_fin = base_config.get("date_fin_book_obs")
    if date_ini and date_fin:
        data_demand = filter_by_date(data_demand, "mis_date", date_ini, date_fin)
    logger.info(f"Global bin learning: {len(data_demand):,} demand records across all segments")

    global_edges: dict[str, list[float]] = {}
    for var_name, bc_raw in bins_config.items():
        has_edges = bc_raw.get("bin_edges") and len(bc_raw["bin_edges"]) >= 2
        if has_edges or bc_raw.get("max_bins") is None:
            continue  # Fixed edges or no learning needed

        source_col = bc_raw["source_col"]
        method = bc_raw.get("method", "quantile")
        max_bins = bc_raw["max_bins"]

        if method == "optimization":
            logger.warning(
                f"Bin method 'optimization' for '{var_name}' is deprecated (target leakage); "
                f"falling back to quantile splits."
            )
        edges = learn_quantile_bins(data_demand, source_col=source_col, max_bins=max_bins)

        global_edges[var_name] = edges
        logger.info(f"Global bin edges for '{var_name}': {edges}")

    return global_edges


def learn_supersegment_bin_edges(
    data: pd.DataFrame | None,
    base_config: dict[str, Any],
    supersegments: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Learn bin edges per reporting supersegment population.

    For each reporting supersegment, filters the demand data (all statuses,
    date-filtered to the observation window) to only the segments in that
    supersegment and learns bin edges from that subpopulation.  This allows
    different supersegments to have bin splits tuned to their own data
    distribution.

    Fixed ``bin_edges`` declared on a supersegment are **configuration, not
    data-derived** — they are extracted even when ``data`` is ``None`` (preload
    failed), so a transient load hiccup can never silently swap the configured
    edges for per-segment quantile learning (audit #27). Only the
    ``learn_own_bin_edges`` path requires data.

    Parameters
    ----------
    data : pd.DataFrame | None
        Full pre-loaded and standardized dataset (all segments), or ``None``
        when preloading failed (fixed edges are still extracted).
    base_config : dict
        Base ``[preprocessing]`` configuration dictionary.
    supersegments : dict
        Mapping of supersegment name → config with ``segment_filters``.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Mapping of supersegment name → {variable name → learned bin edges}.
    """
    from src.data.preprocess_improved import filter_by_date, learn_quantile_bins

    bins_config = base_config.get("bins", {})
    if not bins_config:
        return {}

    # Only learn for bins that need it (no fixed bin_edges, have max_bins).
    # NOTE: fixed supersegment edges below are extracted even when nothing is
    # learnable — they are configuration and must always flow through.
    learnable = {
        var_name: bc_raw
        for var_name, bc_raw in bins_config.items()
        if not (bc_raw.get("bin_edges") and len(bc_raw["bin_edges"]) >= 2) and bc_raw.get("max_bins") is not None
    }

    # Basic quality filters (no segment filter yet).  Learn on the DEMAND
    # population (all statuses) the bins are applied to — not booked-only.
    quality_mask = None
    if data is not None:
        quality_mask = (data["fuera_norma"] == "n") & (data["fraud_flag"] == "n") & (data["nature_holder"] != "legal")
    date_ini = base_config.get("date_ini_book_obs")
    date_fin = base_config.get("date_fin_book_obs")

    result: dict[str, dict[str, list[float]]] = {}
    for ss_name, ss_config in supersegments.items():
        # Check for fixed bin_edges first — these take priority over learning.
        fixed_edges = ss_config.get("bin_edges", {})
        if fixed_edges:
            ss_edges: dict[str, list[float]] = {}
            for var_name, edges in fixed_edges.items():
                if isinstance(edges, list) and len(edges) >= 2:
                    ss_edges[var_name] = edges
                    logger.info(f"Supersegment '{ss_name}' using fixed bin edges for '{var_name}': {edges}")
            if ss_edges:
                result[ss_name] = ss_edges
            continue  # Fixed edges provided — skip learning for this supersegment

        if not ss_config.get("learn_own_bin_edges", False):
            continue  # Use global edges (default)

        if data is None:
            logger.error(
                f"Supersegment '{ss_name}' requests learn_own_bin_edges but data preloading failed — "
                "cannot learn shared edges (the batch gate halts the run in this case)."
            )
            continue

        segment_filters = ss_config.get("segment_filters", [])
        if not segment_filters:
            continue

        # Filter to this supersegment's population. Use EXACT set membership (isin), not
        # str.contains: a substring match let a filter like "premium" absorb "no_premium_*" (or any
        # segment whose full name contains another's), silently pooling the wrong rows into the
        # learned edges (#39). segment_cut_off holds the full segment path, so isin is exact.
        ss_mask = quality_mask & data["segment_cut_off"].isin(segment_filters)
        ss_demand = data[ss_mask]
        if date_ini and date_fin:
            ss_demand = filter_by_date(ss_demand, "mis_date", date_ini, date_fin)

        if len(ss_demand) == 0:
            logger.warning(f"Supersegment '{ss_name}': no demand records for bin learning")
            continue

        ss_edges = {}
        for var_name, bc_raw in learnable.items():
            source_col = bc_raw["source_col"]
            method = bc_raw.get("method", "quantile")
            max_bins = bc_raw["max_bins"]

            try:
                if method == "optimization":
                    logger.warning(
                        f"Bin method 'optimization' for '{var_name}' is deprecated (target leakage); "
                        f"falling back to quantile splits."
                    )
                edges = learn_quantile_bins(ss_demand, source_col=source_col, max_bins=max_bins)
                ss_edges[var_name] = edges
                logger.info(f"Supersegment '{ss_name}' bin edges for '{var_name}': {edges}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Supersegment '{ss_name}' bin learning failed for '{var_name}': {e}")

        if ss_edges:
            result[ss_name] = ss_edges

    return result

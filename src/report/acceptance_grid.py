"""Renderer-independent acceptance-grid building, shared by the HTML report and the Excel workbook.

Cells are three-state (``accepted`` = 1 / 0 / NaN for unobserved phantom cells). The grid is a pivot of
``accepted`` over the first two grid variables (columns, rows), one sub-grid per value of the remaining
variables; a 1-variable grid is a strip.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

MAX_SLICES = 12


def sort_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Sort a pivot's index and columns numerically where possible."""
    try:
        pivot = pivot.sort_index(key=lambda x: pd.to_numeric(x, errors="coerce"))
        pivot = pivot[sorted(pivot.columns, key=lambda x: pd.to_numeric(x, errors="coerce"))]
    except (TypeError, ValueError):
        pass
    return pivot


def acceptance_pivot(df: pd.DataFrame, col_var: str, row_var: str) -> pd.DataFrame:
    """``accepted`` pivoted to rows=``row_var`` x columns=``col_var``, numerically sorted."""
    return sort_pivot(df.pivot_table(index=row_var, columns=col_var, values="accepted", aggfunc="first"))


def slice_groups(df: pd.DataFrame, slice_vars: list[str], context: str) -> list[tuple[str | None, pd.DataFrame]]:
    """Split ``df`` by the 3rd+ grid variables into labelled sub-grids (at most ``MAX_SLICES``).

    ``dropna=False`` keeps a NaN slice value as a visible group instead of silently dropping its cells;
    truncation beyond ``MAX_SLICES`` is logged. Without slice variables the whole frame is one unlabeled group.
    """
    if not slice_vars:
        return [(None, df)]
    groups = list(df.groupby(slice_vars, sort=True, dropna=False))
    if len(groups) > MAX_SLICES:
        logger.warning(f"Acceptance grid: {context} has {len(groups)} slices; showing the first {MAX_SLICES}.")
    items = []
    for key, slice_df in groups[:MAX_SLICES]:
        keys = key if isinstance(key, tuple) else (key,)
        items.append((", ".join(f"{v}={k}" for v, k in zip(slice_vars, keys)), slice_df))
    return items


def strip_counts(accepted) -> tuple[int, int, float]:
    """(accepted, observed, % accepted over observed) — NaN (unobserved) counts as neither."""
    acc = pd.Series(accepted)
    n_accepted = int(acc.sum())
    n_observed = int(acc.notna().sum())
    return n_accepted, n_observed, (100 * n_accepted / n_observed if n_observed > 0 else 0.0)

"""Shared acceptance-grid building (HTML report + Excel workbook)."""

import numpy as np
import pandas as pd
from loguru import logger

from src.report.acceptance_grid import MAX_SLICES, acceptance_pivot, slice_groups, strip_counts


def test_strip_counts_treats_nan_as_unobserved():
    assert strip_counts([1, 0, np.nan, 1]) == (2, 3, 100 * 2 / 3)
    assert strip_counts([np.nan, np.nan]) == (0, 0, 0.0)


def test_acceptance_pivot_sorts_numerically():
    df = pd.DataFrame({"a": [10, 2, 10, 2], "b": [1, 1, 3, 3], "accepted": [1, 0, 1, 1]})
    pivot = acceptance_pivot(df, col_var="a", row_var="b")
    assert list(pivot.columns) == [2, 10]
    assert list(pivot.index) == [1, 3]
    assert pivot.loc[1, 2] == 0


def test_slice_groups_keeps_nan_slice_and_labels():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1], "inc": [1.0, 2.0, np.nan], "accepted": [1, 0, 1]})
    groups = slice_groups(df, ["inc"], "test")
    assert [label for label, _ in groups] == ["inc=1.0", "inc=2.0", "inc=nan"]


def test_slice_groups_without_slice_vars_is_one_group():
    df = pd.DataFrame({"a": [1], "b": [1], "accepted": [1]})
    assert slice_groups(df, [], "test") == [(None, df)]


def test_slice_groups_caps_and_warns():
    n = MAX_SLICES + 3
    df = pd.DataFrame({"a": [1] * n, "b": [1] * n, "inc": range(n), "accepted": [1] * n})
    msgs = []
    hid = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        groups = slice_groups(df, ["inc"], "segment 's'")
    finally:
        logger.remove(hid)
    assert len(groups) == MAX_SLICES
    assert any("showing the first" in m for m in msgs)

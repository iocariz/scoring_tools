"""Frontier invariants of the Cutoff Grids writer, enforced via the QA checker script.

Draws synthetic acceptance pivots through the real ``_write_single_pivot_grid`` and
validates the rendered borders with ``scripts/check_cutoff_frontier.py``: closed
frontier chains, no gaps across grey cells (#219), a single staircase-shaped accept
region, and the one-class behaviour (#220 — all-accept grids keep one outer frontier
instead of boxing interior grey holes).
"""

import importlib.util
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.consolidation import _write_single_pivot_grid

_SPEC = importlib.util.spec_from_file_location(
    "check_cutoff_frontier", Path(__file__).parents[1] / "scripts" / "check_cutoff_frontier.py"
)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

nan = float("nan")


def _render_and_check(acc: np.ndarray):
    """Write *acc* through the real grid writer, re-read it with the checker."""
    pivot = pd.DataFrame(acc, index=range(1, acc.shape[0] + 1), columns=range(1, acc.shape[1] + 1))
    ws = openpyxl.Workbook().active
    _write_single_pivot_grid(ws, pivot, "efx", "octroi", start_row=1)
    grids = checker.find_grids(ws)
    assert len(grids) == 1
    r0, c0, _, nrows, ncols = grids[0]
    assert (nrows, ncols) == acc.shape
    vals, frontier = checker.read_grid(ws, r0, c0, nrows, ncols)
    return vals, frontier, checker.check_grid(vals, frontier)


def test_mixed_grid_staircase_continuous_across_grey():
    """#219: grey cells on/inside the boundary don't fragment the staircase."""
    acc = np.array(
        [
            [0, 0, 0, nan, 0, 1],
            [0, 0, nan, 0, 1, 1],
            [0, nan, 0, 1, 1, 1],
            [0, 0, 1, 1, nan, 1],
            [nan, 1, 1, 1, 1, 1],
        ]
    )
    vals, frontier, problems = _render_and_check(acc)
    assert problems == []
    assert any(f.any() for f in frontier.values())  # a frontier was actually drawn


def test_all_accept_grid_single_outer_frontier():
    """#220: with no rejected cells, interior grey holes are not boxed — the frontier
    is exactly the grid perimeter."""
    acc = np.array(
        [
            [nan, 1, nan, 1],
            [1, nan, 1, 1],
            [1, 1, 1, nan],
        ]
    )
    vals, frontier, problems = _render_and_check(acc)
    assert problems == []
    nrows, ncols = acc.shape
    for i in range(nrows):
        for j in range(ncols):
            expected = {
                "top": i == 0,
                "bottom": i == nrows - 1,
                "left": j == 0,
                "right": j == ncols - 1,
            }
            for side, want in expected.items():
                assert frontier[side][i, j] == want, f"cell ({i},{j}) side {side}"


def test_all_reject_grid_draws_no_frontier():
    """#220: with no accepted cells there is no boundary at all."""
    acc = np.array([[0, nan, 0], [0, 0, nan]])
    _, frontier, problems = _render_and_check(acc)
    assert problems == []
    assert not any(f.any() for f in frontier.values())


@pytest.mark.parametrize("break_side", ["left", "top"])
def test_checker_flags_broken_frontier(break_side):
    """The checker itself must catch a deliberately broken border (guards against the
    checker silently passing everything). Knock out an INTERIOR frontier edge — one
    facing an in-grid R cell — so the accept flood-fill leaks; perimeter edges are
    legitimately allowed to terminate chains and aren't probed here."""
    acc = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    vals, frontier, problems = _render_and_check(acc)
    assert problems == []
    di, dj = (-1, 0) if break_side == "top" else (0, -1)
    candidates = [
        (i, j)
        for i, j in np.argwhere(frontier[break_side]).tolist()
        if i + di >= 0 and j + dj >= 0 and vals[i + di, j + dj] == "R"
    ]
    assert candidates, f"expected an interior {break_side} frontier edge facing an R cell"
    i, j = candidates[0]
    frontier[break_side][i, j] = False
    assert checker.check_grid(vals, frontier), "checker failed to flag a broken frontier"


def test_check_workbook_end_to_end(tmp_path):
    """check_workbook finds multiple grids on a sheet and counts failures."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Cutoff Grids"
    good = pd.DataFrame([[0.0, 1.0], [1.0, 1.0]], index=[1, 2], columns=[1, 2])
    _write_single_pivot_grid(ws, good, "efx", "octroi", start_row=1)
    _write_single_pivot_grid(ws, good, "efx", "octroi", start_row=6)
    path = tmp_path / "wb.xlsx"
    wb.save(path)
    assert checker.check_workbook(str(path), verbose=False) == 0

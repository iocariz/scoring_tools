"""Verify the Cutoff Grids frontier borders are continuous staircases.

A QA helper for the consolidated workbook (companion to
``preview_exec_summary_kpis.py``): it parses every A/R/— acceptance grid on the
"Cutoff Grids" sheet, reconstructs the thick-navy frontier edges cell by cell,
and checks the invariants the grid writer is supposed to guarantee (#215/#219/#220):

1. **No gaps** — flood-filling from the accepted (``A``) cells without crossing a
   frontier edge must never reach a rejected (``R``) cell.
2. **Continuity** — frontier edges form closed chains: every interior lattice
   vertex touches an even number of frontier edges (a dangling edge = a visible
   break in the staircase). Chains may terminate on the grid boundary.
3. **Staircase shape** — the accept side enclosed by the frontier is a monotone
   region: per row one contiguous run anchored at a consistent grid edge, with
   run lengths monotone across rows. Grey (``—``) cells inside the accept region
   count as accept-side (border drawing imputes them), so holes fail loudly.

Single-variable runs render a 1-D strip instead (``_write_acceptance_strip_1d``),
which draws no frontier at all — with one score axis the boundary is a single
threshold, so there is nothing to outline. Those are checked on values instead:
acceptance may flip at most once along the axis. They are still *counted*, and a
run that recognises no grids at all exits 1 rather than reporting a vacuous pass.

Usage:
    uv run python scripts/check_cutoff_frontier.py
    uv run python scripts/check_cutoff_frontier.py path/to.xlsx --sheet "Cutoff Grids"

Exit code 0 when every grid passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import openpyxl

# _SIDE_FRONTIER in src/consolidation.py is thick navy; the normal cell
# separator (_SIDE_GRID / _BORDER_GRID) is medium white.
FRONTIER_STYLE = "thick"
CORNER_SEP = " \\ "  # corner label is "<row_var> \\ <col_var>"
STRIP_ROW_LABEL = "Status"  # single-variable strips label their one data row "Status"


def _is_corner(value) -> bool:
    return isinstance(value, str) and CORNER_SEP in value


def _is_strip_anchor(ws, r: int, c: int, value) -> bool:
    """A single-variable strip anchor (``_write_acceptance_strip_1d``).

    A one-score-axis run has no "row_var \\ col_var" corner: the anchor is the bare
    variable name with the bin headers to its right and the one ``Status`` row below.
    """
    if not isinstance(value, str) or _is_corner(value) or not value.strip():
        return False
    below = ws.cell(row=r + 1, column=c).value
    return isinstance(below, str) and below.strip() == STRIP_ROW_LABEL


def find_grids(ws) -> list[tuple[int, int, str, int, int]]:
    """Locate every pivot grid on the sheet.

    Returns (corner_row, corner_col, label, nrows, ncols) per grid; the corner
    cell holds the "row_var \\ col_var" label with headers to its right and below.
    Single-variable strips are included as 1 x ncols grids — their label is the bare
    variable name (no CORNER_SEP), which is how the caller tells the two apart.
    """
    grids = []
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=r, column=c).value
            if not (_is_corner(v) or _is_strip_anchor(ws, r, c, v)):
                continue
            ncols = 0
            while (h := ws.cell(row=r, column=c + 1 + ncols).value) is not None and not _is_corner(h):
                ncols += 1
            nrows = 0
            while (h := ws.cell(row=r + 1 + nrows, column=c).value) is not None and not _is_corner(h):
                nrows += 1
            if nrows and ncols:
                grids.append((r, c, v, nrows, ncols))
    return grids


def read_grid(ws, r0: int, c0: int, nrows: int, ncols: int):
    """Read cell values and per-side frontier flags for the grid at corner (r0, c0).

    Returns (vals, frontier) where vals is an object array of "A"/"R"/"—" and
    frontier is a dict of four boolean arrays keyed "top"/"bottom"/"left"/"right".
    """
    vals = np.full((nrows, ncols), "", dtype=object)
    frontier = {side: np.zeros((nrows, ncols), bool) for side in ("top", "bottom", "left", "right")}
    for i in range(nrows):
        for j in range(ncols):
            cell = ws.cell(row=r0 + 1 + i, column=c0 + 1 + j)
            vals[i, j] = cell.value
            b = cell.border
            for side in frontier:
                s = getattr(b, side)
                frontier[side][i, j] = s is not None and s.style == FRONTIER_STYLE
    return vals, frontier


def _accept_side(vals, frontier) -> np.ndarray:
    """Flood-fill from the A cells without crossing frontier edges → accept-side mask."""
    nrows, ncols = vals.shape
    acc = np.zeros((nrows, ncols), bool)
    stack = list(zip(*np.where(vals == "A"), strict=True))
    seen = set(stack)
    while stack:
        i, j = stack.pop()
        acc[i, j] = True
        for di, dj, blocked in (
            (-1, 0, frontier["top"][i, j]),
            (1, 0, frontier["bottom"][i, j]),
            (0, -1, frontier["left"][i, j]),
            (0, 1, frontier["right"][i, j]),
        ):
            ni, nj = i + di, j + dj
            if 0 <= ni < nrows and 0 <= nj < ncols and (ni, nj) not in seen and not blocked:
                seen.add((ni, nj))
                stack.append((ni, nj))
    return acc


def check_grid(vals, frontier) -> list[str]:
    """Run the gap / continuity / staircase checks on one grid; returns problems."""
    nrows, ncols = vals.shape
    has_frontier = any(f.any() for f in frontier.values())
    if not (vals == "A").any() or not has_frontier:
        # nothing accepted (or nothing drawn, e.g. all-reject) → no boundary to validate
        return []
    problems: list[str] = []
    acc = _accept_side(vals, frontier)

    # 1. gaps: the fill must never leak onto a rejected cell
    leaked = [(i, j) for i, j in zip(*np.where(acc), strict=True) if vals[i, j] == "R"]
    if leaked:
        problems.append(f"frontier GAP: flood-fill from A reaches R at {leaked[:5]}")

    # 2. continuity: frontier edges as segments on the (nrows+1)x(ncols+1) vertex lattice
    edges = set()
    for i in range(nrows):
        for j in range(ncols):
            if frontier["top"][i, j]:
                edges.add(((i, j), (i, j + 1)))
            if frontier["bottom"][i, j]:
                edges.add(((i + 1, j), (i + 1, j + 1)))
            if frontier["left"][i, j]:
                edges.add(((i, j), (i + 1, j)))
            if frontier["right"][i, j]:
                edges.add(((i, j + 1), (i + 1, j + 1)))
    deg: dict[tuple[int, int], int] = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    dangling = [v for v, d in deg.items() if d % 2 and v[0] not in (0, nrows) and v[1] not in (0, ncols)]
    if dangling:
        problems.append(f"frontier DISCONTINUOUS: odd-degree interior vertices {dangling[:5]}")

    # 3. staircase shape of the accept side
    anchor = None  # which grid edge the accept runs hug: "left" or "right"
    lens = []
    for i in range(nrows):
        idx = np.where(acc[i])[0]
        lens.append(int(idx.size))
        if idx.size == 0:
            continue
        if idx.size != idx[-1] - idx[0] + 1:
            problems.append(f"row {i}: accept cells not contiguous ({idx.tolist()})")
            continue
        a = "left" if idx[0] == 0 else ("right" if idx[-1] == ncols - 1 else None)
        if a is None:
            problems.append(f"row {i}: accept run {idx[0]}..{idx[-1]} anchored at neither grid edge")
        elif idx.size < ncols:  # full rows touch both edges and constrain nothing
            if anchor is None:
                anchor = a
            elif a != anchor:
                problems.append(f"row {i}: anchor flips {anchor}->{a}")
    diffs = np.diff(lens)
    if not ((diffs >= 0).all() or (diffs <= 0).all()):
        problems.append(f"accept run lengths not monotone across rows: {lens}")
    return problems


def check_strip(vals) -> list[str]:
    """Monotonicity check for a single-variable strip; returns problems.

    A 1-D layout draws no thick frontier: with one score axis the boundary is a single
    threshold, so there is no staircase to outline and the border-based checks above do
    not apply. What must still hold is the monotone accept region the optimizer
    guarantees — reading along the score axis and ignoring unobserved cells, acceptance
    may flip at most once (R..RA..A or A..AR..R). Two flips mean an accepted bin sits on
    the far side of a rejected one.
    """
    seq = [v for v in np.asarray(vals).ravel().tolist() if v in ("A", "R")]
    if "A" not in seq:
        return []
    flips = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
    if len(flips) > 1:
        return [f"accept region is not a single run: {len(flips)} flips at bin index {flips[:5]} in {''.join(seq)}"]
    return []


def check_workbook(path: str, sheet: str = "Cutoff Grids", verbose: bool = True) -> int:
    """Check every grid on *sheet*; returns the number of failing grids."""
    wb = openpyxl.load_workbook(path)
    ws = wb[sheet]
    grids = find_grids(ws)
    if verbose:
        print(f"found {len(grids)} grids on {sheet!r}")
    if not grids:
        # A checker that verified nothing must never report green: either the sheet is
        # empty or its layout changed and these invariants are silently unenforced.
        if verbose:
            print(f"\nNOTHING VERIFIED: no acceptance grids recognised on {sheet!r}.")
        return 1
    failures = 0
    for r0, c0, label, nrows, ncols in grids:
        vals, frontier = read_grid(ws, r0, c0, nrows, ncols)
        is_2d = _is_corner(label)
        problems = check_grid(vals, frontier) if is_2d else check_strip(vals)
        failures += bool(problems)
        if verbose:
            counts = {k: int((vals == k).sum()) for k in ("A", "R", "—")}
            status = "FAIL" if problems else "OK "
            shape = f"{nrows}x{ncols}" if is_2d else f"{ncols}-bin strip"
            print(f"[{status}] R{r0}C{c0} ({label}): {shape}, {counts}")
            for p in problems:
                print(f"       - {p}")
    if verbose:
        print(f"\n{failures} grid(s) with problems" if failures else "\nAll grids pass.")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workbook", nargs="?", default="output/consolidated_risk_production.xlsx")
    ap.add_argument("--sheet", default="Cutoff Grids")
    args = ap.parse_args()
    return 1 if check_workbook(args.workbook, args.sheet) else 0


if __name__ == "__main__":
    sys.exit(main())

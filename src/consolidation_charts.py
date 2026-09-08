"""Matplotlib renderings embedded in the consolidated Excel workbook.

Used by the "How We Got Here" sheet (src/consolidation.py): a production bridge
waterfall (Actual → swap-out → swap-in → Proposed) and per-segment efficient-frontier
small multiples with the chosen operating point. Rendering is best-effort — callers
treat a None return as "skip the image, keep the sheet" (trust-layer rule: a missing
artifact shows a note, never crashes the workbook).

Colors reuse the workbook's design tokens. The red/green delta pair and the amber
chosen-point marker are kept in separate figures (amber vs red-coral fails the
normal-vision separation floor side-by-side), and every mark carries a direct text
label so identity never rides on color alone.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

# Workbook design tokens (see src/consolidation.py) on a white chart surface
_NAVY = "#1B2A4A"  # portfolio states (Actual / Proposed), frontier line
_RED = "#EC7063"  # production removed (swap-out)
_GREEN = "#58D68D"  # production added (swap-in)
_AMBER = "#F39C12"  # chosen point marker (never in the same figure as _RED)
_GREY = "#AEB6BF"  # connectors, grid, secondary text
_TEXT = "#2C3E50"

_M = 1e6


def _fmt_eur_m(v: float) -> str:
    return f"€{v / _M:,.1f}M"


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GREY)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.yaxis.grid(True, color="#E8EAED", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def render_production_bridge(totals: dict, out_path: Path) -> Path | None:
    """Waterfall: Actual production → − swap-out → + swap-in → Proposed, with the
    portfolio risk % annotated under the two end states.

    ``totals`` needs actual/swap_out/swap_in/optimum production plus
    actual_risk_pct / optimum_risk_pct (the consolidated TOTAL row's fields).
    """
    try:
        actual = float(totals["actual_production"])
        out = float(totals["swap_out_production"])
        add = float(totals["swap_in_production"])
        proposed = float(totals["optimum_production"])
        risk_a = totals.get("actual_risk_pct")
        risk_p = totals.get("optimum_risk_pct")
    except (KeyError, TypeError, ValueError):
        logger.warning("Production bridge: TOTAL row lacks the required fields — skipping chart")
        return None

    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=150)
    fig.patch.set_facecolor("white")

    xs = [0, 1, 2, 3]
    kept = actual - out
    bars = [
        (0, 0.0, actual, _NAVY, _fmt_eur_m(actual)),
        (1, kept, out, _RED, f"− {_fmt_eur_m(out)}"),
        (2, kept, add, _GREEN, f"+ {_fmt_eur_m(add)}"),
        (3, 0.0, proposed, _NAVY, _fmt_eur_m(proposed)),
    ]
    for x, bottom, height, color, label in bars:
        ax.bar(x, height, bottom=bottom, width=0.58, color=color, zorder=3)
        ax.annotate(
            label,
            (x, bottom + height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=_TEXT,
        )
    # dashed connectors carry the running level across bars
    ax.hlines(actual, 0.29, 1 - 0.29, color=_GREY, linestyle=(0, (3, 2)), linewidth=1, zorder=2)
    ax.hlines(kept, 1 - 0.29, 2 - 0.29, color=_GREY, linestyle=(0, (3, 2)), linewidth=1, zorder=2)
    ax.hlines(kept + add, 2 + 0.29, 3 + 0.29, color=_GREY, linestyle=(0, (3, 2)), linewidth=1, zorder=2)

    def _state(label: str, risk) -> str:
        return f"{label}\nrisk {risk:.2f}%" if pd.notna(risk) else label

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            _state("Today's portfolio", risk_a),
            "Stop lending here\n(high-risk cells)",
            "Newly approved\n(safe cells)",
            _state("Proposed portfolio", risk_p),
        ],
        fontsize=9.5,
        color=_TEXT,
    )
    ax.set_ylabel("Production (€)", fontsize=9.5, color=_TEXT)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / _M:,.0f}M")
    ax.set_ylim(0, max(actual, kept + add, proposed) * 1.16)
    _style_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


def render_frontier_small_multiples(specs: list[dict], out_path: Path, ncols: int = 3) -> Path | None:
    """One panel per segment: the efficient frontier (risk % vs production €M) as a navy
    curve, the segment's risk target as a dashed vertical, and the chosen operating point
    as an amber marker — visually: "of all viable cutoff policies, we took the most
    production that stays left of the target".

    Each spec: {name, risk (sequence, %), production (sequence, €),
    chosen_risk, chosen_production, target (optional)}.
    """
    specs = [s for s in specs if len(s.get("risk", [])) >= 2]
    if not specs:
        return None
    nrows = (len(specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.0 * nrows), dpi=150, squeeze=False)
    fig.patch.set_facecolor("white")

    for i, spec in enumerate(specs):
        ax = axes[i // ncols][i % ncols]
        pts = sorted(zip(spec["risk"], spec["production"], strict=True))
        xs = [p[0] for p in pts]
        ys = [p[1] / _M for p in pts]
        ax.plot(xs, ys, color=_NAVY, linewidth=2, zorder=3)
        ax.plot(xs, ys, linestyle="none", marker="o", markersize=3, color=_NAVY, zorder=3)

        x_mid = (min(xs) + max(xs)) / 2  # labels flip sides in the right half to stay inside the panel
        target = spec.get("target")
        if target is not None and pd.notna(target):
            t = float(target)
            ax.axvline(t, color=_GREY, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
            t_right = t <= x_mid  # text on the empty side of the line
            ax.annotate(
                f"target {t:.2g}%",
                (t, ax.get_ylim()[1]),
                xytext=(3 if t_right else -3, -2),
                textcoords="offset points",
                ha="left" if t_right else "right",
                va="top",
                fontsize=8,
                color=_GREY,
            )
        cr, cp = spec.get("chosen_risk"), spec.get("chosen_production")
        if cr is not None and cp is not None and pd.notna(cr) and pd.notna(cp):
            ax.plot([cr], [cp / _M], marker="*", markersize=15, color=_AMBER, markeredgecolor="white", zorder=4)
            c_right = cr <= x_mid
            y_lo, y_hi = min(ys), max(ys)
            c_low = (cp / _M - y_lo) <= 0.25 * (y_hi - y_lo)  # label above a star near the floor
            ax.annotate(
                f"chosen\n{_fmt_eur_m(cp)} @ {cr:.2f}%",
                (cr, cp / _M),
                xytext=(6 if c_right else -6, 8 if c_low else -16),
                textcoords="offset points",
                ha="left" if c_right else "right",
                va="bottom" if c_low else "top",
                fontsize=8,
                fontweight="bold",
                color=_TEXT,
            )
        ax.set_title(spec["name"], fontsize=9.5, color=_TEXT, fontweight="bold")
        ax.set_xlabel("risk b2_ever_h6 (%)", fontsize=8, color=_GREY)
        ax.set_ylabel("production (€M)", fontsize=8, color=_GREY)
        _style_axes(ax)

    for j in range(len(specs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path

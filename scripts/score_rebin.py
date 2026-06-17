"""Reusable score re-binning tool — consolidate a candidate edge set into ~K balanced groups.

Given a population, a continuous score, a set of candidate bin edges, and (optionally) the risk
inputs and a binary target, this produces:
  • a €/count-balanced grouping at each requested K (merge thin bins; split oversized bins at
    weighted quantiles — adds the minimal number of new interior edges),
  • a risk-monotone grouping (Pool-Adjacent-Violators isotonic merge of b2),
  • b2_ever_h6 per group and the score's Gini (discriminance + binned retention),
  • before/after plots, an old→new crosstab, per-group CSVs, a markdown report, and an optional
    PowerPoint deck.

Defaults reproduce the 2026-06 octroi (score_rf) analysis; override any flag for a new population.

Examples
--------
# reproduce the octroi analysis
uv run python scripts/score_rebin.py

# a different population / score, count-weighted, K=8 and 10, with a deck
uv run python scripts/score_rebin.py \
    --data data/other.sas7bdat --score-col my_score --weight-mode count \
    --groups 8 10 --output-dir output/my_rebin --deck

# different candidate edges from a file (one float per line; use inf / -inf)
uv run python scripts/score_rebin.py --edges-file my_edges.txt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.data_manager import load_data, standardize_columns_and_values
from src.preprocess_improved import filter_by_date

DEFAULT_EDGES = [
    -np.inf,
    0.8894938,
    0.9136649,
    0.9470597,
    0.9654058,
    0.9703869,
    0.9744170,
    0.9777791,
    0.9806289,
    0.9830639,
    0.9851669,
    0.9870236,
    0.9886948,
    0.9902284,
    0.9916906,
    0.9930733,
    0.9943728,
    0.9956056,
    0.9967630,
    0.9977915,
    np.inf,
]


# --------------------------------------------------------------------------- helpers
def fmt(e: float) -> str:
    return "-inf" if e == -np.inf else "+inf" if e == np.inf else f"{e:.7f}"


def parse_edges(s: str) -> list[float]:
    out = []
    for tok in s.replace(" ", "").split(","):
        if tok in ("-inf", "-np.inf"):
            out.append(-np.inf)
        elif tok in ("inf", "+inf", "np.inf"):
            out.append(np.inf)
        elif tok:
            out.append(float(tok))
    return out


def load_population(args) -> pd.DataFrame:
    logger.info(f"Loading {args.data} …")
    if args.data.endswith(".csv"):
        raw = standardize_columns_and_values(pd.read_csv(args.data))
    else:
        raw = standardize_columns_and_values(load_data(args.data))
    if args.date_col and args.date_start and args.date_end:
        raw = filter_by_date(raw, args.date_col, args.date_start, args.date_end)
    if args.segment_col and args.segment_value is not None:
        raw = raw[raw[args.segment_col].astype(str) == str(args.segment_value)]
        logger.info(f"Segment filter {args.segment_col}={args.segment_value}: {len(raw):,} rows")
    cols = {"score": pd.to_numeric(raw[args.score_col], errors="coerce")}
    cols["w"] = (
        pd.Series(1.0, index=raw.index)
        if args.weight_mode == "count"
        else pd.to_numeric(raw[args.weight_col], errors="coerce")
    )
    cols["num"] = pd.to_numeric(raw[args.risk_num], errors="coerce") if args.risk_num in raw else np.nan
    cols["den"] = pd.to_numeric(raw[args.risk_den], errors="coerce") if args.risk_den in raw else np.nan
    cols["bad"] = pd.to_numeric(raw[args.target_col], errors="coerce") if args.target_col in raw else np.nan
    df = pd.DataFrame(cols).dropna(subset=["score", "w"])
    logger.info(
        f"Population: {len(df):,} rows | weight total {df['w'].sum():,.0f} "
        f"({'count' if args.weight_mode == 'count' else args.weight_col})"
    )
    return df


def aggregate(df: pd.DataFrame, edges: list[float], mult: float) -> pd.DataFrame:
    n = len(edges) - 1
    b = pd.cut(df["score"], bins=edges, right=False, labels=False)
    g = df.assign(_b=b).groupby("_b")
    out = pd.DataFrame(index=range(n))
    out["w"] = g["w"].sum().reindex(range(n), fill_value=0.0)
    out["count"] = g.size().reindex(range(n), fill_value=0)
    out["num"] = g["num"].sum().reindex(range(n), fill_value=0.0)
    out["den"] = g["den"].sum().reindex(range(n), fill_value=0.0)
    out["b2"] = np.where(out["den"] > 0, mult * out["num"] / out["den"] * 100, np.nan)
    return out


def weighted_quantiles(vals: np.ndarray, wts: np.ndarray, qs: list[float]) -> list[float]:
    o = np.argsort(vals)
    cw = np.cumsum(wts[o]) / wts.sum()
    return [float(np.interp(q, cw, vals[o])) for q in qs]


def best_partition(weight: np.ndarray, k: int) -> list[int]:
    """Contiguous K-partition minimising Σ(group_weight − total/k)²."""
    n = len(weight)
    if n <= k:
        return list(range(1, n))
    cs = np.concatenate([[0.0], np.cumsum(weight)])
    t = cs[-1] / k
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    par = [[-1] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for g in range(1, k + 1):
        for i in range(g, n - (k - g) + 1):
            for j in range(g - 1, i):
                if dp[g - 1][j] == INF:
                    continue
                c = dp[g - 1][j] + (cs[i] - cs[j] - t) ** 2
                if c < dp[g][i]:
                    dp[g][i], par[g][i] = c, j
    cuts, i = [], n
    for g in range(k, 0, -1):
        j = par[g][i]
        if j > 0:
            cuts.append(j)
        i = j
    return sorted(cuts)


def pava_blocks(num: np.ndarray, den: np.ndarray) -> list[tuple[int, int]]:
    """PAVA non-increasing on b2=num/den → (start,end) spans with strictly decreasing pooled b2."""

    def r(b):
        return (b[0] / b[1]) if b[1] > 0 else -np.inf

    st: list[list] = []
    for i in range(len(num)):
        st.append([num[i], den[i], i, i])
        while len(st) >= 2 and r(st[-2]) <= r(st[-1]):
            x = st.pop()
            y = st.pop()
            st.append([y[0] + x[0], y[1] + x[1], y[2], x[3]])
    return [(b[2], b[3]) for b in st]


def group_table(edges: list[float], agg: pd.DataFrame, cuts: list[int], mult: float) -> pd.DataFrame:
    bnds = [0, *cuts, len(agg)]
    tot = agg["w"].sum()
    rows, fin = [], [edges[0]]
    for gi in range(len(bnds) - 1):
        a, b = bnds[gi], bnds[gi + 1]
        sl = agg.iloc[a:b]
        num, den = sl["num"].sum(), sl["den"].sum()
        rows.append(
            {
                "group": gi + 1,
                "range": f"[{fmt(edges[a])}, {fmt(edges[b])})",
                "lo": edges[a],
                "hi": edges[b],
                "first_bin": a + 1,
                "last_bin": b,
                "weight": sl["w"].sum(),
                "weight_share": sl["w"].sum() / tot,
                "count": int(sl["count"].sum()),
                "b2_ever_h6": (mult * num / den * 100) if den > 0 else np.nan,
            }
        )
        fin.append(edges[b])
    gt = pd.DataFrame(rows)
    gt.attrs["edges"] = fin
    return gt


def scheme_balanced(df, agg_given, edges, k, mult, split_factor=1.5) -> pd.DataFrame:
    target = agg_given["w"].sum() / k
    binned = pd.cut(df["score"], bins=edges, right=False, labels=False)
    exp = list(edges)
    for b in range(len(agg_given)):
        if agg_given["w"].iloc[b] > split_factor * target:
            m = max(2, int(round(agg_given["w"].iloc[b] / target)))
            sub = df[binned == b]
            exp.extend(weighted_quantiles(sub["score"].to_numpy(), sub["w"].to_numpy(), [i / m for i in range(1, m)]))
    exp = sorted(set(exp))
    a = aggregate(df, exp, mult)
    return group_table(exp, a, best_partition(a["w"].to_numpy(), k), mult)


def scheme_monotone(agg_given, edges, k, mult) -> pd.DataFrame:
    num, den = agg_given["num"].to_numpy(), agg_given["den"].to_numpy()
    start = 1 if den[0] == 0 else 0
    s_num, s_den = num.copy(), den.copy()
    if start == 1:
        s_num[1] += s_num[0]
        s_den[1] += s_den[0]
    idx = list(range(start, len(num)))
    spans = pava_blocks(s_num[idx], s_den[idx])
    blocks = [(idx[a] if not (bi == 0 and start == 1) else 0, idx[c]) for bi, (a, c) in enumerate(spans)]
    bk_w = np.array([agg_given["w"].iloc[a : c + 1].sum() for a, c in blocks])
    kk = min(k, len(blocks))
    cuts_b = best_partition(bk_w, kk)
    # map block-level cuts back to the original-bin cut indices
    block_end_bin = [c for _, c in blocks]
    cuts = [block_end_bin[cb - 1] + 1 for cb in cuts_b]
    gt = group_table(edges, agg_given, sorted(cuts), mult)
    gt.attrs["n_blocks"] = len(blocks)
    return gt


def gini_continuous(dfb):
    from sklearn.metrics import roc_auc_score

    return 2 * roc_auc_score(dfb["bad"], -dfb["score"]) - 1


def gini_binned(dfb, edges):
    from sklearn.metrics import roc_auc_score

    grp = pd.cut(dfb["score"], bins=edges, right=False, labels=False)
    rate = dfb["bad"].groupby(grp).transform("mean")
    return 2 * roc_auc_score(dfb["bad"], rate) - 1


# --------------------------------------------------------------------------- plotting
def plot_before_after(given: pd.DataFrame, after: pd.DataFrame, tot: float, k: int, path: str, wlabel: str):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, share, b2, xs, title in [
        (
            axes[0],
            given["w"] / tot * 100,
            given["b2"],
            [str(i + 1) for i in range(len(given))],
            f"BEFORE — {len(given)} candidate bins",
        ),
        (
            axes[1],
            after["weight_share"] * 100,
            after["b2_ever_h6"],
            [str(i) for i in after["group"]],
            f"AFTER — {len(after)} balanced groups",
        ),
    ]:
        x = np.arange(len(share))
        ax.bar(x, share, color="#4C78A8", alpha=0.85)
        ax.axhline(100 / k, color="#888", ls="--", lw=1)
        ax.set_ylabel(f"{wlabel} share (%)", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=8)
        ax.set_title(title, fontsize=12, weight="bold")
        ax2 = ax.twinx()
        bv = np.asarray(b2, dtype=float)
        ax2.plot(x, bv, color="#2CA02C", marker="o", lw=2)
        for i in range(1, len(bv)):
            if not np.isnan(bv[i]) and not np.isnan(bv[i - 1]) and bv[i] > bv[i - 1] + 1e-9:
                ax2.plot([i - 1, i], [bv[i - 1], bv[i]], color="black", lw=3, zorder=5)
        ax2.set_ylabel("b2_ever_h6 (%)", color="#2CA02C")
    axes[1].set_xlabel("group")
    fig.suptitle("Score re-binning: balance + realized risk (black = b2 reversal)", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    logger.info(f"Saved {path}")


# --------------------------------------------------------------------------- report
def write_report(path, args, df, given, tot, wlabel, schemes, mono, ginis, primary_k):
    edges = schemes[primary_k].attrs["edges"]
    arr = "[" + ", ".join(fmt(e) for e in edges) + "]"
    L = [
        f"# {args.title} — findings\n",
        f"**Date generated:** {args.date_tag}  ·  **Score:** `{args.score_col}`  ·  "
        f"**Weight:** {wlabel}  ·  **Population:** {len(df):,} rows, weight {tot:,.0f}\n",
        "## Recommendation\n",
        f"Adopt **K={primary_k}** (`{args.score_col}_bins` below). Weight balance "
        f"min {schemes[primary_k]['weight_share'].min():.1%} / max {schemes[primary_k]['weight_share'].max():.1%} "
        f"(std {schemes[primary_k]['weight_share'].std():.3f}).\n",
        f"```toml\n{args.score_col}_bins = {arr}\n```\n",
        "## Scheme comparison\n",
        "| Scheme | K | weight std | max group | b2 reversals |",
        "|---|---|---|---|---|",
    ]
    for k in args.groups:
        s = schemes[k]
        b2 = s["b2_ever_h6"].to_numpy()
        revs = [i + 1 for i in range(1, len(b2)) if b2[i] > b2[i - 1] + 1e-9]
        L.append(
            f"| balanced | {k} | {s['weight_share'].std():.3f} | {s['weight_share'].max():.1%} | "
            f"{revs if revs else 'none'} |"
        )
    if mono is not None:
        L.append(
            f"| risk-monotone | {len(mono)} | {mono['weight_share'].std():.3f} | "
            f"{mono['weight_share'].max():.1%} | none (strict) |"
        )
    L.append("")
    if ginis:
        L += [
            "## Discriminance (Gini vs target)\n",
            f"Continuous `{args.score_col}` Gini = **{ginis['continuous']:.4f}**.\n",
            "| scheme | Gini | retention |",
            "|---|---|---|",
        ]
        for name, g in ginis["schemes"]:
            L.append(f"| {name} | {g:.4f} | {g / ginis['continuous'] * 100:.1f}% |")
        L.append("")
    L += [
        f"## Recommended K={primary_k} — groups\n",
        "| grp | range | weight share | b2_ever_h6 |",
        "|---|---|---|---|",
    ]
    for _, r in schemes[primary_k].iterrows():
        L.append(f"| {int(r['group'])} | {r['range']} | {r['weight_share']:.1%} | {r['b2_ever_h6']:.2f}% |")
    L += ["\n## Conversion map (old candidate bin → new group)\n", "| old bins | → new group |", "|---|---|"]
    for _, r in schemes[primary_k].iterrows():
        fb, lb = int(r["first_bin"]), int(r["last_bin"])
        L.append(f"| {fb if fb == lb else f'{fb}-{lb}'} | {int(r['group'])} |")
    L.append("\n> Note: a candidate bin that was *split* spans two new groups; see the crosstab CSV.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))
    logger.info(f"Saved {path}")


# --------------------------------------------------------------------------- main
def build_args():
    p = argparse.ArgumentParser(description="Reusable score re-binning tool.")
    p.add_argument("--data", default="data/demanda_ecom_inftel_out.sas7bdat")
    p.add_argument("--score-col", default="score_rf")
    p.add_argument("--weight-mode", choices=["eur", "count"], default="eur")
    p.add_argument("--weight-col", default="oa_amt", help="weight column when --weight-mode eur")
    p.add_argument("--risk-num", default="todu_30ever_h6")
    p.add_argument("--risk-den", default="todu_amt_pile_h6")
    p.add_argument("--multiplier", type=float, default=7.0)
    p.add_argument("--target-col", default="early_bad", help="binary target for Gini (optional)")
    p.add_argument("--date-col", default="mis_date")
    p.add_argument("--date-start", default="2024-06-01")
    p.add_argument("--date-end", default="2025-05-01")
    p.add_argument("--segment-col", default=None)
    p.add_argument("--segment-value", default=None)
    p.add_argument("--edges", default=None, help="comma-separated edges (use -inf/inf); default = octroi 20-bin set")
    p.add_argument("--edges-file", default=None, help="file with one edge per line")
    p.add_argument("--groups", type=int, nargs="+", default=[10, 12, 14])
    p.add_argument("--primary-k", type=int, default=None, help="headline K (default: middle of --groups)")
    p.add_argument("--output-dir", default="output/score_rebin")
    p.add_argument("--prefix", default=None)
    p.add_argument("--title", default="Score re-binning")
    p.add_argument("--date-tag", default="generated")
    p.add_argument("--deck", action="store_true", help="also emit a PowerPoint deck")
    return p.parse_args()


def main() -> None:
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.prefix = args.prefix or args.score_col
    if args.edges_file:
        with open(args.edges_file) as f:
            edges = parse_edges(f.read().replace("\n", ","))
    elif args.edges:
        edges = parse_edges(args.edges)
    else:
        edges = list(DEFAULT_EDGES)
    args.primary_k = args.primary_k or args.groups[len(args.groups) // 2]
    wlabel = "count" if args.weight_mode == "count" else "€"

    df = load_population(args)
    tot = df["w"].sum()
    given = aggregate(df, edges, args.multiplier)
    has_risk = given["den"].sum() > 0

    schemes = {k: scheme_balanced(df, given, edges, k, args.multiplier) for k in args.groups}
    mono = scheme_monotone(given, edges, args.primary_k, args.multiplier) if has_risk else None

    # Gini (optional)
    ginis = None
    if df["bad"].notna().any():
        dfb = df.dropna(subset=["bad"]).copy()
        dfb["bad"] = dfb["bad"].astype(int)
        if dfb["bad"].nunique() >= 2:
            rows = [("candidate bins", gini_binned(dfb, edges))]
            for k in args.groups:
                rows.append((f"K={k} balanced", gini_binned(dfb, schemes[k].attrs["edges"])))
            if mono is not None:
                rows.append((f"risk-monotone (K={len(mono)})", gini_binned(dfb, mono.attrs["edges"])))
            ginis = {"continuous": gini_continuous(dfb), "schemes": rows, "n": len(dfb), "bad_rate": dfb["bad"].mean()}

    # outputs
    given_out = given.assign(bin=range(1, len(given) + 1), weight_share=given["w"] / tot)
    given_out.to_csv(f"{args.output_dir}/{args.prefix}_bins_before.csv", index=False)
    for k in args.groups:
        schemes[k].drop(columns=["lo", "hi"]).to_csv(f"{args.output_dir}/{args.prefix}_groups_k{k}.csv", index=False)
    if mono is not None:
        mono.drop(columns=["lo", "hi"]).to_csv(f"{args.output_dir}/{args.prefix}_groups_monotone.csv", index=False)

    # crosstab old bin -> new group (primary K)
    new_edges = schemes[args.primary_k].attrs["edges"]
    df2 = df.assign(
        old_bin=(pd.cut(df["score"], bins=edges, right=False, labels=False) + 1).astype("Int64"),
        new_grp=(pd.cut(df["score"], bins=new_edges, right=False, labels=False) + 1).astype("Int64"),
    )
    pd.crosstab(df2["old_bin"], df2["new_grp"], values=df2["w"], aggfunc="sum").fillna(0).round(0).to_csv(
        f"{args.output_dir}/{args.prefix}_crosstab_k{args.primary_k}.csv"
    )

    if has_risk:
        plot_before_after(
            given,
            schemes[args.primary_k],
            tot,
            args.primary_k,
            f"{args.output_dir}/{args.prefix}_balance_k{args.primary_k}.png",
            wlabel,
        )

    write_report(
        f"{args.output_dir}/{args.prefix}_report.md", args, df, given, tot, wlabel, schemes, mono, ginis, args.primary_k
    )

    if args.deck:
        try:
            from score_rebin_deck import build_deck  # optional companion

            build_deck(args, schemes, mono, ginis, wlabel)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Deck generation skipped ({e}). Report + CSVs + plot were written.")

    print(
        f"\nDone. Artifacts under {args.output_dir}/ (prefix '{args.prefix}'). "
        f"Recommended K={args.primary_k}:\n    {args.score_col}_bins = "
        f"[{', '.join(fmt(e) for e in new_edges)}]"
    )


if __name__ == "__main__":
    main()

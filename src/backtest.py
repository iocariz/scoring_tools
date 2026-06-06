"""M4 — independent out-of-time backtest of frozen cutoffs.

Applies an already-chosen acceptance policy (the frozen accepted-cell set from a
completed run) **as-is** to a held-out, out-of-time cohort and compares realized vs
predicted risk/production. Unlike the inline MR check (`mr_pipeline.process_mr_period`,
which **re-optimizes** the mask and is risk-only), this never re-fits bins, the model,
or the mask — it is a pure, repeatable backtest of the policy you would actually deploy.

Key design choices (M4):
- **Held-out window is auto-derived** as the cohorts booked *after* the training window
  that are old enough to have a realized H6 outcome: ``(date_fin_book_obs,
  max(mis_date) − maturity_months]``. This trades recency for maturity so the headline
  is *fully realized* (no H3 extrapolation).
- **Production is compared as rates, not absolute €** (the held-out population is a
  different size than training): risk is a rate (directly predicted-vs-realized);
  production is reported as realized acceptance-rate + realized €, compared in-sample
  vs out-of-time as rates.
- The frozen policy is the accepted-cell **set** from ``accepted_cells*.csv`` — a row is
  accepted iff its binned coordinate tuple is in that set (fail-closed on misses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.constants import StatusName
from src.preprocess_improved import _run_data_transformations, filter_by_date
from src.utils import calculate_b2_ever_h6

if TYPE_CHECKING:
    from src.config import PreprocessingSettings


class BacktestError(RuntimeError):
    """Raised when a backtest cannot be run (missing artifacts, unusable config)."""


@dataclass
class BacktestResult:
    """Outcome of backtesting one segment's frozen cutoffs on a held-out cohort."""

    segment: str
    sufficient: bool  # False when there is no mature out-of-time cohort to test
    message: str = ""
    window: dict = field(default_factory=dict)  # start/end/reference_date/maturity_months
    coverage: dict = field(default_factory=dict)  # n_demand / n_booked / n_accepted per period
    predicted: dict = field(default_factory=dict)  # from optimal_solution.csv (risk %, production €)
    in_sample: dict = field(default_factory=dict)  # frozen cutoff on the training window
    out_of_time: dict = field(default_factory=dict)  # frozen cutoff on the held-out window
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-cell predicted vs realized

    def aggregate_row(self) -> dict:
        """Flat one-row summary for the consolidated report."""
        oot_risk = self.out_of_time.get("risk")
        pred_risk = self.predicted.get("risk")
        ins_risk = self.in_sample.get("risk")
        # Same-basis (booked-only realized) drift is the clean signal; predicted carries RI+stress.
        realized_delta = (oot_risk - ins_risk) if (oot_risk is not None and ins_risk is not None) else None
        pred_delta = (oot_risk - pred_risk) if (oot_risk is not None and pred_risk is not None) else None
        return {
            "segment": self.segment,
            "sufficient": self.sufficient,
            "holdout_start": self.window.get("start"),
            "holdout_end": self.window.get("end"),
            "n_booked_oot": self.coverage.get("n_booked_oot"),
            "pct_mature": self.coverage.get("pct_mature"),
            "predicted_risk_pct": pred_risk,
            "in_sample_realized_risk_pct": ins_risk,
            "oot_realized_risk_pct": oot_risk,
            "oot_minus_insample_pp": realized_delta,  # same-basis drift (headline)
            "oot_minus_predicted_pp": pred_delta,  # predicted carries RI+stress conservatism
            "in_sample_acceptance_rate": self.in_sample.get("acceptance_rate"),
            "oot_acceptance_rate": self.out_of_time.get("acceptance_rate"),
            "oot_production_eur": self.out_of_time.get("production"),
            "message": self.message,
        }


# --------------------------------------------------------------------------- #
# Window derivation + policy application
# --------------------------------------------------------------------------- #


def _maturity_months(mis_date: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    """Months between booking and *reference_date* (parity with mr_pipeline._compute_account_maturity)."""
    return (reference_date.year - mis_date.dt.year) * 12 + (reference_date.month - mis_date.dt.month)


def derive_holdout_window(
    data_clean: pd.DataFrame,
    settings: PreprocessingSettings,
    maturity_months: int,
) -> dict:
    """Auto-derive the latest H6-mature out-of-time window after the training window.

    Window = ``(date_fin_book_obs, max(mis_date) − maturity_months]`` — the post-training
    cohorts old enough to have a realized H6 outcome. ``sufficient`` is False when that
    window is empty (the data does not extend far enough past training).
    """
    mis = pd.to_datetime(data_clean["mis_date"])
    reference_date = mis.max()
    train_end = pd.to_datetime(settings.date_fin_book_obs)
    mature_cutoff = reference_date - pd.DateOffset(months=maturity_months)
    sufficient = mature_cutoff > train_end
    return {
        "start": train_end,  # held-out is strictly AFTER train_end (mis_date > start)
        "end": mature_cutoff,  # inclusive upper bound (mis_date <= end)
        "reference_date": reference_date,
        "maturity_months": maturity_months,
        "sufficient": bool(sufficient),
    }


def load_frozen_policy(
    seg_data_dir: Path,
    variables: list[str],
    suffix: str,
) -> tuple[set[tuple[float, ...]], pd.DataFrame]:
    """Load the frozen accepted-cell set + predicted-KPI row from a completed run."""
    accepted_path = seg_data_dir / f"accepted_cells{suffix}.csv"
    opt_path = seg_data_dir / f"optimal_solution{suffix}.csv"
    if not accepted_path.exists():
        raise BacktestError(f"missing frozen policy: {accepted_path}")
    if not opt_path.exists():
        raise BacktestError(f"missing predicted KPIs: {opt_path}")

    accepted_df = pd.read_csv(accepted_path).dropna(subset=variables)
    accepted_set = {tuple(float(row[v]) for v in variables) for _, row in accepted_df.iterrows()}
    opt_sol = pd.read_csv(opt_path)
    if opt_sol.empty:
        raise BacktestError(f"empty optimal solution: {opt_path}")
    return accepted_set, opt_sol


def apply_policy(df: pd.DataFrame, variables: list[str], accepted_set: set[tuple[float, ...]]) -> pd.Series:
    """Boolean Series: True where the row's binned coordinate is in the accepted set (fail-closed)."""
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    coords = list(zip(*[df[v].astype("float64") for v in variables], strict=True))
    return pd.Series([c in accepted_set for c in coords], index=df.index)


# --------------------------------------------------------------------------- #
# Realized metrics
# --------------------------------------------------------------------------- #


def _realized_metrics(booked: pd.DataFrame, accepted: pd.Series, multiplier: float) -> dict:
    """Realized production (€) + risk (%) over the accepted, booked subset."""
    acc = booked[accepted]
    t30 = float(acc["todu_30ever_h6"].sum())
    tamt = float(acc["todu_amt_pile_h6"].sum())
    risk = float(calculate_b2_ever_h6(t30, tamt, multiplier=multiplier, as_percentage=True)) if tamt > 0 else None
    return {
        "production": float(acc["oa_amt_h0"].sum()),
        "risk": risk,
        "n_accepted_booked": int(len(acc)),
        "todu_30ever_h6": t30,
        "todu_amt_pile_h6": tamt,
    }


def _acceptance_rate(demand: pd.DataFrame, accepted: pd.Series) -> float | None:
    """Share of demand rows whose cell is accepted by the policy (coverage)."""
    n = len(demand)
    return float(accepted.sum()) / n if n else None


def _per_cell_calibration(
    booked_oot: pd.DataFrame,
    accepted_oot: pd.Series,
    training_summary: pd.DataFrame,
    variables: list[str],
    accepted_set: set[tuple[float, ...]],
    multiplier: float,
) -> pd.DataFrame:
    """Predicted (training) vs realized (held-out) risk per ACCEPTED cell."""
    # Predicted per-cell risk from the training booked aggregates.
    pred = training_summary.copy()
    if {"todu_30ever_h6_boo", "todu_amt_pile_h6_boo"}.issubset(pred.columns):
        pred["predicted_risk_pct"] = calculate_b2_ever_h6(
            pred["todu_30ever_h6_boo"], pred["todu_amt_pile_h6_boo"], multiplier=multiplier, as_percentage=True
        )
    else:
        pred["predicted_risk_pct"] = pd.NA
    pred_keep = pred[[*variables, "predicted_risk_pct"]]

    # Realized per-cell risk from the held-out accepted booked loans.
    acc = booked_oot[accepted_oot].copy()
    if acc.empty:
        realized = pd.DataFrame(columns=[*variables, "oot_realized_risk_pct", "n_booked_oot"])
    else:
        grp = acc.groupby(variables, dropna=False)
        realized = grp.apply(
            lambda g: pd.Series(
                {
                    "oot_realized_risk_pct": (
                        float(
                            calculate_b2_ever_h6(
                                g["todu_30ever_h6"].sum(),
                                g["todu_amt_pile_h6"].sum(),
                                multiplier=multiplier,
                                as_percentage=True,
                            )
                        )
                        if g["todu_amt_pile_h6"].sum() > 0
                        else None
                    ),
                    "n_booked_oot": int(len(g)),
                }
            ),
            include_groups=False,
        ).reset_index()

    cal = pred_keep.merge(realized, on=variables, how="left")
    # Keep only ACCEPTED cells (the deployed policy) and order by realized sample size.
    cal["accepted"] = [tuple(float(r[v]) for v in variables) in accepted_set for _, r in cal.iterrows()]
    cal = cal[cal["accepted"]].drop(columns="accepted")
    cal["n_booked_oot"] = cal["n_booked_oot"].fillna(0).astype(int)
    cal["risk_delta_pp"] = cal["oot_realized_risk_pct"] - cal["predicted_risk_pct"]
    return cal.sort_values("n_booked_oot", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def backtest_segment(
    full_data: pd.DataFrame,
    settings: PreprocessingSettings,
    seg_output_dir: Path,
    suffix: str = "_base",
    maturity_months: int = 6,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> BacktestResult:
    """Backtest one segment's frozen cutoffs on a held-out out-of-time cohort."""
    segment = settings.segment_filter
    variables = list(settings.variables)
    multiplier = float(settings.multiplier)
    seg_data_dir = Path(seg_output_dir) / "data"

    # Frozen edges are mandatory — re-learning on the held-out population would corrupt the test.
    for var, bc in (settings.bins or {}).items():
        if not bc.bin_edges or len(bc.bin_edges) < 2:
            raise BacktestError(
                f"[{segment}] frozen config bin '{var}' has no explicit bin_edges; cannot reproduce binning"
            )

    accepted_set, opt_sol = load_frozen_policy(seg_data_dir, variables, suffix)
    predicted = {
        "risk": float(opt_sol["b2_ever_h6"].iloc[0]) if "b2_ever_h6" in opt_sol else None,
        "production": float(opt_sol["oa_amt_h0"].iloc[0]) if "oa_amt_h0" in opt_sol else None,
    }

    # Bin the FULL population once with the frozen edges (no re-learning), then slice windows.
    data_clean = _run_data_transformations(full_data, settings)

    # Held-out window (auto-derived unless explicitly overridden).
    if holdout_start and holdout_end:
        window = {
            "start": pd.to_datetime(holdout_start),
            "end": pd.to_datetime(holdout_end),
            "reference_date": pd.to_datetime(data_clean["mis_date"]).max(),
            "maturity_months": maturity_months,
            "sufficient": True,
        }
    else:
        window = derive_holdout_window(data_clean, settings, maturity_months)

    if not window["sufficient"]:
        msg = (
            f"no mature out-of-time cohort: data ends {window['reference_date'].date()}, "
            f"training ends {settings.date_fin_book_obs}, maturity={maturity_months}mo "
            f"→ mature cutoff {window['end'].date()} is not after training."
        )
        logger.warning(f"[{segment}] {msg}")
        return BacktestResult(segment=segment, sufficient=False, message=msg, window=window, predicted=predicted)

    mis = pd.to_datetime(data_clean["mis_date"])
    booked = data_clean[data_clean["status_name"] == StatusName.BOOKED.value].copy()
    booked_mis = pd.to_datetime(booked["mis_date"])

    # In-sample (training window) reference.
    booked_train = filter_by_date(booked, "mis_date", settings.date_ini_book_obs, settings.date_fin_book_obs)
    demand_train = filter_by_date(data_clean, "mis_date", settings.date_ini_book_obs, settings.date_fin_book_obs)

    # Out-of-time (held-out) cohort: booked strictly after training, up to the mature cutoff.
    oot_booked_mask = (booked_mis > window["start"]) & (booked_mis <= window["end"])
    oot_demand_mask = (mis > window["start"]) & (mis <= window["end"])
    booked_oot = booked[oot_booked_mask]
    demand_oot = data_clean[oot_demand_mask]

    acc_train = apply_policy(booked_train, variables, accepted_set)
    acc_oot = apply_policy(booked_oot, variables, accepted_set)
    acc_dem_train = apply_policy(demand_train, variables, accepted_set)
    acc_dem_oot = apply_policy(demand_oot, variables, accepted_set)

    in_sample = _realized_metrics(booked_train, acc_train, multiplier)
    in_sample["acceptance_rate"] = _acceptance_rate(demand_train, acc_dem_train)
    out_of_time = _realized_metrics(booked_oot, acc_oot, multiplier)
    out_of_time["acceptance_rate"] = _acceptance_rate(demand_oot, acc_dem_oot)

    # Coverage: how much of the post-training data is mature (in the window) vs not.
    n_post_train_booked = int((booked_mis > window["start"]).sum())
    coverage = {
        "n_demand_oot": int(len(demand_oot)),
        "n_booked_oot": int(len(booked_oot)),
        "n_accepted_booked_oot": int(acc_oot.sum()),
        "n_post_train_booked": n_post_train_booked,
        "pct_mature": (float(len(booked_oot)) / n_post_train_booked if n_post_train_booked else None),
    }

    training_summary = _load_training_summary(seg_data_dir, suffix)
    calibration = (
        _per_cell_calibration(booked_oot, acc_oot, training_summary, variables, accepted_set, multiplier)
        if training_summary is not None
        else pd.DataFrame()
    )

    return BacktestResult(
        segment=segment,
        sufficient=True,
        window=window,
        coverage=coverage,
        predicted=predicted,
        in_sample=in_sample,
        out_of_time=out_of_time,
        calibration=calibration,
    )


def _load_training_summary(seg_data_dir: Path, suffix: str) -> pd.DataFrame | None:
    """Load the per-cell training booked aggregates (for predicted-risk calibration)."""
    for name in (f"data_summary_desagregado{suffix}.csv", "data_summary_desagregado.csv"):
        path = seg_data_dir / name
        if path.exists():
            return pd.read_csv(path)
    return None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def write_backtest_report(result: BacktestResult, out_dir: Path, suffix: str = "_base") -> dict[str, Path]:
    """Write the per-segment aggregate CSV + per-cell calibration CSV. Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = result.segment
    paths: dict[str, Path] = {}

    agg_path = out_dir / f"backtest_{seg}{suffix}.csv"
    pd.DataFrame([result.aggregate_row()]).to_csv(agg_path, index=False)
    paths["aggregate"] = agg_path

    if result.sufficient and not result.calibration.empty:
        cal_path = out_dir / f"backtest_{seg}{suffix}_calibration.csv"
        result.calibration.to_csv(cal_path, index=False)
        paths["calibration"] = cal_path

    return paths


def write_consolidated_report(results: list[BacktestResult], out_dir: Path, suffix: str = "_base") -> dict[str, Path]:
    """Write the cross-segment consolidated CSV + markdown narrative."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.aggregate_row() for r in results]
    csv_path = out_dir / f"backtest_consolidated{suffix}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    lines = [
        "# Out-of-time backtest — consolidated\n",
        "Frozen cutoffs applied as-is to the auto-derived latest H6-mature out-of-time cohort "
        "(no re-optimization). Production is reported as rates (acceptance %, realized €) — absolute € "
        "is not comparable across periods of different size.\n",
        "**Reading the risk columns:** `predicted` is the optimizer's figure for the chosen solution and "
        "**includes reject-inference + stress conservatism**, so it is not the same basis as a booked-only "
        "realized rate. The clean, same-basis drift signal is **in-sample realized → OOT realized** (both "
        "booked-only under the identical frozen cutoff); the `[OK]/[DRIFT]` flag is based on that. Small "
        "mature windows give noisy per-cell numbers — read the booked counts.\n",
    ]
    for r in results:
        if not r.sufficient:
            lines.append(f"- **{r.segment}**: skipped — {r.message}")
            continue
        pred = r.predicted.get("risk")
        ins = r.in_sample.get("risk")
        oot = r.out_of_time.get("risk")
        # Flag on the same-basis (booked-only realized) drift: in-sample vs OOT.
        realized_delta = (oot - ins) if (ins is not None and oot is not None) else None
        pred_delta = (oot - pred) if (pred is not None and oot is not None) else None
        flag = "—"
        if realized_delta is not None:
            flag = "OK" if abs(realized_delta) <= 0.5 else "DRIFT"
        win = r.window
        lines.append(
            f"- **{r.segment}** [{flag}]: window {win['start'].date()}→{win['end'].date()} "
            f"({r.coverage.get('n_booked_oot')} booked, {_fmt_pct(r.coverage.get('pct_mature'))} mature) | "
            f"realized {_fmt(ins)}% (in-sample) → {_fmt(oot)}% (OOT), Δ {_fmt(realized_delta)} pp | "
            f"predicted {_fmt(pred)}% (RI+stress) → OOT Δ {_fmt(pred_delta)} pp | "
            f"acceptance {_fmt_pct(r.in_sample.get('acceptance_rate'))} → "
            f"{_fmt_pct(r.out_of_time.get('acceptance_rate'))}"
        )
    md_path = out_dir / f"backtest_summary{suffix}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"consolidated": csv_path, "summary": md_path}


def _fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"

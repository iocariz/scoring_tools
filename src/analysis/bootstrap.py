"""Bootstrap confidence intervals for a fixed acceptance mask (booked + reject-inferred)."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from src.core.constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_RISK_MULTIPLIER
from src.core.utils import (
    MAX_PARALLEL_JOBS,
    calculate_b2_ever_h6,
)
from src.optimize.optimization_utils import CellGrid, CutoffSpec


class _RepescaBootstrapContext:
    """Pre-computed arrays for resampling the reject-inferred (repesca) component.

    The repesca cell values (`*_rep`) are deterministic functions of the
    score-rejected loans that fall in each accepted cell (model prediction ×
    stress × RI uplift are all cell-level), so their sampling variance comes
    from the rejected-population composition. Resampling the rejected loans and
    scaling each accepted cell's final rep values by the resampled/original
    amount ratio reproduces that variance without re-running the RI machinery
    (audit #28 — the old bootstrap added the repesca production as a CONSTANT
    to every replicate, assigning the RI'd component zero variance).

    Model/RI *parameter* uncertainty is intentionally NOT captured here — only
    sampling variance of the populations; see the assumptions register.
    """

    def __init__(
        self,
        rejected_loans: pd.DataFrame,
        rep_cells: pd.DataFrame,
        spec: "CutoffSpec",
        variables: list[str],
    ) -> None:
        rep_cols = ["oa_amt_h0_rep", "todu_30ever_h6_rep", "todu_amt_pile_h6_rep"]
        accepted = spec.classify(rep_cells)
        acc = rep_cells.loc[accepted, variables + [c for c in rep_cols if c in rep_cells.columns]]

        cell_pos = {tuple(row): i for i, row in enumerate(map(tuple, acc[variables].values))}
        n_cells = len(cell_pos)
        self.rep_oa = acc["oa_amt_h0_rep"].to_numpy(float) if "oa_amt_h0_rep" in acc.columns else np.zeros(n_cells)
        self.rep_num = (
            acc["todu_30ever_h6_rep"].to_numpy(float) if "todu_30ever_h6_rep" in acc.columns else np.zeros(n_cells)
        )
        self.rep_den = (
            acc["todu_amt_pile_h6_rep"].to_numpy(float) if "todu_amt_pile_h6_rep" in acc.columns else np.zeros(n_cells)
        )

        amt_col = "oa_amt_h0" if "oa_amt_h0" in rejected_loans.columns else "oa_amt"
        loan_pos = np.array([cell_pos.get(k, -1) for k in map(tuple, rejected_loans[variables].values)], dtype=np.int64)
        self.loan_pos = loan_pos
        self.loan_amt = rejected_loans[amt_col].to_numpy(float)
        self.n_loans = len(rejected_loans)
        self.n_cells = n_cells

        in_cells = loan_pos >= 0
        self.base_amt = np.bincount(loan_pos[in_cells], weights=self.loan_amt[in_cells], minlength=n_cells)
        # Cells whose rep values exist but have zero base amount can't be
        # ratio-scaled — keep them constant (ratio 1).
        self._safe_base = np.where(self.base_amt > 0, self.base_amt, 1.0)
        self._scalable = self.base_amt > 0

    def resample(self, rng: np.random.RandomState) -> tuple[float, float, float]:
        """One bootstrap draw of the rejected population → (rep_prod, rep_num, rep_den)."""
        if self.n_loans == 0 or self.n_cells == 0:
            return 0.0, 0.0, 0.0
        idx = rng.randint(0, self.n_loans, self.n_loans)
        pos = self.loan_pos[idx]
        amt = self.loan_amt[idx]
        in_cells = pos >= 0
        resampled_amt = np.bincount(pos[in_cells], weights=amt[in_cells], minlength=self.n_cells)
        ratio = np.where(self._scalable, resampled_amt / self._safe_base, 1.0)
        return (
            float(np.dot(self.rep_oa, ratio)),
            float(np.dot(self.rep_num, ratio)),
            float(np.dot(self.rep_den, ratio)),
        )


def _bootstrap_worker(
    df: pd.DataFrame,
    spec: "CutoffSpec",
    multiplier: float,
    random_state: int | None = None,
    annual_coef: float = 1.0,
    repesca_production: float = 0.0,
    rep_ctx: "_RepescaBootstrapContext | None" = None,
) -> tuple[float, float, float]:
    """Worker function for bootstrap resampling.

    *spec* (todo #65) replaces the previous (cut_map, variables, inv_var1)
    / (mask, grid) double-branch. A CutoffSpec unifies both encodings and
    dispatches via its ``classify`` method.

    Returns ``(production, risk_blended, risk_booked)`` per replicate:

    - ``risk_booked`` — realized risk of the resampled BOOKED loans under the
      cutoff (the only risk the old bootstrap measured).
    - ``risk_blended`` — booked + reject-inferred components on the same basis
      as the optimizer's headline ``b2_ever_h6`` (audit #28). NaN when no
      ``rep_ctx`` is available.
    - ``production`` — booked production (annualized) + the repesca component,
      resampled via *rep_ctx* when available (else the legacy constant).
    """
    # Resample with replacement. Booked draws keep the original integer-seed
    # stream so the legacy (no rep_ctx) path is bit-identical to before; the
    # rejected-population draw uses a derived, independent stream.
    sample = df.sample(frac=1.0, replace=True, random_state=random_state)

    passes = spec.classify(sample)
    passed_df = sample[passes]

    # Use oa_amt_h0 to match the optimization pipeline metric
    prod_col = "oa_amt_h0" if "oa_amt_h0" in passed_df.columns else "oa_amt"
    production_booked = passed_df[prod_col].sum() if not passed_df.empty else 0.0

    # Joint completeness (audit F1 / #41 rule): a loan with a partial H6 outcome is
    # excluded from BOTH risk sums — production above still counts it.
    if not passed_df.empty:
        complete = passed_df["todu_30ever_h6"].notna() & passed_df["todu_amt_pile_h6"].notna()
        risk_num = passed_df.loc[complete, "todu_30ever_h6"].sum()
        risk_den = passed_df.loc[complete, "todu_amt_pile_h6"].sum()
    else:
        risk_num = 0.0
        risk_den = 0.0

    risk_booked = calculate_b2_ever_h6(risk_num, risk_den, multiplier=multiplier, as_percentage=False, decimals=6)

    if rep_ctx is not None:
        rep_rng = np.random.RandomState(None if random_state is None else (random_state + 7919) % (2**31 - 1))
        rep_prod, rep_num, rep_den = rep_ctx.resample(rep_rng)
        production = (production_booked * annual_coef) + rep_prod
        # Blend on the optimizer's basis: the cell `_boo`/`_rep` aggregates are
        # both annualized there, so the booked sums get annual_coef here too
        # (a pure booked ratio cancels the coef; the blend weights do not).
        risk_blended = calculate_b2_ever_h6(
            annual_coef * risk_num + rep_num,
            annual_coef * risk_den + rep_den,
            multiplier=multiplier,
            as_percentage=False,
            decimals=6,
        )
    else:
        production = (production_booked * annual_coef) + repesca_production
        risk_blended = float("nan")

    return production, float(risk_blended), float(risk_booked)


def calculate_bootstrap_intervals(
    data_booked: pd.DataFrame,
    cut_map: dict[float, float] | None = None,
    variables: list[str] | None = None,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    inv_var1: bool = False,
    annual_coef: float = 1.0,
    repesca_production: float = 0.0,
    mask: np.ndarray | None = None,
    grid: object | None = None,
    spec: "CutoffSpec | None" = None,
    rejected_loans: pd.DataFrame | None = None,
    rep_cells: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    """
    Calculate confidence intervals for Risk and Production using bootstrap resampling.

    Two call styles are supported (todo #65):
      1. Pass a pre-built ``spec: CutoffSpec`` (preferred).
      2. Pass one of the legacy encodings directly:
         - 2-var: ``cut_map`` + ``variables`` [+ ``inv_var1``]
         - N-var: ``mask`` + ``grid``
         Internally these are converted to a ``CutoffSpec`` before dispatch.

    Risk-CI basis (audit #28): when *rejected_loans* + *rep_cells* are
    provided, the headline ``risk_ci_*`` is computed on the **blended**
    booked + reject-inferred basis — the same quantity as the optimizer's
    ``b2_ever_h6`` it decorates — with the repesca component resampled
    (see :class:`_RepescaBootstrapContext`). Without them, ``risk_ci_*``
    falls back to the booked-only realized basis (the historic behavior;
    note that basis differs from the blended headline). The basis actually
    used is returned in ``risk_ci_basis``, and the booked-only realized CI
    is always returned separately as ``risk_booked_ci_*``.

    Args:
        data_booked: DataFrame of booked accounts (patient-level data)
        cut_map: (Legacy) 2-var dict mapping var0 bin values to var1 cutoff.
        variables: [var0, var1, ...] — required for the cut_map encoding and
            for the repesca context (cell coordinates of *rep_cells* /
            *rejected_loans*).
        multiplier: Risk multiplier
        n_bootstraps: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95)
        random_state: Seed for reproducibility (default: 42)
        inv_var1: (Legacy) If True, use >= comparison for var1 (higher = safer)
        mask: (Legacy) Binary acceptance mask for N-d classify_by_mask path.
        grid: (Legacy) CellGrid for N-d classify_by_mask path.
        spec: Pre-built CutoffSpec. Takes precedence over legacy kwargs if given.
        rejected_loans: Loan-level score-rejected population (same period and
            bin columns as the data that built *rep_cells*).
        rep_cells: Cell-level frame carrying the final ``*_rep`` columns
            (typically ``data_summary_desagregado``).

    Returns:
        Dictionary with lower/upper bounds for production and risk (blended
        basis when available), ``risk_booked_ci_lower/upper`` (booked-only
        realized basis), and ``risk_ci_basis``.
    """

    logger.info(f"Calculating {confidence_level:.0%} CI with {n_bootstraps} bootstraps...")

    if data_booked.empty:
        logger.warning("Bootstrap CI: data_booked is empty — returning zero CIs")
        return {
            "production_ci_lower": 0.0,
            "production_ci_upper": 0.0,
            "risk_ci_lower": 0.0,
            "risk_ci_upper": 0.0,
            "risk_ci_basis": "booked_realized",
            "risk_booked_ci_lower": 0.0,
            "risk_booked_ci_upper": 0.0,
        }
    if len(data_booked) < 10:
        logger.warning(f"Bootstrap CI: only {len(data_booked)} row(s) in data_booked — CIs may be unreliable")

    # Resolve the cutoff spec: explicit > mask+grid > cut_map+variables.
    if spec is None:
        if mask is not None and grid is not None:
            assert isinstance(grid, CellGrid)  # narrow type for CutoffSpec
            spec = CutoffSpec.from_mask(mask, grid)
        elif cut_map is not None and variables is not None:
            spec = CutoffSpec.from_cut_map(cut_map, variables, inv_var1=inv_var1)
        else:
            raise ValueError(
                "calculate_bootstrap_intervals requires either `spec=CutoffSpec(...)` or one of "
                "the legacy encodings: (cut_map + variables) or (mask + grid)."
            )

    # Optional repesca resampling context (audit #28): gives the RI'd component
    # sampling variance and lets the risk CI match the blended headline basis.
    rep_ctx: _RepescaBootstrapContext | None = None
    if (
        rejected_loans is not None
        and not rejected_loans.empty
        and rep_cells is not None
        and not rep_cells.empty
        and variables
    ):
        try:
            rep_ctx = _RepescaBootstrapContext(rejected_loans, rep_cells, spec, list(variables))
        except Exception:
            logger.warning(
                "Bootstrap CI: failed to build the repesca resampling context — "
                "falling back to the booked-realized basis with constant repesca production.",
                exc_info=True,
            )
            rep_ctx = None

    # Generate per-iteration seeds for reproducibility
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        seeds = rng.randint(0, 2**31 - 1, size=n_bootstraps)
    else:
        seeds = [None] * n_bootstraps

    # Parallel execution (capped to avoid OOM).
    # prefer="threads" avoids heavy fork overhead; numpy releases the GIL so
    # threads still achieve true parallelism for the numerical work.
    results = Parallel(n_jobs=MAX_PARALLEL_JOBS, prefer="threads")(
        delayed(_bootstrap_worker)(
            data_booked,
            spec,
            multiplier,
            random_state=int(seed) if seed is not None else None,
            annual_coef=annual_coef,
            repesca_production=repesca_production,
            rep_ctx=rep_ctx,
        )
        for seed in seeds
    )

    # Unzip results
    productions, risks_blended, risks_booked = zip(*results)

    # Calculate percentiles
    alpha = (1 - confidence_level) / 2
    lower_p = alpha * 100
    upper_p = (1 - alpha) * 100

    prod_lower = np.nanpercentile(productions, lower_p)
    prod_upper = np.nanpercentile(productions, upper_p)
    booked_lower = np.nanpercentile(risks_booked, lower_p)
    booked_upper = np.nanpercentile(risks_booked, upper_p)

    if rep_ctx is not None:
        risk_basis = "blended_booked_plus_ri"
        risk_lower = np.nanpercentile(risks_blended, lower_p)
        risk_upper = np.nanpercentile(risks_blended, upper_p)
    else:
        risk_basis = "booked_realized"
        risk_lower, risk_upper = booked_lower, booked_upper

    return {
        "production_ci_lower": prod_lower,
        "production_ci_upper": prod_upper,
        # Return risk CIs as percentage to match the Risk (%) column convention.
        # Single authoritative conversion point — callers should NOT multiply by 100.
        "risk_ci_lower": risk_lower * 100,
        "risk_ci_upper": risk_upper * 100,
        "risk_ci_basis": risk_basis,
        "risk_booked_ci_lower": booked_lower * 100,
        "risk_booked_ci_upper": booked_upper * 100,
    }

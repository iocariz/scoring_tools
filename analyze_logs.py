"""
Log analyzer that reads pipeline log output, diagnoses issues, and suggests
configuration improvements — including a ready-to-use alternative segments.toml.

Usage:
    uv run python analyze_logs.py pipeline.log
    uv run python analyze_logs.py batch_run.log --verbose
    uv run python analyze_logs.py output/segment/logs/run_*.log -o report.txt
    uv run python analyze_logs.py batch.log --segments-config segments.toml
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class Signal:
    """A single extracted signal from the log."""

    category: str
    key: str
    value: Any
    line_number: int
    raw_line: str
    segment: str = ""


@dataclass
class Recommendation:
    """A single configuration recommendation."""

    priority: str  # HIGH, MEDIUM, LOW
    category: str
    message: str
    setting: str = ""
    current: str = ""
    suggested: str = ""


@dataclass
class ScenarioResult:
    """Results from a single scenario (pessimistic/base/optimistic)."""

    name: str = ""
    risk_threshold: float | None = None
    selected_b2: float | None = None
    production: float | None = None
    is_fallback: bool = False  # True when Pareto solution was unavailable


@dataclass
class MRDecomposition:
    """MR-period optimum decomposition."""

    actual_risk: float | None = None
    actual_prod: float | None = None
    swap_out_risk: float | None = None
    swap_out_prod: float | None = None
    swap_out_pct: float | None = None  # % of actual
    swap_in_risk: float | None = None
    swap_in_prod: float | None = None
    swap_in_pct: float | None = None  # % of actual
    optimum_risk: float | None = None
    optimum_prod: float | None = None


@dataclass
class TierDistribution:
    """Tiered MR risk reconstruction distribution."""

    tier1_actual_h6: int = 0
    tier2_account_h3: int = 0
    tier3_bin_main: int = 0
    tier4_model_fallback: int = 0

    @property
    def total(self) -> int:
        return self.tier1_actual_h6 + self.tier2_account_h3 + self.tier3_bin_main + self.tier4_model_fallback

    def pct(self, tier: int) -> float:
        t = self.total
        if t == 0:
            return 0.0
        counts = [0, self.tier1_actual_h6, self.tier2_account_h3, self.tier3_bin_main, self.tier4_model_fallback]
        return counts[tier] / t * 100 if 1 <= tier <= 4 else 0.0


@dataclass
class SegmentMetrics:
    """Collected metrics for a single segment."""

    name: str = "default"
    segment_filter: str = ""
    # Model
    cv_rmse_mean: float | None = None
    cv_rmse_std: float | None = None
    cv_r2_mean: float | None = None
    cv_r2_std: float | None = None
    train_r2: float | None = None
    model_name: str = ""
    feature_set: str = ""
    # Data
    n_booked: int | None = None
    n_demand: int | None = None
    zero_proportion: float | None = None
    stress_factor: float | None = None
    tasa_fin_pct: float | None = None
    # Optimization
    pareto_n_solutions: int | None = None
    pareto_n_milp: int | None = None
    n_feasible: int | None = None
    ga_fallback: bool = False
    milp_infeasible: bool = False
    grid_dims: str = ""
    b2_range_min: float | None = None
    b2_range_max: float | None = None
    optimum_risk: float | None = None
    n_grid_cells: int | None = None
    n_accepted_cells: int | None = None
    # Scenarios
    scenarios: dict[str, ScenarioResult] = field(default_factory=dict)
    pareto_fallback_targets: list[float] = field(default_factory=list)
    pareto_actual_min: float | None = None  # actual min b2 on the frontier
    # MR
    mr_decomposition: MRDecomposition = field(default_factory=MRDecomposition)
    mr_tiers: TierDistribution = field(default_factory=TierDistribution)
    psi_value: float | None = None
    risk_drift_bins: int = 0
    mr_auto_fallback: bool = False
    h3_floor_bins: int = 0
    h6_h3_ratio_trend_pct: float | None = None
    mr_recal_avg_factor: float | None = None
    mr_recal_h3_avg_factor: float | None = None
    risk_inversion: bool = False
    mr_risk_sources: dict[str, int] = field(default_factory=dict)
    mr_imputed_pct: float | None = None
    mr_low_risk_warning: bool = False
    # Reject inference
    ri_avg_multiplier: float | None = None
    ri_max_multiplier: float | None = None
    ri_min_multiplier: float | None = None
    ri_acceptance_rate_mean: float | None = None
    ri_non_score_rejection_bias: bool = False
    ri_no_feasible: bool = False
    ri_validation_failed: bool = False
    ri_bayesian_smoothing: bool = False
    ri_negative_between_var: bool = False
    # DQ / binning
    dq_warnings: int = 0
    dq_failures: int = 0
    n_anomalous_months: int = 0
    drift_alerts: list[str] = field(default_factory=list)
    outliers_dropped: int = 0
    out_of_range_bins: int = 0
    nan_drop_pct: float | None = None
    n_bootstrap: int | None = None
    monotonicity_directions: dict[str, int] = field(default_factory=dict)
    baseline_mode: bool = False
    # Timing
    elapsed_preprocessing: float | None = None
    elapsed_inference: float | None = None
    elapsed_optimization: float | None = None
    elapsed_total: float | None = None
    # Raw
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Log parsing
# =============================================================================

# Loguru default format: "2026-03-24 11:27:55.987 | LEVEL    | module:func:line - message"
LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*(\w+)\s*\|\s*"
    r"([\w._]+):([\w_]+):(\d+)\s*-\s*(.*)"
)


def parse_log_file(path: Path) -> list[tuple[str, str, str, str, int, str]]:
    """Parse a loguru log file into structured entries.

    Returns list of (timestamp, level, module, function, line_no, message).
    """
    entries = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LOG_PATTERN.match(line.strip())
            if m:
                entries.append((m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5)), m.group(6)))
    return entries


# =============================================================================
# Signal extraction
# =============================================================================


def extract_signals(entries: list[tuple[str, str, str, str, int, str]]) -> tuple[dict[str, SegmentMetrics], list[str]]:
    """Extract key metrics from parsed log entries.

    Returns dict of segment_name -> SegmentMetrics and list of raw errors.
    """
    segments: dict[str, SegmentMetrics] = {}
    current_segment = "default"
    raw_errors: list[str] = []

    def seg() -> SegmentMetrics:
        if current_segment not in segments:
            segments[current_segment] = SegmentMetrics(name=current_segment)
        return segments[current_segment]

    for _timestamp, level, _module, _func, _lineno, msg in entries:
        # Track current segment
        m = re.match(r"PROCESSING SEGMENT:\s+(\S+)", msg)
        if m:
            current_segment = m.group(1)
            seg()
            continue

        m = re.match(r"TRAINING SUPERSEGMENT MODEL:\s+(\S+)", msg)
        if m:
            current_segment = f"_supersegment_{m.group(1)}"
            seg()
            continue

        s = seg()

        # Collect errors and warnings
        if level == "ERROR":
            s.errors.append(msg)
            raw_errors.append(msg)
        if level == "WARNING":
            s.warnings.append(msg)

        # --- Segment filter ---
        m = re.match(r"Segment filter:\s+(\S+)", msg)
        if m:
            s.segment_filter = m.group(1)

        # --- Baseline mode ---
        if "Baseline mode" in msg or "baseline mode" in msg:
            s.baseline_mode = True

        # --- Model performance ---
        m = re.search(r"CV RMSE:\s+([\d.]+)\s*±\s*([\d.]+)", msg)
        if m:
            s.cv_rmse_mean = float(m.group(1))
            s.cv_rmse_std = float(m.group(2))

        m = re.search(r"CV R²:\s+([\d.e+-]+)\s*±\s*([\d.e+-]+)", msg)
        if m:
            s.cv_r2_mean = float(m.group(1))
            s.cv_r2_std = float(m.group(2))

        m = re.search(r"Train R²[^:]*:\s+([\d.e+-]+)", msg)
        if m:
            s.train_r2 = float(m.group(1))

        m = re.search(r"Final model:\s+(.+)", msg)
        if m:
            model_str = m.group(1).strip()
            if " + " in model_str:
                parts = model_str.rsplit(" + ", 1)
                s.model_name = parts[0]
                s.feature_set = parts[1]
            else:
                s.model_name = model_str

        # Model loaded from file
        m = re.search(r"Model loaded.*?(\w+)\s*\|.*?R2=([\d.e+-]+)", msg)
        if m:
            s.model_name = s.model_name or m.group(1)
            s.train_r2 = float(m.group(2))

        # --- Data stats ---
        m = re.search(r"Booked valid records:\s+([\d,]+)", msg)
        if m:
            s.n_booked = int(m.group(1).replace(",", ""))

        m = re.search(r"Zero proportion[^:]*:\s+([\d.]+)%", msg)
        if m:
            s.zero_proportion = float(m.group(1))

        m = re.search(
            r"Preprocessing done.*?booked=([\d,]+)\s+demand=([\d,]+).*?stress=([\d.]+)\s+tasa_fin=([\d.]+)%", msg
        )
        if m:
            s.n_booked = int(m.group(1).replace(",", ""))
            s.n_demand = int(m.group(2).replace(",", ""))
            s.stress_factor = float(m.group(3))
            s.tasa_fin_pct = float(m.group(4))

        m = re.search(r"Data ready.*?([\d,]+)\s+rows", msg)
        if m and s.n_demand is None:
            s.n_demand = int(m.group(1).replace(",", ""))

        # --- Monotonicity ---
        m = re.search(r"direction.*?(\w+)\s*=\s*(-?\d+)", msg)
        if m:
            s.monotonicity_directions[m.group(1)] = int(m.group(2))

        # --- Optimization ---
        m = re.search(
            r"Optimization done.*?mode=(\w+).*?(\d+)\s+solutions.*?(\d+)x(\d+)(?:x(\d+))?\s+grid.*?b2 range:\s*\[([\d.]+)%?,\s*([\d.]+)%?\].*?optimum_risk=([\d.]+)",
            msg,
        )
        if m:
            s.pareto_n_solutions = int(m.group(2))
            dims = f"{m.group(3)}x{m.group(4)}"
            if m.group(5):
                dims += f"x{m.group(5)}"
            s.grid_dims = dims
            s.b2_range_min = float(m.group(6))
            s.b2_range_max = float(m.group(7))
            s.optimum_risk = float(m.group(8))

        m = re.search(r"Pareto frontier:\s+(\d+)\s+solutions.*?(\d+)\s+unique", msg)
        if m:
            s.pareto_n_solutions = int(m.group(1))
            s.pareto_n_milp = int(m.group(2))

        if "No feasible MILP solutions" in msg or "MILP no solutions" in msg:
            s.milp_infeasible = True

        if "GA fallback" in msg or "attempting GA" in msg or "trying GA" in msg:
            s.ga_fallback = True

        m = re.search(r"Number of feasible solutions:\s+(\d+)", msg)
        if m:
            s.n_feasible = int(m.group(1))

        m = re.search(r"[Oo]ptimum.?risk[=:\s]+([\d.]+)", msg)
        if m:
            s.optimum_risk = float(m.group(1))

        # --- Acceptance grid ---
        m = re.search(r"(\d+)\s+cells?,\s+(\d+)\s+accepted", msg)
        if m:
            s.n_grid_cells = int(m.group(1))
            s.n_accepted_cells = int(m.group(2))

        # --- Scenario analysis ---
        m = re.search(
            r"Scenario\s+(\w+)\s*\|\s*risk_threshold=([\d.]+)%\s*\|\s*selected b2=([\d.]+)%\s*\|\s*production=([\d,]+)",
            msg,
        )
        if m:
            scen = ScenarioResult(
                name=m.group(1),
                risk_threshold=float(m.group(2)),
                selected_b2=float(m.group(3)),
                production=float(m.group(4).replace(",", "")),
            )
            s.scenarios[scen.name] = scen

        # --- No feasible solution for scenario ---
        m = re.search(r"Scenario\s+(\w+)\s*\|.*?no feasible solution", msg)
        if m:
            scen = ScenarioResult(name=m.group(1), is_fallback=True)
            s.scenarios[scen.name] = scen

        # --- Pareto fallback ---
        m = re.search(
            r"No Pareto solution with b2_ever_h6 <= ([\d.]+)%.*?Pareto b2 range:\s*\[([\d.]+)%?,\s*([\d.]+)%?\].*?Falling back.*?b2=([\d.]+)%",
            msg,
        )
        if m:
            target = float(m.group(1))
            s.pareto_fallback_targets.append(target)
            s.pareto_actual_min = float(m.group(2))

        # --- MR decomposition ---
        m = re.search(r"Actual:\s+risk=([\d.]+)%\s+prod=([\d,]+)", msg)
        if m and "OPTIMUM DECOMPOSITION" not in msg:
            s.mr_decomposition.actual_risk = float(m.group(1))
            s.mr_decomposition.actual_prod = float(m.group(2).replace(",", ""))

        m = re.search(r"Swap-out:\s+risk=([\d.]+)%\s+prod=([\d,]+)\s+\(([\d.]+)%", msg)
        if m:
            s.mr_decomposition.swap_out_risk = float(m.group(1))
            s.mr_decomposition.swap_out_prod = float(m.group(2).replace(",", ""))
            s.mr_decomposition.swap_out_pct = float(m.group(3))

        m = re.search(r"Swap-in:\s+risk=([\d.]+)%\s+prod=([\d,]+)\s+\(([\d.]+)%", msg)
        if m:
            s.mr_decomposition.swap_in_risk = float(m.group(1))
            s.mr_decomposition.swap_in_prod = float(m.group(2).replace(",", ""))
            s.mr_decomposition.swap_in_pct = float(m.group(3))

        m = re.search(r"Optimum:\s+risk=([\d.]+)%\s+prod=([\d,]+)", msg)
        if m:
            s.mr_decomposition.optimum_risk = float(m.group(1))
            s.mr_decomposition.optimum_prod = float(m.group(2).replace(",", ""))

        # --- Risk inversion ---
        if "RISK INVERSION" in msg:
            s.risk_inversion = True

        # --- Risk sources ---
        m = re.search(
            r"Risk sources:\s+h3_extrapolated=(\d+),\s*mr_observed=(\d+),\s*main_imputed=(\d+),\s*model_fallback=(\d+)",
            msg,
        )
        if m:
            s.mr_risk_sources = {
                "h3_extrapolated": int(m.group(1)),
                "mr_observed": int(m.group(2)),
                "main_imputed": int(m.group(3)),
                "model_fallback": int(m.group(4)),
            }

        # --- Tiered reconstruction ---
        m = re.search(r"Tier 1 \(Actual H6\):\s+([\d,]+)\s+accounts", msg)
        if m:
            s.mr_tiers.tier1_actual_h6 = int(m.group(1).replace(",", ""))
        m = re.search(r"Tier 2 \(Account H3.H6\):\s+([\d,]+)\s+accounts", msg)
        if m:
            s.mr_tiers.tier2_account_h3 = int(m.group(1).replace(",", ""))
        m = re.search(r"Tier 3 \(Bin-level main\):\s+([\d,]+)\s+accounts", msg)
        if m:
            s.mr_tiers.tier3_bin_main = int(m.group(1).replace(",", ""))
        m = re.search(r"Tier 4 \(Model fallback\):\s+([\d,]+)\s+accounts", msg)
        if m:
            s.mr_tiers.tier4_model_fallback = int(m.group(1).replace(",", ""))

        # --- Low MR risk warning ---
        if "LOW MR RISK" in msg:
            s.mr_low_risk_warning = True

        # --- Imputation ratio ---
        m = re.search(r"Imputed accounts:.*?([\d.]+)%\s*\)\s*\|\s*Imputed production:\s*([\d.]+)%", msg)
        if m:
            s.mr_imputed_pct = float(m.group(2))

        # --- Stability ---
        m = re.search(r"Overall PSI:\s+([\d.]+)", msg)
        if m:
            s.psi_value = float(m.group(1))

        m = re.search(r"RISK DRIFT:\s+(\d+)\s+bins", msg)
        if m:
            s.risk_drift_bins = int(m.group(1))

        # --- H3/H6 ---
        m = re.search(r"H6/H3 ratio shows significant trend \(([+-]?[\d.]+)%\)", msg)
        if m:
            s.h6_h3_ratio_trend_pct = float(m.group(1))

        m = re.search(r"H3 floor:\s+(\d+)\s+bin", msg)
        if m:
            s.h3_floor_bins = int(m.group(1))

        # --- MR recalibration ---
        m = re.search(r"MR repesca recalibration:\s+avg factor=([\d.]+)", msg)
        if m:
            s.mr_recal_avg_factor = float(m.group(1))

        m = re.search(r"MR H3 repesca recalibration:\s+avg factor=([\d.]+)", msg)
        if m:
            s.mr_recal_h3_avg_factor = float(m.group(1))

        # --- Reject inference ---
        m = re.search(r"Acceptance rates computed.*?mean=([\d.]+)", msg)
        if m:
            s.ri_acceptance_rate_mean = float(m.group(1))

        m = re.search(r"Swap-in RI multiplier:\s+min=([\d.]+),\s*avg=([\d.]+),\s*max=([\d.]+)", msg)
        if m:
            s.ri_min_multiplier = float(m.group(1))
            s.ri_avg_multiplier = float(m.group(2))
            s.ri_max_multiplier = float(m.group(3))

        if "Non-score rejections exceed" in msg:
            s.ri_non_score_rejection_bias = True

        if "Bayesian smoothing applied" in msg:
            s.ri_bayesian_smoothing = True

        if "between-bin variance is non-positive" in msg:
            s.ri_negative_between_var = True

        if "RI optimizer: no feasible solution" in msg or "RI optimizer: all feasible solutions non-finite" in msg:
            s.ri_no_feasible = True

        if "RI temporal validation failed" in msg:
            s.ri_validation_failed = True

        # --- MR auto-calibration fallback ---
        if "Auto-calibration: insufficient bins" in msg or "falling back to linear" in msg:
            s.mr_auto_fallback = True

        # --- DQ ---
        m = re.search(r"Data Quality:\s+\d+/\d+ passed.*?(\d+)\s+warning.*?(\d+)\s+failure", msg)
        if m:
            s.dq_warnings = int(m.group(1))
            s.dq_failures = int(m.group(2))

        # --- Trend & anomaly ---
        m = re.search(r"anomal(?:ies|ous).*?(\d+)\s+month", msg)
        if m:
            s.n_anomalous_months = int(m.group(1))

        if "DRIFT ALERT" in msg:
            s.drift_alerts.append(msg)

        # --- Binning ---
        m = re.search(r"Dropped.*?(\d+).*?outlier", msg, re.IGNORECASE)
        if m:
            s.outliers_dropped += int(m.group(1))

        m = re.search(r"out-of-range.*?(\d+)", msg, re.IGNORECASE)
        if m:
            s.out_of_range_bins += int(m.group(1))

        m = re.search(r"dropped.*?NaN.*?([\d.]+)%", msg)
        if m:
            pct = float(m.group(1))
            if s.nan_drop_pct is None or pct > s.nan_drop_pct:
                s.nan_drop_pct = pct

        # --- Timing (phase-level) ---
        m = re.search(r"Preprocessing done.*?([\d.]+)s", msg)
        if m:
            s.elapsed_preprocessing = float(m.group(1))

        m = re.search(r"Inference done.*?([\d.]+)s", msg)
        if m:
            s.elapsed_inference = float(m.group(1))

        m = re.search(r"Optimization done.*?([\d.]+)s", msg)
        if m:
            s.elapsed_optimization = float(m.group(1))

        m = re.search(r"Pipeline complete.*?([\d.]+)\s*(?:s|seconds)", msg)
        if m:
            s.elapsed_total = float(m.group(1))

    # Propagate supersegment model metrics to member segments that have R²=0
    for ss_name, ss in segments.items():
        if not ss_name.startswith("_supersegment_"):
            continue
        if ss.train_r2 is None and ss.cv_r2_mean is None:
            continue
        for seg_name, seg_s in segments.items():
            if seg_name.startswith("_supersegment_") or seg_name == "default":
                continue
            if seg_s.cv_r2_mean is not None and seg_s.cv_r2_mean == 0.0 and seg_s.cv_r2_std == 0.0:
                if ss.train_r2 is not None:
                    seg_s.train_r2 = ss.train_r2
                if ss.cv_rmse_mean is not None:
                    seg_s.cv_rmse_mean = ss.cv_rmse_mean
                    seg_s.cv_rmse_std = ss.cv_rmse_std
                if ss.model_name:
                    seg_s.model_name = seg_s.model_name or ss.model_name
                seg_s.cv_r2_mean = None
                seg_s.cv_r2_std = None

    return segments, raw_errors


# =============================================================================
# Recommendation engine
# =============================================================================


def generate_recommendations(segments: dict[str, SegmentMetrics]) -> list[Recommendation]:
    """Generate configuration recommendations from extracted metrics."""
    recs: list[Recommendation] = []

    for name, s in segments.items():
        if name.startswith("_supersegment_"):
            continue
        prefix = f"[{name}] " if len(segments) > 1 else ""

        # --- Hard errors ---
        for err in s.errors:
            if "Unexpected segment error" in err or "Pipeline returned no result" in err:
                recs.append(
                    Recommendation(
                        priority="HIGH",
                        category="Error",
                        message=f"{prefix}Pipeline failed: {err[:200]}",
                    )
                )

        if s.dq_failures > 0:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="Data Quality",
                    message=f"{prefix}{s.dq_failures} DQ check(s) failed. Pipeline may have skipped this segment.",
                    setting="data source / segment_filter",
                    suggested="Check DQ report output. Use --skip-dq-checks only for debugging.",
                )
            )

        # --- Model quality ---
        if s.cv_r2_mean is not None and s.cv_r2_mean < 0.1:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="Model",
                    message=f"{prefix}Very low CV R² ({s.cv_r2_mean:.4f}). Model has weak predictive power.",
                    setting="inference_variables / cv_folds",
                    current=f"R²={s.cv_r2_mean:.4f}",
                    suggested="Try adding more inference_variables, increasing cv_folds to 5, or enabling Optuna tuning.",
                )
            )
        elif s.cv_r2_mean is not None and s.cv_r2_mean < 0.3:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Model",
                    message=f"{prefix}Low CV R² ({s.cv_r2_mean:.4f}). Consider enriching feature set.",
                    setting="inference_variables",
                    current=f"R²={s.cv_r2_mean:.4f}",
                    suggested="Add more predictive variables or try enabling Optuna hyperparameter tuning.",
                )
            )

        if s.train_r2 is not None and s.cv_r2_mean is not None:
            gap = s.train_r2 - s.cv_r2_mean
            if gap > 0.15:
                recs.append(
                    Recommendation(
                        priority="MEDIUM",
                        category="Model",
                        message=f"{prefix}Possible overfitting: Train R²={s.train_r2:.4f} vs CV R²={s.cv_r2_mean:.4f} (gap={gap:.4f}).",
                        setting="cv_folds / model regularization",
                        suggested="Increase cv_folds (e.g. 5), use stronger regularization, or reduce feature count.",
                    )
                )

        if s.cv_rmse_std is not None and s.cv_rmse_mean is not None and s.cv_rmse_mean > 0:
            cv_coeff = s.cv_rmse_std / s.cv_rmse_mean
            if cv_coeff > 0.3:
                recs.append(
                    Recommendation(
                        priority="MEDIUM",
                        category="Model",
                        message=f"{prefix}High CV variance (RMSE CoV={cv_coeff:.2f}). Model is unstable across folds.",
                        setting="cv_folds / sample size",
                        suggested="Increase cv_folds or ensure enough booked records per fold.",
                    )
                )

        # --- Sample size ---
        if s.n_booked is not None and s.n_booked < 500:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="Data",
                    message=f"{prefix}Very small sample: only {s.n_booked} booked records.",
                    setting="date range / segment filter",
                    suggested="Expand date_ini_book_obs/date_fin_book_obs or merge with related segment via modelling_supersegment.",
                )
            )
        elif s.n_booked is not None and s.n_booked < 2000:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Data",
                    message=f"{prefix}Small sample: {s.n_booked} booked records. Model may be unreliable.",
                    setting="date range / modelling_supersegment",
                    suggested="Consider sharing training data via modelling_supersegment or expanding date range.",
                )
            )

        if s.zero_proportion is not None and s.zero_proportion > 95:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Data",
                    message=f"{prefix}Extremely high zero proportion ({s.zero_proportion:.1f}%). Hurdle model recommended.",
                    setting="Model selection",
                    suggested="Ensure HurdleRegressor is available in the model candidates (it should be by default).",
                )
            )

        # --- Optimization: unreachable optimum_risk ---
        if s.pareto_fallback_targets:
            actual_min = s.pareto_actual_min
            if actual_min is not None and s.optimum_risk is not None:
                recs.append(
                    Recommendation(
                        priority="HIGH",
                        category="Optimization",
                        message=(
                            f"{prefix}optimum_risk={s.optimum_risk}% is unreachable. "
                            f"Minimum achievable risk on the Pareto frontier is {actual_min:.2f}%. "
                            f"All scenarios fell back to the minimum-risk solution."
                        ),
                        setting="optimum_risk",
                        current=f"{s.optimum_risk}%",
                        suggested=f"Set optimum_risk >= {actual_min:.2f} (suggest {_round_up(actual_min, 0.1):.2f}).",
                    )
                )

        if s.milp_infeasible:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="Optimization",
                    message=f"{prefix}MILP solver found no feasible solutions."
                    + (" GA fallback was triggered." if s.ga_fallback else ""),
                    setting="optimum_risk / risk_step / monotonicity constraints",
                    suggested=(
                        "Relax optimum_risk (increase target), widen risk_step, "
                        "check monotonicity directions, or enable monotonicity_relaxation_enabled=true."
                    ),
                )
            )

        if s.pareto_n_solutions is not None and s.pareto_n_solutions < 5:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Optimization",
                    message=f"{prefix}Very sparse Pareto frontier ({s.pareto_n_solutions} solutions).",
                    setting="pareto_n_points / risk_step",
                    suggested="Increase pareto_n_points (e.g. 100) or reduce risk_step for finer granularity.",
                )
            )

        # --- Scenario: base didn't improve over actual ---
        base = s.scenarios.get("base")
        if base and s.mr_decomposition.actual_risk is not None and base.selected_b2 is not None:
            if (
                s.mr_decomposition.optimum_risk is not None
                and s.mr_decomposition.optimum_risk > s.mr_decomposition.actual_risk
            ):
                recs.append(
                    Recommendation(
                        priority="MEDIUM",
                        category="MR Validation",
                        message=(
                            f"{prefix}MR optimum risk ({s.mr_decomposition.optimum_risk:.2f}%) is worse than "
                            f"actual risk ({s.mr_decomposition.actual_risk:.2f}%). The proposed cutoffs "
                            f"increase risk in the MR period."
                        ),
                        setting="optimum_risk / reject_inference",
                        suggested="Review if the main-period cutoffs generalize to MR. Consider tightening optimum_risk.",
                    )
                )

        # No feasible solution for a scenario
        for scen_name, scen in s.scenarios.items():
            if scen.is_fallback and scen.selected_b2 is None:
                recs.append(
                    Recommendation(
                        priority="HIGH",
                        category="Optimization",
                        message=f"{prefix}Scenario '{scen_name}' found no feasible solution on the Pareto frontier.",
                        setting="optimum_risk / risk_step",
                        suggested="The risk target for this scenario is likely unreachable. Adjust optimum_risk or risk_step.",
                    )
                )

        # --- Risk inversion ---
        if s.risk_inversion:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="MR Validation",
                    message=(
                        f"{prefix}Risk inversion detected: swap-out bins have lower risk than actual. "
                        f"The risk ordering has shifted between main and MR periods."
                    ),
                    setting="bin edges / MR date range",
                    suggested="Investigate population drift. Re-learn bin edges or adjust MR date range.",
                )
            )

        # --- MR risk sources ---
        if s.mr_risk_sources:
            fb = s.mr_risk_sources.get("model_fallback", 0)
            total = sum(s.mr_risk_sources.values())
            if total > 0 and fb / total > 0.3:
                recs.append(
                    Recommendation(
                        priority="MEDIUM",
                        category="MR Validation",
                        message=(
                            f"{prefix}{fb}/{total} MR bins ({fb / total * 100:.0f}%) use model fallback risk. "
                            f"MR risk estimates may be unreliable."
                        ),
                        setting="mr_min_obs_per_bin / MR date range",
                        suggested="Lower mr_min_obs_per_bin or extend MR observation window for more data.",
                    )
                )

        # --- MR imputation ---
        if s.mr_imputed_pct is not None and s.mr_imputed_pct > 20:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="MR Validation",
                    message=f"{prefix}{s.mr_imputed_pct:.1f}% of MR production comes from model-imputed bins.",
                    setting="bin granularity",
                    suggested="Consider coarser binning (fewer bins) to reduce the number of MR-only cells.",
                )
            )

        # --- Low MR risk warning ---
        if s.mr_low_risk_warning:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="MR Validation",
                    message=f"{prefix}MR-observed H6 risk is suspiciously low (<10% of main-period risk). Likely immature loan dilution.",
                    setting="mr_maturity_months / mr_min_obs_per_bin",
                    suggested="Increase mr_maturity_months to exclude immature accounts from H6 risk computation.",
                )
            )

        # --- Reject inference ---
        if s.ri_non_score_rejection_bias:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Reject Inference",
                    message=f"{prefix}Non-score rejections exceed 10% of demand — acceptance rate bias may be material.",
                    setting="reject_include_all_rejections",
                    suggested="Set reject_include_all_rejections = true to include all rejections in the denominator.",
                )
            )

        if s.ri_avg_multiplier is not None and s.ri_avg_multiplier > 3.0:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Reject Inference",
                    message=f"{prefix}High average RI multiplier ({s.ri_avg_multiplier:.2f}×). Repesca risk uplift is aggressive.",
                    setting="reject_uplift_factor / reject_parceling_method",
                    suggested="Consider reducing reject_uplift_factor or switching to a milder parceling method ('linear').",
                )
            )
        elif s.ri_avg_multiplier is not None and s.ri_avg_multiplier > 2.0:
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="Reject Inference",
                    message=f"{prefix}Moderately high RI multiplier ({s.ri_avg_multiplier:.2f}×).",
                    setting="reject_uplift_factor",
                    suggested="Monitor swap-in actual risk vs predicted risk. If bias emerges, reduce reject_uplift_factor.",
                )
            )

        if s.ri_negative_between_var:
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="Reject Inference",
                    message=f"{prefix}Bayesian smoothing: between-bin variance was non-positive. Data may not support random-effects model.",
                    setting="reject_bayesian_prior_strength",
                    suggested="The configured prior_strength is being used as fallback. Consider increasing it for more smoothing.",
                )
            )

        if s.ri_no_feasible:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Reject Inference",
                    message=f"{prefix}RI optimizer found no feasible solution.",
                    setting="ri_uplift_range / ri_max_mult_range",
                    suggested="Widen search ranges: ri_uplift_range=[0.0, 8.0], ri_max_mult_range=[1.0, 8.0].",
                )
            )

        if s.ri_validation_failed:
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="Reject Inference",
                    message=f"{prefix}RI temporal validation failed (non-blocking).",
                    setting="ri_validation_split",
                    suggested="Increase ri_validation_split (e.g. 0.8) to give more data to training.",
                )
            )

        # --- H3/H6 ratio ---
        if s.h6_h3_ratio_trend_pct is not None and abs(s.h6_h3_ratio_trend_pct) > 50:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="MR Validation",
                    message=f"{prefix}H6/H3 ratio shows significant trend ({s.h6_h3_ratio_trend_pct:+.1f}%). Extrapolation may be inaccurate.",
                    setting="mr_extrapolation_method",
                    suggested="Use mr_extrapolation_method='auto' for data-driven curvature fitting.",
                )
            )

        if s.h3_floor_bins > 3:
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="MR Validation",
                    message=f"{prefix}{s.h3_floor_bins} bin(s) had H6 risk clamped to H3 floor.",
                    setting="mr_extrapolation_curvature",
                    suggested="H3 floor is a safety net. If many bins are floored, the extrapolation may be underestimating H6 risk.",
                )
            )

        # --- MR recalibration ---
        if s.mr_recal_avg_factor is not None:
            if s.mr_recal_avg_factor < 0.5 or s.mr_recal_avg_factor > 2.0:
                recs.append(
                    Recommendation(
                        priority="MEDIUM",
                        category="MR Validation",
                        message=(
                            f"{prefix}MR recalibration factor is extreme (avg={s.mr_recal_avg_factor:.3f}). "
                            f"Repesca risk was scaled by {'<0.5×' if s.mr_recal_avg_factor < 0.5 else '>2.0×'}."
                        ),
                        setting="MR date range / reject_inference parameters",
                        suggested="Large calibration factors suggest main vs MR risk divergence. Review period selection.",
                    )
                )

        # --- Stability ---
        if s.psi_value is not None:
            if s.psi_value > 0.25:
                recs.append(
                    Recommendation(
                        priority="HIGH",
                        category="Stability",
                        message=f"{prefix}Significant population shift (PSI={s.psi_value:.4f} > 0.25).",
                        setting="date ranges / bin edges",
                        suggested="Investigate population drift between main and MR. Consider re-learning bin edges.",
                    )
                )
            elif s.psi_value > 0.1:
                recs.append(
                    Recommendation(
                        priority="MEDIUM",
                        category="Stability",
                        message=f"{prefix}Moderate population shift (PSI={s.psi_value:.4f}). Monitor trend.",
                        setting="date ranges",
                        suggested="Review MR date range or check for data quality changes between periods.",
                    )
                )

        if s.risk_drift_bins > 3:
            recs.append(
                Recommendation(
                    priority="HIGH",
                    category="MR Validation",
                    message=f"{prefix}Risk drift in {s.risk_drift_bins} bins (>20% deviation main vs MR).",
                    setting="mr_extrapolation_method / mr_min_obs_per_bin",
                    suggested="Try mr_extrapolation_method='auto'. Increase mr_min_obs_per_bin for more observations per cell.",
                )
            )

        if s.mr_auto_fallback:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="MR Validation",
                    message=f"{prefix}MR auto-calibration fell back to linear (insufficient data).",
                    setting="mr_min_obs_per_bin / mr_extrapolation_method",
                    suggested="Lower mr_min_obs_per_bin (e.g. 15) or set mr_extrapolation_method='power' with manual curvature.",
                )
            )

        # --- DQ ---
        if s.dq_warnings > 5:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Data Quality",
                    message=f"{prefix}{s.dq_warnings} data quality warnings.",
                    setting="Data pipeline / input data",
                    suggested="Review DQ warnings and fix upstream data issues before production runs.",
                )
            )

        # --- Binning ---
        if s.nan_drop_pct is not None and s.nan_drop_pct > 10:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Binning",
                    message=f"{prefix}High NaN rate during binning ({s.nan_drop_pct:.1f}% dropped).",
                    setting="bin source columns / data imputation",
                    suggested="Check source_col data quality. Consider imputation or adjusting bin edges.",
                )
            )

        if s.out_of_range_bins > 100:
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="Binning",
                    message=f"{prefix}{s.out_of_range_bins} records fell outside bin edges.",
                    setting="bin_edges",
                    suggested="Re-learn bin edges with method='quantile' or extend edge range to cover data.",
                )
            )

        # --- Anomalies ---
        if s.n_anomalous_months > 3:
            recs.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Trends",
                    message=f"{prefix}{s.n_anomalous_months} anomalous months detected in trend analysis.",
                    setting="date range / segment filter",
                    suggested="Investigate anomalous months. Consider narrowing date_fin_book_obs.",
                )
            )

        # --- Performance ---
        if s.elapsed_total is not None and s.elapsed_total > 600:
            recs.append(
                Recommendation(
                    priority="LOW",
                    category="Performance",
                    message=f"{prefix}Pipeline took {s.elapsed_total:.0f}s ({s.elapsed_total / 60:.1f} min).",
                    setting="n_bootstraps / pareto_n_points",
                    suggested="Reduce n_bootstraps (e.g. 500) or pareto_n_points (e.g. 30) for faster iteration.",
                )
            )

    return recs


# =============================================================================
# Alternative segments.toml generation
# =============================================================================


def _round_up(value: float, step: float) -> float:
    """Round value up to the nearest step."""
    return round(math.ceil(value / step) * step, 2)


def _round_down(value: float, step: float) -> float:
    """Round value down to the nearest step."""
    return round(math.floor(value / step) * step, 2)


def generate_alternative_segments_toml(
    segments: dict[str, SegmentMetrics],
    current_segments_path: Path | None = None,
) -> str:
    """Generate an alternative segments.toml based on log analysis.

    Reads the current segments.toml (if available) to preserve structure,
    then applies data-driven adjustments to optimum_risk, risk_step, etc.
    """
    import tomllib

    # Load current config if available
    current_config: dict[str, Any] = {}
    if current_segments_path and current_segments_path.exists():
        with open(current_segments_path, "rb") as f:
            current_config = tomllib.load(f)

    current_segments_cfg = current_config.get("segments", {})

    lines: list[str] = []
    changes: list[str] = []

    lines.append("# =============================================================================")
    lines.append("# ALTERNATIVE segments.toml — generated by analyze_logs.py")
    lines.append("# Review changes marked with # CHANGED before applying.")
    lines.append("# =============================================================================")
    lines.append("")

    # Preserve supersegment definitions verbatim
    for ss_type in ["modelling_supersegments", "reporting_supersegments", "supersegments"]:
        ss_block = current_config.get(ss_type, {})
        if ss_block:
            for ss_name, ss_cfg in ss_block.items():
                lines.append(f"[{ss_type}.{ss_name}]")
                for k, v in ss_cfg.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            lines.append(f"{k}.{sub_k} = {_toml_value(sub_v)}")
                    else:
                        lines.append(f"{k} = {_toml_value(v)}")
                lines.append("")

    # Process each segment
    real_segments = {
        name: s for name, s in segments.items() if not name.startswith("_supersegment_") and name != "default"
    }

    for seg_name, s in sorted(real_segments.items()):
        cur = current_segments_cfg.get(seg_name, {})
        cur_optimum = cur.get("optimum_risk")
        cur_risk_step = cur.get("risk_step")

        new_optimum = cur_optimum
        new_risk_step = cur_risk_step
        seg_changes: list[str] = []

        # --- Adjust optimum_risk if unreachable ---
        if s.pareto_fallback_targets and s.pareto_actual_min is not None:
            suggested = _round_up(s.pareto_actual_min, 0.05)
            if cur_optimum is not None and cur_optimum < s.pareto_actual_min:
                new_optimum = suggested
                seg_changes.append(
                    f"optimum_risk: {cur_optimum} -> {new_optimum} "
                    f"(was unreachable, min Pareto b2={s.pareto_actual_min:.2f}%)"
                )

        # --- Adjust optimum_risk if base scenario selected same as pessimistic ---
        base = s.scenarios.get("base")
        pess = s.scenarios.get("pessimistic")
        opti = s.scenarios.get("optimistic")
        if base and pess and base.selected_b2 == pess.selected_b2 and base.production == pess.production:
            if base.selected_b2 is not None and s.optimum_risk is not None:
                gap = s.optimum_risk - base.selected_b2
                if gap > 0.15:
                    suggested = _round_up(base.selected_b2, 0.05)
                    if cur_optimum is not None and suggested < cur_optimum:
                        new_optimum = suggested
                        seg_changes.append(
                            f"optimum_risk: {cur_optimum} -> {new_optimum} "
                            f"(base=pessimistic at b2={base.selected_b2:.2f}%, target had slack)"
                        )

        # --- Suggest tighter risk_step if frontier is sparse ---
        if s.pareto_n_solutions is not None and s.pareto_n_solutions < 8:
            if cur_risk_step and cur_risk_step >= 0.1:
                new_risk_step = 0.05
                seg_changes.append(
                    f"risk_step: {cur_risk_step} -> {new_risk_step} "
                    f"(sparse frontier with {s.pareto_n_solutions} solutions)"
                )

        # --- Suggest wider risk_step if all 3 scenarios select different solutions ---
        if base and pess and opti:
            if pess.selected_b2 != base.selected_b2 != opti.selected_b2 and cur_risk_step and cur_risk_step <= 0.05:
                new_risk_step = 0.1
                seg_changes.append(
                    f"risk_step: {cur_risk_step} -> {new_risk_step} "
                    f"(all 3 scenarios select distinct solutions, step may be too fine)"
                )

        # --- Write segment ---
        lines.append(f"[segments.{seg_name}]")
        for k, v in cur.items():
            if k == "optimum_risk" and new_optimum is not None:
                if new_optimum != cur_optimum:
                    lines.append(f"optimum_risk = {new_optimum}  # CHANGED (was {cur_optimum})")
                else:
                    lines.append(f"optimum_risk = {v}")
            elif k == "risk_step" and new_risk_step is not None:
                if new_risk_step != cur_risk_step:
                    lines.append(f"risk_step = {new_risk_step}  # CHANGED (was {cur_risk_step})")
                else:
                    lines.append(f"risk_step = {v}")
            elif k == "fixed_cutoffs":
                lines.append("")
                _write_fixed_cutoffs(lines, seg_name, v)
                continue
            else:
                lines.append(f"{k} = {_toml_value(v)}")

        # If segment wasn't in current config, write what we know
        if not cur:
            if s.segment_filter:
                lines.append(f'segment_filter = "{s.segment_filter}"')
            if new_optimum is not None:
                lines.append(f"optimum_risk = {new_optimum}")
            if new_risk_step is not None:
                lines.append(f"risk_step = {new_risk_step}")

        lines.append("")

        if seg_changes:
            changes.extend([f"  [{seg_name}] {c}" for c in seg_changes])

    return "\n".join(lines), changes


def _write_fixed_cutoffs(lines: list[str], seg_name: str, fc: dict) -> None:
    """Write fixed_cutoffs as TOML sub-tables."""
    simple_keys = {}
    table_keys = {}
    for k, v in fc.items():
        if isinstance(v, dict):
            table_keys[k] = v
        else:
            simple_keys[k] = v

    lines.append(f"[segments.{seg_name}.fixed_cutoffs]")
    for k, v in simple_keys.items():
        lines.append(f"{k} = {_toml_value(v)}")

    for k, v in table_keys.items():
        lines.append("")
        lines.append(f"[segments.{seg_name}.fixed_cutoffs.{k}]")
        for sub_k, sub_v in v.items():
            lines.append(f"{sub_k} = {_toml_value(sub_v)}")


def _toml_value(v: Any) -> str:
    """Format a Python value as TOML."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, float) and math.isinf(item):
                parts.append("-inf" if item < 0 else "inf")
            elif isinstance(item, float):
                parts.append(f"{item}" if item != int(item) else f"{item}")
            elif isinstance(item, str):
                parts.append(f'"{item}"')
            else:
                parts.append(str(item))
        return f"[{', '.join(parts)}]"
    if isinstance(v, float) and math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return str(v)


# =============================================================================
# Output formatting
# =============================================================================

PRIORITY_ICON = {"HIGH": "!!!", "MEDIUM": " ! ", "LOW": "   "}
PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def format_report(
    segments: dict[str, SegmentMetrics],
    recommendations: list[Recommendation],
    verbose: bool = False,
    alt_toml: str = "",
    alt_changes: list[str] | None = None,
) -> str:
    """Format analysis results as a text report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PIPELINE LOG ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Filter to real segments for display
    real_segments = {n: s for n, s in segments.items() if not n.startswith("_supersegment_") and n != "default"}

    lines.append(f"Segments analyzed: {len(real_segments)}")
    lines.append("")

    # --- Per-segment summary ---
    for name, s in sorted(real_segments.items()):
        lines.append(f"--- {name} {'(BASELINE)' if s.baseline_mode else ''}---")
        parts = []
        if s.segment_filter:
            parts.append(f"Filter: {s.segment_filter}")
        if s.model_name:
            parts.append(f"Model: {s.model_name}")
        if s.feature_set:
            parts.append(f"Features: {s.feature_set}")
        if s.cv_r2_mean is not None:
            r2_str = f"CV R²: {s.cv_r2_mean:.4f}"
            if s.cv_r2_std:
                r2_str += f" ± {s.cv_r2_std:.4f}"
            parts.append(r2_str)
        if s.train_r2 is not None and s.cv_r2_mean is None:
            parts.append(f"Train R²: {s.train_r2:.4f}")
        if s.n_booked is not None:
            parts.append(f"Booked: {s.n_booked:,}")
        if s.n_demand is not None:
            parts.append(f"Demand: {s.n_demand:,}")
        if s.stress_factor is not None:
            parts.append(f"Stress: {s.stress_factor:.4f}")
        if s.tasa_fin_pct is not None:
            parts.append(f"Tasa fin: {s.tasa_fin_pct:.1f}%")
        if s.grid_dims:
            parts.append(f"Grid: {s.grid_dims}")
        if s.pareto_n_solutions is not None:
            parts.append(f"Pareto: {s.pareto_n_solutions} solutions")
        if s.b2_range_min is not None and s.b2_range_max is not None:
            parts.append(f"Risk range: [{s.b2_range_min:.2f}%, {s.b2_range_max:.2f}%]")
        if s.optimum_risk is not None:
            parts.append(f"Target risk: {s.optimum_risk}%")
        if s.n_grid_cells is not None and s.n_accepted_cells is not None:
            pct = s.n_accepted_cells / s.n_grid_cells * 100 if s.n_grid_cells else 0
            parts.append(f"Acceptance: {s.n_accepted_cells}/{s.n_grid_cells} cells ({pct:.0f}%)")
        if s.psi_value is not None:
            label = "stable" if s.psi_value < 0.1 else "moderate" if s.psi_value < 0.25 else "UNSTABLE"
            parts.append(f"PSI: {s.psi_value:.4f} ({label})")
        if s.ri_avg_multiplier is not None:
            ri_str = f"RI multiplier: avg={s.ri_avg_multiplier:.2f}×"
            if s.ri_max_multiplier is not None:
                ri_str += f", max={s.ri_max_multiplier:.2f}×"
            parts.append(ri_str)
        if s.mr_recal_avg_factor is not None:
            parts.append(f"MR recal factor: {s.mr_recal_avg_factor:.3f}")
        if s.elapsed_total is not None:
            parts.append(f"Time: {s.elapsed_total:.0f}s")
        elif s.elapsed_preprocessing or s.elapsed_inference or s.elapsed_optimization:
            phase_parts = []
            if s.elapsed_preprocessing:
                phase_parts.append(f"preproc={s.elapsed_preprocessing:.0f}s")
            if s.elapsed_inference:
                phase_parts.append(f"infer={s.elapsed_inference:.0f}s")
            if s.elapsed_optimization:
                phase_parts.append(f"optim={s.elapsed_optimization:.0f}s")
            parts.append(f"Time: {', '.join(phase_parts)}")
        if s.errors:
            parts.append(f"ERRORS: {len(s.errors)}")

        for part in parts:
            lines.append(f"  {part}")

        # Scenario results
        if s.scenarios:
            lines.append("  Scenarios:")
            for scen_name in ["pessimistic", "base", "optimistic"]:
                scen = s.scenarios.get(scen_name)
                if scen:
                    if scen.is_fallback and scen.selected_b2 is None:
                        lines.append(f"    {scen_name:12s}: NO FEASIBLE SOLUTION")
                    else:
                        flag = " [FALLBACK]" if scen.is_fallback else ""
                        prod_str = f"€{scen.production:,.0f}" if scen.production else "N/A"
                        lines.append(
                            f"    {scen_name:12s}: threshold={scen.risk_threshold:.2f}%  "
                            f"selected={scen.selected_b2:.2f}%  prod={prod_str}{flag}"
                        )

        # MR risk sources
        if s.mr_risk_sources:
            lines.append("  MR risk sources:")
            for src, count in sorted(s.mr_risk_sources.items(), key=lambda x: -x[1]):
                total = sum(s.mr_risk_sources.values())
                lines.append(f"    {src}: {count} bins ({count / total * 100:.0f}%)")

        # MR tier distribution
        if s.mr_tiers.total > 0:
            t = s.mr_tiers
            lines.append("  MR tier distribution:")
            lines.append(f"    Tier 1 (Actual H6):     {t.tier1_actual_h6:>8,} ({t.pct(1):.1f}%)")
            lines.append(f"    Tier 2 (Account H3→H6): {t.tier2_account_h3:>8,} ({t.pct(2):.1f}%)")
            lines.append(f"    Tier 3 (Bin-level main): {t.tier3_bin_main:>8,} ({t.pct(3):.1f}%)")
            lines.append(f"    Tier 4 (Model fallback): {t.tier4_model_fallback:>8,} ({t.pct(4):.1f}%)")

        # MR decomposition
        mr = s.mr_decomposition
        if mr.actual_risk is not None:
            lines.append("  MR Decomposition:")
            lines.append(
                f"    Actual:   risk={mr.actual_risk:.2f}%  prod=€{mr.actual_prod:,.0f}" if mr.actual_prod else ""
            )
            if mr.swap_out_risk is not None:
                lines.append(
                    f"    Swap-out: risk={mr.swap_out_risk:.2f}%  prod=€{mr.swap_out_prod:,.0f} ({mr.swap_out_pct:.1f}%)"
                )
            if mr.swap_in_risk is not None:
                lines.append(
                    f"    Swap-in:  risk={mr.swap_in_risk:.2f}%  prod=€{mr.swap_in_prod:,.0f} ({mr.swap_in_pct:.1f}%)"
                )
            if mr.optimum_risk is not None:
                lines.append(f"    Optimum:  risk={mr.optimum_risk:.2f}%  prod=€{mr.optimum_prod:,.0f}")

        lines.append("")

    # --- Cross-segment comparison table ---
    if len(real_segments) > 1:
        lines.append("=" * 80)
        lines.append("  CROSS-SEGMENT COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        # Header
        hdr = f"{'Segment':<22s} {'Booked':>8s} {'Target':>7s} {'Sel.b2':>7s} {'Prod(base)':>14s} {'Pareto':>7s} {'PSI':>7s} {'RI avg':>7s} {'Time':>6s}"
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for name, s in sorted(real_segments.items()):
            booked = f"{s.n_booked:,}" if s.n_booked else "?"
            target = f"{s.optimum_risk:.2f}%" if s.optimum_risk else "?"
            base = s.scenarios.get("base")
            sel_b2 = f"{base.selected_b2:.2f}%" if base and base.selected_b2 else "?"
            prod = f"€{base.production:,.0f}" if base and base.production else "?"
            pareto = str(s.pareto_n_solutions) if s.pareto_n_solutions else "?"
            psi = f"{s.psi_value:.4f}" if s.psi_value is not None else "?"
            ri = f"{s.ri_avg_multiplier:.2f}×" if s.ri_avg_multiplier is not None else "?"
            time_s = f"{s.elapsed_total:.0f}s" if s.elapsed_total else "?"
            lines.append(
                f"{name:<22s} {booked:>8s} {target:>7s} {sel_b2:>7s} {prod:>14s} {pareto:>7s} {psi:>7s} {ri:>7s} {time_s:>6s}"
            )

        lines.append("")

        # Flag segments with issues
        problem_segments = []
        for name, s in real_segments.items():
            issues = []
            if s.pareto_fallback_targets:
                issues.append("unreachable optimum_risk")
            if s.risk_inversion:
                issues.append("MR risk inversion")
            if s.psi_value is not None and s.psi_value > 0.25:
                issues.append(f"high PSI ({s.psi_value:.3f})")
            if s.n_booked is not None and s.n_booked < 500:
                issues.append(f"tiny sample ({s.n_booked})")
            if s.milp_infeasible:
                issues.append("MILP infeasible")
            if s.mr_low_risk_warning:
                issues.append("low MR risk")
            if s.errors:
                issues.append(f"{len(s.errors)} error(s)")
            if issues:
                problem_segments.append((name, issues))

        if problem_segments:
            lines.append("Segments needing attention:")
            for name, issues in problem_segments:
                lines.append(f"  {name}: {', '.join(issues)}")
            lines.append("")

    # --- Recommendations ---
    if not recommendations:
        lines.append("No recommendations — everything looks good.")
        lines.append("")
    else:
        recs_sorted = sorted(recommendations, key=lambda r: (PRIORITY_ORDER.get(r.priority, 9), r.category))

        lines.append("=" * 80)
        lines.append("  RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")

        n_high = sum(1 for r in recs_sorted if r.priority == "HIGH")
        n_med = sum(1 for r in recs_sorted if r.priority == "MEDIUM")
        n_low = sum(1 for r in recs_sorted if r.priority == "LOW")
        lines.append(f"  {n_high} high  |  {n_med} medium  |  {n_low} low")
        lines.append("")

        current_priority = None
        for r in recs_sorted:
            if r.priority != current_priority:
                current_priority = r.priority
                lines.append(f"--- {r.priority} priority ---")
                lines.append("")

            icon = PRIORITY_ICON.get(r.priority, "   ")
            lines.append(f"[{icon}] [{r.category}] {r.message}")
            if r.setting:
                lines.append(f"      Setting: {r.setting}")
            if r.current:
                lines.append(f"      Current: {r.current}")
            if r.suggested:
                lines.append(f"      Suggested: {r.suggested}")
            lines.append("")

    # --- Alternative segments.toml ---
    if alt_changes:
        lines.append("=" * 80)
        lines.append("  SUGGESTED CONFIGURATION CHANGES")
        lines.append("=" * 80)
        lines.append("")
        for change in alt_changes:
            lines.append(change)
        lines.append("")
        lines.append("An alternative segments.toml has been written to: segments_suggested.toml")
        lines.append("")

    # --- Verbose: show all warnings/errors ---
    if verbose:
        lines.append("=" * 80)
        lines.append("  ALL WARNINGS AND ERRORS")
        lines.append("=" * 80)
        lines.append("")
        for name, s in segments.items():
            if s.errors or s.warnings:
                lines.append(f"--- {name} ---")
                for err in s.errors:
                    lines.append(f"  [ERROR] {err[:200]}")
                for w in s.warnings[:50]:
                    lines.append(f"  [WARN]  {w[:200]}")
                if len(s.warnings) > 50:
                    lines.append(f"  ... and {len(s.warnings) - 50} more warnings")
                lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pipeline log files, diagnose issues, and suggest configuration improvements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python analyze_logs.py pipeline.log
  uv run python analyze_logs.py batch.log --verbose
  uv run python analyze_logs.py batch.log --segments-config segments.toml
  uv run python analyze_logs.py output/segment/logs/run_*.log -o report.txt
        """,
    )
    parser.add_argument("log_files", nargs="+", type=str, help="Log file(s) to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include all warnings and errors in report")
    parser.add_argument("--output", "-o", type=str, default=None, help="Write report to file instead of stdout")
    parser.add_argument(
        "--segments-config",
        type=str,
        default="segments.toml",
        help="Path to current segments.toml for generating alternative config (default: segments.toml)",
    )

    args = parser.parse_args()

    # Parse all log files
    all_entries = []
    for path_str in args.log_files:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        entries = parse_log_file(path)
        if not entries:
            print(f"Warning: No parseable log entries in: {path}", file=sys.stderr)
            continue
        all_entries.extend(entries)
        print(f"Parsed {len(entries)} log entries from {path}", file=sys.stderr)

    if not all_entries:
        print("Error: No log entries found in any input file.", file=sys.stderr)
        return 1

    print(f"Total: {len(all_entries)} log entries", file=sys.stderr)

    # Extract signals and generate recommendations
    segments, raw_errors = extract_signals(all_entries)
    recommendations = generate_recommendations(segments)

    # Generate alternative segments.toml
    seg_path = Path(args.segments_config)
    alt_toml, alt_changes = generate_alternative_segments_toml(
        segments,
        current_segments_path=seg_path if seg_path.exists() else None,
    )

    # Write alternative segments.toml if there are changes
    if alt_changes:
        alt_path = Path("segments_suggested.toml")
        alt_path.write_text(alt_toml)
        print(f"Alternative config written to {alt_path}", file=sys.stderr)

    # Format and output report
    report = format_report(
        segments,
        recommendations,
        verbose=args.verbose,
        alt_toml=alt_toml,
        alt_changes=alt_changes,
    )

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())

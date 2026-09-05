import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_SENSITIVITY_LEVELS

#: Keys that legitimately appear in a config's ``[preprocessing]`` section (or a frozen segment
#: config) but are consumed by OTHER components — batch orchestration and per-segment constraints —
#: not by :class:`PreprocessingSettings`. They are allowlisted so the unknown-key typo guard does
#: not flag them. Everything else that is not a model field is reported as a likely typo.
_KNOWN_EXTERNAL_CONFIG_KEYS = frozenset(
    {
        "cutoff_ordering_mode",  # batch-level; run_batch.py reads it from the raw TOML
        "cutoff_floor_segment",  # sequential cutoff ordering (segments.toml / batch)
        "min_risk",  # SegmentConstraints
        "max_risk",
        "min_production",
        "locked_sol_fac",
        "supersegment",  # supersegment resolution
        "modelling_supersegment",
        "reporting_supersegment",
    }
)


def _fs_safe(name: str) -> str:
    """Make a segment label safe to embed in a filename. Segment filters use the raw
    ``segment_cut_off`` value (e.g. ``direct/pl/known/nopremium/a-b``), whose ``/`` would
    otherwise be read as sub-directories that don't exist (trend/monthly-metrics writes crashed)."""
    return str(name).replace("/", "_").replace("\\", "_")


@dataclass
class OutputPaths:
    """Centralized output path configuration for the pipeline.

    All pipeline output files are written relative to ``base_dir``.
    Use the directory properties (``data_dir``, ``images_dir``, ``models_dir``)
    for ad-hoc paths, or the helper methods for well-known output files.
    """

    base_dir: Path = field(default_factory=lambda: Path("."))

    # -- directory roots --

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def images_dir(self) -> Path:
        return self.base_dir / "images"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    # -- preprocessing --

    @property
    def risk_vs_production_html(self) -> str:
        return str(self.images_dir / "risk_vs_production.html")

    @property
    def transformation_rate_html(self) -> str:
        return str(self.images_dir / "transformation_rate.html")

    # -- inference --

    @property
    def todu_avg_inference_html(self) -> str:
        return str(self.models_dir / "todu_avg_inference.html")

    @property
    def todu_model_joblib(self) -> str:
        return str(self.models_dir / "todu_model.joblib")

    @property
    def model_base_path(self) -> str:
        return str(self.models_dir)

    # -- optimization --

    @property
    def pareto_solutions_csv(self) -> str:
        return str(self.data_dir / "pareto_optimal_solutions.csv")

    def risk_production_visualizer_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"risk_production_visualizer{suffix}.html")

    def risk_production_summary_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"risk_production_summary_table{suffix}.csv")

    def data_summary_desagregado_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"data_summary_desagregado{suffix}.csv")

    def optimal_solution_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"optimal_solution{suffix}.csv")

    def efficient_frontier_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"efficient_frontier{suffix}.csv")

    @property
    def cutoff_summary_by_segment_csv(self) -> str:
        return str(self.data_dir / "cutoff_summary_by_segment.csv")

    @property
    def cutoff_summary_wide_csv(self) -> str:
        return str(self.data_dir / "cutoff_summary_wide.csv")

    def acceptance_grid_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"acceptance_grid{suffix}.html")

    # -- MR pipeline --

    def mr_summary_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"data_summary_desagregado_mr{suffix}.csv")

    def mr_b2_visualization_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"b2_ever_h6_vs_octroi_and_risk_score_mr{suffix}.html")

    def mr_cutoff_drift_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"mr_cutoff_drift{suffix}.html")

    def mr_cutoff_summary_wide_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"cutoff_summary_wide_mr{suffix}.csv")

    def mr_optimal_solution_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"optimal_solution_mr{suffix}.csv")

    def mr_risk_production_summary_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"risk_production_summary_table_mr{suffix}.csv")

    def mr_risk_comparison_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"mr_risk_comparison{suffix}.csv")

    def stability_report_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"stability_report{suffix}.html")

    def stability_psi_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"stability_psi{suffix}.csv")

    def drift_alerts_json(self, suffix: str = "") -> str:
        return str(self.data_dir / f"drift_alerts{suffix}.json")

    # -- sensitivity / marginal impact --

    def sensitivity_analysis_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"sensitivity_analysis{suffix}.csv")

    def cell_marginal_impact_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"cell_marginal_impact{suffix}.csv")

    def ri_optimizer_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"ri_optimizer_results{suffix}.csv")

    @property
    def cell_ci_csv(self) -> str:
        return str(self.models_dir / "cell_level_ci.csv")

    # -- trends --

    def monthly_metrics_csv(self, segment: str) -> str:
        return str(self.data_dir / f"monthly_metrics_{_fs_safe(segment)}.csv")

    def metric_trends_html(self, segment: str) -> str:
        return str(self.images_dir / f"metric_trends_{_fs_safe(segment)}.html")

    def trend_anomalies_csv(self, segment: str) -> str:
        return str(self.data_dir / f"trend_anomalies_{_fs_safe(segment)}.csv")

    # -- reporting --

    @property
    def segment_report_html(self) -> str:
        return str(self.base_dir / "report.html")

    # -- bin threshold diagnostics --

    def accepted_cells_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"accepted_cells{suffix}.csv")

    def bin_diagnostic_html(self, bin_col: str) -> str:
        return str(self.images_dir / f"bin_diagnostic_{bin_col}.html")

    # -- resimulation artifacts --

    @property
    def data_clean_parquet(self) -> str:
        return str(self.data_dir / "data_clean.parquet")

    @property
    def data_booked_parquet(self) -> str:
        return str(self.data_dir / "data_booked.parquet")

    @property
    def resimulation_meta_json(self) -> str:
        return str(self.data_dir / "resimulation_meta.json")

    @property
    def run_lineage_json(self) -> str:
        """Per-run data lineage / provenance artifact (M2)."""
        return str(self.data_dir / "run_lineage.json")

    @property
    def per_bin_stress_csv(self) -> str:
        return str(self.data_dir / "per_bin_stress.csv")

    @property
    def per_bin_tasa_fin_csv(self) -> str:
        return str(self.data_dir / "per_bin_tasa_fin.csv")

    # -- inference_optimized (main-period visualization) --

    @property
    def b2_visualization_html(self) -> str:
        return str(self.images_dir / "b2_ever_h6_vs_octroi_and_risk_score.html")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (self.data_dir, self.images_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class BinConfig:
    """Configuration for a single binning variable.

    Attributes:
        source_col: Raw column name in the data (e.g. 'score_rf').
        output_col: Name of the binned column created (e.g. 'sc_octroi_new_clus').
        bin_edges: Bin boundary values (must have >= 2 elements).
            When empty and ``max_bins`` is set, edges are learned via
            unsupervised quantiles on the demand population
            (``learn_quantile_bins``).
        max_bins: Optional maximum number of bins for edge learning.
            Only used when ``bin_edges`` is empty.
        method: Edge-learning method. ``"quantile"`` (default, unsupervised
            equal-count splits). ``"optimization"`` is deprecated — it leaked
            the risk target the optimizer maximizes and now falls back to
            quantile with a warning.
    """

    source_col: str
    output_col: str
    bin_edges: list[float] = field(default_factory=list)
    max_bins: int | None = None
    method: str = "quantile"

    def __post_init__(self) -> None:
        if self.bin_edges and len(self.bin_edges) < 2:
            raise ValueError(f"bin_edges for '{self.output_col}' must have at least 2 values")
        if self.bin_edges and len(self.bin_edges) >= 2:
            for i in range(1, len(self.bin_edges)):
                if self.bin_edges[i] <= self.bin_edges[i - 1]:
                    raise ValueError(
                        f"bin_edges for '{self.output_col}' must be in strictly ascending order, "
                        f"but found {self.bin_edges[i - 1]} >= {self.bin_edges[i]} at position {i - 1}-{i}"
                    )
        if not self.bin_edges and self.max_bins is None:
            raise ValueError(f"BinConfig for '{self.output_col}': either bin_edges or max_bins must be provided")


class PreprocessingSettings(BaseModel):
    """Configuration for preprocessing and overall pipeline settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_config_keys(cls, data: Any) -> Any:
        """Warn (loudly) about config keys that are neither a model field nor a known
        cross-component key (#63). Pydantic's default ``extra="ignore"`` otherwise silently drops a
        misspelled governance key (e.g. ``ri_optimizer_methods`` vs ``ri_optimizer_method``,
        ``mr_reoptimize_cutoffs``, ``dq_allow_warnings``), so the run proceeds on the default with no
        signal. We warn rather than hard-``forbid`` for back-compat (legit external keys share the
        section). The value is still dropped — only the silence is fixed."""
        if isinstance(data, dict):
            known = set(cls.model_fields.keys())
            for f in cls.model_fields.values():
                if getattr(f, "alias", None):
                    known.add(f.alias)
            unknown = sorted(k for k in data if k not in known and k not in _KNOWN_EXTERNAL_CONFIG_KEYS)
            if unknown:
                logger.warning(
                    f"Config: ignoring unrecognized key(s) {unknown} — likely a typo. A misspelled "
                    "key silently falls back to its default (no effect). Check the spelling against "
                    "the documented PreprocessingSettings fields."
                )
        return data

    # Required fields
    keep_vars: list[str]
    indicators: list[str]
    segment_filter: str = "unknown"
    date_ini_book_obs: str
    date_fin_book_obs: str
    variables: list[str]
    inference_variables: list[str] | None = None

    # N-variable binning config: maps variable output name -> BinConfig
    bins: dict[str, BinConfig] = Field(default_factory=dict)

    # Dictionary of explicit direction overrides mapping variable name to direction (1 or -1)
    # 1 indicates ascending risk (higher bin index = higher risk)
    # -1 indicates descending risk (higher bin index = lower risk)
    directions: dict[str, int] = Field(default_factory=dict)

    @field_validator("segment_filter")
    @classmethod
    def validate_segment_filter(cls, v: str) -> str:
        if v == "unknown":
            import warnings

            warnings.warn(
                "segment_filter is set to the default 'unknown'. This may filter out all data. "
                "Set an explicit segment_filter in your config.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("directions")
    @classmethod
    def validate_directions_values(cls, v: dict[str, int]) -> dict[str, int]:
        for var, direction in v.items():
            if direction not in (1, -1):
                raise ValueError(
                    f"directions['{var}'] must be 1 or -1, got {direction}. 1 = ascending risk, -1 = descending risk."
                )
        return v

    @field_validator("min_accepted_bin_by_variable")
    @classmethod
    def validate_min_accepted_bin_values(
        cls, v: dict[str, float | dict[str | int | float, float]]
    ) -> dict[str, float | dict[str | int | float, float]]:
        for var, threshold in v.items():
            if not isinstance(var, str) or not var:
                raise ValueError("min_accepted_bin_by_variable keys must be non-empty variable names")
            if isinstance(threshold, dict):
                if not threshold:
                    raise ValueError(f"min_accepted_bin_by_variable['{var}'] cannot be an empty dict")
                for income_bin_value, th in threshold.items():
                    if th is None:
                        raise ValueError(f"min_accepted_bin_by_variable['{var}']['{income_bin_value}'] cannot be null")
                    try:
                        float(th)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"min_accepted_bin_by_variable['{var}']['{income_bin_value}'] must be numeric, got {th}"
                        ) from e
            else:
                if threshold is None:
                    raise ValueError(f"min_accepted_bin_by_variable['{var}'] cannot be null")
                try:
                    float(threshold)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"min_accepted_bin_by_variable['{var}'] must be numeric, got {threshold}") from e
        return v

    # Inversion flags populated during preprocessing based on directions
    inv_vars: list[str] = Field(default_factory=list)

    # Legacy fields — still accepted for backward compatibility.
    # Prefer using ``bins`` dict for new configs.
    octroi_bins: list[float] = Field(default_factory=list)
    efx_bins: list[float] = Field(default_factory=list)

    # Optional fields with defaults
    date_ini_book_obs_mr: str | None = None
    date_fin_book_obs_mr: str | None = None
    score_measures: list[str] | None = None
    data_path: str = "data/demanda_direct_out.sas7bdat"
    n_months: int = 12
    # FIXED accounting constants, not tuning knobs. `todu_amt_pile_hN` is the SUM of
    # outstanding over months H0..HN, so dividing by the month count gives the average
    # monthly outstanding: b2_ever_h6 = 7·Σ(todu_30ever_h6)/Σ(todu_amt_pile_h6).
    # multiplier = 7 because H0..H6 inclusive = 7 months; multiplier_h3 = 4 (H0..H3).
    # These must match how the pile columns are summed — do NOT vary them to "tune"
    # risk (a config-sensitivity sweep will show huge impact precisely because changing
    # them mis-scales the denominator; that's a guardrail signal, not a degree of freedom).
    multiplier: float = Field(default=7.0, gt=0)
    multiplier_h3: float = Field(default=4.0, gt=0)
    # Modified-z threshold for target-based outlier WINSORIZATION in the risk fit
    # (#56: extreme-risk bins are clipped to the ±threshold boundary, not dropped).
    # 0 disables it entirely; smaller = more aggressive clipping of the tail.
    z_threshold: float = Field(default=3.0, ge=0)
    cv_folds: int = Field(default=4, ge=2, le=10)
    # Expert / default-off (audit #6). When True, a meaningful two-part HurdleRegressor is offered
    # as a model candidate, trained on PER-LOAN data (real zero mass in the default indicator) and
    # scored on the same bin-level CV RMSE as the other candidates. When False (default) the hurdle
    # is not offered at all — on the bin-aggregated target it degenerates to plain Ridge/Lasso, so
    # offering it added a misleading "distinct model" and two wasted Optuna tunings. Enabling this
    # can change the selected risk model and therefore the cutoffs (M1 validation required).
    model_hurdle_per_loan: bool = Field(default=False)
    optimum_risk: float = Field(default=1.1, gt=0, le=100)
    risk_step: float = Field(default=0.1, gt=0, le=50)
    cz_config: dict[int, Any] = Field(default_factory=dict)
    log_level: str = "INFO"
    # SAS file encoding for pd.read_sas (audit #19). Default "latin-1" (the verified main-path
    # value); used by both data_manager.load_data and run_batch's preloader so the two read sites
    # never diverge.
    sas_encoding: str = "latin-1"
    # Data-quality strictness (audit #18, M2). Fail-closed by default: when False (default), any DQ
    # warning (e.g. coverage gaps, outliers, small segments, booked-ratio 0.01-0.05) HALTS the run, on
    # top of the FAILED-severity checks (negative counts/amounts, unparseable dates, booked-ratio <0.01)
    # that always halt. Relax with the `--allow-dq-warnings` CLI flag or `dq_allow_warnings=true` in
    # config (proceed past soft warnings); use `--skip-dq-checks` to skip DQ entirely.
    dq_allow_warnings: bool = False
    fixed_cutoffs: dict[str, Any] | None = None
    baseline_mode: bool = False
    base_scenario_only: bool = False
    cutoff_floor_segment: str | None = None
    # Per-variable minimum accepted bin thresholds.
    # Value can be a scalar (applies to all rows) or an income_bin-keyed map.
    # Cells with value < resolved threshold are forced rejected in optimization.
    min_accepted_bin_by_variable: dict[str, float | dict[str | int | float, float]] = Field(default_factory=dict)
    # Note: cutoff_ordering_mode is a batch-level setting read from the raw TOML
    # dict in run_batch.py (not from this model), since it controls cross-segment
    # orchestration rather than per-segment behavior.

    # Sensitivity analysis
    run_sensitivity: bool = False
    sensitivity_levels: list[float] = DEFAULT_SENSITIVITY_LEVELS

    # Reject inference parameter optimization
    run_ri_optimizer: bool = False
    ri_uplift_range: list[float] = [0.0, 5.0]
    ri_max_mult_range: list[float] = [1.0, 5.0]
    ri_uplift_steps: int = 11
    ri_max_mult_steps: int = 9

    # Hybrid MR risk inference
    use_mr_outcomes: bool = False
    mr_min_obs_per_bin: int = Field(default=30, ge=1)
    mr_extrapolation_method: Literal["linear", "power", "logistic", "auto"] = "linear"
    mr_extrapolation_curvature: float = Field(default=1.0, ge=0.3, le=5.0)
    mr_extrapolation_risk_multiplier: float = Field(default=3.0, gt=0, le=10.0)
    mr_extrapolation_hard_cap: float = Field(default=15.0, gt=0, le=100.0)
    mr_maturity_months: int = Field(
        default=6,
        ge=0,
        le=24,
        description=(
            "Minimum months since booking for an MR account to count as mature H6. "
            "Accounts booked more recently are excluded from b2_mr to avoid diluting "
            "risk with immature zeros. Set to 0 to disable maturity filtering."
        ),
    )
    # MR cutoff re-optimization (Expert, default True = legacy behavior):
    # True  — the inline MR check RE-OPTIMIZES the acceptance mask via MILP on the
    #         recalibrated MR risk surface (targeting the main b2). The MR cutoffs
    #         can therefore drift from the main period; MR risk lands ≈ the target
    #         by construction (it is optimized to it).
    # False — the MR period KEEPS the main (frozen) mask and only recomputes
    #         risk/production on those cells — the honest "how does my chosen
    #         policy do on the more recent cohort" view (same basis as the M4
    #         backtest, but with H3→H6 extrapolation instead of full maturity).
    # FIXED-CUTOFF segments are ALWAYS frozen in MR regardless of this flag — a
    # deliberately-fixed policy must stay fixed; the flag only governs the
    # MILP-optimized segments. Flipping it changes MR headline numbers, so it is
    # governance-relevant (validate before relying on it for sign-off).
    mr_reoptimize_cutoffs: bool = Field(default=True)

    # Swap-in (repesca) constraints for MILP optimization
    max_swapin_production_pct: float | None = Field(default=None, ge=0, le=100)
    max_swapin_risk: float | None = Field(default=None, ge=0, le=100)

    # MILP / Pareto / bootstrap tuning
    milp_time_limit: float = Field(default=30.0, gt=0, description="MILP solver time limit in seconds")
    pareto_n_points: int = Field(default=50, ge=5, le=500, description="Number of risk targets in Pareto sweep")
    n_bootstraps: int = Field(
        default=DEFAULT_N_BOOTSTRAPS, ge=100, le=50000, description="Bootstrap replicates for CI estimation"
    )
    # Scenario selection basis (audit #28 Phase C — Expert, default "point"):
    # "point"    — classic rule: max production whose POINT-estimate risk <= target.
    # "ci_upper" — noise-margin rule: max production whose bootstrap risk CI
    #              UPPER bound <= target (per-candidate CI from the selection-aware
    #              bootstrap). More conservative by construction (less production);
    #              enabling it changes cutoffs and needs M5-style sign-off.
    selection_risk_basis: Literal["point", "ci_upper"] = "point"
    # Uncertainty-aware monotonicity relaxation (optional):
    # keep strict monotonicity by default; when enabled, skip local adjacency
    # constraints for pairs that are both sparse and statistically ambiguous.
    monotonicity_relaxation_enabled: bool = False
    monotonicity_uncertainty_min_exposure: float = Field(default=0.0, ge=0.0)
    monotonicity_uncertainty_z_threshold: float = Field(default=1.0, ge=0.0)

    # --- Knob tiers (M3 config-complexity audit; see todo-list.md M3) ---
    # The defaults below ARE the simple/safe path: every fragile feature defaults
    # off (RI "none", optimizer off, MR off, smoothing off, decay None, monotonicity
    # strict, extrapolation "linear", gamma 1.0). The complexity lives in config.toml,
    # not here. Sensitivity ranking (no_premium_cd): stress_mode ±29%, multiplier ±22%,
    # run_ri_optimizer toggle ±19%, ri_calibration_gamma ±24% (only when optimizer on),
    # parceling_method ±10%. NOTE: with run_ri_optimizer=true the manual
    # reject_uplift_factor / reject_max_risk_multiplier are OVERRIDDEN (inert); the
    # active RI levers are then ri_calibration_gamma + ri_*_range. MR knobs are
    # validation-only and do not affect cutoffs (see M3a).

    # Reject inference settings
    reject_inference_method: Literal["none", "parceling"] = "none"
    reject_parceling_method: Literal["linear", "power", "sigmoid"] = "linear"
    reject_uplift_factor: float = Field(default=1.5, ge=0.0, le=10.0)
    reject_max_risk_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    reject_bayesian_smoothing: bool = False
    reject_bayesian_prior_strength: float = Field(default=10.0, gt=0, le=1000)
    reject_enforce_monotonicity: bool = False
    # Deprecated and ignored: reject inference always uses score-only acceptance
    # rates (the swap-in/repesca population is solely score-rejected). Setting
    # True logs a one-time deprecation warning and has no effect.
    reject_include_all_rejections: bool = False
    # Time-awareness for reject inference acceptance rates (selection bias)
    # When enabled, acceptance rates are computed on a more recent demand
    # subset or with exponential time-decay weights based on `mis_date`.
    reject_acceptance_recent_months: int | None = Field(
        default=None, ge=1, description="If set, compute RI acceptance rates using only the last N months."
    )
    reject_acceptance_decay_half_life_months: float | None = Field(
        default=None,
        gt=0,
        description=(
            "If set, apply exponential time-decay weights to RI acceptance rates using this half-life (months). "
            "Takes precedence over reject_acceptance_recent_months."
        ),
    )
    reject_acceptance_date_col: str = Field(
        default="mis_date",
        min_length=1,
        description="Date column used for temporal weighting/windowing in reject inference acceptance-rate estimation.",
    )
    # Default to False to avoid imposing the (often unstable) booked H6/H3 ratio
    # assumption onto the rejected/re-predicted H3 numerator.
    reject_apply_h3_multiplier: bool = False
    # No/low-demand repesca bins (highest selection-bias uncertainty) have their acceptance
    # rate shrunk toward a conservative LOW anchor by confidence, instead of the old
    # anti-conservative median fallback (audit #5). Anchor = this percentile of observed
    # rates; confidence = 1 - exp(-n / scale) (smaller scale ⇒ only sparse bins shrink).
    reject_no_demand_anchor_percentile: float = Field(default=0.10, ge=0.0, le=0.5)
    reject_confidence_scale: float = Field(default=10.0, gt=0.0, le=1000.0)
    ri_calibration_gamma: float = Field(default=1.0, gt=0, le=1)
    ri_optimizer_method: Literal["grid", "optuna"] = "grid"
    ri_optuna_n_trials: int = Field(default=100, ge=10, le=10000)
    ri_validation_split: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of main-period months used for RI optimizer training. "
            "The remaining months are held out for out-of-time validation. "
            "Both splits have fully mature H6 outcomes. Set to 1.0 to disable "
            "(uses all data for training, no validation)."
        ),
    )

    # Stress factor mode: "global" (legacy single scalar), "disabled" (always 1.0),
    # "per_bin" (per-bin computation). When parceling is active, "disabled" avoids
    # double-counting selection bias with the RI multiplier.
    stress_mode: Literal["global", "disabled", "per_bin"] = "global"

    # Per-bin transformation rate: when True, compute tasa_fin per bin combination
    # instead of a single global scalar.  Bins with insufficient data fall back to
    # the global rate.
    per_bin_tasa_fin: bool = False

    @field_validator("keep_vars", "indicators")
    @classmethod
    def validate_non_empty_list(cls, v: list[str], info: Any) -> list[str]:
        if not v:
            raise ValueError(f"'{info.field_name}' must be a non-empty list")
        return v

    @field_validator("variables")
    @classmethod
    def validate_variables_length(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError(f"'variables' must contain at least 1 element, got {len(v)}")
        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(f"'variables' must contain unique elements, found duplicates: {set(duplicates)}")
        return v

    @field_validator("octroi_bins", "efx_bins")
    @classmethod
    def validate_bins_length(cls, v: list[float], info: Any) -> list[float]:
        if v and len(v) < 2:
            raise ValueError(f"'{info.field_name}' must have at least 2 values")
        if v and len(v) >= 2:
            # Validate ascending sort order (allowing -inf at start and inf at end)
            for i in range(1, len(v)):
                if v[i] <= v[i - 1]:
                    raise ValueError(
                        f"'{info.field_name}' must be in strictly ascending order, "
                        f"but found {v[i - 1]} >= {v[i]} at position {i - 1}-{i}"
                    )
        return v

    @field_validator("ri_uplift_range", "ri_max_mult_range")
    @classmethod
    def validate_ri_range(cls, v: list[float], info: Any) -> list[float]:
        if len(v) != 2:
            raise ValueError(f"'{info.field_name}' must have exactly 2 elements [min, max], got {len(v)}")
        if v[0] >= v[1]:
            raise ValueError(f"'{info.field_name}' min ({v[0]}) must be less than max ({v[1]})")
        return v

    @field_validator("date_ini_book_obs", "date_fin_book_obs", "date_ini_book_obs_mr", "date_fin_book_obs_mr")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            # Enforce dayfirst=False to match original implicit behavior but be explicit to suppress warning
            pd.to_datetime(v, dayfirst=False)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date format: {v}. Error: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "PreprocessingSettings":
        # Main period validation
        # Field validators ensure these are valid date strings/can be parsed.
        # We parse them again here to compare.
        try:
            start = pd.to_datetime(self.date_ini_book_obs, dayfirst=False)
            end = pd.to_datetime(self.date_fin_book_obs, dayfirst=False)
        except (ValueError, TypeError):
            # Should not happen if field validators pass, but if it does,
            # we can't validate the range, so we rely on field validators to have caught format errors.
            return self

        if start > end:
            raise ValueError(
                f"Invalid main observation period: start date ({start.date()}) is after end date ({end.date()})"
            )

        # MR period validation check consistency
        has_mr_ini = self.date_ini_book_obs_mr is not None
        has_mr_fin = self.date_fin_book_obs_mr is not None

        if has_mr_ini != has_mr_fin:
            provided = "date_ini_book_obs_mr" if has_mr_ini else "date_fin_book_obs_mr"
            missing = "date_fin_book_obs_mr" if has_mr_ini else "date_ini_book_obs_mr"
            raise ValueError(
                f"Partial MR date configuration: '{provided}' is set but '{missing}' is missing. "
                f"Provide both MR dates or neither."
            )

        if has_mr_ini and has_mr_fin:
            start_mr = pd.to_datetime(self.date_ini_book_obs_mr, dayfirst=False)
            end_mr = pd.to_datetime(self.date_fin_book_obs_mr, dayfirst=False)
            if start_mr > end_mr:
                raise ValueError(
                    f"Invalid MR period: start date ({start_mr.date()}) is after end date ({end_mr.date()})"
                )
            # Warn if main and MR periods overlap (data leakage risk). Use <= because the
            # pipeline date filters are INCLUSIVE on both ends: start_mr == end means the
            # shared boundary day is counted in BOTH periods (a one-day leak), which the old
            # strict `<` let through unwarned.
            if start_mr <= end:
                import warnings

                warnings.warn(
                    f"Main period ({start.date()} to {end.date()}) overlaps with MR period "
                    f"({start_mr.date()} to {end_mr.date()}). This may introduce data leakage.",
                    UserWarning,
                    stacklevel=2,
                )

        return self

    @model_validator(mode="after")
    def _validate_baseline_vs_fixed_cutoffs(self) -> "PreprocessingSettings":
        """Prevent setting both baseline_mode and fixed_cutoffs."""
        if self.baseline_mode and self.fixed_cutoffs:
            raise ValueError(
                "Cannot set both baseline_mode=True and fixed_cutoffs. "
                "baseline_mode shows the current portfolio; fixed_cutoffs applies explicit cuts."
            )
        return self

    def get_date(self, field: str) -> pd.Timestamp:
        val = getattr(self, field)
        return pd.to_datetime(val, dayfirst=False)

    @model_validator(mode="after")
    def _auto_populate_bins(self) -> "PreprocessingSettings":
        """Auto-populate ``bins`` dict from legacy octroi_bins/efx_bins if not set.

        todo #66: emit a DeprecationWarning whenever the legacy fields are
        consulted. The `[preprocessing.bins.*]` TOML path is the supported
        N-variable configuration; ``octroi_bins`` / ``efx_bins`` + their
        hardcoded ``score_rf`` / ``risk_score_rf`` source columns encode a
        2-variable assumption that the N-D pipeline has outgrown. Migration
        helper in ``scripts/migrate_legacy_bins.py``.
        """
        import warnings

        # Known legacy variable → bins/source_col mapping
        _legacy_map = {
            "sc_octroi_new_clus": (self.octroi_bins, "score_rf"),
            "new_efx_clus": (self.efx_bins, "risk_score_rf"),
        }

        _legacy_used: list[str] = []  # names of variables populated via the legacy path

        # Merge legacy octroi_bins/efx_bins into bins dict for variables not already defined
        for var in self.variables:
            if var in self.bins:
                continue
            # Try name-based match first
            if var in _legacy_map:
                edges, src = _legacy_map[var]
                if edges:
                    self.bins[var] = BinConfig(source_col=src, output_col=var, bin_edges=edges)
                    _legacy_used.append(f"{var} (name-match → {src})")
                    continue
            # Positional fallback: var0 → octroi_bins, var1 → efx_bins
            idx = self.variables.index(var)
            if idx == 0 and self.octroi_bins:
                self.bins[var] = BinConfig(source_col="score_rf", output_col=var, bin_edges=self.octroi_bins)
                _legacy_used.append(f"{var} (positional var0 → score_rf)")
            elif idx == 1 and self.efx_bins:
                self.bins[var] = BinConfig(source_col="risk_score_rf", output_col=var, bin_edges=self.efx_bins)
                _legacy_used.append(f"{var} (positional var1 → risk_score_rf)")

        if _legacy_used:
            warnings.warn(
                "Legacy 'octroi_bins' / 'efx_bins' config fields are deprecated and will be "
                "removed in a future release. The following variables were populated via the "
                f"legacy path: {_legacy_used}. Migrate to the explicit "
                "[preprocessing.bins.VAR] TOML blocks — each variable gets a source_col, "
                "output_col, and bin_edges (or max_bins + method). "
                "See scripts/migrate_legacy_bins.py for an automated rewrite.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not self.bins:
            raise ValueError(
                "No binning configuration provided. Set either 'bins' dict or "
                "both 'octroi_bins' and 'efx_bins' in your configuration."
            )

        # Every variable must have a corresponding bins entry
        missing_bins = [v for v in self.variables if v not in self.bins]
        if missing_bins:
            raise ValueError(
                f"Variables {missing_bins} have no binning configuration. "
                f"Provide bin edges via 'bins' dict or legacy 'octroi_bins'/'efx_bins'."
            )

        return self

    @model_validator(mode="after")
    def _default_inference_variables(self) -> "PreprocessingSettings":
        """Default ``inference_variables`` to ``variables`` and validate."""
        if self.inference_variables is None:
            self.inference_variables = list(self.variables)
        if len(self.inference_variables) < 1:
            raise ValueError(
                f"'inference_variables' must contain at least 1 element, got {len(self.inference_variables)}"
            )
        if not set(self.inference_variables).issubset(set(self.variables)):
            extra = set(self.inference_variables) - set(self.variables)
            raise ValueError(f"'inference_variables' must be a subset of 'variables', found extra: {extra}")
        return self

    @model_validator(mode="after")
    def _validate_min_accepted_bin_vars(self) -> "PreprocessingSettings":
        if not self.min_accepted_bin_by_variable:
            return self
        unknown = [v for v in self.min_accepted_bin_by_variable if v not in self.variables]
        if unknown:
            raise ValueError(
                "min_accepted_bin_by_variable contains variables not present in 'variables': "
                f"{unknown}. Allowed: {self.variables}"
            )
        return self

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> "PreprocessingSettings":
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

        # Extract the 'preprocessing' section as in the original code
        prep_config = config_data.get("preprocessing", {})

        # Handle cz_config keys (convert string keys to int)
        if "cz_config" in prep_config:
            prep_config["cz_config"] = {int(k): v for k, v in prep_config["cz_config"].items()}

        # Convert new-style TOML bins section into BinConfig objects
        if "bins" in prep_config and isinstance(prep_config["bins"], dict):
            converted_bins: dict[str, BinConfig] = {}
            for var_name, bin_data in prep_config["bins"].items():
                converted_bins[var_name] = BinConfig(**bin_data)
            prep_config["bins"] = converted_bins

        return cls(**prep_config)

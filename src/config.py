import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    # -- MR pipeline --

    def mr_summary_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"data_summary_desagregado_mr{suffix}.csv")

    def mr_b2_visualization_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"b2_ever_h6_vs_octroi_and_risk_score_mr{suffix}.html")

    def mr_risk_production_summary_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"risk_production_summary_table_mr{suffix}.csv")

    def stability_report_html(self, suffix: str = "") -> str:
        return str(self.images_dir / f"stability_report{suffix}.html")

    def stability_psi_csv(self, suffix: str = "") -> str:
        return str(self.data_dir / f"stability_psi{suffix}.csv")

    def drift_alerts_json(self, suffix: str = "") -> str:
        return str(self.data_dir / f"drift_alerts{suffix}.json")

    # -- trends --

    def monthly_metrics_csv(self, segment: str) -> str:
        return str(self.data_dir / f"monthly_metrics_{segment}.csv")

    def metric_trends_html(self, segment: str) -> str:
        return str(self.images_dir / f"metric_trends_{segment}.html")

    def trend_anomalies_csv(self, segment: str) -> str:
        return str(self.data_dir / f"trend_anomalies_{segment}.csv")

    # -- inference_optimized (main-period visualization) --

    @property
    def b2_visualization_html(self) -> str:
        return str(self.images_dir / "b2_ever_h6_vs_octroi_and_risk_score.html")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (self.data_dir, self.images_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


class PreprocessingSettings(BaseModel):
    """Configuration for preprocessing and overall pipeline settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required fields
    keep_vars: list[str]
    indicators: list[str]
    segment_filter: str = "unknown"
    octroi_bins: list[float]
    efx_bins: list[float]
    date_ini_book_obs: str
    date_fin_book_obs: str
    variables: list[str]

    # Optional fields with defaults
    date_ini_book_obs_mr: str | None = None
    date_fin_book_obs_mr: str | None = None
    score_measures: list[str] | None = None
    data_path: str = "data/demanda_direct_out.sas7bdat"
    n_months: int = 12
    multiplier: float = Field(default=7.0, gt=0)
    z_threshold: float = Field(default=3.0, gt=0)
    optimum_risk: float = 1.1
    risk_step: float = 0.1
    cz_config: dict[int, Any] = Field(default_factory=dict)
    log_level: str = "INFO"
    fixed_cutoffs: dict[str, Any] | None = None
    inv_var1: bool = False

    # Reject inference settings
    reject_inference_method: Literal["none", "parceling"] = "none"
    reject_uplift_factor: float = Field(default=1.5, ge=0.0, le=10.0)
    reject_max_risk_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)

    @field_validator("keep_vars", "indicators")
    @classmethod
    def validate_non_empty_list(cls, v: list[str], info: Any) -> list[str]:
        if not v:
            raise ValueError(f"'{info.field_name}' must be a non-empty list")
        return v

    @field_validator("variables")
    @classmethod
    def validate_variables_length(cls, v: list[str]) -> list[str]:
        if len(v) != 2:
            raise ValueError(f"'variables' must contain exactly 2 elements, got {len(v)}")
        return v

    @field_validator("octroi_bins", "efx_bins")
    @classmethod
    def validate_bins_length(cls, v: list[float], info: Any) -> list[float]:
        if len(v) < 2:
            raise ValueError(f"'{info.field_name}' must have at least 2 values")
        # Convert infs here if needed, or keeping it raw and converting later.
        # Original code used convert_bins to handle inf strings in python/json,
        # but here we expect floats (including float('inf')).
        # We'll allow standard floats.
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

        return self

    def get_date(self, field: str) -> pd.Timestamp:
        val = getattr(self, field)
        return pd.to_datetime(val, dayfirst=False)

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> "PreprocessingSettings":
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

        # Extract the 'preprocessing' section as in the original code
        prep_config = config_data.get("preprocessing", {})

        # Handle cz_config keys (convert string keys to int)
        if "cz_config" in prep_config:
            prep_config["cz_config"] = {int(k): v for k, v in prep_config["cz_config"].items()}

        # Handle infinite values in bins for pydantic validation if they come as strings
        # (Though TOML handles floats, sometimes users might put strings for inf)
        # We assume standard TOML floats or compatible types.

        return cls(**prep_config)

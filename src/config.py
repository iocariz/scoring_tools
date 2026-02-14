import tomllib
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    scenario_step: float = 0.1
    cz_config: dict[int, Any] = Field(default_factory=dict)
    log_level: str = "INFO"
    fixed_cutoffs: dict[str, Any] | None = None
    inv_var1: bool = False

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
            # In pydantic v2 model validators we can just log a warning via standard logging
            # or we can decide to be stricter. The original code only warned.
            # For a "Settings" class, simple consistency is better.
            # Let's verify if we want to enforce both or neither.
            # The original code just appended a warning.
            # We will leave it as is but note it.
            pass

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

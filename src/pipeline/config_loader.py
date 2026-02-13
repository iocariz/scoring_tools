import pandas as pd
from loguru import logger

from src.config import PreprocessingSettings
from src.utils import calculate_annual_coef


def load_and_validate_config(config_path: str) -> tuple[PreprocessingSettings, pd.Timestamp, pd.Timestamp, float]:
    """Load TOML config, validate it using Pydantic, parse dates, and compute annual coefficient.

    Args:
        config_path: Path to the configuration TOML file

    Returns:
        Tuple of (settings, date_ini, date_fin, annual_coef)

    Raises:
        ValidationError: If configuration validation fails
    """
    settings = PreprocessingSettings.from_toml(config_path)
    logger.debug(f"Configuration loaded from {config_path}")

    # Dates are already validated by Pydantic model
    date_ini = settings.get_date("date_ini_book_obs")
    date_fin = settings.get_date("date_fin_book_obs")

    annual_coef = calculate_annual_coef(date_ini, date_fin)

    segment = settings.segment_filter
    logger.info(
        f"Config validated | segment={segment} | "
        f"period={date_ini.date()} to {date_fin.date()} | annual_coef={annual_coef:.2f}"
    )

    return settings, date_ini, date_fin, annual_coef

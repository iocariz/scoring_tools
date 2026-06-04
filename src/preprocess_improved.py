"""
Enhanced data preprocessing module for credit risk scoring pipeline.

This module provides robust preprocessing functions with improved error handling,
performance optimizations, and comprehensive logging.
"""

import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.constants import DEFAULT_RANDOM_STATE, Columns, RejectReason, StatusName

if TYPE_CHECKING:
    from src.config import BinConfig, PreprocessingSettings


def log_dataframe_stats(df: pd.DataFrame, name: str) -> None:
    """
    Log comprehensive statistics about a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to log information about
    name : str
        Name of the DataFrame for reference
    """
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {df.columns.tolist()}")

    missing_values = df.isna().sum().sum()
    logger.info(f"{name} missing values: {missing_values}")

    if missing_values > 0:
        missing_per_col = df.isna().sum()
        missing_cols = missing_per_col[missing_per_col > 0]
        logger.warning(f"{name} columns with missing values:\n{missing_cols}")

    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"{name} memory usage: {memory_mb:.2f} MB")

    # Structural summary only — never log row-level data (PII risk in --log-file)
    if not df.empty:
        dtype_summary = df.dtypes.value_counts().to_dict()
        logger.debug(f"{name} shape: {df.shape}, dtype counts: {dtype_summary}")


def validate_dataframe_columns(df: pd.DataFrame, required_columns: list[str], operation: str) -> None:
    """
    Validate that DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
    operation : str
        Name of operation for error message

    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{operation}: Missing required columns: {missing_cols}")


def preprocess_data(
    df: pd.DataFrame, keep_vars: list[str], indicators: list[str], segment_filter: str = "loan_known_ab"
) -> pd.DataFrame:
    """
    Preprocess data with consistent filtering and transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    keep_vars : List[str]
        List of variables to keep
    indicators : List[str]
        List of indicator columns for calculations
    segment_filter : str, optional
        Segment filter criteria (default='loan_known_ab')

    Returns
    -------
    pd.DataFrame
        Processed data

    Raises
    ------
    ValueError
        If required columns are missing or data is invalid
    """
    start_time = time.time()
    logger.info(f"Starting data preprocessing with {df.shape[0]:,} records")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Validate required columns for filtering
    required_filter_cols = ["fuera_norma", "fraud_flag", "nature_holder", "segment_cut_off"]
    validate_dataframe_columns(df, required_filter_cols, "preprocess_data")

    # Collect all measure columns (m_ct_direct*)
    m_ct_direct_columns = [col for col in df.columns if col.startswith("m_ct_direct")]
    logger.info(f"Found {len(m_ct_direct_columns)} measure columns starting with 'm_ct_direct'")

    # Validate that requested columns exist (H3 columns are optional)
    h3_optional = {Columns.TODU_30EVER_H3, Columns.TODU_AMT_PILE_H3}
    all_requested_columns = list(dict.fromkeys(keep_vars + indicators + m_ct_direct_columns))
    required_columns = [c for c in all_requested_columns if c not in h3_optional]
    validate_dataframe_columns(df, required_columns, "preprocess_data")

    # Apply filter conditions in a single query operation
    query_start = time.time()
    logger.info(
        f"Applying filters: fuera_norma='n', fraud_flag='n', nature_holder!='legal', segment_cut_off='{segment_filter}'"
    )

    try:
        # First apply non-segment filters using explicit boolean masks
        mask = (df["fuera_norma"] == "n") & (df["fraud_flag"] == "n") & (df["nature_holder"] != "legal")
        data_filtered = df[mask].copy()

        # Apply segment filter - support both exact match and regex patterns
        # If segment_filter contains '|' (OR operator), treat as regex pattern
        if "|" in segment_filter:
            # Regex pattern for multiple segments (e.g., supersegment combinations)
            logger.info("Using regex pattern matching for segment_filter")
            segment_mask = data_filtered["segment_cut_off"].astype(str).str.match(segment_filter, na=False)
            data_filtered = data_filtered[segment_mask]
        else:
            # Exact match for single segment
            data_filtered = data_filtered[data_filtered["segment_cut_off"] == segment_filter]
    except (KeyError, pd.errors.UndefinedVariableError) as e:
        logger.error(f"Error applying filters: {e}")
        raise

    filter_time = time.time() - query_start
    records_removed = df.shape[0] - data_filtered.shape[0]
    logger.info(
        f"Filter applied in {filter_time:.2f}s. "
        f"Removed {records_removed:,} records ({records_removed / df.shape[0]:.1%})"
    )
    logger.info(f"{data_filtered.shape[0]:,} records remaining")

    if data_filtered.empty:
        raise ValueError("No records remain after filtering")

    # Select only needed columns (filter out optional columns not present in data)
    select_columns = [c for c in all_requested_columns if c in data_filtered.columns]
    logger.info(f"Selecting {len(select_columns)} columns")
    data_clean = data_filtered[select_columns].copy()

    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    log_dataframe_stats(data_clean, "Preprocessed data")

    return data_clean


def _generate_bin_summary(data: pd.DataFrame, bin_col: str, source_col: str) -> pd.DataFrame:
    """
    Generate a summary table showing bin statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Data with binned column
    bin_col : str
        Name of the binned column (e.g., 'sc_octroi_new_clus')
    source_col : str
        Name of the source column used for binning (e.g., 'score_rf')

    Returns
    -------
    pd.DataFrame
        Summary with columns: bin, min, max, count
    """
    summary = data.groupby(bin_col)[source_col].agg(["min", "max", "count"]).reset_index()
    summary.columns = ["bin", "min", "max", "count"]
    summary = summary.sort_values("bin")
    return summary


def learn_quantile_bins(
    data: pd.DataFrame,
    source_col: str = "income_t1_m",
    max_bins: int = 2,
) -> list[float]:
    """Learn bin edges from quantiles of the source column (unsupervised).

    Uses equally-spaced quantiles so each bin has roughly the same number of
    records.  This avoids using the target variable for feature engineering
    and eliminates the risk of circularity / overfitting.

    Parameters
    ----------
    data : pd.DataFrame
        Population to compute quantiles on.  This should be the **demand**
        population (all applications: booked + rejected + canceled) that the
        bins are later applied to — not booked-only.  Booked is a selected,
        risk-truncated subset, so quantiles computed on it would not yield
        equal counts on the graded population, and rejects (which feed reject
        inference) would be binned by edges that never saw their distribution.
    source_col : str
        Raw column name (default ``"income_t1_m"``).
    max_bins : int
        Number of bins (2 or 3).

    Returns
    -------
    list[float]
        Bin edges suitable for ``pd.cut``, e.g. ``[-inf, median, inf]``.

    Raises
    ------
    ValueError
        If required column is missing or has too few distinct values.
    """
    validate_dataframe_columns(data, [source_col], "learn_quantile_bins")

    n_before = len(data)
    valid = data[source_col].dropna()
    n_dropped = n_before - len(valid)
    if n_dropped > 0:
        logger.info(
            f"learn_quantile_bins: dropped {n_dropped:,} NaN rows ({n_dropped / n_before:.1%}) from '{source_col}'"
        )
    if len(valid) < max_bins * 2:
        raise ValueError(f"learn_quantile_bins: only {len(valid)} valid records for '{source_col}'")

    # Compute quantile edges (e.g. 2 bins → median, 3 bins → terciles)
    quantiles = [i / max_bins for i in range(1, max_bins)]
    thresholds = sorted(valid.quantile(quantiles).unique().tolist())

    if not thresholds:
        raise ValueError(
            f"learn_quantile_bins: degenerate distribution for '{source_col}' "
            f"— quantile splits produced no unique thresholds."
        )

    edges: list[float] = [-np.inf] + thresholds + [np.inf]
    logger.info(
        f"learn_quantile_bins: {len(thresholds)} split(s) for '{source_col}' "
        f"(max_bins={max_bins}): thresholds={thresholds}"
    )
    logger.info(f"  Bin edges: {edges}")
    return edges


def learn_income_bins(
    data_booked: pd.DataFrame,
    source_col: str = "income_t1_m",
    target_col: str = "early_bad",
    max_bins: int = 2,
    min_samples_leaf: int = 500,
) -> list[float]:
    """Learn optimal bin edges for an income variable using a supervised decision tree.

    .. deprecated::
        Use :func:`learn_quantile_bins` instead to avoid target-based feature
        engineering circularity.  This function is kept for backward
        compatibility and can be invoked explicitly via ``bin_method = "supervised"``.

    Fits a ``DecisionTreeClassifier`` on *booked* data to find the split thresholds
    that best separate ``target_col`` (typically ``early_bad``).

    Parameters
    ----------
    data_booked : pd.DataFrame
        Booked population with both ``source_col`` and ``target_col``.
    source_col : str
        Raw income column name (default ``"income_t1_m"``).
    target_col : str
        Binary target column (default ``"early_bad"``).
    max_bins : int
        Maximum number of bins (2 or 3). The tree is fitted with
        ``max_leaf_nodes=max_bins``.
    min_samples_leaf : int
        Minimum samples per leaf node (guards against tiny bins).

    Returns
    -------
    list[float]
        Bin edges suitable for ``pd.cut``, e.g. ``[-inf, threshold, inf]``.

    Raises
    ------
    ValueError
        If required columns are missing or the tree produces no valid splits.
    """
    validate_dataframe_columns(data_booked, [source_col, target_col], "learn_income_bins")

    n_before = len(data_booked)
    valid = data_booked[[source_col, target_col]].dropna()
    n_dropped = n_before - len(valid)
    if n_dropped > 0:
        logger.info(
            f"learn_income_bins: dropped {n_dropped:,} NaN rows ({n_dropped / n_before:.1%}) from '{source_col}'/'{target_col}'"
        )
    if len(valid) < min_samples_leaf * 2:
        raise ValueError(f"learn_income_bins: only {len(valid)} valid records — need at least {min_samples_leaf * 2}")

    X = valid[[source_col]].values
    y = valid[target_col].values.astype(int)

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=DEFAULT_RANDOM_STATE,
    )
    tree.fit(X, y)

    # Extract split thresholds from tree internals
    thresholds = sorted(
        {t for t in tree.tree_.threshold if t != -2.0}  # -2.0 = leaf node sentinel
    )

    if not thresholds:
        raise ValueError(
            f"learn_income_bins: tree produced no splits for '{source_col}' → '{target_col}'. "
            "Check data quality or reduce min_samples_leaf."
        )

    edges = [-np.inf] + thresholds + [np.inf]
    logger.info(
        f"learn_income_bins: {len(thresholds)} split(s) found for '{source_col}' "
        f"(max_bins={max_bins}): thresholds={thresholds}"
    )
    logger.info(f"  Bin edges: {edges}")
    return edges


def learn_optimization_bins(
    data_booked: pd.DataFrame,
    source_col: str = "income_t1_m",
    target_col: str = "early_bad",
    weight_col: str = "oa_amt_h0",
    max_bins: int = 2,
    min_samples_leaf: int = 500,
) -> list[float]:
    """Learn bin edges that maximize production-weighted risk differentiation.

    .. deprecated::
        No longer used by the pipeline.  This method fits bin edges on the
        same risk target (``early_bad``) that the downstream optimizer later
        maximizes (``b2_ever_h6``), so the bins are tuned to in-sample risk
        separation and the resulting risk/production frontier looks better than
        it generalizes (target leakage / circularity).  The pipeline now always
        uses unsupervised :func:`learn_quantile_bins`; configs requesting
        ``method = "optimization"`` fall back to quantile with a warning.  This
        function is retained for backward compatibility only.

    Fits a ``DecisionTreeRegressor`` on *booked* data with ``sample_weight``
    set to the production column (``oa_amt_h0``), so the split maximizes the
    production-weighted variance reduction in the risk target.  This gives the
    MILP optimizer maximal leverage from the income dimension.

    Parameters
    ----------
    data_booked : pd.DataFrame
        Booked population with ``source_col``, ``target_col``, and optionally
        ``weight_col``.
    source_col : str
        Raw column name (default ``"income_t1_m"``).
    target_col : str
        Risk target column (default ``"early_bad"``).
    weight_col : str
        Production weight column (default ``"oa_amt_h0"``).
    max_bins : int
        Maximum number of bins (default 2).
    min_samples_leaf : int
        Minimum samples per leaf node (default 500).

    Returns
    -------
    list[float]
        Bin edges suitable for ``pd.cut``, e.g. ``[-inf, threshold, inf]``.

    Raises
    ------
    ValueError
        If required columns are missing or the tree produces no valid splits.
    """
    validate_dataframe_columns(data_booked, [source_col, target_col], "learn_optimization_bins")

    n_before = len(data_booked)
    valid = data_booked[[source_col, target_col]].dropna()
    n_dropped = n_before - len(valid)
    if n_dropped > 0:
        logger.info(
            f"learn_optimization_bins: dropped {n_dropped:,} NaN rows ({n_dropped / n_before:.1%}) from '{source_col}'/'{target_col}'"
        )

    # Determine sample weights
    weights = None
    if weight_col in data_booked.columns:
        weight_values = data_booked.loc[valid.index, weight_col].fillna(0.0)
        if weight_values.sum() > 0:
            weights = weight_values.values
            logger.info(f"learn_optimization_bins: using production weights from '{weight_col}'")
        else:
            logger.warning(f"learn_optimization_bins: '{weight_col}' is all zeros, falling back to unweighted")
    else:
        logger.warning(f"learn_optimization_bins: '{weight_col}' not found, falling back to unweighted")

    if len(valid) < min_samples_leaf * 2:
        raise ValueError(
            f"learn_optimization_bins: only {len(valid)} valid records — need at least {min_samples_leaf * 2}"
        )

    X = valid[[source_col]].values
    y = valid[target_col].values.astype(float)

    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=DEFAULT_RANDOM_STATE,
    )
    tree.fit(X, y, sample_weight=weights)

    # Extract split thresholds from tree internals
    thresholds = sorted(
        {t for t in tree.tree_.threshold if t != -2.0}  # -2.0 = leaf node sentinel
    )

    if not thresholds:
        raise ValueError(
            f"learn_optimization_bins: tree produced no splits for '{source_col}' → '{target_col}'. "
            "Check data quality or reduce min_samples_leaf."
        )

    edges: list[float] = [-np.inf] + thresholds + [np.inf]
    logger.info(
        f"learn_optimization_bins: {len(thresholds)} split(s) found for '{source_col}' "
        f"(max_bins={max_bins}): thresholds={thresholds}"
    )
    logger.info(f"  Bin edges: {edges}")

    # Holdout stability validation: check that bin edges differentiate risk on unseen data
    if len(valid) >= min_samples_leaf * 3:
        from sklearn.model_selection import train_test_split

        train_fold, holdout_fold = train_test_split(valid, test_size=0.2, random_state=DEFAULT_RANDOM_STATE + 1)
        train_binned = pd.cut(train_fold[source_col], bins=edges, labels=False)
        holdout_binned = pd.cut(holdout_fold[source_col], bins=edges, labels=False)
        train_risk = train_fold.groupby(train_binned, observed=True)[target_col].mean()
        holdout_risk = holdout_fold.groupby(holdout_binned, observed=True)[target_col].mean()
        logger.info(
            f"  Bin stability: train_risk={train_risk.round(4).to_dict()}, holdout_risk={holdout_risk.round(4).to_dict()}"
        )
        if len(train_risk) >= 2 and len(holdout_risk) >= 2:
            train_diff = float(train_risk.iloc[-1] - train_risk.iloc[0])
            holdout_diff = float(holdout_risk.iloc[-1] - holdout_risk.iloc[0])
            if train_diff * holdout_diff < 0:
                logger.warning("  Bin edges may be overfit: risk ordering reversed on holdout fold")
            else:
                logger.info("  Bin stability OK: risk ordering consistent between train and holdout")

    return edges


def assess_binning_gini(
    data: pd.DataFrame,
    raw_col: str,
    binned_col: str,
    target_col: str,
) -> dict[str, float]:
    """Compare discriminatory power of a raw continuous column vs its binned version.

    Computes the Gini coefficient (2 * AUC - 1) for both ``raw_col`` and
    ``binned_col`` against ``target_col``, and reports the retention percentage.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing all three columns.
    raw_col : str
        Name of the raw continuous column (e.g. ``"income_t1_m"``).
    binned_col : str
        Name of the binned column (e.g. ``"income_bin"``).
    target_col : str
        Binary target column (e.g. ``"early_bad"``).

    Returns
    -------
    dict[str, float]
        ``{"gini_raw": ..., "gini_binned": ..., "gini_retention_pct": ...}``
    """
    validate_dataframe_columns(data, [raw_col, binned_col, target_col], "assess_binning_gini")

    n_before = len(data)
    valid = data[[raw_col, binned_col, target_col]].dropna()
    n_dropped = n_before - len(valid)
    if n_dropped > 0:
        logger.info(f"assess_binning_gini: dropped {n_dropped:,} NaN rows ({n_dropped / n_before:.1%})")
    y = valid[target_col].values.astype(int)

    if len(np.unique(y)) < 2:
        logger.warning("assess_binning_gini: target has a single class — Gini is undefined")
        return {"gini_raw": 0.0, "gini_binned": 0.0, "gini_retention_pct": 0.0}

    auc_raw = roc_auc_score(y, valid[raw_col].values)
    gini_raw = 2 * auc_raw - 1

    auc_binned = roc_auc_score(y, valid[binned_col].values)
    gini_binned = 2 * auc_binned - 1

    retention = (gini_binned / gini_raw * 100) if gini_raw != 0 else 0.0

    logger.info(
        f"assess_binning_gini: raw Gini={gini_raw:.4f}, binned Gini={gini_binned:.4f}, retention={retention:.1f}%"
    )
    return {"gini_raw": gini_raw, "gini_binned": gini_binned, "gini_retention_pct": retention}


def apply_binning_transformations(
    data: pd.DataFrame,
    octroi_bins: list[float] | None = None,
    efx_bins: list[float] | None = None,
    *,
    bins_config: dict[str, "BinConfig"] | None = None,
) -> pd.DataFrame:
    """
    Apply binning transformations to the data.

    Supports both the legacy 2-variable interface (octroi_bins/efx_bins) and
    the new N-variable interface (bins_config dict).

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    octroi_bins : list[float], optional
        Legacy bin edges for octroi score (mutually exclusive with bins_config)
    efx_bins : list[float], optional
        Legacy bin edges for efx (risk) score (mutually exclusive with bins_config)
    bins_config : dict[str, BinConfig], optional
        N-variable binning configuration mapping output column names to BinConfig.

    Returns
    -------
    pd.DataFrame
        Data with binned variables

    Raises
    ------
    ValueError
        If required columns are missing or binning fails
    """
    if bins_config is not None:
        return _apply_binning_from_config(data, bins_config)

    # Legacy path — original 2-variable behavior
    if octroi_bins is None or efx_bins is None:
        raise ValueError("Either bins_config or both octroi_bins/efx_bins must be provided")

    from src.config import BinConfig

    legacy_config = {
        "sc_octroi_new_clus": BinConfig(
            source_col="score_rf",
            output_col="sc_octroi_new_clus",
            bin_edges=octroi_bins,
        ),
        "new_efx_clus": BinConfig(
            source_col="risk_score_rf",
            output_col="new_efx_clus",
            bin_edges=efx_bins,
        ),
    }
    return _apply_binning_from_config(data, legacy_config)


def _apply_binning_from_config(
    data: pd.DataFrame,
    bins_config: dict[str, "BinConfig"],
) -> pd.DataFrame:
    """Apply binning for each variable defined in bins_config."""
    start_time = time.time()
    logger.info(f"Starting binning transformations for {len(bins_config)} variable(s)")

    # Validate required source columns
    source_cols = [bc.source_col for bc in bins_config.values()]
    validate_dataframe_columns(data, source_cols, "apply_binning_transformations")

    transformed_data = data.copy()

    for _var_name, bc in bins_config.items():
        logger.info(f"Applying binning '{bc.output_col}' from '{bc.source_col}' with {len(bc.bin_edges)} edges")
        logger.info(f"  Bin edges: {bc.bin_edges}")

        if len(bc.bin_edges) < 2:
            raise ValueError(f"{bc.output_col}: bin_edges must have at least 2 values")

        try:
            transformed_data[bc.output_col] = pd.cut(
                transformed_data[bc.source_col],
                bins=bc.bin_edges,
                labels=False,
                include_lowest=True,
            )

            nan_count = transformed_data[bc.output_col].isna().sum()
            if nan_count > 0:
                nan_pct = nan_count / len(transformed_data)
                src = transformed_data[bc.source_col]
                src_min, src_max = src.min(), src.max()
                lo, hi = min(bc.bin_edges), max(bc.bin_edges)
                # Distinguish out-of-range (snapped into a boundary bin, KEPT — these contaminate the
                # boundary cell that feeds the MILP grid) from true-NaN-source (dropped). NaN source
                # values match neither mask. Flag each explicitly (audit #21).
                nan_mask = transformed_data[bc.output_col].isna()
                below_mask = nan_mask & (src < lo)
                above_mask = nan_mask & (src >= hi)
                n_below = int(below_mask.sum())
                n_above = int(above_mask.sum())
                n_nan_src = int(nan_count - n_below - n_above)
                logger.warning(
                    f"{bc.output_col}: {nan_count:,} records ({nan_pct:.1%}) unbinned — "
                    f"{n_below:,} below-range + {n_above:,} above-range snapped to boundary bins, "
                    f"{n_nan_src:,} NaN-source dropped. "
                    f"{bc.source_col} range: [{src_min}, {src_max}], bin_edges: [{lo}, {hi}]"
                )
                if nan_pct > 0.01:
                    raise ValueError(
                        f"{nan_pct:.1%} of records fall outside {bc.output_col} bin range — exceeds 1% threshold. "
                        f"Check bin_edges configuration or data quality."
                    )
                # Assign out-of-range records to the nearest boundary bin instead of dropping.
                n_bins = len(bc.bin_edges) - 1
                transformed_data.loc[below_mask, bc.output_col] = 0  # first bin (0-indexed)
                transformed_data.loc[above_mask, bc.output_col] = n_bins - 1  # last bin (0-indexed)
                # Any remaining NaNs (NaN source values) are dropped.
                if n_nan_src > 0:
                    logger.info(
                        f"Dropping {n_nan_src:,} records with NaN source values in '{bc.output_col}' "
                        f"({n_nan_src / len(transformed_data):.1%} of data)"
                    )
                    transformed_data = transformed_data.dropna(subset=[bc.output_col])

            # 1-indexed bins
            transformed_data[bc.output_col] = transformed_data[bc.output_col] + 1

        except (ValueError, KeyError) as e:
            logger.error(f"Error in {bc.output_col} binning: {e}")
            raise

        # Log bin summary
        logger.info("=" * 60)
        logger.info(f"BIN SUMMARY: {bc.output_col} (from {bc.source_col})")
        logger.info("=" * 60)
        summary = _generate_bin_summary(transformed_data, bin_col=bc.output_col, source_col=bc.source_col)
        logger.info(f"\n{summary.to_string(index=False)}")

    elapsed_time = time.time() - start_time
    logger.info(f"Binning transformations completed in {elapsed_time:.2f} seconds")

    return transformed_data


def update_oa_amt_h0(data: pd.DataFrame) -> pd.DataFrame:
    """
    Update oa_amt_h0 for records not booked, setting it to oa_amt.

    Parameters
    ----------
    data : pd.DataFrame
        Input data

    Returns
    -------
    pd.DataFrame
        Data with updated oa_amt_h0

    Raises
    ------
    ValueError
        If required columns are missing
    """
    start_time = time.time()
    logger.info("Updating oa_amt_h0 values for non-booked records")

    # Validate required columns
    validate_dataframe_columns(data, ["status_name", "oa_amt", "oa_amt_h0"], "update_oa_amt_h0")

    result = data.copy()

    # Count records to be updated
    update_mask = result["status_name"] != StatusName.BOOKED.value
    update_count = update_mask.sum()
    logger.info(
        f"Updating {update_count:,} records where status_name != 'booked' ({update_count / len(result):.1%} of data)"
    )

    # Use vectorized operation
    result.loc[update_mask, "oa_amt_h0"] = result.loc[update_mask, "oa_amt"]

    # Verify update
    updated_count = (result.loc[update_mask, "oa_amt_h0"] == result.loc[update_mask, "oa_amt"]).sum()
    logger.info(f"Successfully updated {updated_count:,} records")

    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"oa_amt_h0 update completed in {elapsed_time:.2f} seconds")

    return result


def filter_by_date(data: pd.DataFrame, date_field: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter data by date range.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    date_field : str
        Name of the date field to filter on
    start_date : str
        Start date (inclusive)
    end_date : str
        End date (inclusive)

    Returns
    -------
    pd.DataFrame
        Filtered data

    Raises
    ------
    ValueError
        If date_field is missing or date conversion fails
    """
    start_time = time.time()
    logger.info(f"Filtering data by date range: {start_date} to {end_date}")

    validate_dataframe_columns(data, [date_field], "filter_by_date")

    original_record_count = len(data)
    result = data.copy()

    # Convert dates if they aren't already datetime
    if not pd.api.types.is_datetime64_any_dtype(result[date_field]):
        logger.info(f"Converting {date_field} to datetime")
        try:
            result[date_field] = pd.to_datetime(result[date_field], dayfirst=False)
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting {date_field} to datetime: {e}")
            raise ValueError(f"Cannot convert {date_field} to datetime: {e}") from e

    # Apply date filter
    try:
        start_date_dt = pd.to_datetime(start_date, dayfirst=False)
        end_date_dt = pd.to_datetime(end_date, dayfirst=False)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid date format: {e}") from e

    if start_date_dt > end_date_dt:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    mask = (result[date_field] >= start_date_dt) & (result[date_field] <= end_date_dt)
    filtered_data = result[mask].copy()

    # Log results
    filtered_record_count = len(filtered_data)
    removed_record_count = original_record_count - filtered_record_count

    if original_record_count > 0:
        pct_removed = removed_record_count / original_record_count
        logger.info(f"Date filter removed {removed_record_count:,} records ({pct_removed:.1%} of data)")
    logger.info(f"Remaining records: {filtered_record_count:,}")

    # Check for date distribution
    if not filtered_data.empty:
        min_date = filtered_data[date_field].min()
        max_date = filtered_data[date_field].max()
        logger.info(f"Date range in filtered data: {min_date.date()} to {max_date.date()}")

        # Log monthly distribution
        monthly_counts = filtered_data[date_field].dt.to_period("M").value_counts().sort_index()
        logger.info(f"Monthly distribution:\n{monthly_counts}")
    else:
        logger.warning("No records remain after date filtering")

    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Date filtering completed in {elapsed_time:.2f} seconds")

    return filtered_data


def update_status_and_reject_reason(data: pd.DataFrame, score_measures: list[str] | None = None) -> pd.DataFrame:
    """
    Update status_name and reject_reason based on direct measures.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    score_measures : List[str], optional
        List of score measure columns

    Returns
    -------
    pd.DataFrame
        Data with updated status_name and reject_reason

    Raises
    ------
    ValueError
        If required columns are missing
    """
    start_time = time.time()
    logger.info("Updating status_name and reject_reason")

    validate_dataframe_columns(data, ["status_name", "reject_reason"], "update_status_and_reject_reason")

    result = data.copy()

    # Get all direct measure columns
    m_ct_cols = [col for col in result.columns if col.startswith("m_ct_direct")]
    logger.info(f"Found {len(m_ct_cols)} direct measure columns")

    if not m_ct_cols:
        logger.warning("No m_ct_direct* columns found, skipping measure-based updates")
        return result

    # Create a mask for any 'y' in direct measure columns
    measure_mask = result[m_ct_cols].eq("y").any(axis=1)
    measure_count = measure_mask.sum()
    logger.info(
        f"Found {measure_count:,} records with 'y' in direct measure columns "
        f"({measure_count / len(result):.1%} of data)"
    )

    # Update status_name and reject_reason based on the mask
    result.loc[measure_mask, "status_name"] = StatusName.REJECTED.value
    result.loc[measure_mask, "reject_reason"] = RejectReason.OTHER.value

    # Update reject_reason for score measures if provided
    if score_measures:
        logger.info(f"Updating reject_reason for score measures: {score_measures}")
        validate_dataframe_columns(result, score_measures, "update_status_and_reject_reason")

        score_measure_mask = result[score_measures].eq("y").any(axis=1)
        score_measure_count = score_measure_mask.sum()
        logger.info(
            f"Found {score_measure_count:,} records with 'y' in score measure columns "
            f"({score_measure_count / len(result):.1%} of data)"
        )
        result.loc[score_measure_mask, "reject_reason"] = RejectReason.SCORE.value

    # Log status name distribution after update
    status_counts = result["status_name"].value_counts()
    logger.info(f"status_name distribution after update:\n{status_counts}")

    # Log reject reason distribution after update
    reject_counts = result["reject_reason"].value_counts()
    logger.info(f"reject_reason distribution after update:\n{reject_counts}")

    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Status and reject reason update completed in {elapsed_time:.2f} seconds")

    return result


_pipeline_stdout_sink_id: int | None = None


def _configure_pipeline_logging(log_level: str) -> None:
    """Replace the default stderr handler with a stdout handler at the given level.

    Only removes the default sink (id=0) on first call, then tracks and
    replaces its own stdout sink on subsequent calls.  This avoids
    stripping file sinks added by callers (e.g. ``--log-file`` or
    per-segment log files).
    """
    global _pipeline_stdout_sink_id

    # First call: remove the default stderr sink
    try:
        logger.remove(0)
    except ValueError:
        pass

    # Subsequent calls: remove the previous stdout sink we added
    if _pipeline_stdout_sink_id is not None:
        try:
            logger.remove(_pipeline_stdout_sink_id)
        except ValueError:
            pass

    _pipeline_stdout_sink_id = logger.add(sys.stdout, level=log_level)


def _run_data_transformations(df: pd.DataFrame, settings: "PreprocessingSettings") -> pd.DataFrame:
    """Run preprocess_data, apply_binning, update_oa_amt_h0, update_status."""
    from src.config import BinConfig

    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Basic filtering")
    logger.info("=" * 80)
    data_clean = preprocess_data(df, settings.keep_vars, settings.indicators, settings.segment_filter)

    # IMPORTANT: status_name ("booked" vs non-booked) drives bin-edge learning
    # (booked mask) and oa_amt_h0 updates. update_status_and_reject_reason can
    # relabel records, so apply it before learning any bin edges to avoid
    # learning on an inconsistent "booked" definition.
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Update oa_amt_h0 (before bin edge learning)")
    logger.info("=" * 80)
    data_clean = update_oa_amt_h0(data_clean)

    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Update status and reject reasons (before bin edge learning)")
    logger.info("=" * 80)
    data_clean = update_status_and_reject_reason(data_clean, settings.score_measures)

    if settings.bins:
        from src.constants import StatusName

        # Learn bin edges for any BinConfig with max_bins but no bin_edges.
        # Bins are learned (unsupervised quantiles) on the date-filtered DEMAND
        # population — the same population they are applied to — so the
        # equal-count guarantee holds on the graded population and rejects
        # (which feed reject inference) are binned by edges that saw their
        # distribution.  Learning on booked-only would be a selected, truncated
        # subset.  Supervised ("optimization") binning is deprecated because it
        # leaks the risk target the optimizer later maximizes.
        data_demand_for_bins = data_clean
        if settings.date_ini_book_obs and settings.date_fin_book_obs:
            data_demand_for_bins = filter_by_date(
                data_clean, "mis_date", settings.date_ini_book_obs, settings.date_fin_book_obs
            )

        for var_name, bc in settings.bins.items():
            if not bc.bin_edges and bc.max_bins is not None:
                if bc.method == "optimization":
                    logger.warning(
                        f"Bin method 'optimization' for '{var_name}' is deprecated (target leakage); "
                        f"falling back to quantile splits."
                    )
                logger.info(f"Learning bin edges for '{var_name}' using quantile splits (max_bins={bc.max_bins})")
                learned_edges = learn_quantile_bins(
                    data_demand_for_bins,
                    source_col=bc.source_col,
                    max_bins=bc.max_bins,
                )
                # Replace BinConfig with one that has learned edges
                settings.bins[var_name] = BinConfig(
                    source_col=bc.source_col,
                    output_col=bc.output_col,
                    bin_edges=learned_edges,
                    max_bins=bc.max_bins,
                )

        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Binning transformations")
        logger.info("=" * 80)
        data_clean = apply_binning_transformations(data_clean, bins_config=settings.bins)

        # Gini assessment for variables that used supervised edge learning
        for var_name, bc in settings.bins.items():
            if bc.max_bins is not None and bc.source_col in data_clean.columns and bc.output_col in data_clean.columns:
                booked_mask = data_clean["status_name"] == StatusName.BOOKED.value
                data_booked_subset = data_clean[booked_mask]
                if "early_bad" in data_booked_subset.columns and data_booked_subset["early_bad"].nunique() > 1:
                    gini_result = assess_binning_gini(
                        data_booked_subset, raw_col=bc.source_col, binned_col=bc.output_col, target_col="early_bad"
                    )
                    logger.info(
                        f"Binning Gini assessment for '{var_name}': "
                        f"raw={gini_result['gini_raw']:.4f}, "
                        f"binned={gini_result['gini_binned']:.4f}, "
                        f"retention={gini_result['gini_retention_pct']:.1f}%"
                    )
    elif settings.octroi_bins and settings.efx_bins:
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Binning transformations (legacy)")
        logger.info("=" * 80)
        data_clean = apply_binning_transformations(data_clean, settings.octroi_bins, settings.efx_bins)
    else:
        logger.warning("Skipping binning transformations (bins not provided)")

    return data_clean


def _filter_booked_for_period(
    data_clean: pd.DataFrame,
    date_ini: str | None,
    date_fin: str | None,
) -> pd.DataFrame:
    """Filter booked status and date range."""
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Filter booked data by date")
    logger.info("=" * 80)
    data_booked = data_clean[data_clean["status_name"] == StatusName.BOOKED.value].copy()
    logger.info(f"Found {data_booked.shape[0]:,} records with status_name='booked'")

    if date_ini and date_fin:
        data_booked = filter_by_date(data_booked, "mis_date", date_ini, date_fin)
    else:
        logger.warning("Date filtering skipped for booked data (dates not provided)")

    return data_booked


def _filter_demand_for_period(
    data_clean: pd.DataFrame,
    date_ini: str | None,
    date_fin: str | None,
) -> pd.DataFrame:
    """Filter demand data by date range."""
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Filter demand data by date")
    logger.info("=" * 80)

    if date_ini and date_fin:
        data_demand = filter_by_date(data_clean.copy(), "mis_date", date_ini, date_fin)
    else:
        logger.warning("Date filtering skipped for demand data (dates not provided)")
        data_demand = data_clean.copy()

    return data_demand


def _infer_direction_from_bins(
    bin_vals,
    risk_vals,
    weights,
    *,
    n_permutations: int = 1000,
    seed: int = DEFAULT_RANDOM_STATE,
) -> tuple[int, float, float]:
    """Infer the monotonicity direction of risk vs bin index (audit #16).

    Returns ``(direction, rho_w, p_value)`` where ``direction`` is +1 (ascending risk: higher bin
    = riskier) or -1 (descending: higher bin = safer).

    The direction is **always** the sign of the production-weighted rank correlation ``rho_w``
    between bin index and per-bin risk — it is never overridden by significance, so a genuinely
    descending-but-noisy score is inferred descending instead of being forced to ascending. The
    permutation ``p_value`` is reported for **confidence only** (a caller may warn when it is large);
    it is calibrated by permuting the ``(risk_rank, weight)`` pairs **jointly** so the weight travels
    with its risk (an exchangeable null). When there is no monotone signal (``rho_w`` NaN/≈0 — a flat
    risk profile), ``direction`` falls back to the documented default ``+1`` with ``p_value = 1.0``.
    """
    from scipy.stats import rankdata

    rank_x = rankdata(bin_vals)
    rank_y = rankdata(risk_vals)
    weights = np.asarray(weights, dtype=float)

    mean_x = np.average(rank_x, weights=weights)
    mean_y = np.average(rank_y, weights=weights)
    dx = rank_x - mean_x
    dy = rank_y - mean_y
    cov_xy = np.average(dx * dy, weights=weights)
    var_x = np.average(dx * dx, weights=weights)
    var_y = np.average(dy * dy, weights=weights)

    # Coherent production-weighted rank correlation (weighted Pearson-on-ranks ⇒ weighted Spearman).
    if var_x <= 0 or var_y <= 0:
        rho_w = float("nan")
    else:
        rho_w = float(cov_xy / np.sqrt(var_x * var_y))

    # No monotone signal: flat/degenerate risk profile → documented default, no confidence.
    if np.isnan(rho_w) or np.isclose(rho_w, 0.0):
        return 1, rho_w, 1.0

    direction = 1 if rho_w > 0 else -1

    # Calibrated permutation null (confidence only — does NOT change `direction`): permute the
    # (risk_rank, weight) pairs jointly so each risk keeps its own production weight.
    rng = np.random.RandomState(seed)
    abs_cov_obs = abs(cov_xy)
    n = len(rank_x)
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        perm_y = rank_y[perm]
        perm_w = weights[perm]
        pmx = np.average(rank_x, weights=perm_w)
        pmy = np.average(perm_y, weights=perm_w)
        perm_cov = np.average((rank_x - pmx) * (perm_y - pmy), weights=perm_w)
        if abs(perm_cov) >= abs_cov_obs:
            exceed += 1
    p_value = (exceed + 1) / (n_permutations + 1)
    return direction, rho_w, p_value


def _infer_monotonicity(data: pd.DataFrame, settings: "PreprocessingSettings") -> None:
    """
    Infer the monotonicity of binned variables with respect to the risk target.
    Updates settings.directions and settings.inv_vars in place.

    1 = ascending risk (higher bin = riskier)
    -1 = descending risk (higher bin = safer)
    """
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Infer dynamic monotonicity for variables (Weighted)")
    logger.info("=" * 80)

    import numpy as np

    from src.utils import calculate_b2_ever_h6

    for var in settings.variables:
        if var in settings.directions:
            direction = settings.directions[var]
            logger.info(f"[{var}] Using explicitly configured direction: {direction}")
        else:
            # Drop unbinned/missing records for inference
            valid_data = data[[var, "todu_30ever_h6", "todu_amt_pile_h6", "oa_amt"]].dropna()

            # Calculate B2 risk empirically per bin on the booked records
            aggs = valid_data.groupby(var, observed=True)[["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt"]].sum()
            b2_risks = calculate_b2_ever_h6(aggs["todu_30ever_h6"], aggs["todu_amt_pile_h6"])

            valid_idx = ~b2_risks.isna()
            if sum(valid_idx) < 2:
                logger.warning(f"[{var}] Not enough valid risk data to infer direction. Defaulting to ascending (1).")
                direction = 1
            else:
                bin_vals = aggs.index[valid_idx].values
                risk_vals = b2_risks[valid_idx].values
                weights = aggs["oa_amt"][valid_idx].values

                # Direction = sign of the production-weighted rank correlation, ALWAYS (audit #16);
                # the permutation p-value is confidence-only and never overrides the empirical sign.
                direction, rho_w, p_value = _infer_direction_from_bins(bin_vals, risk_vals, weights)

                if np.isnan(rho_w) or np.isclose(rho_w, 0.0):
                    logger.warning(
                        f"[{var}] No production-weighted monotone signal (flat risk profile). "
                        f"Defaulting to ascending (1) — set the direction explicitly in config."
                    )
                elif p_value > 0.10:
                    logger.warning(
                        f"[{var}] Weak monotone signal (rho_w: {rho_w:.3f}, perm_p: {p_value:.3f}); "
                        f"using the empirical sign direction: {direction}. Consider setting it explicitly in config."
                    )
                else:
                    logger.info(f"[{var}] Inferred direction: {direction} (rho_w: {rho_w:.3f}, perm_p: {p_value:.3f}).")

            settings.directions[var] = direction

        # Update inv_vars mapping for the optimization engine
        # In the optimization code, if a variable is in inv_vars, it means higher bin index = SAFER
        # So if direction == -1 (descending risk, higher bin = safer), it should be IN inv_vars
        if direction == -1 and var not in settings.inv_vars:
            settings.inv_vars.append(var)
            logger.info(f"[{var}] Added to inv_vars list for optimization monotonically descending constraint.")
        elif direction == 1 and var in settings.inv_vars:
            settings.inv_vars.remove(var)
            logger.info(f"[{var}] Removed from inv_vars list.")


def complete_preprocessing_pipeline(
    df: pd.DataFrame, settings: "PreprocessingSettings"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    settings : PreprocessingSettings
        Configuration settings object

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (data_clean, data_booked, data_demand) - Cleaned data, booked data, and demand data

    Raises
    ------
    ValueError
        If configuration is invalid or processing fails
    """
    _configure_pipeline_logging(settings.log_level)

    total_start_time = time.time()
    logger.info("=" * 80)
    logger.info("Starting complete preprocessing pipeline")
    logger.info("=" * 80)
    logger.info(f"Input data shape: {df.shape}")
    logger.info(f"Observation period: {settings.date_ini_book_obs} to {settings.date_fin_book_obs}")

    try:
        data_clean = _run_data_transformations(df, settings)
        data_booked = _filter_booked_for_period(data_clean, settings.date_ini_book_obs, settings.date_fin_book_obs)
        data_demand = _filter_demand_for_period(data_clean, settings.date_ini_book_obs, settings.date_fin_book_obs)

        if data_booked.empty:
            raise ValueError(
                f"No booked records found in observation period "
                f"[{settings.date_ini_book_obs}, {settings.date_fin_book_obs}]. "
                f"Cannot infer monotonicity or proceed with optimization."
            )

        # Dynamic monotonicity check using the booked population
        _infer_monotonicity(data_booked, settings)

        # Log final statistics
        logger.info("\n" + "=" * 80)
        logger.info("Preprocessing pipeline completed successfully")
        logger.info("=" * 80)
        logger.info(f"Final data_clean shape: {data_clean.shape}")
        logger.info(f"Final data_booked shape: {data_booked.shape}")
        logger.info(f"Final data_demand shape: {data_demand.shape}")

        total_elapsed_time = time.time() - total_start_time
        logger.info(
            f"Total preprocessing time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)"
        )

        return data_clean, data_booked, data_demand

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}", exc_info=True)
        raise

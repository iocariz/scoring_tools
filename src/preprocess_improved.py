"""
Enhanced data preprocessing module for fraud detection pipeline.

This module provides robust preprocessing functions with improved error handling,
performance optimizations, and comprehensive logging.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
import time
import sys
from loguru import logger


class StatusName(Enum):
    """Enum for status names to avoid string literals."""
    BOOKED = 'booked'
    REJECTED = 'rejected'


class RejectReason(Enum):
    """Enum for reject reasons to avoid string literals."""
    OTHER = '08-other'
    SCORE = '09-score'


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    keep_vars: List[str]
    indicators: List[str]
    segment_filter: str = 'loan_known_ab'
    octroi_bins: Optional[List[float]] = None
    efx_bins: Optional[List[float]] = None
    date_ini_book_obs: Optional[str] = None
    date_fin_book_obs: Optional[str] = None
    score_measures: Optional[List[str]] = None
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.keep_vars:
            raise ValueError("keep_vars cannot be empty")
        if not self.indicators:
            raise ValueError("indicators cannot be empty")
        if self.octroi_bins and len(self.octroi_bins) < 2:
            raise ValueError("octroi_bins must have at least 2 values")
        if self.efx_bins and len(self.efx_bins) < 2:
            raise ValueError("efx_bins must have at least 2 values")


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
    
    # Log a few sample rows if dataframe is not empty
    if not df.empty:
        logger.debug(f"{name} first rows:\n{df.head(3)}")


def validate_dataframe_columns(df: pd.DataFrame, 
                               required_columns: List[str],
                               operation: str) -> None:
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
        raise ValueError(
            f"{operation}: Missing required columns: {missing_cols}"
        )


def preprocess_data(df: pd.DataFrame,
                    keep_vars: List[str],
                    indicators: List[str],
                    segment_filter: str = 'loan_known_ab') -> pd.DataFrame:
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
    required_filter_cols = ['fuera_norma', 'fraud_flag', 'nature_holder', 'segment_cut_off']
    validate_dataframe_columns(df, required_filter_cols, "preprocess_data")
    
    # Collect all measure columns (m_ct_direct*)
    m_ct_direct_columns = [col for col in df.columns if col.startswith('m_ct_direct')]
    logger.info(f"Found {len(m_ct_direct_columns)} measure columns starting with 'm_ct_direct'")
    
    # Validate that requested columns exist
    all_requested_columns = keep_vars + indicators + m_ct_direct_columns
    validate_dataframe_columns(df, all_requested_columns, "preprocess_data")
    
    # Apply filter conditions in a single query operation
    query_start = time.time()
    logger.info(f"Applying filters: fuera_norma='n', fraud_flag='n', "
                f"nature_holder!='legal', segment_cut_off='{segment_filter}'")
    
    try:
        data_filtered = df.query(
            "fuera_norma == 'n' & fraud_flag == 'n' & "
            "nature_holder != 'legal' & segment_cut_off == @segment_filter"
        ).copy()
    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        raise
    
    filter_time = time.time() - query_start
    records_removed = df.shape[0] - data_filtered.shape[0]
    logger.info(f"Filter applied in {filter_time:.2f}s. "
                f"Removed {records_removed:,} records ({records_removed/df.shape[0]:.1%})")
    logger.info(f"{data_filtered.shape[0]:,} records remaining")
    
    if data_filtered.empty:
        raise ValueError("No records remain after filtering")
    
    # Select only needed columns
    logger.info(f"Selecting {len(all_requested_columns)} columns")
    data_clean = data_filtered[all_requested_columns].copy()
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    log_dataframe_stats(data_clean, "Preprocessed data")
    
    return data_clean


def apply_binning_transformations(data: pd.DataFrame,
                                 octroi_bins: List[float],
                                 efx_bins: List[float]) -> pd.DataFrame:
    """
    Apply binning transformations to the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    octroi_bins : List[float]
        Bin edges for octroi score
    efx_bins : List[float]
        Bin edges for efx (risk) score
        
    Returns
    -------
    pd.DataFrame
        Data with binned variables
        
    Raises
    ------
    ValueError
        If required columns are missing or binning fails
    """
    start_time = time.time()
    logger.info("Starting binning transformations")
    logger.info(f"Octroi bins: {octroi_bins}")
    logger.info(f"Efx bins: {efx_bins}")
    
    if len(octroi_bins) < 2:
        raise ValueError("octroi_bins must have at least 2 values")
    if len(efx_bins) < 2:
        raise ValueError("efx_bins must have at least 2 values")
    
    # Validate required columns
    validate_dataframe_columns(data, ['score_rf', 'risk_score_rf'], 
                               "apply_binning_transformations")
    
    # Create a copy to avoid modifying the original data
    transformed_data = data.copy()
    
    # Apply octroi binning
    logger.info("Applying octroi binning to 'score_rf'")
    try:
        transformed_data['sc_octroi_new_clus'] = pd.cut(
            transformed_data['score_rf'],
            bins=octroi_bins,
            labels=False,
            include_lowest=True
        )
        
        # Check for NaN values from binning
        nan_count = transformed_data['sc_octroi_new_clus'].isna().sum()
        if nan_count > 0:
            logger.warning(f"{nan_count:,} records have NaN values after octroi binning")
            score_min = transformed_data['score_rf'].min()
            score_max = transformed_data['score_rf'].max()
            logger.warning(f"score_rf range: [{score_min}, {score_max}]")
            logger.warning(f"octroi_bins range: [{min(octroi_bins)}, {max(octroi_bins)}]")
        
        # Adjust bins to 1-indexed
        transformed_data['sc_octroi_new_clus'] = transformed_data['sc_octroi_new_clus'] + 1
        
    except Exception as e:
        logger.error(f"Error in octroi binning: {e}")
        raise
    
    # Apply efx binning
    logger.info("Applying efx binning to 'risk_score_rf'")
    try:
        transformed_data['new_efx_clus'] = pd.cut(
            transformed_data['risk_score_rf'],
            bins=efx_bins,
            labels=False,
            include_lowest=True
        )
        
        # Check for NaN values from binning
        nan_count = transformed_data['new_efx_clus'].isna().sum()
        if nan_count > 0:
            logger.warning(f"{nan_count:,} records have NaN values after efx binning")
            median_val = transformed_data['new_efx_clus'].median()
            logger.warning(f"Filling NaN values with median: {median_val}")
            transformed_data['new_efx_clus'].fillna(median_val, inplace=True)
        
        # Adjust bins to 1-indexed
        transformed_data['new_efx_clus'] = transformed_data['new_efx_clus'] + 1
        
        # Invert efx_clus (higher is better)
        transformed_data['new_efx_clus'] = len(efx_bins) - transformed_data['new_efx_clus']
        
    except Exception as e:
        logger.error(f"Error in efx binning: {e}")
        raise
    
    # Log distribution of binned variables
    logger.info(f"sc_octroi_new_clus distribution:\n"
                f"{transformed_data['sc_octroi_new_clus'].value_counts().sort_index()}")
    logger.info(f"new_efx_clus distribution:\n"
                f"{transformed_data['new_efx_clus'].value_counts().sort_index()}")
    
    # Log completion
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
    validate_dataframe_columns(data, ['status_name', 'oa_amt', 'oa_amt_h0'], 
                               "update_oa_amt_h0")
    
    result = data.copy()
    
    # Count records to be updated
    update_mask = result['status_name'] != StatusName.BOOKED.value
    update_count = update_mask.sum()
    logger.info(f"Updating {update_count:,} records where status_name != 'booked' "
                f"({update_count/len(result):.1%} of data)")
    
    # Use vectorized operation
    result.loc[update_mask, 'oa_amt_h0'] = result.loc[update_mask, 'oa_amt']
    
    # Verify update
    updated_count = (result.loc[update_mask, 'oa_amt_h0'] == 
                     result.loc[update_mask, 'oa_amt']).sum()
    logger.info(f"Successfully updated {updated_count:,} records")
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"oa_amt_h0 update completed in {elapsed_time:.2f} seconds")
    
    return result


def filter_by_date(data: pd.DataFrame,
                  date_field: str,
                  start_date: str,
                  end_date: str) -> pd.DataFrame:
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
            result[date_field] = pd.to_datetime(result[date_field])
        except Exception as e:
            logger.error(f"Error converting {date_field} to datetime: {e}")
            raise ValueError(f"Cannot convert {date_field} to datetime: {e}")
    
    # Apply date filter
    try:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start_date_dt > end_date_dt:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")
    
    mask = (result[date_field] >= start_date_dt) & (result[date_field] <= end_date_dt)
    filtered_data = result[mask].copy()
    
    # Log results
    filtered_record_count = len(filtered_data)
    removed_record_count = original_record_count - filtered_record_count
    
    if original_record_count > 0:
        pct_removed = removed_record_count / original_record_count
        logger.info(f"Date filter removed {removed_record_count:,} records "
                    f"({pct_removed:.1%} of data)")
    logger.info(f"Remaining records: {filtered_record_count:,}")
    
    # Check for date distribution
    if not filtered_data.empty:
        min_date = filtered_data[date_field].min()
        max_date = filtered_data[date_field].max()
        logger.info(f"Date range in filtered data: {min_date.date()} to {max_date.date()}")
        
        # Log monthly distribution
        monthly_counts = (filtered_data[date_field]
                         .dt.to_period('M')
                         .value_counts()
                         .sort_index())
        logger.info(f"Monthly distribution:\n{monthly_counts}")
    else:
        logger.warning("No records remain after date filtering")
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Date filtering completed in {elapsed_time:.2f} seconds")
    
    return filtered_data


def update_status_and_reject_reason(data: pd.DataFrame, 
                                    score_measures: Optional[List[str]] = None) -> pd.DataFrame:
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
    
    validate_dataframe_columns(data, ['status_name', 'reject_reason'], 
                               "update_status_and_reject_reason")
    
    result = data.copy()
    
    # Get all direct measure columns
    m_ct_cols = [col for col in result.columns if col.startswith('m_ct_direct')]
    logger.info(f"Found {len(m_ct_cols)} direct measure columns")
    
    if not m_ct_cols:
        logger.warning("No m_ct_direct* columns found, skipping measure-based updates")
        return result
    
    # Create a mask for any 'y' in direct measure columns
    measure_mask = result[m_ct_cols].eq('y').any(axis=1)
    measure_count = measure_mask.sum()
    logger.info(f"Found {measure_count:,} records with 'y' in direct measure columns "
                f"({measure_count/len(result):.1%} of data)")
    
    # Update status_name and reject_reason based on the mask
    result.loc[measure_mask, 'status_name'] = StatusName.REJECTED.value
    result.loc[measure_mask, 'reject_reason'] = RejectReason.OTHER.value
    
    # Update reject_reason for score measures if provided
    if score_measures:
        logger.info(f"Updating reject_reason for score measures: {score_measures}")
        validate_dataframe_columns(result, score_measures, 
                                   "update_status_and_reject_reason")
        
        score_measure_mask = result[score_measures].eq('y').any(axis=1)
        score_measure_count = score_measure_mask.sum()
        logger.info(f"Found {score_measure_count:,} records with 'y' in score measure columns "
                    f"({score_measure_count/len(result):.1%} of data)")
        result.loc[score_measure_mask, 'reject_reason'] = RejectReason.SCORE.value
    
    # Log status name distribution after update
    status_counts = result['status_name'].value_counts()
    logger.info(f"status_name distribution after update:\n{status_counts}")
    
    # Log reject reason distribution after update
    reject_counts = result['reject_reason'].value_counts()
    logger.info(f"reject_reason distribution after update:\n{reject_counts}")
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Status and reject reason update completed in {elapsed_time:.2f} seconds")
    
    return result


def complete_preprocessing_pipeline(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    config : PreprocessingConfig
        Configuration object containing all parameters
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (data_clean, data_booked, data_demand) - Cleaned data, booked data, and demand data
        
    Raises
    ------
    ValueError
        If configuration is invalid or processing fails
    """
    # Validate configuration
    config.validate()
    
    # Configure logging based on parameters
    logger.remove() # Remove default handler
    logger.add(sys.stdout, level=config.log_level)
    if config.log_file:
        logger.add(config.log_file, level=config.log_level, rotation="10 MB")
    
    total_start_time = time.time()
    logger.info("=" * 80)
    logger.info("Starting complete preprocessing pipeline")
    logger.info("=" * 80)
    logger.info(f"Input data shape: {df.shape}")
    logger.info(f"Observation period: {config.date_ini_book_obs} to {config.date_fin_book_obs}")
    
    try:
        # Step 1: Basic filtering
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Basic filtering")
        logger.info("=" * 80)
        data_clean = preprocess_data(df, config.keep_vars, config.indicators, 
                                     config.segment_filter)
        
        # Step 2: Apply binning transformations (if bins are provided)
        if config.octroi_bins and config.efx_bins:
            logger.info("\n" + "=" * 80)
            logger.info("Step 2: Binning transformations")
            logger.info("=" * 80)
            data_clean = apply_binning_transformations(data_clean, config.octroi_bins, 
                                                       config.efx_bins)
        else:
            logger.warning("Skipping binning transformations (bins not provided)")
        
        # Step 3: Update oa_amt_h0 for non-booked records
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Update oa_amt_h0")
        logger.info("=" * 80)
        data_clean = update_oa_amt_h0(data_clean)
        
        # Step 4: Update status and reject reasons
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Update status and reject reasons")
        logger.info("=" * 80)
        data_clean = update_status_and_reject_reason(data_clean, config.score_measures)
        
        # Step 5: Filter booked data by date
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Filter booked data by date")
        logger.info("=" * 80)
        data_booked = data_clean[data_clean['status_name'] == StatusName.BOOKED.value].copy()
        logger.info(f"Found {data_booked.shape[0]:,} records with status_name='booked'")
        
        if config.date_ini_book_obs and config.date_fin_book_obs:
            data_booked = filter_by_date(
                data_booked,
                'mis_date',
                config.date_ini_book_obs,
                config.date_fin_book_obs
            )
        else:
            logger.warning("Date filtering skipped for booked data (dates not provided)")
        
        # Step 6: Filter demand data by date
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Filter demand data by date")
        logger.info("=" * 80)
        
        if config.date_ini_book_obs and config.date_fin_book_obs:
            data_demand = filter_by_date(
                data_clean.copy(),
                'mis_date',
                config.date_ini_book_obs,
                config.date_fin_book_obs
            )
        else:
            logger.warning("Date filtering skipped for demand data (dates not provided)")
            data_demand = data_clean.copy()
        
        # Log final statistics
        logger.info("\n" + "=" * 80)
        logger.info("Preprocessing pipeline completed successfully")
        logger.info("=" * 80)
        logger.info(f"Final data_clean shape: {data_clean.shape}")
        logger.info(f"Final data_booked shape: {data_booked.shape}")
        logger.info(f"Final data_demand shape: {data_demand.shape}")
        
        # Log total time
        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total preprocessing time: {total_elapsed_time:.2f} seconds "
                    f"({total_elapsed_time/60:.2f} minutes)")
        
        return data_clean, data_booked, data_demand
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}", exc_info=True)
        raise


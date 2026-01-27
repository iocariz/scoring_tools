import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from loguru import logger
import tomllib
import plotly.graph_objects as go
from src.preprocess_improved import PreprocessingConfig, complete_preprocessing_pipeline, filter_by_date, StatusName
from src.utils import calculate_stress_factor, calculate_and_plot_transformation_rate, get_fact_sol, kpi_of_fact_sol, get_optimal_solutions, calculate_annual_coef, calculate_b2_ever_h6
from src.plots import plot_risk_vs_production, RiskProductionVisualizer
from src.inference_optimized import inference_pipeline, todu_average_inference, run_optimization_pipeline, load_model_for_prediction
import joblib
from src.models import calculate_risk_values
from src.mr_pipeline import process_mr_period
from src import styles


# Required configuration keys
REQUIRED_CONFIG_KEYS = [
    'keep_vars',
    'indicators',
    'segment_filter',
    'octroi_bins',
    'efx_bins',
    'date_ini_book_obs',
    'date_fin_book_obs',
    'variables',
]

OPTIONAL_CONFIG_KEYS = [
    'date_ini_book_obs_mr',
    'date_fin_book_obs_mr',
    'score_measures',
    'data_path',
    'n_months',
    'multiplier',
    'z_threshold',
    'optimum_risk',
    'scenario_step',
    'cz_config',
    'log_level',
]


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_config(config_data: Dict[str, Any]) -> List[str]:
    """
    Validate that all required configuration keys are present and have valid values.

    Args:
        config_data: Configuration dictionary

    Returns:
        List of warning messages (empty if no warnings)

    Raises:
        ConfigValidationError: If required keys are missing or values are invalid
    """
    warnings = []

    # Check required keys
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config_data]
    if missing_keys:
        raise ConfigValidationError(f"Missing required configuration keys: {missing_keys}")

    # Validate keep_vars and indicators are non-empty lists
    if not config_data['keep_vars'] or not isinstance(config_data['keep_vars'], list):
        raise ConfigValidationError("'keep_vars' must be a non-empty list")

    if not config_data['indicators'] or not isinstance(config_data['indicators'], list):
        raise ConfigValidationError("'indicators' must be a non-empty list")

    # Validate variables has exactly 2 elements
    variables = config_data.get('variables', [])
    if len(variables) != 2:
        raise ConfigValidationError(f"'variables' must contain exactly 2 elements, got {len(variables)}")

    # Validate bins have at least 2 values
    if len(config_data['octroi_bins']) < 2:
        raise ConfigValidationError("'octroi_bins' must have at least 2 values")

    if len(config_data['efx_bins']) < 2:
        raise ConfigValidationError("'efx_bins' must have at least 2 values")

    # Validate numeric parameters
    if 'multiplier' in config_data and config_data['multiplier'] <= 0:
        raise ConfigValidationError("'multiplier' must be positive")

    if 'z_threshold' in config_data and config_data['z_threshold'] <= 0:
        raise ConfigValidationError("'z_threshold' must be positive")

    # Check for MR config completeness
    has_mr_ini = 'date_ini_book_obs_mr' in config_data
    has_mr_fin = 'date_fin_book_obs_mr' in config_data
    if has_mr_ini != has_mr_fin:
        warnings.append("MR period dates partially configured - both date_ini_book_obs_mr and date_fin_book_obs_mr should be set")

    return warnings


def validate_date_string(date_str: str, field_name: str) -> pd.Timestamp:
    """
    Validate and convert a date string to pandas Timestamp.

    Args:
        date_str: Date string to validate
        field_name: Name of the field for error messages

    Returns:
        Parsed pandas Timestamp

    Raises:
        ConfigValidationError: If date string is invalid
    """
    if not date_str:
        raise ConfigValidationError(f"'{field_name}' cannot be empty")

    try:
        return pd.to_datetime(date_str)
    except Exception as e:
        raise ConfigValidationError(f"Invalid date format for '{field_name}': {date_str}. Error: {e}")


def validate_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp, range_name: str) -> None:
    """
    Validate that start_date is before end_date.

    Args:
        start_date: Start date
        end_date: End date
        range_name: Name of the date range for error messages

    Raises:
        ConfigValidationError: If start_date is after end_date
    """
    if start_date > end_date:
        raise ConfigValidationError(
            f"Invalid {range_name}: start date ({start_date.date()}) is after end date ({end_date.date()})"
        )


def validate_data_columns(data: pd.DataFrame, required_columns: List[str], context: str = "data") -> List[str]:
    """
    Validate that required columns exist in the DataFrame.

    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        context: Context for error messages

    Returns:
        List of missing columns (empty if all present)

    Raises:
        DataValidationError: If any required columns are missing
    """
    # Normalize column names for comparison
    data_columns = set(data.columns.str.lower())
    missing = [col for col in required_columns if col.lower() not in data_columns]

    if missing:
        raise DataValidationError(f"Missing required columns in {context}: {missing}")

    return []


def validate_data_not_empty(data: pd.DataFrame, context: str = "data") -> None:
    """
    Validate that DataFrame is not empty.

    Args:
        data: DataFrame to validate
        context: Context for error messages

    Raises:
        DataValidationError: If DataFrame is empty
    """
    if data.empty:
        raise DataValidationError(f"{context} is empty")


def convert_bins(bins: List[float]) -> List[float]:
    """
    Convert bin values, replacing float('inf') with numpy constants.

    Args:
        bins: List of bin edge values

    Returns:
        List with inf values replaced by numpy constants
    """
    if not bins:
        return bins
    return [np.inf if x == float('inf') else -np.inf if x == float('-inf') else x for x in bins]


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
    return config_data['preprocessing']


def load_data(df_path: str) -> pd.DataFrame:
    """Load data from SAS file."""
    df = pd.read_sas(df_path, format="sas7bdat", encoding="utf-8")
    return df

def main(config_path: str = "config.toml", model_path: str = None, training_only: bool = False):
    """
    Load and preprocess SAS data using configuration.

    Args:
        config_path: Path to the configuration TOML file (default: config.toml)
        model_path: Optional path to a pre-trained model directory. If provided,
                   skips inference training and loads the existing model.
                   Used for supersegment workflows where model is trained on
                   combined data and optimization runs on individual segments.
        training_only: If True, only runs data preprocessing and model training,
                      skipping optimization and scenario analysis. Used for
                      supersegment model training.

    Returns:
        Tuple of processed DataFrames, or None if processing fails
    """

    # =========================================================================
    # STEP 1: Load and validate configuration
    # =========================================================================
    logger.info(f"Loading configuration from {config_path}...")
    try:
        config_data = load_config(config_path)
        logger.info("Configuration loaded successfully.")

        # Ensure cz_config keys are integers (TOML keys are always strings)
        if 'cz_config' in config_data:
            config_data['cz_config'] = {int(k): v for k, v in config_data['cz_config'].items()}

        # Validate configuration
        warnings = validate_config(config_data)
        for warning in warnings:
            logger.warning(warning)

    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

    # =========================================================================
    # STEP 2: Validate and parse dates
    # =========================================================================
    logger.info("Validating dates...")
    try:
        date_ini = validate_date_string(config_data['date_ini_book_obs'], 'date_ini_book_obs')
        date_fin = validate_date_string(config_data['date_fin_book_obs'], 'date_fin_book_obs')
        validate_date_range(date_ini, date_fin, "main observation period")
        annual_coef = calculate_annual_coef(date_ini, date_fin)
        logger.info(f"Observation period: {date_ini.date()} to {date_fin.date()} (annual_coef: {annual_coef:.2f})")
    except ConfigValidationError as e:
        logger.error(f"Date validation failed: {e}")
        return None

    # =========================================================================
    # STEP 3: Load and validate data
    # =========================================================================
    logger.info("Loading data...")
    try:
        data_path = config_data.get('data_path', "data/demanda_direct_out.sas7bdat")
        data = load_data(data_path)
        validate_data_not_empty(data, "Input data")
        logger.info(f"Data loaded successfully: {data.shape[0]:,} rows × {data.shape[1]} columns")
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {data_path}")
        return None
    except DataValidationError as e:
        logger.error(f"Data validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

    # Standardize column names
    logger.info("Standardizing column names...")
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    logger.debug("Column names standardized")

    # Standardize categorical values
    logger.info("Standardizing categorical values...")
    for col in data.select_dtypes(include=['object', 'category', 'string']).columns:
        data[col] = (data[col]
                    .astype("string")
                    .str.lower()
                    .str.replace(" ", "_")
                    .astype("category"))
    logger.debug("Categorical values standardized")

    # Validate required columns exist after standardization
    try:
        required_cols = config_data['keep_vars'] + config_data['indicators']
        validate_data_columns(data, required_cols, "input data")
        logger.info("All required columns present in data")
    except DataValidationError as e:
        logger.error(f"Data validation failed: {e}")
        return None

    # Display summary
    logger.info("DATA SUMMARY")
    logger.info(f"Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")

    # =========================================================================
    # STEP 4: Preprocessing
    # =========================================================================
    logger.info("Preprocessing data...")

    # Parse bins to replace -inf/inf with numpy constants
    octroi_bins = convert_bins(config_data.get('octroi_bins'))
    efx_bins = convert_bins(config_data.get('efx_bins'))

    config = PreprocessingConfig(
        keep_vars=config_data['keep_vars'],
        indicators=config_data['indicators'],
        segment_filter=config_data['segment_filter'],
        octroi_bins=octroi_bins,
        efx_bins=efx_bins,
        date_ini_book_obs=config_data.get('date_ini_book_obs'),
        date_fin_book_obs=config_data.get('date_fin_book_obs'),
        score_measures=config_data.get('score_measures'),
        log_level=config_data.get('log_level', 'INFO')
    )
    
    data_clean, data_booked, data_demand = complete_preprocessing_pipeline(data, config)

    # Saving risk vs production plot
    logger.info("Saving risk vs production plot...")
    fig = plot_risk_vs_production(data_clean, config_data.get('indicators'), config_data.get('cz_config'), data_booked)
    fig.write_html("images/risk_vs_production.html")
    logger.info("Risk vs production plot saved successfully into images folder")

    # Calculating stress factor
    logger.info("Calculating stress factor...")
    stress_factor = calculate_stress_factor(data_booked)
    logger.info(f"Stress factor: {stress_factor}")

    # Calculating transformation rate
    logger.info("Calculating transformation rate...")
    result = calculate_and_plot_transformation_rate(data_clean, date_col='mis_date', amount_col='oa_amt', n_months=config_data.get('n_months'))
    result['figure'].write_html("images/transformation_rate.html")
    logger.info(f"Finance Rate (last {config_data.get('n_months')} months): {result['overall_rate']:.2%}") 

    # Risk Inference
    if model_path:
        # Load pre-trained model from supersegment
        logger.info(f"Loading pre-trained model from {model_path}...")
        try:
            model, metadata, features = load_model_for_prediction(model_path)
            risk_inference = {
                'best_model_info': {
                    'model': model,
                    'name': metadata.get('model_type', 'Unknown'),
                    'test_r2': metadata.get('test_r2', 0.0),
                },
                'features': features,
                'model_path': model_path
            }
            logger.info(f"Loaded model: {risk_inference['best_model_info']['name']}")
            logger.info(f"Original Test R²: {risk_inference['best_model_info']['test_r2']:.4f}")

            # Load todu model from the same directory
            todu_model_path = Path(model_path).parent / "todu_model.joblib"
            if todu_model_path.exists():
                reg_todu_amt_pile = joblib.load(todu_model_path)
                logger.info(f"Loaded todu model from {todu_model_path}")
            else:
                # Fallback: train todu model on current segment data
                logger.warning(f"Todu model not found at {todu_model_path}, training on current data...")
                _, reg_todu_amt_pile, _ = todu_average_inference(
                    data=data_clean,
                    variables=config_data.get('variables'),
                    indicators=config_data.get('indicators'),
                    feature_col='oa_amt',
                    target_col='todu_amt_pile_h6',
                    z_threshold=config_data.get('z_threshold', 3.0),
                    plot_output_path="models/todu_avg_inference.html",
                    model_output_path=None  # Don't save, it's a fallback
                )
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            return None
    else:
        # Train new model
        logger.info("Calculating risk inference...")
        risk_inference = inference_pipeline(
            data=data_clean,
            bins=(config_data.get('octroi_bins'), config_data.get('efx_bins')),
            variables=config_data.get('variables'),
            indicators=config_data.get('indicators'),
            target_var='todu_30ever_h6',
            multiplier=config_data.get('multiplier', 7),
            test_size=0.4,
            include_hurdle=True,
            save_model=True,
            model_base_path='models',
            create_visualizations=True
        )
        logger.info(f"Best model: {risk_inference['best_model_info']['name']}")
        logger.info(f"Test R²: {risk_inference['best_model_info']['test_r2']:.4f}")

        # Todu Average Inference
        _, reg_todu_amt_pile, _ = todu_average_inference(
            data=data_clean,
            variables=config_data.get('variables'),
            indicators=config_data.get('indicators'),
            feature_col='oa_amt',
            target_col='todu_amt_pile_h6',
            z_threshold=config_data.get('z_threshold', 3.0),
            plot_output_path="models/todu_avg_inference.html",
            model_output_path="models/todu_model.joblib"
        )

    # Early return for training_only mode (supersegment training)
    if training_only:
        logger.info("=" * 80)
        logger.info("TRAINING ONLY MODE - Skipping optimization")
        logger.info("=" * 80)
        logger.info(f"Model trained and saved. Path: {risk_inference.get('model_path', 'models/')}")
        return data_clean, data_booked, data_demand, risk_inference, reg_todu_amt_pile

    # Apply inference model and Optimization Pipeline
    # Using the new refactored function
    
    data_summary_desagregado = run_optimization_pipeline(
        data_booked=data_booked,
        data_demand=data_demand,
        risk_inference=risk_inference,
        reg_todu_amt_pile=reg_todu_amt_pile,
        stressor=stress_factor,
        tasa_fin=result['overall_rate'],
        config_data=config_data,
        annual_coef=annual_coef
    )

    # Cutoff optimization
    logger.info("Optimizing cutoff...")
    values_var0 = sorted(data_summary_desagregado[config_data.get('variables')[0]].unique())
    values_var1 = sorted(data_summary_desagregado[config_data.get('variables')[1]].unique())

    # Get feasible solutions    
    df_v = get_fact_sol(
        values_var0=values_var0,
        values_var1=values_var1,
        chunk_size=10000
    )

    # Obtener KPIs de las soluciones factibles
    data_summary = kpi_of_fact_sol(
        df_v=df_v,
        values_var0=values_var0,
        data_sumary_desagregado=data_summary_desagregado,
        variables=config_data.get('variables'),
        indicadores=config_data.get('indicators'),
        chunk_size=100000  # Adjust based on available memory
    )

    # display not optimal solutions
    data_summary_sample_no_opt = data_summary.sample(10000)

    # Find optimal solutions
    data_summary = get_optimal_solutions(
        df_v=df_v,
        data_sumary=data_summary,
        chunk_size=100000  # Adjust based on available memory
    )

    multiplier = config_data.get('multiplier', 7)
    data_summary_desagregado['b2_ever_h6'] = calculate_b2_ever_h6(
        data_summary_desagregado['todu_30ever_h6'],
        data_summary_desagregado['todu_amt_pile_h6'],
        multiplier=multiplier,
        as_percentage=True
    )
    data_summary_desagregado['text'] = data_summary_desagregado.apply(lambda x: str(
    "{:,.2f}M".format(x['oa_amt_h0']/1000000))+' '+str("{:.2%}".format(x['b2_ever_h6']/100)), axis=1)

    # Scenario Analysis
    base_optimum_risk = config_data.get('optimum_risk', 1.1)
    scenario_step = config_data.get('scenario_step', 0.1)

    # Calculate MR coefficients before loop
    date_ini_mr = pd.to_datetime(config_data.get('date_ini_book_obs_mr'))
    date_fin_mr = pd.to_datetime(config_data.get('date_fin_book_obs_mr'))
    annual_coef_mr = calculate_annual_coef(date_ini_book_obs=date_ini_mr, date_fin_book_obs=date_fin_mr)
    logger.info(f"Annual Coef MR: {annual_coef_mr}")

    scenarios = [base_optimum_risk - scenario_step, base_optimum_risk, base_optimum_risk + scenario_step]
    
    optimal_solution_df_base = None
    
    for scenario_risk in scenarios:
        current_risk = float(round(scenario_risk, 1))
        logger.info(f"Running scenario: optimum_risk = {current_risk}")
        
        visualizer = RiskProductionVisualizer(
             data_summary=data_summary,
             data_summary_disaggregated=data_summary_desagregado,
             data_summary_sample_no_opt=data_summary_sample_no_opt,
             variables=config_data.get('variables'),
             values_var0=values_var0,
             values_var1=values_var1,
             optimum_risk=current_risk,
             tasa_fin=result['overall_rate']
        )
    
        suffix = f"_{current_risk:.1f}"
        
        # Save HTML
        visualizer.save_html(f"images/risk_production_visualizer{suffix}.html")
        
        # Save summary table
        summary_table = visualizer.get_summary_table()
        summary_table.to_csv(f"data/risk_production_summary_table{suffix}.csv", index=False)
        logger.info(f"Risk production summary table saved to data/risk_production_summary_table{suffix}.csv")  
    
        # Save optimal solution
        opt_sol = visualizer.get_selected_solution()
        opt_sol.to_csv(f"data/optimal_solution{suffix}.csv", index=False)
        logger.info(f"Optimal solution saved to data/optimal_solution{suffix}.csv")
        
        if abs(current_risk - base_optimum_risk) < 0.001:
            optimal_solution_df_base = opt_sol
            # Also save as default filenames for backward compatibility
            visualizer.save_html("images/risk_production_visualizer.html")
            summary_table.to_csv("data/risk_production_summary_table.csv", index=False)
            opt_sol.to_csv("data/optimal_solution.csv", index=False)
            logger.info("Base scenario outputs saved to default filenames.")
            
            # Base Scenario MR Processing (Default filenames)
            process_mr_period(
                data_clean=data_clean,
                data_booked=data_booked,
                config_data=config_data,
                risk_inference=risk_inference,
                reg_todu_amt_pile=reg_todu_amt_pile,
                stress_factor=stress_factor,
                tasa_fin=result['overall_rate'],
                annual_coef=annual_coef_mr,
                optimal_solution_df=opt_sol,
                file_suffix=""
            )

        # Scenario MR Processing
        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            config_data=config_data,
            risk_inference=risk_inference,
            reg_todu_amt_pile=reg_todu_amt_pile,
            stress_factor=stress_factor,
            tasa_fin=result['overall_rate'],
            annual_coef=annual_coef_mr,
            optimal_solution_df=opt_sol,
            file_suffix=suffix
        )



    return data_clean, data_booked, data_demand, data_summary_desagregado, data_summary

if __name__ == "__main__":
    main()

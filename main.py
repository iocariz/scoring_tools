import pandas as pd
import numpy as np
from loguru import logger
import tomllib
import plotly.graph_objects as go
from src.preprocess_improved import PreprocessingConfig, complete_preprocessing_pipeline   
from src.utils import calculate_stress_factor, calculate_and_plot_transformation_rate, get_fact_sol, kpi_of_fact_sol, get_optimal_solutions
from src.plots import plot_risk_vs_production, RiskProductionVisualizer
from src.inference_optimized import inference_pipeline, todu_average_inference, run_optimization_pipeline
from src.models import calculate_risk_values

def load_config(config_path="config.toml"):
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
    return config_data['preprocessing']

def load_data(df_path: str):
    df = pd.read_sas(df_path, format="sas7bdat", encoding="utf-8")
    return df

from src.utils import calculate_annual_coef

def main():
    """Load and preprocess SAS data using configuration."""
    
    # Load configuration
    logger.info("Loading configuration...")
    try:
        config_data = load_config()
        logger.info("Configuration loaded successfully.")
        
        # Ensure cz_config keys are integers (TOML keys are always strings)
        if 'cz_config' in config_data:
            config_data['cz_config'] = {int(k): v for k, v in config_data['cz_config'].items()}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

    date_ini = pd.to_datetime(config_data.get('date_ini_book_obs'))
    date_fin = pd.to_datetime(config_data.get('date_fin_book_obs'))
    annual_coef = calculate_annual_coef(date_ini, date_fin) 

    # Load data
    logger.info("Loading data...")
    try:
        data_path = config_data.get('data_path', "data/demanda_direct_out.sas7bdat")
        data = load_data(data_path)
        logger.info(f"Data loaded successfully: {data.shape[0]:,} rows × {data.shape[1]} columns")
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {data_path}")
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
    
    # Display summary
    logger.info("DATA SUMMARY")
    logger.info(f"Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")

    logger.info("Preprocessing data...")
    
    # Parse bins to replace -inf/inf with numpy constants
    octroi_bins = config_data.get('octroi_bins')
    if octroi_bins:
        octroi_bins = [np.inf if x == float('inf') else -np.inf if x == float('-inf') else x for x in octroi_bins]
        
    efx_bins = config_data.get('efx_bins')
    if efx_bins:
        efx_bins = [np.inf if x == float('inf') else -np.inf if x == float('-inf') else x for x in efx_bins]

    config = PreprocessingConfig(
        keep_vars=config_data['keep_vars'],
        indicators=config_data['indicators'],
        segment_filter=config_data['segment_filter'], # Passed as string, ensure it matches what config has
        octroi_bins=octroi_bins,
        efx_bins=efx_bins,
        date_ini_book_obs=config_data.get('date_ini_book_obs'),
        date_fin_book_obs=config_data.get('date_fin_book_obs'),
        score_measures=config_data.get('score_measures'),
        log_level=getattr(logging, config_data.get('log_level', 'INFO')) if 'logging' in locals() else "INFO"
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
    logger.info("Calculating risk inference...")
    risk_inference = inference_pipeline(data=data_clean, bins=(config_data.get('octroi_bins'), config_data.get('efx_bins')), variables=['sc_octroi_new_clus', 'new_efx_clus'], indicators=config_data.get('indicators'), target_var='todu_30ever_h6', multiplier=7, test_size=0.4, include_hurdle=True, save_model=True, model_base_path='models', create_visualizations=True)
    logger.info(f"Best model: {risk_inference['best_model_info']['name']}")
    logger.info(f"Test R²: {risk_inference['best_model_info']['test_r2']:.4f}")

    # Todu Average Inference
    _, reg_todu_amt_pile, _ = todu_average_inference(
        data=data_clean,
        variables=config_data.get('variables'),
        indicators=config_data.get('indicators'),
        feature_col='oa_amt',
        target_col='todu_amt_pile_h6',
        z_threshold=3.0,
        plot_output_path="models/todu_avg_inference.html",
        model_output_path="models/todu_model.joblib"
    )   
  
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

    data_summary_desagregado['b2_ever_h6'] = np.round(
    100*7*data_summary_desagregado['todu_30ever_h6']/data_summary_desagregado['todu_amt_pile_h6'], 2)
    data_summary_desagregado['text'] = data_summary_desagregado.apply(lambda x: str(
    "{:,.2f}M".format(x['oa_amt_h0']/1000000))+' '+str("{:.2%}".format(x['b2_ever_h6']/100)), axis=1)

    visualizer = RiskProductionVisualizer(
     data_summary=data_summary,
     data_summary_disaggregated=data_summary_desagregado,
     data_summary_sample_no_opt=data_summary_sample_no_opt,
     variables=config_data.get('variables'),
     values_var0=values_var0,
     values_var1=values_var1,
     cz2024=1.1,
     tasa_fin=result['overall_rate']
)

    visualizer.save_html("images/risk_production_visualizer.html")
    
    # Save summary table
    summary_table = visualizer.get_summary_table()
    summary_table.to_csv("data/risk_production_summary_table.csv", index=False)
    logger.info("Risk production summary table saved to data/risk_production_summary_table.csv")  

    # Save data
    logger.info("Saving data...")
    data_clean.to_csv("data/data_clean.csv", index=False)
    data_booked.to_csv("data/data_booked.csv", index=False)
    data_demand.to_csv("data/data_demand.csv", index=False) 
    data_summary_desagregado.to_csv("data/data_summary_desagregado.csv", index=False)
    data_summary.to_csv("data/data_summary.csv", index=False)
    logger.info("Data saved successfully")

    return data_clean, data_booked, data_demand, data_summary_desagregado, data_summary

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import logging
import tomllib
from src.preprocess_improved import PreprocessingConfig, complete_preprocessing_pipeline   
from src.utils import calculate_stress_factor, calculate_transformation_rate

def load_config(config_path="config.toml"):
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
    return config_data['preprocessing']

def load_data(df_path: str):
    df = pd.read_sas(df_path, format="sas7bdat", encoding="utf-8")
    return df


def main():
    """Load and preprocess SAS data using configuration."""
    
    # Load configuration
    print("Loading configuration...")
    try:
        config_data = load_config()
        print("✓ Configuration loaded.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

    # Load data
    print("\nLoading data...")
    try:
        data_path = config_data.get('data_path', "data/demanda_direct_out.sas7bdat")
        data = load_data(data_path)
        print(f"✓ Data loaded successfully: {data.shape[0]:,} rows × {data.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Standardize column names
    print("\nStandardizing column names...")
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    print("✓ Column names standardized")

    # Standardize categorical values
    print("\nStandardizing categorical values...")
    for col in data.select_dtypes(include=['object', 'category', 'string']).columns:
        data[col] = (data[col]
                    .astype("string")
                    .str.lower()
                    .str.replace(" ", "_")
                    .astype("category"))
    print("✓ Categorical values standardized")
    
    # Display summary
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")

    print("\nPreprocessing data...")
    
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
        log_level=getattr(logging, config_data.get('log_level', 'INFO'))
    )
    
    data_clean, data_booked, data_demand = complete_preprocessing_pipeline(data, config)

    # Calculating stress factor
    print("\nCalculating stress factor...")
    stress_factor = calculate_stress_factor(data_booked)
    print(f"Stress factor: {stress_factor}")

    # Calculating transformation rate
    print("\nCalculating transformation rate...")
    result = calculate_transformation_rate(data_clean, date_col='mis_date', amount_col='oa_amt', n_months=config_data.get('n_months'))
    print(f"Finance Rate (last {config_data.get('n_months')} months): {result['overall_rate']:.2%}") 

    # Save data
    print("\nSaving data...")
    data_clean.to_csv("data/data_clean.csv", index=False)
    data_booked.to_csv("data/data_booked.csv", index=False)
    data_demand.to_csv("data/data_demand.csv", index=False) 
    print("✓ Data saved")

    return data_clean, data_booked, data_demand

if __name__ == "__main__":
    main()

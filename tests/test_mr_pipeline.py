import pandas as pd
import numpy as np
import pytest
from src.mr_pipeline import calculate_metrics_from_cuts

@pytest.fixture
def sample_data():
    """Create sample data for MR testing"""
    data = pd.DataFrame({
        'sc_octroi_new_clus': [1, 2, 1, 3], # var0
        'new_efx_clus': [1, 5, 10, 2],      # var1
        'todu_30ever_h6_boo': [10, 20, 30, 40],
        'todu_amt_pile_h6_boo': [100, 200, 300, 400],
        'oa_amt_h0_boo': [1000, 2000, 3000, 4000],
        'todu_30ever_h6_rep': [1, 2, 3, 4],
        'todu_amt_pile_h6_rep': [10, 20, 30, 40],
        'oa_amt_h0_rep': [100, 200, 300, 400],
    })
    return data

@pytest.fixture
def optimal_solution_df():
    """Create mock optimal solution with cuts"""
    # Columns '1', '2', '3' represent cuts for bins 1, 2, 3 of var0
    # Bin 1 cut: 5 (allows value 1 and 5)
    # Bin 2 cut: 4 
    # Bin 3 cut: 1
    return pd.DataFrame({
        '1': [5.0],
        '2': [4.0],
        '3': [1.0]
    })

def test_calculate_metrics_from_cuts_basic(sample_data, optimal_solution_df):
    """Test basic calculation of Actual, Swap-in, Swap-out, Optimum"""
    variables = ['sc_octroi_new_clus', 'new_efx_clus']
    
    result = calculate_metrics_from_cuts(
        sample_data,
        optimal_solution_df,
        variables
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert result['Metric'].tolist() == ['Actual', 'Swap-in', 'Swap-out', 'Optimum selected']
    
    # --- Manual Verification of Logic ---
    
    # BIN 1 (Rows 0, 2): Cut = 5.0
    # Row 0: var1=1 <= 5 -> PASS
    # Row 2: var1=10 > 5 -> FAIL
    
    # BIN 2 (Row 1): Cut = 4.0
    # Row 1: var1=5 > 4 -> FAIL
    
    # BIN 3 (Row 3): Cut = 1.0
    # Row 3: var1=2 > 1 -> FAIL
    
    # Expected Passes: Row 0 ONLY
    # Expected Fails: Row 1, 2, 3
    
    # 1. Actual (All Booked)
    expected_actual_prod = 1000 + 2000 + 3000 + 4000 # 10000
    assert result.loc[0, 'Production (€)'] == expected_actual_prod
    
    # 2. Swap-in (Repesca that PASSES) -> Row 0 Rep
    expected_si_prod = 100
    assert result.loc[1, 'Production (€)'] == expected_si_prod
    
    # 3. Swap-out (Booked that FAILS) -> Rows 1, 2, 3 Boo
    expected_so_prod = 2000 + 3000 + 4000 # 9000
    assert result.loc[2, 'Production (€)'] == expected_so_prod
    
    # 4. Optimum (Actual - Swap-out + Swap-in)
    expected_opt_prod = (10000 - 9000) + 100
    assert result.loc[3, 'Production (€)'] == expected_opt_prod
    
    # Check Risk (b2_ever_h6) Calculation for Actual
    # 7 * sum(num) / sum(den)
    # Num = 10+20+30+40 = 100
    # Den = 100+200+300+400 = 1000
    # Risk = 7 * 100 / 1000 = 0.7
    assert np.isclose(result.loc[0, 'Risk (%)'], 0.7)

def test_calculate_metrics_missing_solution(sample_data):
    """Test handling of None/Empty solution"""
    result = calculate_metrics_from_cuts(
        sample_data,
        None,
        ['sc_octroi_new_clus', 'new_efx_clus']
    )
    assert result is None

def test_calculate_metrics_missing_bin(sample_data, optimal_solution_df):
    """Test behavior when a bin in data is missing from solution"""
    # Add a row with Bin 99
    data = sample_data.copy()
    new_row = pd.DataFrame({
        'sc_octroi_new_clus': [99], 
        'new_efx_clus': [1],
        'todu_30ever_h6_boo': [0], 'todu_amt_pile_h6_boo': [1], 'oa_amt_h0_boo': [0],
        'todu_30ever_h6_rep': [0], 'todu_amt_pile_h6_rep': [1], 'oa_amt_h0_rep': [0]
    })
    data = pd.concat([data, new_row], ignore_index=True)
    
    # Function logs warning and defaults to np.inf (Pass all)
    result = calculate_metrics_from_cuts(
        data,
        optimal_solution_df,
        ['sc_octroi_new_clus', 'new_efx_clus']
    )
    
    assert result is not None
    # Verify Bin 99 was processed (implied by it affecting totals if we checked, 
    # but here we just ensure no crash and return)
    

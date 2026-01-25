import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
import gc
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import styles

def get_data_information(df):
    """
    Display DataFrame information and return a DataFrame with variable details.
    """
    # Display DataFrame information
    logger.info(f"Number of rows/records: {df.shape[0]}")
    logger.info(f"Number of columns/variables: {df.shape[1]}")
    logger.info("-" * 50)

    # Create a DataFrame with variable information
    variables_df = pd.DataFrame({
        'Variable': df.columns,
        'Number of unique values': df.nunique(),
        'Variable Type': df.dtypes,
        'Number of missing values': df.isnull().sum(),
        'Percentage missing values': df.isnull().mean() * 100
    })

    # Sort variables by percentage of missing values
    variables_df = variables_df.sort_values(by='Percentage missing values', ascending=False)

    # Return the DataFrame with variable information
    return variables_df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by choosing appropriate dtypes"""
    for col in df.columns:
        # Convert integer columns
        if df[col].dtype == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
       
        # Convert float columns
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
   
    return df

def _get_fact_sol(values_var0, values_var1, inv_var1=False):
    v = [0]+values_var1
    v = pd.DataFrame([[0 for i in v], v]).T.astype(
        'int8').rename(columns={0: 'on'})
    df_v = pd.DataFrame([0], columns=['on'])
    
    i_anterior = None
    for n, i in enumerate(values_var0):
        df_v = df_v.merge(v.rename(columns={1: i}), on='on', how='outer')
        if i > values_var0[0]:
            if inv_var1 == True:
                df_v = df_v[df_v[i_anterior] >= df_v[i]]
            else:
                df_v = df_v[df_v[i_anterior] <= df_v[i]]
        i_anterior = i
        
        logger.info('--Obteniendo soluciones factibles')
        logger.info(f'Bins: {n+1}/{len(values_var0)}')
        logger.info(f'Número de soluciones factibles: {df_v.shape[0]:,.0f}')
    df_v = df_v.drop(columns=['on'])
    df_v = df_v.reset_index(drop=True).reset_index().rename(
        columns={'index': 'sol_fac'})
    return(df_v)

def _kpi_of_fact_sol(df_v, values_var0, data_sumary_desagregado, variables, indicadores, inv_var1=False):
    logger.info('--Calculando KPI de las soluciones factibles')

    df_v_melt = df_v.melt(id_vars=['sol_fac'], value_vars=values_var0,
                          var_name=variables[0], value_name=variables[1]+'_lim')
    df_v_melt_distinct_combination = df_v_melt.drop_duplicates(
        subset=[variables[0], variables[1]+'_lim']).reset_index()[[variables[0], variables[1]+'_lim']]
    data_sumary = df_v_melt_distinct_combination.merge(
        data_sumary_desagregado, how='left', on=variables[0])

    # filtrar solo los optmos para cada valor de riesgo
    if inv_var1 == True:
        data_sumary = data_sumary[data_sumary[variables[1]]
                                  > data_sumary[variables[1]+'_lim']]
    else:
        data_sumary = data_sumary[data_sumary[variables[1]]
                                  <= data_sumary[variables[1]+'_lim']]

    data_sumary = data_sumary.groupby(
        [variables[0], variables[1]+'_lim']).sum().reset_index().drop(columns=[variables[1]])
    data_sumary = df_v_melt.merge(data_sumary, how='left', on=[
                                  variables[0], variables[1]+'_lim']).fillna(0)
    data_sumary = data_sumary.groupby('sol_fac').sum().drop(
        columns=[variables[0], variables[1]+'_lim'])

    for kpi in indicadores:
        data_sumary[kpi+'_cut'] = (data_sumary_desagregado.sum()
                                   [kpi+'_boo']-data_sumary[kpi+'_boo']).apply(lambda x: max(x, 0))

    data_sumary['b2_ever_h6'] = np.round(
        100*7*data_sumary['todu_30ever_h6']/data_sumary['todu_amt_pile_h6'], 2)
    data_sumary['b2_ever_h6_cut'] = np.round(
        100*7*data_sumary['todu_30ever_h6_cut']/data_sumary['todu_amt_pile_h6_cut'], 2)
    data_sumary['b2_ever_h6_rep'] = np.round(
        100*7*data_sumary['todu_30ever_h6_rep']/data_sumary['todu_amt_pile_h6_rep'], 2)
    data_sumary['b2_ever_h6_boo'] = np.round(
        100*7*data_sumary['todu_30ever_h6_boo']/data_sumary['todu_amt_pile_h6_boo'], 2)

    data_sumary = data_sumary[sorted(data_sumary.columns)]
    data_sumary = data_sumary.sort_values(by=["b2_ever_h6", 'oa_amt_h0'])
    return(data_sumary)


def _get_optimal_sol(df_v, data_sumary):
    logger.info('--Obteniendo soluciones optimas')
    data_sumary = data_sumary.sort_values(by=["b2_ever_h6", 'oa_amt_h0'])
    data_sumary = data_sumary.drop_duplicates(
        subset=["b2_ever_h6"], keep='last')
    lim = 0
    list_ = []
    for n, i in enumerate(data_sumary.index):
        value = data_sumary.loc[i, 'oa_amt_h0']
        if (value > lim) | (n == 0):
            lim = value
            list_.append(True)
        else:
            list_.append(False)

    data_sumary = data_sumary[list_]
    data_sumary = df_v.merge(data_sumary.reset_index(
    ), how='inner', on='sol_fac').sort_values(by=["b2_ever_h6", 'oa_amt_h0'])
    logger.info(f'Número de soluciones óptimas: {data_sumary.shape[0]:,.0f}')
    return(data_sumary)

def get_fact_sol(
    values_var0: List[float],
    values_var1: List[float],
    inv_var1: bool = False,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """Memory-optimized version of get_fact_sol"""
    try:
        # Initial setup with optimized dtypes
        v = pd.DataFrame({
            'on': [0] * (len(values_var1) + 1),
            'value': [0] + values_var1
        })
        v = optimize_dtypes(v)
       
        df_v = pd.DataFrame({'on': [0]})
        i_anterior = None
        logger.info('--Getting feasible solutions')
        # Process each value with memory management
        for n, i in tqdm(enumerate(values_var0), desc="Processing Bins"):
            # Memory-efficient merge
            df_v = df_v.merge(
                v.rename(columns={'value': i}),
                on='on',
                how='outer'
            )
           
            # Apply constraints
            if i > values_var0[0]:
                if inv_var1:
                    df_v = df_v[df_v[i_anterior] >= df_v[i]]
                else:
                    df_v = df_v[df_v[i_anterior] <= df_v[i]]
           
            i_anterior = i
           
            # Optimize dtypes after operations
            df_v = optimize_dtypes(df_v)
           
            # Force garbage collection
            gc.collect()
           
            # Progress update
            logger.info(f'Bins: {n+1}/{len(values_var0)}')
            logger.info(f'Number of feasible solutions: {df_v.shape[0]:,}')
       
        # Prepare final output efficiently
        df_v = df_v.drop(columns=['on'])
        df_v = optimize_dtypes(
            df_v.reset_index(drop=True)
            .reset_index()
            .rename(columns={'index': 'sol_fac'})
        )
       
        return df_v
       
    except Exception as e:
        logger.error(f"Error in get_fact_sol: {str(e)}")
        raise

def process_kpi_chunk(
    chunk_data: pd.DataFrame,
    values_var0: List[float],
    data_sumary_desagregado: pd.DataFrame,
    variables: List[str],
    indicadores: List[str],
    inv_var1: bool = False
) -> pd.DataFrame:
    """Process a chunk of data for KPI calculation"""
    # Melt the chunk
    chunk_melt = chunk_data.melt(
        id_vars=['sol_fac'],
        value_vars=values_var0,
        var_name=variables[0],
        value_name=f"{variables[1]}_lim"
    )
   
    # Get distinct combinations for this chunk
    chunk_distinct = chunk_melt.drop_duplicates(
        subset=[variables[0], f"{variables[1]}_lim"]
    ).reset_index()[[variables[0], f"{variables[1]}_lim"]]
   
    # Merge with summary data
    data_sumary = chunk_distinct.merge(
        data_sumary_desagregado,
        how='left',
        on=variables[0]
    )
   
    # Apply filters
    if inv_var1:
        data_sumary = data_sumary[
            data_sumary[variables[1]] > data_sumary[f"{variables[1]}_lim"]
        ]
    else:
        data_sumary = data_sumary[
            data_sumary[variables[1]] <= data_sumary[f"{variables[1]}_lim"]
        ]
   
    # Group and aggregate
    data_sumary = (
        data_sumary.groupby([variables[0], f"{variables[1]}_lim"])
        .sum()
        .reset_index()
        .drop(columns=[variables[1]])
    )
   
    # Merge back with chunk data
    chunk_result = (
        chunk_melt.merge(
            data_sumary,
            how='left',
            on=[variables[0], f"{variables[1]}_lim"]
        )
        .fillna(0)
        .groupby('sol_fac')
        .sum()
        .drop(columns=[variables[0], f"{variables[1]}_lim"])
    )
   
    return chunk_result

def kpi_of_fact_sol(df_v, values_var0, data_sumary_desagregado, variables, indicadores, inv_var1=False, chunk_size=1000):
    try:
        logger.info('--Calculating KPIs for feasible solutions')
        
        # Process in chunks
        chunks_results = []
        total_chunks = len(df_v) // chunk_size + (1 if len(df_v) % chunk_size > 0 else 0)
        
        for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df_v))
            
            # Process chunk
            df_chunk = df_v.iloc[start_idx:end_idx].copy()
            
            # Melt the chunk
            df_melt = df_chunk.melt(
                id_vars=['sol_fac'],
                value_vars=values_var0,
                var_name=variables[0],
                value_name=f"{variables[1]}_lim"
            )
            
            # Ensure numeric types
            df_melt[variables[0]] = df_melt[variables[0]].astype(float)
            
            # Get distinct combinations for this chunk
            df_melt_distinct = df_melt.drop_duplicates(
                subset=[variables[0], f"{variables[1]}_lim"]
            )[[variables[0], f"{variables[1]}_lim"]]
            
            # Merge with summary data
            data_sumary = df_melt_distinct.merge(
                data_sumary_desagregado,
                how='left',
                on=variables[0]
            )
            
            # Apply filters
            if inv_var1:
                data_sumary = data_sumary[
                    data_sumary[variables[1]] > data_sumary[f"{variables[1]}_lim"]
                ]
            else:
                data_sumary = data_sumary[
                    data_sumary[variables[1]] <= data_sumary[f"{variables[1]}_lim"]
                ]
            
            # Group and aggregate numeric columns only
            numeric_cols = data_sumary.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'sum' for col in numeric_cols 
                       if col not in [variables[0], f"{variables[1]}_lim"]}
            
            data_sumary = (
                data_sumary.groupby([variables[0], f"{variables[1]}_lim"])
                .agg(agg_dict)
                .reset_index()
            )
            
            # Process chunk result
            chunk_result = (
                df_melt.merge(
                    data_sumary,
                    how='left',
                    on=[variables[0], f"{variables[1]}_lim"]
                )
                .fillna(0)
                .groupby('sol_fac', observed=True)
                .agg(agg_dict)
                .reset_index()
            )
            
            chunks_results.append(chunk_result)
            
            # Clean up memory
            del df_chunk, df_melt, df_melt_distinct, data_sumary
            gc.collect()
        
        # Combine results from all chunks
        if not chunks_results:
            return pd.DataFrame()
            
        # Combine chunks efficiently
        final_result = pd.concat(chunks_results, ignore_index=True)
        del chunks_results
        gc.collect()
        
        # Group combined results
        final_result = final_result.groupby('sol_fac', observed=True).sum().reset_index()
        
        # Calculate cut metrics
        for kpi in indicadores:
            final_result[f'{kpi}_cut'] = (
                data_sumary_desagregado[f'{kpi}_boo'].sum() - 
                final_result[f'{kpi}_boo']
            ).clip(lower=0)
        
        # Calculate B2 metrics
        metrics = ['', '_cut', '_rep', '_boo']
        for metric in metrics:
            todu_30 = f'todu_30ever_h6{metric}'
            todu_amt = f'todu_amt_pile_h6{metric}'
            if todu_30 in final_result.columns and todu_amt in final_result.columns:
                final_result[f'b2_ever_h6{metric}'] = np.round(
                    100 * 7 * final_result[todu_30].astype(float) /
                    final_result[todu_amt].replace(0, np.nan).astype(float),
                    2
                ).fillna(0)
        
        return final_result.sort_values(['b2_ever_h6', 'oa_amt_h0'])
        
    except Exception as e:
        logger.error(f"Error in kpi_of_fact_sol: {str(e)}")
        raise

def get_optimal_solutions(
    df_v: pd.DataFrame,
    data_sumary: pd.DataFrame,
    chunk_size: int = 1000
) -> pd.DataFrame:
    """Memory-optimized version of get_optimal_solutions"""
    try:
        logger.info('--Getting optimal solutions')
       
        # Sort and deduplicate efficiently
        data_sumary = data_sumary.sort_values(
            by=["b2_ever_h6", 'oa_amt_h0']
        )
        data_sumary = data_sumary.drop_duplicates(
            subset=["b2_ever_h6"],
            keep='last'
        )
       
        # Find Pareto optimal solutions efficiently
        data_sumary['optimal'] = False
        current_max = float('-inf')
       
        for idx in data_sumary.index:
            value = data_sumary.loc[idx, 'oa_amt_h0']
            if value > current_max:
                current_max = value
                data_sumary.loc[idx, 'optimal'] = True
       
        data_sumary = data_sumary[data_sumary['optimal']].drop(columns=['optimal'])
       
        # Merge in chunks
        chunks_results = []
        total_chunks = len(df_v) // chunk_size + (1 if len(df_v) % chunk_size > 0 else 0)
       
        for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df_v))
           
            chunk_result = df_v.iloc[start_idx:end_idx].merge(
                data_sumary.reset_index(),
                how='inner',
                on='sol_fac'
            )
           
            if not chunk_result.empty:
                chunks_results.append(chunk_result)
           
            gc.collect()
       
        # Combine results
        final_result = pd.concat(chunks_results, ignore_index=True)
        del chunks_results
        gc.collect()
       
        # Optimize final datatypes
        final_result = optimize_dtypes(final_result)
       
        logger.info(f'Number of optimal solutions: {len(final_result):,}')
        return final_result.sort_values(by=["b2_ever_h6", 'oa_amt_h0'])
       
    except Exception as e:
        logger.error(f"Error in get_optimal_solutions: {str(e)}")
        raise

def calculate_stress_factor(df: pd.DataFrame, 
                            status_col: str = 'status_name',
                            score_col: str = 'risk_score_rf',
                            num_col: str = 'todu_30ever_h6',
                            den_col: str = 'todu_amt_pile_h6',
                            frac: float = 0.05,
                            target_status: str = 'booked',
                            bad_rate: float = 0.05) -> float:
    
    # Filter for target status
    df_target = df[df[status_col] == target_status].copy()
    
    if df_target.empty:
        logger.warning(f"No records found with {status_col} = {target_status}")
        return 0.0
        
    # Calculate overall bad rate
    total_num = df_target[num_col].sum()
    total_den = df_target[den_col].sum()
    
    overall_bad_rate = (total_num / total_den * 7) if total_den > 0 else bad_rate
    
    # Calculate cutoff using quantile from the known population
    cutoff_score = df_target[score_col].quantile(frac)
    
    # Select worst population based on score cutoff
    # Assuming lower score is worse (ascending=True)
    df_worst = df_target[df_target[score_col] <= cutoff_score]
    
    logger.debug(f"Score cutoff (frac={frac}): {cutoff_score}")
    logger.debug(f"Selected {len(df_worst)}/{len(df_target)} records ({len(df_worst)/len(df_target):.2%}) as worst population")

    # Calculate bad rate for worst fraction
    worst_num = df_worst[num_col].sum()
    worst_den = df_worst[den_col].sum()
    
    worst_bad_rate = (worst_num / worst_den * 7) if worst_den > 0 else 0.0
    
    # Calculate stress factor
    if overall_bad_rate > 0:
        stress_factor = worst_bad_rate / overall_bad_rate
    else:
        stress_factor = 0.0
        
    return float(stress_factor)

def calculate_and_plot_transformation_rate(
    data: pd.DataFrame, 
    date_col: str, 
    amount_col: str = 'oa_amt', 
    n_months: int = None,
    plot_width: int = 1200,
    plot_height: int = 500
):
    """
    Calculates transformation rate and generates a dual-axis plot.
    
    Transformation Rate = Booked Amount / Eligible Amount (Decision in OK/RV)
    
    Returns:
    --------
    dict containing:
      - 'stats': Dictionary of overall stats
      - 'monthly_data': DataFrame of monthly breakdown
      - 'figure': Plotly go.Figure object
    """
    
    # 1. Data Preparation
    # ---------------------------------------------------------
    df = data.copy()
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter for last n months
    if n_months is not None:
        max_date = df[date_col].max()
        cutoff_date = max_date - pd.DateOffset(months=n_months)
        df = df[df[date_col] >= cutoff_date]
    
    # Filter eligible (Denominator: Decision OK or RV)
    eligible_mask = df['se_decision_id'].isin(['ok', 'rv'])
    df_eligible = df[eligible_mask].copy()
    
    # Identify booked (Numerator: Status Booked AND Eligible)
    # Note: We rely on the row being in the eligible set first
    df_eligible['is_booked'] = df_eligible['status_name'] == 'booked'
    
    # 2. Calculation
    # ---------------------------------------------------------
    # Overall Stats
    total_eligible = df_eligible[amount_col].sum()
    total_booked = df_eligible.loc[df_eligible['is_booked'], amount_col].sum()
    overall_rate = (total_booked / total_eligible) if total_eligible > 0 else 0
    
    # Monthly Aggregation
    # We use to_period('M') for grouping, but convert back to timestamp for plotting
    df_eligible['period'] = df_eligible[date_col].dt.to_period('M')
    
    monthly = df_eligible.groupby('period').agg(
        eligible_amt=(amount_col, 'sum'),
        booked_amt=(amount_col, lambda x: x[df_eligible.loc[x.index, 'is_booked']].sum())
    ).reset_index()
    
    monthly['rate'] = monthly['booked_amt'] / monthly['eligible_amt']
    monthly['rate'] = monthly['rate'].fillna(0)
    
    # Convert period back to timestamp for Plotly (uses start of month)
    monthly['plot_date'] = monthly['period'].dt.to_timestamp()

    # 3. Plotting
    # ---------------------------------------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Volume (Bars) - Plotted on Secondary Y (Right)
    # We plot this first so it stays in the background
    fig.add_trace(
        go.Bar(
            x=monthly['plot_date'],
            y=monthly['eligible_amt'],
            name='Eligible Volume (€)',
            opacity=0.3,
            marker_color=styles.COLOR_SECONDARY,
            hovertemplate='Volume: %{y:,.0f} €<extra></extra>'
        ),
        secondary_y=True
    )

    # Trace 2: Transformation Rate (Line) - Plotted on Primary Y (Left)
    fig.add_trace(
        go.Scatter(
            x=monthly['plot_date'],
            y=monthly['rate'],
            name='Transformation Rate',
            mode='lines+markers',
            line=dict(color=styles.COLOR_ACCENT, width=3),
            marker=dict(size=8),
            hovertemplate='Rate: %{y:.1%}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add Overall Average Line
    fig.add_hline(
        y=overall_rate, 
        line_dash="dot", 
        line_color=styles.COLOR_RISK, 
        annotation_text=f"Avg: {overall_rate:.1%}", 
        annotation_position="top left",
        secondary_y=False
    )

    # 4. Layout Improvements
    # ---------------------------------------------------------
    styles.apply_plotly_style(fig, width=plot_width, height=plot_height)
    fig.update_layout(
        title='<b>Monthly Transformation Rate</b><br><sup>(Booked / [OK + RV]) by Amount</sup>',
        plot_bgcolor='white',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        
        # Left Axis (Rate)
        yaxis=dict(
            title="Transformation Rate",
            tickformat=".0%", # Displays 0.5 as 50%
            range=[0, max(monthly['rate'].max() * 1.1, overall_rate * 1.1)],
            showgrid=True,
            gridcolor='lightgrey'
        ),
        
        # Right Axis (Volume)
        yaxis2=dict(
            title="Eligible Volume (€)",
            showgrid=False,
            zeroline=False
        ),
        
        xaxis=dict(
            title="Month",
            showgrid=False
        )
    )

    return {
        'overall_rate': overall_rate,
        'overall_booked_amt': total_booked,
        'overall_eligible_amt': total_eligible,
        'monthly_amounts': monthly.drop(columns=['plot_date']), # Return clean DF
        'figure': fig
    }

def calculate_annual_coef(date_ini_book_obs: pd.Timestamp, date_fin_book_obs: pd.Timestamp) -> float:
    """
    Calculate annual coefficient based on the time range.
    """
    n_month = (
        (date_fin_book_obs.year  - date_ini_book_obs.year)  * 12 +
        (date_fin_book_obs.month - date_ini_book_obs.month) + 1
    )
    annual_coef = 12/n_month
    return annual_coef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact
from colorama import Fore
from sklearn.metrics import roc_curve, auc
from .metrics import ks_statistic, compute_metrics

# Keep track of previous KS annotation y-coordinates
previous_ks_y_positions = []

def plot_roc_curve(ax, y_true, scores, name, color):
    """Plot ROC curve on given axes."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    gini = 2 * auc(fpr, tpr) - 1
    ks = ks_statistic(y_true, scores)
    
    ax.plot(fpr, tpr, lw=2.5, label=f'GINI {name} ({gini:.3f})', color=color)
    ax.fill_between(fpr, tpr, color=color, alpha=0.05)
    ks_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[ks_idx], tpr[ks_idx], color='red', s=60, zorder=5)
    ax.annotate(f'KS={ks:.3f}', (fpr[ks_idx], tpr[ks_idx]), fontsize=12,
                xytext=(-80,-10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))

def visualize_metrics(y_true, scores_dict, ax):
    sns.set_style('whitegrid')
    palette = sns.color_palette("deep", len(scores_dict))

    for (name, scores), color in zip(scores_dict.items(), palette):
        current_ax_roc = ax[0] if 'combined' not in name.lower() else ax[1]
        current_ax_cap = ax[2] if 'combined' not in name.lower() else ax[3]

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        gini = 2 * auc(fpr, tpr) - 1
        current_ax_roc.plot(fpr, tpr, lw=2.5, label=f'GINI {name} ({gini:.3f})', color=color)
        current_ax_roc.fill_between(fpr, tpr, color=color, alpha=0.05)

        # Plot CAP Curve
        y_true_array = np.array(y_true)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_true_values = y_true_array[sorted_indices]
        cumulative_true_positives = np.cumsum(sorted_true_values)
        total_true_positives = np.sum(y_true_array)
        current_ax_cap.plot(np.linspace(0, 1, len(y_true_array)), cumulative_true_positives / total_true_positives, lw=2.5, label=f'CAP {name}', color=color)

    for axis in ax[:2]:
        axis.plot([0, 1], [0, 1], 'k--', lw=1.5)  # Ideal line for ROC
        axis.set_xlabel('False Positive Rate', fontsize=14)
        axis.set_ylabel('True Positive Rate', fontsize=14)
        axis.legend(loc='lower right', fontsize=12)
        axis.set_ylim(0, 1.05)
        axis.grid(True, which='both', linestyle='--', linewidth=0.5)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    for axis in ax[2:]:
        axis.plot([0, 1], [0, 1], 'k--', lw=1.5)  # Ideal line for CAP
        axis.set_xlabel('Fraction of Population', fontsize=14)
        axis.set_ylabel('Fraction of Total Positives', fontsize=14)
        axis.legend(loc='lower right', fontsize=12)
        axis.set_ylim(0, 1.05)
        axis.grid(True, which='both', linestyle='--', linewidth=0.5)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    ax[0].set_title('Individual Models ROC', fontsize=16)
    ax[1].set_title('Combined Models ROC', fontsize=16)
    ax[2].set_title('Individual Models CAP', fontsize=16)
    ax[3].set_title('Combined Models CAP', fontsize=16)

def plot_gini_confidence_intervals(ax, df):
    """
    Plots the Gini confidence intervals for each model.
    """
    # Sort dataframe by Gini Score for a better visualization
    df = df.sort_values(by="Gini Score")
    
    # Extract lower and upper bounds of confidence intervals
    lower_bounds = [ci[0] for ci in df["Gini CI"]]
    upper_bounds = [ci[1] for ci in df["Gini CI"]]
    
    # Create a range for the y axis
    y_range = list(range(len(df)))

    ax.errorbar(df["Gini Score"],
                y_range,
                xerr=[df["Gini Score"] - lower_bounds, upper_bounds - df["Gini Score"]],
                fmt='o',
                color='blue',
                ecolor='gray',
                elinewidth=3,
                capsize=5,
                markersize=8)

    # Annotate with model names
    for y, model in enumerate(df["Model"]):
        ax.annotate(model, (0.05, y), color="black", fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel('Gini Score')
    ax.set_title('Gini Confidence Intervals',fontsize=16)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()  # Higher scores at the top

def plot_group_statistics(data_frame, group_col, binary_outcome):
    """
    Plot the risk rate and frequency for each group.
    """
    grouped = data_frame.groupby(group_col).agg({binary_outcome: ['mean', 'count']})
    grouped.columns = ['Target_mean', "Target_count"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Frequency', color=color)
    ax1.bar(grouped.index.astype(str), grouped["Target_count"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f"Risk rate per groups")
    ax1.set_xticklabels(grouped.index, rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Risk Rate', color=color)
    ax2.plot(grouped.index.astype(str).to_numpy(), grouped["Target_mean"].to_numpy(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def plot_risk_vs_production(
    data: pd.DataFrame,
    indicadores: List[str],
    cz2022: float,
    cz2023: float,
    cz2024: float,
    data_booked: pd.DataFrame,
    rolling_window: int = 6,
    plot_width: int = 1500,
    plot_height: int = 500
) -> go.Figure:
    """
    Creates an interactive plot comparing risk metrics against production data over time.
    """
    # Extract and aggregate booked data
    df_plot = (data[data['status_name'] == 'booked']
               .groupby(['mis_date'], as_index=False)
               .agg({col: 'sum' for col in indicadores}))
   
    # Calculate rolling sums
    risk_cols = ['todu_30ever_h6', 'todu_amt_pile_h6']
    df_plot_MA = df_plot[risk_cols].rolling(rolling_window).sum()
    df_plot_MA.columns = [f"{col}_MA{rolling_window}" for col in df_plot_MA.columns]
    df_plot = pd.concat([df_plot, df_plot_MA], axis=1)
   
    # Calculate risk percentages
    df_plot['b2_ever_h6'] = np.round(
        100 * 7 * df_plot['todu_30ever_h6'] / df_plot['todu_amt_pile_h6'], 2)
    df_plot['b2_ever_h6_MA'] = np.round(
        100 * 7 * df_plot[f'todu_30ever_h6_MA{rolling_window}'] /
        df_plot[f'todu_amt_pile_h6_MA{rolling_window}'], 2)
   
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
   
    # Add production area plot
    fig.add_trace(
        go.Scatter(
            x=df_plot['mis_date'],
            y=df_plot['oa_amt_h0'],
            mode="lines",
            line=dict(width=1.5, color='rgba(116,196,118,1)'),
            opacity=0.1,
            name='Production Amount',
            fill='tozeroy',
            fillcolor='rgba(116,196,118,0.2)'
        ),
        secondary_y=False
    )
   
    # Add risk line plots
    fig.add_trace(
        go.Scatter(
            x=df_plot['mis_date'],
            y=df_plot['b2_ever_h6'],
            mode="lines",
            line=dict(width=1.5, color='black'),
            name='Risk Percentage'
        ),
        secondary_y=True
    )
   
    # Add moving average line (excluding NaN values)
    mask = ~df_plot['b2_ever_h6'].isna()
    fig.add_trace(
        go.Scatter(
            x=df_plot.loc[mask, 'mis_date'],
            y=df_plot.loc[mask, 'b2_ever_h6_MA'],
            mode="lines",
            line=dict(width=1.5, color='black', dash='dot'),
            name=f'Risk Percentage (MA-{rolling_window})'
        ),
        secondary_y=True
    )
   
    # Add CZ lines
    for year, cz_value in {2022: cz2022, 2023: cz2023, 2024: cz2024}.items():
        year_mask = df_plot['mis_date'].dt.year == year
        if year_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df_plot.loc[year_mask, 'mis_date'],
                    y=[cz_value] * year_mask.sum(),
                    mode="lines",
                    line=dict(width=2, color='rgba(165,15,21,1)'),
                    name=f'CZ {year}'
                ),
                secondary_y=True
            )
   
    # Add selected period highlight
    fig.add_vrect(
        x0=data_booked['mis_date'].min() - pd.Timedelta(days=15),
        x1=data_booked['mis_date'].max() + pd.Timedelta(days=15),
        fillcolor="blue",
        opacity=0.1,
        line_width=0,
        annotation_text="Selected period (Booked)",
        annotation_position="bottom left"
    )
   
    # Update layout
    fig.update_layout(
        width=plot_width,
        height=plot_height,
        title='Risk vs Production Analysis',
        plot_bgcolor='white',
        xaxis=dict(
            title="Date",
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            title="Production (€)",
            side="right",
            range=[0, df_plot['oa_amt_h0'].max() * 1.1],
            showgrid=False
        ),
        yaxis2=dict(
            title="Risk (%€)",
            side="left",
            gridcolor='lightgrey',
            range=[0, df_plot['b2_ever_h6'].max() * 1.1],
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
   
    return fig

def plot_3d_graph(data_train, data_surf, variables, var_target):
    data_surf_pivot = data_surf.pivot(
        index=variables[1], columns=variables[0], values=f'{var_target}_pred')

    # gráfico
    fig = go.Figure()

    # Scatter plot for outliers
    fig = fig.add_trace(go.Scatter3d(
        x=data_surf[variables[0]],
        y=data_surf[variables[1]],
        z=data_surf[var_target],
        name=f'{var_target} (outliers)',
        showlegend=True,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.5)
    ))
    # Scatter plot for actual data
    fig = fig.add_trace(go.Scatter3d(
        x=data_train[variables[0]],
        y=data_train[variables[1]],
        z=data_train[var_target],
        name=var_target,
        showlegend=True,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.5)
    ))

    fig = fig.add_trace(go.Surface(
        x=data_surf_pivot.columns,
        y=data_surf_pivot.index,
        z=data_surf_pivot.values,
        colorscale='Viridis',
        name='Regression',
        showlegend=True,
        opacity=0.7
    ))
    fig.update_layout(
        title="Regression Plot",
        width=1200,
        height=900,
        scene=dict(
            xaxis=dict(title=variables[0], gridcolor='white'),
            yaxis=dict(title=variables[1], gridcolor='white'),
            zaxis=dict(title=var_target, gridcolor='white'),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        legend=dict(y=1, x=0.8))
    fig.show()

class RiskProductionVisualizer:
    def __init__(self, data_summary, data_summary_disaggregated, data_summary_sample_no_opt,
                 variables, values_var0, values_var1, cz2024, tasa_fin):
        self.data_summary = data_summary
        self.data_summary_disaggregated = data_summary_disaggregated
        self.data_summary_sample_no_opt = data_summary_sample_no_opt
        self.variables = variables
        self.values_var0 = values_var0
        self.values_var1 = values_var1
        self.cz2024 = cz2024
        self.tasa_fin = tasa_fin
       
        # Calculate initial metrics
        self.calculate_initial_metrics()
       
        # Create the visualization
        self.create_figure()
        self.create_slider()
       
    def calculate_initial_metrics(self):
        """Calculate initial B2 and OA metrics"""
        # Calculate B2_0
        tudu_30_ever = self.data_summary_disaggregated['todu_30ever_h6_boo'].sum()
        tudu_amt_pile = self.data_summary_disaggregated['todu_amt_pile_h6_boo'].sum()
        self.B2_0 = np.round(100 * 7 * tudu_30_ever / tudu_amt_pile, 2)
       
        # Calculate OA_0
        self.OA_0 = self.data_summary_disaggregated['oa_amt_h0_boo'].sum()
       
        # Calculate limits
        self.lim_sup_B2 = self.data_summary['b2_ever_h6'].max()
        self.lim_inf_B2 = 0
        self.lim_sup_OA = self.data_summary['oa_amt_h0'].max()
        self.lim_inf_OA = 0
       
        # Create meshgrid for heatmap
        self.xx, self.yy = np.meshgrid(self.values_var0, self.values_var1)
   
    def create_figure(self):
        """Create the main figure with subplots"""
        self.fig = go.Figure(make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Optimal relationship between production and risk',
                'Production & B2 ever H6 Heatmap / Optimal Cut-Off'
            )
        ))
       
        self._add_scatter_traces()
        self._add_heatmap_traces()
        self._update_layout()
   
    def _add_scatter_traces(self):
        """Add scatter plot traces to the figure"""
        # Non-optimal solutions
        self.fig.add_trace(
            go.Scatter(
                x=self.data_summary_sample_no_opt["oa_amt_h0"],
                y=self.data_summary_sample_no_opt["b2_ever_h6"],
                name='Non-optimal solutions',
                mode='markers',
                marker=dict(size=1.5, color='rgba(209,142,145,0.5)'),
                showlegend=True
            ),
            row=1, col=1
        )
       
        # Optimal line
        self.fig.add_trace(
            go.Scatter(
                x=self.data_summary["oa_amt_h0"],
                y=self.data_summary["b2_ever_h6"],
                name='Optimal line',
                marker=dict(size=0),
                line=dict(color='rgba(157,13,20,.8)', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
       
        # Actual point
        self.fig.add_trace(
            go.Scatter(
                x=[self.OA_0],
                y=[self.B2_0],
                name='Actual',
                mode='markers',
                marker=dict(size=8, color='rgba(0,0,0,1)'),
                showlegend=True
            ),
            row=1, col=1
        )
       
        # Optimum selected point
        self.optimum_trace = go.Scatter(
            x=[0],
            y=[0],
            name='Optimum selected',
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(33,150,243,1)',
                line=dict(width=.2, color='black')
            ),
            showlegend=True
        )
        self.fig.add_trace(self.optimum_trace, row=1, col=1)
   
    def _add_heatmap_traces(self):
        """Add heatmap traces to the figure"""
        # Main heatmap
        self.fig.add_trace(
            go.Heatmap(
                x=self.data_summary_disaggregated[self.variables[0]],
                y=self.data_summary_disaggregated[self.variables[1]],
                z=self.data_summary_disaggregated['b2_ever_h6'],
                colorscale=[
                    (0.00, "rgba(255,255,255,1)"),
                    (1.00, "rgba(157,13,20,1)")
                ],
                zmin=0,
                zmax=20,
                text=self.data_summary_disaggregated['text'],
                texttemplate="%{text}"
            ),
            row=1, col=2
        )
       
        # Mask heatmap
        self.mask_trace = go.Heatmap(
            x=self.values_var0,
            y=self.values_var1,
            z=np.zeros_like(self.yy),
            colorscale=[
                (0.00, "rgba(0,0,0,0.5)"),
                (1.00, "rgba(0,0,0,0)")
            ],
            showscale=False
        )
        self.fig.add_trace(self.mask_trace, row=1, col=2)
   
    def _update_layout(self):
        """Update the figure layout"""
        self.fig.update_layout(
            width=1700,
            height=600,
            legend=dict(y=.1, x=0.33)
        )
       
        # Update axes
        self.fig.update_xaxes(
            title_text='OA AMT (€)',
            range=[self.lim_inf_OA, self.lim_sup_OA],
            row=1, col=1
        )
        self.fig.update_xaxes(
            title_text=self.variables[0],
            dtick=1,
            row=1, col=2
        )
        self.fig.update_yaxes(
            title_text='B2 ever H6 (%€)',
            range=[self.lim_inf_B2, self.lim_sup_B2],
            row=1, col=1
        )
        self.fig.update_yaxes(
            title_text=self.variables[1],
            dtick=1,
            row=1, col=2
        )
   
    def create_slider(self):
        """Create the risk level slider"""
        self.slider = widgets.FloatSlider(
            value=self.cz2024,
            min=self.lim_inf_B2,
            max=self.lim_sup_B2,
            step=0.01,
            description='Risk level',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='',
            layout=widgets.Layout(width='45%')
        )
   
    def update(self, x):
        """Update the visualization based on slider value"""
        # Filter data based on risk level
        data_filtered = self.data_summary[self.data_summary['b2_ever_h6'] <= x].tail(1)
        metrics = data_filtered[['b2_ever_h6', 'oa_amt_h0', 'b2_ever_h6_cut',
                               'oa_amt_h0_cut', 'b2_ever_h6_rep', 'oa_amt_h0_rep']].values[0, :]
        B2, OA, B2_CUT, OA_CUT, B2_REP, OA_REP = metrics
       
        # Update cut-off mask
        CUT_OFF = data_filtered[self.values_var0].values[0]
        z_mask = (self.yy <= CUT_OFF) * 1
       
        # Update visualization
        self.fig.data[5].z = z_mask
        self.fig.data[3].update(x=[OA], y=[B2])
       
        # Print metrics
        self._print_metrics(B2, OA, B2_CUT, OA_CUT, B2_REP, OA_REP)
       
        return CUT_OFF
   
    def _print_metrics(self, B2, OA, B2_CUT, OA_CUT, B2_REP, OA_REP):
        """Print the performance metrics"""
        print('   financing rate (fin/app):', self.tasa_fin)
        print(Fore.BLACK + f'   Actual -------------- Risk: {max(self.B2_0/100, 0.000001):.2%}   ,  Production: {self.OA_0:,.0f}€ ({self.OA_0/self.OA_0:.2%})')
        print(Fore.GREEN + f'   Swap-in ------------- Risk: {max(B2_REP/100, 0.000001):.2%}   ,  Production: {OA_REP:,.0f}€ ({OA_REP/self.OA_0:.2%})')
        print(Fore.RED + f'   Swap-out ------------ Risk: {max(B2_CUT/100, 0.000001):.2%}   ,  Production: {OA_CUT:,.0f}€ ({OA_CUT/self.OA_0:.2%})')
        print(Fore.BLUE + f'   Optimum selected ---- Risk: {max(B2/100, 0.000001):.2%}   ,  Production: {OA:,.0f}€ ({OA/self.OA_0:.2%})')
        print(Fore.BLACK + '   ----------------------------------------------------------------------------')
        print(Fore.BLACK + f'   Summary ------------- Risk: {("+" if B2 > self.B2_0 else "")}{B2-self.B2_0:.2}p.p.',
              f',  Production: {("+" if OA > self.OA_0 else "")}{OA-self.OA_0:,.0f}€',
              f'({("+" if OA > self.OA_0 else "")}{(OA-self.OA_0)/self.OA_0:.2%})')
   
    def display(self):
        """Display the visualization and enable interaction"""
        display(self.fig)
        interact(self.update, x=self.slider)


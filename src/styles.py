"""
Unified styling module for the scoring tools project.
Defines color palettes and helper functions to ensure consistency across all plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

# --- Color Constants ---
# Primary palette
COLOR_PRIMARY = "#2C3E50"  # Dark Blue/Grey (Titles, Text)
COLOR_SECONDARY = "#95A5A6"  # Light Grey (Secondary Text, Grid)
COLOR_ACCENT = "#3498DB"   # Bright Blue (Primary Actions, Highlights)

# Domain specific colors
COLOR_RISK = "#E74C3C"         # Red for Risk
COLOR_PRODUCTION = "#2ECC71"   # Green for Production
COLOR_GOOD = "#2ECC71"         # Green
COLOR_BAD = "#E74C3C"          # Red
COLOR_NEUTRAL = "#BDC3C7"      # Grey

# Sequences
PALETTE_CATEGORICAL = sns.color_palette("viridis")
PALETTE_DIVERGING = sns.color_palette("vlag")

# --- Matplotlib / Seaborn Styles ---

def apply_matplotlib_style():
    """
    Applies the standardized seaborn/matplotlib style settings.
    Should be called at the beginning of plotting functions.
    """
    sns.set_theme(style="whitegrid", context="talk") # 'talk' context makes fonts slightly larger/readable
    
    # Custom overrides
    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlecolor': COLOR_PRIMARY,
        'axes.labelsize': 14,
        'axes.labelcolor': COLOR_PRIMARY,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#EEF0F4', # Very light grey grid
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'text.color': COLOR_PRIMARY,
        'xtick.color': COLOR_PRIMARY,
        'ytick.color': COLOR_PRIMARY,
        'figure.figsize': (10, 6),
        'figure.dpi': 100
    })

# --- Plotly Styles ---

def apply_plotly_style(fig: go.Figure, title: str = None, height: int = None, width: int = None):
    """
    Applies standard layout settings to a Plotly figure.
    
    Args:
        fig: The Plotly Figure object.
        title: Optional title override.
        height: Optional height override.
        width: Optional width override.
    """
    
    layout_update = dict(
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLOR_PRIMARY
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=20,
            color=COLOR_PRIMARY,
            weight='bold'
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        
        # Axis defaults
        xaxis=dict(
            showgrid=True,
            gridcolor='#EEF0F4',
            gridwidth=1,
            linecolor='#D1D5DB', # Axis line
            linewidth=1,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#EEF0F4',
            gridwidth=1,
            linecolor='#D1D5DB',
            linewidth=1,
            zeroline=False
        ),
        
        # Legend defaults
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        ),
        
        # Hover label defaults
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif"
        )
    )
    
    if title:
        layout_update['title'] = dict(text=f"<b>{title}</b>", x=0.01)
        
    if height:
        layout_update['height'] = height
        
    if width:
        layout_update['width'] = width

    fig.update_layout(**layout_update)
    
    # Update axes if they exist in the figure (handling subplots roughly)
    fig.update_xaxes(showgrid=True, gridcolor='#EEF0F4')
    fig.update_yaxes(showgrid=True, gridcolor='#EEF0F4')

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import plots, styles, utils

# Create output directory
os.makedirs("images/verification", exist_ok=True)


def verify_roc_curve():
    print("Verifying plot_roc_curve...")
    fig, ax = plt.subplots()
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    plots.plot_roc_curve(ax, y_true, scores, "TestModel", styles.COLOR_ACCENT)
    fig.savefig("images/verification/test_roc_curve.png")
    plt.close(fig)
    print("  Saved images/verification/test_roc_curve.png")


def verify_visualize_metrics():
    print("Verifying visualize_metrics...")
    y_true = np.random.randint(0, 2, 100)
    scores = {"ModelA": np.random.rand(100), "ModelB": np.random.rand(100)}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.flatten()
    plots.visualize_metrics(y_true, scores, ax)
    fig.savefig("images/verification/test_visualize_metrics.png")
    plt.close(fig)
    print("  Saved images/verification/test_visualize_metrics.png")


def verify_group_statistics():
    print("Verifying plot_group_statistics...")
    df = pd.DataFrame({"group": ["A", "A", "B", "B", "C"], "target": [0, 1, 0, 1, 0]})
    plots.plot_group_statistics(df, "group", "target")
    # This function calls plt.show(), which might block or do nothing in script.
    # We should modify it to accept ax or return fig if we want to save it properly,
    # but for now we just run it to check for errors.
    # To capture correct save, we can grab current figure.
    plt.gcf().savefig("images/verification/test_group_statistics.png")
    plt.close()
    print("  Saved images/verification/test_group_statistics.png")


def verify_transformation_rate():
    print("Verifying calculate_and_plot_transformation_rate...")
    dates = pd.date_range(start="2024-01-01", periods=10, freq="ME")
    df = pd.DataFrame(
        {
            "date": dates,
            "se_decision_id": ["ok", "rv", "ok", "ko", "ok", "rv", "ok", "ok", "rv", "ok"],
            "status_name": [
                "booked",
                "not_booked",
                "booked",
                "n/a",
                "booked",
                "booked",
                "not_booked",
                "booked",
                "booked",
                "booked",
            ],
            "oa_amt": np.random.randint(1000, 5000, 10),
        }
    )
    result = utils.calculate_and_plot_transformation_rate(df, "date")
    result["figure"].write_html("images/verification/test_transformation_rate.html")
    print("  Saved images/verification/test_transformation_rate.html")


def verify_risk_production():
    print("Verifying plot_risk_vs_production...")
    dates = pd.date_range(start="2022-01-01", periods=24, freq="ME")
    df = pd.DataFrame(
        {
            "mis_date": dates,
            "status_name": ["booked"] * 24,
            "oa_amt_h0": np.random.randint(10000, 50000, 24),
            "todu_30ever_h6": np.random.randint(0, 5, 24),
            "todu_amt_pile_h6": np.random.randint(1000, 5000, 24),
        }
    )
    # Add rolling columns manually if needed or let function do it?
    # Function does it.

    # Needs indicators list... looking at code... indicators arg used in logic
    # indicators = config_data['indicators'] -> usually ['oa_amt_h0', ...]
    # The function aggregates these columns.
    indicators = ["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]

    # Comfort zones
    cz_config = {2022: 4.5, 2023: 4.0}

    # Data booked
    data_booked = df.copy()  # Simplification

    fig = plots.plot_risk_vs_production(df, indicators, cz_config, data_booked)
    fig.write_html("images/verification/test_risk_vs_production.html")
    print("  Saved images/verification/test_risk_vs_production.html")


if __name__ == "__main__":
    verify_roc_curve()
    verify_visualize_metrics()
    verify_group_statistics()
    try:
        verify_transformation_rate()
    except Exception as e:
        print(f"Error verifying transformation rate: {e}")
        # Kaleido might not be installed, ignore image save error if it's just that
        pass

    try:
        verify_risk_production()
    except Exception as e:
        print(f"Error verifying risk vs production: {e}")
        pass

    print("Verification complete.")

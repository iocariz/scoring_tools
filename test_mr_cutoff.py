import pandas as pd

from src.mr_pipeline import calculate_metrics_from_cuts

try:
    df_opt = pd.read_csv("output/premium/data/optimal_solution_base.csv")
    print("opt columns:", df_opt.columns.tolist())

    df_mr = pd.read_csv("output/premium/data/data_summary_desagregado_mr_base.csv")
    print("mr columns:", df_mr.columns.tolist())

    # Try running the same code
    variables = ["sc_octroi", "risk_score_rf"]
    inv_vars = ["risk_score_rf"]  # Assuming risk_score_rf is inverted

    res = calculate_metrics_from_cuts(df_mr, df_opt, variables, inv_vars)
    print("Metrics result:")
    if res is not None:
        print(res.to_string())
    else:
        print("Result is None")
except Exception as e:
    print("Error:", e)

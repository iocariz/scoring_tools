import pandas as pd

df_opt = pd.read_csv("output/premium/data/optimal_solution_base.csv")
df_mr = pd.read_csv("output/premium/data/data_summary_desagregado_mr_base.csv")

variables = ["sc_octroi_new_clus", "new_efx_clus"]
inv_vars = []  # Actually we can pass []

bins = sorted(df_mr[variables[0]].unique())
print("Unique bins in MR data:", bins)
print("Columns in optimal solution:", df_opt.columns[-10:])

for bin_val in bins:
    if bin_val in df_opt.columns:
        print(f"Found {bin_val} exactly")
    elif str(bin_val) in df_opt.columns:
        print(f"Found str({bin_val}) -> '{str(bin_val)}'")
    else:
        print(f"Missing: {bin_val} (type {type(bin_val)}) - we would check str={str(bin_val)}")
        if f"{float(bin_val):.1f}" in df_opt.columns:
            print(f"  But wait, {float(bin_val):.1f} IS in columns!")

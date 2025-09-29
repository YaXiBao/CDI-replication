#!/usr/bin/env python3
# run_pca_on_cdi_outputs.py
"""
Reads:
  - cdi_outputs/sales_with_available_dates.csv
  - cdi_outputs/marketcaps_at_availability.csv
Builds panel (date,ticker) with features ['sales','market_cap_usd'],
runs rolling_local_pca (default window=36 months, n_components=2),
and writes outputs to pca_outputs/.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pca_utils import rolling_local_pca

# Config - adjust if needed
SALES_CSV = os.path.join("cdi_outputs", "sales_with_available_dates.csv")
MC_CSV = os.path.join("cdi_outputs", "marketcaps_at_availability.csv")
OUT_DIR = "pca_outputs"
WINDOW_MONTHS = 60
N_COMPONENTS = 1
MIN_NOBS = 2   # min firms required at a date to produce PC scores

os.makedirs(OUT_DIR, exist_ok=True)

def load_and_build_panel(sales_csv, mc_csv):
    if not os.path.exists(sales_csv):
        raise FileNotFoundError(f"{sales_csv} not found.")
    if not os.path.exists(mc_csv):
        raise FileNotFoundError(f"{mc_csv} not found.")

    sales = pd.read_csv(sales_csv, parse_dates=['available_date','period_end'], dayfirst=False)
    mc = pd.read_csv(mc_csv, parse_dates=['available_date','close_used_date'], dayfirst=False)

    # normalize column names
    sales = sales.rename(columns={'available_date':'date','ticker':'ticker','total_revenue_usd':'sales'})
    mc = mc.rename(columns={'available_date':'date','ticker':'ticker','market_cap_usd':'market_cap_usd'})

    # We want a panel indexed by (date,ticker) with columns sales, market_cap_usd
    # Merge sales and marketcap (outer) then set MultiIndex
    merged = pd.merge(sales[['date','ticker','sales']], mc[['date','ticker','market_cap_usd']],
                      on=['date','ticker'], how='outer')
    # drop rows where both features missing
    merged = merged[~(merged['sales'].isna() & merged['market_cap_usd'].isna())].copy()
    # ensure date is timestamp (month-end or available_date as close to month)
    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.set_index(['date','ticker']).sort_index()

    return merged

def main():
    print("Loading sales and marketcap data...")
    panel = load_and_build_panel(SALES_CSV, MC_CSV)
    print("Panel shape (rows):", len(panel))
    print("Sample panel rows:")
    print(panel.head(8).to_string())

    features = ['sales','market_cap_usd']
    print(f"Running rolling_local_pca (window={WINDOW_MONTHS}, n_components={N_COMPONENTS}, min_nobs={MIN_NOBS}) ...")
    pcs_by_date, explained_df = rolling_local_pca(panel, features,
                                                  window_months=WINDOW_MONTHS,
                                                  n_components=N_COMPONENTS,
                                                  min_nobs=MIN_NOBS,
                                                  standardize=True)
    # save pcs_by_date stacked
    all_rows = []
    for d, dfpc in pcs_by_date.items():
        tmp = dfpc.reset_index().rename(columns={'index':'ticker'})
        tmp['date'] = pd.to_datetime(d)
        all_rows.append(tmp)
    if len(all_rows) > 0:
        pcs_all = pd.concat(all_rows, ignore_index=True)
        # order columns
        pc_cols = [c for c in pcs_all.columns if c.startswith('PC')]
        pcs_all = pcs_all[['date','ticker'] + pc_cols]
        out_pcs = os.path.join(OUT_DIR, "pcs_by_date.csv")
        pcs_all.to_csv(out_pcs, index=False)
        print("Saved PC scores:", out_pcs)
    else:
        print("No PC scores computed (not enough data or min_nobs too large).")

    # save explained_df
    if not explained_df.empty:
        explained_df = explained_df.sort_index()
        out_expl = os.path.join(OUT_DIR, "explained_by_date.csv")
        explained_df.to_csv(out_expl)
        print("Saved explained variance by date:", out_expl)
    else:
        print("No explained variance rows to save.")

    # Compute a simple cross-sectional measure: PC1 std across tickers at each date
    if len(all_rows) > 0:
        pcs_all['date'] = pd.to_datetime(pcs_all['date']).dt.to_period('M').dt.to_timestamp('M')
        if 'PC1' in pcs_all.columns:
            pc1_std = pcs_all.groupby('date')['PC1'].std().rename('pc1_crosssec_std').reset_index()
            out_pc1std = os.path.join(OUT_DIR, "pc1_crosssec_std.csv")
            pc1_std.to_csv(out_pc1std, index=False)
            print("Saved PC1 cross-sectional std:", out_pc1std)
        else:
            print("PC1 not present in pcs_by_date; skipping pc1 std computation.")
    print("PCA run complete. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
replicate_regression_with_pca_fixed.py

Robust regression of CDI on PCA-based uncertainty (pc1 cross-sectional std).
Provides diagnostics and fallbacks when there are not enough non-null observations.

Usage:
    python replicate_regression_with_pca_fixed.py
Outputs:
    pca_regression_outputs/merged_cdi_pc1_debug.csv
    (and regression text + plots if enough obs)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- config (change if needed) ---
CDI_MONTHLY = os.path.join("cdi_outputs", "cdi_timeseries_monthly.csv")
PC1_STD = os.path.join("pca_outputs", "pc1_crosssec_std.csv")
OUT_DIR = "pca_regression_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# try windows to attempt if 24-month rolling produces no obs
ROLL_WINDOWS = [24, 12]   # try 24 then 12
MIN_PERIODS = 6           # min_periods for rolling std
MIN_OBS_FOR_REG = 6       # minimum observations to run a regression
# -----------------------------------

def load_and_preview():
    if not os.path.exists(CDI_MONTHLY):
        raise FileNotFoundError(f"{CDI_MONTHLY} not found.")
    if not os.path.exists(PC1_STD):
        raise FileNotFoundError(f"{PC1_STD} not found.")

    cdi = pd.read_csv(CDI_MONTHLY)
    pc1 = pd.read_csv(PC1_STD)

    print("=== CDI file preview ===")
    print("Path:", CDI_MONTHLY)
    print("Columns:", list(cdi.columns))
    print("Rows:", len(cdi))
    print(cdi.head(5).to_string(index=False))
    print()

    print("=== PC1 file preview ===")
    print("Path:", PC1_STD)
    print("Columns:", list(pc1.columns))
    print("Rows:", len(pc1))
    print(pc1.head(8).to_string(index=False))
    print()

    return cdi, pc1

def normalize_cdi(cdi):
    # find date column
    date_col = None
    for c in cdi.columns:
        if c.lower().startswith('date') or 'available' in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = cdi.columns[0]
    # find numeric CDI column
    cdi_col = None
    for c in cdi.columns:
        if 'cdi' in c.lower() or 'spearman' in c.lower() or 'pearson' in c.lower():
            cdi_col = c
            break
    if cdi_col is None:
        numeric_cols = [c for c in cdi.columns if pd.api.types.is_numeric_dtype(cdi[c]) and c != date_col]
        if not numeric_cols:
            raise ValueError("No numeric CDI-like column found.")
        cdi_col = numeric_cols[0]

    cdi2 = cdi[[date_col, cdi_col]].rename(columns={date_col:'date', cdi_col:'cdi'})
    cdi2['date'] = pd.to_datetime(cdi2['date']).dt.to_period('M').dt.to_timestamp('M')
    return cdi2

def normalize_pc1(pc1):
    # assume first column is date, second is pc1 value
    if pc1.shape[1] < 2:
        raise ValueError("pc1_crosssec_std file must have at least two columns (date and value).")
    pc1 = pc1.copy()
    pc1 = pc1.rename(columns={pc1.columns[0]:'date', pc1.columns[1]:'pc1_crosssec_std'})
    pc1['date'] = pd.to_datetime(pc1['date']).dt.to_period('M').dt.to_timestamp('M')
    return pc1[['date','pc1_crosssec_std']]

def try_make_std_pc1(pc1, window, min_periods):
    s = pc1.set_index('date').sort_index()['pc1_crosssec_std']
    std_series = s.rolling(window=window, min_periods=min_periods).std().rename('std_pc1_{}m'.format(window))
    df = std_series.reset_index()
    return df

def main():
    cdi_raw, pc1_raw = load_and_preview()
    cdi = normalize_cdi(cdi_raw)
    pc1 = normalize_pc1(pc1_raw)

    # diagnostic ranges
    print("CDI date range:", cdi['date'].min(), "->", cdi['date'].max(), "n=", len(cdi))
    print("PC1 date range:", pc1['date'].min(), "->", pc1['date'].max(), "n=", len(pc1))
    # save debug merge before rolling
    debug_merge0 = pd.merge(cdi, pc1, on='date', how='outer').sort_values('date')
    debug_merge0.to_csv(os.path.join(OUT_DIR, "merged_cdi_pc1_debug.csv"), index=False)
    print("Wrote debug merged file (no rolling yet):", os.path.join(OUT_DIR, "merged_cdi_pc1_debug.csv"))

    # if pc1 values nonexistent
    if pc1['pc1_crosssec_std'].isna().all():
        print("ERROR: pc1_crosssec_std is all NaN. Cannot compute rolling std. Check pca_outputs/pcs_by_date.csv and pc1_crosssec_std.csv generation.")
        return

    # try rolling windows fallback
    std_df = None
    used_window = None
    for w in ROLL_WINDOWS:
        df_std = try_make_std_pc1(pc1, window=w, min_periods=MIN_PERIODS)
        nonnull = df_std['std_pc1_{}m'.format(w)].notna().sum()
        print(f"Rolling window {w} months produced {nonnull} non-null std rows (min_periods={MIN_PERIODS}).")
        if nonnull >= MIN_OBS_FOR_REG:
            std_df = df_std.rename(columns={f'std_pc1_{w}m':'std_pc1_24m'})  # name unify
            used_window = w
            break

    # fallback: if no rolling std had enough obs, try using raw pc1_crosssec_std as predictor
    fallback_using_raw = False
    if std_df is None:
        print("No rolling std found with enough observations. Will try using raw pc1_crosssec_std as predictor (z-scored).")
        fallback_using_raw = True
        pc1_z = pc1.copy()
        pc1_z['std_pc1_24m'] = (pc1_z['pc1_crosssec_std'] - pc1_z['pc1_crosssec_std'].mean()) / (pc1_z['pc1_crosssec_std'].std(ddof=0) + 1e-12)
        std_df = pc1_z[['date','std_pc1_24m']].copy()

    # merge with CDI
    merged = pd.merge(cdi, std_df, on='date', how='left').sort_values('date')
    # keep also raw pc1 for inspection
    merged = pd.merge(merged, pc1, on='date', how='left')
    merged.to_csv(os.path.join(OUT_DIR, "merged_cdi_pc1_final_debug.csv"), index=False)
    print("Wrote merged with predictor (final) to:", os.path.join(OUT_DIR, "merged_cdi_pc1_final_debug.csv"))

    # show counts
    total_rows = len(merged)
    nonnull_predictor = merged['std_pc1_24m'].notna().sum()
    nonnull_cdi = merged['cdi'].notna().sum()
    both_nonnull = merged.dropna(subset=['cdi','std_pc1_24m']).shape[0]
    print(f"Total rows: {total_rows}, non-null CDI: {nonnull_cdi}, non-null predictor: {nonnull_predictor}, both non-null: {both_nonnull}")

    if both_nonnull < MIN_OBS_FOR_REG:
        print(f"Not enough observations for regression (need >= {MIN_OBS_FOR_REG}).")
        print("Suggestions:")
        print(" - Lower MIN_PERIODS or try smaller rolling windows (edit ROLL_WINDOWS & MIN_PERIODS in script).")
        print(" - Ensure pca_outputs/pc1_crosssec_std.csv has values (check pcs_by_date.csv).")
        print(" - Consider using raw pc1_crosssec_std (already tried) or increasing panel pooling window in PCA.")
        return

    # run regression
    df_reg = merged.dropna(subset=['cdi','std_pc1_24m']).copy()
    X = sm.add_constant(df_reg['std_pc1_24m'])
    y = df_reg['cdi']

    # OLS HC1
    model_hc1 = sm.OLS(y, X).fit(cov_type='HC1')
    print("OLS HC1 summary:")
    print(model_hc1.summary())
    with open(os.path.join(OUT_DIR, "regression_stdpc1_hc1.txt"), "w") as f:
        f.write(model_hc1.summary().as_text())

    # Newey-West HAC (lags = 3)
    try:
        model_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':3})
        with open(os.path.join(OUT_DIR, "regression_stdpc1_hac_l3.txt"), "w") as f:
            f.write(model_hac.summary().as_text())
        print("Saved HAC regression (lags=3).")
    except Exception as e:
        print("HAC regression failed:", e)

    # plots
    # scatter
    plt.figure(figsize=(7,5))
    plt.scatter(df_reg['std_pc1_24m'], df_reg['cdi'], s=50, alpha=0.8)
    b = model_hc1.params
    xs = np.linspace(df_reg['std_pc1_24m'].min(), df_reg['std_pc1_24m'].max(), 50)
    plt.plot(xs, b['const'] + b['std_pc1_24m']*xs, color='C1', lw=2)
    plt.xlabel('std_pc1_24m')
    plt.ylabel('CDI')
    plt.title('CDI vs std_pc1_24m')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scatter_cdi_stdpc1.png'), dpi=150)
    plt.close()
    print("Saved scatter plot.")

    # timeseries
    plt.figure(figsize=(10,5))
    plt.plot(merged['date'], merged['cdi'], label='CDI', linewidth=2)
    plt.plot(merged['date'], merged['std_pc1_24m'], label='std_pc1_24m', linewidth=1.5)
    if 'pc1_crosssec_std' in merged.columns:
        plt.plot(merged['date'], merged['pc1_crosssec_std'], label='pc1_crosssec_std', alpha=0.6)
    plt.legend()
    plt.title('CDI and PC1-based uncertainty')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'timeseries_cdi_vs_stdpc1.png'), dpi=150)
    plt.close()
    print("Saved timeseries plot.")

    print("Regression outputs & plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()

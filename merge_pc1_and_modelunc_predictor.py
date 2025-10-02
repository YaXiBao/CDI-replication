#!/usr/bin/env python3
"""
merge_pc1_and_modelunc_predictor.py

Purpose:
 - Load PC1 outputs (prefer rolling std e.g. pca_outputs/pc1_crosssec_std_rolling_12m.csv,
   fallback to pca_outputs/pc1_crosssec_std.csv)
 - Load model_uncertainty.csv (expects a 'date' column and a 'std_wt_24m' column)
 - Merge into one monthly table; create mixed predictor:
       predictor = first_nonnull(std_pc1_roll, pc1_crosssec_std, std_wt_24m)
 - Create standardized (z-score) predictor and a clipped 0.05-0.95 mapped predictor for use as weight
 - Run OLS regressions: CDI ~ 1 + predictor_z (HC1) and CDI ~ 1 + predictor_z (Newey-West, lags=3)
 - Save outputs: merged CSV, regression txt results, scatter/time plots.

Outputs (folder pca_regression_outputs):
 - merged_with_mixed_predictor.csv
 - regression_hc1.txt, regression_nw.txt
 - plots: predictor_scatter.png, predictor_timeseries.png
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ----------------- CONFIG -----------------
PCA_DIR = Path("pca_outputs")
PC1_ROLL12 = PCA_DIR / "pc1_crosssec_std_rolling_12m.csv"
PC1_ROLL24 = PCA_DIR / "pc1_crosssec_std_rolling_24m.csv"
PC1_RAW = PCA_DIR / "pc1_crosssec_std.csv"

MODEL_UNC = Path("model_uncertainty.csv")   # your file (you said it's at repo root)
CDI_MONTHLY = Path("cdi_outputs/cdi_timeseries_monthly.csv")

OUT_DIR = Path("pca_regression_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Newey-West lags (for HAC)
NW_LAGS = 3

# Clip mapping range (for turning predictor into 0..1 weight then into [0.05,0.95])
CLIP_MIN = 0.05
CLIP_MAX = 0.95

# ----------------- helpers -----------------
def robust_read_csv(path, parse_dates=None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p, parse_dates=parse_dates)

def detect_date_col_and_cdi(df):
    # return (date_col, cdi_col)
    date_col = None
    for c in df.columns:
        if c.lower().startswith("date") or "available" in c.lower():
            date_col = c; break
    if date_col is None:
        # try parseable col
        for c in df.columns:
            try:
                pd.to_datetime(df[c].iloc[0:3])
                date_col = c; break
            except Exception:
                continue
    # find CDI numeric col
    cdi_col = None
    for c in df.columns:
        if 'cdi' in c.lower() or 'spearman' in c.lower() or 'pearson' in c.lower():
            cdi_col = c; break
    if cdi_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
        if not numeric_cols:
            raise ValueError("No numeric CDI column found in CDI file.")
        cdi_col = numeric_cols[0]
    return date_col, cdi_col

def to_month_start(ts):
    return pd.to_datetime(ts).dt.to_period('M').dt.to_timestamp('M')

def extract_intercept_slope(model, pred_name):
    params = model.params
    # intercept candidates
    intercept_keys = ['const','Const','CONSTANT','Intercept','intercept']
    intercept = None
    for k in intercept_keys:
        if k in params.index:
            intercept = float(params[k]); break
    if intercept is None:
        # fallback: if two params and pred_name present, take other as intercept
        if pred_name in params.index and len(params)==2:
            intercept = float(params.drop(index=pred_name).iloc[0])
        else:
            intercept = 0.0
    # slope
    slope = float(params[pred_name]) if pred_name in params.index else None
    if slope is None:
        # pick first non-intercept param
        for name in params.index:
            if name.lower() not in [k.lower() for k in intercept_keys]:
                slope = float(params[name]); break
    if slope is None:
        slope = 0.0
    return intercept, slope

# ----------------- main -----------------
def main():
    # 1) load pc1 file (prefer roll12 -> roll24 -> raw)
    pc1_df = None
    pc1_pred_col = None
    for candidate in [PC1_ROLL12, PC1_ROLL24, PC1_RAW]:
        if candidate.exists():
            pc1_df = pd.read_csv(candidate, parse_dates=['date'])
            # find appropriate col
            roll_cols = [c for c in pc1_df.columns if c.startswith('std_pc1_')]
            if roll_cols:
                pc1_pred_col = roll_cols[0]
            elif 'pc1_crosssec_std' in pc1_df.columns:
                pc1_pred_col = 'pc1_crosssec_std'
            else:
                # fallback to any std-like col
                std_cols = [c for c in pc1_df.columns if 'pc1' in c and 'std' in c]
                pc1_pred_col = std_cols[0] if std_cols else None
            print("Using PC1 file:", candidate, "-> predictor-col:", pc1_pred_col)
            break
    if pc1_df is None:
        raise FileNotFoundError("No PC1 output found. Run run_batch_pca.py first.")

    # normalize pc1_df date to month-timestamp
    pc1_df['date'] = to_month_start(pc1_df['date'])

    # 2) load model_uncertainty.csv
    if not MODEL_UNC.exists():
        raise FileNotFoundError(f"model_uncertainty.csv not found at {MODEL_UNC}")
    model_unc = pd.read_csv(MODEL_UNC, parse_dates=['date'])
    # try to find std_wt_24m col (case-insensitive)
    std_wt_col = None
    for c in model_unc.columns:
        if c.lower().startswith('std_wt') or 'std' in c.lower() and 'wt' in c.lower():
            std_wt_col = c; break
    if std_wt_col is None:
        # try explicit
        if 'std_wt_24m' in model_unc.columns:
            std_wt_col = 'std_wt_24m'
    if std_wt_col is None:
        raise ValueError("model_uncertainty.csv does not contain std_wt_24m (or similar) column.")
    model_unc['date'] = to_month_start(model_unc['date'])

    # 3) load cdi monthly
    if not CDI_MONTHLY.exists():
        raise FileNotFoundError(f"CDI monthly not found: {CDI_MONTHLY}")
    cdi = pd.read_csv(CDI_MONTHLY)
    date_col, cdi_col = detect_date_col_and_cdi(cdi)
    cdi = cdi[[date_col, cdi_col]].rename(columns={date_col:'date', cdi_col:'cdi'})
    cdi['date'] = to_month_start(pd.to_datetime(cdi['date']))

    # 4) merge: start with CDI, left-join pc1 and model_unc
    merged = cdi.merge(pc1_df[['date', pc1_pred_col]] if pc1_pred_col else pc1_df, on='date', how='left')
    merged = merged.merge(model_unc[['date', std_wt_col]], on='date', how='left')

    # 5) create mixed predictor
    # priority: pc1 rolling std (std_pc1_...) -> pc1_crosssec_std -> std_wt_24m
    # our pc1_pred_col already is either std_pc1_12m or pc1_crosssec_std
    merged = merged.rename(columns={pc1_pred_col: 'pc1_pred_temp', std_wt_col: 'std_wt_24m'})
    # if there is separate raw pc1_crosssec_std in pc1 file, prefer rolling if exists
    # attempt to read raw pc1 if present in pc1_df
    if 'pc1_crosssec_std' in pc1_df.columns:
        merged = merged.merge(pc1_df[['date','pc1_crosssec_std']], on='date', how='left')
    else:
        merged['pc1_crosssec_std'] = merged.get('pc1_crosssec_std', np.nan)

    # build predictor: pick first non-null among pc1_pred_temp, pc1_crosssec_std, std_wt_24m
    merged['predictor_raw'] = merged['pc1_pred_temp'].fillna(merged['pc1_crosssec_std']).fillna(merged['std_wt_24m'])

    # 6) create standardized predictor (z-score) using available (non-null) sample
    vals = merged['predictor_raw'].dropna()
    if len(vals) == 0:
        raise RuntimeError("No non-null predictor values available after merging. Check inputs.")
    mu = vals.mean()
    sigma = vals.std(ddof=0)  # population std for z-score; ddof=0 to match prior scripts maybe
    merged['predictor_z'] = (merged['predictor_raw'] - mu) / sigma

    # 7) create clipped 0.05..0.95 weight via min-max scaling then stretch to [CLIP_MIN, CLIP_MAX]
    # Use min/max computed from non-null predictor_raw
    mn = vals.min()
    mx = vals.max()
    if mx == mn:
        # degenerate: fallback to z-score mapping via logistic
        merged['predictor_clipped'] = 0.5
    else:
        scaled = (merged['predictor_raw'] - mn) / (mx - mn)  # 0..1 (may be NaN)
        merged['predictor_clipped'] = scaled * (CLIP_MAX - CLIP_MIN) + CLIP_MIN

    # 8) Save merged table
    out_csv = OUT_DIR / "merged_with_mixed_predictor.csv"
    merged.to_csv(out_csv, index=False)
    print("Saved merged table to:", out_csv)
    print("Merged head:\n", merged.head(12).to_string(index=False))

    # 9) Regression: use predictor_z as primary predictor
    pred = 'predictor_z'
    df_reg = merged.dropna(subset=['cdi', pred]).copy()
    print(f"Regression observations (cdi & {pred} non-null):", len(df_reg))
    if len(df_reg) == 0:
        print("No overlapping observations for regression. Exiting.")
        return

    # add const
    X = sm.add_constant(df_reg[pred], has_constant='add')
    y = df_reg['cdi']

    # HC1
    model_hc1 = sm.OLS(y, X).fit(cov_type='HC1')
    with open(OUT_DIR / "regression_hc1.txt", "w", encoding="utf-8") as f:
        f.write(model_hc1.summary().as_text())
    print("Saved HC1 regression to regression_hc1.txt")

    # Newey-West HAC
    model_nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    with open(OUT_DIR / "regression_nw.txt", "w", encoding="utf-8") as f:
        f.write(model_nw.summary().as_text())
    print("Saved Newey-West regression to regression_nw.txt")

    # 10) Plots: scatter + fitted line (use HC1 model params), and timeseries
    intercept, slope = extract_intercept_slope(model_hc1, pred)
    # scatter plot
    plt.figure(figsize=(7,5))
    plt.scatter(df_reg[pred], df_reg['cdi'], label='obs')
    xs = np.linspace(df_reg[pred].min(), df_reg[pred].max(), 60)
    ys = intercept + slope * xs
    plt.plot(xs, ys, color='C1', label='fit (HC1)')
    plt.xlabel(f'{pred} (z-score)')
    plt.ylabel('CDI')
    plt.title('CDI vs predictor_z (mixed)')
    plt.legend()
    scatter_path = OUT_DIR / "predictor_scatter.png"
    plt.tight_layout(); plt.savefig(scatter_path, dpi=150); plt.close()
    print("Saved scatter plot:", scatter_path)

    # timeseries plot (plot raw predictor and z)
    plt.figure(figsize=(10,4))
    plt.plot(merged['date'], merged['cdi'], label='CDI')
    if 'predictor_raw' in merged.columns:
        plt.plot(merged['date'], merged['predictor_raw'], label='predictor_raw')
    plt.plot(merged['date'], merged['predictor_z'], label='predictor_z')
    plt.legend(); plt.title('CDI and Mixed Predictor (raw & z)')
    ts_path = OUT_DIR / "predictor_timeseries.png"
    plt.tight_layout(); plt.savefig(ts_path, dpi=150); plt.close()
    print("Saved timeseries plot:", ts_path)

    print("Done. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()

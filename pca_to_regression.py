#!/usr/bin/env python3
# pca_to_regression.py (robust version)

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# CONFIG - 编辑路径/参数如需
PCA_DIR = Path("pca_outputs")
PC1_RAW = PCA_DIR / "pc1_crosssec_std.csv"
PC1_ROLL24 = PCA_DIR / "pc1_crosssec_std_rolling_24m.csv"
PC1_ROLL12 = PCA_DIR / "pc1_crosssec_std_rolling_12m.csv"
CDI_MONTHLY_CSV = Path(r"cdi_outputs/cdi_timeseries_monthly.csv")
OUT_DIR = Path("pca_regression_outputs")
NW_LAGS = 3   # Newey-West lags

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_pc1():
    # prefer rolling 12m file, then 24m, then raw
    if PC1_ROLL12.exists():
        df = pd.read_csv(PC1_ROLL12, parse_dates=['date'])
        roll_col = [c for c in df.columns if c.startswith('std_pc1_')]
        print("Using:", PC1_ROLL12)
        return df, (roll_col[0] if roll_col else None)
    if PC1_ROLL24.exists():
        df = pd.read_csv(PC1_ROLL24, parse_dates=['date'])
        roll_col = [c for c in df.columns if c.startswith('std_pc1_')]
        print("Using:", PC1_ROLL24)
        return df, (roll_col[0] if roll_col else None)
    if PC1_RAW.exists():
        df = pd.read_csv(PC1_RAW, parse_dates=['date'])
        print("Using raw pc1:", PC1_RAW)
        return df, 'pc1_crosssec_std'
    raise FileNotFoundError("No PC1 output found in pca_outputs. Run run_batch_pca.py first.")

def load_cdi():
    if not CDI_MONTHLY_CSV.exists():
        raise FileNotFoundError("CDI monthly file not found: " + str(CDI_MONTHLY_CSV))
    df = pd.read_csv(CDI_MONTHLY_CSV)
    # detect date col
    date_col = None
    for c in df.columns:
        if c.lower().startswith('date') or 'available' in c.lower():
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c].iloc[0:3])
                date_col = c
                break
            except:
                continue
    # detect cdi col
    cdi_col = None
    for c in df.columns:
        if 'cdi' in c.lower() or 'spearman' in c.lower() or 'pearson' in c.lower():
            cdi_col = c
            break
    if cdi_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
        if not numeric_cols:
            raise ValueError("No numeric CDI column found.")
        cdi_col = numeric_cols[0]
    df = df[[date_col, cdi_col]].rename(columns={date_col:'date', cdi_col:'cdi'})
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp('M')
    return df

def run_regressions(merged, pred_col='std_pc1_12m'):
    # prepare df for regression: drop NaNs
    df = merged.dropna(subset=['cdi', pred_col]).copy()
    if df.empty:
        print("No overlapping observations for regression with predictor:", pred_col)
        return None
    # Add constant robustly (force add)
    X = sm.add_constant(df[pred_col], has_constant='add')
    y = df['cdi']
    ols_hc1 = sm.OLS(y, X).fit(cov_type='HC1')
    ols_nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    return df, ols_hc1, ols_nw

def extract_intercept_slope(model, pred_col):
    """
    Robustly extract intercept and slope from model.params.
    Handles cases where 'const' or 'Intercept' are missing.
    Returns (intercept, slope).
    """
    params = model.params
    # intercept key candidates
    intercept_keys = ['const','Const','CONSTANT','Intercept','intercept']
    intercept = None
    for k in intercept_keys:
        if k in params.index:
            intercept = float(params[k])
            break
    if intercept is None:
        # no named intercept found. If there are two params, assume first is intercept
        if len(params) == 2:
            # find param that is not the predictor
            if pred_col in params.index:
                intercept = float(params.drop(index=pred_col).iloc[0])
            else:
                intercept = float(params.iloc[0])
        else:
            # fallback to 0
            intercept = 0.0

    # slope: try to get by pred_col name, otherwise take first non-intercept param
    slope = None
    if pred_col in params.index:
        slope = float(params[pred_col])
    else:
        # try find any param that is not intercept
        for name in params.index:
            if name.lower() not in [k.lower() for k in intercept_keys]:
                # choose this as slope
                slope = float(params[name])
                break
    if slope is None:
        # fallback 0
        slope = 0.0
    return intercept, slope

def plot_and_save(merged, df_reg, pred_col, model, out_prefix):
    # Ensure pred_col present in df_reg
    if pred_col not in df_reg.columns:
        raise KeyError(f"Predictor '{pred_col}' not present in regression dataframe.")
    # Use robust extractor for intercept and slope
    intercept, slope = extract_intercept_slope(model, pred_col)

    # scatter + fit
    plt.figure(figsize=(7,5))
    plt.scatter(df_reg[pred_col], df_reg['cdi'], label='obs')
    xs = np.linspace(df_reg[pred_col].min(), df_reg[pred_col].max(), 50)
    ys = intercept + slope * xs
    plt.plot(xs, ys, color='C1', label=f'fit')
    plt.xlabel(pred_col); plt.ylabel('CDI')
    plt.title(f'CDI vs {pred_col}')
    plt.legend()
    plt.tight_layout()
    p = OUT_DIR / f"{out_prefix}_scatter.png"
    plt.savefig(p, dpi=150); plt.close()
    print("Saved scatter:", p)

    # time series
    plt.figure(figsize=(10,4))
    plt.plot(merged['date'], merged['cdi'], label='CDI')
    if pred_col in merged.columns:
        plt.plot(merged['date'], merged[pred_col], label=pred_col)
    plt.legend()
    plt.title("CDI and predictor over time")
    p2 = OUT_DIR / f"{out_prefix}_timeseries.png"
    plt.savefig(p2, dpi=150); plt.close()
    print("Saved timeseries:", p2)

if __name__ == "__main__":
    pc1_df, pc1_col = load_pc1()
    print("PC1 columns:", pc1_df.columns.tolist())
    print("pc1_col chosen:", pc1_col)

    cdi = load_cdi()
    merged = pd.merge(cdi, pc1_df, on='date', how='left')
    merged.to_csv(OUT_DIR / "merged_cdi_pc1.csv", index=False)
    print("Saved merged:", OUT_DIR / "merged_cdi_pc1.csv")

    # choose predictor column
    pred_col = None
    for c in merged.columns:
        if c.startswith('std_pc1_'):
            pred_col = c
            break
    if pred_col is None and 'pc1_crosssec_std' in merged.columns:
        pred_col = 'pc1_crosssec_std'

    if pred_col is None:
        print("No suitable predictor column found in merged. Columns:", merged.columns.tolist())
    else:
        print("Using predictor column:", pred_col)
        out = run_regressions(merged, pred_col=pred_col)
        if out is None:
            print("No regression run (no overlapping obs).")
        else:
            df_reg, model_hc1, model_nw = out
            # save regression summaries
            with open(OUT_DIR / "regression_hc1.txt","w",encoding="utf-8") as f:
                f.write(model_hc1.summary().as_text())
            with open(OUT_DIR / "regression_nw.txt","w",encoding="utf-8") as f:
                f.write(model_nw.summary().as_text())
            print("Saved regression outputs (HC1 & Newey-West).")
            plot_and_save(merged, df_reg, pred_col, model_hc1, "pc1_regression")
            print("Regression done. Observations:", len(df_reg))

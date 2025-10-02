#!/usr/bin/env python3
"""
run_batch_pca.py (fixed)

Batch PCA across pooled firm-date observations.

Fixes:
 - use groupby.transform instead of apply when creating sales_share / mc_share
 - coerce numeric columns to float to avoid dtype buffer mismatch
 - safe handling for zeros / missing values

Outputs into pca_outputs/
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------- CONFIG -----------------
PATH_SALES = r"C:\Users\yaxib\OneDrive\Desktop\CDI-replication\cdi_outputs\sales_with_available_dates.csv"
PATH_MARKETCAP = r"C:\Users\yaxib\OneDrive\Desktop\CDI-replication\cdi_outputs\marketcaps_at_availability.csv"
OUT_DIR = Path("pca_outputs")

FEATURE_SET = "shares"   # 'shares' | 'log' | 'both'
ROLL_WINDOW_MONTHS = 12
ROLL_MIN_PERIODS = 3
MIN_FIRMS_PER_MONTH = 2
N_COMPONENTS = 1
RANDOM_STATE = 0
# ------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p)

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def prepare_panel(sales_path, mc_path):
    sales = safe_read_csv(sales_path)
    mc = safe_read_csv(mc_path)

    s_date_col = find_col(sales, ['available_date','period_end','date','available'])
    s_ticker_col = find_col(sales, ['ticker','symbol'])
    s_sales_col = find_col(sales, ['total_revenue_usd','total_revenue','revenue','sales','totalrevenue_usd'])
    if s_date_col is None or s_ticker_col is None or s_sales_col is None:
        raise ValueError("Could not detect required columns in sales csv.")
    sales = sales.rename(columns={s_date_col:'date', s_ticker_col:'ticker', s_sales_col:'sales'})

    m_date_col = find_col(mc, ['available_date','date','period_end'])
    m_ticker_col = find_col(mc, ['ticker','symbol'])
    m_mc_col = find_col(mc, ['market_cap_usd','market_cap','marketcap','mc'])
    if m_date_col is None or m_ticker_col is None or m_mc_col is None:
        raise ValueError("Could not detect required columns in marketcap csv.")
    mc = mc.rename(columns={m_date_col:'date', m_ticker_col:'ticker', m_mc_col:'market_cap_usd'})

    sales['date'] = pd.to_datetime(sales['date'])
    mc['date'] = pd.to_datetime(mc['date'])

    panel = pd.merge(
        sales[['date','ticker','sales']],
        mc[['date','ticker','market_cap_usd']],
        on=['date','ticker'], how='outer'
    )

    # coerce numeric to floats to avoid dtype issues
    panel['sales'] = pd.to_numeric(panel['sales'], errors='coerce').astype(float)
    panel['market_cap_usd'] = pd.to_numeric(panel['market_cap_usd'], errors='coerce').astype(float)

    panel = panel[~(panel['sales'].isna() & panel['market_cap_usd'].isna())].copy()
    panel = panel.sort_values(['date','ticker']).reset_index(drop=True)
    return panel

def compute_features(panel, feature_set='shares'):
    df = panel.copy()
    # Ensure numeric floats (already coerced in prepare_panel but ensure again)
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').astype(float)
    df['market_cap_usd'] = pd.to_numeric(df['market_cap_usd'], errors='coerce').astype(float)

    # compute per-date sums safely
    sales_sum = df.groupby('date')['sales'].transform('sum')  # aligned Series
    mc_sum = df.groupby('date')['market_cap_usd'].transform('sum')

    if feature_set in ('shares','both'):
        # use transform to produce aligned series
        # avoid division by zero: only divide where sum>0
        df['sales_share'] = np.where(sales_sum > 0, df['sales'] / sales_sum, np.nan)
        df['mc_share'] = np.where(mc_sum > 0, df['market_cap_usd'] / mc_sum, np.nan)

    if feature_set in ('log','both'):
        # log only where >0
        df['log_sales'] = np.where(df['sales'] > 0, np.log(df['sales']), np.nan)
        df['log_mc'] = np.where(df['market_cap_usd'] > 0, np.log(df['market_cap_usd']), np.nan)

    if feature_set == 'shares':
        feature_cols = ['sales_share','mc_share']
    elif feature_set == 'log':
        feature_cols = ['log_sales','log_mc']
    elif feature_set == 'both':
        feature_cols = ['sales_share','mc_share','log_sales','log_mc']
    else:
        raise ValueError("Unknown FEATURE_SET")

    # drop rows where all features are nan
    nonnull_count = df[feature_cols].notna().sum(axis=1)
    df = df[nonnull_count > 0].copy()
    return df, feature_cols

def pool_and_fit_pca(df_features, feature_cols, n_components=1):
    # create X and drop rows that have any NaN in feature_cols
    X_df = df_features[feature_cols].copy()
    mask = X_df.notna().all(axis=1)
    if mask.sum() == 0:
        raise ValueError("No fully-observed rows for chosen features. Consider other FEATURE_SET or imputation.")
    X_full = X_df[mask].values.astype(float)
    idx_full = df_features.index[mask]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_full)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(Xs)

    scores = pca.transform(Xs)
    pooled_df = df_features.loc[idx_full, ['date','ticker']].copy().reset_index(drop=True)
    for k in range(n_components):
        pooled_df[f'pc{k+1}_score'] = scores[:,k]
    return scaler, pca, pooled_df, pca.explained_variance_ratio_

def compute_monthly_pc1_std(pooled_df, min_firms=MIN_FIRMS_PER_MONTH):
    if 'pc1_score' not in pooled_df.columns:
        pc_cols = [c for c in pooled_df.columns if c.endswith('_score')]
        if not pc_cols:
            raise ValueError("No pc score column found.")
        pooled_df = pooled_df.rename(columns={pc_cols[0]:'pc1_score'})
    grouped = pooled_df.groupby('date')['pc1_score']
    pc1_sd = grouped.std(ddof=1).rename('pc1_crosssec_std')
    pc1_count = grouped.count().rename('n_firms')
    pc1_df = pd.concat([pc1_sd, pc1_count], axis=1).reset_index()
    pc1_df.loc[pc1_df['n_firms'] < min_firms, 'pc1_crosssec_std'] = np.nan
    return pc1_df

def apply_rolling(pc1_df, window_months=ROLL_WINDOW_MONTHS, min_periods=ROLL_MIN_PERIODS):
    if window_months is None:
        return pc1_df
    s = pc1_df.set_index('date').sort_index()['pc1_crosssec_std']
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp('M')
    rolling_std = s.rolling(window=window_months, min_periods=min_periods).std()
    out = pc1_df.copy()
    out[f'std_pc1_{window_months}m'] = out['date'].map(rolling_std.to_dict())
    return out

def main():
    try:
        print("=== Batch PCA (fixed) start ===")
        panel = prepare_panel(PATH_SALES, PATH_MARKETCAP)
        print(f"Panel rows: {len(panel)}  sample:")
        print(panel.head(10).to_string(index=False))

        df_features, feature_cols = compute_features(panel, FEATURE_SET)
        print("\nFeature columns used:", feature_cols)
        print("Rows with at least one feature:", len(df_features))

        scaler, pca, pooled_df, explained = pool_and_fit_pca(df_features, feature_cols, n_components=N_COMPONENTS)
        print("\nPCA explained variance ratio:", explained)

        pcs_out = OUT_DIR / "pcs_by_date.csv"
        pooled_df.to_csv(pcs_out, index=False)
        print("Saved pooled PC scores to:", pcs_out)

        pc1_df = compute_monthly_pc1_std(pooled_df, min_firms=MIN_FIRMS_PER_MONTH)
        pc1_out = OUT_DIR / "pc1_crosssec_std.csv"
        pc1_df[['date','pc1_crosssec_std','n_firms']].to_csv(pc1_out, index=False)
        print("Saved PC1 cross-sectional std to:", pc1_out)
        print(pc1_df.head(12).to_string(index=False))

        if ROLL_WINDOW_MONTHS is not None:
            pc1_roll = apply_rolling(pc1_df, window_months=ROLL_WINDOW_MONTHS, min_periods=ROLL_MIN_PERIODS)
            roll_out = OUT_DIR / f"pc1_crosssec_std_rolling_{ROLL_WINDOW_MONTHS}m.csv"
            pc1_roll.to_csv(roll_out, index=False)
            print("Saved rolling pc1 std to:", roll_out)
            print(pc1_roll.head(12).to_string(index=False))

        print("\n=== Batch PCA done ===")
    except Exception as e:
        print("ERROR:", type(e).__name__, e)
        raise

if __name__ == "__main__":
    main()

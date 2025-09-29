#!/usr/bin/env python3
"""
replicate_cdi_with_uncertainty_gen_with_pca.py

Generates model_uncertainty.csv (SE_t, wt_t, std_wt_24m) from close_prices,
and optionally augments/replaces the uncertainty proxy with a PCA-based
measure (pc1_crosssec_std -> std_pc1_24m) if pca_outputs/pc1_crosssec_std.csv exists.

Outputs (OUT_DIR):
 - model_uncertainty.csv
 - cdi_wt_merged.csv (and cdi_wt_merged_with_pc1.csv when PCA present)
 - regression_results_using_pc1_or_stdwt.txt (or regression_results.txt)
 - plot_cdi_vs_stdwt_scatter.png
 - plot_cdi_vs_stdwt.png

Usage:
    python replicate_cdi_with_uncertainty_gen_with_pca.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ----------------- user config (edit as needed) -----------------
# input files
CLOSES_CSV = r"cdi_outputs/close_prices.csv"   # path to your closes (wide or long)
CDI_MONTHLY_CSV = r"cdi_outputs/cdi_timeseries_monthly.csv"  # monthly CDI
# PCA input (optional)
PC1_STD_CSV = r"pca_outputs/pc1_crosssec_std.csv"  # produced by run_pca_on_cdi_outputs.py
# outputs
MODEL_UNC_CSV = "model_uncertainty.csv"   # saved in working dir
OUT_DIR = "cdi_paper_replication_outputs"

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","JNJ","PG"]

# Rolling windows and thresholds
AR_ROLL_WINDOW = 12           # months to fit AR(1)
SE_ROLL_WINDOW = 24           # months for RMSE of errors (SE)
ROLL_STD_WINDOW = 24          # months for rolling std of wt (paper uses 24)
MIN_PERIODS_ROLL_STD = 12     # min periods for rolling std
MIN_FIRMS_FOR_SE_MEAN = 1     # min tickers to average SE across tickers
MIN_OBS_FOR_REG = 6           # min monthly obs to run regression

# PCA fallback windows (try these in order to build std_pc1)
PCA_ROLL_WINDOWS_TRY = [24, 12, 6]
PCA_MIN_PERIODS_TRY = [12, 6, 3]

# -----------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helper functions ----------
def read_close_prices(path):
    raw = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in raw.columns}
    if 'date' in cols_lower:
        date_col = cols_lower['date']
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.rename(columns={date_col: 'Date'})
    else:
        first = raw.columns[0]
        try:
            raw[first] = pd.to_datetime(raw[first])
            raw = raw.rename(columns={first: 'Date'})
        except Exception:
            raise ValueError("Could not detect a Date column in close_prices.csv")
    upper_cols = [c.upper() for c in raw.columns]
    if all(t in upper_cols for t in TICKERS):
        mapping = {}
        for c in raw.columns:
            if c.upper() in TICKERS:
                mapping[c] = c
        df = raw[['Date'] + list(mapping.keys())].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df.columns = [c.upper() for c in df.columns]
        df = df.reindex(columns=TICKERS)
        return df
    if 'ticker' in (c.lower() for c in raw.columns) or 'symbol' in (c.lower() for c in raw.columns):
        name_map = {c.lower(): c for c in raw.columns}
        ticker_col = name_map.get('ticker', name_map.get('symbol'))
        close_col = None
        for cand in ['close','adjclose','adj_close','close_price','price']:
            if cand in name_map:
                close_col = name_map[cand]
                break
        if close_col is None:
            raise ValueError("Long format detected but no close column found.")
        df = raw.rename(columns={ticker_col: 'ticker', close_col: 'close'})
        df['Date'] = pd.to_datetime(df['Date'])
        pivot = df.pivot(index='Date', columns='ticker', values='close')
        pivot.columns = [str(c).upper() for c in pivot.columns]
        pivot = pivot.reindex(columns=TICKERS)
        pivot = pivot.sort_index()
        return pivot
    col_map = {}
    for c in raw.columns:
        for t in TICKERS:
            if t in c.upper():
                col_map[c] = t
    if col_map:
        raw['Date'] = pd.to_datetime(raw['Date'])
        df = raw[['Date'] + list(col_map.keys())].copy().set_index('Date').rename(columns=col_map)
        df.columns = [c.upper() for c in df.columns]
        df = df.reindex(columns=TICKERS)
        df = df.sort_index()
        return df
    raise ValueError("Unrecognized close_prices.csv layout. Provide wide Date+tickers or long Date,ticker,close.")

def month_end_prices(close_df):
    close_df = close_df.copy()
    close_df.index = pd.to_datetime(close_df.index)
    me = close_df.groupby([close_df.index.year, close_df.index.month]).apply(lambda g: g.iloc[-1])
    me.index = pd.to_datetime([pd.Timestamp(year=int(y), month=int(m), day=1) + relativedelta(months=1) - pd.Timedelta(days=1)
                               for (y,m) in me.index])
    me.index.name = 'Date'
    me = me.sort_index()
    me = me[~me.index.duplicated(keep='first')]
    return me

def monthly_returns_from_prices(month_end_df):
    r = month_end_df.pct_change().dropna(how='all')
    r.index = pd.to_datetime(r.index.to_period('M').to_timestamp())
    return r

def rolling_ar1_forecast_errors(series, window=AR_ROLL_WINDOW):
    s = series.dropna().copy()
    if s.shape[0] < window + 1:
        return pd.Series(index=s.index, data=np.nan)
    errors = pd.Series(index=s.index, data=np.nan)
    vals = s.values
    for t in range(window, len(vals)):
        y = vals[(t - window + 0):(t)]
        r_lag = vals[(t - window - 1):(t - 1)]
        if len(r_lag) != len(y):
            continue
        X = np.column_stack([np.ones(len(r_lag)), r_lag])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b = coef[0], coef[1]
        except Exception:
            continue
        r_tm1 = vals[t - 1]
        pred = a + b * r_tm1
        err = vals[t] - pred
        errors.iloc[t] = err
    return errors

def rolling_rmse_of_errors(err_series, window=SE_ROLL_WINDOW):
    sq = err_series.pow(2)
    rmse = sq.rolling(window=window, min_periods=1).mean().pow(0.5)
    return rmse

# PCA helper: attempt to build std_pc1_24m using a series and multiple roll windows
def build_std_pc1_variant(pc1_df, windows_try=PCA_ROLL_WINDOWS_TRY, min_periods_try=PCA_MIN_PERIODS_TRY):
    """
    pc1_df: DataFrame with columns ['date','pc1_crosssec_std'] (date parsed)
    returns: tuple (std_pc1_df, used_window, used_min_periods)
      - std_pc1_df: DataFrame columns ['date','std_pc1_24m'] (name uses window chosen)
    """
    if pc1_df is None or pc1_df.empty:
        return None, None, None
    pc1 = pc1_df.copy()
    pc1['date'] = pd.to_datetime(pc1['date']).dt.to_period('M').dt.to_timestamp('M')
    pc1 = pc1.sort_values('date').set_index('date')
    for w, mp in zip(windows_try, min_periods_try):
        colname = f'std_pc1_{w}m'
        s = pc1['pc1_crosssec_std'].rolling(window=w, min_periods=mp).std()
        nonnull = s.notna().sum()
        if nonnull >= MIN_OBS_FOR_REG:
            out = s.reset_index().rename(columns={colname: 'std_pc1_24m'}).rename(columns={s.name: 'std_pc1_24m'})
            # Actually s has name 'pc1_crosssec_std' so we reset differently:
            out = s.reset_index().rename(columns={0: 'std_pc1_24m'})
            # but safer:
            out = pd.DataFrame({'date': s.index, 'std_pc1_24m': s.values})
            return out, w, mp
    # if none passed, return raw z-scored series as fallback (may be used)
    raw = pc1.reset_index()[['date','pc1_crosssec_std']].copy()
    raw['std_pc1_24m'] = (raw['pc1_crosssec_std'] - raw['pc1_crosssec_std'].mean()) / (raw['pc1_crosssec_std'].std(ddof=0) + 1e-12)
    return raw[['date','std_pc1_24m']], None, None

# ------------------ Main pipeline ------------------
def main():
    print("Loading month-end prices from:", CLOSES_CSV)
    close_df = read_close_prices(CLOSES_CSV)
    close_df = close_df.sort_index()
    print("Close data range:", close_df.index.min(), "to", close_df.index.max())

    me_prices = month_end_prices(close_df)
    print("Month-end prices from", me_prices.index.min(), "to", me_prices.index.max())

    monthly_rets = monthly_returns_from_prices(me_prices)
    print("Monthly returns shape:", monthly_rets.shape)

    # compute per-ticker forecast errors and rolling RMSE (SE_t per ticker)
    se_tickers = {}
    for tkr in TICKERS:
        if tkr not in monthly_rets.columns:
            print(f"Ticker {tkr} missing in returns — skipping.")
            continue
        series = monthly_rets[tkr].dropna()
        if series.shape[0] < AR_ROLL_WINDOW + 1:
            print(f"Ticker {tkr} has {series.shape[0]} months < AR_ROLL_WINDOW+1 ({AR_ROLL_WINDOW+1}) — SE will be sparse.")
        errs = rolling_ar1_forecast_errors(series, window=AR_ROLL_WINDOW)
        se_series = rolling_rmse_of_errors(errs, window=SE_ROLL_WINDOW)
        se_series.name = tkr
        se_tickers[tkr] = se_series
        print(f"Computed SE (ticker={tkr}) — {se_series.dropna().shape[0]} non-null rows")

    # unified DataFrame of SEs across tickers (align by month)
    se_df = pd.DataFrame(se_tickers)
    se_df = se_df.sort_index()

    # cross-sectional mean across tickers to get a single SE_t series
    se_count = se_df.notna().sum(axis=1)
    se_mean = se_df.mean(axis=1, skipna=True)
    se_mean[se_count < MIN_FIRMS_FOR_SE_MEAN] = np.nan
    se_mean = se_mean.rename("se")
    se_mean.index = pd.to_datetime(se_mean.index.to_period('M').to_timestamp())

    # compute Sigma_t (cumulative mean of se^2) and wt
    se2 = se_mean.pow(2)
    sigma_cummean = se2.expanding(min_periods=1).mean()
    wt = sigma_cummean / (se2 + sigma_cummean)
    wt = wt.rename("wt")
    # compute rolling std of wt (ROLL_STD_WINDOW adjustable)
    std_wt = wt.rolling(window=ROLL_STD_WINDOW, min_periods=MIN_PERIODS_ROLL_STD).std().rename("std_wt_24m")

    # create model_unc DataFrame robustly
    model_unc = pd.concat([se_mean, wt, std_wt], axis=1)
    model_unc = model_unc.reset_index()
    first_col = model_unc.columns[0]
    if first_col != 'date':
        model_unc = model_unc.rename(columns={first_col: 'date'})
    model_unc['date'] = pd.to_datetime(model_unc['date']).dt.to_period('M').dt.to_timestamp('M')
    for col in ['se', 'wt', 'std_wt_24m']:
        if col not in model_unc.columns:
            model_unc[col] = np.nan
    model_unc = model_unc[['date', 'se', 'wt', 'std_wt_24m']]
    model_unc = model_unc.sort_values('date').reset_index(drop=True)

    print("DEBUG model_unc columns:", list(model_unc.columns))
    print("DEBUG model_unc head:\n", model_unc.head().to_string(index=False))
    model_unc.to_csv(MODEL_UNC_CSV, index=False)
    print("Saved model_uncertainty.csv ->", MODEL_UNC_CSV)

    # load CDI monthly
    print("Loading CDI monthly:", CDI_MONTHLY_CSV)
    cdi = pd.read_csv(CDI_MONTHLY_CSV)
    # detect date and cdi columns
    date_col = None
    for c in cdi.columns:
        if c.lower().startswith('date') or c.lower().startswith('available'):
            date_col = c
            break
    if date_col is None:
        for c in cdi.columns:
            try:
                pd.to_datetime(cdi[c].iloc[0:3])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise ValueError("Could not find date column in CDI monthly CSV.")
    cdi_col = None
    for c in cdi.columns:
        if 'cdi' in c.lower() or 'spearman' in c.lower() or 'pearson' in c.lower():
            cdi_col = c
            break
    if cdi_col is None:
        numeric_cols = [c for c in cdi.columns if pd.api.types.is_numeric_dtype(cdi[c]) and c != date_col]
        if not numeric_cols:
            raise ValueError("No numeric CDI column found in CDI CSV.")
        cdi_col = numeric_cols[0]
    cdi = cdi[[date_col, cdi_col]].rename(columns={date_col:'date', cdi_col:'cdi'})
    cdi['date'] = pd.to_datetime(cdi['date']).dt.to_period('M').dt.to_timestamp('M')

    # Merge original model_unc with CDI
    merged = pd.merge(cdi, model_unc[['date','se','wt','std_wt_24m']], on='date', how='left')
    merged.to_csv(os.path.join(OUT_DIR, "cdi_wt_merged.csv"), index=False)
    print("Saved merged file:", os.path.join(OUT_DIR, "cdi_wt_merged.csv"))

    # --- PCA integration (optional) ---
    if os.path.exists(PC1_STD_CSV):
        print("Found PCA pc1_crosssec_std file:", PC1_STD_CSV)
        try:
            pc1 = pd.read_csv(PC1_STD_CSV, parse_dates=['date'])
        except Exception:
            # try without parse_dates
            pc1 = pd.read_csv(PC1_STD_CSV)
            if 'date' in pc1.columns:
                pc1['date'] = pd.to_datetime(pc1['date'])
        # canonicalize columns
        if pc1.shape[1] >= 2 and 'pc1_crosssec_std' not in pc1.columns:
            pc1.columns = [pc1.columns[0], 'pc1_crosssec_std'] + list(pc1.columns[2:])
        print("PC1 preview:")
        print(pc1.head().to_string(index=False))
        # build std_pc1_24m (tries multiple windows; falls back to z-scored raw)
        std_pc1_df, used_w, used_mp = build_std_pc1_variant(pc1)
        if std_pc1_df is None:
            print("Could not build std_pc1_24m from PCA output.")
            std_pc1_df = None
        else:
            print("Built std_pc1_24m: rows =", len(std_pc1_df), "used_window =", used_w, "used_min_periods =", used_mp)
            std_pc1_df['date'] = pd.to_datetime(std_pc1_df['date']).dt.to_period('M').dt.to_timestamp('M')
            merged = pd.merge(merged, std_pc1_df[['date','std_pc1_24m']], on='date', how='left')
            merged.to_csv(os.path.join(OUT_DIR, "cdi_wt_merged_with_pc1.csv"), index=False)
            print("Saved merged file with PCA predictor:", os.path.join(OUT_DIR, "cdi_wt_merged_with_pc1.csv"))
    else:
        print("PCA pc1 file not found; skipping PCA augmentation. Expected at:", PC1_STD_CSV)
        std_pc1_df = None

    # Choose regression predictor: prefer std_pc1_24m if available, else std_wt_24m
    if 'std_pc1_24m' in merged.columns:
        merged['std_unc_for_reg'] = merged['std_pc1_24m'].fillna(merged['std_wt_24m'])
    else:
        merged['std_unc_for_reg'] = merged['std_wt_24m']

    # Save merged final debug
    merged.to_csv(os.path.join(OUT_DIR, "cdi_wt_merged_final_debug.csv"), index=False)
    print("Saved final merged debug file:", os.path.join(OUT_DIR, "cdi_wt_merged_final_debug.csv"))

    # regression
    df_reg = merged.dropna(subset=['cdi','std_unc_for_reg']).copy()
    print("Observations available for regression:", len(df_reg))
    if df_reg.empty or len(df_reg) < MIN_OBS_FOR_REG:
        print(f"No/too few observations for regression (need at least {MIN_OBS_FOR_REG}).")
        print("Diagnostics:")
        print(" - merged.head():\n", merged.head().to_string(index=False))
        print(" - merged.tail():\n", merged.tail().to_string(index=False))
        print("Suggestions:")
        print(" - lower MIN_FIRMS_FOR_SE_MEAN, reduce AR_ROLL_WINDOW/SE_ROLL_WINDOW, or reduce ROLL_STD_WINDOW.")
        print(" - increase PCA coverage (adjust PCA script: WINDOW_MONTHS, MIN_NOBS) or use batch PCA.")
        print(" - try using raw pc1_crosssec_std (no rolling std) or smaller rolling windows (6/12 months).")
        return

    X = sm.add_constant(df_reg['std_unc_for_reg'])
    y = df_reg['cdi']
    model = sm.OLS(y, X).fit(cov_type='HC1')
    print(model.summary())
    with open(os.path.join(OUT_DIR, "regression_results_using_pc1_or_stdwt.txt"), "w") as f:
        f.write(model.summary().as_text())
    print("Saved regression results to regression_results_using_pc1_or_stdwt.txt")

    # scatter + fit plot
    plt.figure(figsize=(8,6))
    plt.scatter(df_reg['std_unc_for_reg'], df_reg['cdi'], label='obs')
    xs = np.linspace(df_reg['std_unc_for_reg'].min(), df_reg['std_unc_for_reg'].max(), 50)
    ys = model.params['const'] + model.params['std_unc_for_reg'] * xs
    plt.plot(xs, ys, color='C1', label=f'fit')
    plt.xlabel("uncertainty (std_pc1_24m preferred, fallback std_wt_24m)")
    plt.ylabel("CDI (cdi)")
    plt.title("CDI vs uncertainty (PCA or generated wt)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_cdi_vs_stdwt_scatter.png"), dpi=150)
    plt.close()
    print("Saved scatter plot")

    # timeseries plot
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(merged['date'], merged['cdi'], label='CDI', linewidth=2)
    ax2 = ax1.twinx()
    if 'std_pc1_24m' in merged.columns:
        ax2.plot(merged['date'], merged['std_pc1_24m'], color='C1', label='std_pc1_24m', linewidth=1.5)
    ax2.plot(merged['date'], merged['std_wt_24m'], color='C2', label='std_wt_24m', linewidth=1.0, alpha=0.6)
    ax1.set_xlabel('Date'); ax1.set_ylabel('CDI'); ax2.set_ylabel('std (uncertainty)')
    lines_labels = [l.get_label() for l in ax1.get_lines()] + [l.get_label() for l in ax2.get_lines()]
    ax1.legend(lines_labels)
    plt.title("CDI and rolling std of uncertainty (PCA and generated wt)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_cdi_vs_stdwt.png"), dpi=150)
    plt.close()
    print("Saved time-series plot")

    print("\nAll done. Output folder:", OUT_DIR)
    print("If you want different PCA / rolling settings, edit the top parameters and re-run.")
    return

if __name__ == "__main__":
    main()

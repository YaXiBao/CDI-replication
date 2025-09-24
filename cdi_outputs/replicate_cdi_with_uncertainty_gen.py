#!/usr/bin/env python3
"""
replicate_cdi_with_uncertainty_gen.py

Generates model_uncertainty.csv from your close_prices.csv (if not present) by:
 - computing month-end returns
 - using rolling-window AR(1) one-step-ahead forecasts to produce forecast errors
 - computing rolling RMSE of forecast errors per ticker (SE_t per ticker)
 - aggregating SE_t across tickers by cross-sectional mean -> SE_t (single series)
 - computing wt_t = Sigma_t / (SE_t^2 + Sigma_t) where Sigma_t = cumulative mean(SE^2)
 - computing 24-month rolling std of wt_t

Merges generated model_uncertainty with cdi_timeseries_monthly.csv and runs regression:
   CDI_t ~ 1 + std_wt_24m

Outputs (OUT_DIR):
 - model_uncertainty.csv
 - cdi_wt_merged.csv
 - regression_results.txt
 - plot_cdi_vs_stdwt_scatter.png
 - plot_cdi_vs_stdwt.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ----------------- user config -----------------
CLOSES_CSV = r"C:\Users\yaxib\OneDrive\Desktop\Python\cdi_outputs\close_prices.csv"                      # your close prices CSV (modify if needed)
CDI_MONTHLY_CSV = r"C:\Users\yaxib\OneDrive\Desktop\Python\cdi_outputs\cdi_timeseries_monthly.csv"         # your monthly CDI CSV
MODEL_UNC_CSV = "model_uncertainty.csv"               # will be created if missing
OUT_DIR = "cdi_paper_replication_outputs"
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","JNJ","PG"]

# Rolling windows (months)
AR_ROLL_WINDOW = 12           # window to fit AR(1) and compute forecast errors
SE_ROLL_WINDOW = 24           # window to compute RMSE of errors (can be same as AR_ROLL_WINDOW)
ROLL_STD_WINDOW = 12          # 24-month rolling std for w_t
MIN_PERIODS_ROLL_STD = 12
MIN_FIRMS_FOR_SE_MEAN = 1     # require at least this many tickers to average SE across tickers
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def read_close_prices(path):
    """
    Read close_prices CSV. Supports:
     - wide format: Date + ticker columns
     - long format: Date, ticker, close
    Returns DataFrame indexed by Date with columns = uppercase tickers.
    """
    raw = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in raw.columns}
    # detect date column
    if 'date' in cols_lower:
        date_col = cols_lower['date']
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.rename(columns={date_col: 'Date'})
    else:
        # try first column as date
        first = raw.columns[0]
        try:
            raw[first] = pd.to_datetime(raw[first])
            raw = raw.rename(columns={first: 'Date'})
        except Exception:
            raise ValueError("Could not detect a Date column in close_prices.csv")
    # Wide format?
    upper_cols = [c.upper() for c in raw.columns]
    if all(t in upper_cols for t in TICKERS):
        # pick Date + columns that match tickers
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
    # Long format?
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
    # fallback: try columns containing ticker substring
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
    """Return month-end prices (last available trading day of each month)."""
    close_df = close_df.copy()
    close_df.index = pd.to_datetime(close_df.index)
    # groupby year-month and take last
    me = close_df.groupby([close_df.index.year, close_df.index.month]).apply(lambda g: g.iloc[-1])
    # rebuild month-end dates
    me.index = pd.to_datetime([pd.Timestamp(year=int(y), month=int(m), day=1) + relativedelta(months=1) - pd.Timedelta(days=1)
                               for (y,m) in me.index])
    me.index.name = 'Date'
    me = me.sort_index()
    me = me[~me.index.duplicated(keep='first')]
    return me

def monthly_returns_from_prices(month_end_df):
    """Compute simple returns (P_t/P_{t-1} - 1) from month-end prices."""
    r = month_end_df.pct_change().dropna(how='all')
    r.index = pd.to_datetime(r.index.to_period('M').to_timestamp())
    return r

def rolling_ar1_forecast_errors(series, window=AR_ROLL_WINDOW):
    """
    Rolling AR(1) one-step-ahead forecast errors.
    Returns a Series of errors aligned to the original index (NaN where not computed).
    """
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
    """Compute rolling RMSE of the forecast errors (window in months)."""
    sq = err_series.pow(2)
    rmse = sq.rolling(window=window, min_periods=1).mean().pow(0.5)
    return rmse

# ---------- main ----------
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
            print(f"Ticker {tkr} has {series.shape[0]} months < AR_ROLL_WINDOW+1 ({AR_ROLL_WINDOW+1}) — will produce sparse SEs.")
        errs = rolling_ar1_forecast_errors(series, window=AR_ROLL_WINDOW)
        se_series = rolling_rmse_of_errors(errs, window=SE_ROLL_WINDOW)
        se_series.name = tkr
        se_tickers[tkr] = se_series
        print(f"Computed SE (ticker={tkr}) — {se_series.dropna().shape[0]} non-null rows")

    # make unified DataFrame of SEs across tickers (align by month)
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
    # compute 24-month rolling std of wt
    std_wt_24m = wt.rolling(window=ROLL_STD_WINDOW, min_periods=MIN_PERIODS_ROLL_STD).std().rename("std_wt_24m")

    # ---------- Robust creation of model_unc DataFrame ----------
    model_unc = pd.concat([se_mean, wt, std_wt_24m], axis=1)
    model_unc = model_unc.reset_index()  # index -> a column (named e.g. 'index' or the index name)
    # robustly rename first column to 'date'
    first_col = model_unc.columns[0]
    if first_col != 'date':
        model_unc = model_unc.rename(columns={first_col: 'date'})
    # ensure datetime monthly-aligned
    model_unc['date'] = pd.to_datetime(model_unc['date']).dt.to_period('M').dt.to_timestamp()
    # make sure expected columns exist
    for col in ['se', 'wt', 'std_wt_24m']:
        if col not in model_unc.columns:
            model_unc[col] = np.nan
    # reorder for readability
    model_unc = model_unc[['date', 'se', 'wt', 'std_wt_24m']]
    model_unc = model_unc.sort_values('date').reset_index(drop=True)

    # debug print
    print("DEBUG model_unc columns:", list(model_unc.columns))
    print("DEBUG model_unc head:\n", model_unc.head().to_string(index=False))

    # save model_unc CSV
    model_unc.to_csv(MODEL_UNC_CSV, index=False)
    print("Saved model_uncertainty.csv ->", MODEL_UNC_CSV)

    # Next: load CDI monthly and merge, then run regression
    print("Loading CDI monthly:", CDI_MONTHLY_CSV)
    cdi = pd.read_csv(CDI_MONTHLY_CSV)

    # find date column name
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
    # find CDI numeric column
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
    cdi['date'] = pd.to_datetime(cdi['date']).dt.to_period('M').dt.to_timestamp()
    model_unc['date'] = pd.to_datetime(model_unc['date']).dt.to_period('M').dt.to_timestamp()

    merged = pd.merge(cdi, model_unc[['date','se','wt','std_wt_24m']], on='date', how='left')
    merged.to_csv(os.path.join(OUT_DIR, "cdi_wt_merged.csv"), index=False)
    print("Saved merged file:", os.path.join(OUT_DIR, "cdi_wt_merged.csv"))

    # regression on rows with both cdi and std_wt_24m
    df_reg = merged.dropna(subset=['cdi','std_wt_24m']).copy()
    if df_reg.empty:
        print("No observations with both CDI and std_wt_24m. You can:")
        print(" - lower MIN_PERIODS_ROLL_STD or MIN_FIRMS_FOR_SE_MEAN")
        print(" - check monthly coverage of your close_prices and CDI")
        return

    X = sm.add_constant(df_reg['std_wt_24m'])
    y = df_reg['cdi']
    model = sm.OLS(y, X).fit(cov_type='HC1')
    print(model.summary())
    with open(os.path.join(OUT_DIR, "regression_results.txt"), "w") as f:
        f.write(model.summary().as_text())
    print("Saved regression results to regression_results.txt")

    # scatter + fit plot
    plt.figure(figsize=(8,6))
    plt.scatter(df_reg['std_wt_24m'], df_reg['cdi'], label='obs')
    xs = np.linspace(df_reg['std_wt_24m'].min(), df_reg['std_wt_24m'].max(), 50)
    ys = model.params['const'] + model.params['std_wt_24m'] * xs
    plt.plot(xs, ys, color='C1', label=f'fit')
    plt.xlabel("24-month rolling std of w_t")
    plt.ylabel("CDI (cdi)")
    plt.title("CDI vs volatility of ensemble weight (generated wt)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_cdi_vs_stdwt_scatter.png"), dpi=150)
    plt.close()
    print("Saved scatter plot")

    # timeseries plot
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(merged['date'], merged['cdi'], label='CDI', linewidth=2)
    ax2 = ax1.twinx()
    ax2.plot(merged['date'], merged['std_wt_24m'], color='C1', label='std_wt_24m', linewidth=1.5)
    ax1.set_xlabel('Date'); ax1.set_ylabel('CDI'); ax2.set_ylabel('std_wt_24m')
    ax1.legend(['CDI','std_wt_24m'])
    plt.title("CDI and 24-month rolling std of generated w_t")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_cdi_vs_stdwt.png"), dpi=150)
    plt.close()
    print("Saved time-series plot")

    print("\nAll done. Output folder:", OUT_DIR)
    print("If you want a different uncertainty-generation method, you can change AR_ROLL_WINDOW or use realized volatility instead.")
    return

if __name__ == "__main__":
    main()


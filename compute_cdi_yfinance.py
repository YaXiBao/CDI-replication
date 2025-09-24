#!/usr/bin/env python3
"""
compute_cdi_yfinance.py

Complete script: load close prices, fetch sales from yfinance, apply availability lags,
compute market caps (close * sharesOutstanding approx), compute CDI on availability dates
and (recommended) monthly CDI using most recent available sales as-of each month.

Outputs (OUT_DIR):
 - sales_with_available_dates.csv
 - marketcaps_at_availability.csv
 - cdi_timeseries.csv                 (availability-date CDI; may be sparse)
 - cdi_timeseries_monthly.csv         (monthly CDI using latest available sales)
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr, pearsonr

# ------------- User settings -------------
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","JNJ","PG"]
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
CLOSES_CSV = r"C:\Users\yaxib\OneDrive\Desktop\Python\Stock Prediction\close_prices.csv"
OUT_DIR = "cdi_outputs"

# Tweak these
MIN_FIRMS_REQUIRED = 5      # for availability-date CDI (quick)
MIN_FIRMS_MONTHLY = 5       # for monthly CDI (recommended)
# -----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- helper utilities ----------------
def parse_close_prices(path):
    """
    Read close_prices in either "wide" format (Date + columns per ticker)
    or "long" format (date,ticker,close). Return DataFrame indexed by Date with columns = tickers (uppercase).
    """
    df = pd.read_csv(path)
    # detect date column name (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    if 'date' in cols_lower:
        date_col = cols_lower['date']
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: 'Date'})
    else:
        # maybe index-like; try to coerce index
        if df.columns.size == 0:
            raise ValueError("Empty close_prices CSV.")
        # else expect a 'Date' or similar; if not found, try to parse first column as date
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col])
            df = df.rename(columns={first_col: 'Date'})
        except Exception:
            raise ValueError("Could not find or parse a 'Date' column in close_prices.csv. "
                             "Provide a 'Date' column or long format with 'ticker' and 'close'.")

    # Now we have a Date column.
    # Detect wide format where tickers appear as column headers
    columns_upper = [c.upper() for c in df.columns]
    if all(t in columns_upper for t in TICKERS):
        # pick Date + ticker columns
        mapping = {}
        for c in df.columns:
            if c.upper() in TICKERS:
                mapping[c] = c
        wide = df[['Date'] + list(mapping.keys())].copy()
        wide['Date'] = pd.to_datetime(wide['Date'])
        wide = wide.set_index('Date').sort_index()
        wide.columns = [c.upper() for c in wide.columns]
        # ensure all requested tickers present (if some missing columns, reindex)
        wide = wide.reindex(columns=TICKERS)
        return wide

    # Try long format: must contain ticker (or symbol) and a close-like column
    cols_map = {c.lower(): c for c in df.columns}
    ticker_col = cols_map.get('ticker', cols_map.get('symbol', None))
    close_col = None
    for candidate in ['close','adjclose','adj_close','close_price','price','adj_close']:
        if candidate in cols_map:
            close_col = cols_map[candidate]
            break

    if ticker_col and close_col:
        long = df.rename(columns={ticker_col: 'ticker', close_col: 'close'})
        long['Date'] = pd.to_datetime(long['Date'])
        pivot = long.pivot(index='Date', columns='ticker', values='close')
        # keep requested tickers (case-insensitive)
        pivot.columns = [str(c).upper() for c in pivot.columns]
        pivot = pivot.reindex(columns=TICKERS)
        pivot = pivot.sort_index()
        return pivot

    # Fallback: try to detect columns whose name contains a ticker substring (case-insensitive)
    col_map = {}
    for c in df.columns:
        for t in TICKERS:
            if t in c.upper():
                col_map[c] = t
    if col_map:
        df['Date'] = pd.to_datetime(df['Date'])
        wide = df[['Date'] + list(col_map.keys())].copy().set_index('Date')
        wide = wide.rename(columns=col_map)
        wide.columns = [c.upper() for c in wide.columns]
        wide = wide.reindex(columns=TICKERS)
        wide = wide.sort_index()
        return wide

    raise ValueError("Unrecognized close_prices.csv format. Provide a wide Date + ticker columns or long (Date,ticker,close).")

def ensure_sorted_datetime_index(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def get_nearest_prior_price(close_df, asof_date):
    """
    Return a Series of prices for all tickers using the most recent prior (<=) trading day before asof_date.
    If no prior date exists, returns NaNs for all tickers.
    """
    if not isinstance(asof_date, (pd.Timestamp, pd.DatetimeIndex, datetime)):
        asof = pd.to_datetime(asof_date)
    else:
        asof = pd.to_datetime(asof_date)
    idx = close_df.index
    # If asof is before first trading date, return NaNs
    if asof < idx[0]:
        return pd.Series({t: np.nan for t in close_df.columns}, name=asof)
    # get_indexer with method='ffill' will give index position of last index <= asof
    pos = idx.get_indexer([asof], method='ffill')[0]
    if pos == -1:
        return pd.Series({t: np.nan for t in close_df.columns}, name=asof)
    nearest_date = idx[pos]
    s = close_df.loc[nearest_date]
    # ensure series index contains all tickers
    s = s.reindex(close_df.columns)
    s.name = nearest_date
    return s

def fetch_sales_yfinance(tickers):
    """
    Fetch quarterly and annual income statements via yfinance.
    Returns DataFrame columns: ['ticker','period_end','period_type','total_revenue_usd']
    """
    records = []
    for t in tickers:
        print(f"Downloading financials for {t} ...")
        tk = yf.Ticker(t)
        fin_annual = tk.financials        # annual columns (end dates)
        fin_quarterly = tk.quarterly_financials
        for period_df, ptype in [(fin_quarterly, 'quarterly'), (fin_annual, 'annual')]:
            if period_df is None or period_df.empty:
                continue
            labels = [str(s) for s in period_df.index]
            rev_label = None
            for lbl in labels:
                L = lbl.lower()
                if 'revenue' in L or 'total rev' in L or 'sales' in L:
                    rev_label = lbl
                    break
            if rev_label is None:
                # warn and skip
                print(f"  WARNING: couldn't detect revenue row for {t} ({ptype}). Available rows: {labels[:8]}")
                continue
            for col in period_df.columns:
                try:
                    val = period_df.loc[rev_label, col]
                    if pd.isna(val):
                        continue
                    val = float(val)
                except Exception:
                    continue
                per_end = pd.to_datetime(col).date()
                records.append({
                    'ticker': t,
                    'period_end': pd.to_datetime(per_end),
                    'period_type': ptype,
                    'total_revenue_usd': val
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No sales data fetched from yfinance. Check tickers and yfinance availability.")
    df = df.sort_values(['ticker','period_end']).reset_index(drop=True)
    return df

def apply_availability_lag(sales_df):
    """
    Add 'available_date' = period_end + lag (annual +6 months, quarterly +4 months),
    and filter to START/END window.
    """
    def add_lag(row):
        if row['period_type'] == 'annual':
            return row['period_end'] + relativedelta(months=6)
        else:
            return row['period_end'] + relativedelta(months=4)
    s = sales_df.copy()
    s['available_date'] = s.apply(add_lag, axis=1)
    s['available_date'] = pd.to_datetime(s['available_date'])
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    s = s[(s['available_date'] >= start) & (s['available_date'] <= end)]
    return s

def get_shares_outstanding_for_tickers(tickers):
    """
    Get current sharesOutstanding via yfinance.info. Returns dict ticker -> shares or NaN.
    (yfinance provides current sharesOutstanding, not historical).
    """
    so = {}
    for t in tickers:
        print(f"Fetching shares outstanding for {t} ...")
        tk = yf.Ticker(t)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            # some versions of yfinance may fail on .info for certain tickers
            info = {}
        val = info.get('sharesOutstanding', np.nan)
        if val is None:
            val = np.nan
        try:
            so[t] = float(val) if not pd.isna(val) else np.nan
        except Exception:
            so[t] = np.nan
    return so

def build_marketcaps_at_availability(close_df, sales_with_dates, shares_out_dict):
    """
    For each unique availability date, compute market cap = nearest prior close * shares_outstanding.
    Returns long-format DataFrame with rows for every (available_date, ticker).
    """
    availability_dates = sorted(sales_with_dates['available_date'].unique())
    rows = []
    for ad in availability_dates:
        ad_ts = pd.to_datetime(ad)
        prices = get_nearest_prior_price(close_df, ad_ts)
        for t in TICKERS:
            price = prices.get(t, np.nan) if t in prices.index else np.nan
            shares = shares_out_dict.get(t, np.nan)
            mc = price * shares if (not pd.isna(price) and not pd.isna(shares)) else np.nan
            rows.append({
                'available_date': ad_ts,
                'ticker': t,
                'close_used_date': prices.name if hasattr(prices,'name') else None,
                'close': price,
                'shares_outstanding': shares,
                'market_cap_usd': mc
            })
    df_mc = pd.DataFrame(rows)
    return df_mc

def compute_cdi_time_series(sales_df, mc_df, min_firms=MIN_FIRMS_REQUIRED):
    """
    Compute CDI on availability dates (sparse). Uses MIN_FIRMS_REQUIRED threshold.
    """
    expected_cols = ['available_date', 'n_firms', 'cdi_spearman_rho', 'cdi_spearman_pval',
                     'cdi_pearson_shares', 'cdi_pearson_pval']
    results = []

    if 'available_date' not in sales_df.columns:
        if 'period_end' in sales_df.columns and 'period_type' in sales_df.columns:
            sales_df = apply_availability_lag(sales_df)
        else:
            return pd.DataFrame(columns=expected_cols)

    availability_dates = sorted(sales_df['available_date'].dropna().unique())
    if len(availability_dates) == 0:
        return pd.DataFrame(columns=expected_cols)

    for ad in availability_dates:
        ad_ts = pd.to_datetime(ad)
        s_sub = sales_df[sales_df['available_date'] == ad_ts].copy()
        if 'ticker' in s_sub.columns:
            s_sub = s_sub.set_index('ticker')
        else:
            continue
        mc_sub = mc_df[mc_df['available_date'] == ad_ts].set_index('ticker') if 'available_date' in mc_df.columns else mc_df.set_index('ticker')
        joined = s_sub.join(mc_sub[['market_cap_usd']], how='inner')
        joined = joined.dropna(subset=['total_revenue_usd','market_cap_usd'])
        n = len(joined)
        if n < min_firms:
            # skipped (not enough firms)
            continue
        joined['sales_rank'] = joined['total_revenue_usd'].rank(ascending=False, method='average')
        joined['mc_rank'] = joined['market_cap_usd'].rank(ascending=False, method='average')
        rho, pval = spearmanr(joined['sales_rank'], joined['mc_rank'])
        joined['sales_share'] = joined['total_revenue_usd'] / joined['total_revenue_usd'].sum()
        joined['mc_share'] = joined['market_cap_usd'] / joined['market_cap_usd'].sum()
        pearson, ppear = pearsonr(joined['sales_share'], joined['mc_share'])
        results.append({
            'available_date': pd.to_datetime(ad_ts),
            'n_firms': int(n),
            'cdi_spearman_rho': float(rho) if not np.isnan(rho) else np.nan,
            'cdi_spearman_pval': float(pval) if not np.isnan(pval) else np.nan,
            'cdi_pearson_shares': float(pearson) if not np.isnan(pearson) else np.nan,
            'cdi_pearson_pval': float(ppear) if not np.isnan(ppear) else np.nan
        })

    if len(results) == 0:
        return pd.DataFrame(columns=expected_cols)
    df = pd.DataFrame(results).sort_values('available_date').reset_index(drop=True)
    return df

def compute_monthly_cdi_using_latest_sales(sales_df, close_df, shares_out_dict,
                                           start_date, end_date, freq='M', min_firms=MIN_FIRMS_MONTHLY):
    """
    For each calendar period end (monthly by default), pick the most recent sales record
    whose available_date <= as_of_date for each ticker. Compute marketcap at as_of_date
    using nearest prior close * shares_outstanding (approx). Compute Spearman rank
    correlation (sales rank vs marketcap rank) and Pearson on shares.
    Returns DataFrame with columns: available_date, n_firms, cdi_spearman_rho, ...
    """
    results = []
    sales_df = sales_df.copy()
    sales_df['available_date'] = pd.to_datetime(sales_df['available_date'])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    asof_dates = pd.date_range(start, end, freq=freq)

    for asof in asof_dates:
        eligible = sales_df[sales_df['available_date'] <= asof]
        if eligible.empty:
            continue
        latest = eligible.sort_values('available_date').groupby('ticker', as_index=False).last()
        prices = get_nearest_prior_price(close_df, asof)
        rows = []
        for _, row in latest.iterrows():
            t = row['ticker']
            price = prices.get(t, np.nan) if t in prices.index else np.nan
            shares = shares_out_dict.get(t, np.nan)
            mc = price * shares if (not pd.isna(price) and not pd.isna(shares)) else np.nan
            rows.append({'ticker': t,
                         'total_revenue_usd': row['total_revenue_usd'],
                         'available_date_for_sales': row['available_date'],
                         'close_used_date': prices.name if hasattr(prices, 'name') else None,
                         'close': price,
                         'shares_outstanding': shares,
                         'market_cap_usd': mc})
        df_this = pd.DataFrame(rows).dropna(subset=['total_revenue_usd', 'market_cap_usd'])
        n = len(df_this)
        if n < min_firms:
            continue
        df_this['sales_rank'] = df_this['total_revenue_usd'].rank(ascending=False, method='average')
        df_this['mc_rank'] = df_this['market_cap_usd'].rank(ascending=False, method='average')
        rho, pval = spearmanr(df_this['sales_rank'], df_this['mc_rank'])
        df_this['sales_share'] = df_this['total_revenue_usd'] / df_this['total_revenue_usd'].sum()
        df_this['mc_share'] = df_this['market_cap_usd'] / df_this['market_cap_usd'].sum()
        pearson, ppear = pearsonr(df_this['sales_share'], df_this['mc_share'])
        results.append({
            'available_date': pd.to_datetime(asof),
            'n_firms': int(n),
            'cdi_spearman_rho': float(rho) if not np.isnan(rho) else np.nan,
            'cdi_spearman_pval': float(pval) if not np.isnan(pval) else np.nan,
            'cdi_pearson_shares': float(pearson) if not np.isnan(pearson) else np.nan,
            'cdi_pearson_pval': float(ppear) if not np.isnan(ppear) else np.nan
        })

    if len(results) == 0:
        cols = ['available_date', 'n_firms', 'cdi_spearman_rho', 'cdi_spearman_pval',
                'cdi_pearson_shares', 'cdi_pearson_pval']
        return pd.DataFrame(columns=cols)

    df_res = pd.DataFrame(results).sort_values('available_date').reset_index(drop=True)
    return df_res

def print_monthly_contributors(sales_df, close_df, shares_out_dict, start_date, end_date, freq='M'):
    """
    Diagnostic: prints which tickers contributed to each monthly CDI (their sales available date, close, shares, mc).
    """
    sales_df = sales_df.copy()
    sales_df['available_date'] = pd.to_datetime(sales_df['available_date'])
    asof_dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq=freq)
    for asof in asof_dates:
        eligible = sales_df[sales_df['available_date'] <= asof]
        if eligible.empty:
            continue
        latest = eligible.sort_values('available_date').groupby('ticker', as_index=False).last()
        prices = get_nearest_prior_price(close_df, asof)
        rows = []
        for _, row in latest.iterrows():
            t = row['ticker']
            price = prices.get(t, np.nan) if t in prices.index else np.nan
            shares = shares_out_dict.get(t, np.nan)
            mc = price * shares if (not pd.isna(price) and not pd.isna(shares)) else np.nan
            rows.append((t, row['available_date'], row['total_revenue_usd'], price, shares, mc))
        df_this = pd.DataFrame(rows, columns=['ticker','sales_available_date','sales','close','shares','market_cap'])
        df_this = df_this.dropna(subset=['sales','market_cap'])
        if df_this.empty:
            continue
        print(f"\nAs-of {asof.date()} -> n={len(df_this)} contributors:")
        print(df_this.sort_values('market_cap', ascending=False).to_string(index=False))

# ------------------ Main pipeline ------------------
def main():
    print("Loading close prices...")
    close_df = parse_close_prices(CLOSES_CSV)
    close_df = ensure_sorted_datetime_index(close_df)
    # restrict close_df range a bit for speed
    close_df = close_df[(close_df.index >= pd.to_datetime(START_DATE) - pd.Timedelta(days=60)) &
                        (close_df.index <= pd.to_datetime(END_DATE) + pd.Timedelta(days=60))]
    print(f"Close price data covers {close_df.index.min().date()} to {close_df.index.max().date()} with {len(close_df)} rows.")

    print("Fetching sales (total revenue) from yfinance ...")
    sales_df = fetch_sales_yfinance(TICKERS)
    sales_df = sales_df[(sales_df['period_end'] >= pd.to_datetime(START_DATE)) & (sales_df['period_end'] <= pd.to_datetime(END_DATE))]
    print(f"Fetched {len(sales_df)} sales rows (annual+quarterly) across tickers.")

    print("Applying availability lags (annual +6m, quarterly +4m) ...")
    sales_with_dates = apply_availability_lag(sales_df)

    # DEBUG / sanity checks for sales_with_dates
    print("DEBUG: sales_with_dates rows =", len(sales_with_dates))
    print("DEBUG: sales_with_dates columns =", list(sales_with_dates.columns))
    if len(sales_with_dates) > 0:
        print("DEBUG: sample rows:\n", sales_with_dates.head().to_string(index=False))
    else:
        print("ERROR: No sales records after applying availability lag. Possible causes:")
        print(" - yfinance did not return any income-statement rows for these tickers.")
        print(" - the period_end -> available_date lag filtered all rows outside your START/END window.")
        print("Check the fetched sales (earlier printed) and your START_DATE/END_DATE.")

    sales_with_dates.to_csv(os.path.join(OUT_DIR, "sales_with_available_dates.csv"), index=False)
    print(f"Saved sales_with_available_dates.csv ({len(sales_with_dates)} rows).")

    print("Fetching shares outstanding (approx, current) ...")
    shares_out = get_shares_outstanding_for_tickers(TICKERS)
    print("\nDEBUG shares_out (ticker -> shares):")
    for k, v in shares_out.items():
        print(f"  {k}: {v}")

    print("Building market caps at availability dates ...")
    mc_df = build_marketcaps_at_availability(close_df, sales_with_dates, shares_out)
    mc_df.to_csv(os.path.join(OUT_DIR, "marketcaps_at_availability.csv"), index=False)
    print("Saved marketcaps_at_availability.csv")

    # Debug marketcap counts per availability date
    try:
        tmp = mc_df.groupby('available_date')['market_cap_usd'].apply(lambda s: s.notna().sum())
        print("\nDEBUG marketcap counts per available_date:")
        print(tmp.to_string())
    except Exception:
        pass

    print("Computing CDI time series on availability dates (sparse) ...")
    cdi_ts = compute_cdi_time_series(sales_with_dates, mc_df, min_firms=MIN_FIRMS_REQUIRED)
    cdi_ts.to_csv(os.path.join(OUT_DIR, "cdi_timeseries.csv"), index=False)
    print("Saved cdi_timeseries.csv")

    print("Computing monthly CDI (recommended) using most recent available sales as-of each month ...")
    cdi_monthly = compute_monthly_cdi_using_latest_sales(sales_with_dates, close_df, shares_out,
                                                         START_DATE, END_DATE, freq='M', min_firms=MIN_FIRMS_MONTHLY)
    cdi_monthly.to_csv(os.path.join(OUT_DIR, "cdi_timeseries_monthly.csv"), index=False)
    print("Saved cdi_timeseries_monthly.csv")

    print("\nSample of availability-date CDI (sparse):")
    if cdi_ts.empty:
        print("  (empty — not enough simultaneous reporting across tickers at the chosen MIN_FIRMS_REQUIRED)")
    else:
        print(cdi_ts.head().to_string(index=False))

    print("\nSample of monthly CDI:")
    if cdi_monthly.empty:
        print("  (empty — try lowering MIN_FIRMS_MONTHLY or check shares/prices)")
    else:
        print(cdi_monthly.head(12).to_string(index=False))

    # Optional diagnostic print: who contributed each month
    print("\nContributor diagnostics (monthly):")
    print_monthly_contributors(sales_with_dates, close_df, shares_out, START_DATE, END_DATE, freq='M')

    print("\nDone. Outputs in folder:", OUT_DIR)

if __name__ == "__main__":
    main()


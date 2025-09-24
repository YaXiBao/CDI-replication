#!/usr/bin/env python3
"""
sales_downloader.py (improved)

Standalone sales-data downloader. Saves a daily forward-filled CSV of quarterly revenue
for the requested tickers: sales_data.csv (index = date, columns = tickers).

Improvements in this version:
 - Always tries FinancialModelingPrep (FMP) as a fallback even if no API key provided.
 - Adds a --verbose flag to print what yfinance returns for failed tickers for debugging.
 - Keeps on-disk cache to avoid re-requesting successful tickers.
 - Optional business-day resampling (--business_days).

Usage examples:
 python sales_downloader.py --tickers AAPL,MSFT,GOOGL --start 2015-01-01 --end 2024-01-01 --api_key XLOFNLWRV8NVEZ0A --verbose
 python sales_downloader.py --tickers AAPL,MSFT --cache sales_cache.pkl --business_days

Outputs:
 - sales_data.csv
 - (optional) cache file (default: sales_cache.pkl)

Note: do not commit API keys to source control. Prefer environment variables.
"""

import argparse
import os
import time
import pickle
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf

ALPHA_SLEEP = 12


# ---------------- helpers (AlphaVantage / yfinance / FMP) ----------------

def fetch_sales_alphavantage_single(ticker: str, start_date: str, end_date: str, api_key: str) -> Tuple[Optional[pd.Series], str]:
    base = "https://www.alphavantage.co/query"
    params = {'function': 'INCOME_STATEMENT', 'symbol': ticker, 'apikey': api_key, 'outputsize': 'full'}
    try:
        r = requests.get(base, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        return None, f"request_failed:{e}"

    if isinstance(data, dict) and ('Note' in data or 'Information' in data):
        return None, 'rate_limited'

    if not isinstance(data, dict) or 'quarterlyReports' not in data:
        return None, 'no_data'

    reports = data['quarterlyReports']
    rows = []
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    for rep in reports:
        d = rep.get('fiscalDateEnding')
        if not d:
            continue
        d = pd.to_datetime(d)
        if d < s or d > e:
            continue
        rev = rep.get('totalRevenue') or rep.get('totalNetRevenue') or 0
        try:
            rev = float(rev) / 1e9
        except Exception:
            rev = 0.0
        rows.append({'date': d, 'sales': rev})

    if not rows:
        return None, 'no_quarters_found'

    df = pd.DataFrame(rows).set_index('date').sort_index()
    daily = df.resample('D').ffill().reindex(pd.date_range(s, e)).fillna(0.0)['sales']
    return daily, 'ok'


def fetch_sales_yfinance_single(ticker: str, start_date: str, end_date: str, verbose: bool = False) -> Tuple[Optional[pd.Series], str]:
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    try:
        tk = yf.Ticker(ticker)

        # candidate attributes (common variations across yfinance versions)
        cand_attrs = ['quarterly_income_stmt', 'quarterly_financials', 'quarterly_earnings', 'financials', 'quarterlyBalanceSheet']
        qstmt = None
        found_attr = None
        for a in cand_attrs:
            q = getattr(tk, a, None)
            if isinstance(q, pd.DataFrame) and not q.empty:
                qstmt = q.copy()
                found_attr = a
                break

        if qstmt is None or not isinstance(qstmt, pd.DataFrame) or qstmt.empty:
            return None, 'no_yf_data'

        matches = [idx for idx in qstmt.index if 'revenue' in str(idx).lower()]
        if not matches:
            # fallback: look for common alternate labels
            matches = [idx for idx in qstmt.index if any(x in str(idx).lower() for x in ['total revenue', 'net sales', 'sales', 'totalrevenue', 'total revenue'])]
            if not matches:
                if verbose:
                    print(f"  yfinance: no revenue-like row found for {ticker}. Dataframe index labels: {list(qstmt.index)}")
                    try:
                        print("  Sample of dataframe (transposed):\n", qstmt.head().T)
                    except Exception:
                        pass
                return None, 'no_revenue_row'

        rev_row = qstmt.loc[matches[0]].dropna()
        if rev_row.empty:
            if verbose:
                print(f"  yfinance: revenue row exists but has no values for {ticker}; label={matches[0]}")
            return None, 'no_revenue_values'

        rev_row.index = pd.to_datetime(rev_row.index)
        # If all dates are outside requested range, print debug when verbose
        in_range = rev_row[(rev_row.index >= s) & (rev_row.index <= e)]
        if in_range.empty:
            if verbose:
                print(f"  yfinance: revenue dates found but none within {s.date()}..{e.date()} for {ticker}")
                print("  revenue row index dates:\n", list(rev_row.index))
                try:
                    print("  revenue values:\n", rev_row)
                except Exception:
                    pass
            return None, 'no_revenue_in_range'

        rev_row = in_range
        daily = rev_row.resample('D').ffill().reindex(pd.date_range(s, e)).fillna(0.0) / 1e9
        return daily, 'ok'
    except Exception as exc:
        if verbose:
            print(f"  yfinance exception for {ticker}: {exc}")
        return None, f'yf_exception:{exc}'


def fetch_sales_fmp_single(ticker: str, start_date: str, end_date: str, fmp_key: Optional[str]) -> Tuple[Optional[pd.Series], str]:
    base = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
    params = {'period': 'quarter', 'limit': 40}
    if fmp_key:
        params['apikey'] = fmp_key
    try:
        r = requests.get(base, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        return None, f'fmp_request_failed:{e}'

    if not isinstance(data, list) or len(data) == 0:
        return None, 'fmp_no_data'

    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    rows = []
    for item in data:
        d = item.get('date') or item.get('calendarDate')
        if not d:
            continue
        d = pd.to_datetime(d)
        if d < s or d > e:
            continue
        rev = item.get('revenue') or item.get('totalRevenue') or 0
        try:
            rev = float(rev) / 1e9
        except Exception:
            rev = 0.0
        rows.append({'date': d, 'sales': rev})

    if not rows:
        return None, 'fmp_no_quarters_in_range'

    df = pd.DataFrame(rows).set_index('date').sort_index()
    daily = df.resample('D').ffill().reindex(pd.date_range(s, e)).fillna(0.0)['sales']
    return daily, 'ok'


# ---------------- pipeline ----------------

def fetch_sales_for_tickers(tickers, start_date, end_date, alpha_key=None, fmp_key=None,
                             cache_path: Optional[str] = None, force_refresh=False, verbose=False):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_idx = pd.date_range(start, end)
    sales_dict = {}

    # load cache
    cache = {}
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    alpha_allowed = bool(alpha_key)
    alpha_rate_limited = False

    for t in tickers:
        t = t.upper().strip()
        print(f"\nProcessing {t} ...")

        # use cache if available
        if t in cache and not force_refresh:
            print("  -> loaded from cache")
            series = cache[t]
            if isinstance(series, pd.Series):
                series = series.reindex(all_idx).fillna(0.0)
                sales_dict[t] = series
                continue

        series = None
        status = None

        # 1) try AlphaVantage if allowed
        if alpha_allowed and not alpha_rate_limited:
            series, status = fetch_sales_alphavantage_single(t, start_date, end_date, alpha_key)
            if status == 'ok':
                print("  -> AlphaVantage OK")
                sales_dict[t] = series
                cache[t] = series
                time.sleep(ALPHA_SLEEP)
                continue
            elif status == 'rate_limited':
                alpha_rate_limited = True
                print("  -> Alpha rate-limited; switching to fallbacks")
            else:
                if verbose:
                    print(f"  -> Alpha returned: {status}")

        # 2) try yfinance
        series, status = fetch_sales_yfinance_single(t, start_date, end_date, verbose=verbose)
        if status == 'ok':
            print("  -> yfinance OK")
            sales_dict[t] = series
            cache[t] = series
            continue
        else:
            if verbose:
                print(f"  -> yfinance returned: {status}")

        # 3) try FMP (always attempt FMP as a fallback even without a key)
        series, status = fetch_sales_fmp_single(t, start_date, end_date, fmp_key)
        if status == 'ok':
            print("  -> FMP OK")
            sales_dict[t] = series
            cache[t] = series
            continue
        else:
            if verbose:
                print(f"  -> FMP returned: {status}")

        # final fallback: zeros
        print("  -> filling zeros")
        sales_dict[t] = pd.Series(0.0, index=all_idx)
        cache[t] = sales_dict[t]

    # save cache
    if cache_path:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
            print(f"Cache saved to {cache_path}")
        except Exception as e:
            print(f"Could not save cache: {e}")

    sales_df = pd.DataFrame(sales_dict).reindex(all_idx).fillna(0.0)
    sales_df.index.name = 'date'
    return sales_df


# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', default='AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,JNJ,PG')
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2024-01-01')
    p.add_argument('--api_key', default=os.getenv('ALPHA_VANTAGE_KEY', None), help='Alpha Vantage API key (optional)')
    p.add_argument('--fmp_key', default=os.getenv('FMP_API_KEY', None), help='FinancialModelingPrep API key (optional)')
    p.add_argument('--cache', default='sales_cache.pkl', help='Path to cache pickle file')
    p.add_argument('--force_refresh', action='store_true', help='Ignore cache and fetch anew')
    p.add_argument('--business_days', action='store_true', help='Resample sales_data.csv to business days only')
    p.add_argument('--verbose', action='store_true', help='Print verbose diagnostics for yfinance/fallbacks')
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    sales_df = fetch_sales_for_tickers(tickers, args.start, args.end, alpha_key=args.api_key, fmp_key=args.fmp_key,
                                       cache_path=args.cache, force_refresh=args.force_refresh, verbose=args.verbose)

    if args.business_days:
        bidx = pd.bdate_range(args.start, args.end)
        sales_df = sales_df.reindex(bidx).ffill().fillna(0.0)

    sales_df.to_csv('sales_data.csv', float_format='%.6f')
    print('\nSaved sales_data.csv')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
data_downloader_fallback.py

Tries multiple methods (Alpha Vantage -> yfinance -> FinancialModelingPrep) to obtain quarterly
'Total Revenue' and forward-fills to daily. Saves:
 - close_prices.csv
 - index_returns.csv
 - sales_data.csv

Usage:
 python data_downloader_fallback.py --tickers AAPL,MSFT,... --start 2015-01-01 --end 2024-01-01 --api_key YOUR_ALPHA_KEY
Optionally add: --fmp_key YOUR_FMP_KEY

Notes:
 - If Alpha Vantage returns a rate-limit "Note" the script will stop using AV for remaining tickers
   and fall back to the other sources to avoid wasting daily quota.
 - The script does NOT hard-code any API keys. Pass them via CLI or environment variables.
"""
import argparse
import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

ALPHA_SLEEP = 12  # conservative delay (seconds) between AlphaVantage calls


def download_prices(tickers, index_ticker, start_date, end_date):
    print("Downloading price data via yfinance...")
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    index_data = yf.download(index_ticker, start=start_date, end=end_date)['Close']
    stock_data = stock_data.dropna(how='all').copy()
    index_data = index_data.dropna().copy()

    # Index returns: next-day returns aligned to features (shift -1)
    index_returns = index_data.pct_change().dropna()
    common_dates = stock_data.index.intersection(index_returns.index)
    index_returns = index_returns.loc[common_dates].shift(-1).dropna()
    stock_data = stock_data.loc[index_returns.index]

    print(f"Downloaded close prices shape: {stock_data.shape}, index returns length: {len(index_returns)}")
    return stock_data, index_returns


# ---------------- AlphaVantage helper ----------------
def fetch_sales_alphavantage_single(ticker, start_date, end_date, api_key):
    base = "https://www.alphavantage.co/query"
    params = {'function': 'INCOME_STATEMENT', 'symbol': ticker, 'apikey': api_key, 'outputsize': 'full'}
    try:
        r = requests.get(base, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        print(f"AlphaVantage request failed for {ticker}: {e}")
        return None, "request_failed"

    # detect rate limit / informational note
    if isinstance(data, dict) and ('Note' in data or 'Information' in data):
        note = data.get('Note') or data.get('Information')
        print(f"AlphaVantage rate/limit note encountered for {ticker}: {note}")
        return None, "rate_limited"

    if not isinstance(data, dict) or 'quarterlyReports' not in data:
        msg = data.get('Error Message') if isinstance(data, dict) else 'unexpected_response'
        print(f"AlphaVantage returned no quarterlyReports for {ticker}: {msg}")
        return None, "no_data"

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
        rev = rep.get('totalRevenue')
        try:
            rev = float(rev) / 1e9  # scale to billions
        except:
            rev = 0.0
        rows.append({'date': d, 'sales': rev})
    if not rows:
        return None, "no_quarters_found"
    df = pd.DataFrame(rows).set_index('date').sort_index()
    daily = df.resample('D').ffill().reindex(pd.date_range(s, e)).fillna(0.0)['sales']
    return daily, "ok"


# ---------------- yfinance helper ----------------
def fetch_sales_yfinance_single(ticker, start_date, end_date):
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    try:
        tk = yf.Ticker(ticker)
        cand_attrs = ['quarterly_income_stmt', 'quarterly_financials', 'quarterly_earnings',
                      'financials', 'quarterlyBalanceSheet']
        qstmt = None
        for a in cand_attrs:
            q = getattr(tk, a, None)
            if isinstance(q, pd.DataFrame) and not q.empty:
                qstmt = q.copy()
                break
        if qstmt is None or not isinstance(qstmt, pd.DataFrame) or qstmt.empty:
            return None, "no_yf_data"

        # find a row that contains 'revenue' case-insensitive
        matches = [idx for idx in qstmt.index if 'revenue' in str(idx).lower()]
        if not matches:
            return None, "no_revenue_row"

        rev_row = qstmt.loc[matches[0]].dropna()
        if rev_row.empty:
            return None, "no_revenue_values"

        rev_row.index = pd.to_datetime(rev_row.index)
        rev_row = rev_row[(rev_row.index >= s) & (rev_row.index <= e)]
        if rev_row.empty:
            return None, "no_revenue_in_range"

        daily = rev_row.resample('D').ffill().reindex(pd.date_range(s, e)).fillna(0.0) / 1e9
        return daily, "ok"
    except Exception as exc:
        print(f"yfinance error for {ticker}: {exc}")
        return None, "yf_exception"


# ---------------- FinancialModelingPrep helper ----------------
def fetch_sales_fmp_single(ticker, start_date, end_date, fmp_key):
    base = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
    params = {'period': 'quarter', 'limit': 40}
    if fmp_key:
        params['apikey'] = fmp_key
    try:
        r = requests.get(base, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        print(f"FMP request failed for {ticker}: {e}")
        return None, "fmp_request_failed"
    if not isinstance(data, list) or len(data) == 0:
        return None, "fmp_no_data"
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
        except:
            rev = 0.0
        rows.append({'date': d, 'sales': rev})
    if not rows:
        return None, "fmp_no_quarters_in_range"
    df = pd.DataFrame(rows).set_index('date').sort_index()
    daily = df.resample('D').ffill().reindex(pd.date_range(s, e)).fillna(0.0)['sales']
    return daily, "ok"


# ---------------- Pipeline ----------------
def fetch_sales_all(tickers, start_date, end_date, alpha_key=None, fmp_key=None):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_idx = pd.date_range(start, end)
    sales_dict = {}
    alpha_allowed = bool(alpha_key)
    alpha_rate_limited = False

    for t in tickers:
        t = t.upper().strip()
        print(f"\nFetching sales for {t} ...")
        series = None
        status = None

        # 1) Alpha Vantage (only try if we still think it's allowed)
        if alpha_allowed and not alpha_rate_limited:
            series, status = fetch_sales_alphavantage_single(t, start_date, end_date, alpha_key)
            if status == "ok":
                print(f"AlphaVantage -> OK for {t}")
                sales_dict[t] = series
                time.sleep(ALPHA_SLEEP)
                continue
            elif status == "rate_limited":
                alpha_rate_limited = True
                print("Alpha Vantage rate limit reached — switching to next fallbacks for remaining tickers.")
            else:
                print(f"AlphaVantage did not return usable data for {t}: {status}")

        # 2) yfinance fallback
        series, status = fetch_sales_yfinance_single(t, start_date, end_date)
        if status == "ok":
            print(f"yfinance -> OK for {t}")
            sales_dict[t] = series
            continue
        else:
            print(f"yfinance failed for {t}: {status}")

        # 3) FinancialModelingPrep fallback (if provided)
        if fmp_key:
            series, status = fetch_sales_fmp_single(t, start_date, end_date, fmp_key)
            if status == "ok":
                print(f"FMP -> OK for {t}")
                sales_dict[t] = series
                continue
            else:
                print(f"FMP failed for {t}: {status}")

        # 4) final fallback -> zeros
        print(f"All methods failed for {t}. Filling zeros.")
        sales_dict[t] = pd.Series(0.0, index=all_idx)

    sales_df = pd.DataFrame(sales_dict).reindex(all_idx).fillna(0.0)
    sales_df.index.name = 'date'
    return sales_df


# ---------------- Main ----------------
def main(args):
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    close_prices, index_returns = download_prices(tickers, args.index, args.start, args.end)

    # Save close_prices
    close_prices.to_csv('close_prices.csv', float_format='%.6f')

    # Save index_returns robustly (handle Series or DataFrame)
    idx = index_returns
    if isinstance(idx, pd.Series):
        idx.to_frame(name=args.index).to_csv('index_returns.csv', float_format='%.8f')
    elif isinstance(idx, pd.DataFrame):
        if args.index in idx.columns:
            idx[[args.index]].to_csv('index_returns.csv', float_format='%.8f')
        elif idx.shape[1] == 1:
            idx.to_csv('index_returns.csv', float_format='%.8f')
        else:
            first_col = idx.columns[0]
            print(f"Warning: index_returns has multiple columns {list(idx.columns)}; saving first column '{first_col}'.")
            idx[[first_col]].to_csv('index_returns.csv', float_format='%.8f')
    else:
        pd.DataFrame(idx).to_csv('index_returns.csv', float_format='%.8f')

    print("Saved close_prices.csv and index_returns.csv")

    # sales (with fallback pipeline)
    sales_df = fetch_sales_all(tickers, args.start, args.end, alpha_key=args.api_key, fmp_key=args.fmp_key)
    sales_df.to_csv('sales_data.csv', float_format='%.6f')
    print("Saved sales_data.csv")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', default='AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,JNJ,PG',
                   help='Comma-separated tickers (default: large caps)')
    p.add_argument('--index', default='SPY', help='Index ticker for target returns (default: SPY)')
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2024-01-01')
    p.add_argument('--api_key', default=os.getenv('ALPHA_VANTAGE_KEY', None), help='Alpha Vantage API key (optional)')
    p.add_argument('--fmp_key', default=os.getenv('FMP_API_KEY', None), help='FinancialModelingPrep API key (optional)')
    args = p.parse_args()
    main(args)

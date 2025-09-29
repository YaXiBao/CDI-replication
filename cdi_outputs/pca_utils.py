# pca_utils.py  -- robust rolling_local_pca with automatic n_components adjustment + debug
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def rolling_local_pca(panel_df, features, window_months=36, n_components=2, min_nobs=5, standardize=True, debug=False):
    """
    Robust Rolling local PCA.

    panel_df: DataFrame indexed by MultiIndex (date,ticker), columns include 'features'
    features: list of column names to use (e.g. ['sales','market_cap_usd'])
    window_months: lookback window length in months (use most recent window_months months to fit PCA)
    n_components: target number of PCs (function will lower this if not enough samples/features)
    min_nobs: minimum number of tickers in cross-section at date T to produce scores
    standardize: whether to z-score columns using pooled window mean/std
    debug: if True, print per-date diagnostics

    Returns:
      - pcs_by_date: dict mapping date -> DataFrame(index=ticker, columns=PC1..PCk)
      - explained_df: DataFrame index=date columns=PC1..PCk (explained variance ratio)
    """
    if not isinstance(panel_df.index, pd.MultiIndex):
        raise ValueError("panel_df must be indexed by MultiIndex (date, ticker)")

    dates = sorted(set(panel_df.index.get_level_values(0)))
    pcs_by_date = {}
    explained_rows = []

    for i, T in enumerate(dates):
        start_idx = max(0, i - window_months + 1)
        window_dates = dates[start_idx:(i+1)]
        pooled = []
        for d in window_dates:
            try:
                sub = panel_df.loc[d]
            except KeyError:
                continue
            if not set(features).issubset(sub.columns):
                continue
            subf = sub[features].dropna()
            if subf.shape[0] == 0:
                continue
            pooled.append(subf.values)   # each is n_firms_at_d x n_features

        if len(pooled) == 0:
            if debug:
                print(f"[{T.date()}] skipped: no pooled data in window")
            continue

        X = np.vstack(pooled)  # pooled observations x features
        n_samples, n_features = X.shape[0], X.shape[1]

        # If too few pooled observations or features, skip or reduce components
        k_allowed = min(n_components, n_samples, n_features)
        if k_allowed < 1:
            if debug:
                print(f"[{T.date()}] skipped: not enough data (n_samples={n_samples}, n_features={n_features}, need at least 1)")
            continue

        # standardize pooled X per column
        if standardize:
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=0)
            sd[sd == 0] = 1.0
            Xs = (X - mu) / sd
        else:
            mu = np.zeros(n_features)
            sd = np.ones(n_features)
            Xs = X.copy()

        # Fit PCA with k_allowed components
        k = int(k_allowed)
        if debug:
            print(f"[{T.date()}] pooled shape = {X.shape}, using k = {k}")

        try:
            pca = PCA(n_components=k)
            pca.fit(Xs)
        except Exception as e:
            if debug:
                print(f"[{T.date()}] PCA fit failed: {e}")
            continue

        # compute scores for cross-section at date T
        try:
            subT = panel_df.loc[T]
        except KeyError:
            if debug:
                print(f"[{T.date()}] skip: no cross-section at T")
            continue
        if not set(features).issubset(subT.columns):
            if debug:
                print(f"[{T.date()}] skip: features missing at T")
            continue
        subT_f = subT[features].dropna()
        if subT_f.shape[0] < min_nobs:
            if debug:
                print(f"[{T.date()}] skip: cross-section too small ({subT_f.shape[0]} firms, min_nobs={min_nobs})")
            continue

        XT = subT_f.values.astype(float)
        # standardize using pooled mu/sd
        XTs = (XT - mu) / sd
        try:
            scores = pca.transform(XTs)
        except Exception as e:
            if debug:
                print(f"[{T.date()}] PCA.transform failed: {e}")
            continue

        cols = [f'PC{i+1}' for i in range(scores.shape[1])]
        pcs_by_date[T] = pd.DataFrame(scores, index=subT_f.index, columns=cols)
        explained_rows.append((T, pca.explained_variance_ratio_))

    # build explained DataFrame
    if explained_rows:
        maxk = max(len(arr) for _, arr in explained_rows)
        explained_df = pd.DataFrame(index=[d for d,_ in explained_rows],
                                    columns=[f'PC{i+1}' for i in range(maxk)], dtype=float)
        for d, arr in explained_rows:
            for j, v in enumerate(arr):
                explained_df.loc[d, f'PC{j+1}'] = v
        explained_df.index.name = 'date'
    else:
        explained_df = pd.DataFrame()

    return pcs_by_date, explained_df


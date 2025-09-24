#!/usr/bin/env python3
# cdi_robust_checks.py
"""
Run three robustness checks:
 A) HAC (Newey-West) OLS: CDI ~ 1 + std_wt_24m
 B) Standardize std_wt_24m (z-score) and run OLS with HC1 (robust SE)
 C) Scatter plot + fitted line, and descriptive stats

Reads merged data from one of likely paths (tries several).
Saves plots to ./cdi_analysis_outputs/
Prints regression summaries to console and saves text files.
"""
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

CANDIDATE_PATHS = [
    "cdi_paper_replication_outputs/cdi_wt_merged.csv",
    "cdi_wt_merged.csv",
    "cdi_outputs/cdi_wt_merged.csv",
    "cdi_paper_replication_outputs/cdi_wt_merged.CSV",
    "cdi_wt_merged.CSV"
]

OUT_DIR = "cdi_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def find_merged_file():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    # fallback: search for any file with name pattern
    for f in glob.glob("**/cdi_wt_merged.csv", recursive=True):
        return f
    for f in glob.glob("**/*cdi*wt*merged*.csv", recursive=True):
        return f
    return None

def load_merged(path):
    df = pd.read_csv(path)
    # normalize date column
    date_col = None
    for c in df.columns:
        if c.lower() in ('date','available_date','time'):
            date_col = c
            break
    if date_col is None:
        # try find any column that parses as a date
        for c in df.columns:
            try:
                pd.to_datetime(df[c].iloc[:5])
                date_col = c
                break
            except Exception:
                continue
    if date_col is not None:
        df = df.rename(columns={date_col: 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    # ensure std_wt_24m & cdi columns exist
    if 'std_wt_24m' not in df.columns:
        # try to find column name with 'std' and 'wt'
        for c in df.columns:
            if 'std' in c.lower() and 'wt' in c.lower():
                df = df.rename(columns={c: 'std_wt_24m'})
                break
    if 'cdi' not in df.columns:
        for c in df.columns:
            if 'cdi' in c.lower():
                df = df.rename(columns={c: 'cdi'})
                break
    return df

def save_text(path, txt):
    with open(path, "w", encoding="utf8") as f:
        f.write(txt)

def main():
    path = find_merged_file()
    if path is None:
        print("ERROR: 找不到 cdi_wt_merged.csv。请把它放在工作目录或 cdi_paper_replication_outputs/ 下。")
        sys.exit(1)
    print("Loading merged file:", path)
    df = load_merged(path)
    print("Columns found:", df.columns.tolist())

    # keep only rows with cdi and std_wt_24m
    df_reg = df.dropna(subset=['cdi','std_wt_24m']).copy()
    if df_reg.empty:
        print("ERROR: 没有同时含 cdi 和 std_wt_24m 的行。请检查合并文件。")
        sys.exit(1)

    # ensure numeric
    df_reg['cdi'] = pd.to_numeric(df_reg['cdi'], errors='coerce')
    df_reg['std_wt_24m'] = pd.to_numeric(df_reg['std_wt_24m'], errors='coerce')
    df_reg = df_reg.dropna(subset=['cdi','std_wt_24m']).reset_index(drop=True)
    print(f"Using {len(df_reg)} observations for regressions.")

    # ---------- A) Newey-West (HAC) OLS ----------
    print("\n=== A) Newey-West (HAC) OLS: CDI ~ 1 + std_wt_24m ===")
    X = sm.add_constant(df_reg['std_wt_24m'])
    y = df_reg['cdi']
    # choose maxlags: rule of thumb ~ floor(4*(n_obs/100)^(2/9)) but small sample -> use 3
    hac_lags = 3
    res_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    print(res_hac.summary())
    save_text(os.path.join(OUT_DIR, "regression_hac.txt"), res_hac.summary().as_text())

    # ---------- B) Standardize std_wt_24m and OLS with HC1 ----------
    print("\n=== B) Standardized std_wt_24m (z-score) and OLS (HC1) ===")
    df_reg['std_wt_z'] = (df_reg['std_wt_24m'] - df_reg['std_wt_24m'].mean()) / df_reg['std_wt_24m'].std(ddof=0)
    Xz = sm.add_constant(df_reg['std_wt_z'])
    res_z = sm.OLS(df_reg['cdi'], Xz).fit(cov_type='HC1')
    print(res_z.summary())
    save_text(os.path.join(OUT_DIR, "regression_stdz_hc1.txt"), res_z.summary().as_text())

    # report standardized effect: coefficient on std_wt_z
    coef_z = res_z.params.get('std_wt_z', np.nan)
    print(f"\nStandardized effect: 1 std increase in std_wt_24m -> CDI change = {coef_z:.4f}")

    # ---------- C) Scatter plot + fitted lines, descriptive stats ----------
    print("\n=== C) Scatter plot + fitted line and descriptive stats ===")
    desc = df_reg[['std_wt_24m','cdi']].describe().T
    print("\nDescriptive statistics:\n", desc.to_string())

    # Scatter plot with two fitted lines: HAC fit (on original scale) and standardized-fit (same line, just scaled)
    plt.figure(figsize=(8,6))
    plt.scatter(df_reg['std_wt_24m'], df_reg['cdi'], label='observations', alpha=0.7)
    # HAC line:
    coef_hac = res_hac.params
    xs = np.linspace(df_reg['std_wt_24m'].min(), df_reg['std_wt_24m'].max(), 100)
    ys_hac = coef_hac['const'] + coef_hac['std_wt_24m'] * xs
    plt.plot(xs, ys_hac, color='red', label=f'HAC fit: CDI = {coef_hac["const"]:.3f} + {coef_hac["std_wt_24m"]:.3f} * std_wt')

    # Standardized line (res_z): convert std_z line back to original x scale for plotting
    coef_z_ = res_z.params
    mean_x = df_reg['std_wt_24m'].mean()
    sd_x = df_reg['std_wt_24m'].std(ddof=0)
    # model: CDI = const_z + beta_z * ((x-mean)/sd)
    ys_z = coef_z_['const'] + coef_z_['std_wt_z'] * ((xs - mean_x) / sd_x)
    plt.plot(xs, ys_z, color='green', linestyle='--', label=f'Z-fit: CDI = {coef_z_["const"]:.3f} + {coef_z_["std_wt_z"]:.3f} * std_w_z')

    plt.xlabel('std_wt_24m')
    plt.ylabel('CDI')
    plt.title('CDI vs std_wt_24m (HAC and standardized fits)')
    plt.legend()
    plt.grid(alpha=0.25)
    scatter_path = os.path.join(OUT_DIR, "cdi_vs_stdwt_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot -> {scatter_path}")

    # time-series plot of CDI and std_wt_24m
    if 'date' in df_reg.columns:
        fig, ax1 = plt.subplots(figsize=(10,4))
        ax1.plot(df_reg['date'], df_reg['cdi'], label='CDI', linewidth=2)
        ax1.set_ylabel('CDI')
        ax2 = ax1.twinx()
        ax2.plot(df_reg['date'], df_reg['std_wt_24m'], color='C1', label='std_wt_24m', linewidth=1.5)
        ax2.set_ylabel('std_wt_24m')
        ax1.set_xlabel('date')
        plt.title('Time series: CDI and std_wt_24m')
        plt.tight_layout()
        ts_path = os.path.join(OUT_DIR, "cdi_stdwt_timeseries.png")
        plt.savefig(ts_path, dpi=150)
        plt.close()
        print(f"Saved time-series plot -> {ts_path}")

    # save descriptive stats to csv and text
    desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))
    save_text(os.path.join(OUT_DIR, "descriptive_stats.txt"), desc.to_string())
    print("Saved descriptive stats.")

    print("\nAll done. Check outputs in folder:", OUT_DIR)
    print("Notes:")
    print(f" - HAC Newey-West lags used = {hac_lags}. 如需不同 lag 请修改脚本并重跑。")
    print(" - 标准化系数表示 '1 std 的 std_wt_24m 变化对应的 CDI 变化'。")

if __name__ == "__main__":
    main()

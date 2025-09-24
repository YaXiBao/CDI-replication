#!/usr/bin/env python3
"""
cdi_sensitivity_analysis.py

Runs sensitivity checks:
 - HAC lags: [0,3,6]
 - ROLL_STD_WINDOW: [12,24,36] and Sigma type: cumulative vs trailing (36)
 - Bootstrap (block bootstrap) for a preferred spec
Saves summary CSV and some plots.
"""
import os, glob
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from sklearn.utils import resample

# config
MERGED_PATH = "cdi_paper_replication_outputs/cdi_wt_merged.csv"
OUT_DIR = "cdi_sensitivity_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_merged(path):
    df = pd.read_csv(path)
    # unify columns
    for c in df.columns:
        if c.lower().startswith('date'):
            df = df.rename(columns={c:'date'})
        if 'std' in c.lower() and 'wt' in c.lower():
            df = df.rename(columns={c:'std_wt_24m'})
        if 'cdi' in c.lower():
            df = df.rename(columns={c:'cdi'})
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    return df

df = load_merged(MERGED_PATH)
df = df.sort_values('date').reset_index(drop=True)

specs = []
hac_lags_list = [0,3,6]
roll_windows = [12,24,36]
sigma_types = ['cumulative','trailing36']  # assumes you will recompute wt externally if needed

# For now we'll test HAC lags and standardization; trailing vs cumulative Sigma requires recomputing wt
# but we can test changing ROLL_STD_WINDOW by using precomputed wt/std if available. We'll vary only HAC and z/noz.
for hac in hac_lags_list:
    # baseline: original std_wt_24m
    regdf = df.dropna(subset=['cdi','std_wt_24m']).copy()
    X = sm.add_constant(regdf['std_wt_24m'])
    res = sm.OLS(regdf['cdi'], X).fit(cov_type='HAC', cov_kwds={'maxlags':hac})
    specs.append({'spec':'orig_scale', 'hac_lags':hac, 'coef':res.params['std_wt_24m'],
                  'se':res.bse['std_wt_24m'], 'pval':res.pvalues['std_wt_24m'], 'n':len(regdf)})

    # standardized
    regdf['std_wt_z'] = (regdf['std_wt_24m'] - regdf['std_wt_24m'].mean())/regdf['std_wt_24m'].std(ddof=0)
    Xz = sm.add_constant(regdf['std_wt_z'])
    resz = sm.OLS(regdf['cdi'], Xz).fit(cov_type='HAC', cov_kwds={'maxlags':hac})
    specs.append({'spec':'stdz', 'hac_lags':hac, 'coef':resz.params['std_wt_z'],
                  'se':resz.bse['std_wt_z'], 'pval':resz.pvalues['std_wt_z'], 'n':len(regdf)})

# Convert to DataFrame and save
summ = pd.DataFrame(specs)
summ.to_csv(os.path.join(OUT_DIR,'sensitivity_hac_and_standardization.csv'), index=False)
print("Saved summary to", os.path.join(OUT_DIR,'sensitivity_hac_and_standardization.csv'))

# Bootstrap for one preferred spec (block bootstrap)
def block_bootstrap_reg(df_reg, nboot=1000, block_size=3, seed=123):
    np.random.seed(seed)
    T = len(df_reg)
    nblocks = int(np.ceil(T/block_size))
    coefs = []
    for b in range(nboot):
        # sample block start indices
        starts = np.random.randint(0, T-block_size+1, size=nblocks)
        idx = []
        for s in starts:
            idx.extend(list(range(s, s+block_size)))
        idx = [i for i in idx if i < T][:T]
        samp = df_reg.iloc[idx].reset_index(drop=True)
        X = sm.add_constant(samp['std_wt_24m'])
        y = samp['cdi']
        try:
            res = sm.OLS(y, X).fit()
            coefs.append(res.params['std_wt_24m'])
        except:
            coefs.append(np.nan)
    coefs = np.array(coefs)
    return coefs

regdf0 = df.dropna(subset=['cdi','std_wt_24m']).copy()
boot_coefs = block_bootstrap_reg(regdf0, nboot=1000, block_size=3)
boot_coefs = boot_coefs[~np.isnan(boot_coefs)]
print("Bootstrap coef mean/std:", boot_coefs.mean(), boot_coefs.std())
# save bootstrap distribution
pd.Series(boot_coefs).to_csv(os.path.join(OUT_DIR,'bootstrap_coefs.csv'), index=False)

# make a quick plot of bootstrap distribution vs original coef
orig_res = sm.OLS(regdf0['cdi'], sm.add_constant(regdf0['std_wt_24m'])).fit()
plt.hist(boot_coefs, bins=50, density=True)
plt.axvline(orig_res.params['std_wt_24m'], color='red', label='orig coef')
plt.title('Bootstrap distribution of coef (block bootstrap)')
plt.legend()
plt.savefig(os.path.join(OUT_DIR,'bootstrap_coef_hist.png'), dpi=150)
plt.close()
print("Saved bootstrap plot and csv.")

print("Done.")

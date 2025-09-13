
import itertools
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from .utils.io import load_json, save_json

def _pairwise(iterable):
    items = list(iterable)
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            yield items[i], items[j]

def jaccard_overlap_by_month(picks_by_seed: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    months = sorted(set.intersection(*[set(df['year_month']) for df in picks_by_seed.values()]))
    rows = []
    for m in months:
        sets = {s: set(df.loc[df['year_month']==m, 'ticker']) for s, df in picks_by_seed.items()}
        vals = []
        for (s1, s2) in _pairwise(sets.keys()):
            a, b = sets[s1], sets[s2]
            denom = len(a | b) if (a or b) else 1
            vals.append(len(a & b) / denom)
        if len(vals):
            rows.append({
                "year_month": m,
                "avg_jaccard": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_pairs": int(len(vals))
            })
    return pd.DataFrame(rows).set_index("year_month")

def return_correlation(series_by_seed: Dict[int, pd.Series]) -> pd.DataFrame:
    months = sorted(set.intersection(*[set(s.index) for s in series_by_seed.values()]))
    aligned = [series_by_seed[k].reindex(months) for k in series_by_seed.keys()]
    mat = np.corrcoef(np.vstack([a.values for a in aligned]))
    keys = list(series_by_seed.keys())
    return pd.DataFrame(mat, index=keys, columns=keys)

def params_table(params_by_seed: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(params_by_seed).T.sort_index()
    rename = {"learning_rate":"eta", "reg_lambda":"lambda", "reg_alpha":"alpha"}
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df

def params_dispersion(df_params: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for col in df_params.columns:
        vals = pd.to_numeric(df_params[col], errors='coerce').dropna()
        if len(vals) < 2: continue
        rng = vals.max() - vals.min()
        std = vals.std(ddof=1)
        mean = vals.mean()
        cv = std / (abs(mean) + 1e-12)
        stats.append({"param": col, "mean": float(mean), "std": float(std), "cv": float(cv), "range": float(rng)})
    return pd.DataFrame(stats).set_index("param").sort_values("cv", ascending=False)

def euclidean_distance_matrix(df_params: pd.DataFrame) -> pd.DataFrame:
    num = df_params.apply(pd.to_numeric, errors='coerce')
    z = (num - num.mean()) / (num.std(ddof=1) + 1e-12)
    seeds = z.index.tolist()
    dists = np.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            diff = z.iloc[i].fillna(0) - z.iloc[j].fillna(0)
            dists[i, j] = np.sqrt(np.nansum(diff.values**2))
    return pd.DataFrame(dists, index=seeds, columns=seeds)

def summarize_stability(jaccard_df: pd.DataFrame, corr_df: pd.DataFrame, params_var_df: pd.DataFrame):
    summary = {}
    if jaccard_df is not None and not jaccard_df.empty:
        summary.update({
            "avg_jaccard_mean": float(jaccard_df["avg_jaccard"].mean()),
            "avg_jaccard_p25": float(jaccard_df["avg_jaccard"].quantile(0.25)),
            "avg_jaccard_p75": float(jaccard_df["avg_jaccard"].quantile(0.75))
        })
    if corr_df is not None and not corr_df.empty:
        mask = ~np.eye(corr_df.shape[0], dtype=bool)
        summary["avg_return_corr"] = float(corr_df.values[mask].mean())
    if params_var_df is not None and not params_var_df.empty:
        summary["top_unstable_params"] = params_var_df.head(5).index.tolist()
    return summary

def save_stability_report(out_dir: str,
                          jaccard_df: pd.DataFrame,
                          corr_df: pd.DataFrame,
                          params_df: pd.DataFrame,
                          params_var_df: pd.DataFrame,
                          extras):
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    if jaccard_df is not None and not jaccard_df.empty:
        jaccard_df.to_csv(p / "monthly_jaccard.csv")
    if corr_df is not None and not corr_df.empty:
        corr_df.to_csv(p / "return_correlation.csv")
    if params_df is not None and not params_df.empty:
        params_df.to_csv(p / "best_params_by_seed.csv")
    if params_var_df is not None and not params_var_df.empty:
        params_var_df.to_csv(p / "params_dispersion.csv")
    from .utils.io import save_json
    save_json(dict(extras), p / "stability_summary.json")

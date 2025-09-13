
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from ..data.loaders import load_prepared_dataframe
from ..data.preprocess import apply_user_preprocessing
from ..modeling.datasets import make_rank_dataset
from ..modeling.models import make_xgb_ranker, fit_ranker, predict_scores
from .portfolio import build_topk_per_month, evaluate_portfolio
from .utils.io import load_json
from .shap_utils import compute_shap, save_waterfall_for_row, save_global_summary, save_dependence_plots

def simulate_year_with_params(path_or_df,
                              params_json_path: str,
                              year: int,
                              k_select: int = 10,
                              out_dir: str = "outputs/simulation",
                              save_shap: bool = True) -> Dict[str,Any]:
    meta = load_json(params_json_path)
    params = meta["best_params"]
    df, features, target = load_prepared_dataframe(path_or_df)
    df = apply_user_preprocessing(df)
    df["year_month"] = pd.PeriodIndex(df["year_month"].astype(str), freq="M").astype(str)

    months = sorted(df["year_month"].unique())
    months_year = [m for m in months if m.startswith(str(year))]
    results = []
    picks_all = []

    for m in months_year:
        idx = months.index(m)
        train_months = months[:idx]
        infer_months = [m]
        if len(train_months) < 3:
            continue

        train_df = df[df["year_month"].isin(train_months)].copy()
        infer_df  = df[df["year_month"].isin(infer_months)].copy()

        X_tr, y_tr, g_tr, _ = make_rank_dataset(train_df, features, target)
        X_te, y_te, g_te, meta_te = make_rank_dataset(infer_df, features, target)

        booster_params, n_rounds = make_xgb_ranker(params)
        booster = fit_ranker(X_tr, y_tr, g_tr, params=booster_params, num_boost_round=n_rounds, early_stopping_rounds=50)

        scores = predict_scores(booster, X_te)
        topk = build_topk_per_month(meta_te, scores, k=k_select)
        realized = infer_df[["year_month","ticker", target]].rename(columns={target:"target"})
        met = evaluate_portfolio(topk, realized, bench_returns=None)
        ser = met["series"]
        results.append(ser)
        picks_all.append(topk)

        if save_shap:
            month_dir = Path(out_dir) / f"{year}" / m
            month_dir.mkdir(parents=True, exist_ok=True)
            shap_vals, base_val = compute_shap(booster, X_te, features)
            save_global_summary(shap_vals, X_te, features, str(month_dir / "global"))
            save_dependence_plots(shap_vals, X_te, features, str(month_dir / "dependence"), max_features=12)
            merged = topk.merge(infer_df[["year_month","ticker"] + features], on=["year_month","ticker"], how="left")
            for _, row in merged.iterrows():
                ticker = row["ticker"]
                idxs = np.where((meta_te["ticker"].values == ticker) & (meta_te["year_month"].values == m))[0]
                if len(idxs)==0: continue
                i = int(idxs[0])
                out_path = month_dir / f"waterfall_{ticker}.png"
                save_waterfall_for_row(shap_vals[i], base_val, X_te[i], features, str(out_path), max_display=20)

            topk.to_csv(month_dir / "topk.csv", index=False)

    series = pd.concat(results).sort_index() if results else pd.Series(dtype=float)
    picks = pd.concat(picks_all) if picks_all else pd.DataFrame(columns=["year_month","ticker","score"])

    out_root = Path(out_dir) / f"{year}"
    out_root.mkdir(parents=True, exist_ok=True)
    series.to_csv(out_root / "monthly_portfolio_return.csv")
    picks.to_csv(out_root / "all_picks.csv", index=False)

    summary = {
        "year": year,
        "n_months": int(series.shape[0]) if series.shape[0] else 0,
        "mean": float(series.mean()) if series.shape[0] else None,
        "std": float(series.std(ddof=1)) if series.shape[0] else None,
        "sharpe": float(series.mean()/(series.std(ddof=1)+1e-12)) if series.shape[0] else None,
    }
    with open(out_root / "summary.json", "w") as f:
        import json; json.dump(summary, f, indent=2)
    return {"series_csv": str(out_root / "monthly_portfolio_return.csv"),
            "picks_csv": str(out_root / "all_picks.csv"),
            "summary_json": str(out_root / "summary.json")}

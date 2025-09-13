
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.backtestz.optimize_yearly import optimize_for_next_year
from src.backtestz.simulate_with_params import simulate_year_with_params
from src.analysis.stability import (
    jaccard_overlap_by_month, return_correlation, params_table, params_dispersion,
    euclidean_distance_matrix, summarize_stability, save_stability_report
)
from src.utils.io import load_json

def run_stability(
    data_path: str,
    target_year: int,
    seeds: List[int],
    metric_name: str = "log_alpha",
    metric_kwargs: Dict[str,Any] = None,
    n_trials: int = 20,
    k_select: int = 10,
    base_out: str = "outputs/stability"
):
    params_dir = Path("outputs/params"); params_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = Path("outputs/simulation_seeded") / str(target_year)
    sim_dir.mkdir(parents=True, exist_ok=True)

    params_paths = {}
    for s in seeds:
        _ = optimize_for_next_year(
            path_or_df=data_path,
            target_year=target_year,
            k_select=k_select,
            metric_name=metric_name,
            metric_kwargs=metric_kwargs,
            n_trials=n_trials,
            seed=s,
            params_out_dir=str(params_dir)
        )
        src = params_dir / f"params_{target_year}.json"
        dst = params_dir / f"params_{target_year}_seed{s}.json"
        dst.write_text(src.read_text())
        params_paths[s] = str(dst)

    series_by_seed = {}
    picks_by_seed = {}
    for s in seeds:
        out = simulate_year_with_params(
            path_or_df=data_path,
            params_json_path=params_paths[s],
            year=target_year,
            k_select=k_select,
            out_dir=str(sim_dir / f"seed_{s}"),
            save_shap=False
        )
        series = pd.read_csv(out["series_csv"], index_col=0, parse_dates=True).iloc[:,0]
        picks  = pd.read_csv(out["picks_csv"])
        series_by_seed[s] = series
        picks_by_seed[s] = picks

    from src.analysis.stability import summarize_stability, save_stability_report
    jac_df = jaccard_overlap_by_month(picks_by_seed)
    corr_df = return_correlation(series_by_seed)
    params_by_seed = {s: load_json(params_paths[s])["best_params"] for s in seeds}
    df_params = params_table(params_by_seed)
    df_disp   = params_dispersion(df_params)
    dist_mat  = euclidean_distance_matrix(df_params)
    (Path(base_out) / str(target_year)).mkdir(parents=True, exist_ok=True)
    dist_mat.to_csv(Path(base_out) / str(target_year) / "params_distance_matrix.csv")

    summary = summarize_stability(jac_df, corr_df, df_disp)
    summary.update({
        "target_year": target_year,
        "seeds": seeds,
        "metric": metric_name,
        "metric_kwargs": metric_kwargs or {},
        "avg_monthly_jaccard": float(jac_df["avg_jaccard"].mean()) if jac_df is not None and not jac_df.empty else None,
        "avg_return_corr": float(corr_df.values[~np.eye(corr_df.shape[0], dtype=bool)].mean()) if corr_df is not None and not corr_df.empty else None
    })

    report_dir = Path(base_out) / str(target_year)
    save_stability_report(str(report_dir), jac_df, corr_df, df_params, df_disp, summary)
    print("Stability report saved to:", report_dir)

if __name__ == "__main__":
    data_path = "data/example.csv"
    target_year = 2021
    seeds = [101,202,303,404,505]
    run_stability(
        data_path=data_path,
        target_year=target_year,
        seeds=seeds,
        metric_name="log_alpha",
        metric_kwargs={"alpha": 5.0},
        n_trials=10,
        k_select=10,
        base_out="outputs/stability"
    )

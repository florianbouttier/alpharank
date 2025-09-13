
from src.backtestz.optimize_yearly import optimize_for_next_year
from src.backtestz.simulate_with_params import simulate_year_with_params

if __name__ == "__main__":
    data_path = "data/example.csv"
    target_year = 2021
    metric_name = "log_alpha"
    metric_kwargs = {"alpha": 5.0}
    k_select = 10

    best = optimize_for_next_year(
        path_or_df=data_path,
        target_year=target_year,
        k_select=k_select,
        metric_name=metric_name,
        metric_kwargs=metric_kwargs,
        n_trials=20,
        seed=123,
        params_out_dir="outputs/params"
    )
    print("Saved params for", target_year, best)

    params_path = f"outputs/params/params_{target_year}.json"
    sim = simulate_year_with_params(
        path_or_df=data_path,
        params_json_path=params_path,
        year=target_year,
        k_select=k_select,
        out_dir="outputs/simulation",
        save_shap=True
    )
    print("Simulation outputs:", sim)

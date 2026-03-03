
import optuna
import optuna.logging   
def sample_xgb_space(trial: optuna.Trial):
    if trial is None:
        study = optuna.create_study()
        trial = study.ask()
        
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0)
    }

"""{
        "n_estimators": trial.suggest_int("n_estimators", 20, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "seed": trial.suggest_int("seed", 1, 10000),
    }"""
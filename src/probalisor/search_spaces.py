
import optuna
import optuna.logging   
def sample_xgb_space(trial: optuna.Trial):
    if trial is None:
        study = optuna.create_study()
        trial = study.ask()
        
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "subsample": trial.suggest_float("subsample", 0.2, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 10.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "loss_type" : trial.suggest_categorical("loss_type", ["hinge", "asymmetric","none"]),
        "scale": trial.suggest_float("scale", 0.5, 5),
        "epsilon": trial.suggest_float("epsilon", 0.0, 0.1)
    }


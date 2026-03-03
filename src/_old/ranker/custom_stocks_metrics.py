
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Callable

def _as_series(x):
    if isinstance(x, pd.Series):
        return x.sort_index()
    return pd.Series(np.asarray(x)).sort_index()

def mean_return(port: pd.Series, bench: Optional[pd.Series]=None, **kwargs) -> float:
    s = _as_series(port)
    return float(s.mean())

def median_return(port: pd.Series, bench: Optional[pd.Series]=None, **kwargs) -> float:
    s = _as_series(port)
    return float(s.median())

def sharpe_ratio(port: pd.Series, bench: Optional[pd.Series]=None, annualize: bool=False, **kwargs) -> float:
    s = _as_series(port)
    val = s.mean() / (s.std(ddof=1) + 1e-12)
    if annualize:
        val *= np.sqrt(12)
    return float(val)

def sortino_ratio(port: pd.Series, bench: Optional[pd.Series]=None, annualize: bool=False, **kwargs) -> float:
    s = _as_series(port)
    downside = s[s<0.0]
    denom = downside.std(ddof=1) + 1e-12
    val = s.mean() / denom
    if annualize:
        val *= np.sqrt(12)
    return float(val)

def log_alpha_objective(port: pd.Series, bench: Optional[pd.Series], alpha: float=5.0, **kwargs) -> float:
    s = _as_series(port)
    if bench is not None:
        b = _as_series(bench).reindex(s.index).fillna(0.0)
        x = (1.0 + s) / (1.0 + b) - 1.0
    else:
        x = s
    vals = np.log(1.0 + alpha * x)
    vals = np.where(np.isfinite(vals), vals, -1e9)
    return float(np.mean(vals))

def cumulative_log_return(port: pd.Series, **kwargs) -> float:
    s = _as_series(port)
    return float(np.sum(np.log1p(s)))

REGISTRY = {
    "mean": mean_return,
    "median": median_return,
    "sharpe": sharpe_ratio,
    "sortino": sortino_ratio,
    "log_alpha": log_alpha_objective,
    "cum_log": cumulative_log_return,
}

def get_objective_fn(name: str):
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown objective '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]

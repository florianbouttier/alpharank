
from typing import Dict, Optional
import numpy as np
import pandas as pd
# %%
def build_topk_per_month(df: pd.DataFrame, scores, k: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["score"] = np.asarray(scores)
    out = (d.sort_values(["year_month","score"], ascending=[True, False])
             .groupby("year_month")
             .head(k)
             .reset_index(drop=True))
    return out[["year_month","ticker","score"]]

def evaluate_portfolio(topk: pd.DataFrame, realized: pd.DataFrame, bench_returns: Optional[pd.Series]=None) -> Dict:
    merged = topk.merge(realized, on=["year_month","ticker"], how="left")
    rets = (merged.groupby("year_month")["target"].mean()).sort_index()
    rets.index = pd.PeriodIndex(rets.index, freq="M").to_timestamp()
    out = {"series": rets}
    if bench_returns is not None:
        bench = bench_returns.reindex(rets.index).fillna(0.0)
        rel = (1.0 + rets) / (1.0 + bench) - 1.0
        out["relative"] = rel
    return out

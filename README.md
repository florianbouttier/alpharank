# AlphaRank

AlphaRank is a quantitative equity research repository organized around two active tracks:

- **Legacy**: current production-style baseline workflow
- **Boosting**: experimentation pipeline using XGBoost + time-fold walk-forward backtests

## Install

Use editable install (recommended):

```bash
pip install -e .
```

Alternative:

```bash
pip install -r requirements.txt
```

> The repository follows a `src/` layout. Imports must use `alpharank.*` (no `sys.path.append(...)`).

## Active Structure

```text
src/
  alpharank/
    __init__.py
    legacy/               # public legacy API
    boosting/             # public boosting API
    backtest/             # boosting modules (data/features/folds/tuning/shap/report)
    data/
    features/
    models/
    strategy/
    utils/
    visualization/
  _old/                   # archived historical modules (reference only)
```

## Main Entry Scripts

- Legacy pipeline: `scripts/run_legacy.py`
- Boosting pipeline: `scripts/run_backtest.py`

## Library APIs

### Legacy

```python
from alpharank.legacy import StrategyLearner, ModelEvaluator
```

### Boosting

```python
from alpharank.boosting import BacktestConfig, run_boosting_backtest
```

## Boosting Pipeline Outputs

`run_backtest.py` / `run_boosting_backtest` generates:

- fold-level train/val/test metrics
- Optuna trials and selected hyperparameters
- SHAP plots (beeswarm + individual explanation)
- lift/learning curves
- consolidated HTML training report
- parquet/csv/json artifacts in `outputs/`

## Notes for Contributors

- Put new production/experiment code in `src/alpharank/`.
- Keep `src/_old/` untouched unless explicitly cleaning archives.
- Prefer modular code in library modules over logic embedded in scripts.
- Keep legacy and boosting concerns separated.

## Current Focus

- **Production baseline**: legacy workflow
- **R&D**: boosting workflow and fold-based model experimentation

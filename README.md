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

## Clean Conda Environment (Recommended)

Create a dedicated environment and install AlphaRank package + all requirements:

```bash
bash scripts/setup_conda_env.sh alpharank
conda activate alpharank
python scripts/run_backtest.py
```

Manual equivalent:

```bash
conda env create -n alpharank -f environment.yml
conda activate alpharank
python -m pip install -e .
python scripts/run_backtest.py
```

If the environment already exists and dependencies changed:

```bash
conda env update -n alpharank -f environment.yml --prune
conda run -n alpharank python -m pip install -e .
```

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
- Data lineage audit: `scripts/audit_data_lineage.py`

## Data Snapshotting

Refreshing `data/` now keeps an immutable snapshot under `data/_snapshots/<timestamp>/` and updates `data/latest_snapshot.json`.

Each legacy run also writes its own input manifest to:

```text
outputs/YYYY-MM-DD/data_input_manifest.json
```

This makes it possible to distinguish:

- source files changed
- same source files, different processing code

To compare the two latest legacy run manifests:

```bash
python scripts/audit_data_lineage.py
```

## Library APIs

### Legacy

```python
from alpharank.legacy import StrategyLearner, ModelEvaluator
```

### Boosting

```python
from alpharank.boosting import (
    BacktestConfig,
    run_learning_phase,
    run_backtest_phase,
    run_boosting_backtest,
)
```

## Boosting Pipeline Outputs

`run_backtest.py` / `run_boosting_backtest` generates:

- fold-level train/val/test KPIs
- Optuna trials + best hyperparameters + Optuna interactive HTML visualizations
- SHAP outputs (beeswarm PNG/PDF, individual PNG/PDF, dependence plots PNG/PDF for top beeswarm features)
- learning curve (train/validation)
- sorted prediction bucket analysis (20 buckets by default): predicted vs realized frequency on validation and test
- backtest analysis vs SP500 (cumulative, drawdown, active return)
- consolidated HTML report embedding all fold/global assets
- parquet/csv/json artifacts in `outputs/`

## Notes for Contributors

- Put new production/experiment code in `src/alpharank/`.
- Keep `src/_old/` untouched unless explicitly cleaning archives.
- Prefer modular code in library modules over logic embedded in scripts.
- Keep legacy and boosting concerns separated.

## Current Focus

- **Production baseline**: legacy workflow
- **R&D**: boosting workflow and fold-based model experimentation

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
- Python source selection example: `scripts/backtest_data_source_examples.py`
- Open-source price transition audit: `scripts/open_source/run_price_transition.py`
- Unified open-source ingestion: `scripts/open_source/run_ingestion.py`
- Nightly ingestion runner: `scripts/open_source/nightly_ingestion.py`
- Nightly launchd installer: `scripts/open_source/install_nightly_launchd.py`
- Data lineage audit: `scripts/audit_data_lineage.py`

## Open-Source Price Transition

To materialize Yahoo-based price history in the repo's canonical parquet shape and audit it against the existing EODHD reference data:

```python
from scripts.open_source.run_price_transition import main

main(start_date="2005-01-01")
```

This writes a reusable price dataset under `data/open_source/price_transition_20050101/` with:

- `US_Finalprice.parquet`
- `SP500Price.parquet`
- HTML audit reports and per-ticker deep dives

You can then test the backtests with open-source prices only while keeping the existing EODHD financial statements:

```bash
./.venv/bin/python scripts/run_backtest_open_source_prices.py
```

Or for the legacy runner:

```python
from scripts.run_legacy import main

main(
    final_price_path="data/open_source/price_transition_20050101/US_Finalprice.parquet",
    sp500_price_path="data/open_source/price_transition_20050101/SP500Price.parquet",
)
```

## Open-Source Live Ingestion

The live ingestion pipeline writes:

- raw normalized source tables
- clean consolidated tables with lineage
- legacy-compatible parquet exports
- optional HTML audits
- immutable per-run deltas and manifests

Bootstrap the historical store:

```python
from scripts.open_source.run_ingestion import main

main(
    mode="bootstrap",
    start_date="2005-01-01",
    audit_years=(2025,),
)
```

Daily incremental update:

```python
from scripts.open_source.run_ingestion import main

main(
    mode="daily",
    start_date="2005-01-01",
    audit_years=(2025,),
)
```

Default live storage layout:

- `data/open_source/live/raw/`
- `data/open_source/live/clean/`
- `data/open_source/live/clean/legacy_compatible/`
- `data/open_source/live/audits/`
- `data/open_source/live/manifests/`
- `data/open_source/live/runs/`

For the full ingestion contract, lineage rules, natural keys, and the "never delete raw data" policy, see:

- `docs/open_source_ingestion_architecture.md`

## Nightly Ingestion

If you want the ingestion to run automatically during the night, the simplest Python-first setup is:

1. Edit the constants in `scripts/open_source/nightly_ingestion.py`
2. Run `scripts/open_source/install_nightly_launchd.py` once

The nightly runner itself is just:

```python
from scripts.open_source.nightly_ingestion import main

main()
```

By default it refreshes the union of:

- the current S&P 500 universe from `SP500_Constituents.csv`
- tickers already present in `data/open_source/live/raw/`

That means a nightly run does not silently narrow the live store after a broader bootstrap. Already-ingested delisted names stay in the raw store and stay present in the rebuilt clean/legacy exports unless someone manually purges the raw parquet files.

The launchd installer writes a macOS LaunchAgent that runs the nightly Python script using the repo `.venv`.

Logs are written under:

- `logs/open_source_ingestion/stdout.log`
- `logs/open_source_ingestion/stderr.log`

## Python-First Backtest Source Selection

Backtests can now be pointed to a dataset source directly from Python, without using CLI flags.

```python
from alpharank.backtest import BacktestDataSource
from scripts.run_backtest import default_config, run

source = BacktestDataSource.open_source_live()
config = source.apply(default_config())
artifacts = run(config)
```

Available source profiles:

- `BacktestDataSource.eodhd()`
- `BacktestDataSource.open_source_live()`
- `BacktestDataSource.open_source_prices_only()`
- `BacktestDataSource.custom(...)`

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

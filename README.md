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
    backtest/             # boosting modules (data/features/folds/tuning/shap/report)
    multihorizon/         # current multi-horizon boosting and live scoring
    portfolio/            # shared holdings, simulation, metrics and artifacts
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
- Common Legacy/Alpha replay gate: `scripts/validate_common_portfolio_engine.py`
- Python source selection example: `scripts/backtest_data_source_examples.py`
- EODHD exact-name mirror builder: `scripts/sync_eodhd_output.py`
- Open-source price transition audit: `scripts/open_source/run_price_transition.py`
- Unified open-source ingestion: `scripts/open_source/run_ingestion.py`
- Guarded price roll-forward: `scripts/open_source/build_roll_forward_price_package.py`
- Point-in-time SEC raw rebuild: `scripts/open_source/build_sec_raw_version_candidate.py`
- Immutable production snapshot composer: `scripts/open_source/build_composed_model_snapshot.py`
- Open-source exact-name output builder: `scripts/open_source/build_output_package.py`
- Nightly ingestion runner: `scripts/open_source/nightly_ingestion.py`
- Nightly launchd installer: `scripts/open_source/install_nightly_launchd.py`
- Data lineage audit: `scripts/audit_data_lineage.py`
- Strict Legacy replay validation: `scripts/validate_legacy_replay_package.py`

Legacy and boosting keep separate signal/training logic, then adapt finalized
monthly decisions to the same `alpharank.portfolio` contract. See
`docs/common_portfolio_backtest_engine.md` for timing, weighting, cost,
performance, and parity rules.

## Production Data Contract And Anti-Leakage Controls

New production runs must resolve the immutable package referenced by
`data/model_inputs/manifests/latest.json`. Do not use mutable
`data/open_source/output` as current production truth. The composed package
contains:

- strict SEC/GAAP fundamentals, with every raw filing version retained;
- a guarded price package that refreshes active securities from one complete
  Yahoo vintage and preserves the frozen EODHD-seeded inactive/delisted history;
- a persistent published-price registry: once a ticker has appeared in a
  validated snapshot, its complete retained history survives later refreshes
  even when it was first downloaded from Yahoo, is absent from EODHD, leaves
  the active universe, or is no longer downloadable;
- historical S&P 500 membership, SPY total-return prices, source manifests, and
  SHA-256 hashes for every model input.

The 2026-08-16 validated reference is ingestion `20260816_103942`, composed
snapshot `alpharank_input_20260816_120458_2a01288bab06`. Prices and SEC filing
dates reach 2026-08-14. Legacy run `20260816_142810` and Boosting run
`outputs/production_refresh_20260816/boosting_latest_common_v3` consume the
same retained Legacy `input_snapshot/`; the common replay is
`outputs/production_refresh_20260816/common_replay_v3`.

### What Is Rejected Before Publication

The price revision gate fails closed when it finds any of the following outside
the declared seven-day mutable tail:

- an active ticker missing from the single fresh Yahoo vintage;
- active rows mixed across Yahoo vintages or sourced from another provider;
- a historical adjusted-close daily-return revision above 1 bp;
- a historical return changing between available and unavailable;
- a historical price key disappearing;
- a frozen inactive EODHD seed key disappearing;
- any previously published inactive ticker/date disappearing, including
  Yahoo-only histories that have no EODHD seed;
- an unexplained split-adjustment transition, ticker-reuse splice, or price
  bridge with insufficient overlap or a gap above ten calendar days.

Confirmed splits/dividends must be declared in the versioned corporate-action
registry and produce a new historized derived package. Published snapshots and
the frozen EODHD seed are never edited in place.

`scripts/open_source/build_roll_forward_price_package.py` resolves the previous
validated lineage from `data/model_inputs/manifests/latest.json` by default and
writes `lineage/persistent_price_history_registry.parquet`. The registry records
each ticker's first/last date, row count, sources, latest vintage, active status,
and persistence class. An explicit previous-lineage path remains available only
for controlled replay or migration.

Fundamental publication is also fail-closed:

- official model values may come only from SEC Companyfacts, filing-level SEC
  XBRL, or an explicitly labelled derivation using only SEC facts;
- the raw natural key includes `filing_date`, so later restatements cannot
  overwrite what was originally filed;
- model exports select the earliest available filing version, and monthly
  features become available only from their filing/report date;
- historical revisions beyond the declared mutable window require a written,
  retained migration review and cannot be silently accepted.

### Causal Model Calendar

- Information observed in decision month `t` can only create holdings for
  month `t+1`.
- The still-open calendar month is retained only as freshness evidence and is
  removed before Legacy feature construction.
- Historical S&P membership and monthly tradability are evaluated point in
  time, rather than from today's constituent list.
- Boosting uses chronological expanding train/validation/test folds. Legacy EMA
  winner features for each fold are restricted to winners known by that fold's
  training cutoff.
- A future H6 label is forced to null whenever its return window extends beyond
  the last completed month. Pending rows may be scored and explained by SHAP,
  but cannot enter model-quality metrics or realized performance.
- Backtest performance includes a holding month only after its complete
  one-month stock and SPY total returns exist.

### Permanent Ticker Quarantine

`configs/data_quality/historical_ticker_exclusions_v1.json` excludes these ten
entire trajectories from both Legacy and Boosting:

| Ticker | Why the full trajectory is excluded |
| --- | --- |
| `SII.US` | ticker reuse, identity mismatch, post-delisting contamination |
| `CBE.US` | observations continuing after the 2012 delisting |
| `TIE.US` | post-delisting contamination and impossible price scale |
| `CPWR.US` | corrupt returns and unresolved adjustment chain |
| `BMC.US` | impossible scale, OHLC errors, post-delisting contamination |
| `COL.US` | wrong security/price series and post-delisting contamination |
| `GR.US` | wrong, illiquid series continuing after delisting |
| `EP.US` | ticker reuse between El Paso and Empire Petroleum |
| `SW.US` | pre-2024 history spliced onto a security first listed in 2024 |
| `HAR.US` | impossible scale, OHLC errors, post-delisting contamination |

The registry includes source links, official start/terminal dates, and the
reason for every exclusion. It is hash-recorded in each run manifest. Adding or
removing a ticker is a versioned data-quality decision, not an ad hoc backtest
parameter.

Separately, `monthly_price_eligibility_v1` excludes only a ticker-month when it
has fewer than 10 price observations, median daily dollar volume below USD 1m,
or more than 5% invalid OHLC rows. A ticker can become eligible again later.
Legacy additionally requires point-in-time S&P membership, market
capitalization, `0 < PE < 100`, and applies its sector cap. The current public
Boosting profile uses price/relative-EMA features and does not use fundamental
features, but it must consume and hash the same snapshot.

### Verification Suite

The production-data suite covers snapshot composition, price lineage and
corporate actions, SEC-only exports and filing versions, atomic publication,
historical revision guards, Legacy lineage, completed-month target masking,
walk-forward behavior, common portfolio simulation, CAGR attribution, and the
research dashboard:

```bash
PYTHONPATH=. ./.venv/bin/pytest -q \
  tests/test_composed_snapshot.py \
  tests/test_price_composition.py tests/test_price_gates.py \
  tests/test_price_corporate_actions.py tests/test_price_seed.py \
  tests/test_open_source_price_quality.py tests/test_open_source_sec_only.py \
  tests/test_sec_raw_versions.py tests/test_open_source_financial_versions.py \
  tests/test_open_source_transaction.py tests/test_open_source_publishing.py \
  tests/test_open_source_consolidation.py tests/test_open_source_earnings.py \
  tests/test_open_source_legacy_export.py tests/test_open_source_price_fallback.py \
  tests/test_open_source_sec_mapping.py tests/test_open_source_yahoo.py \
  tests/test_open_source_freshness.py tests/test_open_source_fundamental_quality.py \
  tests/test_open_source_refresh_policy.py tests/test_open_source_revision_guard.py \
  tests/test_run_legacy_open_source_lineage.py \
  tests/test_multihorizon_research.py tests/test_multihorizon_confirmation.py \
  tests/test_portfolio_engine.py tests/test_portfolio_attribution.py \
  tests/test_central_research_dashboard.py
```

After a production Legacy run, also validate its immutable replay package:

```bash
./.venv/bin/python scripts/validate_legacy_replay_package.py \
  --strict-code outputs/<run>/data_input_manifest.json
```

The targeted production-data suite passes 185 tests; the full repository suite
passes 250 tests. The current Legacy package also passes strict replay
validation.
See `docs/monthly_portfolio_runbook.md`,
`docs/open_source_ingestion_architecture.md`, and
`docs/legacy_boosting_methodology.md` for the complete operational contracts.

## Open-Source Price Transition

To materialize Yahoo-based price history in the repo's canonical parquet shape and audit it against the existing EODHD reference data:

```python
from scripts.open_source.run_price_transition import main

main(start_date="2005-01-01")
```

This writes a reusable price dataset under `data/open_source/audit/price_transition_20050101/` with:

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
    final_price_path="data/open_source/audit/price_transition_20050101/US_Finalprice.parquet",
    sp500_price_path="data/open_source/audit/price_transition_20050101/SP500Price.parquet",
)
```

## Open-Source Official Ingestion

The official ingestion pipeline writes:

- raw normalized source tables
- target consolidated tables with lineage
- target earnings and general-reference consolidations with lineage
- published exact-name outputs for backtests/manual inspection
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

- `data/open_source/official/raw/`
- `data/open_source/official/target/`
- `data/open_source/official/target/legacy_compatible/`
- `data/open_source/output/`
- `data/open_source/output/lineage/`
- `data/open_source/history/output/`
- `data/open_source/official/manifests/`
- `data/open_source/official/runs/`
- `data/open_source/audit/`
- `data/open_source/archive/`

Legacy reference datasets are mirrored in:

- `data/eodhd/output/`

For the full ingestion contract, lineage rules, natural keys, and the "never delete raw data" policy, see:

- `docs/open_source_ingestion_architecture.md`
- `docs/sec_fundamentals_contract.md`

If you want the exact legacy filenames in one user-facing folder, open:

- `data/open_source/output/`

The associated lineage package is:

- `data/open_source/output/lineage/`

The user-facing lineage folder now contains the official exported lineage of the new model, including:

- `financials_open_source_consolidated.parquet`
- `financials_open_source_lineage.parquet`
- `financials_open_source_source_summary.parquet`
- `general_reference.parquet`
- `general_reference_lineage.parquet`
- `earnings_open_source_consolidated.parquet`
- `earnings_open_source_lineage.parquet`
- `earnings_open_source_long.parquet`
- `manifest.json`

## Official Data Packages

There are now three package roles to keep distinct:

1. `data/eodhd/output/`
   - legacy mirror and historical reference package
2. `data/open_source/output/`
   - multi-source open-source package
   - source priority can mix SEC / filing / SimFin / Yahoo depending on dataset
3. `data/sec/output/`
   - fundamentals-only SEC package
   - one source of truth for fundamentals: SEC
   - no Yahoo, no SimFin, no EODHD fallback inside this package

Current operating status for fundamentals:

- default reference package: `data/sec/output/`
- current KPI objective: close the gaps on `epsActual`, `revenue`, and `net_income`
- quality target: less than `1 %` missing on the worst audited year
- best current candidate: `outputs/sec_kpi_hybrid_output_latest/`
- latest external-facing status note: `docs/sec_open_source_status.md`
- scenario comparison report: `outputs/sec_kpi_scenario_comparison_latest/summary.md`

Important:

- `data/open_source/output/` remains useful for the broader multi-source ingestion track
- it is **not** the package currently used to pilot the `<1 %` KPI-gap objective
- that objective is tracked on the SEC-only branch and its overlay experiments

Useful SEC-only candidate workflows:

- automated candidate rebuild: `scripts/open_source/run_sec_q4_fix2_candidate.py`
- scenario comparison builder: `scripts/open_source/build_sec_kpi_scenario_comparison.py`

Use these docs when touching the model:

- `docs/open_source_ingestion_architecture.md`
- `docs/sec_fundamentals_contract.md`
- `docs/sec_open_source_status.md`

The SEC package exists to keep the fundamental lineage simple and explicit:

- source canonique: SEC
- access path 1: `companyfacts`
- access path 2: `filing-level XBRL`
- no external vendor fallback
- historical/delisted ticker recovery can use a local `ticker -> CIK` bridge, but final fundamental values still come from SEC only

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
- tickers already present in `data/open_source/official/raw/`

That means a nightly run does not silently narrow the official store after a broader bootstrap. Already-ingested delisted names stay in the raw store and stay present in the rebuilt target/legacy exports unless someone manually purges the raw parquet files.

The launchd installer writes a macOS LaunchAgent that runs the nightly Python script using the repo `.venv`.

Logs are written under:

- `logs/open_source_ingestion/stdout.log`
- `logs/open_source_ingestion/stderr.log`

Nightly run state is written under:

- `data/open_source/official/manifests/nightly_status.json`
- `data/open_source/official/manifests/nightly.lock.json`

The lock file prevents two nightly runs from writing into the store at the same time. If one run is already active when launchd fires again, the second process exits cleanly instead of overlapping the first one.

## Python-First Backtest Source Selection

Backtests can now be pointed to a dataset source directly from Python, without using CLI flags.

```python
from alpharank.backtest import BacktestDataSource
from scripts.run_backtest import default_config, run

source = BacktestDataSource.open_source_official()
config = source.apply(default_config())
artifacts = run(config)
```

Available source profiles:

- `BacktestDataSource.eodhd()`
- `BacktestDataSource.open_source_official()`
- `BacktestDataSource.open_source_live()`
- `BacktestDataSource.open_source_prices_only()`
- `BacktestDataSource.custom(...)`

`BacktestDataSource.eodhd()` now points to `data/eodhd/output/`, and `BacktestDataSource.open_source_official()` points to `data/open_source/output/`. The folder switch is therefore symmetric.

## Data Snapshotting

Production model packages are immutable directories under
`data/model_inputs/history/`. The only mutable entrypoint is the small pointer
`data/model_inputs/manifests/latest.json`; consumers resolve it once, then copy
the referenced package into the run's timestamped `input_snapshot/` using APFS
copy-on-write cloning where available.

Each Legacy run writes its own input manifest to:

```text
outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json
```

Monthly portfolio production is documented in
[`docs/monthly_portfolio_runbook.md`](docs/monthly_portfolio_runbook.md).

The manifest records input hashes, critical-code hashes, exclusion-policy hash,
liquidity thresholds, completed-month cutoff, storage method, and source
lineage. This distinguishes:

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

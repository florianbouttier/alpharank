# Legacy Benchmark Repo

Toolkit de benchmark pour `run_legacy` (version polars-only) sur un subset contrôlé.

## Structure
- `scripts/create_subset.py`: construit un subset déterministe des données legacy.
- `scripts/run_benchmark.py`: exécute benchmark wall-time polars (warmup + runs mesurés).
- `data/`: subsets générés.
- `results/`: métriques benchmark (JSON/CSV + outputs/checkpoints).
- `logs/`: logs complets des runs.

## 1) Générer un subset clair
```python
from scripts.create_subset import main

main(
    source_data_dir="/Users/nicolas.rusinger/AlphaRank/data",
    target_data_dir="data/subset_legacy_v1",
    start_date="2019-01-01",
    end_date="2022-12-31",
    max_tickers=80,
    fundamentals_lookback_days=730,
)
```

## 2) Lancer benchmark
```python
from scripts.run_benchmark import main

main(
    alpharank_repo="/Users/nicolas.rusinger/AlphaRank",
    data_dir="data/subset_legacy_v1",
    results_dir="results/legacy_v1",
    logs_dir="logs/legacy_v1",
    n_trials=2,
    first_date="2020-01",
    warmups=1,
    runs=3,
)
```

## Sorties
- `results/.../benchmark_summary.json`
- `results/.../benchmark_summary.csv`
- `results/.../outputs/`
- `results/.../checkpoints/`

## Rejouer complètement le benchmark
1. Recréer le subset via `create_subset.py` (step 1).
2. Relancer `run_benchmark.py` avec un nouveau dossier de résultats (ex: `results/legacy_rerun_YYYYMMDD`).
3. Comparer les fichiers `benchmark_summary.json` entre runs.

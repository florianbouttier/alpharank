# Legacy Benchmark Repo

Toolkit de benchmark pour `run_legacy` (version polars-only) sur un subset contrôlé.

## Structure
- `scripts/create_subset.py`: construit un subset déterministe des données legacy.
- `scripts/run_benchmark.py`: exécute benchmark wall-time polars (warmup + runs mesurés).
- `data/`: subsets générés.
- `results/`: métriques benchmark (JSON/CSV + outputs/checkpoints).
- `logs/`: logs complets des runs.

## 1) Générer un subset clair
```bash
python3 scripts/create_subset.py \
  --source-data-dir /Users/nicolas.rusinger/AlphaRank/data \
  --target-data-dir data/subset_legacy_v1 \
  --start-date 2019-01-01 \
  --end-date 2022-12-31 \
  --max-tickers 80 \
  --fundamentals-lookback-days 730
```

## 2) Lancer benchmark
```bash
python3 scripts/run_benchmark.py \
  --alpharank-repo /Users/nicolas.rusinger/AlphaRank \
  --data-dir data/subset_legacy_v1 \
  --results-dir results/legacy_v1 \
  --logs-dir logs/legacy_v1 \
  --n-trials 2 \
  --first-date 2020-01 \
  --warmups 1 \
  --runs 3
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

# Experiment log

> **Invalidation notice:** the initial screening and shortlisted CPCV sections
> below are retained as an incident history only. A raw
> `weight_normalized` Legacy column leaked into the economic feature list.
> Their numerical conclusions are invalid. The corrected source of truth is
> `performance_report.md` and run
> `outputs/multihorizon_boosting/screening_clean_20260725`.

## 2026-07-25 — implementation

Hypothesis:

> A common, leakage-aware comparison of classification, robust regression and
> grouped ranking over several horizons can identify whether the prior failures
> came from the objective, the one-month horizon or the feature representation.

Data:

- frozen snapshot:
  `outputs/2026-07-19/runs/20260719_194418/input_snapshot`;
- exact associated detailed Legacy output:
  `outputs/2026-07-19/runs/20260719_194418/legacy_detailed_returns_polars.parquet`;
- exact associated monthly Legacy output:
  `outputs/2026-07-19/runs/20260719_194418/legacy_monthly_returns_polars.parquet`.

Implementation:

- Added generic grouped XGBoost ranking to sibling `mlcraft`.
- Added the dedicated `alpharank.multihorizon` package.
- Added strict calendar targets, horizon-aware walk-forward, interval-purged
  CPCV, train-only preprocessing, calibrated classifiers, portfolio metrics and
  out-of-sample SHAP.
- Added horizons 24 and 36 as exploratory tracks.

Validation completed before the full run:

- `7` focused `mlcraft` tests passed.
- `6` focused AlphaRank tests passed.
- Real snapshot frame built: `113351` rows, `386` columns, `335` eligible
  features, January 2005 through April 2026.
- One-fold regression smoke produced predictions, portfolio metrics and all
  three SHAP artifacts.

Operational note:

The initial run attempted to duplicate the 128 MB research panel and filled the
nearly full volume. Only the newly generated smoke and interrupted run were
removed. The CLI now stores input paths, panel dimensions and protocol in the
manifest and does not duplicate the panel unless `--save-research-frame` is
explicitly passed.

## Screening command

```bash
./.venv/bin/python scripts/experiments/run_multihorizon_boosting.py \
  --data-dir outputs/2026-07-19/runs/20260719_194418/input_snapshot \
  --legacy-detailed outputs/2026-07-19/runs/20260719_194418/legacy_detailed_returns_polars.parquet \
  --legacy-monthly outputs/2026-07-19/runs/20260719_194418/legacy_monthly_returns_polars.parquet \
  --run-dir outputs/multihorizon_boosting/screening_20260725 \
  --horizons 1,3,6,12,24,36 \
  --methods classification,regression,ranking,teacher \
  --min-train-months 72 \
  --validation-months 24 \
  --test-months 12 \
  --step-months 12 \
  --num-boost-round 100 \
  --shap-sample-per-fold 50
```

Primary results:

| method | horizon | folds | monthly IC | top-10 horizon excess | top-10 one-month excess | Legacy overlap |
|---|---:|---:|---:|---:|---:|---:|
| regression | 24 | 7 | 0.0959 | 17.71% | 1.15% | 10.67% |
| ranking | 24 | 7 | 0.0485 | 12.86% | 1.01% | 23.82% |
| classification | 24 | 7 | 0.0243 | 13.24% | 0.97% | 19.03% |
| regression | 36 | 4 | 0.0492 | 24.50% | 1.27% | 18.85% |
| classification | 36 | 4 | 0.0571 | 14.92% | 1.36% | 21.75% |
| Legacy teacher | 1 | 8 | 0.0049 | 0.60% | 0.60% | 45.83% |

The 24-month regression is the primary economic candidate. The 24-month
ranker is the best economic bridge to Legacy. The 36-month results are not
promoted because only four annual test blocks are available.

Fold dispersion is material: regression-24 monthly IC is
`0.0959 +/- 0.1101`, and top-10 one-month excess is
`1.15% +/- 1.06%` (mean +/- fold standard deviation).

SHAP summary:

- regression-24: EPS growth, long volatility, earnings yield, volatility
  ratios, SPY volatility and medium/long momentum;
- ranking-24: Bollinger bandwidth, short/medium volatility, earnings yield and
  long relative EMA z-scores;
- Legacy teacher: cross-sectional relative EMA ranks and z-scores dominate.

Decision:

- run three-trial purged CPCV only on classification/regression/ranking at 24
  months, using the teacher as a diagnostic control;
- preserve 36 months as exploratory;
- use directional SHAP samples to check whether high feature values raise or
  lower each shortlisted score.

## 2026-07-25 — shortlisted purged CPCV

Run:

`outputs/multihorizon_boosting/shortlist_cpcv_20260725`

Configuration: three parameter candidates, four inner chronological groups,
one purged test group at a time, and the three most recent outer annual blocks.
Every input file and both Legacy outputs are SHA-256 recorded in the run
manifest.

| method | IC | top-10 horizon excess | top-10 one-month excess | Legacy overlap |
|---|---:|---:|---:|---:|
| regression-24 | 0.0158 | 8.78% | 2.12% | 14.54% |
| ranking-24 | -0.0334 | 0.58% | 0.67% | 20.02% |
| classification-24 | -0.0016 | -3.87% | 1.13% | 14.74% |
| teacher-1 | 0.0066 | 1.53% | 1.53% | 41.53% |

On the exact same last three fixed-parameter folds, regression-24 had IC
0.0164, horizon excess 5.63%, one-month excess 0.84%, and overlap 12.32%.
CPCV therefore improved its top-10 economics without improving IC. Ranking
tuning degraded IC; fixed parameters won all three teacher searches.

Directional SHAP files are present in every shortlisted directory. The
regressor penalizes high 24-month stock volatility and high 24-month SPY
volatility, while favoring higher earnings yield and 36-month momentum in the
sample. These are model-behavior summaries, not causal statements.

Decision:

- retain regression-24 as the primary economic R&D candidate;
- keep ranking-24 as the strongest Legacy-overlap diagnostic;
- do not promote classification-24 or any 36-month result;
- next modeling step is a return/risk multi-head built on the regression-24
  score, using the already generated future volatility and downside targets.

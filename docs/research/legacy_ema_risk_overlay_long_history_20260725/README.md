# Exact Legacy EMA risk heads — long-history result

Date: 2026-07-25

Status: `risk_models_useful_overlay_not_validated`

Implementation commit: `ecb9e33`

> **Superseded on 2026-07-26.** The `20260719_194418` data package fails the
> official replay-lineage validator, and this document used the monthly
> mean/std Sharpe instead of the canonical Legacy report convention. Keep this
> file as the v1 audit trail, but use
> [`../legacy_ema_risk_overlay_long_history_clean_v2_20260726/README.md`](../legacy_ema_risk_overlay_long_history_clean_v2_20260726/README.md)
> for all headline comparisons.

## Objective

Keep the boosting-only exact-EMA alpha ranking unchanged, estimate future risk
with separate boosting heads, and test whether those estimates improve monthly
allocation on the longest defensible point-in-time history.

The risk score is never blended into the alpha score.

## Why the test starts in July 2011

- Daily price data starts in January 2005.
- The first winning Legacy EMA pair is observable in February 2010.
- Selecting that pair before it was known would leak future Legacy model
  selection into the past.
- With 62 train months, 6 validation months and a conservative 6-month purge,
  the first valid test month is July 2011.

The resulting OOS period is July 2011 through October 2025: 172 months and 15
outer folds. The final incomplete test block is retained because its model is
fixed before all months in that block.

## Data

- Frozen input snapshot:
  `outputs/2026-07-19/runs/20260719_194418/input_snapshot`
- Detailed Legacy schedule:
  `outputs/2026-07-19/runs/20260719_194418/legacy_detailed_returns_polars.parquet`
- Legacy monthly returns:
  `outputs/2026-07-19/runs/20260719_194418/legacy_monthly_returns_polars.parquet`
- Alpha predictions:
  `outputs/multihorizon_boosting/legacy_ema_long_history_v1_20260725`
- Risk and allocation outputs:
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725`

Hashes and exact paths are recorded in the output `manifest.json`.
The final rerun is attached to commit `ecb9e33`; hashes of the model metrics,
allocation performance and acceptance-gate files were identical across two
complete runs.

## Targets

For horizons 1, 3 and 6 months:

1. realized volatility regression:
   annualized sample standard deviation of strictly future daily returns;
2. daily downside regression:
   annualized square root of mean squared negative future daily returns;
3. high-volatility classification:
   probability of belonging to the cross-sectional top 20% of future realized
   volatility.

Every future target month requires at least 10 valid daily observations. A
daily return crossing a gap longer than seven calendar days is excluded.

All heads use:

- only exact Legacy EMA pairs observable at the fold train cutoff;
- fold-local preprocessing;
- early stopping on the historical validation block;
- the same conservative 6-month purge as the alpha head;
- fixed XGBoost parameters through local `mlcraft`;
- no Optuna search.

## Commands

Long-history alpha:

```bash
./.venv/bin/python scripts/experiments/operations/run_multihorizon_boosting.py \
  --data-dir outputs/2026-07-19/runs/20260719_194418/input_snapshot \
  --legacy-detailed outputs/2026-07-19/runs/20260719_194418/legacy_detailed_returns_polars.parquet \
  --legacy-monthly outputs/2026-07-19/runs/20260719_194418/legacy_monthly_returns_polars.parquet \
  --run-dir outputs/multihorizon_boosting/legacy_ema_long_history_v1_20260725 \
  --horizons 6 --methods classification --start-month 2005-01 \
  --min-train-months 62 --validation-months 6 \
  --test-months 12 --step-months 12 --include-partial-test-window \
  --n-trials 0 --num-boost-round 100 --shap-sample-per-fold 80 \
  --feature-mode legacy_winners_pit_ema_only
```

Risk heads and allocation:

```bash
./.venv/bin/python scripts/experiments/legacy/run_legacy_ema_risk_heads.py \
  --spec configs/research/legacy_ema_risk_overlay_long_history_v1.json \
  --output-dir outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725 \
  --bootstrap-samples 2000 --shap-sample-per-fold 30
```

HTML papers:

```bash
./.venv/bin/python scripts/experiments/legacy/render_legacy_ema_risk_papers.py \
  --spec configs/research/legacy_ema_risk_overlay_long_history_v1.json \
  --output-dir outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725
```

## Alpha result on the longer period

Top 5 equal weight, 10 bps times turnover:

- net total return: `+3693.9%`;
- CAGR: `28.88%`;
- Sharpe: `0.778`;
- maximum drawdown: `-35.43%`.

Comparators on the same 172 months:

- S&P 500: CAGR `14.34%`, Sharpe `1.016`, drawdown `-23.93%`;
- Legacy: CAGR `17.07%`, Sharpe `0.890`, drawdown `-23.38%`.

This longer test is less favorable than the selected 2013-2025 result. Alpha
retains a strong return advantage, but not a risk-adjusted advantage.

## Risk model metrics

| Target | Horizon | Monthly Spearman | R2 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| realized volatility | 1m | 0.393 | 0.150 | — | — |
| realized volatility | 3m | 0.440 | 0.152 | — | — |
| realized volatility | 6m | 0.441 | 0.113 | — | — |
| daily downside | 1m | 0.330 | 0.079 | — | — |
| daily downside | 3m | 0.385 | 0.085 | — | — |
| daily downside | 6m | 0.398 | 0.061 | — | — |
| high volatility | 1m | 0.407 | — | 0.747 | 0.499 |
| high volatility | 3m | 0.457 | — | 0.783 | 0.562 |
| high volatility | 6m | 0.467 | — | 0.784 | 0.569 |

The 3-month realized-volatility regression has positive fold R2 in 13 of 15
folds. The 3-month high-volatility classifier has ROC-AUC from 0.761 to 0.820
across all 15 folds.

Conclusion: the exact EMA features contain a stable and economically
interpretable risk signal.

## Allocation tests

The 3-month horizon was fixed as primary before allocation outcomes were read.
Horizons 1 and 6 are sensitivity diagnostics.

| Strategy | CAGR | Sharpe | Volatility | Max DD |
|---|---:|---:|---:|---:|
| alpha top 5 equal | 28.88% | 0.778 | 42.89% | -35.43% |
| inverse vol 1m | 27.08% | 0.789 | 39.07% | -33.05% |
| inverse vol 3m, primary | 26.96% | 0.781 | 39.50% | -31.80% |
| inverse vol 6m | 27.29% | 0.780 | 40.12% | -32.32% |
| inverse downside 6m | 27.66% | 0.779 | 40.81% | -32.74% |
| inverse vol 3m + max 2 names/sector + sector cap | 24.35% | 0.719 | 40.09% | -35.62% |

Primary inverse-vol 3m versus equal weight:

- Sharpe difference: `+0.003`;
- maximum drawdown improvement: `+3.63 percentage points`;
- CAGR difference: `-1.91 percentage points`;
- paired block bootstrap Sharpe difference 95% CI:
  `[-0.051, +0.073]`;
- probability that annualized mean-return difference is non-positive:
  `88.45%`.

The sector-constrained version respects the 40% sector cap but loses 4.53
percentage points of CAGR and worsens the maximum drawdown.

## Acceptance decision

Pre-registered gates:

- Sharpe above equal weight;
- drawdown improves by at least 5 percentage points;
- CAGR loss no larger than 3 percentage points;
- maximum sector weight no larger than 40%;
- Sharpe advantage remains at 50 bps times turnover.

No tested primary allocation passes every gate.

Decision:

- keep the risk heads for probability, volatility, downside and explanation;
- do not activate inverse-vol sizing;
- do not activate the sector rule in its current form;
- keep alpha top 5 equal weight as the research allocation baseline;
- any high-volatility veto or softer risk tilt must receive a new
  pre-registration and cannot be selected on these same 172 months as if it
  were confirmatory.

## SHAP

Main 3-month realized-volatility features:

- `relative_ema_ratio_27_106`;
- `relative_ema_ratio_100_326`;
- `relative_ema_ratio_7_333`;
- `relative_ema_ratio_95_72`;
- `relative_ema_ratio_12_150`.

The high-volatility probability relies more heavily on cross-sectional
standardizations, especially:

- `100/326 z_month`;
- `100/326 raw`;
- `95/72 z_month`;
- `75/364 raw`;
- `27/106 z_month`.

SHAP describes the fitted model. It is not evidence of a causal effect.

## HTML papers

- Index:
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725/html/index.html`
- Results:
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725/html/risk_results_paper.html`
- Methodology:
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725/html/methodology_paper.html`

The results paper was rendered in Chrome at 1440 px and 480 px widths and
visually checked.

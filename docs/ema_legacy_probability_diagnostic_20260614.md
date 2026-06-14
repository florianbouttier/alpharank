# Diagnostic EMA vs XGBoost vs legacy - 2026-06-14

## Verdict

Oui, l'idee EMA-only etait la bonne experience a faire.

Sur les 11 mois recents comparables, le modele XGBoost limite aux features EMA bat le full model precedent. Encore plus important : le correcteur residual avec `base_margin` marche tres bien quand le modele de base est EMA-only, alors qu'il echouait sur les features SHAP/volatilite.

Conclusion : le probleme n'est pas "XGBoost est nul". Le probleme est que le full model apprend un signal large, assez bon en AUC, mais moins bon pour le top-N momentum/EMA qui drive l'allocation.

## Runs

- EMA-only residual : `outputs/residual_init_score_experiment_20260613_112301`
- SHAP top 12 residual : `outputs/residual_init_score_experiment_20260613_030717`
- SHAP top 20 residual : `outputs/residual_init_score_experiment_20260613_030809`
- Full model walk-forward : `outputs/xgboost_timefold_backtest_20260612_175250`
- Diagnostic legacy/probas : `outputs/legacy_probability_diagnostics_20260614`

## Features EMA testees

16 variables :

- `ema_ratio_2_6`
- `ema_ratio_3_6`
- `ema_ratio_3_12`
- `ema_ratio_6_12`
- `ema_ratio_6_18`
- `ema_ratio_12_24`
- `ema_ratio_18_24`
- `ema_ratio_6_24`
- `ema_ratio_3_24`
- `ema_ratio_3_18`
- `ema_ratio_2_12`
- `ema_ratio_2_3`
- `price_to_ema_3`
- `price_to_ema_6`
- `price_to_ema_12`
- `price_to_ema_24`

## Comparaison des experiences

Fenetre : 11 mois, holdings `2025-06` a `2026-04`.

| modele | features | total return | actif vs SPY | AUC test moy. | hit-rate top 10 | pire mois actif |
|---|---:|---:|---:|---:|---:|---:|
| full model, proba brute | 63 | +26.8% | +4.9% | n/a ici | 37.3% | -10.3% |
| full model, risk-adjusted | 63 | +18.9% | -2.5% | 0.614 | 33.6% | -5.1% |
| SHAP base | 12 | +8.5% | -11.0% | 0.619 | 33.6% | -7.4% |
| SHAP residual | 12 | +3.3% | -15.4% | 0.619 | 32.7% | -7.1% |
| SHAP base | 20 | +8.1% | -11.6% | 0.617 | 32.7% | -8.3% |
| SHAP residual | 20 | +7.9% | -11.5% | 0.617 | 31.8% | -6.4% |
| EMA base | 16 | +37.9% | +12.5% | 0.560 | 30.0% | -10.9% |
| EMA residual | 16 | +52.2% | +24.8% | 0.562 | 35.5% | -4.7% |

Point cle : EMA residual a une AUC mediocre, mais c'est le meilleur panier ML teste jusqu'ici. L'AUC globale ne mesure pas correctement ce qui nous interesse : les 10 meilleurs noms du mois.

## Legacy sur la meme fenetre

Depuis `outputs/2026-06-07/legacy_monthly_returns_polars.parquet`.

| legacy model | total return | benchmark | actif compose | mois actifs gagnants | pire mois actif | meilleur mois actif |
|---|---:|---:|---:|---:|---:|---:|
| Combined_Equal | +122.6% | +21.9% | +86.4% | 8/11 | -6.2% | +24.3% |
| Combined_Frequency | +125.7% | +21.9% | +89.5% | 8/11 | -5.4% | +28.2% |

Le legacy reste largement au-dessus. Donc EMA-only ML est une amelioration nette, mais il ne replique pas encore le legacy.

## Probabilites du full model sur les actions legacy

J'ai joint les actions legacy avec les predictions du full model walk-forward.

Coverage : 142 lignes legacy, 142 matchees.

Les paniers `Combined_Equal` et `Combined_Frequency` sont identiques en tickers sur cette fenetre.

| metrique | valeur |
|---|---:|
| proba full moyenne sur actions legacy | 0.337 |
| proba full mediane sur actions legacy | 0.342 |
| proba full globale moyenne | 0.215 |
| proba full globale mediane | 0.200 |
| proba full globale p90 | 0.310 |
| rang full moyen des actions legacy | 41.4 |
| rang full median des actions legacy | 25 |
| overlap actions legacy avec top 10 full model | 14.1% |
| actions legacy avec proba full > 50% | 0% |
| hit-rate legacy sur target +5% | 50.7% |
| exces futur moyen des actions legacy | +6.0% |
| overlap actions legacy avec top 10 EMA residual | 26.8% |
| rang EMA residual moyen des actions legacy | 25.8 |
| rang EMA residual median des actions legacy | 19 |

Interpretation : le full model ne deteste pas les actions legacy. Au contraire, il leur donne des probas au-dessus du p90 global. Mais il ne les pousse pas assez souvent dans le top 10. Les probas absolues ne depassent jamais 50%, donc un seuil fixe `p > 0.5` est inutile ici. La bonne notion de threshold est le cutoff mensuel du top N, pas un seuil absolu.

## Pourquoi une regle EMA peut battre le full XGBoost

Hypothese la plus probable : le legacy optimise implicitement un signal de ranking momentum/EMA, alors que notre full XGBoost optimise une classification binaire `future_excess_return > 5%`.

Ce decouplage cree trois effets :

1. Le full model apprend mieux l'univers global, donc meilleure AUC.
2. Les features volatilite/fondamentaux ajoutent du signal moyen mais peuvent casser la convexite du top 10.
3. Le legacy et EMA-only favorisent davantage les noms a fort momentum, ce qui paie fortement sur la fenetre recente.

Le modele full est "raisonnable". Le legacy est plus agressif et plus proche du payoff reel de l'allocation.

## Decision

Je ne pousserais pas le residual SHAP.

Je pousserais la voie EMA-only :

- garder EMA residual comme nouveau benchmark ML ;
- comparer chaque future experience a EMA residual, pas seulement au full model ;
- arreter d'utiliser AUC comme metrique principale de decision ;
- optimiser top-N active return / ranking mensuel.

## Prochaine piste

La prochaine experience qui a le plus de chances d'expliquer le legacy :

1. Distillation legacy : apprendre `legacy_selected` ou `legacy_n_models` comme target auxiliaire.
2. Puis combiner :
   - score EMA residual ;
   - probabilite `future_excess > 5%` ;
   - probabilite d'etre selectionne par legacy.
3. Selection finale via ranking mensuel, pas threshold absolu.

Autre piste forte : remplacer la classification par regression/ranking du `future_excess_return` continu, parce que le legacy semble capter la magnitude des gagnants, pas seulement la probabilite de depasser +5%.

# Rapport Boosting CPCV - Seuil relatif 5 %

Date du run : 2026-06-11  
Run dir : `outputs/xgboost_timefold_backtest_20260611_013248`  
Code de référence : commits `85a2b12`, `e69537b`, `903207b`, `02ba3bf`

## Résumé

Le seuil de classification est maintenant défini à `5 %` d'outperformance mensuelle relative à SPY/S&P 500.

La target apprise est :

```text
future_relative_return = (1 + future_return_action) / (1 + benchmark_future_return)
target_label = 1 si future_relative_return - 1 > 0.05
```

Autrement dit, le modèle apprend la probabilité que l'action fasse mieux que l'indice de plus de 5 % le mois suivant. Ce seuil est raisonnable sur les données actuelles : le taux positif global est `22.3 %`. Un seuil à 15 % aurait été très rare, puisque le percentile 95 de `future_excess_return` est seulement `13.3 %`.

Le run exploratoire donne :

- `21` fenêtres CPCV complétées
- `147 844` lignes dans le frame modèle
- `147 226` prédictions agrégées action/mois
- `315` mois backtestés
- AUC test moyenne : `0.606`
- hit rate top 10 : `35.6 %` contre un taux positif de base de `22.3 %`
- performance active moyenne : `+2.51 %` par mois

Verdict : il y a un signal exploitable dans le ranking, mais le run n'est pas encore une validation production. La stabilité dépend des blocs temporels et les résultats portefeuille sont portés par des mois très extrêmes au début de l'historique.

## Données

Sources chargées via le manifest du run :

- manifest du run : `outputs/xgboost_timefold_backtest_20260611_013248/data_input_manifest.json`
- snapshot d'entrée : `20260611_013249`
- snapshot source : `20260426_154730`
- statut de matching : `full_match`

Périmètre modèle :

- période : `2000-01-01` à `2026-04-01`
- mois distincts : `316`
- tickers distincts : `793`
- lignes modèle : `147 844`
- features retenues : `63`
- features supprimées pour sparsité : `41`
- exclusions qualité appliquées : `SII.US`, `CBE.US`, `TIE.US`, `CPWR.US`

Distribution de la variable future relative :

| Stat | Valeur |
| --- | ---: |
| moyenne `future_excess_return` | `0.75 %` |
| médiane | `0.10 %` |
| p05 | `-12.15 %` |
| p95 | `13.34 %` |
| taux positif au seuil 5 % | `22.26 %` |

Le seuil 5 % garde donc une classe positive assez sélective sans rendre la classification trop déséquilibrée.

## Entraînement

Le training boosting passe par `mlcraft`, avec XGBoost comme backend.

Configuration du run :

- `fold_strategy = "cpcv"`
- groupes CPCV externes : `7`
- groupes test par combinaison : `2`
- fenêtres externes obtenues : `21`
- CPCV interne pour tuning : `5` groupes
- seuil target : `0.05`
- `top_n = 10`
- `n_optuna_trials = 5` pour ce run exploratoire
- SHAP désactivé sur ce run (`shap_sample_size=0`) pour limiter le temps

Métriques folds :

| Métrique | Valeur |
| --- | ---: |
| AUC train moyenne | `0.655` |
| AUC validation moyenne | `0.607` |
| AUC test moyenne | `0.606` |
| score pénalisé moyen | `0.351` |
| score pénalisé min | `0.021` |
| score pénalisé max | `0.635` |
| part folds score > 0 | `100 %` |
| taux positif train moyen | `21.9 %` |
| taux positif validation moyen | `24.3 %` |
| taux positif test moyen | `22.4 %` |

Meilleurs folds par score pénalisé :

| Fold | Score | AUC train | AUC val | AUC test | Taux positif test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | `0.635` | `0.642` | `0.638` | `0.641` | `25.5 %` |
| 1 | `0.615` | `0.637` | `0.595` | `0.633` | `27.6 %` |
| 9 | `0.535` | `0.643` | `0.567` | `0.625` | `18.8 %` |
| 4 | `0.518` | `0.620` | `0.637` | `0.646` | `25.5 %` |
| 13 | `0.444` | `0.648` | `0.635` | `0.614` | `20.9 %` |

Folds faibles :

| Fold | Score | AUC train | AUC val | AUC test | Commentaire |
| ---: | ---: | ---: | ---: | ---: | --- |
| 7 | `0.021` | `0.733` | `0.560` | `0.615` | overfit fort, validation atypique |
| 21 | `0.056` | `0.674` | `0.610` | `0.571` | période test récente plus faible |
| 15 | `0.109` | `0.679` | `0.631` | `0.584` | gap train/test notable |
| 17 | `0.225` | `0.677` | `0.594` | `0.602` | score modéré |
| 14 | `0.287` | `0.650` | `0.637` | `0.589` | test plus faible que validation |

Lecture : le modèle classe mieux que le hasard dans tous les folds test, mais le gap train/test varie. Le fold 7 montre un cas d'overfit marqué malgré une AUC test correcte.

## Hyperparamètres observés

Les trials gagnants convergent vers des modèles conservateurs :

- `max_depth` moyen : `4.1`
- `learning_rate` moyen : `0.016`
- `num_boost_round` moyen : `475`
- `subsample` moyen : `0.753`
- `colsample_bytree` moyen : `0.764`

Observation qualitative pendant le run : les configurations profondes (`max_depth` 8-10) et/ou learning rate élevé sont presque toujours fortement pénalisées par l'objectif `AUC_val - lambda * gap`.

Recommandation : pour un run plus sérieux, resserrer l'espace de recherche autour de :

- `max_depth = 2..5`
- `learning_rate = 0.005..0.05`
- `num_boost_round = 150..900`
- conserver une régularisation significative

## Backtest Top 10

Le backtest prend les probabilités agrégées par action/mois, classe les actions par `prediction`, puis conserve le top 10 mensuel.

Résultats globaux :

| Série | Total return | CAGR | Vol annualisée | Sharpe | Max drawdown | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Portefeuille | `1339.3x` | `31.6 %` | `51.7 %` | `0.57` | `-70.8 %` | `59.0 %` |
| Benchmark | `7.1x` | `8.3 %` | `15.2 %` | `0.42` | `-50.8 %` | `63.5 %` |
| Active | `295.9x` | `24.2 %` | `45.0 %` | `0.54` | `-61.5 %` | `54.9 %` |

Statistiques mensuelles :

- rendement portefeuille moyen : `3.28 %`
- rendement benchmark moyen : `0.76 %`
- rendement actif moyen : `2.51 %`
- meilleur mois actif : `+100.4 %`
- pire mois actif : `-25.3 %`
- hit rate moyen du top 10 : `35.6 %`
- prédiction moyenne univers : `22.0 %`
- prédiction moyenne sélection top 10 : `39.3 %`

Le lift de hit rate est important :

```text
hit_rate_top10 / base_positive_rate = 35.6 % / 22.3 % = 1.6x
```

Cela valide mieux le ranking que la performance brute, car la performance brute est dominée par quelques mois extrêmes.

## Années Récentes

La stratégie ne surperforme pas chaque année. Sur les années récentes :

| Année | Total portefeuille | Total benchmark | Active total | Active win rate |
| ---: | ---: | ---: | ---: | ---: |
| 2019 | `54.9 %` | `31.2 %` | `20.2 %` | `66.7 %` |
| 2020 | `19.9 %` | `18.3 %` | `11.4 %` | `41.7 %` |
| 2021 | `16.7 %` | `28.7 %` | `-9.6 %` | `33.3 %` |
| 2022 | `-29.2 %` | `-18.2 %` | `-11.5 %` | `41.7 %` |
| 2023 | `-5.7 %` | `26.2 %` | `-23.5 %` | `41.7 %` |
| 2024 | `-2.7 %` | `24.9 %` | `-22.4 %` | `41.7 %` |
| 2025 | `34.4 %` | `17.7 %` | `13.9 %` | `58.3 %` |
| 2026 YTD | `2.3 %` | `5.0 %` | `-1.8 %` | `50.0 %` |

Lecture : le backtest long terme paraît spectaculaire, mais la période 2021-2024 est défavorable. Il faut donc éviter de conclure à une robustesse production sans analyse par régime.

## Tickers Les Plus Sélectionnés

Top tickers historiques par nombre de sélections :

| Ticker | Sélections |
| --- | ---: |
| `AMD.US` | `83` |
| `BBBY.US` | `78` |
| `NVDA.US` | `62` |
| `NFLX.US` | `59` |
| `EP.US` | `56` |
| `BKNG.US` | `43` |
| `FSLR.US` | `42` |
| `THC.US` | `40` |
| `ANDV.US` | `38` |
| `AKAM.US` | `37` |

Attention : plusieurs tickers sont historiques, delistés ou non investissables aujourd'hui. C'est normal pour un backtest historique S&P 500, mais il ne faut pas utiliser cette liste directement comme portefeuille courant.

## Limites Importantes

Ce run est exploratoire, pas une validation finale.

Limites méthodologiques :

- `n_optuna_trials=5` seulement, pour obtenir un résultat rapide ; le preset reste prévu pour plus d'essais.
- SHAP désactivé, donc pas d'analyse d'importance de variables dans ce rapport.
- CPCV évalue la robustesse hors bloc, mais ce n'est pas un backtest strictement live walk-forward : certains modèles CPCV peuvent apprendre avec des mois postérieurs au bloc testé.
- Les métriques portefeuille par fold ne sont pas interprétables après agrégation CPCV des prédictions ; la conclusion backtest doit se lire sur le portefeuille global agrégé.
- La performance globale est très influencée par des mois extrêmes de 2001-2009.
- Pas de coûts de transaction, slippage, liquidité, capacité, turnover, ni contraintes investissables actuelles.

Limites data :

- le run utilise le snapshot local disponible, pas une reconstruction data fraîche
- les tickers historiques peuvent inclure des noms delistés
- la target dépend de la qualité des prix mensuels action et indice

## Conclusion

Le seuil 5 % est meilleur comme première cible que 15 % : il correspond à une classe positive sélective mais encore assez fréquente (`22.3 %`).

Le modèle boosting `mlcraft` apprend un signal de ranking mesurable :

- AUC test moyenne au-dessus de `0.60`
- lift top 10 d'environ `1.6x` sur la probabilité de battre SPY de plus de 5 %
- backtest top 10 globalement très au-dessus du benchmark

Mais la robustesse production n'est pas encore démontrée :

- performance récente 2021-2024 faible
- CPCV non équivalent à une simulation live mensuelle
- optimisation trop courte
- forte sensibilité aux régimes et aux mois extrêmes

Prochaine étape recommandée : lancer un run plus long avec espace hyperparamètres resserré, puis comparer trois vues séparées :

1. CPCV pur pour robustesse modèle.
2. Walk-forward strict pour performance investissable.
3. Application mensuelle actuelle avec contraintes de tickers vivants, liquidité et turnover.

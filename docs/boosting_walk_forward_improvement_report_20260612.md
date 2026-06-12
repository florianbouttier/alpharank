# Rapport boosting walk-forward - 2026-06-12

## Synthese

J'ai ajoute une voie de training/backtest plus proche de l'usage mensuel reel :

- target binaire : `((1 + rendement_action) / (1 + rendement_SPY) - 1) > 5%` sur le mois suivant ;
- entrainement via `mlcraft` depuis la racine du repo ;
- validation interne en CPCV pour Optuna ;
- evaluation externe en walk-forward strict : train passe, validation 12 mois, test 1 mois ;
- objectif Optuna aligne avec le backtest : precision dans le top N, pas seulement AUC ;
- warm-start Optuna depuis les meilleurs hyperparametres du run CPCV precedent ;
- score de selection optionnel `prediction - risk_penalty * volatility_12m`.

Verdict court : il y a du signal de ranking, mais pas encore assez stable pour en faire un bon algo d'allocation autonome. Sur les 11 mois recents exploitables, le portefeuille risk-adjusted fait +18.9%, mais SPY fait +22.5%. L'actif est donc negatif a -2.5%. La version top 10 en probabilite brute aurait fait +26.8%, soit +4.9% actif compose, mais avec un pire mois actif plus violent.

## Commits

- `52da0ad add walk-forward warm-start top-n tuning`

Ce commit ajoute :

- `walk_forward_windows(...)` dans `src/alpharank/backtest/time_folds.py` ;
- support `fold_strategy="walk_forward"` ;
- warm-start Optuna par `study.enqueue_trial(...)` ;
- objectif `top_n_precision` ;
- selection top N sur colonne configurable ;
- score risk-adjusted ;
- metadata du run enrichie ;
- tests cibles sur folds temporels et ranking portfolio.

## Donnees et features

Run analyse :

- dossier : `outputs/xgboost_timefold_backtest_20260612_175250`
- source de warm-start : `outputs/xgboost_timefold_backtest_20260611_013248`
- features utilisees : 63
- target : outperformance relative action/SPY superieure a 5%
- top N backtest : 10 actions par mois
- periode test exploitable : holding months `2025-06` a `2026-04`

Le modele utilise bien des variables techniques et financieres :

- techniques : momentum, ratios EMA, RSI, Bollinger, distances aux plus hauts/bas, volatilites, ratios de volatilite ;
- financieres : marge nette TTM, ROE, ROA, asset turnover, equity/assets, croissance revenue et net income.

SHAP exploratoire du run precedent :

| rang | feature | famille |
|---:|---|---|
| 1 | `volatility_12m` | technique |
| 2 | `volatility_24m` | technique |
| 3 | `volatility_36m` | technique |
| 4 | `asset_turnover_ttm` | financier |
| 5 | `volatility_48m` | technique |
| 6 | `total_revenue_ttm_growth_4q` | financier |
| 7 | `dist_to_21m_high` | technique |
| 8 | `bollinger_percent_b_6m` | technique |
| 9 | `dist_to_12m_high` | technique |
| 10 | `net_margin_ttm` | financier |

Lecture : le signal dominant vient des variables techniques de volatilite/range/momentum. Les fondamentaux existent et apportent du signal, mais moins que la structure de prix recente dans cette experience.

## Parametrage du run

Commande executee :

```bash
MPLCONFIGDIR=/private/tmp/mpl-alpharank ./.venv/bin/python -u -c "from pathlib import Path; from scripts.run_backtest import default_config, run; cfg=default_config(fold_strategy='walk_forward', max_walk_forward_windows=12, n_optuna_trials=3, warm_start_params_path=Path('outputs/xgboost_timefold_backtest_20260611_013248'), warm_start_top_k=3, selection_score='risk_adjusted', risk_penalty=0.25, save_optuna_all_plots=False, shap_sample_size=0, show_optuna_progress=False, verbose=True); artifacts=run(cfg)"
```

Un premier essai `24 folds x 10 trials` a ete interrompu volontairement : le premier fold a pris 4m40 et l'ETA etait environ 1h47. Le run final complet est donc un run diagnostic borne : `12 folds x 3 trials`.

Comme `n_optuna_trials=3` et `warm_start_top_k=3`, le run sert surtout a valider les meilleurs hyperparametres connus en walk-forward strict. Il ne teste pas encore une vraie exploration Optuna large.

## Resultats principaux

KPI du run walk-forward risk-adjusted :

| strategie | total return | CAGR | Sharpe | max drawdown | win rate | hit-rate top 10 |
|---|---:|---:|---:|---:|---:|---:|
| Portfolio | +18.9% | +20.8% | 0.82 | -14.4% | 45.5% | 33.6% |
| SPY | +22.5% | +24.8% | 1.78 | -5.8% | 81.8% | n/a |
| Actif | -2.5% | -2.8% | -0.18 | -11.1% | 36.4% | n/a |

Moyennes folds :

- AUC test moyenne : 0.614
- AUC test mediane : 0.619
- AUC test min/max : 0.347 / 0.769
- hit-rate top 10 moyen : 33.6%
- actif mensuel moyen : -0.14%

Le modele classe donc mieux que le hasard en moyenne, mais la dispersion mensuelle est trop forte.

## Mois qui expliquent le resultat

| holding month | portfolio | SPY | actif | hit-rate |
|---|---:|---:|---:|---:|
| 2025-06 | +4.9% | +5.1% | -0.2% | 20% |
| 2025-07 | -0.3% | +2.3% | -2.6% | 30% |
| 2025-08 | +9.2% | +2.1% | +7.1% | 70% |
| 2025-09 | +4.0% | +3.6% | +0.4% | 30% |
| 2025-10 | -1.5% | +2.4% | -3.9% | 30% |
| 2025-11 | -2.8% | +0.2% | -3.0% | 40% |
| 2025-12 | +6.6% | +0.1% | +6.5% | 60% |
| 2026-01 | -3.6% | +1.5% | -5.0% | 10% |
| 2026-02 | -6.0% | -0.9% | -5.1% | 30% |
| 2026-03 | -5.6% | -4.9% | -0.6% | 10% |
| 2026-04 | +14.6% | +9.8% | +4.8% | 40% |

Les meilleurs mois existent, mais janvier/fevrier 2026 detruisent beaucoup de l'actif. Le probleme principal est la stabilite de regime.

## Risk-adjusted vs probabilite brute

J'ai compare le panier backteste (`selection_score = prediction - 0.25 * volatility_12m`) avec un panier hypothetique top 10 par probabilite brute sur les memes predictions.

| selection | total return | actif compose | mois actifs gagnants | hit-rate | pire mois actif | meilleur mois actif |
|---|---:|---:|---:|---:|---:|---:|
| risk-adjusted | +18.9% | -2.5% | 4/11 | 33.6% | -5.1% | +7.1% |
| probabilite brute | +26.8% | +4.9% | 5/11 | 37.3% | -10.3% | +9.4% |

Conclusion : la penalisation simple par volatilite n'est pas validee. Elle reduit certains accidents, mais elle retire aussi trop de convexite positive. A ce stade, je ne la mettrais pas en production par defaut ; je garderais `selection_score="prediction"` et je testerais le risque dans une couche separee de sizing/constraints.

## Warm-start et Optuna

Les 11 folds ont tous choisi le meme jeu d'hyperparametres warm-start :

- `max_depth=4`
- `learning_rate=0.007842`
- `subsample=0.6636`
- `colsample_bytree=0.9296`
- `min_child_weight=2.4818`
- `gamma=3.2470`
- `num_boost_round=607`
- `alpha=2.935`
- `lambda=0.174`

Interpretation : avec seulement 3 trials et 3 warm-starts, le run valide surtout que le meilleur candidat precedent reste competitif dans cette configuration. Il ne prouve pas qu'il est optimal.

Tester plus de trials peut ameliorer la perf, mais je ne m'attends pas a ce que ce soit le levier principal. Le cout observe est eleve : environ 1m45 par fold pour 3 trials, et environ 4m40 pour 10 trials sur un fold recent. Un run `24 folds x 20 trials` serait plutot un run long/offline.

## Est-ce assez bon pour allocation ?

Pas encore.

Arguments positifs :

- AUC test moyenne 0.614 en walk-forward strict ;
- certaines periodes capturent bien les gagnants relatifs ;
- les features SHAP sont coherentes economiquement : volatilite, range, momentum, asset turnover, marge, croissance revenue.

Arguments negatifs :

- actif negatif sur le run walk-forward risk-adjusted ;
- mois de regime tres faibles : AUC 0.347 en janvier 2026 ;
- hit-rate top 10 moyen seulement 33.6% pour une target dont la base varie fortement ;
- la penalisation volatilite naive degrade le rendement ;
- le run large Optuna n'a pas encore ete execute a cause du cout.

Je le considererais comme un moteur de ranking R&D utilisable pour produire des candidats, pas comme une allocation finale autonome.

## Pistes prioritaires

1. Revenir a `selection_score="prediction"` pour le backtest de reference, puis traiter le risque dans le sizing.
2. Ajouter une calibration mensuelle des probabilites : isotonic ou Platt par walk-forward, puis deciles de probabilite calibree.
3. Optimiser directement une metrique portefeuille : top-N active return moyen, hit-rate conditionnel et drawdown actif, pas uniquement top-N precision.
4. Tester plusieurs seuils de target : 0%, 2.5%, 5%, 7.5%. Le seuil 5% est clair, mais il peut etre trop strict et instable selon le regime.
5. Ajouter des features de regime marche : trend SPY, volatilite SPY, breadth, dispersion cross-sectionnelle, taux si disponible.
6. Ajouter une couche de contraintes portfolio : max exposition secteur/facteur, cap volatilite par ligne, turnover max, exclusion des tickers trop selectionnes mais destructeurs.
7. Faire une vraie recherche Optuna offline : par exemple `24 folds x 20 trials`, puis warm-start top 10 et rerun `selection_score="prediction"`.
8. Refaire SHAP sur le run walk-forward final quand le parametrage sera stabilise. Le SHAP actuel est utile, mais il vient du run CPCV precedent.

## Prochaine commande recommandee

Pour le prochain run serieux, je recommande de ne pas penaliser la selection et d'augmenter Optuna en offline :

```bash
MPLCONFIGDIR=/private/tmp/mpl-alpharank ./.venv/bin/python -u -c "from pathlib import Path; from scripts.run_backtest import default_config, run; cfg=default_config(fold_strategy='walk_forward', max_walk_forward_windows=24, n_optuna_trials=20, warm_start_params_path=Path('outputs/xgboost_timefold_backtest_20260611_013248'), warm_start_top_k=5, selection_score='prediction', save_optuna_all_plots=False, shap_sample_size=0, show_optuna_progress=True, verbose=True); run(cfg)"
```

Ce run risque de durer plusieurs heures selon la machine. Il donnera une reponse plus solide a la question "plus de trials Optuna ameliore-t-il vraiment la perf ?".

# Catalogue central des modeles boosting / legacy-copy

Date de creation de cette doc : 2026-06-14.

Cette page est le point central pour se rappeler quels modeles ont ete testes,
comment ils sont construits, quand ils ont ete construits, et quelles metriques
doivent servir de juge.

Clarification active du 2026-06-27 soir : il faut maintenant separer deux
questions qui ont ete melangees dans les premiers essais.

```text
1. Diagnostic Legacy : comprendre quels signaux permettent de retrouver les
   actions Legacy.
2. Allocation autonome : predire un rendement relatif futur et construire un
   portefeuille sans utiliser Legacy dans l'objectif d'entrainement ou de
   tuning.
```

Le premier axe explique pourquoi une regle EMA simple peut recomposer Legacy.
Le second axe est le vrai candidat trading : boosting seul ou boosting avec
contraintes de portefeuille, mais sans objectif Legacy.

## Pourquoi cette doc existe

Le constat principal est maintenant clair : chercher directement a battre
legacy avec un classifieur `future_excess_return > 5%` n'est pas le bon premier
objectif.

Correction importante ajoutee le 2026-06-14 soir : `distill_legacy` est un
modele teacher/oracle. Il est utile pour mesurer le plafond de verre et
comprendre ce que Legacy encode, mais il ne doit pas etre considere comme un
algo final parce qu'il apprend directement la decision Legacy.

Avant de generaliser, on veut prouver que le boosting sait reproduire les
signaux et les trades legacy. Le probleme prioritaire est donc :

```text
features decision-month -> score qui retrouve les trades legacy du mois suivant
```

La metrique centrale n'est ni le rendement, ni l'AUC, ni la precision globale.
La metrique centrale est le pourcentage de recomposition du panier Legacy :

```text
nombre d'actions communes entre modele et Legacy
/
nombre d'actions choisies par Legacy ce mois-la
```

Pour chaque mois, le modele choisit exactement le meme nombre de tickers que
Legacy. Si Legacy choisit 4 tickers, le modele en choisit 4. Si Legacy en
choisit 7, le modele en choisit 7. Si Legacy en choisit 10, le modele en choisit
10. Le score du mois est seulement la part de tickers communs.

### Definition concrete du score de recomposition

Quand cette doc dit `2 070 / 2 070 = 100%`, cela veut dire uniquement :

```text
Sur l'ensemble des mois testes, Legacy a choisi 2 070 lignes ticker/mois.
Pour chaque mois, le modele a choisi le meme nombre de tickers que Legacy.
Les tickers choisis par le modele sont tombes 2 070 fois sur un ticker choisi
par Legacy.
```

Exemple simple :

```text
Mois M:
Legacy choisit 7 actions.
Le modele choisit aussi 7 actions.
Si 5 actions sont communes entre les deux paniers, le mois vaut 5 / 7 = 71.4%.
Si les 7 actions sont communes, le mois vaut 7 / 7 = 100%.
```

Donc `100%` ne veut pas dire que le rendement futur est predit parfaitement. Ca
veut dire que les paniers mensuels du modele ont exactement les memes actions
que les paniers mensuels Legacy sur le test mesure. C'est une metrique de
recouvrement des trades, pas une metrique de performance boursiere.

### Fiche de lecture obligatoire par methode

Pour chaque methode, il faut garder quatre questions separees :

1. **Ce qui est appris** : regression, classification, ranking, ou score
   deterministe.
2. **Les variables disponibles** : features prix/EMA/fondamentales, features
   atomiques Legacy, ou features d'experts EMA generalisables.
3. **L'ensemble de test** : periode et nombre de lignes Legacy a recomposer.
4. **Ce qu'on retrouve** : nombre de tickers communs avec Legacy, pas rendement.

Table de synthese des runs longs importants :

| methode | ce qui est appris ou score | variables utilisees | ensemble de test | ce qu'on retrouve |
|---|---|---|---|---|
| `tradable_ema_regression_optuna` | futur rendement relatif : regression de `future_excess_return` clippe, tuning Optuna avec warm starts JSON | 88 variables EMA calculables au mois de decision : 16 EMA de base, rangs mensuels, z-scores mensuels, flags top/bottom quartile, agregats horizontaux | run large corrige `2021-06` a `2026-04`, 59 mois, 457 lignes Legacy | `159 / 457 = 34.8%`, mediane mensuelle 33.3%; meilleur run court recent = `68 / 169 = 40.2%` |
| `tradable_ema_regression_optuna_no_legacy_objective` | futur rendement relatif : meme regression, mais Optuna optimise seulement la validation future, jamais Legacy | memes 88 variables EMA ; pas de warm-start issu de Legacy ; objectifs testes : rendement top 10 validation et precision top K validation `K=10/20/30/50` | sweep court propre `2025-06` a `2026-04`, 11 mois, 71 lignes Legacy | recomposition tres basse (`3/71` a `9/71`) ; meilleur trading court = precision top 30, top 30 `+65.9%`, Sharpe 3.15 |
| `tradable_technical_regression_optuna` | futur rendement relatif : meme regression, mais avec toutes les familles techniques disponibles | 283 variables techniques : ROC prix, EMA, RSI, Bollinger, stochastique, distance high/low, range position, volatilite, rangs/z-scores/flags/agregats mensuels | run court `2024-06` a `2026-04`, 23 mois, 169 lignes Legacy | `55 / 169 = 32.5%`; moins bon que EMA-only sur la meme periode |
| `tradable_ema_residual_regression` | futur rendement relatif : base EMA-only puis regression du residu `future_excess_return_clippe - prediction_EMA` | base : 88 variables EMA ; residu : variables techniques non-EMA avec rangs mensuels, z-scores, flags top/bottom quartile et agregats ; score final `EMA + shrinkage * residu` | run court `2024-06` a `2026-04`, 23 mois, 169 lignes Legacy | meilleur shrinkage recomposition `0.25` : `69 / 169 = 40.8%`; mieux que le meilleur EMA court d'un ticker, mais backtest plus faible que la base EMA fixe |
| `portfolio_boosting_top_return_classifier` | allocation autonome : classification `future_excess_return` dans le top 10% du mois suivant | 283 variables techniques calculables au mois de decision : momentum/ROC, EMA, RSI, Bollinger, stochastique, distances high/low, volatilite, rangs/z-scores/flags/agregats mensuels ; aucune variable Legacy dans le modele | backtest 2015+ `2015-02` a `2026-04`, 135 mois de performance | KPI principal trading : meilleur top 7 = +364.3%, CAGR 14.6%, Sharpe 0.37, max DD -47.0%; sous Legacy et trop risque |
| `portfolio_boosting_rank_regression` | allocation autonome : regression du rang percentile mensuel futur de `future_excess_return` | memes variables techniques calculables que ci-dessus ; target = rang relatif futur dans le mois, pas le rendement brut ; aucune variable Legacy dans le modele | backtest 2015+ `2015-02` a `2026-04`, 135 mois de performance | meilleur baseline top 20 = +348.3%, CAGR 14.3%, Sharpe 0.54, max DD -33.9%; 3 trials/fold degrade a +253.1%, Sharpe 0.43 |
| `deterministic_ema_signal_diagnostic` | pas de training : score tradable = rang mensuel d'un signal EMA / momentum observe | signaux calculables au mois de decision : `ema_ratio_2_12_rank_month`, `ema_ratio_3_12_rank_month`, `technical_z_mean`, `technical_rank_mean`; aucun objectif Legacy en training car il n'y a pas de training | diagnostic 2015+ `2015-01` a `2026-04`, 136 mois, 1 258 lignes Legacy | KPI strict `legacy_k`: `ema_ratio_3_12_rank_month` retrouve `655 / 1 258 = 52.1%`; `ema_ratio_2_12_rank_month` retrouve `647 / 1 258 = 51.4%` |
| `ema_rich_future_target` | futur rendement relatif : regression de `future_excess_return`, classification `>0%` / `>5%`, ranking mensuel | 16 EMA de base + rang mensuel de chaque EMA + z-score mensuel + flags top quartile + agregats `ema_rank_mean`, `ema_rank_max`, `ema_z_mean`, `ema_z_max`, `ema_top25_vote_count` | holdings Legacy `2010-02` a `2026-04`, 195 mois, 2 070 lignes Legacy | meilleur modele = ranking futur rendement relatif, `492 / 2 070 = 23.8%` |
| `legacy_atomic_recomposition` | pas de ML : decomposition des blocs Legacy existants | sorties des 4 blocs `Legacy_Optuna_11`, `12`, `21`, `22`; chaque bloc contient un couple EMA, un nombre cible d'actions, une limite secteur | paniers Legacy sur la longue periode, 2 088 lignes Legacy dans ce diagnostic | union des 4 blocs = `2 088 / 2 088 = 100%`; un seul bloc monte entre 64.1% et 76.5% |
| `atomic_feature_future_target` | futur rendement relatif : regression de `future_excess_return` clippe, classification `>0%` / `>5%`, ranking mensuel | features atomiques derivees des blocs Legacy : votes par bloc, flags `Legacy_Optuna_*`, quantiles/ratios `mtr`, rangs mensuels atomiques; certaines variantes ajoutent aussi EMA | holdings Legacy `2010-02` a `2026-04`, 195 mois, 2 070 lignes Legacy | regression = `2 070 / 2 070 = 100%`; classifieur `>5%` = `2 041 / 2 070 = 98.6%` |
| `generalized_ema_expert` | score deterministe et modeles ML sur futur rendement relatif | experts EMA parametriques `short/long/n_actions/secteur`, selectionnes chaque mois par performance passee; features ticker = votes/scores/mtr/rangs des experts actifs | holdings Legacy `2010-02` a `2026-04`, 195 mois, 2 070 lignes Legacy | meilleur vrai candidat generalisable = somme des scores experts EMA, `1 254 / 2 070 = 60.6%`; XGBoost `>5%` = 42.1% |

Lecture importante :

- `atomic_feature_future_target` est le meilleur pour **recomposer Legacy**, mais
  il utilise des variables qui viennent des briques Legacy exactes. C'est une
  preuve que la representation contient le signal, pas une methode tradable a
  poursuivre.
- `generalized_ema_expert` reste un diagnostic utile, mais ce n'est pas la voie
  active demandee : le score deterministe bat le ML, alors que la consigne
  actuelle impose de travailler les regressions.
- La voie active au 2026-06-21 est donc `tradable_ema_regression_optuna` :
  apprendre le futur rendement relatif avec des variables EMA tradables, puis
  mesurer seulement combien d'actions Legacy sont retrouvees.
- Les modeles `future_excess_return` purs sur EMA enrichies progressent avec
  warm starts Optuna, mais ne suffisent pas encore : le test large corrige est a
  34.8%, sous l'objectif de 50%.
- Diagnostic ajoute le 2026-06-27 : un simple rang EMA tradable retrouve deja
  plus de 50% de Legacy sur 2015+. Le probleme n'est donc pas "il n'y a pas de
  signal EMA". Le probleme est que les regressions boosting propres diluent ce
  signal quand elles optimisent un objectif futur trop global.
- Diagnostic ajoute le 2026-06-27 soir : le boosting seul n'est pas mort, mais
  il est insuffisant tel quel. Le meilleur boosting pur long bat legerement SPY
  en rendement brut, mais avec Sharpe plus faible et drawdown plus profond, et
  il reste tres loin de Legacy. Ajouter quelques trials Optuna degrade le run
  rank-regression, ce qui suggere un overfit de validation plutot qu'un manque
  simple de trials.

Regle de suite ajoutee apres clarification utilisateur :

- ne plus developper de candidat final avec des variables `legacy_atomic_*` ou
  `legacy_optuna_*` ;
- utiliser seulement des variables calculables en live pour trader : prix
  relatifs a l'indice, EMA relatives, filtres d'univers, secteur, et scores
  d'experts calcules sur rendement passe ;
- garder les runs atomiques uniquement comme diagnostic historique de plafond de
  replication.

## Timeline

| date | commit / run | objet |
|---|---|---|
| 2026-06-11 | `85a2b12`, `e69537b`, `903207b`, `02ba3bf`, `c1ac76c`, `324d385` | premiere voie boosting mlcraft, CPCV, target +5%, rapport SHAP |
| 2026-06-12 | `52da0ad` | walk-forward strict, warm-start Optuna, objectif top-N, selection configurable |
| 2026-06-12 | `224ff4d` | rapport walk-forward : full model proba brute, risk-adjusted, SHAP |
| 2026-06-13 | `7bb6f65` | experience residual `base_margin/init_score` sur features selectionnees |
| 2026-06-14 | `780af35` | EMA-only residual et diagnostic probabilites des actions legacy |
| 2026-06-14 | `f7d1b29` | construction des 7 variantes signal-copy et diagnostic clone legacy |
| 2026-06-14 | run `outputs/ema_rich_future_target_20260614_222201` | premier test EMA enrichies, target futur rendement relatif uniquement |
| 2026-06-15 | run `outputs/ema_rich_future_target_20260615_201243` | test long 2010-2026, KPI unique de recomposition |
| 2026-06-16 | run `outputs/legacy_atomic_recomposition_20260616_122045` | decomposition atomique des blocs Optuna Legacy |
| 2026-06-16 | run `outputs/legacy_atomic_feature_frame_20260616_195618` | frame ML enrichi par les signaux atomiques Legacy |
| 2026-06-16 | run `outputs/atomic_feature_future_target_20260616_200236` | training futur rendement relatif sur features atomiques |
| 2026-06-16 | run `outputs/generalized_ema_expert_frame_20260616_231854` | experts EMA selectionnes par performance passee |
| 2026-06-16 | run `outputs/generalized_ema_expert_models_20260616_232041` | training sur features d'experts EMA generalisables |
| 2026-06-16 | run `outputs/generalized_ema_expert_sweep_20260616_234440` | sweep memoire/top experts sur candidats EMA observes |
| 2026-06-16 | run `outputs/generalized_ema_expert_sweep_20260616_234500` | sweep memoire/top experts sur voisins EMA |
| 2026-06-16 | run `outputs/generalized_ema_expert_models_20260616_234556` | training sur le meilleur cadre EMA generalisable |
| 2026-06-20 | script `scripts/experiments/run_tradable_ema_regression_optuna.py` | nouvelle regression EMA tradable avec Optuna random startup puis TPE et warm starts JSON |
| 2026-06-20 | run `outputs/tradable_ema_regression_optuna_20260620_231702` | baseline sans warm-start, 23 mois test |
| 2026-06-20 | run `outputs/tradable_ema_regression_optuna_20260620_232506` | regression warm-startee, 23 mois test |
| 2026-06-21 | run `outputs/tradable_ema_regression_optuna_20260621_001753` | regression warm-startee corrigee mlcraft, 23 mois test |
| 2026-06-21 | run `outputs/tradable_ema_regression_optuna_20260621_003954` | regression warm-startee corrigee mlcraft, test large 59 mois |
| 2026-06-23 | run `outputs/tradable_ema_regression_optuna_20260623_131514` | test court 100 trials/fold : validation meilleure, test moins bon |
| 2026-06-26 | run `outputs/tradable_ema_regression_trading_backtest_20260626_110356` | backtest trading de la regression EMA tradable vs Legacy |
| 2026-06-26 | run `outputs/tradable_technical_regression_optuna_20260626_212142` | regression avec toutes les features techniques disponibles, test court |
| 2026-06-27 | run `outputs/tradable_technical_trading_backtest_20260627_012929` | backtest trading du score technique complet |
| 2026-06-27 | run `outputs/tradable_ema_trading_backtest_20260627_012929` | backtest EMA-only sur la meme periode courte que le run technique |
| 2026-06-27 | script `scripts/experiments/run_tradable_ema_residual_regression.py` | test residual : base EMA-only puis regression du residu sur technique non-EMA |
| 2026-06-27 | run `outputs/tradable_ema_residual_regression_20260627_020253` | residual regression, 23 mois test, shrinkages 0.25/0.50/1.00 |
| 2026-06-27 | runs `outputs/ema_plus_residual_0_25_trading_backtest_20260627_020626`, `outputs/ema_plus_residual_0_50_trading_backtest_20260627_020654`, `outputs/ema_base_trading_backtest_20260627_020633` | backtests residual et base EMA fixe comparable |
| 2026-06-27 | script `scripts/experiments/run_tradable_ema_regression_optuna.py` | ajout d'objectifs Optuna sans Legacy : rendement top K et precision top K en validation |
| 2026-06-27 | runs `outputs/tradable_ema_regression_optuna_20260627_021449` a `outputs/tradable_ema_regression_optuna_20260627_023621` | sweep court sans objectif Legacy, sans warm-start, objectifs rendement top 10 et precision top 10/20/30/50 |
| 2026-06-27 | runs `outputs/tradable_ema_trading_backtest_20260627_024211`, `024213`, `024215`, `024216`, `024218` | backtests du sweep sans objectif Legacy |
| 2026-06-27 | runs `outputs/tradable_ema_regression_optuna_20260627_121144`, `125020`, `133211`, `140859`, `144652` | sweep 2015+ sans objectif Legacy, 136 folds mensuels par objectif |
| 2026-06-27 | runs `outputs/tradable_ema_trading_backtest_20260627_152139`, `152141`, `152143`, `152145`, `152147` | backtests 2015+ des cinq objectifs Optuna propres |
| 2026-06-27 | script `scripts/experiments/analyze_legacy_factor_exposures.py`, run `outputs/legacy_factor_exposure_20260627_154437` | diagnostic Legacy 2015+ : exposition features, secteurs, tickers, blocs EMA atomiques |
| 2026-06-27 | script `scripts/experiments/build_deterministic_signal_predictions.py`, run `outputs/deterministic_signal_predictions_20260627_154617` | generation de predictions deterministes tradables pour tester les scores EMA simples |
| 2026-06-27 | runs `outputs/ema_ratio_2_12_rank_month_trading_backtest_20260627_154632`, `outputs/ema_ratio_3_12_rank_month_trading_backtest_20260627_154632`, `outputs/technical_z_mean_trading_backtest_20260627_154632`, `outputs/technical_rank_mean_trading_backtest_20260627_154632` | backtests 2015+ des temoins deterministes EMA / technique |
| 2026-06-27 | script `scripts/experiments/run_portfolio_boosting_top_return_classifier.py`, run `outputs/portfolio_boosting_top_return_classifier_20260627_225903` | boosting seul mlcraft : classifier top 10% futur rendement relatif, baseline Optuna enqueued |
| 2026-06-27 | script `scripts/experiments/run_portfolio_boosting_rank_regression.py`, runs `outputs/portfolio_boosting_rank_regression_20260627_230913`, `outputs/portfolio_boosting_rank_regression_20260627_233738` | boosting seul mlcraft : regression du rang mensuel futur, test baseline puis 3 trials/fold |

Runs principaux :

- `outputs/xgboost_timefold_backtest_20260612_175250`
- `outputs/residual_init_score_experiment_20260613_112301`
- `outputs/legacy_probability_diagnostics_20260614`
- `outputs/signal_copy_models_20260614_214711`
- `outputs/legacy_clone_diagnostics_20260614`
- `outputs/ema_rich_future_target_20260614_222201`
- `outputs/ema_rich_future_target_20260615_201243`
- `outputs/legacy_atomic_recomposition_20260616_122045`
- `outputs/legacy_atomic_feature_frame_20260616_195618`
- `outputs/atomic_feature_future_target_20260616_200236`
- `outputs/generalized_ema_expert_frame_20260616_231854`
- `outputs/generalized_ema_expert_models_20260616_232041`
- `outputs/generalized_ema_expert_sweep_20260616_234440`
- `outputs/generalized_ema_expert_sweep_20260616_234500`
- `outputs/generalized_ema_expert_frame_20260616_234521`
- `outputs/generalized_ema_expert_models_20260616_234556`
- `outputs/tradable_ema_regression_optuna_20260620_231702`
- `outputs/tradable_ema_regression_optuna_20260620_232506`
- `outputs/tradable_ema_regression_optuna_20260621_001753`
- `outputs/tradable_ema_regression_optuna_20260621_003954`
- `outputs/tradable_ema_regression_optuna_20260623_131514`
- `outputs/tradable_ema_regression_trading_backtest_20260626_110356`
- `outputs/tradable_technical_regression_optuna_20260626_212142`
- `outputs/tradable_technical_trading_backtest_20260627_012929`
- `outputs/tradable_ema_trading_backtest_20260627_012929`
- `outputs/tradable_ema_residual_regression_20260627_020253`
- `outputs/ema_plus_residual_0_25_trading_backtest_20260627_020626`
- `outputs/ema_plus_residual_0_50_trading_backtest_20260627_020654`
- `outputs/ema_base_trading_backtest_20260627_020633`
- `outputs/tradable_ema_regression_optuna_20260627_021449`
- `outputs/tradable_ema_regression_optuna_20260627_022020`
- `outputs/tradable_ema_regression_optuna_20260627_022531`
- `outputs/tradable_ema_regression_optuna_20260627_023102`
- `outputs/tradable_ema_regression_optuna_20260627_023621`
- `outputs/tradable_ema_trading_backtest_20260627_024211`
- `outputs/tradable_ema_trading_backtest_20260627_024213`
- `outputs/tradable_ema_trading_backtest_20260627_024215`
- `outputs/tradable_ema_trading_backtest_20260627_024216`
- `outputs/tradable_ema_trading_backtest_20260627_024218`
- `outputs/tradable_ema_regression_optuna_20260627_121144`
- `outputs/tradable_ema_regression_optuna_20260627_125020`
- `outputs/tradable_ema_regression_optuna_20260627_133211`
- `outputs/tradable_ema_regression_optuna_20260627_140859`
- `outputs/tradable_ema_regression_optuna_20260627_144652`
- `outputs/tradable_ema_trading_backtest_20260627_152139`
- `outputs/tradable_ema_trading_backtest_20260627_152141`
- `outputs/tradable_ema_trading_backtest_20260627_152143`
- `outputs/tradable_ema_trading_backtest_20260627_152145`
- `outputs/tradable_ema_trading_backtest_20260627_152147`
- `outputs/legacy_factor_exposure_20260627_154437`
- `outputs/deterministic_signal_predictions_20260627_154617`
- `outputs/ema_ratio_2_12_rank_month_trading_backtest_20260627_154632`
- `outputs/ema_ratio_3_12_rank_month_trading_backtest_20260627_154632`
- `outputs/technical_z_mean_trading_backtest_20260627_154632`
- `outputs/technical_rank_mean_trading_backtest_20260627_154632`
- `outputs/portfolio_boosting_top_return_classifier_20260627_225903`
- `outputs/portfolio_boosting_rank_regression_20260627_230913`
- `outputs/portfolio_boosting_rank_regression_20260627_233738`

## Donnees communes

Source principale :

- `outputs/xgboost_timefold_backtest_20260612_175250/model_frame.parquet`

Source legacy :

- `outputs/2026-06-07/legacy_detailed_returns_polars.parquet`

Fenetre de test commune :

- holdings `2025-06` a `2026-04`
- 11 mois exploitables
- walk-forward strict
- top 10 fixe pour les tests allocation
- nombre de tickers Legacy variable pour les tests de recomposition

Features :

- `full_features` : 63 features du run full model.
- `ema_features` : 16 features EMA :
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

## Les 7 modeles signal-copy

Les 7 modeles ci-dessous ont ete construits dans :

- script : `scripts/experiments/run_signal_copy_models.py`
- commit : `f7d1b29 add legacy clone signal diagnostics`
- date de construction : 2026-06-14
- run : `outputs/signal_copy_models_20260614_214711`

Ils utilisent tous XGBoost natif. Ce sont des experiences fixes, sans Optuna,
pour comparer les familles de signal rapidement et proprement.

### 1. `distill_legacy`

But : apprendre directement les trades legacy.

Construction :

- features : `full_features` ;
- target : `legacy_selected` ;
- `legacy_selected = 1` si le ticker est present dans les portfolios legacy
  `Combined_Equal` ou `Combined_Frequency` pour le mois de holding ;
- entrainement limite aux mois ou la source legacy existe ;
- modele : XGBoost `binary:logistic` ;
- rounds : 250 ;
- selection : top 10 par `distill_legacy`.

Interpretation :

Ce modele ne cherche pas directement le rendement futur. Il cherche a reproduire
la decision legacy.

Statut :

- meilleur modele de clonage ;
- meilleur modele des 7 en top 10 allocation ;
- statut corrige : teacher/oracle de diagnostic, pas candidat final.

### 2. `rank_pairwise`

But : apprendre un ranking mensuel plutot qu'une classification binaire.

Construction :

- features : `full_features` ;
- target : rang percentile mensuel de `future_excess_return` ;
- objectif XGBoost : `rank:pairwise` ;
- groupement : par `year_month` ;
- rounds : 250 ;
- selection : top 10 par `rank_pairwise`.

Interpretation :

Ce modele teste l'idee que legacy capte un ordre relatif mensuel, pas seulement
la proba de depasser un seuil fixe.

Statut :

- positif en allocation ;
- faible en clonage legacy.

### 3. `regression_excess`

But : predire directement le rendement relatif futur.

Construction :

- features : `full_features` ;
- target : `future_excess_return` clippe entre `-30%` et `+30%` ;
- objectif XGBoost : `reg:squarederror` ;
- rounds : 250 ;
- selection : top 10 par rendement relatif predit.

Interpretation :

Ce modele evite la perte d'information du label binaire `> 5%`.

Statut :

- hit-rate correct ;
- faible clonage legacy.

### 4. `monotone_ema_full`

But : forcer le modele full a respecter le sens du signal EMA.

Construction :

- features : `full_features` ;
- target : `future_excess_return > 5%` ;
- objectif XGBoost : `binary:logistic` ;
- contraintes monotones :
  - `+1` sur chaque feature EMA ;
  - `0` sur les autres features ;
- rounds : 250 ;
- selection : top 10 par `monotone_ema_full`.

Interpretation :

Le modele garde tout l'univers de variables, mais il ne peut pas apprendre une
relation negative sur les features EMA.

Statut :

- n'a pas marche ;
- probablement trop rigide ou mauvais signe monotone pour certaines EMA.

### 5. `two_stage_ema_full`

But : utiliser EMA comme signal principal, puis laisser le full model corriger.

Construction :

1. Entrainer un score EMA residual :
   - features : `ema_features` ;
   - target : `future_excess_return > 5%` ;
   - base model EMA + residual XGBoost avec `base_margin`.
2. Ajouter ce score EMA comme feature supplementaire aux `full_features`.
3. Entrainer un XGBoost `binary:logistic` sur :
   - `full_features + ema_score`.
4. Selection : top 10 par `two_stage_ema_full`.

Interpretation :

Le modele full a acces au signal EMA explicite et peut l'utiliser ou le corriger.

Statut :

- n'a pas assez copie legacy ;
- meilleur en AUC que certains modeles, mais pas en trades.

### 6. `gated_full_after_ema`

But : utiliser EMA comme filtre dur avant le full model.

Construction :

1. Entrainer un score EMA residual.
2. Entrainer un full model `binary:logistic` sur `future_excess_return > 5%`.
3. Pour chaque mois :
   - garder seulement les top 50 tickers par score EMA ;
   - mettre le score des autres a `-1e9`.
4. Selection : top 10 par score full model, mais uniquement dans le gate EMA.

Interpretation :

Le modele full ne peut choisir que parmi les noms deja valides par EMA.

Statut :

- n'a pas marche dans cette version ;
- le gate EMA top 50 n'est pas suffisant pour reproduire legacy.

### 7. `weighted_top_classifier`

But : forcer le classifieur a regarder davantage les exemples qui comptent.

Construction :

- features : `full_features` ;
- target : `future_excess_return > 5%` ;
- objectif : XGBoost `binary:logistic` ;
- poids d'echantillon :
  - poids de base = 1 ;
  - `+3` si `legacy_selected = 1` ;
  - `+2` si `future_excess_return > 5%` ;
- rounds : 250 ;
- selection : top 10 par `weighted_top_classifier`.

Interpretation :

Ce modele garde la target rendement, mais il donne plus d'importance aux trades
legacy et aux futurs gagnants.

Statut :

- echec net en clonage ;
- a eviter dans cette forme.

## Benchmarks et controles non comptes dans les 7

### `tradable_ema_regression_optuna`

Construit le 2026-06-20 / 2026-06-21 dans :

- script : `scripts/experiments/run_tradable_ema_regression_optuna.py`
- run court corrige : `outputs/tradable_ema_regression_optuna_20260621_001753`
- run large corrige : `outputs/tradable_ema_regression_optuna_20260621_003954`
- warm-start actuel a reutiliser :
  `outputs/tradable_ema_regression_optuna_20260621_003954/warm_start_candidates.json`

But :

Revenir exactement a la demande corrigee : ne pas apprendre Legacy directement,
ne pas utiliser de score deterministe d'experts, mais entrainer une regression
boosting sur le futur rendement relatif, puis regarder si le classement obtenu
retrouve les actions Legacy.

Ce qui est regresse :

```text
future_excess_return clippe a +/-30%
```

Lecture metier :

```text
future_excess_return = rendement futur de l'action sur le mois suivant
                       relatif au rendement de l'indice
```

Donc le modele essaie d'apprendre : "quelle action devrait faire mieux que
l'indice le mois prochain", pas "quelle action Legacy a choisi".

Variables utilisees :

- 16 variables EMA de base deja presentes dans le frame :
  `ema_ratio_*` et `price_to_ema_*` ;
- pour chaque variable EMA :
  - rang mensuel du ticker dans l'univers du mois ;
  - z-score mensuel ;
  - flag top quartile ;
  - flag bottom quartile ;
- agregats horizontaux sur les EMA :
  - `ema_rank_mean`, `ema_rank_max`, `ema_rank_min` ;
  - `ema_z_mean`, `ema_z_max`, `ema_z_min` ;
  - `ema_top25_vote_count`, `ema_bottom25_vote_count`.

Total : 88 variables. Toutes sont calculables au mois de decision a partir des
prix/EMA disponibles. Les variables interdites ne sont pas utilisees :
`legacy_selected`, `legacy_atomic_*`, `legacy_optuna_*`, sorties directes des
blocs Legacy.

Modele et tuning :

- librairie boosting : `mlcraft`, via `ModelFactory.create("xgboost")` ;
- type : regression ;
- Optuna : `TPESampler` ;
- demarrage : essais aleatoires controles par `startup_trials` ;
- suite : recherche TPE / bayesienne ;
- warm starts : les meilleurs trials sont sauves en JSON, puis reenqueues au
  debut des runs suivants ;
- politique de selection de trial maintenant configurable :
  - `best_objective` : meilleur objectif validation penalise, comportement
    historique ;
  - `warm_only` : les nouveaux trials enrichissent le JSON, mais le modele final
    du fold choisit seulement parmi les warm-starts connus ;
  - `top10_min_gap` : parmi les 10 meilleurs objectifs, choisir le trial avec le
    plus petit ecart train/validation ;
- objectif de tuning par fold :

```text
overlap validation avec Legacy
- 0.25 * abs(overlap train - overlap validation)
```

Le terme de penalite evite de choisir un trial qui colle beaucoup mieux le train
que la validation.

Selection et KPI :

Pour chaque mois de test :

```text
K = nombre d'actions choisies par Legacy ce mois-la
modele = top K actions par rendement relatif futur predit
score du mois = actions communes entre modele et Legacy / K
```

Le KPI communique pour ce travail est seulement :

```text
nombre d'actions communes entre modele et Legacy
/
nombre d'actions choisies par Legacy ce mois-la
```

Runs :

| run | protocole | actions communes | actions Legacy | recomposition | mediane mensuelle |
|---|---|---:|---:|---:|---:|
| `outputs/tradable_ema_regression_optuna_20260620_231702` | 23 mois, 8 trials/fold, 3 random startup, sans warm-start | 21 | 169 | 12.4% | 0.0% |
| `outputs/tradable_ema_regression_optuna_20260620_232506` | 23 mois, 16 trials/fold, 4 random startup, warm-start depuis le run precedent | 67 | 169 | 39.6% | 40.0% |
| `outputs/tradable_ema_regression_optuna_20260621_001753` | 23 mois, 16 trials/fold, 4 random startup, warm-start, contrat mlcraft corrige | 68 | 169 | 40.2% | 33.3% |
| `outputs/tradable_ema_regression_optuna_20260621_003954` | 59 mois, 16 trials/fold, 4 random startup, warm-start depuis le run corrige court | 159 | 457 | 34.8% | 33.3% |
| `outputs/tradable_ema_regression_optuna_20260623_131514` | 23 mois, 100 trials/fold, 20 random startup, warm-start depuis le run large corrige | 63 | 169 | 37.3% | 37.5% |
| `outputs/tradable_technical_regression_optuna_20260626_212142` | 23 mois, 8 trials/fold, 2 random startup, 55 features techniques de base + rangs/z-scores/flags/agregats | 55 | 169 | 32.5% | 33.3% |

Details du run large corrige :

- periode test : `2021-06` a `2026-04` ;
- mois testes : 59 ;
- mois avec recomposition >= 50% : 15 / 59 ;
- meilleur mois : 71.4% ;
- pire mois : 0.0% ;
- quartiles mensuels : 25% = 22.2%, mediane = 33.3%, 75% = 50.0%.

Lecture :

- le warm-start Optuna aide vraiment : sur le test court, la recomposition passe
  de 12.4% sans warm-start a environ 40% avec warm-start ;
- augmenter a 100 trials/fold n'a pas ameliore le test court : la validation
  moyenne monte de 37.9% a 40.3%, mais le test baisse de 40.3% a 37.6% en
  moyenne mensuelle ;
- le replay des 2300 trials du run 100 a montre que la meilleure politique sur
  test reste `warm_only` : choisir seulement parmi les 12 warm-starts enqueued
  donne `68 / 169 = 40.2%`, alors que choisir le meilleur objectif Optuna donne
  `63 / 169 = 37.3%` ;
- decision du 2026-06-23 : ne pas lancer le run large 100 trials brut. Plus de
  trials cree surtout de l'overfit validation avec les features EMA actuelles ;
- test du 2026-06-26 : ajouter toutes les familles techniques brutes n'a pas
  ameliore. Le run technique complet baisse a `55 / 169 = 32.5%` sur la meme
  periode courte ou EMA-only fait `68 / 169 = 40.2%` ;
- la regression EMA tradable est maintenant une baseline propre, non trichee,
  mais elle ne retrouve pas encore 50% de Legacy sur test large ;
- le meilleur point de depart actuel pour le prochain run est le JSON du run
  large corrige :
  `outputs/tradable_ema_regression_optuna_20260621_003954/warm_start_candidates.json` ;
- la suite logique n'est pas seulement "plus de trials" : il faut aussi enrichir
  les EMA tradables de facon controlee. Ajouter toutes les features techniques
  d'un coup dilue le signal ; la piste suivante doit etre selection/gating par
  famille technique ou apprentissage en deux etages, pas "tout mettre dans le
  meme modele".

#### Correction methodologique du 2026-06-27 : Optuna sans objectif Legacy

Probleme identifie :

```text
L'ancien protocole entrainait une regression sur future_excess_return,
mais choisissait les hyperparametres Optuna avec la recomposition Legacy.
```

Ce protocole etait utile pour comprendre Legacy, mais il n'est pas propre pour
construire un algo d'allocation autonome. Legacy influencait les hyperparametres.
A partir de ce test, Legacy doit etre reserve au diagnostic final, jamais a
l'objectif Optuna quand on evalue un modele de trading.

Changement code :

- script modifie : `scripts/experiments/run_tradable_ema_regression_optuna.py`
- nouveaux objectifs :
  - `val_topk_mean_return` : moyenne mensuelle du rendement relatif futur du
    top K en validation ;
  - `val_topk_precision` : precision mensuelle du top K en validation, avec
    succes defini par `future_excess_return > 0` ;
- `--objective-top-k` permet de tester `K=10/20/30/50` ;
- `--objective-return-col` permet de choisir `future_excess_return` ou
  `future_return` pour l'objectif rendement ;
- Legacy reste calcule dans `recomposition_summary.csv`, mais seulement comme
  diagnostic hors objectif.

Premier sweep court propre :

- features : EMA-only, memes 88 variables ;
- target modele : `future_excess_return` clippe a `[-30%, +30%]` ;
- aucun warm-start Legacy : `--warm-start-top-k 0` ;
- Optuna : 8 trials/fold, 3 random startup puis TPE ;
- penalite train/val gap desactivee : `--lambda-gap 0.0` ;
- test : `2025-06` a `2026-04`, 11 mois ;
- Legacy test disponible : 71 actions.

Runs :

| objectif Optuna | training | backtest | recomposition Legacy test | meilleur scenario trading | rendement total | CAGR | Sharpe | max DD |
|---|---|---|---:|---|---:|---:|---:|---:|
| rendement top 10 validation | `outputs/tradable_ema_regression_optuna_20260627_021449` | `outputs/tradable_ema_trading_backtest_20260627_024211` | `8 / 71 = 11.3%` | `tradable_ema_top_30` | 29.5% | 32.6% | 1.38 | -5.6% |
| precision top 10 validation | `outputs/tradable_ema_regression_optuna_20260627_022020` | `outputs/tradable_ema_trading_backtest_20260627_024213` | `9 / 71 = 12.7%` | `tradable_ema_top_20` | 33.5% | 37.0% | 1.38 | -8.2% |
| precision top 20 validation | `outputs/tradable_ema_regression_optuna_20260627_022531` | `outputs/tradable_ema_trading_backtest_20260627_024215` | `4 / 71 = 5.6%` | `tradable_ema_top_30` | 34.7% | 38.4% | 1.76 | -5.6% |
| precision top 30 validation | `outputs/tradable_ema_regression_optuna_20260627_023102` | `outputs/tradable_ema_trading_backtest_20260627_024216` | `3 / 71 = 4.2%` | `tradable_ema_top_30` | 65.9% | 73.7% | 3.15 | -5.6% |
| precision top 50 validation | `outputs/tradable_ema_regression_optuna_20260627_023621` | `outputs/tradable_ema_trading_backtest_20260627_024218` | `7 / 71 = 9.9%` | `tradable_ema_top_30` | 34.4% | 38.1% | 1.57 | -5.6% |

Metrices top K sur le test :

| objectif Optuna | top10 excess | top10 precision | top20 excess | top20 precision | top30 excess | top30 precision | top50 excess | top50 precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rendement top 10 validation | -1.05% | 44.5% | -0.18% | 42.3% | 0.57% | 46.7% | 0.33% | 47.8% |
| precision top 10 validation | 0.09% | 50.0% | 0.86% | 49.1% | 0.30% | 49.4% | 0.05% | 47.3% |
| precision top 20 validation | 0.07% | 47.3% | 0.90% | 48.2% | 0.90% | 49.1% | 0.42% | 48.2% |
| precision top 30 validation | 1.48% | 52.7% | 1.29% | 51.4% | 2.88% | 53.6% | 1.12% | 48.5% |
| precision top 50 validation | -0.76% | 44.5% | 0.16% | 48.6% | 0.92% | 48.8% | 0.11% | 44.0% |

Lecture :

- le test confirme la fuite methodologique : des qu'Optuna n'utilise plus
  Legacy, la recomposition Legacy tombe entre 4.2% et 12.7% sur ce sweep court ;
- cela ne veut pas dire que le modele ne trade pas : `precision top 30` donne le
  meilleur backtest court, mais sans retrouver Legacy ;
- il faut separer explicitement deux questions :
  - **allocation autonome** : optimiser rendement/precision/risk-adjusted return
    en validation, sans Legacy ;
  - **comprehension Legacy** : mesurer pourquoi Legacy selectionne d'autres
    noms, mais ne pas utiliser Legacy pour choisir les hyperparametres ;
- le meilleur candidat propre a etendre est `precision top 30 validation`, mais
  le test ne couvre que 11 mois. Il faut le rejouer sur une periode plus longue
  et avec plus de trials avant d'en tirer une conclusion robuste.

#### Sweep 2015+ sans objectif Legacy du 2026-06-27

But :

Verifier les objectifs Optuna propres sur une longue periode, avec rotation
mensuelle et reentrainement complet.

Protocole :

- source : `outputs/xgboost_timefold_backtest_20260612_175250/model_frame.parquet`
- features : EMA-only, 88 variables ;
- target modele : `future_excess_return` clippe a `[-30%, +30%]` ;
- debut test vise : `2015-01` decision month ;
- premiere performance realisee : `2015-02`, car le rendement porte sur le mois
  de detention suivant ;
- split mensuel :
  - train = tout l'historique avant la validation ;
  - validation = 12 mois ;
  - test = 1 mois ;
  - puis rotation d'un mois ;
- `min_train_months=168`, `val_months=12`, `test_months=1` ;
- 136 folds mensuels par objectif ;
- Optuna : 4 trials/fold, 2 random startup puis TPE ;
- aucun warm-start Legacy : `--warm-start-top-k 0` ;
- Legacy utilise seulement pour `recomposition_summary.csv`, pas pour choisir les
  hyperparametres.

Runs 2015+ :

| objectif Optuna | training | backtest | recomposition Legacy | meilleur scenario ML | rendement total | CAGR | Sharpe | max DD |
|---|---|---|---:|---|---:|---:|---:|---:|
| rendement top 10 validation | `outputs/tradable_ema_regression_optuna_20260627_121144` | `outputs/tradable_ema_trading_backtest_20260627_152139` | `80 / 1236 = 6.5%` | `tradable_ema_top_50` | 268.7% | 12.3% | 0.57 | -25.0% |
| precision top 10 validation | `outputs/tradable_ema_regression_optuna_20260627_125020` | `outputs/tradable_ema_trading_backtest_20260627_152141` | `73 / 1236 = 5.9%` | `tradable_ema_top_50` | 293.9% | 13.0% | 0.62 | -24.4% |
| precision top 20 validation | `outputs/tradable_ema_regression_optuna_20260627_133211` | `outputs/tradable_ema_trading_backtest_20260627_152143` | `81 / 1236 = 6.6%` | `tradable_ema_top_50` | 269.1% | 12.3% | 0.57 | -24.1% |
| precision top 30 validation | `outputs/tradable_ema_regression_optuna_20260627_140859` | `outputs/tradable_ema_trading_backtest_20260627_152145` | `71 / 1236 = 5.7%` | `tradable_ema_top_50` | 231.1% | 11.2% | 0.45 | -38.5% |
| precision top 50 validation | `outputs/tradable_ema_regression_optuna_20260627_144652` | `outputs/tradable_ema_trading_backtest_20260627_152147` | `83 / 1236 = 6.7%` | `tradable_ema_top_30` | 314.5% | 13.5% | 0.62 | -23.3% |

Baselines communes sur la meme periode :

| modele | rendement total | CAGR | Sharpe | max DD | mois positifs |
|---|---:|---:|---:|---:|---:|
| Legacy `Combined_Frequency` | 1118.1% | 24.9% | 0.93 | -23.5% | 63.0% |
| Legacy `Combined_Equal` | 851.1% | 22.2% | 0.85 | -23.6% | 60.0% |
| SPY | 332.2% | 13.9% | 0.79 | -23.9% | 69.6% |

Tous les top N des objectifs 2015+ :

| objectif Optuna | top 5 | top 7 | top 10 | top 20 | top 30 | top 50 | legacy K |
|---|---:|---:|---:|---:|---:|---:|---:|
| rendement top 10 validation | 45.0% | 93.6% | 124.9% | 195.1% | 238.0% | 268.7% | 105.8% |
| precision top 10 validation | 31.3% | 70.7% | 157.6% | 201.2% | 245.9% | 293.9% | 87.7% |
| precision top 20 validation | 52.5% | 105.9% | 159.1% | 219.5% | 255.9% | 269.1% | 78.8% |
| precision top 30 validation | 66.2% | 72.2% | 133.7% | 131.0% | 157.9% | 231.1% | 51.1% |
| precision top 50 validation | 47.8% | 101.9% | 160.0% | 252.6% | 314.5% | 278.7% | 103.2% |

Lecture :

- le resultat court `2025-06 -> 2026-04` ne generalise pas tel quel : sur
  `2015-02 -> 2026-04`, aucun objectif EMA-only propre ne bat Legacy ;
- le meilleur ML propre long est `precision top 50 validation`, applique en
  top 30 : 314.5% de rendement total, presque au niveau SPY en rendement total
  mais avec un Sharpe plus faible ;
- Legacy reste tres au-dessus : `Combined_Frequency` fait 1118.1% de rendement
  total et 24.9% de CAGR ;
- la recomposition Legacy reste tres faible (`5.7%` a `6.7%`), ce qui confirme
  que le signal Legacy n'est pas retrouve naturellement quand Optuna optimise
  uniquement le rendement/proba future ;
- decision : ne pas promouvoir ces objectifs EMA-only propres en algo
  d'allocation. Ils deviennent une baseline longue non contaminee. La prochaine
  recherche doit ajouter d'autres signaux ou une architecture de ranking/risk,
  pas seulement augmenter les trials EMA-only.

#### Boosting seul portfolio 2015+ du 2026-06-27 soir

But :

Repondre directement a la question : "si on prend seulement le boosting, sans
recopier Legacy et sans blend deterministe EMA/momentum, est-ce que ca marche en
allocation ?"

Point methodologique corrige :

Les scripts courts avec `n_trials=1` pouvaient auparavant lancer un seul trial
aleatoire Optuna. Ce n'etait pas un baseline fiable. Les scripts portfolio
boosting injectent maintenant toujours un trial de depart conservateur avant les
trials aleatoires/TPE. Cela rend les runs `n_trials=1` interpretables comme un
baseline, et pas comme un tirage chanceux.

##### `portfolio_boosting_top_return_classifier`

Ce qui est appris :

```text
target = 1 si l'action est dans le top 10% mensuel du futur rendement relatif
target = 0 sinon
```

Lecture metier :

Le modele essaie d'apprendre la probabilite qu'une action soit parmi les grandes
surperformances du mois suivant, relativement a l'indice.

Variables :

- toutes les variables techniques calculables au mois de decision ;
- familles : momentum/prix, EMA, prix vs EMA, RSI, Bollinger, stochastique,
  distances aux plus hauts/bas, position dans le range, volatilite ;
- pour chaque variable : valeur brute, rang mensuel, z-score mensuel, flags top
  quartile / bottom quartile ;
- agregats transverses par famille ;
- aucune variable `legacy_selected`, `legacy_atomic_*`, `legacy_optuna_*`, ni
  decision Legacy dans le training.

Protocole :

- script : `scripts/experiments/run_portfolio_boosting_top_return_classifier.py`
- run propre baseline : `outputs/portfolio_boosting_top_return_classifier_20260627_225903`
- librairie : `mlcraft` + backend XGBoost classification ;
- tuning : baseline Optuna enqueued, `n_trials=1` sur ce run ;
- split : train passe, validation 12 mois, test 1 mois, rotation mensuelle ;
- periode de performance : `2015-02` a `2026-04`.

Resultats :

| modele | rendement total | CAGR | Sharpe | max DD | vol mensuelle | mois positifs |
|---|---:|---:|---:|---:|---:|---:|
| Legacy `Combined_Frequency` | 1118.1% | 24.9% | 0.93 | -23.5% | 7.1% | 63.0% |
| Legacy `Combined_Equal` | 851.1% | 22.2% | 0.85 | -23.6% | 6.9% | 60.0% |
| boosting classifier top 7 | 364.3% | 14.6% | 0.37 | -47.0% | 9.8% | 59.3% |
| boosting classifier top 50 | 361.3% | 14.6% | 0.54 | -35.0% | 6.8% | 63.7% |
| boosting classifier top 30 | 359.0% | 14.5% | 0.45 | -42.0% | 8.0% | 60.0% |
| SPY | 332.2% | 13.9% | 0.79 | -23.9% | 4.4% | 69.6% |
| boosting classifier top 5 | 223.8% | 11.0% | 0.24 | -48.5% | 10.9% | 54.8% |

Lecture :

- le boosting classifier pur n'est pas nul : plusieurs top N battent SPY en
  rendement brut ;
- il ne bat pas Legacy ;
- le risque est mauvais : Sharpe faible et drawdown beaucoup plus profond que
  Legacy ;
- top 5 est trop concentre ; top 30/50 dilue mieux le risque mais ne transforme
  pas le signal en avantage robuste.

##### `portfolio_boosting_rank_regression`

Ce qui est appris :

```text
target = rang percentile mensuel du futur rendement relatif
```

Exemple :

Si un ticker est dans les 10% meilleurs rendements relatifs futurs de son mois,
sa target est proche de `1.0`. S'il est dans les pires du mois, elle est proche
de `0.0`.

Important :

Ce n'est pas encore une vraie loss XGBoost `rank:pairwise`, car le contrat
`mlcraft` disponible expose ici regression/classification mais pas un task type
ranking avec groupes mensuels. Cette variante utilise donc `mlcraft` en
regression XGBoost sur une target de rang mensuel. C'est propre vis-a-vis de la
contrainte "boosting via mlcraft", mais ce n'est pas encore le meilleur objectif
mathematique possible pour un top mensuel.

Variables :

Memes variables techniques calculables que le classifier ci-dessus. Aucune
variable Legacy dans le modele.

Protocole baseline :

- script : `scripts/experiments/run_portfolio_boosting_rank_regression.py`
- run baseline : `outputs/portfolio_boosting_rank_regression_20260627_230913`
- librairie : `mlcraft` + backend XGBoost regression ;
- tuning : baseline Optuna enqueued, `n_trials=1` ;
- objectif validation : rendement moyen du top 20 ;
- split : train passe, validation 12 mois, test 1 mois, rotation mensuelle ;
- periode de performance : `2015-02` a `2026-04`.

Resultats baseline :

| modele | rendement total | CAGR | Sharpe | max DD | vol mensuelle | mois positifs |
|---|---:|---:|---:|---:|---:|---:|
| Legacy `Combined_Frequency` | 1118.1% | 24.9% | 0.93 | -23.5% | 7.1% | 63.0% |
| Legacy `Combined_Equal` | 851.1% | 22.2% | 0.85 | -23.6% | 6.9% | 60.0% |
| rank-regression top 20 | 348.3% | 14.3% | 0.54 | -33.9% | 6.5% | 59.3% |
| rank-regression top 10 | 335.1% | 14.0% | 0.45 | -46.2% | 7.7% | 58.5% |
| SPY | 332.2% | 13.9% | 0.79 | -23.9% | 4.4% | 69.6% |
| rank-regression top 30 | 289.9% | 12.9% | 0.52 | -32.5% | 6.0% | 63.0% |
| rank-regression top 50 | 276.3% | 12.5% | 0.53 | -28.1% | 5.7% | 64.4% |
| rank-regression top 5 | 161.0% | 8.9% | 0.21 | -59.2% | 9.4% | 51.9% |

Protocole hyperparametre reel :

- run : `outputs/portfolio_boosting_rank_regression_20260627_233738`
- `n_trials=3`, `startup_trials=2` ;
- baseline enqueued puis deux essais supplementaires par fold ;
- objectif validation : rendement moyen du top 20 ;
- sortie warm-start :
  `outputs/portfolio_boosting_rank_regression_20260627_233738/warm_start_candidates.json`.

Resultats 3 trials/fold :

| modele | rendement total | CAGR | Sharpe | max DD | vol mensuelle | mois positifs |
|---|---:|---:|---:|---:|---:|---:|
| Legacy `Combined_Frequency` | 1118.1% | 24.9% | 0.93 | -23.5% | 7.1% | 63.0% |
| Legacy `Combined_Equal` | 851.1% | 22.2% | 0.85 | -23.6% | 6.9% | 60.0% |
| SPY | 332.2% | 13.9% | 0.79 | -23.9% | 4.4% | 69.6% |
| rank-regression 3 trials top 20 | 253.1% | 11.9% | 0.43 | -42.0% | 6.7% | 62.2% |
| rank-regression 3 trials top 30 | 201.0% | 10.3% | 0.38 | -41.6% | 6.3% | 62.2% |
| rank-regression 3 trials top 50 | 194.7% | 10.1% | 0.39 | -38.5% | 6.0% | 63.7% |
| rank-regression 3 trials top 5 | 100.1% | 6.4% | 0.13 | -60.4% | 9.5% | 55.6% |

Lecture :

- augmenter legerement Optuna n'ameliore pas ce modele ; cela degrade le
  rendement et le drawdown ;
- l'explication probable est l'overfit de validation mensuelle : certains
  trials gagnent sur 12 mois de validation mais cassent en regime de crise,
  notamment autour de 2020 ;
- lancer 50 ou 100 trials/fold sans changer l'objectif risque d'amplifier ce
  probleme. La prochaine optimisation doit etre contrainte, par exemple :
  - selectionner seulement des warm starts robustes sur plusieurs regimes ;
  - penaliser le drawdown ou la volatilite dans l'objectif validation ;
  - utiliser une vraie loss de ranking mensuel avec groupes par mois ;
  - construire l'allocation par score/confiance au lieu d'un top N fixe 100%
    investi.

Decision :

Le meilleur boosting pur actuel est le baseline `rank-regression top 20` ou le
`classifier top 30/50`, selon le critere. Aucun n'est encore un bon algo final :
ils ont du signal, mais pas assez de controle du risque pour remplacer Legacy.

#### Backtest trading vs Legacy du 2026-06-26

Construit dans :

- script : `scripts/experiments/run_tradable_ema_regression_trading_backtest.py`
- run : `outputs/tradable_ema_regression_trading_backtest_20260626_110356`
- rapport HTML :
  `outputs/tradable_ema_regression_trading_backtest_20260626_110356/trading_backtest_comparison.html`

But :

Passer du KPI de recomposition Legacy au resultat trading reel du score
`tradable_ema_regression`, sur les memes mois que le run large corrige.

Periode :

- debut : `2021-06`
- fin : `2026-04`
- mois : 59

Regles testees :

- `tradable_ema_top_5` : top 5 actions par score du modele ;
- `tradable_ema_top_7` : top 7 actions par score du modele ;
- `tradable_ema_top_10` : top 10 actions par score du modele ;
- `tradable_ema_legacy_k` : top K actions par score du modele, ou K est le
  nombre d'actions Legacy du mois. Ce scenario est un diagnostic comparable a
  Legacy, pas une regle autonome pure.

Comparaison :

| modele | rendement total | CAGR | Sharpe | max drawdown | vol annualisee | mois positifs | nb actions moyen |
|---|---:|---:|---:|---:|---:|---:|---:|
| tradable_ema_legacy_k | 228.3% | 27.4% | 0.70 | -25.1% | 36.1% | 61.0% | 7.7 |
| Combined_Frequency | 228.2% | 27.3% | 0.83 | -20.4% | 30.6% | 61.0% | n/a |
| tradable_ema_top_10 | 223.3% | 27.0% | 0.73 | -26.2% | 34.3% | 59.3% | 10.0 |
| Combined_Equal | 195.1% | 24.6% | 0.76 | -20.2% | 29.6% | 55.9% | n/a |
| tradable_ema_top_7 | 178.6% | 23.2% | 0.54 | -39.9% | 39.1% | 57.6% | 7.0 |
| tradable_ema_top_5 | 99.0% | 15.0% | 0.33 | -51.2% | 38.9% | 59.3% | 5.0 |
| SPY | 82.0% | 13.0% | 0.69 | -23.9% | 15.8% | 64.4% | n/a |

Lecture :

- en rendement brut, la regression EMA est beaucoup meilleure que ce que le KPI
  de recomposition pouvait laisser penser ;
- `tradable_ema_legacy_k` fait quasiment jeu egal avec `Combined_Frequency` en
  rendement total, mais avec un Sharpe plus faible et un drawdown plus profond ;
- la meilleure regle autonome simple est `tradable_ema_top_10` : elle bat
  `Combined_Equal` en rendement total et reste proche de `Combined_Frequency`,
  mais avec plus de risque ;
- `top_5` est trop concentre : le rendement baisse fortement et le drawdown
  devient mauvais ;
- conclusion actuelle : le signal est tradable, mais il faut travailler la
  construction portefeuille / risque avant de parler d'allocation solide.

#### Test technique complet du 2026-06-26 / 2026-06-27

Construit dans :

- training : `outputs/tradable_technical_regression_optuna_20260626_212142`
- backtest technique :
  `outputs/tradable_technical_trading_backtest_20260627_012929`
- backtest EMA-only comparable :
  `outputs/tradable_ema_trading_backtest_20260627_012929`

Variables ajoutees :

- `price_roc_*`
- `ema_ratio_*`
- `price_to_ema_*`
- `rsi_*`
- `rsi_ratio_*`
- `bollinger_*`
- `stoch_*`
- `dist_to_*_high`
- `dist_to_*_low`
- `range_position_*`
- `volatility_*`
- `volatility_ratio_*`

Pour chaque variable technique, le run ajoute aussi rang mensuel, z-score
mensuel, flag top quartile et flag bottom quartile. Total final : 283 variables
techniques. Les fondamentaux et les sorties Legacy restent exclus.

Comparaison recomposition sur la periode courte commune :

| modele | periode | actions communes | actions Legacy | recomposition |
|---|---|---:|---:|---:|
| EMA-only `outputs/tradable_ema_regression_optuna_20260621_001753` | `2024-06` a `2026-04` | 68 | 169 | 40.2% |
| technical full `outputs/tradable_technical_regression_optuna_20260626_212142` | `2024-06` a `2026-04` | 55 | 169 | 32.5% |

Comparaison backtest trading sur la meme periode courte :

| modele | rendement total | CAGR | Sharpe | max drawdown | vol annualisee |
|---|---:|---:|---:|---:|---:|
| EMA-only top 10 | 165.1% | 66.3% | 1.67 | -13.5% | 38.4% |
| EMA-only legacy K | 152.4% | 62.1% | 1.57 | -12.5% | 38.3% |
| EMA-only top 5 | 137.9% | 57.2% | 1.45 | -11.0% | 38.1% |
| Combined_Frequency | 136.4% | 56.7% | 1.43 | -19.6% | 38.2% |
| technical top 5 | 125.8% | 53.0% | 1.07 | -19.9% | 47.7% |
| Combined_Equal | 117.4% | 50.0% | 1.37 | -17.5% | 35.1% |
| technical legacy K | 72.0% | 32.7% | 0.67 | -21.1% | 45.7% |
| technical top 10 | 53.8% | 25.2% | 0.64 | -18.0% | 36.4% |
| SPY | 38.7% | 18.6% | 1.36 | -7.6% | 12.2% |

Lecture :

- l'intuition "plus de granularite technique va ameliorer" ne se verifie pas
  dans ce test brut ;
- EMA-only reste meilleur que technical-full a la fois en recomposition et en
  backtest trading ;
- le technical-full apporte surtout du bruit ou de l'instabilite : la volatilite
  augmente et les meilleurs scenarios restent sous EMA-only ;
- decision : ne pas lancer le run large technical-full brut. La prochaine piste
  defendable est de tester les familles techniques separement, puis de faire un
  gating/stacking seulement sur les familles qui battent EMA-only hors test.

#### Test residual EMA + technique non-EMA du 2026-06-27

Construit dans :

- script : `scripts/experiments/run_tradable_ema_residual_regression.py`
- run : `outputs/tradable_ema_residual_regression_20260627_020253`
- backtest shrinkage `0.25` :
  `outputs/ema_plus_residual_0_25_trading_backtest_20260627_020626`
- backtest shrinkage `0.50` :
  `outputs/ema_plus_residual_0_50_trading_backtest_20260627_020654`
- backtest base EMA fixe du meme run :
  `outputs/ema_base_trading_backtest_20260627_020633`

Question testee :

```text
Est-ce qu'un modele technique peut apprendre seulement ce que le modele EMA-only
ne predit pas ?
```

Ce qui est regresse :

1. modele de base :
   `future_excess_return` clippe a `[-30%, +30%]` ;
2. modele residual :
   `future_excess_return_clippe - prediction_EMA_base` ;
3. score final :
   `prediction_EMA_base + shrinkage * prediction_residuelle`.

Variables utilisees :

- base EMA : memes 88 variables que `tradable_ema_regression_optuna`
  (16 EMA de base + rangs mensuels + z-scores mensuels + flags quartile +
  agregats horizontaux) ;
- residu : familles techniques hors EMA, donc `price_roc_*`, `rsi_*`,
  `rsi_ratio_*`, `bollinger_*`, `stoch_*`, `dist_to_*`,
  `range_position_*`, `volatility_*`, `volatility_ratio_*`, plus rangs
  mensuels, z-scores mensuels, flags quartile et agregats horizontaux ;
- pas de target Legacy, pas de features atomiques Legacy, pas de variable
  future non tradable.

Ensemble de test :

- meme periode courte que le test technical : `2024-06` a `2026-04` ;
- 23 mois avec panier Legacy ;
- 169 actions Legacy a recomposer ;
- 11 513 lignes de predictions.

Recomposition Legacy :

| modele | actions communes | actions Legacy | recomposition | mediane mensuelle |
|---|---:|---:|---:|---:|
| EMA base fixe du run residual | 61 | 169 | 36.1% | 33.3% |
| EMA + 25% residu technique | 69 | 169 | 40.8% | 40.0% |
| EMA + 50% residu technique | 66 | 169 | 39.1% | 42.9% |
| EMA + 100% residu technique | 45 | 169 | 26.6% | 25.0% |

Comparaison a l'ancien meilleur court :

| modele | actions communes | actions Legacy | recomposition |
|---|---:|---:|---:|
| ancien EMA Optuna par fold `outputs/tradable_ema_regression_optuna_20260621_001753` | 68 | 169 | 40.2% |
| EMA + 25% residu technique | 69 | 169 | 40.8% |

Lecture recomposition :

- apprendre le residu technique aide legerement le KPI demande : `+1` action
  commune vs le meilleur run EMA court existant ;
- le residu doit etre fortement shrinke. A `100%`, il detruit le ranking Legacy ;
- le gain est trop petit pour appeler ca une rupture. C'est un signal faible,
  pas une nouvelle base de production.

Metrices de prediction sur le futur rendement relatif :

| modele | RMSE clippee | Pearson clippe | Spearman global | Spearman mensuel moyen | top 10 futur excess moyen |
|---|---:|---:|---:|---:|---:|
| EMA base fixe | 8.177% | 3.04% | 0.10% | 1.52% | 3.84% |
| EMA + 25% residu | 8.172% | 3.80% | 1.64% | 1.30% | 2.20% |
| EMA + 50% residu | 8.168% | 4.07% | 2.17% | 0.98% | 1.90% |
| EMA + 100% residu | 8.166% | 4.07% | 2.71% | 0.66% | 1.72% |

Lecture prediction :

- oui, en metrique globale de regression, ajouter le residu technique ameliore
  legerement la prediction ;
- non, cette amelioration globale ne se traduit pas automatiquement par une
  meilleure selection mensuelle ;
- plus le residu est fort, plus la RMSE globale baisse, mais plus le top 10
  mensuel se degrade. La metrique globale voit un petit signal diffus, alors que
  le portefeuille depend de l'extreme top du classement.

Comparaison backtest trading sur la meme periode :

| modele | rendement total | CAGR | Sharpe | max drawdown |
|---|---:|---:|---:|---:|
| EMA base fixe top 7 | 232.0% | 87.0% | 1.88 | -8.2% |
| EMA base fixe legacy K | 219.7% | 83.4% | 1.84 | -7.5% |
| EMA base fixe top 10 | 206.2% | 79.3% | 2.03 | -7.3% |
| EMA + 50% residu top 5 | 219.0% | 83.2% | 1.88 | -18.0% |
| ancien EMA Optuna court top 10 | 165.1% | 66.3% | 1.67 | -13.5% |
| EMA + 25% residu top 5 | 125.3% | 52.8% | 1.33 | -19.2% |
| Combined_Frequency | 136.4% | 56.7% | 1.43 | -19.6% |
| SPY | 38.7% | 18.6% | 1.36 | -7.6% |

Lecture trading :

- la base EMA fixe, issue du meilleur warm-start large, backteste mieux que le
  run EMA Optuna court qui optimisait la recomposition par fold ;
- le residu `0.25` est le meilleur pour recomposer Legacy, mais pas pour trader ;
- le residu `0.50` peut etre interessant sur top 5, mais il augmente fortement
  le drawdown et degrade top 7/top 10 ;
- decision : ne pas remplacer la base EMA par le residual. La piste utile est
  plutot un gating : utiliser le residu technique seulement pour departager les
  premiers noms EMA ou construire un top 5 satellite, pas pour reranker tout
  l'univers.

### `ema_rich_*` future-target, run long

Construit le 2026-06-15 dans :

- script : `scripts/experiments/run_ema_rich_future_target_models.py`
- run verifie : `outputs/ema_rich_future_target_20260615_201243`

But :

Tester l'hypothese suivante sans tricher avec la target Legacy :

```text
plus de transformations EMA + apprentissage du futur excess return
-> meilleur signal de selection
-> recouvrement partiel des trades Legacy
```

KPI unique :

```text
nombre d'actions communes entre modele et Legacy
/
nombre d'actions choisies par Legacy ce mois-la
```

Selection :

- pour chaque mois, le modele choisit le meme nombre de tickers que Legacy ;
- Legacy sert uniquement au calcul de recomposition ;
- aucun modele n'apprend la target `legacy_selected`.

Periode mesuree :

- holdings Legacy de `2010-02` a `2026-04` ;
- 195 mois avec panier Legacy disponible ;
- 2 070 lignes Legacy a recomposer.

Construction commune des features :

- features de base : les 16 features EMA existantes ;
- features ajoutees :
  - rang mensuel de chaque EMA ;
  - z-score mensuel de chaque EMA ;
  - flag top quartile mensuel de chaque EMA ;
  - agregats horizontaux `ema_rank_mean`, `ema_rank_max`, `ema_z_mean`,
    `ema_z_max`, `ema_top25_vote_count` ;
- aucune target Legacy dans l'entrainement ;
- Legacy est utilise uniquement apres coup pour mesurer le recouvrement.

Modeles testes :

- `ema_signal_count` : score deterministe = nombre de signaux EMA dans le top
  quartile mensuel ;
- `future_excess_regression` : regression de `future_excess_return` clippe entre
  `-30%` et `+30%` ;
- `future_excess_classifier_gt0` : classification `future_excess_return > 0%` ;
- `future_excess_classifier_gt5` : classification `future_excess_return > 5%` ;
- `future_excess_rank_pairwise` : ranking mensuel du futur excess return.

Resultats recomposition :

| modele | actions communes | actions Legacy | recomposition |
|---|---:|---:|---:|
| future_excess_rank_pairwise | 492 | 2 070 | 23.8% |
| future_excess_classifier_gt5 | 297 | 2 070 | 14.3% |
| ema_signal_count | 240 | 2 070 | 11.6% |
| future_excess_classifier_gt0 | 105 | 2 070 | 5.1% |
| future_excess_regression | 73 | 2 070 | 3.5% |

Lecture :

- l'hypothese "on n'a pas assez d'EMA" est probablement vraie ;
- le nombre de signaux EMA seul aide, mais ne suffit pas ;
- apprendre `future_excess_return > 5%` fait mieux que `future_excess_return > 0%` ;
- la regression directe du rendement relatif futur est mauvaise pour retrouver
  les paniers Legacy ;
- le ranking mensuel du futur rendement relatif est la meilleure methode testee,
  mais 23.8% de recomposition reste insuffisant.

### Decomposition atomique Legacy

Construit le 2026-06-16 dans :

- script : `scripts/experiments/run_legacy_atomic_recomposition.py`
- run verifie : `outputs/legacy_atomic_recomposition_20260616_122045`

But :

Verifier si le plafond a 23.8% vient du boosting ou du fait que les features ML
ne contiennent pas les briques Legacy exactes.

Principe :

- cible : union des paniers `Combined_Equal` et `Combined_Frequency` ;
- candidats : les quatre blocs `Legacy_Optuna_11`, `Legacy_Optuna_12`,
  `Legacy_Optuna_21`, `Legacy_Optuna_22` ;
- unite atomique : `portfolio_model + selected_model`, ou `selected_model`
  contient le couple EMA, le nombre cible d'actions et la limite secteur ;
- KPI : actions communes / actions Legacy.

Resultats par bloc Optuna :

| bloc | actions communes | actions Legacy | recomposition |
|---|---:|---:|---:|
| union des 4 blocs Optuna | 2 088 | 2 088 | 100.0% |
| Legacy_Optuna_21 | 1 598 | 2 088 | 76.5% |
| Legacy_Optuna_11 | 1 497 | 2 088 | 71.7% |
| Legacy_Optuna_12 | 1 416 | 2 088 | 67.8% |
| Legacy_Optuna_22 | 1 338 | 2 088 | 64.1% |

Meilleurs modeles atomiques observes :

| modele atomique | periode | actions communes | actions Legacy | recomposition |
|---|---:|---:|---:|---:|
| Optuna_22, EMA short=95 / long=71, 16 actions, secteur 2 | 2011-02 a 2012-01 | 120 | 120 | 100.0% |
| Optuna_12, EMA short=95 / long=71, 16 actions, secteur 2 | 2011-02 a 2012-01 | 120 | 120 | 100.0% |
| Optuna_12, EMA short=54 / long=162, 30 actions, secteur 2 | 2018-02 a 2019-01 | 145 | 145 | 100.0% |
| Optuna_22, EMA short=54 / long=162, 30 actions, secteur 2 | 2018-02 a 2019-01 | 145 | 145 | 100.0% |
| Optuna_11, EMA short=95 / long=72, 30 actions, secteur 2 | 2010-02 a 2014-01 | 419 | 424 | 98.8% |

Lecture :

- atteindre 50% est possible quand on redescend au niveau des briques Legacy ;
- le meilleur modele ML actuel ne voit pas ces briques exactes, donc il plafonne
  a 23.8% ;
- la prochaine etape n'est pas plus de trials Optuna sur les features actuelles,
  mais la generation des features atomiques Legacy exactes dans le frame ML.

### Frame atomique + apprentissage du futur rendement

Construit le 2026-06-16 dans :

- builder : `scripts/experiments/build_legacy_atomic_feature_frame.py`
- training : `scripts/experiments/run_atomic_feature_future_target_models.py`
- frame : `outputs/legacy_atomic_feature_frame_20260616_195618`
- run training : `outputs/atomic_feature_future_target_20260616_200236`

But :

Tester si un modele qui apprend le futur rendement relatif peut retrouver Legacy
quand les signaux atomiques exacts sont enfin visibles dans les features.

Features ajoutees au `model_frame` :

- `legacy_atomic_vote_count` : nombre de blocs Optuna Legacy qui selectionnent le
  ticker ce mois-la ;
- quantiles et ratios `mtr` atomiques max/moyens ;
- flags par bloc `Legacy_Optuna_11`, `12`, `21`, `22` ;
- rangs mensuels des signaux atomiques principaux.

Controle deterministe sur le frame :

| score | actions communes | actions Legacy | recomposition |
|---|---:|---:|---:|
| legacy_atomic_vote_count | 2 070 | 2 070 | 100.0% |
| legacy_atomic_max_quantile_mtr | 2 070 | 2 070 | 100.0% |

Training sur futur rendement relatif :

| modele | target d'entrainement | actions communes | actions Legacy | recomposition |
|---|---|---:|---:|---:|
| atomic_regression | `future_excess_return` clippe | 2 070 | 2 070 | 100.0% |
| atomic_classifier_gt5 | `future_excess_return > 5%` | 2 041 | 2 070 | 98.6% |
| atomic_classifier_gt0 | `future_excess_return > 0%` | 1 100 | 2 070 | 53.1% |
| atomic_plus_ema_rank_pairwise | ranking mensuel futur rendement relatif | 409 | 2 070 | 19.8% |
| atomic_rank_pairwise | ranking mensuel futur rendement relatif | 41 | 2 070 | 2.0% |

Lecture :

- oui, `atomic_regression` retrouve toutes les actions Legacy sur tous les mois
  testes : minimum mensuel = 100%, mediane mensuelle = 100%, 0 mois sous 100% ;
- on depasse l'objectif 50% des que les features atomiques exactes sont dans le
  frame ;
- la classification `>5%` et la regression recuperent presque exactement Legacy,
  sans target `legacy_selected`, parce que les features portent le signal Legacy
  lui-meme ;
- ce n'est pas un candidat final sain : les variables atomiques viennent des
  briques Legacy exactes, donc ce run doit etre lu comme un plafond de
  replication / diagnostic de representation, pas comme une generalisation ;
- ne pas utiliser cette famille pour produire un portefeuille tradable ;
- la prochaine etape doit separer deux usages :
  1. mode "replication Legacy" : utiliser les features atomiques exactes ;
  2. mode "generalisation" : generer des couples EMA candidats hors winners
     Legacy et verifier si le futur rendement relatif choisit les memes familles
     de signaux.

### Generalisation par experts EMA

Construit le 2026-06-16 dans :

- builder : `scripts/experiments/build_generalized_ema_expert_frame.py`
- training : `scripts/experiments/run_generalized_ema_expert_models.py`
- sweep : `scripts/experiments/sweep_generalized_ema_expert_params.py`
- frame : `outputs/generalized_ema_expert_frame_20260616_231854`
- run training : `outputs/generalized_ema_expert_models_20260616_232041`

But :

Sortir des features `legacy_atomic_*`, donc ne plus utiliser les selections
Legacy comme signal direct. Les experts EMA sont des regles parametriques
`short/long/n_actions/secteur`, puis ils sont selectionnes chaque mois selon
leur performance passee en futur rendement relatif.

Premier perimetre :

- 44 experts observes dans Legacy ;
- scoring par performance moyenne passee sur 36 mois ;
- minimum de 6 mois d'historique ;
- top 10 experts actifs par mois ;
- features ticker = votes/scores des experts actifs qui selectionnent le ticker.

Resultats recomposition :

| modele | actions communes | actions Legacy | recomposition |
|---|---:|---:|---:|
| vote des experts EMA appris sur passe | 1 134 | 2 070 | 54.8% |
| somme des scores experts appris sur passe | 1 101 | 2 070 | 53.2% |
| classifieur `future_excess_return > 5%` | 856 | 2 070 | 41.4% |
| classifieur `future_excess_return > 0%` | 552 | 2 070 | 26.7% |
| ranking mensuel futur rendement relatif | 41 | 2 070 | 2.0% |
| regression futur rendement relatif | 41 | 2 070 | 2.0% |

Lecture :

- l'objectif 50% est atteint sans utiliser les sorties atomiques Legacy comme
  features ;
- le score deterministe d'experts appris sur le passe bat les modeles XGBoost
  supervises dans ce premier run ;
- le prochain levier n'est pas plus de tuning XGBoost, mais l'elargissement de
  l'univers d'experts : voisins des couples EMA observes, puis echantillonnage
  plus large de l'espace `n_short=1..100`, `n_long=50..400`.

#### Tuning generalisable du 2026-06-16 soir

Deux axes ont ete testes ensuite :

- garder seulement les 44 experts EMA observes dans Legacy, mais faire varier la
  memoire de performance passee et le nombre d'experts actifs ;
- generer 356 voisins EMA autour des couples observes avec les deltas
  `n_short +/- 10` et `n_long +/- 40`.

Runs :

- observes, sweep : `outputs/generalized_ema_expert_sweep_20260616_234440`
- voisins, sweep : `outputs/generalized_ema_expert_sweep_20260616_234500`
- meilleur frame observe : `outputs/generalized_ema_expert_frame_20260616_234521`
- training meilleur frame observe :
  `outputs/generalized_ema_expert_models_20260616_234556`

Meilleurs resultats deterministes :

| perimetre candidats | memoire | experts actifs | score | actions communes | actions Legacy | recomposition | mediane mensuelle |
|---|---:|---:|---|---:|---:|---:|---:|
| observes Legacy | 60 mois | 20 | somme des scores experts | 1 254 | 2 070 | 60.6% | 70.0% |
| observes Legacy | 60 mois | 10 | vote des experts | 1 199 | 2 070 | 57.9% | 66.7% |
| voisins EMA | 36 mois | 50 | vote des experts | 1 111 | 2 070 | 53.7% | 63.6% |
| voisins EMA | 60 mois | 50 | vote des experts | 1 100 | 2 070 | 53.1% | 62.5% |

Training sur le meilleur frame observe (`60 mois`, `20 experts`) :

| modele | actions communes | actions Legacy | recomposition | mediane mensuelle |
|---|---:|---:|---:|---:|
| somme des scores experts EMA | 1 254 | 2 070 | 60.6% | 70.0% |
| vote des experts EMA | 1 231 | 2 070 | 59.5% | 70.0% |
| classifieur `future_excess_return > 5%` | 872 | 2 070 | 42.1% | 54.5% |
| classifieur `future_excess_return > 0%` | 790 | 2 070 | 38.2% | 42.9% |
| ranking mensuel futur rendement relatif | 41 | 2 070 | 2.0% | 0.0% |
| regression futur rendement relatif | 41 | 2 070 | 2.0% | 0.0% |

Lecture :

- la meilleure piste generalisable actuelle est le score deterministe d'experts
  EMA observes, selectionnes sur 60 mois de performance passee ;
- ajouter des voisins EMA de facon brute ne suffit pas : l'univers grossit mais
  le signal se dilue ;
- mlcraft/XGBoost n'ameliore pas encore cette representation : le classifieur
  `>5%` retombe a 42.1%, donc le boosting apprend une partie du futur rendement
  mais ne choisit pas les memes trades que le score expert ;
- prochaine piste defendable : ne pas ajouter tous les voisins, mais apprendre
  ou selectionner des familles de voisins qui ont elles-memes une performance
  passee stable avant d'entrainer le modele final.

### `ema_residual_benchmark`

Construit le 2026-06-14 dans `780af35`, puis reutilise comme benchmark dans
`f7d1b29`.

Construction :

- features : `ema_features` uniquement ;
- target : `future_excess_return > 5%` ;
- modele de base XGBoost ;
- predictions OOF du modele de base ;
- residual XGBoost avec `base_margin = logit(base_prediction_oof)`.

Pourquoi ce benchmark est important :

- il a montre que le signal EMA est beaucoup plus proche de legacy que les
  features SHAP/volatilite ;
- il reste moins bon que `distill_legacy` pour copier les trades.

### `blend_ema_distill`

Construit le 2026-06-14 dans `f7d1b29`.

Construction :

```text
blend_ema_distill = zscore(ema_residual_benchmark) + zscore(distill_legacy)
```

Pourquoi ce controle est important :

- il teste si combiner "copie legacy" et "momentum EMA" ajoute de la stabilite ;
- il est presque aussi performant que `distill_legacy` ;
- il a le meilleur overlap legacy moyen en top 10 fixe.

## Ancien controle top 10 allocation

Run : `outputs/signal_copy_models_20260614_214711`.

Cette section est conservee comme historique. Elle ne doit pas etre utilisee
comme juge principal du travail actuel.

| modele | total return | actif compose | overlap legacy moyen | hit-rate +5% |
|---|---:|---:|---:|---:|
| distill_legacy | +100.3% | +67.5% | 43.6% | 44.5% |
| blend_ema_distill | +99.1% | +66.5% | 46.4% | 45.5% |
| ema_residual_benchmark | +61.4% | +33.3% | 16.4% | 34.5% |
| rank_pairwise | +45.9% | +20.1% | 19.1% | 34.5% |
| regression_excess | +33.9% | +9.6% | 18.2% | 40.0% |
| two_stage_ema_full | +29.0% | +5.8% | 11.8% | 31.8% |
| monotone_ema_full | +20.7% | -0.7% | 9.1% | 34.5% |
| weighted_top_classifier | +16.8% | -4.1% | 0.9% | 36.4% |
| gated_full_after_ema | +14.8% | -5.7% | 9.1% | 32.7% |

Lecture :

- `distill_legacy` gagne.
- `blend_ema_distill` est tres proche et a un meilleur overlap top 10 moyen.
- `ema_residual_benchmark` reste un bon benchmark ML.
- les autres modeles apprennent parfois du rendement, mais ne copient pas legacy.

## Ancien controle de recomposition

Run : `outputs/legacy_clone_diagnostics_20260614`.

Selection clone :

```text
Pour chaque mois :
  K = nombre exact de trades legacy
  selection modele = top K du score modele
```

| modele | recomposition | jaccard | trades retrouves | clone return | actif compose |
|---|---:|---:|---:|---:|---:|
| distill_legacy | 53.5% | 37.2% | 38 / 71 | +143.8% | +104.4% |
| blend_ema_distill | 50.7% | 34.8% | 36 / 71 | +142.2% | +103.1% |
| rank_pairwise | 16.4% | 9.4% | 12 / 71 | +48.9% | +22.6% |
| ema_residual_benchmark | 15.0% | 8.8% | 11 / 71 | +16.8% | -4.5% |
| two_stage_ema_full | 10.9% | 6.3% | 8 / 71 | +18.7% | -2.6% |
| gated_full_after_ema | 5.5% | 3.0% | 4 / 71 | +10.7% | -9.2% |
| regression_excess | 5.2% | 2.9% | 4 / 71 | +31.5% | +7.9% |
| monotone_ema_full | 4.2% | 2.4% | 3 / 71 | +11.9% | -8.3% |
| weighted_top_classifier | 0.0% | 0.0% | 0 / 71 | +9.5% | -10.4% |

Lecture :

- `distill_legacy` est la seule vraie piste de clonage.
- `blend_ema_distill` est presque au meme niveau.
- objectif court terme historique : passer de 53.5% a plus de 70% de
  recomposition.

## Quelle metrique dit qu'on est bon ?

### Metrique primaire pour le projet actuel

```text
nombre d'actions communes entre modele et Legacy
/
nombre d'actions choisies par Legacy ce mois-la
```

Pourquoi :

- on veut prouver qu'on reproduit les trades legacy ;
- le rendement peut mentir sur une petite fenetre ;
- l'AUC mesure le classement global, pas la reproduction du panier.

Interpretation actuelle :

- `< 50%` : pas un clone.
- `50%-70%` : piste serieuse.
- `> 70%` : clone exploitable.
- `> 80%` : clone probablement assez solide pour commencer la generalisation.

### Metriques secondaires de clonage

- `jaccard@legacy_k`
- `rank moyen des noms legacy`
- `precision@legacy_k`
- stabilite mensuelle de la recomposition
- erreur de poids si on apprend `legacy_weight_normalized`

### Metriques allocation

Une fois le clone correct :

- `active_compounded` vs SPY ;
- `active_max_drawdown` ;
- `worst_active_month` ;
- `avg_top10_excess` ;
- `hit-rate +5%`.

### Metriques a ne pas utiliser comme juge principal

- AUC seule ;
- logloss seule ;
- proba absolue `> 0.5`.

Les probas absolues sont mal calibrees pour ce probleme. Le bon seuil est
mensuel et relatif : top K ou top N.

## Correction de cap 2026-06-27

Clarification importante apres retour utilisateur :

Le but final n'est pas de recopier Legacy ni de produire une version simplifiee
de Legacy. Le but est :

```text
estimer le rendement relatif futur moyen de chaque action
-> construire un portefeuille controle a partir de ces estimations
-> comparer la performance et le risque a Legacy
```

La recomposition Legacy reste seulement un diagnostic de proximite de famille de
signal. Elle ne doit pas etre l'objectif final et elle ne doit pas remplacer la
question centrale :

```text
quand le modele predit mieux une action, est-ce que cette action rapporte
effectivement plus en moyenne le mois suivant vs SPY ?
```

Tout score deterministe EMA documente ci-dessous doit donc etre lu comme un
temoin explicatif, pas comme une solution d'allocation.

## Diagnostic 2026-06-27 : ce que Legacy achete vraiment

Run principal :

```text
outputs/legacy_factor_exposure_20260627_154437
```

Script :

```text
scripts/experiments/analyze_legacy_factor_exposures.py
```

Fenetre :

- `2015-01` a `2026-04`;
- `Combined_Frequency`;
- 136 mois ;
- 1 258 positions Legacy ;
- 9.2 positions par mois en moyenne, avec un minimum de 5 et un maximum de 22.

Lecture performance :

| mesure | valeur |
|---|---:|
| Rendement mensuel Legacy equal-weight | `1.9%` |
| Excess return mensuel Legacy equal-weight | `0.7%` |
| Excess return mensuel univers equal-weight | `-0.2%` |
| Lift Legacy vs univers en excess return | `0.9%` |
| Positions Legacy avec excess return positif | `51.9%` |
| Univers avec excess return positif | `48.2%` |

Lecture signal :

| feature tradable | rang moyen des actions Legacy | lift vs univers | IC futur excess return |
|---|---:|---:|---:|
| `ema_ratio_2_12` | `0.970` | `0.469` | `0.013` |
| `ema_ratio_3_12` | `0.969` | `0.468` | `0.014` |
| `ema_ratio_3_6` | `0.966` | `0.465` | `0.012` |
| `technical_z_mean` | `0.965` | `0.464` | `0.014` |
| `price_to_ema_12` | `0.961` | `0.460` | `0.013` |

Conclusion :

- Legacy achete presque toujours l'extreme haut des rangs EMA / momentum court
  contre moyen terme.
- Le signal est tres lisible en cross-section, mais son IC futur mensuel est
  faible. Ca explique pourquoi une regression globale du futur excess return
  peut predire un peu mieux en RMSE tout en selectionnant moins bien les tout
  premiers noms.
- Le bon modele ne doit pas juste predire la moyenne conditionnelle. Il doit
  preserver le signal d'extreme ranking, puis apprendre a filtrer ou ponderer
  ces extremes pour reduire volatilite et drawdown.

Secteurs et concentration :

- secteurs les plus presents : Technology, Consumer Cyclical, Industrials,
  Healthcare, Financial Services ;
- Technology est le principal moteur positif visible : 210 positions, excess
  return moyen `2.4%` ;
- tickers les plus frequents : `NVDA.US` 61 mois, `NFLX.US` 38 mois,
  `LRCX.US` 25 mois, `NEM.US` 21 mois, `ALGN.US` 20 mois.

Blocs EMA atomiques les plus presents dans Legacy 2015+ :

- `159.0-54.0-30|asset=18|sector=2`;
- `162.0-54.0-30|asset=30|sector=2`;
- `172.0-46.0-30|asset=11|sector=2`;
- `260.0-71.0-30|asset=5|sector=2`;
- `112.0-49.0-30|asset=5|sector=2`;
- `251.0-37.0-30|asset=5|sector=2`.

Ces fenetres doivent inspirer le prochain generateur EMA, mais pas etre
utilisees comme cible Legacy.

## Temoins deterministes 2015+ non-solutions

Runs :

```text
outputs/deterministic_signal_predictions_20260627_154617
outputs/ema_ratio_2_12_rank_month_trading_backtest_20260627_154632
outputs/ema_ratio_3_12_rank_month_trading_backtest_20260627_154632
outputs/technical_z_mean_trading_backtest_20260627_154632
outputs/technical_rank_mean_trading_backtest_20260627_154632
```

Ces temoins ne sont pas des modeles entraines et ne repondent pas au but final.
Ils servent seulement a verifier si un score simple, calculable au mois de
decision, explique une partie de Legacy et tient en backtest.

Table performance principale :

| score | selection | recomposition stricte | total return | CAGR | Sharpe | max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| `ema_ratio_3_12_rank_month` | top 5 | `404 / 1 258 = 32.1%` | `+1 498.1%` | `27.7%` | `0.75` | `-33.0%` |
| `ema_ratio_2_12_rank_month` | legacy K | `647 / 1 258 = 51.4%` | `+1 291.0%` | `26.1%` | `0.78` | `-27.2%` |
| `ema_ratio_3_12_rank_month` | legacy K | `655 / 1 258 = 52.1%` | `+883.7%` | `22.4%` | `0.68` | `-26.8%` |
| `technical_z_mean` | legacy K | `580 / 1 258 = 46.1%` | `+1 154.1%` | `25.0%` | `0.76` | `-34.7%` |
| `technical_rank_mean` | legacy K | `528 / 1 258 = 42.0%` | `+1 147.4%` | `24.9%` | `0.92` | `-31.7%` |
| `Combined_Frequency` | Legacy | reference | `+1 133.7%` | `24.8%` | `0.93` | `-23.5%` |
| `Combined_Equal` | Legacy | reference | `+847.4%` | `21.9%` | `0.84` | `-23.6%` |
| `SPY` | benchmark | reference | `+319.4%` | `13.5%` | `0.76` | `-23.9%` |

Lecture :

- Le seuil de 50% de recomposition est atteint sans triche par un score EMA
  deterministe : `ema_ratio_3_12_rank_month` en K dynamique Legacy.
- Certains scores simples battent Legacy en rendement total/CAGR, mais pas en
  controle du risque. Le meilleur rendement est plus volatil et plus profond en
  drawdown.
- `technical_rank_mean` est interessant pour le controle : rendement comparable
  a `Combined_Frequency`, Sharpe proche, mais drawdown encore trop eleve.
- Les regressions boosting propres testees avant ne doivent pas etre promues :
  elles apprennent bien une partie du futur excess return, mais perdent le
  comportement extreme-rank qui fait Legacy.

Decision R&D apres ce diagnostic :

1. Construire une regression/ranking qui part d'un score EMA extreme-rank comme
   score de base, puis apprend un ajustement sur futur excess return.
2. Optimiser Optuna sur validation future seulement, avec une metrique de
   portefeuille, pas sur Legacy : rendement top K, Sharpe mensuel ou penalite
   drawdown/volatilite.
3. Ajouter ensuite des contraintes d'allocation pour construire un portefeuille
   plus controle : max overlap Legacy, max secteur, volatilite realisee, et
   limite de concentration ticker.
4. Garder le KPI Legacy comme diagnostic de famille de signal, pas comme
   objectif d'optimisation.

## Calibration des vraies predictions boosting 2015+

Run principal :

```text
outputs/return_forecast_calibration_20260627_161954
```

Script :

```text
scripts/experiments/analyze_return_forecast_calibration.py
```

Objectif :

- prendre les vrais runs boosting propres qui predisent `future_excess_return`;
- separer chaque mois les actions en 10 bins selon le score predit ;
- verifier si le meilleur bin rapporte plus que le pire bin ;
- mesurer les top K sans utiliser Legacy comme objectif.

Runs analyses :

- `outputs/tradable_ema_regression_optuna_20260627_121144`
- `outputs/tradable_ema_regression_optuna_20260627_125020`
- `outputs/tradable_ema_regression_optuna_20260627_133211`
- `outputs/tradable_ema_regression_optuna_20260627_140859`
- `outputs/tradable_ema_regression_optuna_20260627_144652`

Synthese :

| run | objectif Optuna | corr decile/rendement | top decile excess | bottom decile excess | top-bottom | meilleur top K | excess top K |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260627_121144` | rendement top 10 validation | `-0.0368` | `0.00%` | `0.06%` | `-0.06%` | `50` | `-0.09%` |
| `20260627_125020` | precision top 10 validation | `-0.0772` | `-0.38%` | `-0.46%` | `0.08%` | `50` | `-0.05%` |
| `20260627_133211` | precision top 20 validation | `0.1455` | `-0.12%` | `0.47%` | `-0.59%` | `50` | `-0.09%` |
| `20260627_140859` | precision top 30 validation | `0.2977` | `-0.32%` | `-0.65%` | `0.32%` | `50` | `-0.15%` |
| `20260627_144652` | precision top 50 validation | `0.0038` | `0.12%` | `-0.12%` | `0.24%` | `30` | `0.00%` |

Conclusion dure :

- Ces regressions boosting propres ne quantifient pas encore assez bien le
  rendement moyen futur action par action.
- Il n'y a pas de relation monotone robuste entre le score predit et le
  rendement relatif realise.
- Les meilleurs top K ont un lift faible ou nul vs univers.
- Donc ces runs ne sont pas une base suffisante pour construire le portefeuille
  controle que l'on cherche.

Decision suivante :

1. arreter de juger principalement par recomposition Legacy ;
2. garder la recomposition seulement comme diagnostic secondaire ;
3. travailler le probleme comme un probleme de prevision de rendement relatif
   calibre ;
4. la prochaine experience doit optimiser directement une metrique portfolio sur
   validation temporelle : rendement moyen, Sharpe mensuel, drawdown ou
   penalite volatilite ;
5. ensuite seulement construire un portefeuille contraint avec les predictions.

## Candidat boosting allocation 2026-06-27

Runs principaux :

```text
outputs/portfolio_boosting_top_return_classifier_20260627_165036
outputs/portfolio_boosting_blend_backtest_20260627_171645
```

Scripts :

```text
scripts/experiments/run_portfolio_boosting_top_return_classifier.py
scripts/experiments/run_portfolio_boosting_blend_backtest.py
```

Ce qui est appris :

- modele `mlcraft` XGBoost classification ;
- target mensuelle : action dans le top 10% des `future_excess_return` du mois
  suivant ;
- features : 283 features techniques tradables deja disponibles dans la frame
  backtest, incluant EMA, prix/EMA, ROC, RSI, Bollinger, stochastic, distances
  high/low, volatilite, rangs/z-scores/flags mensuels ;
- Optuna : 1 trial par fold pour ce premier run complet, objectif validation =
  rendement moyen du top 5 ;
- walk-forward mensuel 2015-02 a 2026-04, 135 mois testes.

Point dur :

- le boosting pur ne marche pas encore : top 5 = CAGR `5.0%`, sous SPY et tres
  loin de Legacy ;
- le score utile est hybride : rang mensuel de la proba boosting + petit prior
  momentum technique.

Score hybride teste :

```text
score = (rank_month(prediction_boosting) + 0.10 * technical_z_mean) / 1.10
selection = top 5 actions par mois
```

Puis construction portefeuille :

```text
poids final = x% strategie hybride + (1 - x)% SPY
```

Comparaison sur la meme fenetre :

| modele | total return | CAGR | Sharpe | max drawdown | vol mensuelle | mois positifs |
|---|---:|---:|---:|---:|---:|---:|
| `boosting_momentum_top5_100pct` | `+2 121.6%` | `31.7%` | `0.86` | `-38.2%` | `10.0%` | `63.7%` |
| `boosting_momentum_top5_80pct_spy_20pct` | `+1 618.9%` | `28.8%` | `0.90` | `-33.4%` | `8.5%` | `63.7%` |
| `boosting_momentum_top5_70pct_spy_30pct` | `+1 392.2%` | `27.2%` | `0.93` | `-30.9%` | `7.8%` | `63.7%` |
| `boosting_momentum_top5_60pct_spy_40pct` | `+1 184.2%` | `25.5%` | `0.95` | `-28.5%` | `7.1%` | `62.2%` |
| `Combined_Frequency` | `+1 118.1%` | `24.9%` | `0.93` | `-23.5%` | `7.1%` | `63.0%` |
| `Combined_Equal` | `+851.1%` | `22.2%` | `0.85` | `-23.6%` | `6.9%` | `60.0%` |
| `SPY` | `+332.2%` | `13.9%` | `0.79` | `-23.9%` | `4.4%` | `69.6%` |

Lecture honnete :

- On a enfin un backtest boosting-based qui bat `Combined_Frequency` en total
  return, CAGR et Sharpe : le candidat `60% strategie hybride / 40% SPY`.
- Il ne bat pas Legacy sur le max drawdown : `-28.5%` vs `-23.5%`.
- Le modele pur ne suffit pas ; le prior momentum reste indispensable.
- Ce n'est donc pas encore la strategie finale, mais c'est le premier candidat
  allocation qui passe le seuil "mieux que Legacy" sur plusieurs metriques
  importantes sans optimiser Legacy.

Prochaine decision technique :

1. reduire le drawdown du candidat `60/40` sans perdre son avantage CAGR/Sharpe ;
2. tester un overlay de risque dynamique, pas fixe : baisse d'exposition quand
   les predictions top 5 sont peu separees du reste, quand la volatilite du
   panier est trop haute, ou quand le regime SPY est defavorable ;
3. relancer avec plus de trials Optuna et warm starts uniquement si la metrique
   d'allocation validation inclut le risque, pas seulement le rendement.

## Decision actuelle

Ne pas industrialiser les modeles suivants :

- `weighted_top_classifier`
- `gated_full_after_ema`
- `monotone_ema_full`
- `two_stage_ema_full` dans sa version actuelle

Garder comme benchmarks :

- `ema_residual_benchmark`
- full model proba brute
- `distill_legacy` comme teacher/oracle de diagnostic uniquement
- `blend_ema_distill` comme controle de plafond, pas comme algo final

Pousser en priorite :

- `ema_rich_rank_pairwise`
- un generateur de features EMA plus proche de Legacy
- des objectifs de ranking mensuel du futur excess return

## Prochaine construction recommandee

Construire d'abord un vrai frame de signaux EMA compatible Legacy :

```text
outputs/legacy_ema_signal_frame.parquet
```

Colonnes a produire :

- ratios court / long pour beaucoup plus de couples EMA ;
- `mtr` Legacy exact quand les prix bruts le permettent ;
- rang mensuel, quantile normalise mensuel et top-N flags ;
- nombre de votes EMA par ticker ;
- membership universe/stocks_filter au mois de decision.

Puis entrainer en priorite :

1. ranker mensuel sur `future_excess_return` ;
2. classifier `future_excess_return > 5%` seulement comme benchmark ;
3. regression du futur excess return seulement comme benchmark.

Le dataset teacher reste utile pour comprendre le plafond :

```text
outputs/legacy_teacher_frame.parquet
```

Colonnes cibles :

- `legacy_selected`
- `legacy_n_models`
- `legacy_weight_normalized`
- `legacy_rank_in_month`

Modeles teacher a entrainer :

1. classifier `legacy_selected`;
2. ranker/regresseur `legacy_n_models`;
3. regresseur `legacy_weight_normalized`;
4. blend final calibre par mois.

Objectif :

```text
recomposition > 70%
```

Mais l'objectif final ne doit pas etre d'apprendre `legacy_selected`. L'objectif
final est d'apprendre le futur excess return et de verifier que le signal appris
retrouve suffisamment les trades Legacy pour prouver que la famille de signaux
est correcte.

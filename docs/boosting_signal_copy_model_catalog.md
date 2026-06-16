# Catalogue central des modeles boosting / legacy-copy

Date de creation de cette doc : 2026-06-14.

Cette page est le point central pour se rappeler quels modeles ont ete testes,
comment ils sont construits, quand ils ont ete construits, et quelles metriques
doivent servir de juge.

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

- on depasse l'objectif 50% des que les features atomiques exactes sont dans le
  frame ;
- la classification `>5%` et la regression recuperent presque exactement Legacy,
  sans target `legacy_selected`, parce que les features portent le signal Legacy
  lui-meme ;
- ce n'est pas encore une preuve de generalisation hors Legacy : c'est un
  diagnostic de representation ;
- la prochaine etape doit separer deux usages :
  1. mode "replication Legacy" : utiliser les features atomiques exactes ;
  2. mode "generalisation" : generer des couples EMA candidats hors winners
     Legacy et verifier si le futur rendement relatif choisit les memes familles
     de signaux.

### Generalisation par experts EMA

Construit le 2026-06-16 dans :

- builder : `scripts/experiments/build_generalized_ema_expert_frame.py`
- training : `scripts/experiments/run_generalized_ema_expert_models.py`
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

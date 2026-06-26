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
| `tradable_technical_regression_optuna` | futur rendement relatif : meme regression, mais avec toutes les familles techniques disponibles | 283 variables techniques : ROC prix, EMA, RSI, Bollinger, stochastique, distance high/low, range position, volatilite, rangs/z-scores/flags/agregats mensuels | run court `2024-06` a `2026-04`, 23 mois, 169 lignes Legacy | `55 / 169 = 32.5%`; moins bon que EMA-only sur la meme periode |
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

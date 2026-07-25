# Rapport de performance propre — tous horizons

Date du recalcul : 2026-07-25.

Run valide :

`outputs/multihorizon_boosting/screening_clean_20260725`

Les runs `screening_20260725` et `shortlist_cpcv_20260725` ont été invalidés :
une colonne source `weight_normalized` issue de la jointure teacher Legacy était
restée dans les features numériques. Les résumés de l'incident sont conservés
sous `outputs/multihorizon_boosting/invalid_weight_leak_20260725/`, mais leurs
résultats ne doivent pas être utilisés.

Le run propre contient 334 features. Les colonnes `weight_normalized`,
`legacy_weight_normalized`, `legacy_selected` et `legacy_n_models` sont toutes
interdites aux modèles économiques.

## 1. Historique réellement utilisé

Le panel de features commence en janvier 2005 et se termine en avril 2026.
L'entraînement est croissant à partir de 2005. Chaque test est strictement
postérieur à l'entraînement et à la validation ; la maturité et la purge des
labels dépendent de l'horizon.

| Horizon cible | Premier mois de décision test | Dernier mois de décision test | Mois test | Folds annuels |
|---:|---:|---:|---:|---:|
| 1 mois | 2013-01 | 2025-12 | 156 | 13 |
| 3 mois | 2013-05 | 2025-04 | 144 | 12 |
| 6 mois | 2013-11 | 2025-10 | 144 | 12 |
| 12 mois | 2014-11 | 2024-10 | 120 | 10 |
| 24 mois | 2016-11 | 2023-10 | 84 | 7 |
| 36 mois | 2018-11 | 2022-10 | 48 | 4 |

Le signal est recalculé chaque mois, mais le booster est gelé sur chaque bloc
test de 12 mois. Les fins de période diffèrent car une cible à 24 ou 36 mois
exige autant de futur réalisé pour pouvoir être évaluée.

## 2. Métriques ML hors échantillon

### Classification du top décile futur

Le taux positif est proche de 10 %. La PR-AUC doit donc être comparée à une
baseline d'environ 0,10.

| Horizon | ROC-AUC | PR-AUC | Lift PR/baseline | Brier | Erreur calibration |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,634 | 0,164 | 1,62x | 0,0892 | 0,0044 |
| 3 | 0,633 | 0,167 | 1,65x | 0,0890 | 0,0060 |
| 6 | 0,620 | 0,165 | 1,63x | 0,0898 | 0,0114 |
| 12 | 0,614 | 0,152 | 1,50x | 0,0897 | 0,0106 |
| 24 | 0,624 | 0,149 | 1,47x | 0,0902 | 0,0109 |
| 36 | 0,600 | 0,142 | 1,40x | 0,0911 | 0,0203 |

Conclusion : les probabilités les plus utiles et les mieux calibrées sont à
1-3 mois. Le pouvoir discriminant reste modeste, mais réel.

### Régression du rendement relatif cumulé

| Horizon | RMSE | RMSE / écart-type cible | MAE | R² | IC Spearman mensuel |
|---:|---:|---:|---:|---:|---:|
| 1 | 7,35 % | 1,002 | 5,26 % | -0,003 | 0,011 |
| 3 | 12,39 % | 1,003 | 9,18 % | -0,005 | 0,028 |
| 6 | 18,16 % | 1,005 | 13,37 % | -0,010 | 0,032 |
| 12 | 25,41 % | 1,012 | 19,05 % | -0,025 | 0,059 |
| 24 | 37,89 % | 1,019 | 27,59 % | -0,038 | 0,096 |
| 36 | 46,53 % | 1,030 | 33,90 % | -0,060 | 0,049 |

Conclusion : aucune régression ne prédit correctement l'amplitude absolue des
rendements. Tous les R² sont négatifs. La régression 24 mois contient néanmoins
le meilleur signal d'ordre cross-sectionnel, avec un IC de 0,096.

### Ranking mensuel

| Horizon | IC Spearman | NDCG@10 | NDCG@10 sans signal | Lift NDCG | Overlap Legacy top-10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,001 | 0,514 | 0,505 | +0,010 | 11,01 % |
| 3 | -0,007 | 0,513 | 0,505 | +0,008 | 11,60 % |
| 6 | -0,015 | 0,520 | 0,505 | +0,015 | 19,66 % |
| 12 | -0,018 | 0,535 | 0,505 | +0,030 | 15,43 % |
| 24 | 0,048 | 0,521 | 0,505 | +0,016 | 23,82 % |
| 36 | 0,053 | 0,483 | 0,505 | -0,022 | 23,43 % |

Conclusion : le ranking 12 mois est le meilleur pour l'extrême top-10 selon
NDCG. Le 24 mois donne le meilleur compromis IC/overlap Legacy. Le top-10 à
36 mois est moins bon qu'un score sans signal malgré un IC global positif.

### Teacher Legacy

Sur janvier 2018 à décembre 2025 :

- ROC-AUC : `0,961`;
- PR-AUC : `0,356`;
- Brier : `0,0132`;
- erreur de calibration : `0,0037`;
- part du panier Legacy retrouvée par un top-10 : `47,60 %`.

Le teacher confirme que les features EMA permettent de comprendre une grande
partie de Legacy, sans constituer un objectif économique acceptable.

## 3. Backtest trading

Règles :

- score produit hors échantillon ;
- top-10 égal-pondéré ;
- rebalancement mensuel ;
- détention pendant le mois suivant la décision ;
- coût de `10 bps × turnover mensuel` ;
- pas de cash, levier, overlay de risque ni contrainte sectorielle ;
- comparaison SPY et `Combined_Frequency` sur exactement les mêmes mois.

### Période native de chaque horizon — top-10 net

Les rendements totaux ne sont pas directement comparables entre lignes car les
périodes diffèrent.

| Modèle | H | Période test | Rendement net | CAGR | Sharpe | Max DD | SPY | Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification | 1 | 2013-01→2025-12 | +981,9 % | 20,1 % | 0,70 | -48,3 % | +479,0 % | +809,4 % |
| Classification | 3 | 2013-05→2025-04 | +702,9 % | 19,0 % | 0,64 | -49,4 % | +345,8 % | +492,7 % |
| Classification | 6 | 2013-11→2025-10 | +551,2 % | 16,9 % | 0,61 | -52,5 % | +364,7 % | +511,8 % |
| Classification | 12 | 2014-11→2024-10 | +395,5 % | 17,4 % | 0,61 | -40,8 % | +246,9 % | +327,8 % |
| Classification | 24 | 2016-11→2023-10 | +311,4 % | 22,4 % | 0,73 | -46,3 % | +133,9 % | +151,4 % |
| Classification | 36 | 2018-11→2022-10 | +171,0 % | 28,3 % | 0,84 | -26,8 % | +58,3 % | +74,3 % |
| Ranking | 1 | 2013-01→2025-12 | +1 432,4 % | 23,4 % | 0,75 | -41,7 % | +479,0 % | +809,4 % |
| Ranking | 3 | 2013-05→2025-04 | +429,2 % | 14,9 % | 0,55 | -53,4 % | +345,8 % | +492,7 % |
| Ranking | 6 | 2013-11→2025-10 | +658,0 % | 18,4 % | 0,65 | -36,5 % | +364,7 % | +511,8 % |
| Ranking | 12 | 2014-11→2024-10 | +461,7 % | 18,8 % | 0,64 | -44,1 % | +246,9 % | +327,8 % |
| Ranking | 24 | 2016-11→2023-10 | +340,4 % | 23,6 % | 0,75 | -28,5 % | +133,9 % | +151,4 % |
| Ranking | 36 | 2018-11→2022-10 | +72,0 % | 14,5 % | 0,66 | -35,4 % | +58,3 % | +74,3 % |
| Régression | 1 | 2013-01→2025-12 | +452,4 % | 14,1 % | 0,64 | -46,8 % | +479,0 % | +809,4 % |
| Régression | 3 | 2013-05→2025-04 | +843,7 % | 20,6 % | 0,76 | -44,6 % | +345,8 % | +492,7 % |
| Régression | 6 | 2013-11→2025-10 | +790,3 % | 20,0 % | 0,76 | -41,3 % | +364,7 % | +511,8 % |
| Régression | 12 | 2014-11→2024-10 | +365,1 % | 16,6 % | 0,59 | -53,0 % | +246,9 % | +327,8 % |
| Régression | 24 | 2016-11→2023-10 | +360,1 % | 24,4 % | 0,71 | -43,6 % | +133,9 % | +151,4 % |
| Régression | 36 | 2018-11→2022-10 | +139,1 % | 24,3 % | 0,70 | -42,7 % | +58,3 % | +74,3 % |

### Comparaison commune 2018-11→2022-10

Sur ces 48 mois, SPY fait `+58,3 %`, CAGR `12,2 %`, Sharpe `0,68`, max DD
`-23,9 %`. Legacy fait `+74,3 %`, CAGR `14,9 %`, Sharpe `0,74`, max DD
`-23,4 %`.

Les meilleurs rendements top-10 sont :

- ranking 12 mois : `+190,7 %`, CAGR `30,6 %`, Sharpe `0,76`,
  max DD `-44,1 %`;
- classification 36 mois : `+171,0 %`, CAGR `28,3 %`, Sharpe `0,84`,
  max DD `-26,8 %`;
- ranking 1 mois : `+162,6 %`, CAGR `27,3 %`, Sharpe `0,70`,
  max DD `-41,7 %`;
- régression 24 mois : `+140,0 %`, CAGR `24,5 %`, Sharpe `0,66`,
  max DD `-43,6 %`;
- régression 6 mois : `+133,9 %`, CAGR `23,7 %`, Sharpe `0,72`,
  max DD `-41,3 %`.

Cette fenêtre commune est courte et inclut seulement quatre folds pour le
36 mois. Elle ne suffit pas à promouvoir ce dernier malgré ses bons chiffres.

### Candidats longue période en changeant top-N

- `ranking-1 top-20`, 2013-2025 : `+1 166,1 %`, CAGR `21,6 %`,
  Sharpe `0,80`, max DD `-34,3 %`. Legacy : `+809,4 %`, Sharpe `0,98`,
  max DD `-23,4 %`.
- `regression-6 top-5`, 2013-2025 : `+1 027,2 %`, CAGR `22,4 %`,
  Sharpe `0,77`, max DD `-36,9 %`. Legacy sur cette période :
  `+511,8 %`, Sharpe `0,87`, max DD `-23,4 %`.

Ils battent Legacy en rendement brut/net de coût simplifié, mais pas en qualité
de risque. Legacy reste nettement meilleur sur drawdown et généralement sur
Sharpe.

## 4. Conclusion opérationnelle

Il n'existe pas un horizon gagnant pour tous les objectifs :

- probabilités : classification 1 ou 3 mois ;
- prévision de magnitude : aucune régression n'est satisfaisante ;
- signal ordinal économique : régression 24 mois en IC, mais pas en R² ;
- ranking du top extrême : ranking 12 mois ;
- proximité avec Legacy : ranking 24 mois ;
- trading longue période : ranking 1 mois top-20 ou régression 6 mois top-5 ;
- contrôle du risque : Legacy reste la référence à battre.

La prochaine étape défendable est un modèle boosting multi-tête :

1. probabilité top-décile à 1-3 mois ;
2. score ordinal rendement à 6-24 mois ;
3. volatilité et downside futurs ;
4. allocation optimisée sur rendement prévu moins risque prévu, avec objectif
   et validation de portefeuille explicitement pénalisés par le drawdown.

## Artefacts

- `test_coverage.csv` : historique exact par modèle/horizon ;
- `model_horizon_summary.csv` : métriques ML complètes ;
- `trading_backtest_all.csv` : top-5/10/20 sur périodes natives ;
- `trading_backtest_common_period.csv` : comparaison sur période commune ;
- `<model>_h<horizon>/trading_monthly.csv` : trajectoire mensuelle détaillée ;
- `<model>_h<horizon>/shap_*` : analyses SHAP hors échantillon.

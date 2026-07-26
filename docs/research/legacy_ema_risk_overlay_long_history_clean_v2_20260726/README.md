# Audit Legacy et replay exact-EMA sur snapshot validé

Date : 2026-07-26

Statut : `clean_replay_research_only_overlay_not_validated`

> **Superseded by the semantic data audit.** This replay fixed package lineage
> and metric conventions but still selected a corrupt `BMC.US` price series and
> inherited ticker-reuse/universe problems. Its 33,73 % alpha CAGR is invalid.
> Use
> [`../legacy_ema_data_integrity_audit_20260726/README.md`](../legacy_ema_data_integrity_audit_20260726/README.md)
> for the current conclusion.

Ce dossier remplace les chiffres de référence du rapport du 25 juillet. Le
modèle, ses hyperparamètres, ses horizons et les règles d'allocation n'ont pas
été modifiés après lecture des résultats v1. La seule modification expérimentale
est le remplacement du package de données par le dernier package dont le
validateur de replay passe.

## Pourquoi l'audit était nécessaire

Le premier rapport annonçait un CAGR Legacy de 17,07 %, alors que les rapports
historiques du projet montraient plus de 20 %. Trois vérifications étaient donc
nécessaires :

1. recomposer indépendamment `Combined_Frequency` depuis les lignes détaillées ;
2. vérifier l'alignement entre mois de décision et mois de détention ;
3. distinguer l'historique Legacy complet de la fenêtre commune accessible au
   modèle.

L'audit a aussi trouvé deux défauts dans la présentation v1 :

- le snapshot `20260719_194418` échoue au contrôle officiel de lignée
  (`open_source_run_id_match=false` et fichiers publiés non concordants) ;
- le Sharpe ML était le Sharpe mensuel annualisé, alors que les rapports Legacy
  utilisent `(CAGR - 2 %) / volatilité annualisée`.

Les chiffres v1 restent des résultats reproductibles sur un package figé, mais
ils ne doivent plus être utilisés comme référence de production ou comparaison
centrale.

## Ce qui a été vérifié sur Legacy

`Combined_Frequency` est bien l'union des quatre blocs
`Legacy_Optuna_11/12/21/22`.

Pour chaque mois :

- chaque titre reçoit une fréquence égale au nombre de blocs qui le
  sélectionnent, divisé par quatre ;
- ces fréquences sont renormalisées entre les titres sélectionnés ayant un
  rendement valide ;
- le rendement mensuel est la somme des rendements titres pondérés.

La recomposition indépendante de
`legacy_detailed_returns_polars.parquet` reproduit
`legacy_monthly_returns_polars.parquet` avec une erreur absolue maximale de
`1,76e-16`.

L'alignement est également exact :

- décisions ML : juillet 2011 à octobre 2025 ;
- détentions/rendements : août 2011 à novembre 2025 ;
- 172 mois uniques ;
- aucun mois Legacy absent ;
- écart maximal après jointure : zéro.

## Package de données propre

Package utilisé :

`outputs/2026-07-13/runs/20260713_201639`

Commande de validation :

```bash
./.venv/bin/python scripts/validate_legacy_replay_package.py \
  outputs/2026-07-13/runs/20260713_201639/data_input_manifest.json
```

Résultat : `Legacy replay package is valid.`

Le validateur signale seulement que le fichier de code
`src/alpharank/data/processing.py` a été modifié depuis le run. Cela ne modifie
pas le snapshot, les sorties Legacy figées ni le replay ML qui lit ce snapshot.

## Réconciliation des performances Legacy

Convention commune :

- CAGR géométrique ;
- Sharpe Legacy = `(CAGR - 2 %) / volatilité annualisée` ;
- drawdown sur la courbe composée ;
- pire année calculée uniquement sur les années calendaires complètes ;
- SPY calculé sur `adjusted_close`, donc dividendes réinvestis.

| Fenêtre commune Legacy/SPY | Série | Mois | CAGR | Sharpe Legacy | Max DD | Pire année complète |
|---|---|---:|---:|---:|---:|---:|
| 2010-02 à 2026-05 | Legacy | 196 | 23,33 % | 0,858 | -28,44 % | 2015 : -10,83 % |
| 2010-02 à 2026-05 | SPY total return | 196 | 14,72 % | 0,880 | -23,93 % | 2022 : -18,18 % |
| 2015-02 à 2026-04 | Legacy | 135 | 22,00 % | 0,821 | -28,44 % | 2022 : -1,09 % |
| 2015-02 à 2026-04 | SPY total return | 135 | 13,96 % | 0,791 | -23,93 % | 2022 : -18,18 % |
| 2011-08 à 2025-11 | Legacy | 172 | 16,43 % | 0,669 | -28,44 % | 2015 : -10,83 % |
| 2011-08 à 2025-11 | SPY total return | 172 | 14,34 % | 0,865 | -23,93 % | 2022 : -18,18 % |

Le souvenir d'un Legacy supérieur à 20 % est donc correct. Le CAGR de 16,43 %
répond à une autre question : « que fait Legacy uniquement pendant les 172
mois de détention pour lesquels le challenger six mois produit des prédictions
OOS ? »

La différence de fenêtre est particulièrement importante ici :

- février 2010 à juillet 2011 : Legacy gagne `+53,44 %` ;
- décembre 2025 à mai 2026 : Legacy gagne `+126,19 %` ;
- ces deux segments ne sont pas accessibles à la comparaison ML six mois.

## Replay alpha exact sans retuning

Configuration :

`configs/research/legacy_ema_risk_overlay_long_history_clean_v2.json`

Alpha :

- boosting XGBoost classification du décile supérieur de surperformance future ;
- horizon six mois ;
- uniquement les couples EMA gagnants Legacy observables au cutoff train ;
- top 5 mensuel équipondéré ;
- 10 points de base multipliés par le turnover ;
- 62 mois de train minimum, 6 validation, purge 6, blocs test 12 mois ;
- 15 folds et 76 916 observations test.

Métriques modèle alpha :

- ROC-AUC : `0,584` ;
- PR-AUC : `0,148`, soit `1,46x` la prévalence ;
- Brier : `0,091` ;
- log-loss : `0,332` ;
- ECE : `0,011` ;
- NDCG@10 lift contre absence de signal : `+0,026`.

## Comparaison portefeuille sur le même intervalle

Toutes les lignes suivantes utilisent août 2011–novembre 2025, soit les mêmes
172 rendements mensuels. Les allocations ML sont nettes de 10 pb multipliés par
le turnover.

| Méthode | CAGR | Sharpe Legacy | Max DD | Pire année complète |
|---|---:|---:|---:|---:|
| alpha top 5 égal | 33,73 % | 0,804 | -31,65 % | 2024 : -15,85 % |
| inverse volatilité 1 mois | 32,24 % | 0,838 | -31,06 % | 2024 : -16,66 % |
| inverse volatilité 3 mois | 32,60 % | 0,836 | -28,97 % | 2024 : -16,68 % |
| inverse volatilité 6 mois | 32,79 % | 0,832 | -29,46 % | 2024 : -15,62 % |
| inverse downside 1 mois | 32,09 % | 0,835 | -31,71 % | 2024 : -16,69 % |
| inverse downside 3 mois | 32,36 % | 0,829 | -31,28 % | 2024 : -16,36 % |
| inverse downside 6 mois | 32,51 % | 0,830 | -30,76 % | 2024 : -16,00 % |
| inverse vol 3 mois + contrainte secteur | 25,74 % | 0,625 | -39,33 % | 2018 : -19,94 % |
| Legacy `Combined_Frequency` | 16,43 % | 0,669 | -28,44 % | 2015 : -10,83 % |
| SPY total return | 14,34 % | 0,865 | -23,93 % | 2022 : -18,18 % |

Le replay propre confirme un rendement alpha historique très élevé. Il ne
justifie toutefois pas une promotion :

- le modèle et la variante six mois ont été choisis après de nombreux essais ;
- le Deflated Sharpe précédent n'atteignait pas le seuil confirmatoire ;
- le risque reste nettement supérieur à SPY ;
- le top 5 égal reste légèrement plus profond en drawdown que Legacy ;
- aucun holdout nouveau six mois n'est encore mûr.

## Têtes de risque et SHAP

| Cible | Horizon | Métrique principale | Résultat |
|---|---:|---|---:|
| volatilité réalisée | 1 mois | Spearman / R2 | 0,354 / 0,141 |
| volatilité réalisée | 3 mois | Spearman / R2 | 0,398 / 0,125 |
| volatilité réalisée | 6 mois | Spearman / R2 | 0,398 / 0,085 |
| downside journalier | 1 mois | Spearman / R2 | 0,296 / 0,071 |
| downside journalier | 3 mois | Spearman / R2 | 0,349 / 0,057 |
| downside journalier | 6 mois | Spearman / R2 | 0,354 / 0,040 |
| forte volatilité | 1 mois | ROC-AUC / PR-AUC | 0,731 / 0,476 |
| forte volatilité | 3 mois | ROC-AUC / PR-AUC | 0,765 / 0,534 |
| forte volatilité | 6 mois | ROC-AUC / PR-AUC | 0,765 / 0,540 |

Signaux SHAP dominants :

- alpha six mois : `1/298`, `71/260`, `71/260 z`, `17/331`,
  `87/201 z` ;
- volatilité trois mois : `81/56`, `1/298`, `71/260`, `17/331`,
  `85/199` ;
- probabilité high-vol trois mois : `71/260`, `87/201`, `87/201 z`,
  `85/199 z`, `1/298 z` ;
- downside trois mois : `81/56`, `1/298`, `71/260`, `85/199`,
  `17/331`.

SHAP explique les prédictions du modèle ; il ne prouve pas une relation
causale.

## Décision

L'inverse volatilité trois mois améliore le drawdown de 2,68 points et le
Sharpe Legacy de 0,032, mais perd 1,13 point de CAGR. L'intervalle de bootstrap
de la différence de Sharpe mensuel contient zéro (`-0,038` à `+0,094`).

Aucun overlay ne passe tous les garde-fous pré-enregistrés. La bonne décision
reste donc :

- conserver le top 5 égal comme baseline de recherche ;
- conserver les probabilités/volatilités/downside et SHAP comme sorties
  explicatives ;
- ne pas activer les overlays de pondération ;
- attendre un holdout nouveau et mûr avant toute revendication de supériorité.

## Artefacts

- Alpha :
  `outputs/multihorizon_boosting/legacy_ema_long_history_clean_v2_20260726`
- Risque/allocation :
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_clean_v2_20260726`
- Rapport HTML :
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_clean_v2_20260726/html/risk_results_paper.html`
- Table comparable :
  `allocation_performance_legacy_convention.csv`
- Réconciliation des fenêtres :
  `reference_performance_windows.csv`

Le HTML a été vérifié dans un navigateur réel à 1280 px et 390 px. Aucun
overflow horizontal de page ni erreur console n'a été trouvé ; les tables sont
scrollables horizontalement sur mobile.

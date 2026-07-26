# Réaudit Legacy, prix et univers historique

Date : 2026-07-26

Statut : `performance_claims_blocked_by_semantic_data_integrity`

Ce document est la conclusion de l'audit déclenché par l'écart entre le CAGR
Legacy attendu par l'utilisateur (supérieur à 20 %) et le CAGR de 16–17 %
affiché dans le premier rapport long historique.

## Conclusion courte

La construction mathématique de Legacy est correcte et a été recomposée à
l'identique. Le CAGR supérieur à 20 % existe bien sur les longues fenêtres.

En revanche, ni Legacy ni le challenger ML ne doivent encore être présentés
comme un backtest qualifié :

- le premier run ML utilisait un package dont la lignée officielle échoue ;
- le replay sur package validé contenait un rendement non tradable de `+347 %`
  sur `BMC.US` ;
- Legacy contient lui-même des rendements aberrants `CPWR.US` en 2011 ;
- l'univers historique contient des réutilisations de tickers et des
  appartenances économiquement impossibles.

Le problème principal n'est donc plus le calcul du CAGR. C'est la validité
sémantique de l'identité des titres et de l'univers point-in-time.

## 1. Ce que fait exactement Legacy

`Combined_Frequency` agrège les quatre blocs :

- `Legacy_Optuna_11` ;
- `Legacy_Optuna_12` ;
- `Legacy_Optuna_21` ;
- `Legacy_Optuna_22`.

Chaque titre reçoit un poids proportionnel au nombre de blocs qui le
sélectionnent. Les poids sont renormalisés entre les titres ayant un rendement
valide, puis les rendements titres pondérés sont additionnés.

La recomposition indépendante depuis
`legacy_detailed_returns_polars.parquet` reproduit la série publiée avec une
erreur absolue maximale de `1,76e-16`.

L'alignement temporel ML est exact :

- décision : juillet 2011 à octobre 2025 ;
- rendement détenu : août 2011 à novembre 2025 ;
- 172 mois uniques ;
- aucun mois Legacy manquant ;
- aucune erreur de jointure.

## 2. Pourquoi Legacy est parfois à 16 % et parfois au-dessus de 20 %

Package rejouable utilisé :

`outputs/2026-07-13/runs/20260713_201639`

Le validateur officiel répond :

`Legacy replay package is valid.`

Convention :

- CAGR composé ;
- Sharpe Legacy = `(CAGR - 2 %) / volatilité annualisée` ;
- SPY sur `adjusted_close`, dividendes réinvestis ;
- pire année uniquement sur les années calendaires complètes.

| Fenêtre | Série | CAGR | Sharpe Legacy | Max DD |
|---|---|---:|---:|---:|
| 2010-02 à 2026-05 | Legacy publié | 23,33 % | 0,858 | -28,44 % |
| 2010-02 à 2026-05 | SPY total return | 14,72 % | 0,880 | -23,93 % |
| 2015-02 à 2026-04 | Legacy publié | 22,00 % | 0,821 | -28,44 % |
| 2015-02 à 2026-04 | SPY total return | 13,96 % | 0,791 | -23,93 % |
| 2011-08 à 2025-11 | Legacy publié | 16,43 % | 0,669 | -28,44 % |
| 2011-08 à 2025-11 | SPY total return | 14,34 % | 0,865 | -23,93 % |

Le chiffre de 16,43 % exclut :

- les 18 premiers mois, très favorables à Legacy ;
- décembre 2025 à mai 2026, où Legacy gagne `+126,19 %`.

Le souvenir d'un Legacy à plus de 20 % est donc juste, mais il correspond à une
autre fenêtre.

## 3. Anomalies de prix trouvées

### BMC dans le challenger

Le replay ML sur le package validé sélectionne `BMC.US` pour décembre 2012.
La source passe de `532` à `2380`, tandis que `open=2575` et `high=532`.
Le rendement calculé est `+347,37 %` et transforme le mois du portefeuille en
`+79,12 %`.

Effet sur le CAGR alpha top 5 :

| Traitement de BMC | CAGR |
|---|---:|
| rendement brut publié | 33,73 % |
| BMC mis en cash | 29,22 % |
| BMC retiré et quatre autres titres renormalisés | 29,42 % |
| mois entier retiré | 28,58 % |

Le chiffre de 33,73 % est donc invalidé.

### CPWR dans Legacy

Legacy sélectionne `CPWR.US` en 2011 avec des rendements non plausibles :

- mai 2011 : `-90,0 %` ;
- juin 2011 : `+138,1 %` ;
- août 2011 : `+300,0 %`.

Une sensibilité qui retire CPWR des holdings existants et renormalise les autres
poids donne :

| Fenêtre | Legacy publié | Sensibilité sans CPWR |
|---|---:|---:|
| 2010-02 à 2026-05 | 23,33 % | 21,46 % |
| 2011-08 à 2025-11 | 16,43 % | 14,82 % |
| 2015-02 à 2026-04 | 22,00 % | 22,00 % |

Cette sensibilité ne remplace pas un rerun intégral de Legacy, car retirer CPWR
avant le signal peut modifier les quantiles et les sélections. Elle montre
toutefois que le chiffre 2015+ supérieur à 20 % n'est pas expliqué par CPWR,
alors que le début de l'historique l'est partiellement.

## 4. Réutilisation de tickers et biais d'univers

Exemples vérifiés :

- `EP` est El Paso Corporation dans le fichier de constituants 2011, mais
  `US_General` décrit Empire Petroleum ;
- `COL` est Rockwell Collins dans les constituants, tandis que la série de prix
  vaut quelques centimes avec presque aucun volume ;
- `GR` est Goodrich dans les constituants, mais son nom de référence est un
  identifiant numérique et sa série est illiquide ;
- `SW` est Smurfit Westrock, société créée en 2024, mais apparaît dans le
  fichier de constituants dès janvier 1990 ;
- janvier 1990 contient 526 membres, pas environ 500.

Le problème vient du fichier de référence
`data/SP500_Constituents.csv`, copié tel quel dans les packages open-source. Le
package peut donc être parfaitement rejouable au niveau fichiers tout en étant
économiquement incorrect.

Le replay BMC-exclu choisissait encore 41 holdings non tradables sur 40 mois :
30 provenaient de `COL.US`.

## 5. Garde-fou causal implémenté

Un filtre point-in-time a été ajouté au frame multi-horizon. Il utilise
uniquement les données connues dans le mois de décision :

- au moins 10 observations de prix ;
- volume dollar journalier médian d'au moins 1 million USD ;
- au plus 5 % de lignes OHLC incohérentes ;
- exclusions historiques documentées, dont `CPWR.US` et `BMC.US`.

Le filtre est appliqué avant les rangs cross-sectionnels, l'entraînement et le
test. Il ne regarde aucun rendement futur.

Configuration :

`configs/research/legacy_ema_risk_overlay_tradability_v4.json`

Tests :

- le filtre accepte un mois liquide et cohérent ;
- il rejette un mois dont 50 % des OHLC sont impossibles ;
- l'ensemble des 24 tests multi-horizon passe.

## 6. Résultat v4 : diagnostic, pas validation

Après réentraînement complet avec le filtre causal :

| Méthode, mêmes 172 mois | CAGR | Sharpe Legacy | Max DD | Pire année |
|---|---:|---:|---:|---:|
| alpha top 5 égal | 33,78 % | 0,886 | -28,01 % | 2018 : -6,46 % |
| inverse vol 1 mois | 35,14 % | 0,949 | -26,59 % | 2018 : -8,91 % |
| inverse vol 3 mois | 35,04 % | 0,943 | -26,40 % | 2018 : -10,36 % |
| inverse vol 6 mois | 35,22 % | 0,944 | -26,96 % | 2018 : -7,97 % |
| inverse downside 1 mois | 35,54 % | 0,964 | -26,15 % | 2018 : -7,77 % |
| inverse downside 3 mois | 35,20 % | 0,944 | -27,27 % | 2018 : -8,97 % |
| inverse downside 6 mois | 35,26 % | 0,944 | -26,97 % | 2018 : -7,16 % |
| vol 3 mois + contrainte secteur | 33,73 % | 0,915 | -34,42 % | 2018 : -6,60 % |
| Legacy publié | 16,43 % | 0,669 | -28,44 % | 2015 : -10,83 % |
| SPY total return | 14,34 % | 0,865 | -23,93 % | 2022 : -18,18 % |

Le résultat reste très élevé, mais il n'est pas une preuve :

- le fichier de constituants historique n'est toujours pas corrigé ;
- le filtre de tradabilité a été défini après observation des anomalies ;
- la variante alpha six mois/top 5 vient d'un grand espace d'essais antérieurs ;
- le meilleur mois, avril 2020, vaut `+61,52 %`, notamment avec APA
  `+213,84 %` et DVN `+80,46 %` ; ces prix sont cohérents mais montrent une
  exposition extrême aux rebonds ;
- sans le meilleur mois, le CAGR tombe à `29,57 %` ;
- sans les trois meilleurs mois, il tombe à `24,04 %` ;
- le fichier de constituants peut encore injecter des titres avant leur vraie
  entrée dans l'indice.

Les modèles de risque restent mesurables :

- volatilité trois mois : Spearman `0,404`, R2 `0,116` ;
- high-vol trois mois : ROC-AUC `0,769`, PR-AUC `0,541` ;
- downside trois mois : Spearman `0,354`, R2 `0,064`.

Aucun overlay pré-enregistré ne passe tous les garde-fous.

## 7. Décision

Ce que l'audit permet d'affirmer :

- la formule et l'agrégation Legacy sont reproduites correctement ;
- Legacy est bien supérieur à 20 % sur les fenêtres longue et 2015+ publiées ;
- le chiffre Legacy sur la fenêtre ML est inférieur à cause de la période ;
- les performances brutes Legacy et ML restent contaminées par la qualité
  sémantique des historiques.

Ce qu'il ne permet pas encore d'affirmer :

- que le challenger bat réellement Legacy ;
- que le CAGR Legacy complet est économiquement tradable ;
- que l'univers est strictement point-in-time.

Prochaine condition de passage :

1. reconstruire un univers historique avec identifiant société stable
   (CIK/FIGI/CUSIP quand disponible), dates d'entrée/sortie et ponts de ticker ;
2. auditer prix et identité sur chaque holding Legacy et ML ;
3. rerun Legacy et ML sur ce même univers scellé ;
4. comparer sur la même fenêtre et conserver un holdout neuf.

## Artefacts

- rapport HTML v4 :
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_tradability_v4_20260726/html/risk_results_paper.html`
- alpha v4 :
  `outputs/multihorizon_boosting/legacy_ema_long_history_tradability_v4_20260726`
- risque/allocation v4 :
  `outputs/multihorizon_boosting/legacy_ema_risk_overlay_tradability_v4_20260726`
- audit précédent BMC :
  `docs/research/legacy_ema_risk_overlay_long_history_clean_v2_20260726/README.md`

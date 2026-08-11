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

## 8. Audit d'identité et quarantaine complète v1

Cette section remplace la liste provisoire d'exclusions utilisée dans v4.

### Hypothèse et règle

Hypothèse : les rendements extrêmes restants proviennent en partie de collisions
de symboles, séries de prix mal ajustées et trajectoires qui continuent après une
acquisition ou un delisting.

La décision retenue est volontairement conservatrice :

- une anomalie statistique seule ne déclenche jamais une suppression ;
- une exclusion requiert une anomalie mesurée dans le snapshot et une preuve
  externe officielle de l'identité, du niveau de prix ou de la date de
  cotation ;
- un ticker exclu est retiré sur **toutes les dates**, avant les prix mensuels,
  EMA, rangs cross-sectionnels, entraînements, sélections et rendements ;
- le registre est une quarantaine propre à ce dataset symbol-keyed, pas une
  interdiction permanente de la société.

Registre versionné :

`configs/data_quality/historical_ticker_exclusions_v1.json`

Les dix exclusions sont :

`SII.US`, `CBE.US`, `TIE.US`, `CPWR.US`, `BMC.US`, `COL.US`, `GR.US`,
`EP.US`, `SW.US`, `HAR.US`.

Les cas couvrent notamment :

- un ticker réutilisé entre deux sociétés : `SII` et `EP` ;
- une série continuant après disparition du titre historique : `CBE`, `TIE`,
  `CPWR`, `BMC`, `COL`, `GR`, `HAR` ;
- une série présente avant la première cotation officielle : `SW` ;
- une échelle de prix incompatible avec les documents officiels : `BMC`, `COL`,
  `GR`, `HAR`.

Le registre contient vingt liens de preuve. Les sources prioritaires sont les
8-K, 10-K et proxies SEC, complétés par les pages investisseurs des acquéreurs
ou émetteurs.

### Crible exhaustif des titres effectivement détenus

Commande :

```bash
./.venv/bin/python \
  scripts/experiments/audit_historical_ticker_price_integrity.py
```

Entrées :

- snapshot validé
  `outputs/2026-07-13/runs/20260713_201639/input_snapshot` ;
- holdings Legacy publiés du même run ;
- holdings ML v2 publiés ;
- registre `historical_ticker_exclusions_v1`.

Résultat sur l'union des holdings Legacy et ML :

| Statut | Nombre |
|---|---:|
| screen pass | 404 |
| exclus parmi les holdings | 7 |
| revue manuelle, non exclus | 9 |
| total contrôlé | 420 |

Les neuf cas en revue sont `ATI`, `CNX`, `FOXA`, `GE`, `HIG`, `IBM`, `SLB`,
`UA` et `UBER`. Les faibles similarités de nom sont surtout des acronymes ou
changements de dénomination. `HIG` a un vrai mouvement de `+102,36 %` le
2008-12-05 avec OHLC cohérents et volume élevé ; `UA` n'a qu'une violation OHLC
isolée. Ils ne sont donc pas supprimés.

### Sensibilité des holdings publiés

Cette table filtre et renormalise les holdings déjà choisis. Elle mesure
l'exposition directe mais **n'est pas** un rerun causal, puisque les rangs et
modèles n'ont pas été recalculés.

| Fenêtre | Série | CAGR | Sharpe Legacy | Max DD |
|---|---|---:|---:|---:|
| 2010-02 à 2026-05 | Legacy publié | 23,33 % | 0,858 | -28,44 % |
| 2010-02 à 2026-05 | Legacy, sensibilité sans les 10 tickers | 21,46 % | 0,801 | -28,44 % |
| 2010-02 à 2026-05 | SPY total return | 14,72 % | 0,880 | -23,93 % |
| 2011-08 à 2025-11 | ML v2 égal publié | 34,52 % | 0,824 | -31,44 % |
| 2011-08 à 2025-11 | ML v2, sensibilité sans les 10 tickers | 32,39 % | 0,899 | -31,17 % |
| 2011-08 à 2025-11 | Legacy publié | 16,43 % | 0,669 | -28,44 % |
| 2011-08 à 2025-11 | SPY total return | 14,34 % | 0,865 | -23,93 % |

### Rerun ML complet v6

Hypothèse : vérifier si le résultat ML survit quand les dix tickers sont retirés
avant toute transformation, tout en conservant exactement l'alpha EMA,
l'horizon six mois, les fenêtres temporelles, les têtes de risque et les règles
d'allocation gelés en v4.

Run alpha :

`outputs/multihorizon_boosting/legacy_ema_long_history_ticker_quarantine_v6_20260726`

Run risque et allocation :

`outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726`

Métriques classification hors échantillon, 15 folds et 76 534 lignes test :

- ROC AUC `0,5894` ;
- PR AUC `0,1556`, soit `1,535x` la prévalence ;
- Brier `0,0906` ;
- log loss `0,3336` ;
- ECE `0,0112` ;
- lift NDCG@10 `0,0518`.

Performance sur les mêmes 172 mois :

| Méthode | CAGR | Sharpe Legacy | Max DD | Pire année |
|---|---:|---:|---:|---:|
| alpha top 5 égal | 37,47 % | 1,045 | -27,88 % | 2015 : -6,25 % |
| inverse volatilité 3 mois | 38,41 % | 1,094 | -27,43 % | 2014 : -7,70 % |
| inverse downside 1 mois | 38,76 % | 1,105 | -27,24 % | 2014 : -7,65 % |
| Legacy publié | 16,43 % | 0,669 | -28,44 % | 2015 : -10,83 % |
| SPY total return | 14,34 % | 0,865 | -23,93 % | 2022 : -18,18 % |

Les têtes de risque restent prédictives : Spearman mensuel `0,30–0,44` selon
la cible et l'horizon ; le classifieur de forte volatilité atteint ROC AUC
`0,735–0,767`. Malgré cela, aucune allocation ne passe tous les garde-fous
pré-enregistrés.

Le CAGR ML plus élevé après quarantaine n'est pas un argument causal en faveur
des exclusions : le retrait en amont change les rangs, le training et toutes les
sélections. Le résultat reste diagnostique à cause du fichier de constituants
non réparé, du multiple testing déjà consommé et de l'absence de nouveau holdout.

### État du rerun Legacy

Mise à jour du 2026-08-07 : le rerun Legacy complet sur l'univers mis en
quarantaine a finalement abouti. Le package rejouable validé est :

`outputs/2026-07-27/runs/20260727_221253`

Il utilise les 17 fenêtres Optuna, 30 essais par fenêtre, le registre
`historical_ticker_exclusions_v1` appliqué avant les EMA, rangs et sélections,
et le snapshot immuable `20260727_200005`. Le validateur officiel répond
`Legacy replay package is valid.`

Réconciliation de la baisse du CAGR `Combined_Frequency` :

| Étape | Fin comparable | CAGR |
|---|---|---:|
| run publié du 13/07, preprocessing antérieur et séries corrompues | 2026-05 | 23,33 % |
| run du 19/07 après correction du preprocessing, mais lignée invalide | 2026-05 | 21,24 % |
| rerun causal propre avec quarantaine en amont | 2026-05 | 19,55 % |
| même rerun après ajout de juin réalisé | 2026-06 | 20,17 % |
| même série prolongée avec juillet complet | 2026-07 | 18,36 % |

Le rendement de juillet du panier décidé le 30 juin est `-20,90 %`, contre
environ `+0,03 %` pour SPY ajusté. Sur février 2010-juillet 2026, la comparaison
équitable avec SPY sur `adjusted_close` donne Legacy `18,36 %` de CAGR contre
SPY `14,47 %`; le Sharpe Legacy est toutefois inférieur (`0,74` contre `0,87`).

Les prix et ratios communs des runs corrigés du 19 et du 27 juillet sont
pratiquement identiques sur l'ancien historique. Le changement restant vient
de la quarantaine appliquée avant le calcul : retirer les trajectoires de
tickers corrompues modifie les rangs, l'optimisation et les sélections de toute
la période. La sensibilité post-sélection à `21,46 %` ne devait donc pas être
interprétée comme le résultat du rerun causal complet.

### Attribution détaillée du CAGR courant

Audit du 2026-08-09 sur `Combined_Frequency`, février 2010 à juillet 2026 :

`outputs/legacy_attribution_20260809/`

Restitution principale HTML :

`outputs/legacy_attribution_20260809/html/index.html`

Publication publique :

`https://alpharank.net/research/legacy-attribution/index.html`

Le dépôt voisin `portfolio` copie automatiquement le dernier dossier
`outputs/legacy_attribution_*` pendant `make research-sync` et `make prod-build`.

Les trois rapports reliés sont :

- `html/ticker_attribution.html` : 374 tickers, filtres, tri et concentration ;
- `html/monthly_attribution.html` : 198 mois, richesse et contributions ;
- `html/preprocessing_impact.html` : décomposition avant/après par année,
  mois et métrique fondamentale.

Le script reproductible est
`scripts/experiments/analyze_legacy_return_attribution.py`. Il publie :

- `ticker_contributions.csv` : contribution agrégée des 374 tickers ;
- `ticker_month_contributions.csv` : chaque position ticker/mois ;
- `monthly_contributions.csv` : contribution des 198 mois ;
- `summary.json` et `README.md` : contrôles et synthèse.

Le générateur de restitution est
`scripts/experiments/render_legacy_attribution_reports.py`. Les CSV restent des
artefacts machine de contrôle ; ils ne sont plus le point d'entrée utilisateur.

La décomposition additive porte sur `log(1 + rendement)`. La somme des
contributions ticker et la somme des contributions mensuelles valent exactement
`16,8594 %` annualisés ; `exp(16,8594 %) - 1` redonne le CAGR composé de
`18,3639 %`. La colonne `cash_cagr_impact_pp` est un contre-factuel intuitif
mais non additif : elle remplace le ticker ou le mois par du cash.

Concentration du run courant : `NVDA.US` est le premier contributeur avec
`+1,8507` point de log-rendement annualisé et `+2,1336` points de CAGR dans le
contre-factuel cash. Les cinq premiers tickers représentent `32,81 %` du
log-rendement net, les vingt premiers `72,51 %`. Aucun rendement ticker/mois du
run propre ne dépasse `100 %` en valeur absolue. Cinq tickers dépassent `50 %`
au moins une fois : `ANF.US`, `NFLX.US`, `SNDK.US`, `THC.US` et `WDC.US`.

Le screen automatique des 374 tickers détenus donne 366 passages et huit cas
en revue. Sept sont des faux positifs de similarité de nom. `HIG.US` est signalé
pour un jour à `+102,36 %` en décembre 2008, hors fenêtre du backtest et hors de
son unique mois détenu, février 2012. Aucun des dix tickers quarantainés
n'apparaît dans les holdings du rerun causal.

Il reste une position sans rendement : `DFS.US` pèse `5,1282 %` dans le
portefeuille de juin 2025 alors que son dernier prix est le 16 mai et que son
acquisition par Capital One est finalisée le 18 mai. Legacy retire la ligne et
renormalise les autres poids, ce qui transforme le rendement mensuel de
`5,7245 %` avec la poche manquante conservée en cash en `6,0339 %`. L'effet sur
le CAGR complet est `+0,0210` point (`18,34298 %` contre `18,36395 %`). Ce cas
doit être corrigé dans l'univers point-in-time ; son impact est mesuré, mais la
convention actuelle n'est pas économiquement neutre.

`SNDK.US` demande une correction de référence distincte : l'univers conserve
correctement deux périodes séparées, l'ancien SanDisk jusqu'en mai 2016 puis le
nouveau Sandisk à partir de décembre 2025. Les trois holdings commencent en
février 2026 et utilisent uniquement les prix de la nouvelle cotation. En
revanche, `US_General.parquet` conserve le CIK historique `0001000180`, alors
que la nouvelle société cotée depuis février 2025 utilise le CIK SEC
`0002023554`. Cela n'explique pas le rendement d'avril 2026, mais le mapping
d'identité générale doit être rendu temporel pour éviter une future jointure
incorrecte.

### Décomposition de la correction de preprocessing

La baisse comparable du 13 au 19 juillet, de `23,33 %` à `21,24 %` de CAGR,
correspond à `1,7069` point de log-rendement annualisé. Elle ne vient pas d'une
seule révision de fin de série :

- 5 085 clés ticker/mois changent dans l'univers filtré ;
- sur 73 105 clés communes, 51 975 `PE` changent (`71,1 %`) ;
- les quatre branches Optuna changent de modèle sur 160 à 184 des 196 mois ;
- les 196 rendements mensuels `Combined_Frequency` changent.

Les plus gros écarts mensuels avant moins après correction sont août 2011
`+34,03` points, juin 2011 `+15,26`, janvier 2026 `+13,31`, avril 2015
`+10,49` et mai 2015 `-10,20`. Août 2011 contient l'exposition `CPWR` du run
ancien ; la correction des fondamentaux a aussi changé sa sélection. Les
contributions annuelles au gap de log-rendement se compensent fortement :
2011 `+1,5879` point, 2015 `-1,6118`, 2026 `+1,6416`, et toutes les autres
années entre `-0,7464` et `+0,8058`. Le mécanisme est donc global : les ratios
point-in-time déterminent l'univers cross-sectionnel, puis les 17 fenêtres
annuelles réoptimisent les modèles sur des entrées différentes.

Rapport HTML principal :

`outputs/data_quality/historical_ticker_price_audit_20260726/price_identity_audit.html`

Rapport annuel toutes méthodes, même calendrier :

`outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/html/annual_returns_all_methods.html`

Comparaison allocation-only Top 5 contre Top 10 :

`outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726/html/top5_vs_top10.html`

SHAP alpha complet et holdings mensuels Legacy/Top 5/Top 10 :

`outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726/html/alpha_shap_and_monthly_portfolios.html`

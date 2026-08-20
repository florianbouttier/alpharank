# Journal Boosting — 25 et 26 juillet 2026

> Fragment chronologique conservé depuis le catalogue central.


## 2026-07-25 — comparaison boosting multi-horizon — RUN INITIAL INVALIDE

> Les tableaux de cette section et de la shortlist CPCV sont conservés comme
> trace d'incident, mais ne doivent pas être utilisés. La jointure teacher
> laissait la colonne source `weight_normalized` dans les features économiques.
> Le run propre et les conclusions corrigées sont dans la section suivante et
> dans `docs/research/multihorizon_boosting_20260725/performance_report.md`.

### Hypothese et protocole

La nouvelle piste compare sur une base commune :

- classification du top decile futur ;
- regression du rendement cumule futur relatif au S&P 500 ;
- ranking groupe par mois ;
- classifier teacher de `Combined_Frequency`, diagnostic uniquement.

Les horizons economiques sont `1, 3, 6, 12, 24, 36` mois. Les horizons 24 et
36 mois restent exploratoires : peu de regimes independants et fort
chevauchement naturel des labels.

Le package dedie est `src/alpharank/multihorizon/`. Le protocole detaille,
l'audit de Legacy, les commandes et le journal sont dans
`docs/research/multihorizon_boosting_20260725/`.

Controles essentiels :

- grille de 45 paires EMA relatives declaree ex ante, pas extraite des choix
  finaux de Legacy ;
- constituants historiques et fondamentaux joints par date de publication ;
- ecart calendaire exact pour chaque label ;
- walk-forward externe avec modele gele sur 12 mois ;
- maturite des labels et purge adaptees a chaque horizon ;
- CPCV purge uniquement dans l'historique pre-test ;
- filtrage des variables, medianes et calibration appris dans chaque fold ;
- SHAP calcule seulement sur les observations out-of-sample ;
- overlap Legacy publie comme diagnostic, jamais utilise comme objectif des
  modeles economiques.

### Donnees et execution

Snapshot de recherche coherent :

`outputs/2026-07-19/runs/20260719_194418/input_snapshot`

Sorties Legacy associees :

- `legacy_detailed_returns_polars.parquet` ;
- `legacy_monthly_returns_polars.parquet`.

Le frame reel contient 113 351 observations, 387 colonnes et 335 variables
candidates de janvier 2005 a avril 2026. Ce snapshot est coherent avec le run
Legacy utilise pour les labels, mais cette experience ne le requalifie pas en
package open-source production-clean.

Run de screening :

`outputs/multihorizon_boosting/screening_20260725`

### Resultats du screening fixe

Les chiffres ci-dessous sont hors echantillon. L'overlap est bien le nombre de
noms communs divise par la taille du panier Legacy.

| modele | h | folds | IC Spearman mensuel | top10 excess sur h | top10 excess a 1 mois | overlap Legacy top10 |
|---|---:|---:|---:|---:|---:|---:|
| regression | 24 | 7 | `0.0959` | `+17.71%` | `+1.15%` | `10.67%` |
| ranking | 24 | 7 | `0.0485` | `+12.86%` | `+1.01%` | `23.82%` |
| classification | 24 | 7 | `0.0243` | `+13.24%` | `+0.97%` | `19.03%` |
| regression | 36 | 4 | `0.0492` | `+24.50%` | `+1.27%` | `18.85%` |
| classification | 36 | 4 | `0.0571` | `+14.92%` | `+1.36%` | `21.75%` |
| teacher Legacy | 1 | 8 | `0.0049` | `+0.60%` | `+0.60%` | `45.83%` |

Lecture :

- piste economique prioritaire : regression 24 mois. C'est le meilleur IC avec
  sept blocs annuels et le meilleur top10 24 mois parmi les candidats mieux
  couverts ;
- meilleur pont avec Legacy : ranking 24 mois. Il double environ l'overlap de
  la regression 24 mois tout en gardant un rendement relatif positif ;
- 36 mois reste interessant mais non concluant avec quatre folds seulement ;
- le teacher atteint `ROC AUC 0.9608`, Brier `0.0131` et `45.8%` d'overlap
  top10. Il prouve que les features contiennent une grande partie de la
  mecanique Legacy, mais pas toute sa recomposition ;
- les dispersions par fold restent elevees. Pour la regression 24 mois,
  IC `0.0959 +/- 0.1101` et excess mensuel top10
  `1.15% +/- 1.06%` (moyenne +/- ecart type). Ce n'est pas encore un signal
  production-ready.

SHAP separe clairement les deux problemes :

- regression 24 mois : croissance EPS TTM 4 trimestres, volatilites 36/24 mois,
  earnings yield, ratio de volatilite 12/36, regime de volatilite SPY et
  momentum 12/24/36 mois ;
- ranking 24 mois : largeur Bollinger 12/6 mois, volatilites 3/6/12/24 mois,
  earnings yield et EMA relatives longues `100/400`, `100/360`, `80/400` ;
- teacher Legacy : surtout rangs/z-scores d'EMA relatives (`80/90`, `20/260`,
  `60/120`, `40/180`, `10/180`) puis largeur Bollinger.

La faible intersection entre les SHAP economiques, domines par risque,
valorisation et croissance, et les SHAP teacher, domines par EMA
cross-sectionnelles, explique pourquoi copier Legacy et prevoir la
surperformance ne sont pas le meme probleme.

Decision : CPCV cible sur regression, ranking et classification 24 mois, avec
teacher comme controle. Ne pas promouvoir 36 mois sans nouvelles annees ou
une analyse de stabilite plus forte.

### Shortlist CPCV

Run :

`outputs/multihorizon_boosting/shortlist_cpcv_20260725`

Le CPCV utilise trois trials sur les trois derniers blocs annuels. Ce run sert a
tester la stabilite recente, pas a remplacer le screening sept-folds.

| modele | IC | top10 excess 24m | top10 excess 1m | overlap Legacy |
|---|---:|---:|---:|---:|
| regression 24 | `0.0158` | `+8.78%` | `+2.12%` | `14.54%` |
| ranking 24 | `-0.0334` | `+0.58%` | `+0.67%` | `20.02%` |
| classification 24 | `-0.0016` | `-3.87%` | `+1.13%` | `14.74%` |
| teacher 1 | `0.0066` | `+1.53%` | `+1.53%` | `41.53%` |

Comparaison exacte aux memes trois derniers folds avec parametres fixes :

- regression : CPCV ameliore top10 24 mois (`8.78%` vs `5.63%`) et excess un
  mois (`2.12%` vs `0.84%`), IC presque inchange (`0.0158` vs `0.0164`) ;
- ranking : CPCV degrade l'IC (`-0.0334` vs `-0.0085`) malgre un meilleur
  overlap et un meilleur excess un mois ;
- classification : CPCV reduit la perte 24 mois et ameliore l'excess un mois,
  mais le rendement 24 mois reste negatif ;
- teacher : les parametres fixes gagnent les trois recherches internes et
  gardent un meilleur overlap.

Conclusion honnete : seul le regresseur 24 mois merite de poursuivre le tuning.
Le ranking 24 mois reste un bon diagnostic de proximite Legacy, mais son
optimisation economique recente est instable. Trois folds ne suffisent pas pour
declarer que le CPCV a augmente la performance generale.

SHAP directionnel sur la shortlist :

- regression : volatilite propre 24 mois elevee et volatilite SPY 24 mois
  elevee diminuent le score ; earnings yield et momentum 36 mois eleves
  l'augmentent. Le ratio de volatilite 12/36 est positif, ce qui signale une
  forme plus complexe qu'un simple veto de volatilite ;
- ranking : largeur Bollinger, volatilite propre, EMA longue `24/36`, EMA
  relative longue et momentum 36 mois eleves augmentent plutot le score ;
- classification : volatilite SPY elevee diminue la probabilite, tandis que
  dispersion et volatilite propres elevees l'augmentent dans cet echantillon ;
- teacher : plusieurs z-scores EMA relatifs eleves augmentent la probabilite.
  Certains rangs EMA ont une correlation SHAP negative : avec des variables
  fortement correlees et des interactions, ce resume de direction n'est pas
  une contrainte monotone ni une interpretation causale.

## 2026-07-25 — correction anti-fuite et backtest complet

Run valide :

`outputs/multihorizon_boosting/screening_clean_20260725`

Correction :

- suppression de la colonne source `weight_normalized` apres creation de
  l'alias teacher explicitement exclu `legacy_weight_normalized` ;
- test de regression reproduisant la fuite ;
- verification effective de 334 features : aucun poids, label, nombre de votes
  ou selection Legacy n'est accessible aux modeles economiques ;
- rerun complet classification/regression/ranking aux horizons
  `1, 3, 6, 12, 24, 36`.

Historique test :

| h | debut | fin | mois | folds |
|---:|---:|---:|---:|---:|
| 1 | 2013-01 | 2025-12 | 156 | 13 |
| 3 | 2013-05 | 2025-04 | 144 | 12 |
| 6 | 2013-11 | 2025-10 | 144 | 12 |
| 12 | 2014-11 | 2024-10 | 120 | 10 |
| 24 | 2016-11 | 2023-10 | 84 | 7 |
| 36 | 2018-11 | 2022-10 | 48 | 4 |

Conclusions ML :

- classification : meilleurs ROC-AUC a 1 mois (`0.634`) et 3 mois (`0.633`) ;
  PR-AUC `0.164-0.167` pour une prevalence proche de `0.10` ;
- regression : tous les R2 sont negatifs et RMSE normalisees superieures a 1 ;
  les amplitudes ne sont pas bien prevues. Le meilleur IC est a 24 mois
  (`0.096`) ;
- ranking : meilleur lift NDCG@10 a 12 mois (`+0.030` vs score sans signal) ;
  meilleur compromis IC/overlap Legacy a 24 mois (`IC 0.048`,
  overlap top10 `23.82%`) ;
- teacher : ROC-AUC `0.961`, PR-AUC `0.356`, Brier `0.0132`, overlap top10
  `47.60%`.

Backtest mensuel top-N :

- scores strictement out-of-sample ;
- top-N egal-pondere, reequilibre mensuellement ;
- rendement du mois suivant ;
- cout `10 bps x turnover` ;
- comparaison SPY et `Combined_Frequency` sur les memes mois.

Candidats longue periode :

- ranking 1 mois top20, 2013-2025 : `+1166.1%`, CAGR `21.6%`, Sharpe `0.80`,
  max DD `-34.3%`; Legacy `+809.4%`, Sharpe `0.98`, DD `-23.4%` ;
- regression 6 mois top5, 2013-2025 : `+1027.2%`, CAGR `22.4%`,
  Sharpe `0.77`, max DD `-36.9%`; Legacy `+511.8%`, Sharpe `0.87`,
  DD `-23.4%`.

Conclusion : certains paniers boosting battent Legacy en rendement, mais pas
en risque. Legacy reste meilleur en drawdown et generalement en Sharpe. Il n'y
a pas un horizon unique :

- probabilites : 1-3 mois ;
- trading longue periode : ranking 1 mois ou regression 6 mois ;
- signal ordinal : 12-24 mois ;
- proximite Legacy : ranking 24 mois ;
- 36 mois : exploratoire seulement, quatre folds.

Rapport complet :

`docs/research/multihorizon_boosting_20260725/performance_report.md`

## 2026-07-25 — diagnostic des couples EMA Legacy exacts

Hypothese utilisateur :

> Commencer par les EMA qui ont effectivement gagne dans Legacy avant
> d'elargir la representation.

Donnees :

- run fige `outputs/2026-07-19/runs/20260719_194418` ;
- `legacy_detailed_returns_polars.parquet` ;
- quatre chemins `Legacy_Optuna_11`, `12`, `21`, `22`.

Resultat :

- chaque chemin utilise 13 couples `n_short/n_long` distincts sur les
  portefeuilles 2010-2026 ;
- l'union contient 32 couples exacts ;
- la grille multi-horizon actuelle contient 45 couples arrondis declares ex
  ante ;
- intersection exacte entre grille actuelle et gagnants Legacy :
  **`0 / 32`**.

Exemples de gagnants absents de la grille actuelle :

`5/257`, `7/333`, `34/189`, `45/92`, `63/150`, `71/260`,
`95/72`, `100/326`.

Decision :

1. ajouter un benchmark `legacy_winners_pit_ema_only` ;
2. dans chaque fold, construire l'union des couples gagnants connus strictement
   avant validation/test, jamais l'union finale 2010-2026 injectee dans le
   passe ;
3. tester ces EMA exactes seules, puis EMA exactes + risque/fondamentaux ;
4. conserver comme oracle diagnostique une variante utilisant les quatre
   couples Legacy actifs au mois de decision ;
5. comparer d'abord les horizons `1, 3, 6, 12`, puis `24, 36`.

La variante oracle depend de Legacy et n'est donc pas un challenger autonome.
La variante union historique train-only est le benchmark propre. Les
contraintes `n_asset`, filtre fondamental et plafond sectoriel doivent etre
separees de la qualite du signal EMA pour savoir ce qui explique la
performance.

## 2026-07-25 — résultats exact-EMA point-in-time corrigés

Rapport source de vérité :

`docs/research/exact_legacy_ema_20260725/README.md`

Hypothèse :

> Un booster entraîné uniquement sur les couples EMA ayant déjà gagné dans
> Legacy au moment de chaque fold peut préserver le bon inductive bias sans
> recevoir les décisions futures de Legacy.

Données et protocole :

- snapshot figé `outputs/2026-07-19/runs/20260719_194418/input_snapshot` ;
- calendrier Legacy
  `outputs/2026-07-19/runs/20260719_194418/legacy_detailed_returns_polars.parquet` ;
- horizons `1, 3, 6, 12, 24, 36` ;
- walk-forward 72 mois train, 24 validation, 12 test, purges par horizon ;
- coûts `10 bps x turnover` ;
- `n_trials=0`.

Runs corrigés :

- `outputs/multihorizon_boosting/screening_clean_20260725` ;
- `outputs/multihorizon_boosting/legacy_winners_pit_ema_only_20260725` ;
- `outputs/multihorizon_boosting/legacy_winners_pit_ema_plus_20260725` ;
- `outputs/multihorizon_boosting/legacy_active_oracle_20260725`.

Incident et correction :

- la probabilité calibrée isotone créait des plateaux ;
- le top-N départageait alors certains ex aequo par ordre de ticker ;
- les anciennes sorties sont conservées sous
  `outputs/multihorizon_boosting/invalid_calibrated_tie_20260725` ;
- le rerun utilise le score brut pour ordonner les titres et la probabilité
  calibrée uniquement pour Brier, log-loss et ECE.

Résultat principal autonome :

- `legacy_winners_pit_ema_only`, classification top décile futur à 6 mois,
  top 5 ;
- test novembre 2013-octobre 2025, 144 mois, 12 folds ;
- net `+2628.5%`, CAGR `31.7%`, Sharpe `0.969`, max DD `-30.3%` ;
- Legacy mêmes mois : `+511.8%`, CAGR `16.3%`, Sharpe `0.873`,
  max DD `-23.4%`.

Robustesse :

- top 10 : `+1135.1%`, Sharpe `0.835`, DD `-36.4%` ;
- top 20 : `+791.4%`, Sharpe `0.766`, DD `-38.2%` ;
- hors année 2016 : CAGR `25.3%`, Sharpe `0.814`, contre Legacy
  CAGR `14.7%`, Sharpe `0.784` ;
- avantage donc réel mais concentré dans les cinq premiers titres.

Statistiques modèles :

- meilleure discrimination future : EMA-plus classification 3 mois,
  ROC-AUC `0.641`, PR-AUC `0.172` ;
- meilleur trading futur : EMA-only classification 6 mois top 5 ;
- régressions : tous les R2 test négatifs ; meilleur IC autour de 24 mois
  (`0.096` broad, `0.092` EMA-plus) ;
- ranking 36 mois donne plus d'overlap mais seulement quatre folds : ne pas
  promouvoir ;
- meilleure copie autonome Legacy : EMA-only teacher, ROC-AUC `0.975`,
  PR-AUC `0.342`, overlap top20 `77.4%`.

SHAP du champion 6 mois :

- principaux signaux :
  `95/72 z-score`, `100/326 z-score`, `95/72 rank`, `92/183 z-score`,
  `27/106 z-score`, `7/333 raw` ;
- les représentations cross-sectionnelles dominent ;
- les directions SHAP sont descriptives et non causales.

Décision :

1. geler `EMA winners PIT only / classification 6m / top5` comme challenger ;
2. valider sur un nouveau meta-holdout jamais utilisé pour choisir la variante ;
3. garder teacher 1 mois pour rationalisation probabiliste de Legacy ;
4. ajouter ensuite des têtes volatilité/downside et la calibration, sans
   modifier le ranking avant validation indépendante ;
5. ne pas considérer l'oracle actif comme autonome.

## 2026-07-25 — challenger verrouillé et validation anti-sélection

Spécification gelée :

`configs/research/locked_legacy_ema_challenger_v1.json`

Rapport central :

`docs/research/locked_challenger_confirmation_20260725/README.md`

Papiers HTML :

- `outputs/multihorizon_boosting/locked_challenger_confirmation_20260725/html/methodology_paper.html` ;
- `outputs/multihorizon_boosting/locked_challenger_confirmation_20260725/html/results_paper.html`.

Le challenger reste :

`exact EMA winners PIT / classification top 10% / horizon 6m / top5`

Aucun paramètre, feature, horizon ou top-N n'a été modifié après le verrou.

Contrôles ajoutés :

- 10 000 bootstraps circulaires par blocs de 12 mois ;
- Deflated Sharpe avec 162 essais autonomes ;
- replay de sélection annuel parmi 108 candidats, utilisant seulement les
  36 mois OOS précédents ;
- sensibilité aux coûts `0/10/25/50/100 bps x turnover` ;
- analyse par fold, ticker, secteur et calibration ;
- tentative explicite de holdout partiel, laissée indisponible faute de cible
  six mois mûre après octobre 2025.

Résultats :

- écart de rendement moyen annualisé vs Legacy `+16.30 pp`,
  IC95% `[+3.88, +28.90] pp` ;
- écart de Sharpe vs Legacy `+0.096`,
  IC95% `[-0.326, +0.499]` : non confirmé ;
- Deflated Sharpe probability `76.1%` après 162 essais : inférieur au seuil
  confirmatoire de 95% ;
- à 100 bps de coût, le candidat conserve CAGR `25.1%`, Sharpe `0.816`,
  DD `-31.7%` ;
- meta-selector temporel 2018-2024 : `+167.8%`, CAGR `15.5%`, Sharpe `0.589`,
  DD `-33.0%`, contre Legacy Sharpe `0.684`, DD `-23.4%` ;
- le meta-selector ne choisit jamais exactement le champion final ;
- le secteur mensuel dominant pèse en moyenne `44.3%` et atteint parfois
  `100%`.

Conclusion :

- le surplus de rendement historique est robuste et résiste aux coûts ;
- la supériorité de Sharpe n'est pas statistiquement établie ;
- le choix du champion reste exposé au multiple testing ;
- le statut est `paper_challenger_not_production_ready`.

Décision suivante :

1. archiver des scores mensuels prospectifs sans retuning ;
2. versionner séparément les têtes volatilité/downside ;
3. conserver l'ordre alpha exact du challenger ;
4. ne promouvoir qu'après un holdout réellement nouveau et mûr.

## 2026-07-25 — historique long et têtes de risque exact-EMA

Rapport central :

`docs/research/legacy_ema_risk_overlay_long_history_20260725/README.md`

Spécification pré-enregistrée :

`configs/research/legacy_ema_risk_overlay_long_history_v1.json`

Implémentation : commit `ecb9e33`.

Hypothèse :

> Garder le classement alpha boosting exact-EMA inchangé, estimer séparément
> volatilité, downside et probabilité high-vol, puis n'utiliser le risque que
> pour les poids ou une contrainte sectorielle explicite.

Historique maximal sans fuite de sélection :

- prix à partir de janvier 2005 ;
- première paire EMA gagnante observable en février 2010 ;
- 62 mois train, 6 validation, purge conservatrice 6 mois ;
- test juillet 2011-octobre 2025, 172 mois, 15 folds ;
- remonter avant juillet 2011 nécessiterait de choisir rétroactivement une EMA
  qui n'était pas encore gagnante dans Legacy.

Alpha top 5 équipondéré sur l'historique long :

- net `+3693.9%`, CAGR `28.88%`, Sharpe `0.778`, DD `-35.43%` ;
- S&P 500 : CAGR `14.34%`, Sharpe `1.016`, DD `-23.93%` ;
- Legacy : CAGR `17.07%`, Sharpe `0.890`, DD `-23.38%`.

Le rendement reste fort, mais l'avantage ajusté du risque du résultat sélectionné
2013-2025 ne se répète pas.

Têtes de risque OOS :

- volatilité réalisée 3 mois : Spearman mensuel `0.440`, R2 `0.152`,
  13/15 folds avec R2 positif ;
- downside 3 mois : Spearman `0.385`, R2 `0.085` ;
- high-vol 3 mois : ROC-AUC `0.783`, PR-AUC `0.562` ;
- high-vol 6 mois : ROC-AUC `0.784`, PR-AUC `0.569`.

Les cibles utilisent uniquement les rendements journaliers des mois futurs,
avec au moins 10 observations par mois. Toutes les têtes partagent la purge de
six mois et les EMA winners train-only.

Allocation primaire pré-enregistrée :

- top 5 alpha égal : CAGR `28.88%`, Sharpe `0.778`, DD `-35.43%` ;
- mêmes titres inverse-vol 3 mois : CAGR `26.96%`, Sharpe `0.781`,
  DD `-31.80%` ;
- différence de Sharpe `+0.003`, IC95% bootstrap
  `[-0.051, +0.073]` ;
- la règle sectorielle 40% : CAGR `24.35%`, Sharpe `0.719`,
  DD `-35.62%`.

Décision :

- les têtes de risque sont conservées comme sorties utiles et explicables ;
- aucun overlay ne passe tous les garde-fous pré-enregistrés ;
- ne pas activer l'inverse-vol ni la contrainte sectorielle ;
- toute piste high-vol veto ou tilt plus doux reçoit un nouvel identifiant et
  ne doit pas être présentée comme confirmatoire sur ces mêmes 172 mois.

Papiers HTML :

- `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725/html/index.html` ;
- `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725/html/risk_results_paper.html` ;
- `outputs/multihorizon_boosting/legacy_ema_risk_overlay_long_history_v1_20260725/html/methodology_paper.html`.

## 2026-07-26 — réaudit Legacy et replay sur snapshot validé

Rapport source de vérité :

`docs/research/legacy_ema_risk_overlay_long_history_clean_v2_20260726/README.md`

Le rapport du 25 juillet est conservé comme trace v1, mais ses chiffres ne
doivent plus être utilisés comme référence centrale :

- le package `20260719_194418` échoue au validateur officiel de lignée ;
- son Sharpe utilisait la moyenne mensuelle annualisée et non la convention des
  rapports Legacy.

Le replay exact, sans changement de modèle ni de paramètre, utilise le package
validé `outputs/2026-07-13/runs/20260713_201639`.

Réconciliation Legacy :

- historique complet commun Legacy/SPY, février 2010-mai 2026 :
  Legacy CAGR `23,33%`, Sharpe Legacy `0,858`, DD `-28,44%` ;
- fenêtre historique 2015-02 à 2026-04 :
  Legacy CAGR `22,00%`, Sharpe Legacy `0,821`, DD `-28,44%` ;
- fenêtre strictement commune au ML, août 2011-novembre 2025 :
  Legacy CAGR `16,43%`, Sharpe Legacy `0,669`, DD `-28,44%`.

La recomposition indépendante de `Combined_Frequency` depuis le détail reproduit
les rendements mensuels à `1,76e-16` près. Le faible chiffre Legacy sur le test
ML vient donc de la fenêtre, pas d'une mauvaise agrégation.

Sur les mêmes 172 mois, avec Sharpe `(CAGR - 2%)/volatilité` :

- alpha top 5 égal : CAGR `33,73%`, Sharpe `0,804`, DD `-31,65%`,
  pire année complète 2024 `-15,85%` ;
- inverse volatilité 3 mois : CAGR `32,60%`, Sharpe `0,836`,
  DD `-28,97%`, pire année 2024 `-16,68%` ;
- Legacy : CAGR `16,43%`, Sharpe `0,669`, DD `-28,44%`,
  pire année 2015 `-10,83%` ;
- SPY total return : CAGR `14,34%`, Sharpe `0,865`, DD `-23,93%`,
  pire année 2022 `-18,18%`.

Le replay confirme le rendement historique fort du booster, mais pas sa
promotion : risque supérieur à SPY, avantage de Sharpe non confirmé par
bootstrap, multiple testing antérieur et absence de holdout six mois neuf.
Aucun overlay ne passe les garde-fous pré-enregistrés.

### Correction sémantique le même jour

Le replay ci-dessus est à son tour déclassé pour les performances ML : il
sélectionne `BMC.US` sur un saut de prix non tradable de `+347%`.

Audit source de vérité :

`docs/research/legacy_ema_data_integrity_audit_20260726/README.md`

Autres problèmes trouvés :

- Legacy contient `CPWR.US` à `-90%`, `+138%` et `+300%` en 2011 ;
- `EP`, `COL`, `GR` et `SW` mélangent identités historiques et tickers
  réutilisés ;
- Smurfit Westrock apparaît dès 1990 dans les constituants ;
- janvier 1990 contient 526 membres.

Une sensibilité Legacy sans CPWR ramène le CAGR complet de `23,33%` à
`21,46%`, mais ne change pas la fenêtre 2015-2026 à `22,00%`.

Un filtre causal de tradabilité a été ajouté avant les rangs et le training :
10 observations mensuelles, volume dollar médian >= 1 M$, moins de 5% de lignes
OHLC incohérentes. Le replay v4 donne alpha top5 CAGR `33,78%`, Sharpe Legacy
`0,886`, DD `-28,01%`, mais reste diagnostique : l'univers de constituants n'est
pas encore réparé et la variante a subi du multiple testing.

Décision : ne plus revendiquer que le ML bat Legacy tant qu'un univers
historique à identité stable n'a pas été reconstruit et que Legacy et ML n'ont
pas été rejoués ensemble sur ce même univers.

### Quarantaine historique v1 et rerun ML v6

Hypothèse : les collisions d'identité et les trajectoires post-delisting doivent
être supprimées avant toute EMA, rang ou cible, et non corrigées mois par mois
après observation des rendements.

Registre :

`configs/data_quality/historical_ticker_exclusions_v1.json`

Dix tickers sont exclus sur toute leur trajectoire :
`SII`, `CBE`, `TIE`, `CPWR`, `BMC`, `COL`, `GR`, `EP`, `SW`, `HAR`.
Chaque décision combine une anomalie mesurée dans le snapshot avec des sources
officielles externes. L'audit passe au crible les 420 tickers détenus par
Legacy ou ML : 404 passent, 7 sont exclus parmi les holdings et 9 restent en
revue sans exclusion automatique.

La sensibilité post-sélection ramène Legacy, février 2010-mai 2026, de
`23,33%` à `21,46%` de CAGR ; ce n'est pas un rerun causal. Sur la fenêtre
2011-08 à 2025-11, ML v2 passe de `34,52%` à `32,39%`, Legacy publié vaut
`16,43%` et SPY `14,34%`.

Le rerun ML complet v6 retire les dix tickers avant features/rangs/training :

- classification horizon 6 mois, 15 folds, 76 534 observations test ;
- ROC AUC `0,5894`, PR AUC `0,1556` (`1,535x` la prévalence), Brier
  `0,0906`, ECE `0,0112` ;
- alpha top 5 égal : CAGR `37,47%`, Sharpe Legacy `1,045`, max DD `-27,88%` ;
- inverse vol 3 mois : CAGR `38,41%`, Sharpe `1,094`, max DD `-27,43%` ;
- aucune allocation risque ne passe tous les garde-fous.

Conclusion : le rerun prouve que la mécanique d'exclusion causale fonctionne,
pas que le challenger est validé. L'univers de constituants reste sémantiquement
fragile, le budget de multiple testing est consommé et le rerun Legacy complet
n'est pas terminé. Le rapport source de vérité est
`docs/research/legacy_ema_data_integrity_audit_20260726/README.md`.

### Allocation-only Top 5 contre Top 10 — v7

Hypothèse : augmenter le portefeuille de cinq à dix titres peut réduire la
concentration et le drawdown sans trop diluer le signal alpha.

Protocole :

- réutilisation byte-for-byte des prédictions alpha et risque OOS de v6 ;
- aucune modification du modèle, des EMA, de la cible six mois, des 15 folds,
  de la calibration, des exclusions ou des horizons risque ;
- seule variable modifiée : `top_n`, de 5 à 10 ;
- mêmes 172 mois, août 2011 à novembre 2025 ;
- mêmes coûts de 10 pb multipliés par le turnover ;
- reconstruction top 5 vérifiée à `5,55e-17` d'erreur mensuelle maximale.

Commande :

```bash
./.venv/bin/python \
  scripts/experiments/run_topn_allocation_comparison.py
```

Run :

`outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726`

Résultat principal équipondéré :

| Méthode | CAGR | Sharpe Legacy | Max DD | Pire année |
|---|---:|---:|---:|---:|
| Top 5 égal | 37,47 % | 1,045 | -27,88 % | 2015 : -6,25 % |
| Top 10 égal | 25,24 % | 0,793 | -33,73 % | 2015 : -12,86 % |
| Legacy publié | 16,43 % | 0,669 | -28,44 % | 2015 : -10,83 % |
| SPY total return | 14,34 % | 0,865 | -23,93 % | 2022 : -18,18 % |

Le bootstrap apparié par blocs de 12 mois donne Top 10 moins Top 5 :

- écart moyen annualisé `-10,73` points, IC 95 % `[-16,55 ; -4,57]` ;
- écart de Sharpe bootstrap `-0,191`, IC 95 % `[-0,360 ; -0,017]` ;
- probabilité bootstrap d'un écart de Sharpe négatif : `98,2 %`.

Diagnostic des holdings Top 10 avant coûts :

- rangs 1–5 : CAGR `38,19 %`, Sharpe `1,066`, max DD `-27,68 %` ;
- rangs 6–10 : CAGR `12,24 %`, Sharpe `0,326`, max DD `-50,51 %`.

Le turnover moyen baisse légèrement (`45,12 %` à `44,48 %`) : la dégradation
ne vient pas des frais mais de la dilution par les rangs 6–10. Les cinq
garde-fous de promotion échouent. Décision : conserver Top 5 comme référence et
ne pas promouvoir Top 10.

Rapport HTML détaillé :

`outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726/html/top5_vs_top10.html`

### SHAP alpha complet et portefeuilles mensuels

Le test Top 10 étant allocation-only, le dernier modèle reste le classifieur
alpha v6 à horizon six mois. Un rapport séparé publie sans troncature :

- un beeswarm regroupant les `185` variables présentes sur l'union des folds ;
- `185` graphiques individuels valeur de variable contre SHAP ;
- un ordre identique pour le beeswarm et les graphiques individuels :
  `mean(|SHAP|)` OOS strictement décroissant ;
- un lexique exact des `185` colonnes et des `37` couples EMA : spans
  numérateur/dénominateur, formule, suffixe, unité et interprétation ;
- le nombre d'observations et de folds actifs pour chaque variable ;
- les quantiles de valeur et de SHAP et la corrélation valeur–SHAP ;
- les `172` portefeuilles mensuels Legacy publié, Alpha Top 5 égal et
  Alpha Top 10 égal, avec tickers, rangs, poids, secteurs, scores,
  probabilités calibrées et rendements réalisés ;
- les rendements mensuels Legacy, Top 5, Top 10 et SPY sur le même calendrier.

Commande :

```bash
./.venv/bin/python \
  scripts/experiments/render_alpha_shap_portfolio_report.py
```

Précision d'interprétation : les SHAP expliquent la marge brute XGBoost
(log-odds), avant calibration isotone. Ils ne sont pas exprimés en points de
probabilité calibrée. Le sampling contient `1 200` observations OOS, soit
`80` par fold ; une variable non disponible dans un fold reste manquante et
n'est jamais imputée pour le graphique.

Rapport :

`outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726/html/alpha_shap_and_monthly_portfolios.html`

Lexique CSV :

`outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726/alpha_shap_feature_lexicon.csv`

### Passage au scoring mensuel sur données 2026

Hypothèse : le classifieur EMA-only retenu peut être exécuté sur le même
snapshot récent que Legacy, sans réentraîner sur une cible non mature et sans
utiliser le mois calendaire partiel.

Données :

- registre officiel des changements 2026 :
  `configs/data_quality/sp500_constituent_changes_2026.json`;
- snapshots mensuels des constituants étendus d'avril à juillet 2026 ;
- package open-source immuable `20260727_200005`, snapshot
  `data/open_source/history/output/open_source_output_20260727_200058`;
- `3 716 206` observations de prix sur `732` tickers historiques ;
- les `503` membres de juillet ont tous un prix ajusté, avec une date commune
  au `2026-07-23` et une date maximale au `2026-07-24`.

Commandes :

```bash
./.venv/bin/python scripts/open_source/refresh_sp500_constituents.py \
  --target-month 2026-07-01

./.venv/bin/python scripts/open_source/refresh_current_constituent_prices.py

./.venv/bin/python scripts/run_legacy.py \
  --n-trials 30 \
  --n-jobs 1 \
  --first-date 2010-01 \
  --open-source-run-id 20260727_200005 \
  --output-dir outputs \
  --checkpoints-dir outputs/checkpoints_open_source_20260727_constituents \
  --ticker-exclusion-registry configs/data_quality/historical_ticker_exclusions_v1.json
```

Le candidat live est figé sur la classification XGBoost à horizon six mois,
les seuls couples EMA relatifs gagnants de Legacy et la cible décile supérieur
de surperformance face au S&P 500. Pour une décision juin 2026 :

- les labels d'entraînement/calibration s'arrêtent en décembre 2025 ;
- les six derniers mois matures servent à l'arrêt anticipé et à la calibration
  isotone, donc leurs métriques ne sont pas présentées comme un nouveau test
  scellé ;
- le rang utilise la marge brute XGBoost et non la probabilité calibrée ;
- Top 5 reste le portefeuille retenu, Top 10 reste un diagnostic non promu ;
- le rapport compare Alpha et Legacy sur le même mois de détention, juillet
  2026, à partir du même `input_snapshot/`.

La mise à jour de juillet sert à prouver la fraîcheur des données. Le mois étant
incomplet, il n'est pas utilisé comme décision pour un portefeuille août.

Résultat du replay :

- run Legacy `outputs/2026-07-27/runs/20260727_221253`;
- paquet accepté par `scripts/validate_legacy_replay_package.py` et quatre ids
  de lignée égaux à `20260727_200005`;
- `Combined_Frequency` : CAGR historique affiché `18,82 %` contre `12,22 %`
  pour le S&P 500 sur le calendrier du replay ; cette statistique Legacy
  inclut sa mécanique historique complète et ne constitue pas un nouveau test
  Alpha ;
- portefeuille Legacy juillet : `LRCX`, `MU`, `TER` à `22,22 %`, puis `FIX`,
  `STLD`, `WDC` à `11,11 %`.

Résultat Alpha live :

- run
  `outputs/live_alpha/ema_classification_h6_202606_20260727_production_candidate_v3`;
- entraînement janvier 2005-juin 2025, `101 631` observations ;
- validation/calibration juillet-décembre 2025, `2 939` observations ;
- ROC AUC `0,6615`, PR AUC `0,2106` pour une prévalence `0,1014`, soit un lift
  PR de `2,077x` ; Brier `0,0861`, log loss `0,3060` ;
- `35` couples EMA Legacy, `175` variables après préparation train-only ;
- univers live : `503` lignes du mois de décision, puis `498` après
  intersection avec l'univers juillet ; `CAG`, `CPB`, `EPAM`, `POOL` et `SATS`
  sont retirés car absents du snapshot de détention ;
- Top 5 égal : `ZTS`, `ACN`, `CHTR`, `IT`, `LULU`, cours ajustés datés du
  `2026-06-30`, poids `20 %` chacun ;
- Top 10 diagnostic : Top 5 plus `MU`, `SNDK`, `INTU`, `WDC`, `PODD`, poids
  `10 %` chacun.

Décision : ce run rend le challenger exécutable et auditable en mensuel, mais
ne le promeut pas au-dessus de Legacy. Les métriques 2025 servent à l'arrêt
anticipé et à la calibration. La preuve de performance hors échantillon reste
le backtest temporel documenté plus haut ; le panier live doit être suivi en
paper trading face à Legacy et au S&P 500 sur les mêmes mois.

Rapports :

- données/constituants :
  `outputs/data_refresh/20260727_214718/html/constituent_refresh_audit.html`;
- couverture prix :
  `data/open_source/official/runs/20260727_200005/current_constituent_price_coverage.html`;
- Alpha et Legacy juillet :
  `outputs/live_alpha/ema_classification_h6_202606_20260727_production_candidate_v3/html/live_alpha_portfolio.html`;
- performance Legacy :
  `outputs/2026-07-27/runs/20260727_221253/performance_of_models_polars2026-07-27.html`.

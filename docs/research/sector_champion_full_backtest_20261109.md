# Backtest complet des champions sectoriels — 9 novembre 2026

## Verdict

Le plafond d'une action par secteur fonctionne sur tout l'historique causal :
les 183 décisions de juillet 2011 à septembre 2026 produisent chacune cinq
positions appartenant à cinq secteurs distincts. Le backtest porte sur les 182
mois déjà réalisés, d'août 2011 à septembre 2026 ; la décision de septembre
pour octobre est conservée comme `horizon_pending` et n'est pas valorisée.

La diversification sectorielle réduit le risque mais aussi le rendement. Elle
ne domine aucune des deux variantes Boosting de référence et reste donc une
option R&D, sans promotion ni changement de portefeuille de production.

## Résultats sur la fenêtre commune

Toutes les stratégies utilisent le même calendrier, SPY total return issu de
`adjusted_close`, et 10 bps multipliés par le turnover pour les portefeuilles.
Les « années > SPY » portent sur les 14 années civiles complètes 2012-2025.

| stratégie | CAGR | volatilité | max drawdown | Sharpe | Calmar | années > SPY | pire année complète | pire année vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Boosting Top 5 | 28,23 % | 36,97 % | -35,77 % | 0,71 | 0,79 | 9/14 | 2014 : -2,13 % | 2021 : -23,92 % |
| Boosting Top 5, 1/secteur | 19,29 % | 33,12 % | -27,34 % | 0,52 | 0,71 | 8/14 | 2018 : -13,27 % | 2017 : -14,85 % |
| Boosting tendance Top 5 | 26,90 % | 31,54 % | -26,27 % | 0,79 | 1,02 | 8/14 | 2018 : -12,37 % | 2021 : -23,89 % |
| Boosting tendance Top 5, 1/secteur | 20,94 % | 26,08 % | -26,47 % | 0,73 | 0,79 | 10/14 | 2018 : -11,33 % | 2023 : -25,00 % |
| Legacy | 26,70 % | 28,00 % | -25,57 % | 0,88 | 1,04 | 9/14 | 2018 : -14,25 % | 2018 : -10,14 % |
| SPY total return | 14,40 % | 14,23 % | -23,93 % | 0,87 | 0,60 | référence | 2022 : -18,18 % | référence |

Le natif sectoriel abandonne 8,94 points de CAGR mais améliore le drawdown de
8,42 points. La tendance sectorielle abandonne 5,97 points de CAGR, réduit sa
volatilité de 5,46 points, mais ne réduit pas son pire drawdown. Elle bat SPY
sur davantage d'années complètes que la tendance non contrainte, sans améliorer
son Calmar.

## Stabilité selon le point de départ

Les valeurs ci-dessous sont les CAGR jusqu'à septembre 2026. La ligne 2026 ne
contient que neuf mois et son CAGR est donc une annualisation, pas une année
civile réalisée.

| départ | Boosting | Boosting 1/secteur | tendance | tendance 1/secteur | SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2011 | 28,23 % | 19,29 % | 26,90 % | 20,94 % | 14,40 % |
| 2015 | 33,80 % | 18,82 % | 29,79 % | 21,32 % | 13,66 % |
| 2020 | 28,57 % | 17,92 % | 42,05 % | 31,51 % | 15,25 % |
| 2023 | 31,67 % | 18,03 % | 72,61 % | 40,98 % | 21,73 % |
| 2024 | 32,22 % | 14,60 % | 98,23 % | 62,97 % | 20,16 % |
| 2025 | 52,75 % | 10,92 % | 150,16 % | 65,75 % | 17,54 % |
| 2026 | 88,84 % | 24,50 % | 361,67 % | 146,46 % | 17,29 % |

Sur janvier-septembre 2026 en rendement cumulé non annualisé, Boosting fait
+61,09 %, Boosting sectoriel +17,86 %, tendance +214,95 %, tendance sectorielle
+96,71 % et SPY +12,71 %. Ces chiffres récents sont très concentrés dans une
fenêtre courte et ne constituent pas un critère de promotion.

## Contrat causal et contrôles

1. Les prédictions sont celles du replay candidat figé du 31 octobre ; aucun
   score, hyperparamètre ou modèle n'est recalculé après observation des
   rendements.
2. La classification sectorielle est la dernière classification SEC acceptée
   et connue à la décision ; une classification future ne peut pas remonter
   dans le passé et une classification absente exclut le candidat.
3. La gate d'événement terminal est appliquée avant le classement ; les titres
   déjà détenus conservent leur résolution de rendement auditée.
4. Le classement est fait avant la disponibilité du rendement réalisé. Une
   absence partielle ou mature dans les cinq titres sélectionnés fait échouer le
   run ; seul le dernier mois entièrement `horizon_pending` est exclu.
5. Les rendements, le turnover, les coûts, SPY et tous les KPI viennent de
   `alpharank.portfolio`; le script de replay ne possède aucun moteur financier
   parallèle.

La compatibilité `legacy_month_only` du simulateur est conservée uniquement
parce que les prédictions historiques auditées ne stockent pas les cinq
timestamps explicites du contrat moderne. La causalité de la nouvelle donnée
sectorielle est, elle, démontrée par `sector_known_at_selected` et la release
DEF ; ce replay ne prétend pas reconstruire rétroactivement les timestamps de
features du modèle historique.

## Lignée et reproduction

- snapshot candidat : composition du 31 octobre 2026, replay commun
  `replay_headc315821/candidate/common_end_2026_09_trend` ;
- prédictions : SHA-256 `838e0246206940019e137c4e6b01a04fb38396f7949cbdbe370a9281a977da5f` ;
- release secteur : `55fbc0c519ea8b50d77fb87b7c0d62ef7ab02d6ee2694c0c2e4faf0f8b5eda84` ;
- table secteur : SHA-256 `4a67a6ac6b1d137bac0ec18cc6e22d6d041345a74182a79e5534baada48884e6` ;
- holdings natifs sectoriels : SHA-256 `f9c26e2c716537dcb79efd92a3e7b61b7bdd83da5ad1bda9b450966844a74de4` ;
- holdings tendance sectoriels : SHA-256 `1fa8490ab9d670e6249cf259049f4b5df55aa7da515f5e65f5464ada1253f036` ;
- package : `outputs/sector_champion_backtest/REPLAY-014_20261109/` ;
- manifeste : SHA-256 `0d480534c64ea2f544f6da9c3170da526bad1da0e523c2ef1d7058869514cc54`.

Commande :

```bash
.venv/bin/python scripts/experiments/portfolio/run_sector_champion_backtest.py \
  --native-holdings outputs/sector_history_coverage_20261109/native/sector_champion_holdings.parquet \
  --trend-holdings outputs/sector_history_coverage_20261109/trend/sector_champion_holdings.parquet \
  --reference-monthly outputs/data_refresh_replay_20261031/replay_headc315821/candidate/common_end_2026_09_trend/comparison_common_monthly.parquet \
  --output-dir outputs/sector_champion_backtest/REPLAY-014_20261109
```

Le package contient les holdings contraints, les rendements mensuels, les KPI
globaux, les rendements annuels, les profils multi-départs, leurs CSV lisibles
et un manifeste qui scelle toutes les entrées et sorties.

## Dashboard standard — preuve du 15 novembre 2026 (`REPORT-018`)

Le [standard de reporting](../performance_reporting_standard.md) est réutilisé
sans nouvelle simulation : six séries sur 182 mois réalisés, 33 KPI et 136
fenêtres annuelles. « Tout » commence en août 2011 et non en 2010 ; les départs
2015 et 2020 sont directement accessibles. Le mode Composer inclut les cinq
poches (31 combinaisons) ; les frontières gardent leurs trois poches standard.
Le [contrat portefeuille](../common_portfolio_backtest_engine.md#canonical-holdings-contract)
rappelle les cinq titres équipondérés et les onze secteurs SIC AlphaRank,
également visibles dans la carte Méthodes.

- Source : `REPLAY-014`, composition
  `e0e6045a5b9b2702840a4f96c94618aecc708272adacfea8a2c2e843790ef95a`, replay
  commun `c315821921d951240ad3ac5f2de54a557b5e4f2b` ; frais de 10 bps ×
  turnover et SPY total return sur le calendrier réalisé commun.
- Artefacts : `outputs/performance_reports/REPORT-018_20261115/`.
  HTML SHA-256 `8a75190b5040faefd18f332372778626100fd796696cbb72c4e0279bf7147c2c` ;
  manifeste SHA-256 `ffc85f8c2147cacc60ad630c2c3778c96d5450341f490dd0d861f977ed640967`.
- Parité : 1 092 lignes de rendements/turnover/coûts identiques au backtest ;
  54 KPI CAGR, volatilité et max drawdown vérifiés à `1e-12` sur les départs
  effectifs d'août 2011, janvier 2015 et janvier 2020. Les 6 285 positions
  historiques incluent 1 820 positions contraintes, avec cinq secteurs et
  cinq poids cibles de 20 % à chaque décision réalisée.
- Dernier panier : 28 lignes d'octobre 2026 décidées fin septembre, dont dix
  sectorielles ; dernière séance de la preuve de marché : 30 octobre 2026.
  Ce panier est figé et n'entre pas dans les KPI. Aucun signal actuel, aucune
  promotion et aucune copie dans le site Portfolio.
- QA navigateur 1 200 × 900 et 390 × 844 : filtres, courbes, model cards,
  composition, frontières, panier figé, historique et taxonomie ; zéro erreur
  et zéro requête externe. Script embarqué validé par `node --check` ;
  30 tests ciblés reporting, replay secteur et moteur de comparaison réussis.
- Ruff, documentation, inventaire des dépendances et plafond par dossier
  validés (reporting : 19 fichiers sur 20, sans dérogation). La gate statique
  globale reste rouge sur des écarts préexistants hors de ce lot : baseline de
  taille, inventaire des lecteurs data et typage du registre terminal.

Commande (profil à six séries avec le code de `72f9e89` ; le code actuel
construit le profil unifié `REPORT-019`). Choisir une nouvelle destination pour
conserver l'artefact original :

```bash
.venv/bin/python scripts/research/build_backtest_performance_report.py \
  --common-replay-dir outputs/data_refresh_replay_20261031/replay_headc315821/candidate/common_end_2026_09_trend \
  --legacy-run-dir outputs/data_refresh_replay_20261031/replay_head3b17eb7/candidate/legacy/2026-10-31/runs/20261031_175053 \
  --snapshot-manifest outputs/data_refresh_replay_20261031/composed_history/alpharank_input_20261031_161144_e0e6045a5b9b/snapshot_manifest.json \
  --portfolio-as-of-evidence outputs/data_refresh_replay_20261031/price_candidate/audit/benchmark_return_extension_audit.parquet \
  --sector-backtest-dir outputs/sector_champion_backtest/REPLAY-014_20261109 \
  --sector-history-dir data/warehouse/def/sector_history/sector_history_55fbc0c519ea \
  --output-dir outputs/performance_reports/<nouveau-run>
```

## Dashboard unifié — preuve du 15 novembre 2026 (`REPORT-019`)

La même commande, dans une nouvelle destination
`outputs/performance_reports/REPORT-019_20261115/`, réunit les onze séries
standard et les deux Top 5 sectoriels. Ce sont 13 séries, 182 mois communs
d'août 2011 à septembre 2026 et 136 fenêtres de 33 KPI. Les références
Legacy PE diagnostiques restent hors du rapport standard. Le composeur garde
les deux Legacy et les quatre Top 5 (63 combinaisons) ; les autres tailles
sont toutes disponibles en comparaison. Le panier figé d'octobre demeure hors KPI.

- Même composition `e0e6045a5b9b2702840a4f96c94618aecc708272adacfea8a2c2e843790ef95a`
  et replay commun `c315821921d951240ad3ac5f2de54a557b5e4f2b`, SPY total return.
- Sur les 2 366 lignes mensuelles, 2 184 sont identiques à leurs sources.
  Les 182 mois Legacy Equal sont valorisés par le moteur commun à 10 bps ×
  turnover à partir des mêmes positions et des mêmes rendements bruts. Le
  benchmark mensuel est aligné à `1e-12`. Les 594 cellules KPI communes au
  `REPORT-018` pour les départs 2011/2015/2020 concordent à `1e-12`.
- 25 310 positions historiques et 126 positions du panier figé d'octobre.
  La règle des cinq secteurs et leur taxonomie sont inchangées.
- HTML SHA-256 `399619d85854f2dbcd586daa816ca299250bbbf39b41b5005cfdf4290872fff0` ;
  manifeste `470d65172c62fd9a0feb8331bffbcf60661245c003ae218f41f9d50542151d50` ;
  Parquet Equal `c6d655b946df0e4e426bfe66527aac610dacddea9c9557a4890b1c93fa5462a6`.
- 32 tests ciblés verts. QA navigateur à 1 200 et 390 px : 13 courbes,
  survol, filtres 2015/2020, KPI, model cards, paniers et modes Composer et
  Frontières, sans débordement global ni requête externe.
- Aucun refresh, réentraînement, changement de production ni publication Portfolio.

## Suivi prix au 21 novembre 2026 (`REPORT-020`)

Demande : retélécharger les prix et montrer la performance récente des stratégies.
Ce suivi est **une valorisation des paniers figés d'octobre**, décidés fin
septembre, pas un refresh complet des entrées modèles ou un nouveau backtest.
Les prix sont téléchargés le 21 novembre à 12:24 UTC ; la dernière clôture
commune est celle du **20 novembre 2026**. Aucun signal de novembre n'a été
produit. Les résultats de novembre et depuis l'entrée supposent de conserver
le panier d'octobre au-delà de son mois prévu, sans rééquilibrage.

### Sources, couverture et artefacts

- Source : `REPORT-019`, même composition `e0e6045a…ef95a` et HTML inchangé
  au SHA-256 `399619d85854f2dbcd586daa816ca299250bbbf39b41b5005cfdf4290872fff0`.
  Ses 182 mois de KPI et ses positions historiques ne sont pas réécrits.
- 126 positions, 12 stratégies, 32 titres distincts et SPY. Les 33 séries ont
  chacune 38 observations du 30 septembre au 20 novembre : **100 % de couverture**
  sur ce calendrier observé, sans remplissage ni titre écarté.
- Téléchargement par `YahooFinanceClient` existant, avec reprises par titre
  après les erreurs de lot. Les observations reçues et normalisées par cet
  adaptateur sont archivées en JSON RAW, séparément des snapshots publiés :
  `data/warehouse/raw/yfinance/frozen_portfolio_valuations/`.
  Reçu `receipts/valuation_20261121_122307.json`, objet prix SHA-256
  `36882f7bef544d2ec80f02eacbb42d20535bea557852e032651654221b895a4c`.
- Sortie canonique du suivi : `outputs/performance_reports/REPORT-020_20261121_02/`.
  `valuation_report.md` contient les 13 séries, dont SPY, et les quatre fenêtres ;
  `valuation.csv`, `frozen_holdings.parquet` et `downloaded_prices.parquet`
  conservent les chiffres non arrondis et les entrées.
- Manifeste SHA-256
  `fecd2caea6989956209db3c66810096528639e0cf219f55aae6b387330dd2a76` ;
  valorisations CSV `30bdad13bf8e101eb1bfe090f732243d7e73530c5ece565ebf75c165046f932d` ;
  validation `89f04776fa2ddd1d0d812283e9e93d926415b994b6040f79a5e0488fb59c24da`.
  Le manifeste conserve aussi les hashes du script, du moteur et de l'allocation,
  ainsi que Python 3.11.15 et les dates exactes.

### Résultats comparables

Rendements de richesse en USD depuis les clôtures indiquées, avec
`adjusted_close`, bruts des frais d'entrée/sortie et sans change. Les poids
dérivent depuis le 30 septembre ; aucun retour fictif à 20 % début novembre.
Les frais de 10 bps du backtest mensuel ne sont pas substitués à ces rendements
bruts de valorisation. Aucun CAGR n'est annualisé sur ces fenêtres courtes.

| Stratégie | Octobre : 30/09 → 30/10 | Novembre : 30/10 → 20/11 | Semaine : 13/11 → 20/11 |
| --- | ---: | ---: | ---: |
| Legacy Frequency | −12,80 % | −24,67 % | −15,75 % |
| Legacy Equal | −14,87 % | −24,87 % | −16,39 % |
| Boosting Top 5 | +8,12 % | −10,85 % | −13,27 % |
| Boosting Top 10 | +5,32 % | −7,49 % | −11,04 % |
| Boosting Top 15 | −0,11 % | −11,37 % | −11,33 % |
| Boosting Top 20 | −1,48 % | −13,10 % | −11,52 % |
| Boosting tendance Top 5 | −8,40 % | −21,50 % | −13,24 % |
| Boosting tendance Top 10 | −12,23 % | −23,23 % | −14,61 % |
| Boosting tendance Top 15 | −12,06 % | −22,46 % | −13,88 % |
| Boosting tendance Top 20 | −10,69 % | −21,95 % | −12,78 % |
| Boosting Top 5 sectoriel | +3,37 % | −3,37 % | −10,16 % |
| Boosting tendance Top 5 sectoriel | +7,62 % | −17,84 % | −12,66 % |
| SPY total return | −0,91 % | −8,72 % | −8,56 % |

### Cohérence contrôlée et limites

- Identité avant/après de l'objet complet des paniers (titres, poids, scores et
  métadonnées) : SHA-256
  `6508340a92484ddfea4d723ef859c3d1ce50e778dc471ef3c4b2366872957367`.
  Aucun entraînement, reclassement, changement d'univers ou promotion.
- Parité des douze rendements bruts d'octobre avec
  `simulate_weighted_portfolio()` : erreur maximale `2,78e-17`.
  L'échappatoire historique `legacy_month_only` n'est utilisée que pour cette
  vérification des paniers scellés, pas comme preuve d'un nouveau signal causal.
- Comparaison des anciens prix du snapshot et de l'extension SPY : 759 points,
  33 séries et 23 séances communes. Les niveaux bridgés sont comparés après
  normalisation au 30 septembre, pas directement à l'échelle Yahoo.
  `price_overlap.csv` et `october_vintage_comparison.csv` conservent les écarts.
- Seul MPC dépasse `1e-12` sur le parcours normalisé : clôtures non ajustées
  identiques à `5,12e-13` près, réajustement des niveaux ajustés et micro-écarts
  compatibles avec leur précision numérique. Écart normalisé maximal
  `5,9234e-8`, impact portefeuille maximal `5,1216e-9` en fraction de rendement,
  soit `0,0000005122` point de pourcentage. L'événement d'entreprise à l'origine
  de ce réajustement n'est pas confirmé indépendamment : aucune promotion ni
  garantie de parité stricte à `1e-12` entre ces deux vintages prix n'est revendiquée.
- Un contrôle supplémentaire de l'API Yahoo a reçu HTTP 429 après le
  téléchargement réussi. Il n'y a pas de contre-vérification par un second
  fournisseur. Les prix conservés ne sont pas remplacés à cause de cette limite.
- L'égalité des paniers figés ne prouve pas qu'un nouvel entraînement les aurait
  reproduits : **le replay inter-vintages n'a pas été exécuté**. Il reste
  obligatoire pour tout futur refresh complet, avec comparaison du dernier
  mois décidé dans le run précédent. Aucun fondamental SEC n'est retéléchargé ici.

### Code et validations

Calcul dans `src/alpharank/portfolio/valuation.py`, API publique dans
`src/alpharank/portfolio/__init__.py`, CLI mince dans
`scripts/research/value_frozen_portfolios.py`, tests dans
`tests/unit/portfolio/test_portfolio_valuation.py`. Les README de ces trois
dossiers, les deux contrats portefeuille/reporting, l'index recherche, les
inventaires de code et de dossiers et `ROADMAP.md` accompagnent cette tâche.

Tests de dérive des poids, fenêtres exactes, prix invalides/manquants,
invariance aux prix futurs et parité avec le moteur commun ; **72 tests réussis**
(64 portefeuille/reporting et huit de qualité/indépendance des chemins), via
`.venv/bin/python -m pytest`. Ruff et son contrôle différentiel, formatage,
mypy du nouveau module, documentation, liens Markdown et inventaires validés.
Le dossier portefeuille contient 19 modules sur 20 autorisés. La gate de taille
globale conserve huit régressions préexistantes hors des fichiers de ce lot ;
aucune nouvelle violation de taille n'est ajoutée. Aucun rendu HTML n'est modifié.

Reproduction depuis le reçu conservé, dans une **nouvelle destination** :

```bash
.venv/bin/python scripts/research/value_frozen_portfolios.py \
  --source-report outputs/performance_reports/REPORT-019_20261115/backtest_performance_report.html \
  --raw-receipt data/warehouse/raw/yfinance/frozen_portfolio_valuations/receipts/valuation_20261121_122307.json \
  --entry-date 2026-09-30 --as-of 2026-11-21 \
  --window Octobre:2026-09-30:2026-10-30 \
  --window Novembre:2026-10-30:2026-11-20 \
  --window Semaine:2026-11-13:2026-11-20 \
  --window Depuis_entree:2026-09-30:2026-11-20 \
  --output-dir outputs/performance_reports/<nouveau-run>
```

## Livraison HTML complète du suivi (`REPORT-021`, 21 novembre 2026)

Correction demandée par le propriétaire : le tableau du chat et le Markdown
de `REPORT-020` ne sont pas le livrable attendu. Toute présentation de
performances doit livrer **le même HTML canonique complet**, pas une page réduite.
La règle est inscrite dans `AGENTS.md` et le contrat de reporting ; la commande
`value_frozen_portfolios.py` produit désormais ce HTML par défaut, en plus de
ses compléments techniques. La commande de reproduction ci-dessus reste valide,
en choisissant une nouvelle destination.

### Artefact et parcours unique

- Rapport : `outputs/performance_reports/REPORT-021_20261121_02/backtest_performance_report.html`.
  HTML SHA-256 `a8c52aca684ff457563c50f93e0c6d465c1c30d192ff4f3891b502a2e258625d` ;
  manifeste HTML `46c020a80bc94615692a44b72db51bccdfe2a23e458f2117491081e1735d9668` ;
  manifeste de valorisation `3774d3e685477fa041d3edd3e33524280b476bea776ad46b3695a7ec0e707ab6`.
- Entrée stable : `outputs/performance_reports/index.html`, également accessible
  sur `http://127.0.0.1:8765/outputs/performance_reports/` tant que le serveur
  local du dépôt tourne. L'index redirige vers l'artefact daté ; il n'efface ni
  l'ancien HTML, ni ses entrées. Une page étrangère au générateur n'est pas écrasée.
- Le mode initial « Suivi récent » partage les sélecteurs et le graphique du
  studio, avec quatre fenêtres, survol daté au jour, drawdown et richesse relative
  au SPY. Les courbes et les deux KPI non annualisés sont calculés en amont.
- « Backtest complet », « Composer » et « Frontières » conservent les 13 séries,
  182 mois réalisés, 33 KPI, model cards cumulées et annuelles, 63 compositions,
  frontières, 25 310 positions historiques, 126 positions figées et toutes les
  méthodes précédentes. Une carte décrit en plus la méthode du suivi prix.

### Parité et absence de nouveau run modèle

Le HTML reprend les **mêmes prix téléchargés le 21 novembre** sous le reçu et le
hash RAW de `REPORT-020`, jusqu'au 20 novembre, sur les mêmes paniers d'octobre.
Il n'y a pas de deuxième acquisition, entraînement, nouveau signal, promotion ou
copie dans Portfolio. Après le mois prévu, le suivi reste un maintien hypothétique,
brut de frais et sans change ; il n'est pas ajouté aux rendements mensuels nets.

- Aucun champ du payload historique n'a changé, hors date technique de génération.
  L'ajout `recent_valuation` porte seul les nouvelles projections.
- Les 48 rendements stratégie/fenêtre sont **strictement identiques** au CSV de
  `REPORT-020` ; le CSV entier est inchangé, pas seulement ses arrondis affichés.
- `validation.json`, SHA-256
  `513f046b20fbe1cf5d5a3149fa50d13e7f702846bdbf08a8b998ed22b4be0663`,
  conserve ces contrôles, les hashes et les bornes de couverture.
- Le max drawdown quotidien réutilise le champ indépendant de la fréquence du
  moteur commun, avec la clôture initiale incluse. Aucun CAGR/Sharpe/volatilité
  mensuelle n'est calculé à partir de quelques séances quotidiennes.

### Implémentation et vérifications

- `portfolio/valuation.py` et son API publique : projections quotidiennes
  cohérentes avec les valorisations existantes, sans lecture des prix futurs.
- `reporting/performance_followup.py` : raccordement au générateur existant,
  préservation du payload et publication atomique de l'index possédé.
- Les modules HTML, styles, état du studio, graphique, tableaux et interactions
  du rapport sont enrichis en place ; aucun nouveau moteur frontend ou KPI local.
- La CLI existante, les tests portefeuille/reporting, leurs README, les contrats,
  conventions, roadmap et inventaires accompagnent le changement. `build-dashboard`
  guide les contrôles de sources ; son export générique absent de l'installation
  est remplacé par le générateur canonique demandé. `frontend-design` conserve
  l'identité du desk et un parcours unique plutôt qu'une deuxième page.
- **81 tests ciblés réussis** : portefeuille, reporting, inventaires et indépendance
  des chemins. Ruff ciblé et différentiel, formatage, mypy du module portefeuille,
  documentation, liens Markdown, inventaires et syntaxe JavaScript validés.
  Reporting reste au plafond autorisé de 20 fichiers Python, sans dérogation.
- QA Chrome à 1 280 et 390 px : mode récent, fenêtres, survol, drawdown,
  comparaison SPY, backtest depuis 2015, 33 lignes KPI, 156 cellules de model cards,
  composition des deux Boosting, frontières, cinq positions sectorielles et carte
  de méthodologie ajoutée. Aucun débordement global, erreur navigateur ou appel
  externe. L'index et le desk fonctionnent aussi en `file://`, hors serveur.
- Limites inchangées : pas de replay inter-vintages ni validation actuelle des
  signaux ; réserve MPC déjà décrite dans `REPORT-020`. Le contrôle global de
  taille garde les huit écarts préexistants hors lot, sans nouvelle violation.

## Refonte de l'analyse partagée (`REPORT-022`, 21 novembre 2026)

Demande du propriétaire : remplacer globalement le frontend d'analyse, pas
retoucher une seule sortie. Le défaut visible de richesse relative sur 2026 est
reproduit : les anciens points commençaient **après** le rendement de janvier,
sans matérialiser le capital initial. Le rapport n'inventait pas un rendement
antérieur, mais sa présentation empêchait de voir le départ à 1.

### Composant, interaction et direction visuelle

- Les canvas et survols artisanaux des séries temporelles et frontières sont
  remplacés par un adaptateur Plotly unique sous
  `src/alpharank/reporting/analysis_ui/`. Le générateur canonique l'embarque pour
  tous ses profils, suivi récent et composeur compris ; les HTML déjà archivés
  ne sont pas réécrits.
- Une colonne de sélection présente les familles Legacy, Boosting, tendance et
  sectorielles, avec cases, recherche et actions tout cocher/tout retirer. Les
  deux Boosting Top 5 et SPY sont affichés initialement. Tout retirer garde SPY
  seul ; changer les séries pilote aussi le scoreboard, les 33 KPI et les model
  cards. Le composeur conserve ses six poches et leurs poids équipondérés.
- Les bornes réellement disponibles sont visibles en mois/année. Les presets
  « Depuis 2015/2020/2023 » et les filtres cachés sont supprimés. Le graph domine
  le panneau, avec axe calendrier, fond neutre, SPY pointillé, survol commun,
  zoom et réinitialisation explicites, export PNG natif et thèmes cohérents.
- `frontend-design` a guidé cette direction de comparateur financier compact,
  plutôt que l'accumulation de contrôles et de courbes. Inspiration fonctionnelle :
  [Koyfin, model portfolios](https://www.koyfin.com/features/model-portfolios/).
  Implémentation fondée sur les API primaires Plotly :
  [rendu et redimensionnement](https://plotly.com/javascript/plotlyjs-function-reference/),
  [événements](https://plotly.com/javascript/plotlyjs-events/) et
  [axes et survol](https://plotly.com/javascript/reference/layout/).
- Plotly était déjà une dépendance du dépôt : aucune nouvelle bibliothèque à
  installer. Runtime vérifié `plotly.py 7.1.0 / Plotly.js 4.1.1`, JavaScript et
  notice MIT complète embarqués, version inscrite dans le HTML. La construction
  wheel hors réseau contient bien le chargeur, JS, CSS, README et licence.
- `build-dashboard` a guidé la préservation des sources et la QA. Son export
  générique n'est pas présent dans cette installation ; le générateur canonique
  demandé et les deux preuves JSON ci-dessous assurent la livraison complète.
  Aucun second moteur de KPI ni application dans Portfolio n'est créé.

### Correction des bases graphiques, sans changer le résultat

Les trois vues partagent la même projection mensuelle : repère initial à 1 pour
la richesse et la richesse relative, à 0 pour le drawdown, puis **tous** les
rendements de la fenêtre. Ce repère est nommé au survol et daté de la fin
calendaire du mois précédent ; ce n'est pas une cotation supplémentaire. Les
autres points utilisent la fin calendaire du mois détenu, pas son premier jour.

- Non-régression synthétique janvier–février 2026 : stratégie `+50 %, −10 %`,
  SPY `+10 %, +5 %`. L'ancien graphe relatif débutait à `1,363636…` ; le nouveau
  débute à `1` et conserve ce même point de janvier, puis la même valeur finale
  `1,1688311688…`. Les observations hors fenêtre ne sont pas utilisées.
- Sur la vraie vue janvier–septembre 2026, chacune des treize séries possède dix
  points avec le repère du 31 décembre 2025. Les deux Top 5 relatifs finissent
  toujours à `1,4292784045562694` et `2,7943821795450963`, SPY à 1. Le premier
  mois n'a pas été supprimé pour fabriquer visuellement le départ à 1.
- Les courbes quotidiennes récentes ont déjà leur clôture initiale : leurs
  points et leurs dates restent inchangés. Aucun drawdown scalaire ou autre KPI
  n'est recalculé par la correction graphique.

### Artefact complet et parité économique

- Rapport : `outputs/performance_reports/REPORT-022_20261121_05/backtest_performance_report.html`.
  SHA-256 `d3f02bdba00d934fae23ac924123035efab34720141bd74f1256b4c91fa8e800`,
  taille 8 175 454 octets (environ 7,8 Mio avec la bibliothèque hors ligne).
  Le rendu depuis le payload correspond exactement au fichier livré.
- Manifeste HTML `a6becbcb5444988714ef68be6f46f7479f605434ffccc08d91c146f66e046f28` ;
  manifeste de valorisation `49b86a217b5dac5b05956e770aa60d8b593b3c05d23b14ef952349997c8884cc`.
  L'entrée possédée `outputs/performance_reports/index.html` désigne ce rapport,
  via HTTP local ou `file://`, sans supprimer les générations précédentes.
- Comparaison intégrale à `REPORT-021` au SHA `a8c52aca…625d` : mêmes champs et
  valeurs, hors `generated_at_utc` et
  `recent_valuation.provenance.generated_at_utc`. Payload ainsi normalisé SHA-256
  `1d54130f022dd9ebca806a4547a5b7e0c41ff73e279f0311768bf0bdca32f1a7`.
- Conservés : 13 séries, 182 mois d'août 2011 à septembre 2026, 136 fenêtres,
  33 KPI, model cards cumulées/annuelles, 63 compositions, 168 allocations de
  frontière, 25 310 lignes historiques, 126 positions figées et neuf cartes
  méthodologiques. Toutes les variantes standard et sectorielles restent présentes.
- Les 48 rendements récents sont identiques ; CSV complet inchangé au SHA-256
  `30bdad13bf8e101eb1bfe090f732243d7e73530c5ece565ebf75c165046f932d`.
  Même reçu prix du 21 novembre, même RAW `36882f7b…5a4c`, dernière clôture le
  20 novembre. La commande de reproduction de `REPORT-020` régénère le nouveau
  frontend dans une nouvelle destination, sans deuxième téléchargement.
- `validation.json` conserve les parités et les hashes ; `qa_browser.json`, SHA
  `541f9faa507283515ff5a522dc5a17d53b954e4b6d526674299bf0a218b5ad2c`,
  conserve les résultats de QA. Les captures desktop, mobile clair et sombre
  sont dans le même dossier d'artefact.

### Vérifications et limites

- **89 tests ciblés réussis** : portefeuille, reporting, nouvelles projections,
  inventaires et indépendance des chemins. Deux avertissements existants
  (dépréciation pandas et corrélation dégénérée NumPy), aucun test en échec.
- Ruff du reporting et des tests modifiés, formatage des nouveaux modules et
  mypy du chargeur validés ; exception `import-untyped` limitée à Plotly sans
  stubs, motivée à cette frontière. Ruff différentiel : zéro diagnostic nouveau,
  248 diagnostics historiques restants. Les deux blocs JavaScript du HTML et
  l'adaptateur passent `node --check`.
- Documentation, liens Markdown, cohérence des dépendances et inventaires
  validés. Reporting reste à 20 Python directs ; son enfant `analysis_ui/` a un
  seul Python. Aucune dérogation de plafond. La gate globale de taille échoue
  toujours sur les huit régressions préexistantes hors des fichiers de ce lot ;
  aucune nouvelle régression de taille n'est introduite ici.
- QA Chrome 1 440 et 390 px : ancrage des treize séries dans les trois vues,
  fenêtre 2015–2020 à 73 points et valeur finale identique, bornes inversées,
  SPY seul, recherche sectorielle, 33 lignes KPI, model cards de la sélection,
  composeur deux Boosting, deux frontières à 168 points, paniers sectoriels à
  cinq titres, survol souris des trois séries, zoom/reset et vrai export PNG.
  Thème sombre des séries et frontières, index hors ligne, aucune erreur
  navigateur, aucun appel externe ni débordement global.
- Les menus KPI gardent les **bornes annuelles pré-calculées**, y compris les
  mois partiels de bord : pas de choix libre de toute séance ou de tout mois.
  Le zoom ne filtre pas les KPI et l'interface l'indique explicitement.
- **Aucun nouveau téléchargement ni run modèle**, aucun changement de snapshot,
  de signal, d'hyperparamètre, de rendement ou de coût. Ce travail ne constitue
  pas un replay inter-vintages ni une nouvelle preuve d'absence de fuite.
  Le suivi au-delà d'octobre reste hypothétique sur les paniers figés ; les
  réserves économiques et data de `REPORT-020` sont inchangées.

## Reconstitution journalière et fenêtres libres (`REPORT-023`, 21 novembre 2026)

Le propriétaire demande de vraies observations quotidiennes, des bornes au jour
près et des KPI évalués sur la même fenêtre que les courbes. Ce lot construit le
moteur et sa preuve ; le remplacement du frontend et la livraison sont suivis
séparément dans `REPORT-024` et `REPORT-025`.

- Source : HTML `REPORT-022_20261121_05`, SHA-256
  `d3f02bdba00d934fae23ac924123035efab34720141bd74f1256b4c91fa8e800`.
  Même composition `e0e6045a…ef95a`, prix
  `US_Finalprice.parquet` SHA `50493c34…66e8` et SPY SHA `6b29b4c9…cfdb`.
  Aucun téléchargement, entraînement, reranking ou changement de positions.
- `portfolio/daily/` revalorise les 13 séries sur 3 815 clôtures, repère du
  29 juillet 2011 inclus, jusqu'au 30 septembre 2026 : **49 595 lignes**.
  Les quantités restent fixes au sein de chaque panier mensuel et les frais
  existants sont débités à sa première clôture, comme montant fixe relativement
  au capital d'entrée. Le net reste exactement `gross_return - transaction_cost`.
- **2 307 fins de mois observables rapprochées**, écart absolu maximal
  `5,273559366969494e-16`. Les **59 autres lignes stratégie/mois** n'ont pas de
  cours à la clôture finale et restent non vérifiables quotidiennement, sans
  les convertir artificiellement en cash ou en cotations terminales.
- **58 344 entrées de KPI mensuels** du rapport source sont recalculées par le
  moteur étendu ; écart maximal `8,43769498715119e-15`, inférieur à `1e-12`.
  La fréquence mensuelle par défaut et ses chiffres archivés sont préservés.
  La nouvelle fréquence quotidienne déclare 252 séances et une durée CAGR
  calendaire réelle ; volatilité/drawdown quotidiens ne sont donc pas supposés
  égaux à leurs anciennes observations mensuelles.
- Exemple libre effectivement évalué : rendements du **10 février au 17 août
  2026**, ancrés à la clôture du 9 février. Les deux richesses partent de 1 ;
  la période n'est pas remplacée par janvier–décembre. Les fêtes et week-ends
  restent distincts des dates de clôture réellement utilisées.
- Le composeur quotidien conserve un rééquilibrage mensuel entre poches, pas
  une réallocation quotidienne des actions. Les frontières réutilisent les
  mêmes 168 poids admissibles et les sélecteurs Pareto/profils canoniques. Un
  profil non défini sur une courte période, par exemple Calmar sans drawdown,
  reste absent, sans faux infini.

### Couverture : trous conservés, jamais interpolés

Le snapshot contient **155 clés titre/séance absentes** utilisées par ces
paniers, touchant 733 valorisations stratégie/jour. Deux sont des trous internes
isolés : `JCP.US` le 7 janvier 2014 et `FISV.US` le 12 novembre 2025. Les 153
autres sont des fins de série avant la fin du mois détenu : KRFT juillet 2015,
GMCR mars 2016, WFM août 2017, CSRA avril 2018, ESRX décembre 2018, APC août
2019, TSS septembre 2019, XLNX février 2022, NLSN octobre 2022, ABMD décembre
2022, SIVB mars 2023 et PXD mai 2024. Cette observation de couverture ne prétend
pas résoudre la contrepartie économique de chaque événement.

Ces points restent `null` et sont auditables dans `daily_price_gaps.parquet`.
Les rendements/CAGR de bornes observées restent évaluables ; les risques qui
exigent le chemin complet ne le sont pas. Le drawdown après un trou reste
inconnu puisque son plus-haut peut être inconnu. Le capital du mois suivant
reprend la référence mensuelle scellée sans inventer le chemin intermédiaire.
Les fenêtres quotidiennes de 2026 sont complètes pour les treize séries.
Il s'agit d'une reconstitution de recherche, jamais d'une promotion corrigée.

### Artefacts et contrôles

Dossier : `outputs/performance_reports/REPORT-023_20261121_02/`.

| Artefact | SHA-256 |
| --- | --- |
| `daily_ledger.parquet` | `dc2603e7d2320bffbad086666ed13b3f9bb54b4538ceb019d76d7eddf3f65fb8` |
| `monthly_reconciliation.parquet` | `778debcd6da9f9ffd9297c373c5421a3e092c04cf92d7c5c35f762d9e5651b43` |
| `daily_price_gaps.parquet` | `f1084ada5d17a400833b30c152cb9c515c7684a63b02956f13bf28f72a112aa7` |

`validation.json` porte les comparaisons, bornes et identités. **114 tests
ciblés réussis**, dont les nouveaux cas de dates libres, risque après trou,
frais, parité, mutation du futur, année complète avec jour férié, annualisation,
composition, poids nuls et frontière courte. Ruff, formatage et mypy des neuf
modules modifiés du moteur sont validés. Le sous-package quotidien conserve des
fonctions de moins de 80 lignes et complexité au plus 10. Les contrôles
documentaires, liens et inventaires sont exécutés avant commit. Aucun replay
inter-vintages modèle ni nouvelle preuve d'absence de fuite n'est revendiqué.

## Frontend React et moteur local (`REPORT-024`, 21 novembre 2026)

La demande porte sur le composant d'analyse partagé, pas un habillage d'un
rapport unique. Le générateur canonique embarque désormais React 19.3.1 et
Lightweight Charts 5.3.0, sans Plotly, CDN ni police distante. Les sources, le
lockfile, les licences et le bundle reproductible vivent dans `analysis_ui/`.
Les anciens adaptateurs et scripts Python frontaux ont été retirés, après
inventaire de leurs lecteurs, avec récupération possible au commit `b658363`.
Aucune donnée, sortie archivée ou documentation historique n'a été supprimée.

- Le panneau commun comporte dates libres au jour, sélection par familles,
  capital et richesse relative commençant à 1, survol, zoom et drawdown en
  dessous. Jour/semaine/mois n'altère que la résolution visuelle. L'action
  « Analyser cette plage » applique explicitement le zoom aux KPI.
- Les nouvelles fenêtres sont calculées dans `portfolio/daily/`, via
  `reporting/strategy_analysis/`. Le service écoute seulement sur loopback,
  vérifie les artefacts hashés et ne permet que des lectures bornées. Ni
  téléchargement, ni exécution de modèle, ni ordre, ni publication tierce.
- Le fichier HTML garde son état initial et l'intégralité du payload source.
  Les **requêtes personnalisées nécessitent le service Python local**. Une
  erreur ou une couverture insuffisante ne laisse pas d'anciens KPI affichés
  comme ceux de la nouvelle période. L'archive mensuelle conserve ses chiffres
  et ses bornes pré-calculées, séparément du journalier.
- Le même panneau garde composition, corrélations de rendements et rendements
  actifs vs SPY, frontières quotidiennes, détails KPI et model cards Viridis.
  Portefeuilles, Méthodes et Sources restent accessibles par navigation.
  L'historique réalisé net de frais reste distinct du maintien récent
  hypothétique brut de frais, sans nouveau panier ou prolongation OOS inventée.

### Preuves d'interaction et de distribution

La QA réelle de `REPORT-024_20261121_02`, sur le service `127.0.0.1:8765`, est
conservée sous `outputs/performance_reports/REPORT-024_e2e_20261121_01/` :
`e2e.json`, SHA-256
`e5bcc0126c6bf1b634007c16207c1f276c7eab013faf5fbb2267438525663be0`,
et cinq captures Chrome desktop 1 440 px/mobile 390 px inspectées.
Aucune interception ou simulation de l'API n'a été utilisée.

- Fenêtres du 10 février au 17 août et du 13 mars au 22 mai 2026 : repères aux
  clôtures précédentes et égalité des KPI visibles avec des requêtes moteur
  indépendantes. La première contient 131 points, repère inclus.
- Résolutions jour/semaine/mois sans variation des KPI, capital relatif au SPY,
  vrai portefeuille deux Boosting, deux frontières de 168 points et changement
  effectif de leurs dates. Sources récentes demandées au 21 novembre, dernière
  clôture observée le 20 novembre, puis retour à l'historique.
- Fenêtre week-end sans séance : API 400, anciens graphiques et KPI retirés.
  Dates hors couverture refusées. Archives à 13 séries, 63 compositions et 168
  allocations, neuf méthodes et paniers actuels/historiques accessibles.
- Aucune erreur JavaScript, requête externe ou débordement global. La QA finale
  `REPORT-025` porte aussi sur les derniers ajustements de libellés : volumes
  d'archives dynamiques et absence de rééquilibrage dans le suivi récent.
- 127 tests Python ciblés réussis (portefeuille/reporting et contrôles sectoriels),
  15 tests d'inventaires, chemins et documentation, cinq tests JavaScript, build
  React, Prettier, Ruff et mypy du nouveau service validés. `npm audit --omit=dev`
  ne rapporte aucune vulnérabilité. Le wheel Python est construit et inspecté :
  bundle, CSS, versions, licences et service inclus ; aucun `node_modules` ni
  ancien adaptateur. Node n'est pas nécessaire pour ouvrir ou servir le rapport.
- Les inventaires code, dossiers et tests sont synchronisés. Les anciennes
  mesures de tests non relancés restent identifiées comme conservées ; les
  fichiers concernés et les fichiers nouveaux sont mesurés à nouveau.
  Zéro nouveau diagnostic Ruff ; les huit régressions de taille préexistantes
  restent hors des fichiers modifiés, sans nouvelle dérogation.

Les formules financières ne sont pas recodées en JavaScript. Aucun changement
des positions, prix ou résultats mensuels scellés, aucun replay inter-vintages
ni nouvelle preuve de causalité ; les trous détaillés dans `REPORT-023` restent
visibles. La livraison finale et ses hashes sont suivis dans `REPORT-025`.

### Fenêtre d'une seule séance (`REPORT-026`)

Le test ajouté avant correction reproduit une volatilité de 0 avec un unique
rendement quotidien. `close_window_metrics()` laisse désormais ce risque
d'échantillon à `null` ; le rendement total et le drawdown observé restent
disponibles. Deux observations réellement constantes peuvent toujours produire
une volatilité de 0. La convention mensuelle historique n'est pas changée.
Les deux cas sont testés, avec la suite portefeuille/reporting, Ruff, formatage
et mypy. Aucun prix, signal, portefeuille ni résultat mensuel modifié.

### Cohérence des libellés de source (`REPORT-027`)

La QA du premier candidat `REPORT-025_20261121` a détecté un bandeau composé et
une note de frontière disant « mensuel » dans le suivi récent, malgré une
sidebar correcte. Le moteur calculait déjà l'allocation à l'entrée sans
rééquilibrage. `sourcePolicy()` centralise les trois textes et distingue
historique net/mensuel de récent brut/entrée seule. Le test de présentation a
été ajouté avant correction ; six tests JavaScript, build, formatage et trois
tests Python de rendu passent. Chrome avec la vraie API confirme les libellés
des deux sources, sans modification des calculs. Le premier candidat et sa
preuve diagnostique restent conservés ; la livraison utilise un nouvel immuable.

## Livraison du rapport quotidien complet (`REPORT-025`, 21 novembre 2026)

Rapport livré :
`outputs/performance_reports/REPORT-025_20261121_02/backtest_performance_report.html`.
Consultation interactive : **`http://127.0.0.1:8765/`**, service loopback en
lecture seule. L'index local possédé désigne aussi ce rapport. Le fichier seul
conserve l'état initial et les archives ; les nouvelles fenêtres utilisent le
service Python. Aucun accès fournisseur ou modèle n'est déclenché à l'ouverture.

### Source, intégrité et information conservée

La source reste `REPORT-022_20261121_05`, SHA-256
`d3f02bdba00d934fae23ac924123035efab34720141bd74f1256b4c91fa8e800`,
même snapshot/composition `e0e6045a…ef95a` et mêmes fichiers prix scellés.
Le suivi récent conserve le reçu du 21 novembre au SHA
`9720db41b280ab73378d6e975084a6a9b566ac3432c367e01a3d4a7968a30a61`,
RAW `36882f7b…5a4c`. **Aucun retéléchargement ni rerun d'algo dans ce lot UI.**

La comparaison du payload complet est exacte, à l'exception de l'horodatage
technique et de la nouvelle branche `analysis`. Sont conservés :

- les 13 séries, 182 mois réalisés d'août 2011 à septembre 2026, 136 fenêtres,
  33 KPI et leurs **58 344 valeurs archivées** ;
- les 25 310 positions historiques, 126 positions figées d'octobre, huit
  méthodes historiques plus la méthode de suivi récent, soit neuf cartes ;
- les 63 compositions, 168 allocations et l'intégralité des données de leurs
  archives ; les 48 rendements récents hors benchmark sont inchangés.

Le ledger historique est **identique par SHA-256** à `REPORT-023` : 49 595
lignes, 2 307 fins de mois observables rapprochées, écart maximal
`5,273559366969494e-16`. Les 59 fins de mois sans clôture observable et les
155 clés prix absentes affectant 733 valorisations restent signalées, jamais
interpolées. Les KPI quotidiens nécessitant le chemin complet restent null.
Le snapshot, les positions, les prix et les résultats mensuels ne changent pas.

L'état initial affiche le backtest du 1er janvier au 30 septembre 2026. Le suivi
du panier figé va séparément jusqu'à la clôture du 20 novembre, même si le 21
novembre est demandé (samedi). Il ne constitue pas un portefeuille de novembre
recalculé ni une extension mensuelle réalisée. Les risques quotidiens diffèrent
des risques mensuels par construction : sur la fenêtre 2026, le max drawdown
Boosting Top 5 passe de −14,00 % mensuel à −22,53 % quotidien, tendance de
−26,27 % à −38,89 % et SPY de −5,76 % à −8,88 %, avec mêmes rendements totaux
aux bornes à moins de `1e-12`. Ce n'est pas un drift de portefeuille ou de données.

### Manifestes et preuves

| Artefact | SHA-256 |
| --- | --- |
| HTML complet, 3 834 026 octets | `7b19c3a81d41500581c81aa462e57add60c4bc398d9853eb731826df363fa8c6` |
| `analysis_manifest.json` | `e875543926a296652c6d9b90f7149a336902d8c2b5f8fbb778ec9a29d2be0188` |
| manifeste du rapport | `3023f9907fb18f1ac92cdfbdbffcb2cc858c33d3ccd45d7f40f274973e4a5325` |
| ledger historique | `dc2603e7d2320bffbad086666ed13b3f9bb54b4538ceb019d76d7eddf3f65fb8` |
| ledger récent, 494 lignes | `7194528f264c7082f97fbaaaae775d8e9173778ac8766cf1d1a4913dc31e8e71` |
| `qa_browser.json` | `f0335836d81ad3c0074b420f612d2e95303baf383b82505aa18d46ca14e2e446` |
| `validation.json` | `453c378c2dbdc113f60063b408a609057a1406c93aa52aa94d5c129ba9162315` |

Le manifeste capture les hashes du moteur, du service, du rendu et des assets,
ainsi que Python 3.11.15 et Polars 2.0.0. Les hashes sont revérifiés contre le
code livré par `REPORT-027` (`2524dc8`). `validation.json` atteste la parité du
payload, des ledgers et de trois requêtes HTTP avec un appel direct au moteur.

**146 tests Python ciblés réussis**, six tests JavaScript, build et formatage
frontend validés. La QA Chrome réelle (sans mock/interception) conserve treize
captures hashées et vérifie : fenêtres de milieu de mois, base 1, survol capital
et drawdown, molette puis « Analyser cette plage », invariance des KPI aux
résolutions, ratio SPY, deux Boosting, frontières à 168 points, source récente,
libellés de rééquilibrage/coûts, volatilité indisponible sur une séance, erreurs
sans anciens chiffres, archives et export CSV des cinq positions filtrées.
Desktop 1 440 px et mobile 390 px : aucune erreur JavaScript, requête externe
ni débordement global. Le HTML final est aussi vérifié en `file://` : état
initial lisible, service requis explicitement pour une nouvelle fenêtre.

Ruff, mypy et packaging sont détaillés dans `REPORT-024` ; les validations
documentaires, liens et inventaires sont réexécutées. Huit régressions de taille
globales préexistantes restent hors des changements de ce lot. Aucun nouvel
audit inter-vintages ni certification d'absence de fuite modèle n'est revendiqué.
Les anciens rapports, y compris le candidat diagnostique avant `REPORT-027`,
restent récupérables ; aucune donnée ou sortie historique n'est supprimée.

Reproduction sans acquisition ou entraînement :

```bash
.venv/bin/python scripts/research/build_daily_performance_report.py \
  --source-report outputs/performance_reports/REPORT-022_20261121_05/backtest_performance_report.html \
  --output-dir outputs/performance_reports/<nouveau-run-immuable>
.venv/bin/python scripts/research/serve_performance_report.py \
  --report-dir outputs/performance_reports/<nouveau-run-immuable> --port 8765
```

## Diagnostic des points absents et repli (`REPORT-028`, 21 novembre 2026)

La frontière vide signalée est reproduite sur le rapport `REPORT-025_20261121_02`
au SHA `7b19c3a8…fa8c6`, même snapshot `e0e6045a…ef95a`, du 1er janvier 2020 au
31 août 2026. Les 1 674 rendements quotidiens de Boosting Top 5 sont complets ;
les deux autres poches de la frontière ne le sont pas :

| Poche de la frontière | Titre | Séances manquantes | Poids cible du mois |
| --- | --- | --- | ---: |
| Legacy Frequency | NLSN.US | 13, du 13 au 31 octobre 2022 | 7,6923 % |
| Boosting tendance Top 5 | NLSN.US | les mêmes 13 séances | 20 % |
| Legacy Frequency | ABMD.US | 4, du 27 au 30 décembre 2022 | 5,1282 % |

Le plafond de 70 % interdit une allocation 100 % dans la seule poche complète.
Chacune des 168 combinaisons contient donc une poche à chemin incomplet : aucun
risque quotidien n'est optimisable. Ce n'est pas une absence de toutes les
données depuis 2020, ni un problème SEC. Le frontend antérieur masquait aussi
les références complètes lorsque la liste des mélanges calculables était vide ;
ce correctif de présentation est suivi séparément dans `REPORT-029`.

### Source des absences et limite économique

Les archives EODHD explicites `eodhd_output_20260402_170450` et
`eodhd_output_20260426_134831` contiennent les mêmes fins de séries, sans les
séances absentes. Nielsen a annoncé son rachat en numéraire le 11 octobre 2022,
28 USD par action et arrêt de cotation à partir du 12 octobre
([annonce Nielsen](https://www.nielsen.com/news-center/2022/nielsen-announces-closing-of-transaction-with-evergreen-and-brookfield-led-consortium/?redirected=1),
[8-K SEC](https://www.sec.gov/Archives/edgar/data/1492633/000119312522260583/d407513d8k.htm)).
Johnson & Johnson a finalisé Abiomed le 22 décembre 2022 pour 380 USD en
numéraire et un droit conditionnel non négociable
([annonce J&J](https://www.jnj.com/media-center/press-releases/johnson-johnson-completes-acquisition-of-abiomed)).

Une réacquisition de cours n'invente pas une cotation après ces événements.
La source archivée porte même des lignes plates post-événement : NLSN à 27,98
le 12 octobre et ABMD à 381,02 les 22–23 décembre. Les rendements mensuels de
ces positions reproduisent la dernière observation, pas une revalorisation
complète du cash/CVR. Le registre revu sait valoriser NLSN à 28 ; la ligne ABMD
actuelle ne bloque que les nouvelles entrées et ne valorise pas le CVR.
**Aucune de ces valeurs économiques n'est corrigée implicitement par ce lot.**
Leur rapprochement est suivi dans `DATA-048`, travail data/méthodologique distinct ; le repli garde
le statut de recherche et ne certifie pas des exécutions réelles.

### Moteur et preuve du repli

`monthly_reference` observe les rendements mensuels scellés, aux seules fins
de mois effectivement présentes dans le calendrier SPY : 2 379 lignes pour les
13 séries, repère initial compris. Il conserve les rendements nets et applique
le moteur commun à 12 observations/an, sans remplir le moindre prix quotidien.
Le CAGR garde une durée calendaire réelle et le drawdown inclut le capital
initial ; les anciens 33 KPI archivés sont conservés séparément et inchangés.

Le service exige des mois calendaires complets. Pour un intervalle partiel,
il ne recadre rien à la demande : il propose des mois inclus, que l'utilisateur
pourra appliquer explicitement. Exemple du 10 janvier 2020 au 17 août 2026 :
proposition du 1er février 2020 au 31 juillet 2026. Le récent hypothétique reste
une autre source, sans ce repli. L'audit titre/séance est embarqué dans le HTML
hashé puis joint aux holdings scellés, sans lookup mutable.

Sur 2020–août 2026, le journalier strict conserve ses 1 675 points et zéro
mélange quotidien calculable ; le repli mensuel contient 81 points, repère
inclus, et **168 mélanges calculables**. L'audit conserve les trois lignes
NLSN/ABMD ci-dessus dans les deux modes. Le code de rendu et la livraison finale
avec la QA navigateur sont suivis dans `REPORT-029`.

**Validation du moteur** : 120 tests portefeuille/reporting réussis, 15 tests
d'inventaires/chemins/documentation, Ruff, formatage, complexité et mypy des
15 modules concernés. Les 2 366 rendements de la nouvelle grille mensuelle
retrouvent les valeurs scellées avec écart maximal `3,3306690738754696e-16`.
Les séries quotidiennes de l'état initial 2026 sont exactement identiques à
`REPORT-025`, champs et points compris. Le code laisse l'exposition du nouveau
mode au frontend à `REPORT-029` : l'ancien bundle ne reçoit pas un sélecteur
mensuel dont il ne saurait pas étiqueter la fréquence. Les huit régressions de
taille globales préexistantes sont inchangées ; aucune nouvelle n'est introduite.

## Livraison du diagnostic et du repli (`REPORT-029`, 21 novembre 2026)

Le rapport complet corrigé est
`outputs/performance_reports/REPORT-029_20261121/backtest_performance_report.html`,
servi par le même viewer local `http://127.0.0.1:8765/`. Le générateur commun
expose maintenant la source `monthly_reference` avec le bundle React capable
d'en nommer la fréquence. Les anciens HTML restent intacts.

- Le journalier garde ses vrais trous, mais les références complètes SPY et
  Boosting Top 5 ne sont plus masquées lorsqu'aucune combinaison n'est calculable.
- Le panneau de couverture est filtré aux stratégies affichées, aux poches du
  composé ou aux trois poches de la frontière. Il donne titre, mois, séances,
  dates détaillées et poids. La référence mensuelle garde ce diagnostic visible.
- Le bouton de repli annonce les bornes avant le clic. Sur 2020–août 2026, il
  conserve les mêmes bornes et rend les 168 combinaisons calculables, aux seules
  80 observations mensuelles plus le repère. Sur des bornes partielles, seule
  une action explicite applique les mois complets inclus proposés par Python.
- Source, KPI, VaR/CVaR, corrélations, composer, model cards et frontières
  portent tous la fréquence choisie. Aucun rendement n'est calculé en React.
  Le zoom mensuel reste une loupe, sans action d'application au jour près.
- Les champs de dates modifiés mais non appliqués sont désormais signalés.
  La capture utilisateur montrait ainsi une fin saisie en février différente
  de la fenêtre encore calculée jusqu'en août ; ce n'était pas un nouveau run.

### Parité et artefacts

| Artefact | SHA-256 |
| --- | --- |
| HTML complet, 3 846 164 octets | `3e49631481a32ee4aee009b3e284464a8ae44ae1cba19688bb4516d455e2d66d` |
| manifeste d'analyse | `30a65494dacb50f7bcf115bb046d52597627cc275d73b1ed910d73af5e9dcb5d` |
| manifeste du rapport | `c13faed9b953ffcd59889bce1826a302ddc44f61d71a2f1693b2a4c671413cc2` |
| référence mensuelle, 2 379 lignes | `54cba74f2d38b5a403f642724e232f16e2f56b7a77e26ab1b7504d439b769d02` |
| QA navigateur | `d912804c515a64bfa14f1c0d585c4dfc104ac56862ac97e2484f3270a61648d5` |
| validation | `4d208395a1674fc36b350e18313530f9d0504afc9c42b8226f448b9834bf21fe` |

Le payload source est identique à `REPORT-025` hors horodatage technique et
branche `analysis` enrichie : 13 séries, 25 310 positions historiques, 126
positions figées, 58 344 KPI mensuels archivés, 48 rendements récents hors SPY,
63 compositions et 168 poids conservés. Les neuf cartes méthodologiques
existantes restent présentes, avec une explication du repli ajoutée.
Les ledgers quotidien/récent, le rapprochement mensuel et l'audit de trous
conservent leurs hashes de `REPORT-025`. Les champs et points quotidiens de
l'état initial 2026 sont exactement identiques. Aucune donnée, position,
sélection, sortie ou hypothèse terminale existante n'est réécrite.

**154 tests Python ciblés et 11 tests JavaScript réussis** ; le test de rendu
des références seules a échoué avant correction (zéro SVG au lieu de deux).
Build, Prettier, syntaxe du bundle et validations documentaires/liens passent.
Chrome avec la vraie API, sans mock, vérifie quotidien strict, deux références,
audit NLSN/ABMD, 81 points mensuels et 168 mélanges, labels/corrélations mensuels,
composition, repli sur sous-fenêtre inclusif seulement après clic, réponse 400
sans anciens chiffres sur une borne mensuelle invalide, retour au journalier,
suivi récent inchangé et archives. Cinq captures hashées desktop/mobile sont
inspectées ; aucune erreur JavaScript, requête externe ou débordement global.

Les limites de risque mensuel et la dette économique `DATA-048` restent
explicites. Aucun téléchargement ou modèle relancé, aucun nouvel audit
inter-vintages ni promotion. Un incident de capture `ENOSPC` a été contourné
en limitant le cache temporaire Chrome, sans suppression ; la QA finale est
complète. Le volume Data du Mac ne dispose plus que d'environ 280 Mio libres :
ce risque machine est distinct du calcul et aucun nettoyage utilisateur n'a été
entrepris. Les régressions globales de taille préexistantes restent inchangées.

## Catalogue des vrais backtests (`REPORT-030`, 22 novembre 2026)

La navigation demandée porte sur les versions économiques, pas sur les refontes
successives du HTML. Trois sources explicites sont capturées dans
`outputs/performance_reports/REPORT-030_20261122_02/`, sans acquisition,
entraînement, modification d'archive ou promotion :

| Replay central | Identifiant stable | Snapshot | Séries | Positions historiques |
| --- | --- | --- | ---: | ---: |
| 31 octobre | `bt-c4b1904c147b215b` (défaut) | `e0e6045a…ef95a` | 13 | 25 310 |
| 24 octobre | `bt-12c2fcb8f48eed8b` | `c340a9b4…8b7e` | 11 | 24 650 |
| 8 octobre | `bt-1824905b20bc3f38` | `dbaac67b…e598` | 11 | 24 650 |

Sources immuables, toutes sous `outputs/performance_reports/` :
`REPORT-029_20261121/backtest_performance_report.html` (`3e496314…d66d`),
`data_refresh_20261024/backtest_performance_report.html` (`d7b0a8c8…5105`) et
`data_refresh_20261008_095842/backtest_performance_report.html` (`e4e9dfd2…1a92`).
Les 38 références de lignée et les six fichiers prix sont hash-conformes ;
la préparation quotidienne vérifie aussi les rapprochements mensuels.
Les trois versions gardent leurs 182 mois d'août 2011 à septembre 2026.

Chaque champ économique du payload source est identique après capture, hors
date technique de rendu et enrichissement `analysis`. Les ledgers historique
et récent de la version par défaut sont byte-identiques à `REPORT-029`.
Les copies occupent de nouveaux dossiers, les anciens HTML restent intacts.
Un rendu ou suivi du panier figé ne change pas l'identifiant économique ; le
catalogue refuse de les compter comme deux backtests.

Les API restent attachées à la version choisie. La comparaison impose les
mêmes bornes demandées/effectives et la même fréquence ; ni recadrage caché,
ni suivi hypothétique, ni politique de frontière absente inventée. Le cas réel
janvier–septembre 2026 rapproche les 11 séries communes du 31/24 octobre et
compte 639 clés de position/poids différentes dans 84 paniers. Il s'agit d'un
rapprochement d'archives, pas d'un nouvel audit de causalité.

Limites : Legacy Equal ancien conserve ses frais archivés nuls, contre
10 pb × turnover dans la version unifiée ; les deux variantes sectorielles
sont absentes des anciennes versions. Code/runtime, registre terminal et
catalogue de paires ont aussi changé. Aucun écart n'est présenté comme
« data-only » ; les trois versions restent de recherche, non promues.
Le lot n'ajoute aucun portefeuille de novembre ni rendement récent à
l'historique net. La navigation utilisateur est suivie dans `REPORT-031`.

| Artefact | SHA-256 |
| --- | --- |
| HTML par défaut | `8cce59750cd63147720a58ce94ba25a645f5a883ff286abee42dc4bb6ad1efce` |
| manifeste d'analyse | `0c8094a6cc860dd8a853a7995186d8e5f6dbfd25df1cb67c7cc44e47d5a42697` |
| catalogue de versions | `2621a7d333baff19f66b566635d3d1cc3aeb4b126f173c423051739d3c6c524d` |

Validation : 144 tests portefeuille/reporting réussis, dont identité,
immuabilité, doublons économiques, hashes/chemins, routing HTTP, comparaison
quotidienne/mensuelle aux dates exactes, absence de frontière et rejet d'une
requête hors couverture. Ruff/complexité, mypy des neuf modules, documentation,
liens et inventaires contrôlés. Les huit régressions globales de taille déjà
signalées restent hors des fichiers modifiés ; aucune nouvelle n'est ajoutée.

## Navigation par versions et livraison (`REPORT-031`, 22 novembre 2026)

Rapport complet livré :
`outputs/performance_reports/REPORT-031_20261122_02/backtest_performance_report.html`.
L'entrée locale et le service canonique conservent `http://127.0.0.1:8765/`.
Le sélecteur global remplace le menu « Données » et pilote analyse,
portefeuilles, méthodes et sources. Cinq vrais vintages sont capturés :

| Replay central | Version économique | Fin de performance | Séries | Positions historiques |
| --- | --- | --- | ---: | ---: |
| 31 octobre | `bt-c4b1904c147b215b` (défaut) | 30 septembre 2026 | 13 | 25 310 |
| 24 octobre | `bt-12c2fcb8f48eed8b` | 30 septembre 2026 | 11 | 24 650 |
| 8 octobre | `bt-1824905b20bc3f38` | 30 septembre 2026 | 11 | 24 650 |
| 9 septembre | `bt-4fb54c864abd03de` | 31 août 2026 | 11 | 24 522 |
| 29 août | `bt-82d28c4aff74fc2e` | 31 juillet 2026 | 11 | 24 392 |

Les trois premières sources et leurs snapshots sont ceux de `REPORT-030`.
Les deux archives supplémentaires sont, sous `outputs/performance_reports/`,
`data_refresh_20260909/backtest_performance_report.html`
(`632961099368de7f02f82a058bd10612f7bdfd70c530c0d5fe7d023519f3353d`)
et `sats_echo_20260830_current_portfolio/backtest_performance_report.html`
(`51675425c854eb9f227c74756a049f380963d2656195de6d8e6027599425a871`).
Leurs compositions sont respectivement `35f0244f…8421` et `bb1f90a9…8375`.
Le rapport du 30 août porte le replay du 29 et conserve le composeur/panier
archivé : sa date de rendu n'est pas utilisée comme date économique.

### Parcours et limites visibles

- La date d'arrêt du backtest est visible dès l'ouverture ; un prix récent
  du panier figé ne la remplace plus dans l'en-tête. Chaque version conserve
  sa vraie couverture et ses stratégies, sans ajout rétroactif de secteur.
- Début/fin et Jour/Semaine/Mois restent des réglages d'analyse. Le repli
  mensuel est proposé seulement dans le diagnostic ; son retour au journalier
  conserve exactement les bornes explicitement adoptées.
- Le suivi d'octobre, jusqu'au 20 novembre, est un panneau dépliable séparé,
  marqué « hors backtest ». Son maintien en novembre n'est jamais un nouveau
  signal ou un rééquilibrage de novembre.
- L'onglet Versions compare les archives sur les mêmes dates/fréquence et
  montre les écarts de code, snapshot, conventions et frais. Les exceptions
  Legacy Equal restent visibles ; aucune attribution « data-only » implicite.
  Les variantes absentes de l'autre version sont nommées, pas complétées.
- Les couvertures plus courtes sont refusées sans recadrage. La comparaison
  avec la version d'août refuse septembre et ne passe qu'après choix explicite
  d'une fin au plus au 31 juillet ; celle de septembre exige au plus le 31 août.
  Un retry à choix inchangés fonctionne aussi après une erreur.
- Les réponses doivent correspondre aux identifiants et à la fenêtre demandés.
  Aucun résultat précédent n'est présenté sous de nouveaux choix appliqués.
  Sources et méthodes restent sélectionnées lors d'un changement de version.

### Preuves et conservation

Le fichier `validation.json` vérifie chaque clé du payload source hors
horodatage de rendu et enrichissement `analysis` pour les cinq versions ;
zéro champ économique modifié. Les prix et rapports source sont inchangés.
Les cinq artefacts quotidiens/mensuels, de réconciliation et de trous de la
version par défaut sont byte-identiques à `REPORT-029`.
Les API retrouvent exactement les moteurs directs pour les cinq versions sur
le 11 mars–23 juillet 2026, ainsi que pour les huit comparaisons
quotidiennes/mensuelles du défaut contre les quatre archives sur janvier–juillet.
Les hashes du code capturé correspondent aux sources exécutées.

| Artefact final | SHA-256 |
| --- | --- |
| HTML complet, 3 865 453 octets | `f34ac2e639a939aa659d1e71d83eb7bab1fede955901946182d4d36c0ef7e247` |
| manifeste d'analyse | `c437810660949c764b134fd18a188c2349c10e3ec963ac504670240ce3ccd2be` |
| catalogue des cinq versions | `194b5faaaf8688a6ee12e641eca9e30d5fb4392fd309c6e881b228db54938810` |
| validation/parité | `1fd4a0790f0de8bd601a790030a052a55c3b14823e7bcdfc079e0b2f9c3b4830` |
| QA réelle et cinq captures hashées | `991d12620cb0995807ee6a16fbea70c4e9b7c6648aa63824fb491dd40ad2d32d` |
| bundle React | `9f1b4b9eadcf26c8a46e5cfee1c252bbd512925e770f0d39f07552869a1a060e` |
| CSS | `ac954df090af2d42a2dcebecf3fd70997bef08f3ec3530cb771d9a55e2e15384` |

Validation : 159 tests Python ciblés, 15 tests JavaScript, formatage et build
verrouillé, contrôles documentaires/liens et syntaxe du bundle. Le test qui
refuse le menu « Données » a échoué avant correction. Chrome réel, API réelle,
sans mock, valide cinq cycles de versions, onglets Sources/Méthodes conservés,
dates libres, repli/retour, composé, frontières, suivi séparé, archive mensuelle,
CAGR/volatilité/drawdown entre versions, erreurs sans vieux chiffres, retry et
mobile 390 px. Les cinq captures sont inspectées ; zéro erreur JS, appel externe
ou débordement global. Les huit régressions Python de taille préexistantes
restent hors du lot ; aucune nouvelle dette Python n'est introduite.

Pas de téléchargement, nouveau signal, replay modèle, promotion ou correction
terminale `DATA-048`. La fin de performance de la version la plus récente reste
le 30 septembre : ce lot rend ce retard explicite et ne le masque pas par un
panier figé ou une date d'interface. Les sorties de preview et les anciennes
versions sont préservées. Le disque dispose d'environ 395 Mio libres après QA ;
aucune donnée utilisateur n'a été supprimée pour livrer ce changement.

## 22 novembre — risques quotidiens avec dernier cours conservé (`REPORT-032`)

### Demande, périmètre et versions

Le propriétaire demande de calculer le rendement du portefeuille même les
séances où un titre n'a plus de clôture. Le moteur `METH-011` est livré avec la
politique explicite `last_observed_mark_v1` ; `REPORT-032` la capture dans le
HTML complet existant, sans créer un second moteur ou un dashboard réduit.

Livrable : `outputs/performance_reports/REPORT-032_20261122/`.
La version par défaut `bt-a3dae277cf4646f8` est un **scénario de valorisation** du
même replay que la version stricte `bt-c4b1904c147b215b`, également conservée
dans son catalogue. Les quatre autres vintages d'août à octobre restent
accessibles : six versions au total, treize séries dans le défaut, tous les
panneaux, méthodes, pseudo-codes, archives et portefeuilles préservés.

Il n'y a ni téléchargement, ni réapprentissage, ni nouvelle sélection, ni
promotion. Le snapshot reste `e0e6045a…ef95a`, l'historique réalisé va du
1er août 2011 au 30 septembre 2026, les coûts mensuels et SPY total return sont
inchangés. Le suivi du panier archivé d'octobre, prix au 20 novembre, reste
séparé et hypothétique au-delà du mois prévu. Aucun résultat n'est inventé
pour 2010 ni après septembre dans le backtest.

### Règle effectivement implémentée

Seul le titre sans clôture conserve son dernier **cours ajusté déjà connu**
depuis l'ancre du mois. Son facteur de prix brut reste à 1 (rendement simple
nul) pendant le trou ; les autres actions varient, les poids dérivent et les
frais archivés restent débités. Une reprise de cotation conserve tout son
rattrapage. Une ancre d'entrée ou un benchmark absent reste bloquant.

Ce n'est **pas** une vente réelle au dernier cours ni du cash garanti. Les
points concernés et leurs risques portent le statut estimé, dans les KPI et
dans un diagnostic limité à la sélection et aux dates actives. Chaque repli
expose son titre, poche, poids, séance, date du dernier cours et valeur dans
les unités ajustées du snapshot. Les écarts mensuels sont rapprochés ; le
capital et les compositions suivent le rendement revalorisé, pas un retour
artificiel vers l'ancien capital scellé.

La correction des contreparties terminales reste `DATA-048`, distincte du
scénario. Les [28 USD de NLSN](https://www.sec.gov/Archives/edgar/data/1492633/000119312522260583/d407513d8k.htm)
ne sont pas une liquidation observée à 27,98 ; les
[380 USD et le CVR non coté d'ABMD](https://www.sec.gov/Archives/edgar/data/815094/000119312522311074/d353287d8k.htm)
ne sont pas une valeur récupérable égale à 381,02. Surtout, la
[protection des déposants de Silicon Valley Bank](https://www.fdic.gov/news/press-releases/2023/pr23017.html)
ne protégeait pas les actionnaires : figer SIVB peut fortement sous-estimer le
risque. Aucun cash de faillite, CVR nul ou titre successeur n'est supposé dans
ce lot. Les unités raw/adjusted et les dates de connaissance restent à traiter
dans la correction sourcée, sans modifier les archives.

### Replay de valorisation et parité

`valuation_replay_validation.json` vérifie les treize séries sur tout
l'historique et compare les 25 310 positions : **zéro changement de titre,
poids ou panier**. Les allocations courantes sont aussi identiques. Toutes les
clés économiques du payload initial restent conservées ; seules la convention
explicite de valorisation, ses vues et la date de rendu sont ajoutées.

- 155 clés titre/séance absentes, réparties sur quatorze titre/mois ;
- 733 NAV de portefeuille auparavant manquantes deviennent estimées, sur
  49 595 lignes, avec zéro NAV restante manquante ;
- 2 366 rendements mensuels rapprochés : 2 307 fins observées et 59 fins
  estimées ; écart maximal `5.273559366969494e-16`, inférieur à `1e-12` ;
- tous les risques quotidiens deviennent calculables dans ce scénario, mais
  ne deviennent pas pour autant intégralement observés ;
- les 168 candidats de frontière ont des risques calculables sur la fenêtre
  testée ; le portefeuille 50/50 Boosting Top 5 / tendance Top 5 conserve le
  rééquilibrage mensuel et le statut estimé de ses séances touchées.

Les quatorze groupes sont JCP janvier 2014 (1), KRFT juillet 2015 (20), GMCR
mars 2016 (20), WFM août 2017 (4), CSRA avril 2018 (19), ESRX décembre 2018 (5),
APC août 2019 (16), TSS septembre 2019 (9), XLNX février 2022 (7), NLSN octobre
2022 (13), ABMD décembre 2022 (4), SIVB mars 2023 (16), PXD mai 2024 (20) et
FISV novembre 2025 (1). Ils ne sont pas tous des faillites ou des titres sans
valeur : JCP/FISV requièrent notamment un diagnostic de trou/continuité, et les
opérations cash/actions doivent rester distinguées du scénario.

### Fenêtre de la capture utilisateur : janvier 2022–septembre 2026

Valeurs **estimées sous la politique décrite**, mêmes dates, mêmes frais et
benchmark. CAGR sur durée réelle ; volatilité sur 252 séances ; drawdown avec
capital initial ; années complètes 2022–2025 uniquement pour les deux dernières
colonnes. 2026 reste visible mais partielle.

| Série | CAGR | Volatilité | Max drawdown | Pire année | Années devant SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy Frequency | 49,76 % | 39,82 % | −38,09 % | +2,20 % | 3 / 4 |
| Tendance Top 5 | 53,84 % | 43,97 % | −38,89 % | −0,20 % | 3 / 4 |
| Tendance Top 5 sectoriel | 28,12 % | 32,86 % | −32,74 % | −10,55 % | 3 / 4 |
| SPY total return | 11,97 % | 17,33 % | −24,50 % | −18,18 % | 0 / 4 |

Le `2 / 2` et la pire année Legacy de +36,90 % de l'ancien affichage omettaient
les années dont les bornes étaient manquantes. Désormais, le mode strict rend
ces résumés indisponibles en nommant les années non calculables, plutôt que les
supprimer. Le scénario réparable réintègre les quatre années et donne les
chiffres ci-dessus. Cela corrige une présentation, pas une sélection du modèle.

### Preuves capturées

Le HTML source strict `REPORT-031` garde son hash `f34ac2e6…e247`.
Les 24 hashes de code capturés correspondent aux sources réellement exécutées.

| Artefact | SHA-256 |
| --- | --- |
| HTML complet, 3 872 072 octets | `1ee5b9b78f7f336245c08fe736f69d141550bb07fba686f8c631e53e8cf28c09` |
| Manifeste d'analyse | `64a53dcf7830b17153fd059eef839bc3a4b5df9428c28c95c6d21687a11ac8bb` |
| Catalogue des six versions | `0fa952e37063fe19397c4b56dd35e5b35a701b4e13c1ff69a8ae59e212a2234f` |
| Validation du replay de valorisation | `14d4a2d8bf08cfa717a907e48789fdd0169390c95ce6d711fc61a4033ed70e16` |
| Journal des cours conservés | `07922e2275046e2e494ebed9eae8888f4d64669968d863d7b410d157ad7e84aa` |
| Rapprochement mensuel | `85806daea828f2415cd6b592be9203638b32b990679a3d707c9e814c04a3c437` |
| Bundle React | `beffbd2447ddc82cb5f702f82913bb227279bc81cdf932c1d612060ef5dfa24b` |
| CSS | `c68e7600db83ea908e52b4ae84329b66530226b4e5c927eaa3290325bdc6c134` |

Validation automatisée : 157 tests Python portefeuille/reporting/catalogue,
18 tests JavaScript, mypy sur 17 modules, Ruff et complexité ciblés, build
verrouillé et formatage. Les tests de capture/politique et les trois tests UI
ont échoué avant implémentation. Aucun nouvel écart de taille Python ; les
huit régressions antérieures hors périmètre ne sont pas masquées.

Ce replay n'est pas un audit causal inter-vintages des signaux : il n'entraîne
aucun modèle et ne télécharge aucun nouveau snapshot. Les 25 310 positions
identiques prouvent la préservation des paniers scellés dans ce lot, pas une
nouvelle garantie générale d'absence de fuite en amont.

QA réelle : `qa_browser.json`, SHA-256
`4c415ad832a2061c5bdcef2f80900f829fc2a8dc209b55b5478e01df387c9fff`,
avec cinq captures hashées et inspectées (scénario/strict/frontières desktop,
scénario/frontières mobile 390 px). Elle confirme les risques et badges des
trois séries de la capture, les quatre lignes NLSN/ABMD et leurs journaux, la
base 1 au 31 décembre 2021, le retour au strict, le composé 50/50, les 168 mixes
et les archives de positions/méthodes. Le comparateur rapproche 7 668 positions
sur 2022–septembre 2026 sans différence de titre/poids, avec le seul contrat
`daily_valuation` changé. Le texte et les six étapes du scénario restent
distincts des neuf cartes méthodologiques initiales ; aucune carte supprimée.
Zéro erreur JS/console, appel externe ou débordement global. Le serveur de QA
8766 est arrêté ; le rapport complet et ses calculs restent servis sur le port
canonique 8765.

Fichiers du lot : `scripts/research/build_daily_performance_report.py` ; sous
`reporting/strategy_analysis/`, `dataset.py`, `coverage.py`, `service.py`,
`report.py`, `versions.py`, `version_catalog.py` et README ; sous
`reporting/analysis_ui/`, `coverage.jsx`, `tables.jsx`, `versions.jsx`,
`audit.jsx`, `layout.css`, tests et bundle verrouillé. Le test de service, les
inventaires de dépendances/collection/catalogue, `ROADMAP.md`, le standard de
reporting et cette preuve sont mis à jour. Les sorties et les captures restent
ignorées par Git ; leurs hashes sont suivis ici, sans indexer les données.

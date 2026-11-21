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

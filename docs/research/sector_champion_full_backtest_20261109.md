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

Reproduction (choisir une nouvelle destination pour conserver cet artefact) :

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

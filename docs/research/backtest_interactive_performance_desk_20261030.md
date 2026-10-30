# Desk de performance interactif du 30 octobre 2026

## Statut

Cette preuve clôt `REPORT-017`. La refonte applique le processus
`frontend-design` en deux passes : direction visuelle explicite, puis critique
sur captures desktop/mobile avant finalisation. Elle conserve le payload
économique et toutes les vues d'audit du rapport précédent.

## Direction et simplification

L'objet est un outil de décision pour un investisseur quantitatif : choisir une
ou plusieurs stratégies, les comparer à SPY sur une fenêtre, puis approfondir
le risque. Le graphique est donc la surface dominante. Le premier écran ne
reproduit plus la structure technique du payload.

La palette froide conserve le navy AlphaRank, l'or SPY, le vert et le rouge
économiques. IBM Plex Sans porte toute la lecture ; le mono reste limité aux
valeurs, dates et axes. Les éléments génériques retirés sont le hero décoratif,
les cartes de couverture, les six cartes KPI, les labels numérotés et les
menus redondants.

Les dix stratégies sont maintenant des chips visibles et activables en un
clic. Les fenêtres complète, 2015, 2020 et 2023 sont directes ; une période
personnalisée reste disponible. Le résumé principal indique le meilleur CAGR
et le nombre d'indicateurs clés qui surpassent SPY. Les six KPI forment un
scoreboard `stratégies × métriques` sous le graphique ; les 33 KPI, model cards,
corrélations, holdings, méthodologies et hashes restent dans leurs vues
détaillées.

## Entrées et parité

- replay commun :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/common_end_2026_09_trend_clean` ;
- run Legacy :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/legacy/2026-10-24/runs/20261024_172912` ;
- composition :
  `c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e` ;
- couverture : 11 séries, 33 KPI, 182 mois, 24 650 holdings historiques et
  126 holdings courants.

Le payload de `REPORT-016` et celui de `REPORT-017`, comparés hors
`generated_at_utc`, sont exactement égaux. Aucun score, rendement, portefeuille,
snapshot ou statut de promotion ne change.

## Artefacts et QA

- rapport :
  `outputs/performance_reports/interactive_performance_desk_20261030/backtest_performance_report.html` ;
- SHA-256 du rapport :
  `bf9c79b63d2fe4d736ea8024a6ed82a5e1a5ca158bb303f57b580a990bfbb2b0` ;
- manifeste :
  `outputs/performance_reports/interactive_performance_desk_20261030/backtest_performance_report_manifest.json` ;
- SHA-256 du manifeste :
  `5e4f4e2403185185a0d0228468c7d2aecb8a03c199493a5b8c31e76fb2389684`.

Les 19 tests reporting, Ruff, le format, la documentation, les liens et la
syntaxe des 15 911 891 caractères JavaScript passent. La QA navigateur valide
les dix chips, l'ajout d'une troisième stratégie, le passage à 2020, la mise à
jour du graphique et du scoreboard, le mode Composer et ses corrélations, ainsi
que l'absence d'erreur console et de ressource réseau externe.

À 1 139 × 863, le document et le scoreboard n'ont aucun débordement. À
390 × 844, les trois modes restent sur une ligne, les stratégies deviennent un
ruban tactile, aucune section n'est cachée et le document n'a aucun débordement
global ; seul le scoreboard détaillé défile localement, comme les autres tables.

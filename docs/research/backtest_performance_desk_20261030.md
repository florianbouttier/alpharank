# Performance desk du 30 octobre 2026

## Statut

Cette preuve clôt `REPORT-016`. Elle corrige la refonte `REPORT-015`, dont le
rail latéral et les cartes horizontales restaient trop proches du rapport
historique et réduisaient la largeur utile du graphique. Le nouveau rendu est
un poste d'analyse pleine largeur. Aucun payload économique n'est modifié.

## Direction de conception

Le dashboard suit le workflow `build-dashboard` : contexte et fraîcheur en
haut, décision et filtres ensuite, mouvement principal dans un graphique, KPI
de diagnostic dans le même viewport, puis détails et preuves. La navigation
est horizontale pour restituer 248 pixels au studio. Le graphique occupe la
colonne principale ; les six KPI synthétiques forment une grille adjacente et
ne déclenchent plus de défilement horizontal.

La palette, IBM Plex Sans/Mono, les couleurs stratégie/SPY et les états de
surperformance restent ceux d'AlphaRank. La mise en page évite le rail fixe,
les cartes molles répétitives et le hero marketing. La seule surface forte est
le poste graphique/KPI ; les contrôles et métadonnées restent sobres.

## Entrées et parité

- replay commun :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/common_end_2026_09_trend_clean` ;
- run Legacy :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/legacy/2026-10-24/runs/20261024_172912` ;
- composition :
  `c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e` ;
- couverture : 11 séries, 33 KPI, 182 mois, 24 650 holdings historiques et
  126 holdings courants.

Le payload de `REPORT-015` et celui de `REPORT-016`, décodés puis comparés hors
`generated_at_utc`, sont exactement égaux. La refonte ne change donc ni score,
ni rendement, ni portefeuille, ni fenêtre, ni statut de promotion.

## Artefacts et QA

- rapport :
  `outputs/performance_reports/performance_desk_20261030/backtest_performance_report.html` ;
- SHA-256 du rapport :
  `8ead3f81143f0367c820cb6dc28a7c3ce34a16e8efa414072127e568e35b4a53` ;
- manifeste :
  `outputs/performance_reports/performance_desk_20261030/backtest_performance_report_manifest.json` ;
- SHA-256 du manifeste :
  `829e4e02a226c8b0e7b21c2983ff7a1b48799c942c2c984a3c42e6d342ff7618`.

La QA navigateur HTTP locale valide à 1 139 × 863 une colonne graphique de
642 pixels, une colonne KPI de 390 pixels, six cartes visibles, zéro
débordement global et aucune erreur console. Les modes stratégies, portefeuille
composé et frontières fonctionnent ; la richesse relative, les corrélations,
les trois recommandations et les 15 lignes multi-horizon sont présentes.
Les 19 tests reporting, Ruff, le format, les validateurs documentaires et la
syntaxe des 15 909 451 caractères JavaScript extraits du rapport final passent.

À 390 × 844, la navigation reste accessible par défilement local masqué, le
studio passe sur une colonne, les KPI sur deux colonnes, aucune section n'est
cachée et le document n'a aucun débordement horizontal global. Le rapport
historique du 8 octobre n'est pas écrasé : le nouvel artefact est ouvert dans
un nouvel onglet visible et conservé comme livrable de la tâche.

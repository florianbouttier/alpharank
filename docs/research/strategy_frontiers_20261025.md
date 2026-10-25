# Frontières efficientes multi-stratégies du 25 octobre 2026

## Statut

Cette preuve clôt `REPORT-014`. Elle ajoute un diagnostic post-hoc au rapport
de performance ; elle ne promeut aucune allocation et ne modifie ni les
signaux, ni les portefeuilles sources, ni le snapshot publié.

## Entrées et calendrier

- replay commun candidat :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/common_end_2026_09_trend_clean` ;
- run Legacy candidat :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/legacy/2026-10-24/runs/20261024_172912` ;
- composition immuable :
  `c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e` ;
- calendrier réalisé commun : 182 mois, d'août 2011 à septembre 2026 ;
- portefeuille courant séparé : décision septembre 2026, détention octobre
  2026, preuve de marché au 23 octobre, rendement encore exclu des KPI.

Le replay source est mécaniquement comparable mais reste
`publication_eligible=false` et `post_hoc_research_diagnostic`. Le rapport ne
change pas ce statut.

## Contrat de composition

Les poches sont `Legacy · Frequency`, `Boosting · Top 5` et
`Boosting tendance · Top 5`. Les 168 compositions admissibles sont longues,
sans levier, par pas de 5 %, plafonnées à 70 % par poche et rééquilibrées
mensuellement. Elles combinent les rendements déjà nets de chaque stratégie,
sans coût inter-poche supplémentaire. Le moteur canonique calcule CAGR,
volatilité, max drawdown, Sharpe et Calmar sur 136 fenêtres ; le navigateur ne
recalcule aucun KPI.

## Lecture de la fenêtre la plus longue

La demande « 2010 à aujourd'hui » commence effectivement en août 2011, premier
mois comparable.

| Profil post-hoc | Legacy | Boosting | Tendance | CAGR | Volatilité | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| volatilité minimale | 70 % | 30 % | 0 % | 22,07 % | 24,89 % | -22,52 % |
| Sharpe maximal | 25 % | 30 % | 45 % | 24,08 % | 26,47 % | -21,87 % |
| Calmar maximal | 15 % | 35 % | 50 % | 24,33 % | 26,78 % | -21,87 % |
| SPY, référence non optimisée | — | — | — | 14,40 % | 14,23 % | -23,93 % |

La composition Calmar maximise ici le couple croissance/drawdown parmi la
grille et la composition Sharpe est presque identique en risque. SPY conserve
toutefois une volatilité nettement plus faible et un Sharpe de `0,871`, contre
`0,834` pour le meilleur candidat Sharpe : « efficient entre les trois poches »
ne signifie donc pas supérieur au benchmark sur chaque dimension.

## Sensibilité à la date de départ

| Début effectif | Composition Sharpe maximal Legacy / Boosting / Tendance | CAGR | Volatilité | Max drawdown |
| --- | --- | ---: | ---: | ---: |
| août 2011 | 25 % / 30 % / 45 % | 24,08 % | 26,47 % | -21,87 % |
| janvier 2015 | 35 % / 35 % / 30 % | 26,84 % | 27,46 % | -21,42 % |
| janvier 2020 | 30 % / 10 % / 60 % | 35,02 % | 36,83 % | -23,48 % |
| janvier 2023 | 35 % / 10 % / 55 % | 53,67 % | 40,07 % | -23,44 % |
| janvier 2025 | 50 % / 5 % / 45 % | 102,58 % | 52,90 % | -25,12 % |

La poche tendance prend davantage de poids sur les fenêtres récentes, mais les
CAGR très élevés des horizons courts sont annualisés sur peu d'observations et
la volatilité augmente fortement. La recommandation de recherche pour une
lecture longue est donc de comparer en priorité les profils `25/30/45` et
`15/35/50`, pas de retenir automatiquement l'optimum d'une fenêtre récente.
Une allocation de production exigerait une sélection walk-forward, des coûts
inter-poches et une validation scellée ; ce travail n'est pas effectué ici.

## Artefacts et validations

- rapport autonome :
  `outputs/performance_reports/strategy_frontiers_20261025/backtest_performance_report.html` ;
- SHA-256 du rapport :
  `1812f1753df6fda53f0a462b8517caf04350fd325bf2e7fcef71b428ced738ea` ;
- manifeste :
  `outputs/performance_reports/strategy_frontiers_20261025/backtest_performance_report_manifest.json` ;
- SHA-256 du manifeste :
  `3c46a7ce4147b751fe962357117f258fb37943d99f2dfc5ccd70bef48eeb86d0` ;
- tests ciblés : 4 passés ;
- Ruff, formatage Ruff et mypy ciblé : passés ;
- syntaxe JavaScript extraite du rapport réel : valide avec `node --check`.

La collecte élargie `tests/unit/portfolio tests/unit/reporting` obtient 52 tests
passés et trois échecs hors périmètre dans `test_portfolio_execution.py` :
`execution.py` tente encore un cast direct `String -> Date`, retiré par Polars
2.0. Le contrôle de taille ne signale aucune régression de `REPORT-014`, mais
reste rouge sur deux régressions préexistantes de l'explorateur SEC et du script
de publication SEC. Ces dettes ne sont pas maquillées dans cette tâche.

La navigation automatisée vers le fichier local a été refusée par la politique
du navigateur. Aucun contournement n'a été tenté ; les contrats DOM, le payload
compressé, l'unicité des identifiants, les fonctions interactives et la syntaxe
du script sont couverts par les tests et contrôles statiques.

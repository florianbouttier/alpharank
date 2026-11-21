# Recherche

Responsabilité : comparaisons et rendus méthodologiques reproductibles.

Entrées : artefacts de recherche explicitement nommés.

Sorties : rapports et mesures non promus en production.

Commandes :

- `build_backtest_performance_report.py` construit le rapport HTML canonique à
  partir d'un replay commun, d'un run Legacy et d'un manifeste snapshot tous
  explicitement nommés ; il ne résout jamais un dossier par récence. Les
  options `--sector-backtest-dir` et `--sector-history-dir` ajoutent les deux
  Top 5 sectoriels au rapport standard et alignent Legacy Equal sur les frais
  communs à partir des positions figées.
- `value_frozen_portfolios.py` valorise les paniers d'un rapport scellé à partir
  d'un reçu prix RAW explicite, avec fenêtres datées, contrôles de hashes et
  HTML canonique complet obligatoire ; les CSV/Parquet/Markdown restent des
  compléments. L'index `outputs/performance_reports/index.html` désigne le
  nouveau HTML daté ; aucun téléchargement, signal ou replay implicite.

Dossiers enfants : aucun.

Interdit ici : publication mensuelle et mutation de données canoniques.

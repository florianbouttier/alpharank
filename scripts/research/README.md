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

Dossiers enfants : aucun.

Interdit ici : publication mensuelle et mutation de données canoniques.

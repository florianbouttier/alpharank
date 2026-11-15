# Recherche

Responsabilité : comparaisons et rendus méthodologiques reproductibles.

Entrées : artefacts de recherche explicitement nommés.

Sorties : rapports et mesures non promus en production.

Commandes :

- `build_backtest_performance_report.py` construit le rapport HTML canonique à
  partir d'un replay commun, d'un run Legacy et d'un manifeste snapshot tous
  explicitement nommés ; il ne résout jamais un dossier par récence. Les
  options `--sector-backtest-dir` et `--sector-history-dir` activent le profil
  des deux Top 5 sectoriels sur leur replay scellé.

Dossiers enfants : aucun.

Interdit ici : publication mensuelle et mutation de données canoniques.

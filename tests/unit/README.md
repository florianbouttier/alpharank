# Tests unitaires

Ce dossier contient les contrats déterministes et isolés, sans réseau ni
artefact de production local. Les fichiers restent nommés selon le domaine
qu'ils protègent ; les fixtures propres à un seul fichier restent locales.

Les sous-dossiers `backtest/`, `boosting/`, `data/`, `governance/`, `legacy/`,
`portfolio/`, `quality/` et `reporting/` reflètent le propriétaire du contrat
testé. Les rares fichiers conservés directement ici sont des ancrages de
compatibilité explicitement recensés dans
`../../docs/architecture/test_code_move_map_v1.json`.

La suite se lance avec `python -m pytest -m unit`.

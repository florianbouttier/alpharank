# Scripts AlphaRank

Points d'entrée CLI et orchestration. La logique réutilisable doit vivre dans
`src/alpharank/`; un script assemble des services, écrit des artefacts et expose
des arguments.

## Dossiers enfants

- `open_source/` : ingestion, reconstruction, composition et audits de données.
- `experiments/` : expériences et générateurs de rapports R&D.
- `quality/` : contrôles statiques différentiels sans mutation des sources.
- `maintenance/` : inventaires et opérations de rangement réversibles.
- `_old/` : scripts archivés, conservés uniquement pour référence.

## Entrées principales

- `run_legacy.py` : production mensuelle canonique.
- `run_backtest.py` : pipeline Boosting historique/R&D.
- `build_common_legacy_boosting_replay.py` : comparaison même snapshot.
- `validate_legacy_replay_package.py` : validation d'un package Legacy.
- `validate_common_portfolio_engine.py` : parité mécanique du simulateur.
- `validate_documentation.py` : couverture des README et liens locaux.

Lire `../docs/monthly_portfolio_runbook.md` avant un run mensuel.

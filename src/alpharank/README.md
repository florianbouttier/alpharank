# Package AlphaRank

## Dossiers enfants

- `strategy/` : signal et agrégation Legacy.
- `multihorizon/` : Boosting causal actuel et scoring live.
- `backtest/` : pipeline Boosting historique et composants génériques.
- `portfolio/` : contrat de holdings, simulation et KPI partagés.
- `production/` : orchestration testable des commandes canoniques.
- `governance_contracts/` : implémentations des contrats de baseline,
  promotion, confirmation, parité économique et provenance runtime ;
  `governance.py` reste leur façade publique stable.
- `data/` : chargement, snapshots, lignée, prix et ingestion open source.
- `features/` : indicateurs communs de bas niveau.
- `models/` : wrappers historiques XGBoost/SHAP.
- `quality/` : contrôles différentiels de qualité du code et baselines associées.
- `visualization/` : rapports Legacy historiques.
- `utils/` : utilitaires transverses sans propriété métier.

Séparer la génération des signaux Legacy/Boosting, puis utiliser
`portfolio/` pour toute comparaison de performance.

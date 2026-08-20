# Package AlphaRank

## Dossiers enfants

- `strategy/` : signal et agrégation Legacy.
- `multihorizon/` : Boosting causal actuel et scoring live.
- `backtest/` : pipeline Boosting historique et composants génériques.
- `portfolio/` : contrat de holdings, simulation et KPI partagés.
- `data/` : chargement, snapshots, lignée, prix et ingestion open source.
- `features/` : indicateurs communs de bas niveau.
- `models/` : wrappers historiques XGBoost/SHAP.
- `visualization/` : rapports Legacy historiques.
- `utils/` : utilitaires transverses sans propriété métier.

Séparer la génération des signaux Legacy/Boosting, puis utiliser
`portfolio/` pour toute comparaison de performance.

# Package AlphaRank

## Dossiers enfants

- `strategy/` : signal et agrégation Legacy.
- `multihorizon/` : Boosting causal actuel et scoring live.
- `backtest/` : pipeline Boosting historique et composants génériques.
- `portfolio/` : contrat de holdings, simulation et KPI partagés.
- `production/` : orchestration testable des commandes canoniques.
- `replay/` : snapshots causaux, replays Legacy/Boosting, comparaison commune,
  rapprochement méthodologique et package recalculable.
- `governance_contracts/` : implémentations des contrats de baseline,
  promotion, confirmation, parité économique et provenance runtime ;
  `governance.py` reste leur façade publique stable.
- `data/` : chargement, snapshots, lignée, prix et ingestion open source.
- `features/` : indicateurs communs de bas niveau.
- `models/` : wrappers historiques XGBoost/SHAP.
- `quality/` : contrôles différentiels de qualité du code et baselines associées.
- `reporting/` : calculs de payloads et tables, séparés du rendu des dashboards.
- `visualization/` : rapports Legacy historiques.
- `utils/` : utilitaires transverses sans propriété métier.

`observability.py` fournit le logger structuré commun. Les commandes durables y
lient `run_id`, `snapshot_id`, composant et étape avant le premier jalon métier.

Les modules racine `*_v2.py`, `causal_snapshot.py` et `replay_validation.py`
sont uniquement des façades de compatibilité. Leur propriétaire et leur API
sont enregistrés dans
`docs/architecture/root_module_ownership_v1.json`; tout nouveau lecteur interne
importe `alpharank.replay` ou l'un de ses modules nommés.

Séparer la génération des signaux Legacy/Boosting, puis utiliser
`portfolio/` pour toute comparaison de performance.

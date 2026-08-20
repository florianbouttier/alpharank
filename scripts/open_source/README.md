# Scripts de données open source

Orchestration des sources de prix, SEC, composition S&P 500 et publication de
snapshots.

## Séquence de production

1. `run_ingestion.py` construit un candidat réseau horodaté.
2. `build_roll_forward_price_package.py` préserve l'historique publié et
   rafraîchit les titres actifs.
3. `build_sec_output_package.py` construit les fondamentaux SEC-only.
4. `build_composed_model_snapshot.py` assemble un snapshot immutable.
5. Legacy et Boosting consomment exactement ce snapshot.

Les scripts `probe_*`, `audit_*`, `repair_*` et `reconstruct_*` produisent des
candidats ou diagnostics ; ils ne publient pas silencieusement la production.
`build_sec_quality_dashboard.py` rend les tables calculées par
`src/alpharank/reporting/sec_quality_data.py`.
Contrat complet : `../../docs/open_source_ingestion_architecture.md`.

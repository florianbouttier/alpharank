# Scripts de données open source

Orchestration des sources de prix, SEC, composition S&P 500 et publication de
snapshots.

## Responsabilités

- `ingestion/` acquiert ou répare les données sources ;
- `publication/` compose et promeut les packages publiables ;
- `audit/` mesure la couverture et prépare les diagnostics ;
- `reporting/` construit les tableaux et rapports de qualité.

Les quelques fichiers Python conservés directement dans ce dossier sont des
façades de compatibilité pour les commandes déjà documentées. Une nouvelle
implémentation doit être rangée dans le sous-dossier responsable.

## Séquence de production

1. `ingestion/run_ingestion.py` construit un candidat réseau horodaté.
2. `publication/build_roll_forward_price_package.py` préserve l'historique publié et
   rafraîchit les titres actifs.
3. `publication/build_acquired_price_package.py` requalifie, sans réseau, un run
   entièrement acquis dont une gate a été revue après coup, en rejouant sa
   réconciliation de rendements sans reprendre l'historique provider brut.
4. `build_sec_output_package.py` est la commande publique qui transmet ses
   arguments à `publication/build_sec_output_package.py` pour construire les
   fondamentaux SEC-only du run demandé.
5. `publication/build_composed_model_snapshot.py` assemble un snapshot immutable.
6. Legacy et Boosting consomment exactement ce snapshot.

Les scripts `probe_*`, `audit_*`, `repair_*` et `reconstruct_*` produisent des
candidats ou diagnostics ; ils ne publient pas silencieusement la production.
`reporting/build_sec_quality_dashboard.py` rend les tables calculées par
`src/alpharank/reporting/sec_quality_data.py`.
Contrat complet : `../../docs/open_source_ingestion_architecture.md`.

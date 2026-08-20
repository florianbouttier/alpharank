# Données et lignée

Services partagés pour charger, transformer, figer et valider les entrées.

## Dossiers enfants

- `open_source/` : connecteurs et pipeline d'ingestion multi-source/SEC.
- `prices/` : composition d'historiques, corporate actions et gardes de
  révision.

Les modules racine gèrent les datasets Legacy, snapshots composés, stockage,
lignée, intégrité ticker et éligibilité mensuelle. Aucun consommateur ne doit
résoudre plusieurs fois un pointeur mutable pendant un run.

`raw_contracts.py` valide le registre fournisseur et résout chaque cible RAW en
cas de doute en arrêtant l'exécution ; `warehouse.py` construit uniquement des
identifiants fournisseur `lower_snake_case` sous `warehouse/raw`.

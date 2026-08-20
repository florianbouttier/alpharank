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

`open_source/raw_archive.py` écrit un reçu immuable pour chaque tentative RAW,
y compris sans payload en cas d'échec. Les octets reçus sont stockés une seule
fois sous leur SHA-256 ; plusieurs reçus identiques référencent le même objet et
le manifeste fournisseur recompte et revalide tous ses reçus.

`staging.py` porte la normalisation fournisseur-neutre : types et colonnes sont
harmonisés, mais aucune priorité de source n'est acceptée. Deux fournisseurs en
désaccord sur la même clé métier restent deux lignes reliées à leurs reçus RAW.

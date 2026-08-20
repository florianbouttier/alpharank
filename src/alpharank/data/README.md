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

`definitive.py` applique une règle de priorité versionnée à un cutoff de
connaissance explicite. Il conserve le reçu et le hash sélectionnés, distingue
un vrai zéro d'une valeur absente et produit une décision auditée même lorsque
la clé reste irrésolue.

`mart.py` résout l'entrée modèle canonique. Il refuse une cible hors de
`warehouse/mart`, exige les preuves de parité DEF/source et revérifie le hash de
chaque fichier modèle avant de rendre le dossier au consommateur.

`snapshot_publication.py` publie ce MART par référence, sans seconde copie de
données. Son manifeste immuable contient chaque chemin, taille et SHA-256 ; le
pointeur atomique conserve le hash du manifeste et de l'inventaire complet.

`historical_migration.py` catalogue les anciennes racines avant leur bascule.
Chaque fichier reste à sa place et reçoit une référence de chemin, taille et
SHA-256 ; le catalogue refuse une mutation et atteste zéro copie et zéro
téléchargement.

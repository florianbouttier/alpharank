# Architecture du dépôt

Ce dossier explique l'organisation du projet sans raconter l'historique des
runs ni redécrire les deux modèles.

- [`repository_map.md`](repository_map.md) : où se trouve chaque responsabilité,
  aujourd'hui et dans la cible.
- [`data_lifecycle.md`](data_lifecycle.md) : rôle précis de `raw`, `stg`, `def`,
  `mart`, des snapshots et des sorties.
- [`code_dependency_inventory_v1.json`](code_dependency_inventory_v1.json) :
  graphe versionné des points d'entrée, imports et lecteurs Python suivis.
- [`data_location_inventory_v1.json`](data_location_inventory_v1.json) :
  fichiers/packages de données actuels, volumes observés et lecteurs actifs.
- [`data_reader_migration_v1.json`](data_reader_migration_v1.json) : comparaison
  hashée ancien/MART et décision explicite pour chaque lecteur Legacy actif.
- [`legacy_data_archive_policy_v1.json`](legacy_data_archive_policy_v1.json) :
  gel contractuel, fenêtre d'observation et retour arrière des anciennes racines.
- [`test_catalog_v1.json`](test_catalog_v1.json) : domaine, suite, durée et
  résultat observé de chaque fichier de test suivi.
- [`test_collection_v1.json`](test_collection_v1.json) : collecte Pytest
  canonique, indépendante du dossier parent.
- [`test_split_audit_v1.json`](test_split_audit_v1.json) : empreintes des corps
  de tests protégées lors de la découpe des deux anciens modules monolithiques.
- [`test_fixture_inventory_v1.json`](test_fixture_inventory_v1.json) : seule
  fixture partagée et règle de propriété locale par défaut.

Les migrations nécessaires sont suivies dans [`../../ROADMAP.md`](../../ROADMAP.md).
Les règles applicables aux nouveaux changements sont sous
[`../standards/`](../standards/).

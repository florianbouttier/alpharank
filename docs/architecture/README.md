# Architecture du dépôt

Ce dossier explique l'organisation du projet sans raconter l'historique des
runs ni redécrire les deux modèles.

- [`repository_map.md`](repository_map.md) : où se trouve chaque responsabilité,
  aujourd'hui et dans la cible.
- [`data_lifecycle.md`](data_lifecycle.md) : rôle précis de `raw`, `stg`, `def`,
  `mart`, des snapshots et des sorties.
- [`code_dependency_inventory_v1.json`](code_dependency_inventory_v1.json) :
  graphe versionné des points d'entrée, imports et lecteurs Python suivis.
- [`python_directory_inventory_v1.json`](python_directory_inventory_v1.json) :
  nombre de modules par dossier et preuve du respect du plafond de 20 fichiers.
- [`../../configs/quality/python_size_baseline_v1.json`](../../configs/quality/python_size_baseline_v1.json) :
  dette historique de taille et complexité, bloquée contre toute aggravation.
- [`data_location_inventory_v1.json`](data_location_inventory_v1.json) :
  fichiers/packages de données actuels, volumes observés et lecteurs actifs.
- [`data_reader_migration_v1.json`](data_reader_migration_v1.json) : comparaison
  hashée ancien/MART et décision explicite pour chaque lecteur Legacy actif.
- [`legacy_data_archive_policy_v1.json`](legacy_data_archive_policy_v1.json) :
  gel contractuel, fenêtre d'observation et retour arrière des anciennes racines.
- [`run_root_inventory_v1.json`](run_root_inventory_v1.json) : famille, date,
  statut explicite disponible et volume des 346 racines historiques de `outputs/`.
- [`../run_organization.md`](../run_organization.md) : contrat des nouveaux
  chemins `outputs/<famille>/<run_id>/`.
- [`run_retention_report_v1.json`](run_retention_report_v1.json) : mesure des
  doublons exacts et proposition de rétention sans suppression automatique.
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

# Rapports open source

Responsabilité : produire les vues SEC et qualité à partir de résultats calculés.

Entrées : tables et diagnostics structurés.

Sorties : rapports HTML, JSON ou Parquet dérivés.

Dossiers enfants : aucun.

Interdit ici : recalcul métier et promotion de données.

## Explorateur SEC par entreprise

La commande exige un run explicite ; elle ne choisit jamais un dossier parce
qu'il semble être le plus récent :

```bash
python scripts/open_source/reporting/build_sec_fundamental_explorer.py \
  --raw-run-dir data/open_source/official/runs/<run_id>/raw
```

Elle écrit un `report.html` autonome et son `manifest.json` sous
`outputs/sec_fundamental_explorer/<run_id>/`. Le rapport filtre par société,
trace les valeurs par trimestre, conserve toutes les versions et donne accès
aux lignes brutes et aux hashes des fichiers sources.

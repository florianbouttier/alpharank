# Commandes d'ingestion open source

Responsabilité : lancer les acquisitions, refreshs et réparations gouvernées.

Entrées : configurations, caches et cibles explicites.

Sorties : transactions d'ingestion et reçus.

`reconstruct_historical_sec_companyfacts.py` produit un package SEC historique
quarantiné. La commande ne modifie ni l'entrepôt, ni un snapshot modèle, ni le
pointeur de production.

`build_historical_sector_history.py` reconstruit une release DEF sectorielle
causale depuis les SIC des en-têtes de dépôts annuels SEC. Les réponses et
échecs sont archivés sous les racines RAW SEC, l'heure de connaissance est
l'acceptation officielle du dépôt et les intervalles ticker/CIK bornent chaque
observation. La commande écrit une nouvelle release adressée par contenu ; elle
ne modifie ni `latest.json`, ni un modèle, ni un portefeuille.

Exemple :

```bash
.venv/bin/python \
  scripts/open_source/ingestion/build_historical_sector_history.py \
  --ticker-universe <predictions.parquet> \
  --general-reference-lineage \
    data/sec/output/lineage/general_reference_lineage.parquet \
  --start-year 2009 \
  --end-year 2026 \
  --run-id <run-id-unique>
```

Dossiers enfants : aucun.

Interdit ici : rendu de dashboard et décision économique.

# Tests

Tests unitaires et d'intégration de lignée, ingestion, causalité, stratégies,
simulation et rapports.

Conventions :

- nommer le test selon le contrat protégé, pas selon l'incident du jour ;
- utiliser des fixtures synthétiques pour les règles étroites ;
- conserver un test de replay pour les changements à fort impact ;
- ne pas dépendre d'un dossier `outputs/` mutable sauf test explicitement
  marqué comme intégration locale.

Validation documentaire : `test_documentation_structure.py` vérifie que chaque
dossier actif possède son README et que les liens Markdown locaux existent.

## Suites logiques

La politique versionnée `configs/quality/test_suites_v1.json` attribue chaque
fichier à une seule suite pendant la collecte, sans changer son emplacement :

```bash
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m replay
python -m pytest -m network
python -m pytest -m production
```

`network` désigne les contrats de frontière fournisseur ; aucun accès live
n'est implicite. `production` valide les workflows sur des fixtures contrôlées
et ne publie ni ne promeut aucun artefact.

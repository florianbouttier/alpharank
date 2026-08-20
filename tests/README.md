# Tests

Tests unitaires et d'intégration de lignée, ingestion, causalité, stratégies,
simulation et rapports.

- `unit/` : contrats isolés et déterministes ;
- `integration/` : composants et fichiers contrôlés ;
- `integration/network/` : frontières fournisseur sans accès live implicite ;
- `replay/` : snapshots, lignée, causalité et parité économique ;
- `production/` : workflows et publication sur fixtures contrôlées.

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
fichier à une seule suite selon son emplacement maintenu :

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

Le catalogue versionné
[`../docs/architecture/test_catalog_v1.json`](../docs/architecture/test_catalog_v1.json)
associe chaque fichier suivi à son domaine, sa suite, sa frontière réseau et sa
durée mesurée. Une mesure `failed_missing_local_artifacts` signifie que le test
requiert encore un ancien artefact local non versionné ; elle reste visible tant
que ce test n'a pas été rendu autonome.

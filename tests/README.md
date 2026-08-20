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

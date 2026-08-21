# Expériences et rapports

Scripts R&D non utilisés directement par la production mensuelle. Ils couvrent
les challengers Boosting, les diagnostics EMA, l'attribution et les rapports
HTML.

## Responsabilités

- `modeling/` construit et évalue les signaux et modèles ;
- `portfolio/` transforme des prédictions en scénarios de portefeuille ;
- `legacy/` caractérise et recompose la méthode Legacy ;
- `operations/` orchestre les expériences multihorizon et live ;
- `protocols/` matérialise les conventions et protocoles expérimentaux ;
- `reports/` produit les analyses et rapports de recherche.

Les fichiers Python conservés directement dans ce dossier sont des façades de
compatibilité. Une nouvelle implémentation doit être rangée dans le
sous-dossier qui porte sa responsabilité.

`reports/render_central_research_dashboard.py` ne porte que le rendu et l'orchestration
du rapport central ; ses calculs sont propriétaires de
`src/alpharank/reporting/central_research_data.py`.
Les commandes de comparaison temporelle délèguent leurs KPI et calendriers à
`src/alpharank/portfolio/comparison.py`.

Règles :

- commencer par un smoke test étroit ;
- écrire dans un dossier `outputs/` horodaté ou nommé par run ;
- conserver snapshot, configuration, commande et métrique primaire ;
- promouvoir les conclusions dans les documents canoniques ;
- ne pas réimplémenter les KPI du package `alpharank.portfolio`.

`protocols/build_execution_convention_bridge.py` rapproche la série canonique de clôture
et la sensibilité à la prochaine ouverture. Il échoue si le dossier de sortie
existe déjà et conserve les hashes de chaque entrée et sortie.

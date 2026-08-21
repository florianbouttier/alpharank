# Tests d'intégration

Ce dossier valide plusieurs composants, des frontières de fichiers contrôlées
et les packages de données synthétiques. Aucun contenu de `outputs/` mutable ne
devient une fixture implicite.

`network/` isole les contrats de fournisseurs, désactivés en accès réel par
défaut. La suite principale se lance avec `python -m pytest -m integration`.

Les contrats sont rangés sous `fundamentals/`, `ingestion/`, `prices/`,
`publishing/`, `replay/` et `warehouse/`. Le helper privé
`publishing/_legacy_export_support.py` construit seulement la référence
synthétique commune aux exports Legacy et n'est pas collecté comme module de
test. Les rares fichiers conservés directement ici sont documentés dans
`../../docs/architecture/test_code_move_map_v1.json`.

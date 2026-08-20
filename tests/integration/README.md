# Tests d'intégration

Ce dossier valide plusieurs composants, des frontières de fichiers contrôlées
et les packages de données synthétiques. Aucun contenu de `outputs/` mutable ne
devient une fixture implicite.

`network/` isole les contrats de fournisseurs, désactivés en accès réel par
défaut. La suite principale se lance avec `python -m pytest -m integration`.

Les grands contrats SEC et d'export Legacy sont répartis par responsabilité.
Le helper privé `_legacy_export_support.py` construit seulement leur référence
synthétique commune et n'est pas collecté comme module de test.

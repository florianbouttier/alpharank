# Tests d'intégration

Ce dossier valide plusieurs composants, des frontières de fichiers contrôlées
et les packages de données synthétiques. Aucun contenu de `outputs/` mutable ne
devient une fixture implicite.

`network/` isole les contrats de fournisseurs, désactivés en accès réel par
défaut. La suite principale se lance avec `python -m pytest -m integration`.

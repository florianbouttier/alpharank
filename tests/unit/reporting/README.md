# Tests unitaires reporting

Préparation et rendu des rapports complets de recherche et scénarios SEC, sans
réimplémenter les calculs économiques ni tester une application de suivi.

`test_daily_analysis_service.py` vérifie les dates libres, les hashes source,
la conservation du payload, l'immuabilité des rapports et le service HTTP local
en lecture seule. `test_strategy_analysis_ui.py` contrôle le bundle React,
les licences, l'encodage du payload et la syntaxe ; les interactions réelles
sont vérifiées séparément dans Chrome avec le service et le moteur Python.

Le service teste aussi l'audit titre/séance, la source de repli mensuelle,
le refus d'un arrondi silencieux et la proposition de mois complets inclus.

L'explorateur SEC est testé sur un petit run synthétique : conservation des
versions amendées, scellement de l'identifiant de run, hashes, payload autonome
et absence de ressource réseau.

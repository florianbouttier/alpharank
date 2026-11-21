# Valorisation et analyse journalières

Responsabilité : reconstruire les valorisations quotidiennes des paniers scellés
et mesurer une fenêtre de dates libre dans le moteur financier commun.

Entrées : positions, rendements et coûts mensuels archivés, prix ajustés du même
snapshot et calendrier SPY explicite. Aucun pointeur mutable ni réseau.

Sorties : ledger quotidien, rapprochement mensuel, trous de couverture et KPI
calculés sur les dates demandées ; combinaisons rééquilibrées mensuellement.

`monthly_reference.py` adapte les rendements mensuels déjà scellés en une
grille distincte de fins de mois. Le même moteur mesure cette grille à 12
observations/an, sans créer de cours quotidien ; ses limites sont définies dans
le [contrat commun](../../../../docs/common_portfolio_backtest_engine.md#sealed-monthly-reference-report-028).

Dossiers enfants : aucun.

Interdit ici : signal, classement, entraînement, remplissage de prix manquant,
réécriture de snapshot et promotion implicite d'une reconstitution historique.

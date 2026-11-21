# Valorisation et analyse journalières

Responsabilité : reconstruire les valorisations quotidiennes des paniers scellés
et mesurer une fenêtre de dates libre dans le moteur financier commun.

Entrées : positions, rendements et coûts mensuels archivés, prix ajustés du même
snapshot et calendrier SPY explicite. Aucun pointeur mutable ni réseau.

Sorties : ledger quotidien, rapprochement mensuel, trous de couverture et KPI
calculés sur les dates demandées ; combinaisons rééquilibrées mensuellement.

Dossiers enfants : aucun.

Interdit ici : signal, classement, entraînement, remplissage de prix manquant,
réécriture de snapshot et promotion implicite d'une reconstitution historique.

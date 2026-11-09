# Expériences portefeuille

Responsabilité : assemblages de scores expérimentaux au niveau portefeuille.

Entrées : prédictions et rendements de recherche.

Sorties : courbes et comparaisons non promues.

Commandes : `run_sector_champion_allocation.py` produit le Top 5 Boosting
causal limité à une action par secteur, son journal de couverture et ses hashes.
`run_sector_champion_backtest.py` compare les variantes native et tendance à
leurs références, Legacy et SPY sur le calendrier réalisé commun.

Dossiers enfants : aucun.

Interdit ici : moteur KPI local et ordre réel.

# Tests unitaires portefeuille

Exécution, benchmark, maturité, attribution et événements terminaux du moteur
de portefeuille commun. La valorisation des paniers figés vérifie aussi les
dates communes, le rejet des prix manquants, la dérive des poids et l'invariance
des résultats passés aux modifications des prix futurs.

`test_daily_analysis.py` couvre les fenêtres au jour près, le repère initial,
les frais et la parité mensuelle, l'absence d'interpolation des trous, les
années complètes, l'annualisation quotidienne et les compositions/frontières.

`test_monthly_reference.py` vérifie le repli scellé : aucune observation
intermédiaire inventée, rendements mensuels conservés, fréquence déclarée,
composition/frontières communes et rejet d'un calendrier incomplet.

# Stratégie Legacy

Baseline de production mensuelle : signal EMA relatif, filtres point-in-time,
optimisation annuelle, votes et agrégation `Combined_Equal` /
`Combined_Frequency`.

`legacy.py` conserve les façades publiques d'apprentissage et d'évaluation.
`legacy_aggregation.py` agrège les votes, `legacy_selection.py` applique les
contraintes point-in-time et extrait les positions mensuelles, tandis que
`legacy_artifacts.py` prépare les tableaux de comparaison. `base.py` porte le
contrat, `analytics.py` les calculs de diagnostic et `portfolio.py` la
compatibilité historique. Les performances publiées passent ensuite par
`../portfolio/`.

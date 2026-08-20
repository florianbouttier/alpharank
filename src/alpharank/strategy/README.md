# Stratégie Legacy

Baseline de production mensuelle : signal EMA relatif, filtres point-in-time,
optimisation annuelle, votes et agrégation `Combined_Equal` /
`Combined_Frequency`.

`legacy.py` porte le workflow principal ; `base.py` le contrat ; `analytics.py`
les calculs de diagnostic ; `portfolio.py` la compatibilité historique. Les
performances publiées passent ensuite par `../portfolio/`.

# Qualité des données

Registres versionnés appliqués avant la construction des signaux.

- `historical_ticker_exclusions_v1.json` exclut une trajectoire entière lorsque
  l'identité ou les prix sont irrécupérables.
- `confirmed_corporate_actions.json` décrit les corrections de splits et
  dividendes autorisées.
- `sp500_constituent_changes_2026.json` conserve les changements récents de
  composition avec leur date effective.

Toute modification doit être sourcée, produire un nouveau hash et déclencher
les tests de données ainsi qu'un replay Legacy/Boosting comparable.

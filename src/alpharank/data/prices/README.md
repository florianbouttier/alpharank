# Historique des prix

Composition du seed EODHD figé avec les rafraîchissements ouverts, persistance
des titres inactifs, corrections d'actions corporate et gardes fail-closed.

- `contracts.py` définit les schémas et politiques.
- `composition.py` assemble les sources.
- `history.py` et `seed.py` protègent la continuité historique.
- `corporate_actions.py` applique les corrections versionnées.
- `gates.py` bloque les révisions inexpliquées.

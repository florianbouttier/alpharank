# Moteur de portefeuille partagé

Unique source de vérité après génération des signaux.

## Dossier enfant

- `adapters/` : conversion des sorties Legacy et Boosting vers le contrat
  commun.

Les modules racine définissent holdings, allocation, simulation, benchmark SPY,
KPI, comparaison, attribution exacte du CAGR, lignée et artefacts.
`comparison.py` centralise aussi les grilles par sous-période et les comparaisons
par année de départ utilisées par les commandes de reporting. `combinations.py`
pré-calcule les rendements et KPI de toutes les combinaisons équipondérées d'au
plus dix poches, sans ajouter de coût entre poches aux frais déjà contenus dans
leurs rendements nets. Toute nouvelle
implémentation locale de CAGR, Sharpe, drawdown, turnover ou rendement annuel est
interdite. Voir `../../../docs/common_portfolio_backtest_engine.md`.

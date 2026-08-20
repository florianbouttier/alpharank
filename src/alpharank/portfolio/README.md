# Moteur de portefeuille partagé

Unique source de vérité après génération des signaux.

## Dossier enfant

- `adapters/` : conversion des sorties Legacy et Boosting vers le contrat
  commun.

Les modules racine définissent holdings, allocation, simulation, benchmark SPY,
KPI, comparaison, attribution exacte du CAGR, lignée et artefacts. Toute nouvelle
implémentation locale de CAGR, Sharpe, drawdown, turnover ou rendement annuel est
interdite. Voir `../../../docs/common_portfolio_backtest_engine.md`.

# Archive CODE-010 du 20 août 2026

Ces sept scripts sont conservés byte pour byte après vérification de leurs
imports, appels, liens documentaires, automatisations et tests :

- `run_full_backtest.py` : ancien pipeline remplacé par `scripts/run_backtest.py` ;
- `backtest_data_source_examples.py` et `run_backtest_open_source_prices.py` :
  exemples path-coupled remplacés par le snapshot canonique et la configuration
  de `scripts/run_backtest.py` ;
- `download_data.py` et `download_data_incremental.py` : anciennes acquisitions
  EODHD, incompatibles avec l'archive fournisseur figée ;
- `generate_financial_report.py` : rapport ad hoc remplacé par les rapports SEC
  maintenus ;
- `run_boosting_vs_legacy.py` : comparaison remplacée par le replay commun
  Legacy/Boosting.

Le registre machine-lisible canonique est
[`../../../docs/architecture/script_archival_audit_v1.json`](../../../docs/architecture/script_archival_audit_v1.json).

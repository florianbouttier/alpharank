# Configurations

Ce dossier contient les décisions versionnées qui modifient un run sans être du
code Python.

## Dossiers enfants

- `data_contracts/` : schémas versionnés des familles de configuration.
- `data_quality/` : exclusions de tickers, actions corporate confirmées et
  changements de composition d'indice.
- `quality/` : baselines versionnées des contrôles différentiels de code.
- `research/` : configurations figées des expériences et challengers.

Une configuration utilisée en production ou dans une comparaison publiée doit
être hashée dans le manifeste du run. Ne pas modifier rétroactivement un fichier
déjà consommé : créer une nouvelle version.

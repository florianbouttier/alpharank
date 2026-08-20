# Configurations

Ce dossier contient les décisions versionnées qui modifient un run sans être du
code Python.

## Dossiers enfants

- `data_quality/` : exclusions de tickers, actions corporate confirmées et
  changements de composition d'indice.
- `research/` : configurations figées des expériences et challengers.

Une configuration utilisée en production ou dans une comparaison publiée doit
être hashée dans le manifeste du run. Ne pas modifier rétroactivement un fichier
déjà consommé : créer une nouvelle version.

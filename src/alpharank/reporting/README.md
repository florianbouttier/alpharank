# Rapports d'audit statiques

Ce package transforme des payloads d'audit déjà calculés en tables ou en HTML
autonome. Il ne contient ni application de suivi, ni calcul économique, ni
commande interactive.

- `sec_quality_data.py` construit les tables de couverture, trous trimestriels
  et anomalies d'actions du rapport SEC.
- `refresh_replay_html.py` rend le rapport humain du refresh sans recalculer
  les scores, holdings ou KPI.

Les rapports HTML/Markdown conservés dans AlphaRank sont des preuves statiques
d'audit. L'interface de portefeuille et le monitoring appartiennent au dépôt
Portfolio.

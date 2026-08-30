# Rapports d'audit statiques

Ce package transforme des payloads d'audit déjà calculés en tables ou en HTML
autonome. Il ne contient ni application de suivi, ni second calcul économique,
ni commande interactive : les KPI de performance proviennent du moteur commun.

- `sec_quality_data.py` construit les tables de couverture, trous trimestriels
  et anomalies d'actions du rapport SEC.
- `sec_fundamental_explorer.py` vérifie un run RAW explicite, conserve toutes
  ses versions SEC et prépare le payload de l'explorateur autonome par société.
- `_sec_explorer_html.py` et `_sec_explorer_script.py` possèdent uniquement le
  rendu et les interactions locales du rapport ; ils ne sélectionnent aucune
  valeur fondamentale.
- `refresh_replay_html.py` rend le rapport humain du refresh sans recalculer
  les scores, holdings ou KPI.
- `performance_report.py` prépare la vue complète d'un replay explicitement
  nommé ; `_performance_report_html.py`, `_performance_report_styles.py` et
  `_performance_report_script.py` rendent la comparaison multi-stratégie, le
  multiselect global des courbes, les graphiques pleine largeur, les matrices
  cumulées et annuelles et les portefeuilles sans redéfinir les KPI.

Les rapports HTML/Markdown conservés dans AlphaRank sont des preuves statiques
d'audit. L'interface de portefeuille et le monitoring appartiennent au dépôt
Portfolio.

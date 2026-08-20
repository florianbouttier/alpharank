# Données des rapports

Ce package calcule les payloads et tables des rapports maintenus. Il ne contient
ni page HTML, ni style, ni commande interactive.

- `central_research_data.py` construit les séries, lignées, diagnostics et
  payloads du dashboard central ;
- `sec_quality_data.py` construit les tables de couverture, trous trimestriels
  et anomalies d'actions du rapport SEC.

Les scripts sous `scripts/experiments/` et `scripts/open_source/` restent
responsables de l'orchestration disque et du rendu HTML/Markdown.

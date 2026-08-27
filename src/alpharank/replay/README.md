# Contrats de replay

Ce package possède les validations et reconstructions qui rendent un résultat
AlphaRank causal, comparable et recalculable :

- `causal_snapshot.py` scelle les entrées, politiques et sources du run ;
- `legacy.py` et `boosting.py` recalculent séparément leurs artefacts ;
- `common.py` construit et valide la comparaison Legacy/Boosting/SPY ;
- `common_strategy.py` construit le replay public même snapshot ;
- `reconciliation.py` explique le pont économique entre méthodologies ;
- `refresh_compare.py` compare les clés naturelles et les valeurs au cutoff ;
- `refresh_drift.py` relie le refresh aux deux portefeuilles et bloque tout
  écart non attribué ;
- `refresh_provenance.py` détaille chaque différence de code, paramètre,
  dépendance et seed sans confondre les chemins de sortie avec la méthode ;
- `refresh_sources.py` déclare les sources téléchargées, conservées ou non
  démarrées lorsqu'une gate amont arrête le refresh ;
- `refresh_attribution.py` compare les quatre scénarios baseline, prix seuls,
  SEC seuls et candidat complet pour séparer scores, Top-N et positions ;
- `validation.py` scelle et recalcule un package de replay autonome.

L'API canonique courte est exposée par `alpharank.replay`. Les anciens modules
`alpharank.*_v2`, `alpharank.causal_snapshot` et
`alpharank.replay_validation` restent des façades de compatibilité : ils ne
possèdent plus d'implémentation.

Le script public `scripts/build_common_legacy_boosting_replay.py` ne contient
que la CLI et une façade de compatibilité. Les scripts orchestrent ces contrats,
mais aucun module de ce package n'importe un script. Les calculs de portefeuille
restent la propriété de `alpharank.portfolio` et la promotion reste celle de
`alpharank.governance`.

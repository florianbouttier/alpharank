# Contrats de replay

Ce package possède les validations et reconstructions qui rendent un résultat
AlphaRank causal, comparable et recalculable :

- `causal_snapshot.py` scelle les entrées, politiques et sources du run ;
- `legacy.py` et `boosting.py` recalculent séparément leurs artefacts ;
- `common.py` construit et valide la comparaison Legacy/Boosting/SPY ;
- `reconciliation.py` explique le pont économique entre méthodologies ;
- `validation.py` scelle et recalcule un package de replay autonome.

L'API canonique courte est exposée par `alpharank.replay`. Les anciens modules
`alpharank.*_v2`, `alpharank.causal_snapshot` et
`alpharank.replay_validation` restent des façades de compatibilité : ils ne
possèdent plus d'implémentation.

Les scripts orchestrent ces contrats, mais aucun module de ce package n'importe
un script. Les calculs de portefeuille restent la propriété de
`alpharank.portfolio` et la promotion reste celle de `alpharank.governance`.

# Contrats de gouvernance

Ce package sépare les contrôles de gouvernance sans changer l'API publique
`alpharank.governance` :

- `promotion.py` réserve les runs et gère promotion/rollback ;
- `confirmation.py` porte la confirmation finale scellée ;
- `baseline.py` scelle et valide les baselines immuables ;
- `economic_prefix.py` contrôle la parité économique historique ;
- `runtime_provenance.py` capture et valide le runtime ;
- `contracts.py` et `common.py` contiennent seulement les contrats et primitives
  partagés.

Un consommateur métier importe la façade `alpharank.governance`. Les modules de
ce package sont les propriétaires internes des implémentations.

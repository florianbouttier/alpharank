# Références d'identité

Petits fichiers versionnés servant à résoudre les tickers historiques sans
injecter une valeur fondamentale externe.

`sec_historical_ticker_bridge.csv` associe ticker, fenêtre de validité et CIK.
Toute nouvelle ligne exige une vérification d'identité et un audit point-in-time
avant promotion dans un snapshot.

`security_identity_registry.csv` sépare les instruments lorsqu'un symbole est
réutilisé par un autre émetteur. Son `canonical_ticker` est la clé interne
stable appliquée aux prix, fondamentaux et constituants ; les intervalles ne
peuvent pas se chevaucher.

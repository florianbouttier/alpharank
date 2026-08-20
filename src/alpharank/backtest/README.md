# Backtest Boosting historique

Pipeline modulaire de chargement, features, datasets causaux, folds temporels,
tuning, explicabilité et reporting.

`application.py` orchestre le cas d'usage ; `pipeline.py` assemble le run ;
`datasets.py` et `time_folds.py` protègent le calendrier ; `portfolio.py` reste
une couche de compatibilité et doit déléguer au moteur partagé lorsque possible.

Pour le challenger public multi-horizon, préférer `../multihorizon/`.

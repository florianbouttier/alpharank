# Boosting multi-horizon

Implémentation active du challenger Boosting : configuration, données,
preprocessing, splits walk-forward, entraînement XGBoost, tuning, métriques,
risque, trading, SHAP et scoring live.

Le score et la cible appartiennent à ce dossier. Les holdings finalisés sont
convertis vers `../portfolio/` pour les performances et comparaisons. Le profil
public est lancé avec `--latest-common-comparison-profile`. Les filtres
d'allocation expérimentaux appliqués après émission des scores appartiennent à
`../replay/prediction_universes.py`, pas à ce package de modèle.

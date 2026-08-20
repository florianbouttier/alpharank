# Journal Boosting — 20 août 2026

> Entrée présente dans le worktree avant le découpage DOC-009.

## 2026-08-20 — centre de recherche clôture, portefeuilles et SHAP exhaustif

Hypothèse contrôlée : l'écart de performance récent pouvait venir d'un changement
de données ou de modèle. Le diagnostic montre au contraire trois effets à ne pas
mélanger : la période affichée, la convention clôture/prochaine ouverture et une
réutilisation anormale du symbole SNDK.

- Données : snapshot composé immuable `9a2058c9…425ad`, Legacy
  `20260820_055245`, Boosting `boosting_latest_common_live021_final_r2` et replay
  commun `common_replay_live021_final`.
- Commande/run : générateur
  `scripts/experiments/render_latest_common_dashboard.py`, sortie
  `outputs/production_refresh_20260820/dashboard_live021_analysis_close_v1/`.
- Résultat principal : sur janvier 2012 à décembre 2024, le CAGR Legacy passe de
  `16,4033 %` à `17,0905 %`, soit seulement `+0,6871` point à dates identiques.
  L'affichage complet jusqu'en juillet 2026 monte à `22,0295 %` parce qu'il ajoute
  une période 2025–2026 exceptionnellement forte.
- Sensibilité d'exécution : clôture moins prochaine ouverture sur les mêmes 180
  mois vaut `+2,8669` points de CAGR Top 5, `+2,0782` points Top 10 et `+1,0754`
  point Legacy. La clôture reste la convention AlphaRank ; l'ouverture est une
  sensibilité.
- Anomalie ouverte : SNDK représente un pont exact de `+4,6879` points de CAGR
  Top 5 et `+1,9435` point Top 10. L'ancien SanDisk et la nouvelle cotation ne
  sont pas encore séparés dans l'identité de sécurité ; `LIVE-022` bloque donc
  la publication économique.
- Explicabilité : `88 948 / 88 948` lignes de prédiction possèdent une ligne SHAP,
  sur 181 mois. Le schéma expose les 195 variables, leur valeur d'entrée et leur
  contribution pour chaque ticker/mois, ainsi que score, probabilité, rangs Top
  5/10, statuts de cible et rendements futurs lorsqu'ils sont évaluables.
- Décision suivante : implémenter `LEG-005`, corriger SNDK sous `LIVE-022`,
  reconstruire les runs et n'accorder le statut publiable qu'après rapprochement
  mensuel et final à `1e-12`.
- Cible live du mois : la page
  `outputs/production_refresh_20260820/dashboard_live021_trade_20260819_v1/html/alpharank_research_center.html`
  expose le signal d'août formé sur juillet complet et les clôtures exactes du
  19 août. Top 5 : `CHTR`, `BSX`, `ZTS`, `LULU`, `DELL`. Top 10 : ces cinq titres
  puis `TTD`, `FISV`, `ACN`, `INTU`, `IT`. Aucun SNDK n'est présent dans les
  paniers live ; l'anomalie reste un blocage des performances historiques.

# Rapports d'audit et d'analyse

Responsabilité : préparer les vues et rapports des résultats déjà calculés par
le moteur commun, sans application d'exécution ni nouveau calcul financier local.

Entrées : payloads/source artifacts explicites, snapshots et prix hashés.

Sorties : HTML React complet, tableaux SEC, preuves d'audit et service local de
consultation en lecture seule.

Dossiers enfants :

- `analysis_ui/` possède le frontend React, les graphiques Lightweight Charts,
  les styles, les notices et leur build reproductible. Aucun KPI JavaScript.
- `strategy_analysis/` résout les sources du même snapshot, prépare les données
  quotidiennes via `portfolio/daily/`, écrit le rapport et sert les requêtes de
  dates libres via le moteur Python, sans acquisition ni modification d'entrée.

Modules maintenus :

- `performance_report.py` prépare le profil complet Legacy/Boosting/SPY et ses
  variantes sectorielles ; les formules restent dans `portfolio/`.
- `_performance_report_analytics.py`, `_performance_report_sector.py` et
  `_performance_report_methods.py` préparent respectivement les grilles de
  référence, les données sectorielles scellées et les cartes méthodologiques.
- `_performance_report_html.py` rend le frontend commun et conserve tout le
  payload compressé ; les anciens scripts/styles de rendu ont été remplacés,
  pas empilés derrière une deuxième interface active.
- `performance_followup.py` ajoute le suivi quotidien des paniers figés au
  même rapport, distinct du backtest mensuel net de frais.
- `sec_quality_data.py`, `sec_fundamental_explorer.py`,
  `_sec_explorer_html.py` et `_sec_explorer_script.py` possèdent les vues de
  qualité et l'explorateur SEC, sans sélectionner de valeur fondamentale.
- `refresh_replay_html.py` rend les audits de refresh et de stabilité
  inter-vintages sans recalculer de scores ni de holdings.

Interdit ici : ordre réel, connexion broker, entraînement, classement de titres,
remplissage de prix manquants et seconde définition d'un KPI. Le monitoring des
avoirs personnels reste dans Portfolio ; les analyses AlphaRank restent ici.

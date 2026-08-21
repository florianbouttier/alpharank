# Journal Boosting — 27 juillet au 12 août 2026

> Fragment chronologique conservé depuis le catalogue central.


## Centre de recherche HTML consolidé — 2026-07-27

Hypothèse : une lecture fiable du track nécessite de rapprocher, dans un même
artefact, le screening méthode/horizon, le backtest, les actions mensuelles,
SHAP, le risque et le candidat live sans changer les résultats sources.

Commande :

```bash
./.venv/bin/python \
  scripts/experiments/reports/render_central_research_dashboard.py
```

Rapport :
`outputs/research_dashboard/legacy_ema_alpha_central_20260810_full_shap/html/alpharank_research_center.html`.
Le manifeste adjacent enregistre les empreintes SHA-256 des 20 fichiers
sources et les conventions d'interprétation.

Contenu :

- huit onglets : synthèse, modèles/horizons, backtest approfondi, actions par
  mois, SHAP mensuel, risque, production 2026, documentation/audit ;
- 172 mois de portefeuilles Top 5, Top 10 et Legacy avec entrées, sorties,
  poids, score/probabilité disponibles et rendement réalisé ;
- 76 534 observations SHAP OOS exhaustives dans les vues mensuelles, 185
  variables et 172 mois de décision, avec importance, beeswarm, analyse
  individuelle et lexique filtrables par mois ;
- comparaison commune août 2011-novembre 2025 contre Legacy et SPY : richesse,
  drawdown, performances, années, régimes, coûts, bootstrap et gates ;
- métriques des modèles alpha et risque, ainsi que le candidat live juillet
  2026 et sa lignée temporelle.

Interprétation temporelle à préserver : le backtest réentraîne une fois par
fold externe, généralement avant un bloc de douze mois de test. Le filtre SHAP
mensuel explique les observations du mois sous le modèle de ce fold ; il ne
signifie pas qu'un nouveau modèle historique a été ajusté ce mois-là. Le
runner live réentraîne en revanche à chaque exécution mensuelle.

Chargement SHAP : les vues mensuelles utilisent toutes les lignes du mois,
soit 361 à 497 observations. La vue « tous les mois » utilise encore 80 lignes
par fold, soit 1 200 lignes, et son statut échantillonné est visible dans l'UI.

Décision : ce rapport devient le point d'entrée de lecture du track. Il ne
modifie aucun modèle et ne promeut aucune variante supplémentaire. Top 5 égal
reste le challenger, Top 10 et les overlays risque restent non promus, Legacy
et SPY restent les contrôles obligatoires.

### Deep dive temporel interactif — 2026-08-05

Hypothèse : les résultats figés sur 2011-2025 masquent la concentration
temporelle éventuelle de l'alpha et rendent la géométrie H6 difficile à
comprendre. Le centre doit donc recalculer les métriques sur toute période
choisie, sans réentraîner ni modifier les prédictions OOS.

Données : les 172 rendements mensuels communs déjà sauvegardés dans
`monthly_portfolio_returns.csv`. Aucun nouveau résultat de modèle, prix ou
label n'est injecté.

Commande :

```bash
./.venv/bin/python \
  scripts/experiments/reports/render_central_research_dashboard.py
```

Résultat : l'onglet backtest explique désormais explicitement la chaîne mois
de décision → features EMA → cible de surperformance H6 → fold externe → Top
5 → détention un mois. Un sélecteur début/fin et des presets 3/5/10 ans
recalculent sur le sous-échantillon commun : rendement total, CAGR,
volatilité, Sharpe Legacy, Sortino, Calmar, drawdown, information ratio, beta,
alpha, corrélation, hit rate, VaR/CVaR historiques, captures, Omega, extrêmes,
années et épisodes de drawdown. Les diagnostics glissants sont disponibles sur
12/24/36/60 mois.

La sélection des EMA est elle aussi point-in-time : le manifeste par fold
commence avec 3 paires gagnantes et 15 variables au fold 1, puis s'élargit
jusqu'à 37 paires et 185 variables au fold 15. Le modèle historique ne reçoit
donc pas dès 2011 la liste finale connue en 2025.

Décision : ces découpes sont des audits des prédictions OOS figées et non de
nouveaux tests indépendants. Une fenêtre inférieure à 36 mois est signalée
comme fragile ; sous 12 mois les ratios annualisés ne doivent pas motiver une
promotion. Legacy et SPY restent toujours calculés sur exactement les mêmes
mois.

### Mutualisation du moteur portefeuille — 2026-08-09

Hypothèse : les comparaisons pouvaient encore dériver parce que Legacy, le
backtest boosting générique et le track multi-horizon possédaient des
implémentations différentes du portefeuille et des KPI. Les signaux doivent
rester propres à chaque méthodologie, mais une décision mensuelle identique doit
produire strictement le même rendement, le même turnover et les mêmes
statistiques quel que soit son adaptateur d'origine.

Implémentation : création de `src/alpharank/portfolio/` avec contrats de
holdings, pondérations égales/inverse-risk/sectorielles, simulateur, coûts,
performance, alignement temporel, artefacts standard et adaptateurs Legacy et
boosting. `backtest.portfolio`, `backtest.kpis`, `strategy.analytics`,
`multihorizon.trading`, `multihorizon.risk` et les scripts top-N/risk délèguent
maintenant à ces briques communes.

Données de validation : Legacy validé
`outputs/2026-07-27/runs/20260727_221253` et allocation Alpha/risk
`outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726`.

Commande :

```bash
./.venv/bin/python scripts/validate_common_portfolio_engine.py \
  --legacy-detailed outputs/2026-07-27/runs/20260727_221253/legacy_detailed_returns_polars.parquet \
  --legacy-aggregated outputs/2026-07-27/runs/20260727_221253/legacy_aggregated_returns_polars.parquet \
  --alpha-holdings outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/allocation_holdings.parquet \
  --alpha-monthly outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/allocation_monthly.csv \
  --legacy-data-manifest outputs/2026-07-27/runs/20260727_221253/data_input_manifest.json \
  --alpha-data-manifest outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/manifest.json \
  --allow-distinct-snapshots \
  --output outputs/common_portfolio_engine_validation_20260809.json
```

Résultat corrigé le 2026-08-10 : la parité mécanique passe à la tolérance
`1e-12`. Erreur maximale Legacy
`2.08e-16`, rendement Alpha `1.67e-16`, turnover `2.22e-16`. Le moteur commun
rejoue 197 mois complets pour chacun des deux portefeuilles Legacy et 1 376
strategy-months pour les huit allocations Alpha/risk. En revanche, ce couple
n'est pas comparable sur les données : Legacy utilise le snapshot 27 juillet,
Alpha celui du 13 juillet, et les sept hashes d'entrée consommés par Alpha
diffèrent. Le rapport porte donc `comparison_eligible=false`.

Décision : ce changement ne promeut aucun nouveau modèle et ne modifie pas les
signaux. Top 5 égal reste le challenger retenu sur son snapshot historique, Top
10 et les overlays restent diagnostiques. Toute future comparaison doit passer
à la fois le replay commun et le contrôle de lineage ; la parité du moteur seule
ne suffit plus.

### Centre Legacy / Boosting complet et diagnostics par fold — 2026-08-10

Hypothèse : le centre publié permettait déjà de filtrer le backtest et les SHAP
par mois, mais ne montrait pas les métriques train/calibration/test de chaque
modèle externe et expliquait Legacy moins précisément que le boosting.

Données : champion gelé
`legacy_ema_long_history_ticker_quarantine_v6_20260726`, comparaison Top 5
`legacy_ema_top5_vs_top10_quarantine_v7_20260726`, et replay diagnostic
`legacy_ema_fold_full_shap_20260810` depuis le snapshot du 13 juillet.

Résultat : les `76 534` prédictions test du replay diagnostic correspondent au
champion ligne par ligne ; écart maximal du score brut et de la probabilité
calibrée : `0`. Les 15 folds exposent désormais train, calibration et test,
leurs périodes, volumes, variables PIT, ROC AUC, PR AUC, lift, NDCG, excès Top
5 à un mois et overlap Legacy. Les SHAP couvrent désormais les `76 534`
observations test : `361` à `497` actions par mois, médiane `449`, avec une
égalité exacte entre prédictions et explications pour chacun des `172` mois.
Le détail mensuel est exhaustif et chargé depuis un fichier compressé auditable.
Seule la vue globale reste limitée à `80` observations par fold (`1 200` au
total) et elle est explicitement libellée comme échantillon.

Le rapport public compare uniquement Boosting Top 5, Legacy et SPY dans son
tableau principal. Les 14 878 fenêtres contiguës possibles utilisent
`advanced_performance_statistics()` du moteur portefeuille commun pour le
CAGR, la volatilité, le Sharpe, le Sortino, le Calmar, le max drawdown et les
autres indicateurs avancés. Le choix de période tranche les rendements OOS
figés ; il ne réentraîne aucun modèle.

Cause de l'ancien affichage à `1`–`22` points par mois : le runner sauvegardait
`80` lignes par fold sur l'ensemble de son bloc test, puis le dashboard
redistribuait ce sample entre les mois. Ce n'était pas une faiblesse de
couverture du modèle, mais une présentation mensuelle trompeuse. Le replay
exhaustif conserve exactement les `76 534` scores et probabilités du champion
(écart maximal `0`) ; seul le calcul SHAP est étendu à toutes les lignes.

Décision : aucune promotion nouvelle. Cette publication augmente l'auditabilité
du champion existant et conserve Top 10 uniquement comme diagnostic de dilution.

### Réconciliation Legacy / SPY sur intervalle — 2026-08-11

Hypothèse : les chiffres janvier 2012-décembre 2024 du centre public et d'un
recalcul Legacy autonome étaient supposés représenter la même série.

Résultat : l'hypothèse est fausse. Le centre utilise le snapshot commun au
Boosting `20260713_201639` : `Combined_Frequency` CAGR `15,8643 %`, performance
cumulée `578,1795 %`. Le dernier replay Legacy validé `20260727_221253` donne
`16,4033 %` et `620,3603 %`. Les `156/156` rendements mensuels diffèrent, dès
janvier 2012, car le second run intègre le preprocessing causal et le paquet de
données corrigé. Les deux séries ne doivent jamais être fusionnées ou porter
le même libellé sans snapshot.

Une deuxième ambiguïté indépendante concernait SPY. L'ancien modèle Legacy
`SP500` calcule le price return depuis `close` : `12,5863 %` de CAGR sur la
période. Le benchmark standard du moteur commun est le total return depuis
`adjusted_close` : `14,6440 %`. Les formules de CAGR étaient identiques ; les
entrées ne l'étaient pas.

Correction : `src/alpharank/portfolio/benchmark.py` centralise désormais les
deux conventions sous des identifiants non ambigus. Le standard est
`spy_total_return_adjusted_close`. `scripts/build_legacy_common_replay.py`
reconstruit un replay auditable depuis n'importe quel run figé, sans relancer
Optuna, avec manifeste et empreintes. Le site affiche dynamiquement les deux
Legacy pour l'intervalle choisi et interdit d'interpréter le price return comme
benchmark standard.

### Audit de cohérence des snapshots du centre — 2026-08-10

Hypothèse : la centralisation du simulateur pouvait donner l'impression que le
Legacy validé du 27 juillet et le boosting historique du 13 juillet étaient
comparés directement, alors que l'identité du snapshot n'était pas un contrat
du moteur commun.

Données contrôlées : manifestes du champion, du risque, du Top-N, du candidat
live, des runs Legacy des 13 et 27 juillet, et les 172 rendements Legacy
effectivement embarqués dans `monthly_portfolio_returns.csv`.

Résultat : le graphique historique est cohérent sur l'ancien snapshot. Les 172
rendements Legacy correspondent exactement au run `20260713_201639` avec une
erreur maximale `0`, et le boosting historique utilise le même snapshot. Ils ne
correspondent pas au run Legacy du 27 juillet. Le candidat live Legacy/Alpha
utilise séparément `20260727_221253`. Le validateur mécanique du moteur commun,
lui, associait bien deux snapshots différents et ne prouvait donc aucune
comparabilité de performance.

Décision : ajout de `src/alpharank/portfolio/lineage.py`, contrôle bloquant des
hashes avant comparaison, séparation explicite de `engine_parity_passed` et
`comparison_eligible`, et affichage dans le dashboard des contextes historique,
live et dernières données historisées. Le champion historique reste gelé ; il
doit être rerun sur un snapshot actuel commun avant toute nouvelle conclusion.

### Attribution exacte du CAGR Boosting / Legacy — 2026-08-11

Hypothèse : le niveau de performance du Top 5 pouvait provenir d'un titre ou
d'un petit nombre de mois extrêmes, sans moyen de relier exactement les
positions au CAGR affiché.

Données : mêmes artefacts historiques comparables du snapshot
`20260713_201639`, soit 172 mois d'août 2011 à novembre 2025 : holdings et
mensuels Top 5/Top 10 du run
`legacy_ema_top5_vs_top10_quarantine_v7_20260726`, replay commun Legacy/SPY
`legacy_20260713_201639_spy_total_return`.

Commande/run :
`scripts/experiments/reports/render_central_research_dashboard.py --output-dir outputs/research_dashboard/legacy_ema_alpha_central_20260811_cagr_attribution`.
Le moteur commun produit 5 408 lignes d'attribution. Chaque rendement mensuel
simple et chaque CAGR sont réconciliés à `1e-12`; les coûts restent une ligne
séparée.

Résultat : sur toute la fenêtre, le Top 5 affiche `37,47 %` de CAGR, soit
`31,82 %` de log-rendement annualisé additif. Les cinq meilleurs tickers
(`NVDA`, `NFLX`, `FANG`, `ALGN`, `MU`) représentent `27,8 %` du log-CAGR net et
les dix premiers `41,9 %`; la performance ne dépend donc pas d'un seul ticker,
mais reste concentrée. Les années les plus contributrices sont 2023 (`+5,59`
points log annualisés), 2016 (`+5,18`) et 2019 (`+5,17`). Les mois dominants
sont novembre 2020 (`+2,74`) et avril 2020 (`+2,71`); juin 2022 (`-1,53`) et
octobre 2018 (`-1,36`) sont les principaux freins. Plusieurs positions Top 5
ont réalisé `+40 %` à `+73 %` en un mois, notamment `NFLX` en janvier 2012 et
`FCX` en février 2016. Les coûts retirent `0,53` point de log-CAGR annualisé.

Décision : aucune promotion nouvelle. Le diagnostic confirme une performance
forte et concentrée, mais pas une anomalie arithmétique ni un unique ticker
magique. Toute décision doit encore tenir compte du multiple testing, des
révisions d'univers et du fait que le champion historique reste figé sur le
snapshot du 13 juillet. Le site permet désormais d'auditer toute fenêtre qui
affiche par exemple `31,5 %`, sans supposer que cette valeur correspond à la
fenêtre complète.

Contrôle explicite de la fenêtre février 2013-décembre 2022 (`119` mois) : CAGR
`31,49 %`, log-rendement annualisé `27,38 %`. Les principaux apports ticker sont
`FANG +2,29`, `NFLX +2,28`, `NVDA +1,85`, `ALGN +1,62` et `MU +1,46` points de
log annualisé ; les principaux retraits sont `CNX -1,56`, `MRNA -1,40` et
`RRC -1,08`. Les années 2016 et 2019 apportent respectivement `+7,50` et
`+7,47` points ; les coûts retirent `0,52` point.

### Rerun actuel Legacy / XGBoost sur un snapshot identique — 2026-08-11

Hypothèse : les écarts historiques du site provenaient d'abord de snapshots
différents. Il fallait donc rerun les deux méthodes sur une seule copie
immuable, puis bloquer la comparaison sur les hashes et les transformations de
qualité de données.

Données : snapshot open source retenu
`open_source_output_20260811_014746`, run id `20260811_001503`, prix actions et
SPY jusqu'au 10 août 2026. Legacy : `20260811_035522`. Boosting :
`legacy_ema_latest_common_score_tail_20260811_001503_standard`. Les sept jeux requis ont
les mêmes SHA-256 dans les deux manifestes. L'audit ultérieur a toutefois trouvé
une quarantaine différente : trois tickers côté Legacy contre dix côté
Boosting. Cet artefact ne passe donc plus le contrat de comparaison renforcé.

Protocole Boosting : classification H6, walk-forward expansif, validation six
mois, test douze mois, modèle gelé par fold, variables
`legacy_winners_pit_ema_only`, seed 42. Les métriques de modèle utilisent
uniquement les cibles H6 matures ; la queue récente est seulement scorée et
n'entre dans le portefeuille que si son rendement à un mois est réalisé.

Résultat modèle sur 15 folds : 81 267 prédictions, dont 78 006 cibles H6
évaluables, 1 996 en attente et 1 265 indisponibles par ticker. ROC AUC test
`0,60343`, NDCG@10 `0,54369` contre `0,50503` sans signal, PR AUC `0,16205`.
Les SHAP sont exhaustifs : 81 267 explications pour 81 267 prédictions sur 180
mois, sans échantillonnage.

Résultat portefeuille commun août 2011-juillet 2026 : Top 5 CAGR `27,8676 %`,
volatilité `38,3422 %`, Sharpe `0,6747`, max drawdown `-35,7550 %`; Top 10 CAGR
`23,5175 %`; Legacy CAGR `19,2049 %`, Sharpe `0,7706`; SPY total return CAGR
`14,3779 %`, Sharpe `0,8653`. Décision corrigée : chiffres historiques
auditables, mais comparaison de méthodes non éligible tant que les deux runs
n'utilisent pas exactement la même quarantaine. Aucune promotion du Top 5 ne
peut reposer sur cet artefact.

Publication :
`outputs/research_dashboard/legacy_ema_alpha_central_20260811_latest_common_001503_standard` et
`https://alpharank.net/research/alpharank_research_center.html`. Le site charge
les SHAP mensuels compressés, expose chaque fold et décompose exactement le
CAGR par action, année et mois via le moteur commun.

Reproductibilité entre les deux snapshots propres du 11 août : les 81 267
scores, probabilités calibrées et cibles communes sont identiques au bit près.
Legacy conserve exactement les mêmes tickers et rendements économiques ; les
seules différences des ledgers communs sont des arrondis flottants au plus égaux
à `2,22e-16`. Le run intermédiaire sans seuils standard n'est pas un résultat de
modèle comparable : il avait utilisé `1 / 0 / 1,0` au lieu de
`10 / 1 000 000 / 0,05`. Le profil `legacy_ema_latest_common_v1` et le replay
commun bloquent désormais cet écart de configuration.

Le contrat a ensuite été durci une seconde fois :
`scripts/build_common_legacy_boosting_replay.py` et le renderer exigent aussi
des listes `excluded_tickers` identiques. Legacy active désormais
`historical_ticker_exclusions_v1` par défaut. Le site retenu ci-dessus précède
ce garde-fou et doit être régénéré après un rerun aligné.

### Eligible shared-filter rerun — 2026-08-12

Hypothèse : l'écart de performance historique ne peut être interprété que si
Legacy et Boosting utilisent les mêmes entrées immuables, la même quarantaine
de tickers et le même filtre mensuel de qualité des prix.

Données et runs : snapshot open source `20260811_001503`, Legacy
`20260812_171646`, Boosting
`legacy_ema_latest_common_shared_eligibility_final_20260812`, replay commun
`legacy_boosting_shared_eligibility_final_20260812`. La comparaison couvre 180
mois détenus réalisés, d'août 2011 à juillet 2026.

Résultat : tous les contrôles de hashes, exclusions et politique de prix
passent. CAGR : Top 10 `24,2818 %`, Top 5 `23,6497 %`, Legacy `17,0257 %`, SPY
total return `14,3779 %`. Sharpe : `0,6788`, `0,5990`, `0,6740`, `0,8653` ; le
max drawdown Top 10 atteint `-40,3548 %`.

Décision : conserver Boosting comme challenger, sans le promouvoir sur le seul
CAGR. Sa performance ajustée du risque reste inférieure à Legacy et SPY. La
prochaine décision doit utiliser des preuves temporelles appariées et les
diagnostics de concentration/attribution.

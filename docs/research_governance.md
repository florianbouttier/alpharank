# Gouvernance des résultats de recherche

Dernière mise à jour : 2026-08-20.

Ce document définit les règles approuvées pour conserver, comparer et promouvoir
les résultats Legacy et Boosting. Il complète les contrats méthodologiques et de
simulation sans transformer une baseline reproductible en preuve causale.

## Baseline `v1-audited-biased`

Les artefacts audités du 16 août 2026 sont figés comme témoin historique :

- Legacy : run `20260816_142810` ;
- Boosting : `outputs/production_refresh_20260816/boosting_latest_common_v3` ;
- comparaison : `outputs/production_refresh_20260816/common_replay_v3` ;
- dashboard : `outputs/research_dashboard/alpharank_common_20260816_pit_validated`.

Cette baseline conserve les entrées, configurations, prédictions, positions,
rendements et rapports qui ont produit les métriques auditées. Son statut est
`audited_biased_not_causal`. Elle ne doit jamais être présentée comme validation
du Boosting ni être modifiée en place.

La façade publique reste `src/alpharank/governance.py`; les implémentations sont
attribuées par contrat dans `src/alpharank/governance_contracts/`. Le package de
baseline est créé avec `scripts/seal_methodology_baseline.py`. Chaque fichier
du payload possède une taille et un SHA-256 dans `baseline_manifest.json`. Le
SHA-256 du manifeste est conservé séparément dans
`baseline_manifest.sha256`. Les fichiers et répertoires perdent tous leurs bits
d'écriture après renommage atomique du package temporaire.

## Décisions approuvées

- Les structures, mois, tickers, rangs et décisions doivent être strictement
  identiques pour une migration déclarée neutre.
- La tolérance maximale sur les calculs numériques est `1e-12`; les fichiers
  seulement copiés doivent être identiques par SHA-256.
- Toute différence économique crée une nouvelle version et un rapprochement
  mois par mois. Elle ne réécrit jamais `v1`.
- Un run R&D sale reste autorisé si son patch est capturé. Une promotion finale
  exige un commit propre.
- Commit, état Git, diff, code critique, commande, configuration, seeds,
  interpréteur, dépendances, données et modèles appartiennent à la provenance
  obligatoire. Les secrets en sont exclus.
- Toute dérogation doit porter une approbation humaine explicite dans le
  manifeste.

## Contrat normatif `v2-causal`

Cette section est l'index normatif des corrections méthodologiques. Un statut
`Implémenté` dans la roadmap signifie que le contrôle existe ; il ne devient
`Validé` qu'après production, rapprochement et promotion du replay causal `v2`.
Les documents spécialisés restent propriétaires des détails, mais aucune règle
ci-dessous ne peut être contredite par un rapport ou un dashboard.

| Domaine | Règle normative | Propriétaire code / configuration | Test d'acceptation |
|---|---|---|---|
| Temps et cible (`BST-001`, `BST-003`, `QA-001`) | Une décision utilise seulement les features disponibles au cutoff. La maturité de la cible d'entraînement ne filtre jamais les titres classables et les mutations post-cutoff ne changent aucune décision passée. | `scripts/run_backtest.py`, `src/alpharank/data/feature_availability.py`; cible et horizon résolus dans le manifeste du run | `tests/unit/test_portfolio_engine.py::test_boosting_selection_ignores_future_return_availability`, `tests/replay/test_future_mutation_invariance.py::test_future_mutations_do_not_change_past_decisions` |
| Prix et vintages (`PRC-001` à `PRC-003`) | Un prix historique publié appartient à un vintage immuable. Toute révision crée un nouveau package et conserve le diff, la date de connaissance et le registre persistant ; un appel réseau ne réécrit jamais un historique scellé. | `src/alpharank/data/prices/`, `src/alpharank/data/price_revisions.py`; politiques et hashes dans les manifests de composition | `tests/integration/test_price_revisions.py::test_price_revision_requires_new_vintage`, `tests/integration/test_composed_snapshot.py::test_price_registry_promotion_preserves_payload` |
| Univers, secteurs et fondamentaux (`UNI-001` à `UNI-004`, `FND-001` à `FND-004`, `LEG-001`) | Membership, secteur et filing sont joints point-in-time avec provenance complète. Une couverture sectorielle partielle désactive le cap ; une donnée fondamentale officielle est SEC-only et devient disponible après le délai opérationnel de 24 heures. | `src/alpharank/data/open_source/constituents.py`, `src/alpharank/data/sector_history.py`, `src/alpharank/data/fundamental_coverage.py`, `src/alpharank/data/feature_availability.py`; politiques `sec-only-exclude-ex-ante-v1` et `sec-filing-availability-v1` | `tests/integration/test_open_source_constituents.py::test_membership_effective_at_decision_time`, `tests/unit/test_sector_history.py::test_sector_used_was_known_at_decision_date`, `tests/unit/test_strategy_legacy.py::test_legacy_sector_cap_uses_pit_sector` |
| Événements terminaux (`BST-002`, `SIM-001`) | Une sélection sans rendement réalisé échoue par défaut. Seul un événement terminal sourcé, connu et effectif pendant la détention peut résoudre le rendement ; `renormalize_available` est réservé au replay historique nommé. | `src/alpharank/portfolio/terminal_returns.py`, `src/alpharank/portfolio/simulation.py`; `missing_return_policy=raise` | `tests/unit/test_portfolio_engine.py::test_terminal_return_is_included`, `tests/unit/test_portfolio_engine.py::test_unresolved_terminal_return_does_not_promote_a_survivor` |
| Exécution (`LEG-003`, `LEG-005`, `SIM-004`) | Décision approuvée le 2026-08-20 : la performance AlphaRank canonique simule l'achat à la clôture de référence et mesure le rendement de clôture ajustée à clôture ajustée. La prochaine ouverture observée reste une sensibilité obligatoire, jamais la série publiée par défaut ; le VWAP n'est utilisé que s'il est observé. | Le replay commun conserve la série `adjusted_close` ; `src/alpharank/portfolio/execution.py` porte le défaut runtime `reference_close_adjusted_close_v1`, conserve `next_session_open_v1` et écrit le pont des deux séries. | `tests/unit/test_portfolio_execution.py` refuse l'inversion des rôles et tout écart de titres, mois, poids ou barème de coûts entre les deux séries. |
| Allocation et coûts (`SIM-002`, `SIM-003`) | Le turnover part des poids dérivés pré-trade, cash inclus. Les coûts sont des scénarios nommés et rapprochent spread, slippage, impact, commission/minimum et FX avec `net = brut - coûts`. | `src/alpharank/portfolio/allocation.py`, `src/alpharank/portfolio/costs.py`; `TransactionCostModel` versionné dans le manifeste | `tests/unit/test_portfolio_engine.py::test_turnover_uses_drifted_pretrade_weights`, `tests/unit/test_portfolio_engine.py::test_cost_model_is_monotonic_and_reconciled` |
| Limites, promotion et statut (`GOV-001` à `GOV-005`, `BST-006`) | `v1-audited-biased` reste immuable. Toute correction économique publie une autre version ; promotion et rollback sont atomiques, approuvés et conservent la version supersédée. Une confirmation finale est scellée avant ouverture. | façade `src/alpharank/governance.py`, propriétaires `governance_contracts/baseline.py`, `promotion.py` et `confirmation.py` ; pointeur de promotion, inventaire SHA-256 et protocole `sealed-confirmation-v1` | `tests/unit/test_governance_baseline.py::test_baseline_package_is_immutable`, `tests/production/test_governance_promotion.py::test_promotion_is_atomic_and_reversible` |
| Replay et provenance (`GOV-003`, `QA-002`, `QA-003`) | Un résultat publiable capture tout le runtime et doit être recalculé depuis ses entrées scellées. Toute mutation du code moteur, de la configuration, des entrées ou du modèle invalide le replay. | `src/alpharank/replay/validation.py`, `src/alpharank/governance_contracts/runtime_provenance.py`, façades `src/alpharank/replay_validation.py` et `src/alpharank/governance.py`, `.github/workflows/methodology-validation.yml` | `tests/unit/test_governance_runtime_provenance.py::test_manifest_captures_complete_runtime_provenance`, `tests/replay/test_recomputable_replay.py::test_replay_recomputes_outputs_from_sealed_inputs` |

Les sources de vérité détaillées sont :

- signal, cible et point-in-time :
  [`legacy_boosting_methodology.md`](./legacy_boosting_methodology.md) ;
- simulation, benchmark, coûts et comparaison :
  [`common_portfolio_backtest_engine.md`](./common_portfolio_backtest_engine.md) ;
- données SEC et prix :
  [`sec_fundamentals_contract.md`](./sec_fundamentals_contract.md) et
  [`sec_data_robustness_plan.md`](./sec_data_robustness_plan.md) ;
- production et replay mensuel :
  [`monthly_portfolio_runbook.md`](./monthly_portfolio_runbook.md) ;
- statut d'implémentation et preuves :
  [`METHODOLOGY_AUDIT_ROADMAP.md`](../METHODOLOGY_AUDIT_ROADMAP.md).

## Validation

`validate_baseline_package` recalcule l'inventaire et échoue si un fichier a été
ajouté, supprimé, modifié ou rendu inscriptible. La baseline n'est valide que si
le manifeste, son sceau détaché et l'intégralité du payload concordent.

## Garde du préfixe économique

`scripts/validate_economic_prefix.py` compare une référence publiée et un
candidat de migration. Le dernier mois de la référence définit le préfixe ; les
nouveaux mois du candidat restent hors comparaison. Les clés
stratégie/décision/détention/ticker, les rangs et les champs de décision sont
exacts. Les poids, rendements, turnover et coûts utilisent la tolérance absolue
approuvée de `1e-12`, obligatoirement accompagnée de sa justification.

Le rapport contient les SHA-256 canoniques, les clés manquantes ou inattendues et
l'écart maximal de chaque colonne. Une différence sur un mois publié interdit de
qualifier la migration de neutre ; elle doit être traitée comme correction
économique et produire une nouvelle version.

## Provenance runtime

Chaque nouveau run Legacy ou Boosting écrit un bloc `runtime_provenance` dans
son `data_input_manifest.json` et un artefact `runtime_git_patch.json` dans son
répertoire immuable. Le bloc enregistre la commande exacte, la configuration
résolue, les seeds, l'interpréteur et la plateforme, l'inventaire complet des
dépendances installées, les hashes du code critique et les identifiants de
données. Les valeurs dont le nom évoque un secret, mot de passe, token, clé API
ou credential sont remplacées par `<redacted>`.

La section Git contient le commit, la branche, l'état dirty réel, le hash de
l'état porcelain, le hash et la taille du patch suivi, ainsi que le nombre et le
hash d'inventaire des fichiers non suivis. Le bundle conserve le patch Git
binaire complet des fichiers suivis et les empreintes SHA-256 des fichiers non
suivis. Un run R&D sale est donc rejouable et attribuable sans embarquer le
contenu potentiellement sensible ou volumineux des fichiers non suivis.

`validate_runtime_provenance` échoue si un champ obligatoire ou l'artefact de
patch manque, si un hash diverge, ou si le manifeste déclare `git_dirty=false`
alors que le dépôt est sale. La validation contre le worktree courant est une
preuve de capture immédiate ; un replay ultérieur vérifie le manifeste et son
bundle sans exiger que le dépôt soit resté dans le même état.

## Mutations du futur

`tests/replay/test_future_mutation_invariance.py` constitue le garde transversal
anti-look-ahead. Il modifie séparément une cible future, un prix futur, un
événement de membership futur, un reclassement sectoriel futur et un filing
futur, puis exige l'identité des décisions, features ou attributs antérieurs au
cutoff. Les attributs datés utilisent `join_point_in_time_attributes`, qui
retient dans la sortie le timestamp effectif sélectionné et refuse les versions
dupliquées. Ce garde est une condition nécessaire de promotion ; les tâches
UNI/FND restent responsables de brancher toutes les sources de production sur
ces contrats point-in-time.

## Appartenance à l'univers

Les événements S&P sont appliqués à leur instant effectif, par défaut minuit
`America/New_York` de `effective_date` lorsqu'aucune heure plus précise n'est
fournie. `membership_at_decision_time` exige une décision timezone-aware et
rejoue les opérations jusqu'à cet instant inclus. Les snapshots mensuels datés
du premier jour représentent l'univers utilisable à la décision de fin de ce
mois : un événement effectif au milieu du mois n'est donc plus décalé au mois
suivant. Le snapshot de base est lui aussi rapproché avec les événements de son
mois et chaque opération reste dans l'audit.

La clé canonique d'un snapshot est `(Date, Ticker)`. Les doublons sont résolus
par le nom normalisé le plus fréquent ; une égalité est tranchée
lexicographiquement. Chaque groupe dupliqué conserve le nombre de lignes, tous
les noms candidats et leurs fréquences, le nom retenu et l'identifiant de la
règle. Sur le fichier actif contrôlé le 2026-08-17, 1 067 groupes à date non
nulle sont ainsi audités et la sortie de 225 620 lignes ne contient plus aucune
clé dupliquée.

Chaque événement d'univers doit désormais fournir `event_id`, `source_url`,
`observed_at`, `effective_at`, `effective_date` et `confidence`. Le registre est
refusé avant toute décision si l'un de ces champs manque, si l'identifiant est
dupliqué, si les timestamps ne sont pas zonés ou si l'observation est postérieure
à l'effet. Lorsqu'une source officielle publie seulement une date sans heure, la
connaissance est placée conservativement à 23:59:59 `America/New_York`. Le journal
de reconstruction propage ces champs jusqu'à chaque opération appliquée.

Les secteurs suivent le même principe : une classification n'est utilisable que
si son instant d'effet et son instant d'observation sont tous deux antérieurs à
la décision, avec identifiant, source et confiance. Le cap sectoriel est marqué
`sector_constraint_enabled=false` pour toute décision dont au moins un candidat
n'a pas de secteur point-in-time complet. Une table statique de secteurs courants
est explicitement incompatible avec ce contrat ; son branchement au moteur Legacy
relève de `LEG-001`.

La politique officielle pour une absence de fondamentaux est
`sec-only-exclude-ex-ante-v1` : seuls les ensembles SEC déjà disponibles à la
décision rendent un candidat éligible. Aucun fallback Yahoo, SimFin ou EODHD
n'est autorisé. L'exclusion ne reçoit ni rendement futur ni statut de survie et
produit un statut par ticker-mois ainsi qu'un rapport de couverture par année.
Les 3 023 ticker-mois et 62 tickers constatés dans la baseline auditée doivent
être recalculés par le replay `v2`, pas copiés comme une constante de décision.

Les métriques de flux TTM (`revenue`, résultats, EBITDA, FCF et EPS) sont des
sommes de quatre trimestres fiscaux distincts et contigus. Les moyennes de bilan
utilisées par les ratios exigent la même fenêtre complète. Une fenêtre de un à
trois trimestres, ou quatre observations séparées par un trimestre manquant,
reste indisponible : elle n'est jamais extrapolée en la multipliant par quatre.

Le calendrier earnings a pour clé canonique `(ticker, period_end)`. En cas de
doublon SEC, la sélection préfère un dépôt daté après la fin de période, puis le
délai le plus court, puis la date et l'accession lexicographiquement. Le lineage
conserve le nombre de candidats, toutes les accessions et l'identifiant de cette
règle ; l'ordre d'arrivée des lignes ne peut donc pas modifier le résultat.

La disponibilité des features issues d'un filing suit
`sec-filing-availability-v1`. Le timestamp d'acceptation SEC est prioritaire ;
s'il manque, la publication est supposée connue seulement à 23:59:59 New York
le jour du dépôt. Un délai opérationnel de 24 heures est ensuite ajouté. Chaque
valeur conserve son accession/version et le motif du timestamp ; le join de
décision impose mécaniquement `available_at <= decision_at`.

Les runners Legacy, backtest Boosting et multihorizon réservent leur dossier de
run par une création atomique avec `exist_ok=False`. Une collision d'identifiant
échoue avant toute lecture d'entrée ou écriture d'artefact ; aucun run existant
ne peut être repris, complété ou écrasé implicitement.

La promotion méthodologique utilise un unique pointeur JSON remplacé
atomiquement. Il conserve l'inventaire SHA-256 de chaque version, l'approbateur,
le motif et le journal promotion/rollback. Une version précédente devient
`superseded` mais son dossier n'est ni déplacé ni supprimé. Un rollback recalcule
tous ses hashes avant de la réactiver et refuse toute version modifiée.

Le turnover commun compare la nouvelle cible aux poids de pré-trade dérivés de
la performance de la période précédente, et non à l'ancienne cible. Le calcul
inclut le cash résiduel dans les deux vecteurs et traite toute entrée, sortie ou
perte totale dans la même demi-distance L1. Le premier investissement part d'un
portefeuille 100 % cash.

Le coût de transaction est maintenant un scénario nommé et décomposé en spread,
slippage, impact, commission/minimum de frais et change. Chaque composante est
exprimée en fraction de NAV, leur somme rapproche exactement
`transaction_cost`, et `net_return = gross_return - transaction_cost`. Un
scénario nul reproduit strictement le brut ; augmenter les paramètres ne peut
pas améliorer le net.

Le chemin R&D `run_backtest` conserve désormais la matrice brute jusqu'au fold
externe. Le filtre de colonnes trop creuses et les médianes de repli sont appris
sur le train de ce fold seulement, puis sérialisés dans
`fold_XX/preprocessor.json`. Les lignes de validation ou test ne peuvent donc
modifier ni la liste de features ni les valeurs d'imputation du passé.

Chaque fold Boosting écrit aussi un modèle natif `model.ubj` et un
`model_manifest.json` contenant son SHA-256, le préprocesseur, l'ordre des
features, la seed, le nombre d'itérations retenu et les bornes temporelles du
fold. Le cas mono-classe est sérialisé comme probabilité constante explicite.
Le chargeur de replay vérifie le hash avant de produire scores et rangs sans
réentraînement.

## Confirmation finale scellée

Le protocole `sealed-confirmation-v1` déclare avant toute observation la période
finale, son inventaire SHA-256 et la liste exhaustive des identifiants
d'expérience autorisés. Chaque variante doit enregistrer hypothèse, commande,
hash de configuration et hash du manifeste de résultat avant l'ouverture.

L'ouverture est unique et échoue en invalidant le protocole si le registre est
incomplet ou si les données ont changé. Toute nouvelle optimisation, seconde
ouverture ou mutation du registre après ouverture rend ensuite la promotion
impossible. `validate_confirmation_for_promotion` exige l'état `opened`, le
dataset intact et le hash exact du registre observé lors de l'ouverture.

## Promotion du snapshot de prix persistant

Un snapshot composé issu du contrat de prix persistant `v2` incorpore dans son
identifiant de composition le hash du manifeste de prix et celui du registre
`persistent_price_history_registry.parquet`. La composition copie le registre
avec son lineage, puis compare avant promotion le SHA-256, le nombre de lignes,
le nombre de clés uniques et le hash canonique de toutes les séries du fichier
`US_Finalprice.parquet`. Le validateur rejoue ces contrôles depuis le snapshot
publié ; une différence de transport, de clé ou de valeur bloque la promotion.

## Cap sectoriel Legacy point-in-time

Legacy résout désormais la classification de chaque candidat à l'instant de
décision via le contrat `UNI-004`. Le cap est appliqué avant la sélection finale
uniquement lorsque tous les candidats du mois disposent d'une classification
complète, observée et effective avant l'ordre. Une table statique de secteurs
courants, une classification future ou une couverture partielle désactive le
cap pour toute la date et laisse une raison explicite dans les sorties.

## Rendements Legacy et benchmark

Les titres et SPY utilisent tous deux `adjusted_close`, donc une convention de
rendement total incluant distributions et ajustements de splits. Le rendement
relatif est calculé seulement sur des dates de marché observées simultanément
pour le titre et le benchmark. Le benchmark n'est plus étendu par forward-fill
sur un calendrier civil ; toute date absente d'un côté reste indisponible au
lieu de créer une interpolation asymétrique.

## Exécution Legacy

La décision utilisateur du 2026-08-20 confirme la convention historique
AlphaRank : achat simulé à la clôture de référence, puis rendement de clôture
ajustée à clôture ajustée sur le mois détenu. L'interprétation introduite par
`LEG-003` le 2026-08-18, qui faisait de `next_session_open_v1` la convention
canonique, est supersédée ; ses artefacts restent conservés comme contrôle de
sensibilité.

Chaque rapport doit donc afficher séparément la série canonique de clôture et la
sensibilité à la prochaine ouverture, calculées sur les mêmes titres, mois,
poids et frais. Le VWAP de séance n'est affiché que lorsqu'une valeur réellement
observée existe ; aucun proxy n'est reconstruit depuis OHLC. `LEG-005` rend ce
choix explicite et bloquant : chaque nouveau run déclare
`reference_close_adjusted_close_v1` comme convention canonique et
`next_session_open_v1` comme sensibilité obligatoire. Le pont versionné vérifie
l'identité des titres, mois et poids ainsi que le même barème de coûts, sans
réécrire les anciennes baselines.

## Protocole de recherche Legacy

Le protocole `legacy-optuna-search-v1` verrouille l'espace entier des quatre
hyperparamètres, 30 trials par split annuel de janvier, les seeds 42 et 41, le
début de calibration `2010-01`, les cinq ancres et la règle de départage. Le
jeu de confirmation final est explicitement interdit pour la sélection.

Chaque run écrit `legacy_search_protocol.json` avec tous les trials Optuna,
leurs paramètres, scores et états, puis tous les candidats raffinés, le gagnant
et le motif de rejet des autres. Les exécutions smoke restent autorisées mais
sont marquées non promouvables si le nombre de trials, la période ou `n_jobs`
diffère du protocole verrouillé.

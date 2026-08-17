# Roadmap de remédiation — données, Legacy, Boosting et portefeuille

- Dernière mise à jour : 2026-08-17
- Périmètre : dépôt `alpharank` et dashboard du dépôt frère `../portfolio`
- État : Gate G0 franchie et quatorze corrections supplémentaires implémentées ; replay causal `v2` et promotion encore à faire
- Commit de création du document : `bafe06ba1afbbebb6e64657fae85db4422d5abc9`

## 1. Objectif et règle de non-réécriture

Cette feuille de route transforme les constats d'audit en changements atomiques,
testables et traçables. Elle couvre l'ingestion des données, les univers historiques,
les fondamentaux, Legacy, Boosting, le simulateur de portefeuille, l'ingestion IBKR,
le dashboard, la reproductibilité et la documentation.

La contrainte « les rendements passés ne doivent pas bouger » est appliquée de deux
façons différentes selon la nature du changement :

1. **Migration sans effet économique** : refactorisation, déplacement de données,
   enrichissement de lineage ou changement de format. Les rendements historiques
   doivent être strictement identiques. Le test porte sur les séries mensuelles,
   les positions, les turnovers et leurs empreintes cryptographiques.
2. **Correction d'un biais économique** : fuite du rendement futur, univers non
   point-in-time, rendement de radiation manquant, convention d'exécution ou coût
   erroné. Il serait incorrect de forcer la nouvelle méthode à reproduire le résultat
   biaisé. La série existante est alors figée, conservée et marquée `superseded`; une
   nouvelle version est publiée avec un rapprochement mois par mois. L'historique
   publié n'est jamais écrasé silencieusement.

Cette distinction est un garde-fou central : une parité parfaite peut reproduire un
biais, tandis qu'une correction causale peut légitimement modifier un rendement.

## 2. Références auditées

| Dépôt | HEAD observé | État lors de l'audit | Usage du hash |
|---|---|---|---|
| `portfolio` | `040d40099c58f9570866e7e02dcc99313e3832fd` | Worktree non propre | Référence de lecture seulement, pas une preuve de build reproductible |
| `../alpharank` | `05805bc3dd3bfb00fd49148fe0f5ecf649effb70` | Worktree non propre | Référence de lecture seulement, pas une preuve de build reproductible |

Contrôles déjà exécutés pendant l'audit :

- suite AlphaRank : `255 passed`, avec 147 avertissements ;
- suite backend Portfolio : `53 passed`, avec 2 avertissements ;
- build frontend : réussi, avec avertissement sur la taille du bundle ;
- replay Legacy strict : valide selon le validateur actuel ;
- snapshot composé : 9 empreintes contrôlées ;
- replay du moteur commun : 1 620 lignes stratégie-mois reproduites, écart maximal
  de turnover `2.22e-16`.

Ces résultats prouvent la stabilité technique du comportement actuel. Ils ne prouvent
pas à eux seuls l'absence de biais méthodologique.

Repères quantitatifs à figer dans `GOV-001`, sans les considérer comme une preuve de
validité causale :

| Série auditée | Mesure observée | Statut d'usage |
|---|---:|---|
| Boosting Top 5 | CAGR 28,1562 % | Baseline de rapprochement, résultat méthodologiquement contesté |
| Boosting Top 10 | CAGR 26,5717 % | Baseline de rapprochement, résultat méthodologiquement contesté |
| Boosting Top 10 matched-PE | CAGR 26,0726 % | Baseline de rapprochement, résultat méthodologiquement contesté |
| Legacy | CAGR 18,9965 % | Baseline de rapprochement ; corrections PIT et exécution encore à tester |
| SPY | CAGR 14,3975 % | Benchmark de la baseline auditée |

État de fraîcheur constaté le 2026-08-17 dans `portfolio` : prix locaux jusqu'au
2026-02-27, positions jusqu'au 2026-07-30 et transactions jusqu'au 2026-07-13. Ces
dates doivent rester des éléments de preuve de l'audit, pas des valeurs codées en dur.

## 3. Légende de suivi

### Criticité

| Niveau | Définition | Règle de publication |
|---|---|---|
| `P0 — Bloquant` | Peut invalider la causalité, la sélection ou la performance publiée | Aucun résultat Boosting/Legacy comparatif ne doit être présenté comme validé avant correction |
| `P1 — Critique` | Peut modifier sensiblement le risque, le rendement ou la reproductibilité | Correction exigée avant promotion en production |
| `P2 — Élevée` | Réduit la fidélité économique ou rend l'audit difficile | À traiter avant la prochaine version méthodologique stable |
| `P3 — Moyenne` | Dette de structure, observabilité ou documentation | Peut suivre la validation économique, avec ticket et échéance |
| `P4 — Faible` | Amélioration non bloquante | À planifier selon capacité |

### Statuts autorisés

- `À faire` : aucun commit d'implémentation validé.
- `En cours` : changement commencé, critères d'acceptation non satisfaits.
- `Bloqué` : dépendance explicite manquante.
- `Implémenté` : code présent, validation complète encore manquante.
- `Validé` : code, tests, documentation et preuves d'acceptation sont complets.
- `Abandonné` : décision documentée avec justification.
- `Supersédé` : version conservée pour audit mais remplacée par une version plus récente.

### Règle du champ commit

Le champ `Commit` reste `—` tant que le changement et son test ne sont pas contenus
dans un commit identifiable. Utiliser le hash Git complet de 40 caractères. Le HEAD
observé pendant l'audit ne doit jamais être copié comme commit de réalisation d'une
tâche qui n'y est pas effectivement implémentée.

## 4. Résumé des constats qui pilotent la priorité

- **Boosting — fuite future P0** : le classement est actuellement effectué après
  exclusion de lignes dont `future_return_1m` est nul. Sur 180 mois réalisés, cette
  condition change 17 sélections Top 5, 30 Top 10 et 47 Top 20.
- **Radiations et censure P0** : 1 497 observations hors échantillon disposent d'un
  rendement futur du benchmark mais pas du rendement futur du titre. Les supprimer
  favorise les titres survivants.
- **Univers historique P0/P1** : les événements de composition sont décalés au mois
  suivant alors que le fichier les décrit comme effectifs au début du mois daté ;
  des doublons sont présents et la provenance historique exacte doit être renforcée.
- **Secteurs P1** : la contrainte sectorielle Legacy repose sur une classification
  statique, non point-in-time.
- **Simulation P0/P1** : le mode par défaut peut renormaliser les titres dont le
  rendement est disponible et le turnover est calculé entre poids cibles, non entre
  poids de fin de période dérivés et nouvelle cible.
- **Dashboard P0/P1** : les performances sont reconstruites depuis le ledger de
  transactions sans rapprochement bloquant avec les snapshots de positions ; en
  absence de prix historique, un `mark_price` courant peut être répété dans le passé.
- **IBKR P1** : trois crédits d'intérêts, soit 41,45 EUR lors de l'audit, sont classés
  comme dépôts et neutralisés dans le TWR.
- **Reproductibilité P0/P1** : l'audit initial constatait que les packages de replay
  ne capturaient pas tout le code, l'environnement, les modèles et l'état Git ; le
  contrat runtime commun de `GOV-003` corrige désormais la capture des nouveaux runs.

## 5. Chemin critique et gates

1. **Gate G0 — Gel — franchie le 2026-08-17** : `GOV-001`, `GOV-002`, `GOV-003`.
2. **Gate G1 — Causalité Boosting** : `BST-001`, `BST-002`, `BST-003`, `SIM-001`,
   `SIM-004`, `QA-001`.
3. **Gate G2 — Point-in-time** : `UNI-001` à `UNI-004`, `FND-001` à `FND-004`.
4. **Gate G3 — Réalisme portefeuille** : `SIM-002`, `SIM-003`, `LEG-002`,
   `LEG-003`.
5. **Gate G4 — Reproductibilité** : `GOV-004`, `GOV-005`, `BST-005`, `QA-002`.
6. **Gate G5 — Production Portfolio** : `DASH-001` à `DASH-006`, `QA-003`.
7. **Gate G6 — Publication** : `DOC-001`, `DOC-002`, puis comparaison scellée
   Legacy/Boosting.

Une gate n'est franchie que lorsque toutes ses tâches P0 et P1 sont `Validé`.

### Tableau de progression courant

| Catégorie | Total | P0 | P1 | P2 | Implémenté | Validé | Progression validée |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gouvernance | 5 | 2 | 3 | 0 | 0 | 3 | 60 % |
| Prix et vintages | 3 | 0 | 3 | 0 | 1 | 0 | 0 % |
| Univers et secteurs | 4 | 1 | 3 | 0 | 4 | 0 | 0 % |
| Fondamentaux | 4 | 1 | 1 | 2 | 3 | 0 | 0 % |
| Boosting | 6 | 3 | 3 | 0 | 3 | 0 | 0 % |
| Legacy | 4 | 0 | 3 | 1 | 0 | 0 | 0 % |
| Simulation | 4 | 2 | 1 | 1 | 2 | 0 | 0 % |
| Dashboard et IBKR | 6 | 1 | 4 | 1 | 0 | 0 | 0 % |
| Qualité et documentation | 5 | 2 | 1 | 2 | 1 | 0 | 0 % |
| **Total** | **41** | **12** | **22** | **7** | **14** | **3** | **7,3 %** |

Mettre ce tableau à jour dans le commit documentaire de suivi immédiatement après
chaque commit d'action. Le total des criticités doit toujours égaler le total des tâches.

## 6. Roadmap détaillée

### A. Gouvernance, invariance et versionnement

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `GOV-001` | P0 | Figer les artefacts actuels Legacy, Boosting et benchmark en baseline `v1-audited-biased`; inclure entrées, sorties, configuration et rapport. | `test_baseline_package_is_immutable` : toute réécriture d'un fichier scellé échoue ; inventaire et SHA-256 complets. | Validé | `f526a11ff1aab53e39edbfdd7c99f309e0f8d3b4` | Aucun ; 266 fichiers / 297 256 217 octets conservés exactement |
| `GOV-002` | P0 | Ajouter un garde de préfixe économique commun aux migrations sans effet économique. Dépend de `GOV-001`. | `test_economic_prefix_is_bitwise_stable` : mêmes mois, tickers, poids, rendements bruts/nets et turnover ; écart maximal `0` ou tolérance explicitement justifiée pour la sérialisation. | Validé | `3f2f8aa235329197b759f4a7d84fcc0e2700adf9` | Identique sur la baseline : écarts maximaux nuls |
| `GOV-003` | P1 | Étendre les manifests avec commit Git, état dirty, diff hash, dépendances, interpréteur, configuration, code critique et identifiants de données. | `test_manifest_captures_complete_runtime_provenance` : échec si un champ requis est absent ou si `git_dirty` est faux alors que le worktree est sale. | Validé | `2f89e39b519355a51be569aaf118a50f9fc46d31` | Aucun ; manifests Legacy et Boosting enrichis |
| `GOV-004` | P1 | Rendre les répertoires de run uniques et atomiques ; interdire `exist_ok=True` sur un identifiant déjà utilisé. Dépend de `GOV-003`. | `test_run_directory_cannot_be_overwritten` : un second run avec le même ID échoue avant toute écriture. | À faire | — | Aucun attendu |
| `GOV-005` | P1 | Définir promotion, rollback et supersession : pointeur canonique atomique, ancienne version conservée, motif et approbation enregistrés. | `test_promotion_is_atomic_and_reversible` : interruption simulée sans pointeur partiel ; rollback retrouve tous les hashes précédents. | À faire | — | Aucun écrasement ; nouvelle version si méthode corrigée |

### B. Prix, ajustements et vintages de données

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `PRC-001` | P1 | Recomposer le snapshot canonique avec le contrat de prix persistant récent et son registre de lineage, sans changer les octets de `US_Finalprice`. Dépend de `GOV-002`. | `test_price_registry_promotion_preserves_payload` : SHA-256 du prix, nombre de lignes, clés et séries économiques identiques avant/après. | À faire | — | Doit rester identique |
| `PRC-002` | P1 | Formaliser les révisions de prix, splits, dividendes et corrections fournisseur par vintage et date de connaissance. | `test_price_revision_requires_new_vintage` : une valeur historique modifiée ne peut pas remplacer le vintage canonique sans nouveau package et rapport de diff. | Implémenté | `30904d777eef48abcea662d1f12c34505fe46de5` | Ancien vintage immuable ; nouveau résultat versionné ; package réel à produire pour validation |
| `PRC-003` | P1 | Construire un cache de prix du dashboard figé par date d'ingestion ; interdire qu'un appel réseau modifie implicitement une période historique déjà publiée. | `test_dashboard_history_is_stable_when_provider_changes` : deux réponses fournisseur différentes donnent le même historique pour un vintage scellé. | À faire | — | Doit rester identique pour un vintage donné |

### C. Univers d'investissement et secteurs point-in-time

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `UNI-001` | P0 | Construire l'appartenance à l'univers `as-of` à l'instant de décision à partir de la date/heure effective, sans décalage mensuel implicite. | `test_membership_effective_at_decision_time` avec VEEV, MRVL, FLEX, EA et FERG : présence exacte avant et après chaque événement. | Implémenté | `0acf0963ec94e69a234f1f82c08055027f16eecd` | Nouvelle baseline requise : les événements intra-mois ne sont plus décalés au mois suivant |
| `UNI-002` | P1 | Imposer une clé unique documentée pour les constituants et résoudre les doublons avec une règle déterministe et auditée. | `test_constituent_snapshot_has_unique_key` : zéro doublon non résolu ; les 214 groupes historiques détectés ont une décision traçable. | Implémenté | `1c67a9801a7ccb606a6ede2107bc57fe0772f6e6` | Fichier actif : 1 067 groupes datés audités, sortie de 225 620 clés uniques ; replay v2 restant |
| `UNI-003` | P1 | Ajouter provenance source, date d'observation, date effective, identifiant d'événement et niveau de confiance à chaque changement historique. | `test_membership_event_lineage_is_complete` : 100 % des événements utilisés par une décision ont une provenance et une date effective. | Implémenté | `3b816e4ca494d27e49c5f0bed23655abaa869ccb` | Payload économique inchangé ; 10 événements et 17 opérations du registre actif ont une provenance complète |
| `UNI-004` | P1 | Fournir une classification sectorielle point-in-time ; à défaut, désactiver la contrainte sectorielle sur les périodes non couvertes au lieu d'utiliser le secteur courant. | `test_sector_used_was_known_at_decision_date` : aucun secteur dont la date de disponibilité est postérieure à la décision ; scénario de changement de secteur inclus. | Implémenté | `3690fa93ba8e6bbce51c528de4e12f8e5ca35ee4` | Nouvelle baseline Legacy probable ; branchement du cap au contrat PIT suivi par `LEG-001` |

### D. Fondamentaux SEC, disponibilité et TTM

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `FND-001` | P1 | Définir une politique explicite pour les 3 023 ticker-mois sans données SEC et les 62 tickers concernés : fallback, exclusion ex ante ou indicateur de couverture. | `test_missing_fundamentals_policy_is_ex_ante` : la décision ne dépend ni du rendement futur ni de la survie ultérieure ; rapport de couverture par année. | Implémenté | `89ac75cce09db0c7f89691b21dd1e469aea196ab` | SEC uniquement, exclusion ex ante et statut ticker-mois ; couverture réelle à recalculer au replay v2 |
| `FND-002` | P2 | Remplacer la moyenne glissante multipliée par quatre avec `min_samples=1` par un vrai TTM fondé sur quatre trimestres ou un état d'indisponibilité explicite. | `test_ttm_requires_four_distinct_quarters` : aucun TTM sur 1 à 3 trimestres ; somme vérifiée sur exemples de filings. | Implémenté | `33f9c47fc5a619ec769d0e670c6fada4b2c319e0` | Nouvelle baseline requise : TTM partiels supprimés, lacunes trimestrielles laissées indisponibles |
| `FND-003` | P2 | Résoudre les doublons du calendrier earnings avec une clé et une priorité de source déterministes. | `test_earnings_calendar_key_is_unique` : zéro doublon après consolidation ; les dix cas observés ont un résultat attendu fixé. | Implémenté | `c4600c55377491c7853ba63a7ac40ad5b02c9597` | Aucun attendu sur les clés non conflictuelles ; accessions candidates et règle conservées pour chaque conflit |
| `FND-004` | P0 | Appliquer partout une disponibilité point-in-time stricte : `filing_date`, heure de publication, délai opérationnel et version du filing. | `test_feature_availability_precedes_decision` : pour chaque valeur de feature, `available_at <= decision_at`; mutations d'un filing futur sans effet sur le passé. | À faire | — | Nouvelle baseline requise si une fuite est trouvée |

### E. Algorithme Boosting

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `BST-001` | P0 | Classer tous les candidats éligibles avec le score disponible à la décision, sélectionner Top N, puis seulement joindre le rendement réalisé. Ne jamais filtrer sur `future_return_1m` avant sélection. | `test_boosting_selection_ignores_future_return_availability` : rendre nuls les rendements futurs du titre et du benchmark sans changer le Top N. | Implémenté | `36ec9be79a7faa70c1ae4abb93a5bad60f766247` | Nouvelle baseline Boosting obligatoire ; validation finale après replay `v2` |
| `BST-002` | P0 | Résoudre le rendement terminal total actionnaire pour radiations, faillites, acquisitions et changements de ticker. Dépend de `PRC-002`. | `test_terminal_return_is_included` : cas cash merger, échange d'actions, radiation à perte totale et pont de ticker ; aucun survivant implicite. | Implémenté | `a6feee33fd8fffcfa5f2255c5d55a3ddb079773b` | Nouvelle baseline Boosting obligatoire ; validation finale après replay `v2` |
| `BST-003` | P0 | Définir la censure de la cible d'entraînement : distinguer horizon non encore réalisé, donnée manquante et événement terminal ; supprimer toute exclusion corrélée à la survie. Dépend de `BST-002`. | `test_training_target_missingness_is_not_survival_filter` : les 1 497 observations sont classées par motif, sans drop générique ; rapport de censure par fold. | Implémenté | `34df93c7f99ff1f3961bc36b1d1b4f9422e38ce1` | Nouvelle baseline modèle obligatoire ; replay bloqué tant que les 1 497 cibles matures ne sont pas résolues |
| `BST-004` | P1 | Corriger ou déprécier `scripts/run_backtest.py`, dont sélection sparse et médiane sont actuellement calculées sur l'échantillon complet avant les folds. | `test_preprocessing_is_fit_inside_each_outer_fold` : mutation du futur sans effet sur features, colonnes retenues et imputations du passé. | À faire | — | Nouvelle baseline pour ce chemin R&D uniquement |
| `BST-005` | P1 | Sérialiser chaque modèle, préprocesseur, liste de features, seed et métadonnées de fold ; permettre un replay sans réentraînement. Dépend de `GOV-003`. | `test_serialized_model_reproduces_oos_predictions` : prédictions, rangs et portefeuille hors échantillon identiques après rechargement. | À faire | — | Doit rester identique pour le même run |
| `BST-006` | P1 | Sceller un jeu de confirmation final et enregistrer toutes les variantes testées pour limiter le biais de sélection et le multiple testing. | `test_sealed_period_is_single_use` : toute lecture prématurée ou nouvelle optimisation après ouverture invalide la promotion ; registre des expériences complet. | À faire | — | Aucun recalage rétroactif autorisé |

### F. Stratégie Legacy

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `LEG-001` | P1 | Brancher le cap sectoriel sur `UNI-004` et déclarer le comportement lorsque le secteur PIT manque. | `test_legacy_sector_cap_uses_pit_sector` : un reclassement futur ne modifie aucune décision passée ; cap vérifié avant l'ordre. | À faire | — | Nouvelle baseline Legacy probable |
| `LEG-002` | P1 | Harmoniser l'éligibilité prix du titre et du benchmark, les ajustements dividendes/splits et le calcul d'excès de rendement. | `test_asset_and_benchmark_return_conventions_match` : même calendrier, même convention de total return, aucune interpolation asymétrique. | À faire | — | À mesurer |
| `LEG-003` | P1 | Déclarer l'instant exact du signal et de l'exécution ; comparer clôture, prochaine ouverture et VWAP avec données réellement disponibles. | `test_order_price_occurs_after_signal_cutoff` : timestamp d'exécution strictement postérieur au cutoff ; rapport de sensibilité obligatoire. | À faire | — | Nouvelle série de sensibilité ; convention canonique à versionner |
| `LEG-004` | P2 | Verrouiller le protocole Optuna et les ancres : espace de recherche, seeds, période de calibration, candidats rejetés et règle de choix. | `test_legacy_search_protocol_is_locked` : même manifeste, mêmes trials et même gagnant ; aucune donnée de validation finale utilisée pour choisir. | À faire | — | Aucun retuning rétroactif autorisé |

### G. Simulateur commun de portefeuille

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `SIM-001` | P0 | Remplacer `renormalize_available` comme défaut par un mode `raise` ou par un rendement terminal explicitement résolu. | `test_missing_selected_return_fails_closed_by_default` : un titre sélectionné sans rendement produit une erreur qualifiée ; l'ancien replay demande explicitement `renormalize_available`. | Implémenté | `0cf79a9c5c19f2a83b47508e09b27aa885e402ac` | Nouvelle baseline si des mois réalisés sont touchés ; l'ancienne baseline reste reproductible à `2.08e-16` |
| `SIM-002` | P1 | Calculer le turnover entre les poids dérivés après performance et les nouveaux poids cibles, avec gestion du cash et des entrées/sorties. | `test_turnover_uses_drifted_pretrade_weights` : exemples analytiques à deux actifs et comparaison indépendante. | À faire | — | Nouvelle baseline nette ; ordre de grandeur audité : environ 0,44 point cumulé sur Boosting Top 10 à 10 pb |
| `SIM-003` | P2 | Ajouter spreads, slippage, impact, minimum de frais, change et scénarios de coûts, séparés des rendements bruts. | `test_cost_model_is_monotonic_and_reconciled` : coût nul reproduit le brut ; coût croissant ne peut améliorer le net ; somme des coûts rapproche le P&L. | À faire | — | Séries nettes nouvelles, brut inchangé |
| `SIM-004` | P0 | Imposer la frontière causale globale décision-exécution-rendement et intégrer les événements terminaux résolus par `BST-002`. | `test_holding_return_starts_after_trade` : première observation de rendement après exécution ; aucune donnée de la période détenue dans le signal. | Implémenté | `5b96074747c94058693607bdd7b5ef828aaa4804` | Nouvelle baseline requise ; `v1` demande désormais explicitement `legacy_month_only` |

### H. Ingestion IBKR et performance du dashboard

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `DASH-001` | P1 | Classer explicitement intérêts, dividendes, frais, taxes et transferts ; faire échouer ou mettre en quarantaine les descriptions cash inconnues au lieu de les transformer en dépôt/retrait. | `test_interest_cash_is_not_external_flow` : les trois exemples audités totalisant 41,45 EUR augmentent le rendement et ne sont pas neutralisés dans le TWR. | À faire | — | Nouvelle baseline dashboard requise |
| `DASH-002` | P0 | Rapprocher quotidiennement ledger, corporate actions et `positions_history.parquet`, qui reste la source de vérité ; bloquer la publication si l'écart inexpliqué dépasse la tolérance. | `test_ledger_reconciles_to_position_snapshots` : quantités par compte/ticker/date identiques, avec exceptions FX typées ; splits et transferts couverts. | À faire | — | Nouvelle baseline si écarts ou corporate actions détectés |
| `DASH-003` | P1 | Exposer séparément fraîcheur positions, cash, transactions, prix, FX et valorisation ; avertissement visible et statut final/provisoire. | `test_freshness_contract_reports_each_source` : dates exactes, aucune date agrégée trompeuse, seuils de staleness testés. | À faire | — | Aucun attendu |
| `DASH-004` | P1 | Interdire la répétition d'un `mark_price` actuel sur tout l'historique lorsqu'un prix manque ; utiliser prix versionné ou état indisponible. Dépend de `PRC-003`. | `test_current_mark_is_never_backfilled_into_history` : fournisseur indisponible et historique partiel donnent une erreur/rupture documentée, pas une série plate artificielle. | À faire | — | Nouvelle baseline possible |
| `DASH-005` | P2 | Décomposer `backend/app/analytics/engine.py` en modules ingestion, pricing, positions, cashflows, performance, attribution et risque, sans changement économique. Dépend de `GOV-002`. | Tests de caractérisation plus `make test` : mêmes réponses API et hashes économiques sur fixtures avant/après. | À faire | — | Doit rester identique |
| `DASH-006` | P1 | Remplacer les `except` larges et fallbacks silencieux par erreurs typées, métriques et lineage de fallback dans l'API. | `test_fallback_is_visible_and_typed` : chaque panne simulée indique source, cause, fallback et dégradation ; aucun `except Exception: pass`. | À faire | — | Aucun attendu, sauf suppression d'un fallback invalide |

### I. Tests anti-biais, validation, CI et documentation

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `QA-001` | P0 | Créer une suite de tests sémantiques par mutation du futur : cible, prix futur, membership futur, secteur futur et filing futur. | `test_future_mutations_do_not_change_past_decisions` : scores et ordres antérieurs au cutoff restent identiques pour chaque mutation. | Implémenté | `68a1f557aeb146b5a1f031570c67086ef86d5365` | Aucun ; contrat générique prêt, branchement production UNI/FND et replay v2 restants |
| `QA-002` | P0 | Étendre les validateurs pour recalculer les sorties depuis le package, pas seulement vérifier quelques hashes de fichiers. Inclure tout le moteur commun et les règles d'éligibilité. | `test_replay_recomputes_outputs_from_sealed_inputs` : environnement neuf, sorties identiques ; échec à toute mutation de code, config, entrée ou modèle. | À faire | — | Doit reproduire exactement une version donnée |
| `QA-003` | P1 | Ajouter une matrice CI des deux dépôts : tests unitaires, tests anti-look-ahead, replay court, validation documentation et build frontend. | Pipeline : AlphaRank complet, `make test`, `npm run build`, validateurs de docs et replay smoke tous verts sur commit propre. | À faire | — | Aucun attendu |
| `DOC-001` | P2 | Mettre à jour les sources de vérité après chaque correction : contrat temporel, univers, prix, cible, exécution, coûts, limites et procédure de replay. | `test_documentation_structure.py` et revue croisée code/doc : chaque règle normative pointe vers son test et sa configuration. | À faire | — | Aucun attendu |
| `DOC-002` | P2 | Afficher dans les rapports et le dashboard version méthodologique, vintage de données, commit, statut `provisional/final/superseded` et avertissements connus. | `test_report_exposes_methodology_identity` : informations présentes et cohérentes avec le manifeste ; impossible de publier sans identité complète. | À faire | — | Aucun attendu |

## 7. Protocole de validation par changement

Chaque tâche suit le même ordre. Un test de parité seul n'autorise jamais la promotion
d'une méthode dont la causalité n'a pas été vérifiée.

1. **Fixture minimale** : reproduire le défaut sur un cas synthétique et, lorsque
   possible, sur un exemple réel audité.
2. **Test rouge** : ajouter le test d'acceptation et confirmer qu'il échoue sur la
   version de référence.
3. **Correction minimale** : modifier le plus petit périmètre compatible avec le
   contrat cible.
4. **Test ciblé** : exécuter le test associé et les tests du module.
5. **Test causal** : muter les données postérieures au cutoff et confirmer que les
   décisions passées restent identiques.
6. **Replay économique** : recalculer positions, rendements bruts, turnover, coûts et
   rendements nets.
7. **Qualification de l'effet** :
   - migration neutre : preuve de parité exacte ;
   - correction économique : diff mois par mois, explication de chaque rupture et
     publication en nouvelle version.
8. **Validation complète** : suites des deux dépôts, build frontend et validateurs.
9. **Commit atomique** : code, test et documentation ensemble ; reporter le hash dans
   cette roadmap.
10. **Promotion** : seulement après satisfaction des critères et mise à jour du journal.

## 8. Commandes minimales de contrôle

Depuis la racine du dépôt `alpharank` :

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python scripts/validate_common_portfolio_engine.py
.venv/bin/python scripts/validate_legacy_replay_package.py --help
python3 scripts/validate_documentation.py
```

Depuis `../portfolio` :

```bash
make test
cd frontend && npm run build
python3 scripts/validate_documentation.py
```

Le validateur Legacy doit être appelé avec le package exact à promouvoir. La commande
finale complète, incluant le chemin du package, doit être copiée dans la preuve de la
tâche concernée ; `--help` ci-dessus ne constitue pas une validation.

## 9. Définition de terminé

Une tâche ne peut passer à `Validé` que si toutes les conditions suivantes sont vraies :

- le changement est implémenté dans le bon dépôt ;
- le test associé échoue sur la référence fautive et réussit après correction, lorsque
  cette démonstration est praticable ;
- les tests ciblés et la suite complète pertinente réussissent ;
- le contrat temporel et la disponibilité des données sont explicités ;
- l'effet sur chaque série historique est classé : identique, expliqué ou non applicable ;
- les artefacts avant/après et leurs SHA-256 sont conservés ;
- la documentation normative est alignée avec le code ;
- le commit complet est renseigné dans la ligne de roadmap ;
- le journal de suivi ci-dessous contient le lien vers la preuve ;
- aucun résultat `provisional` ou `superseded` n'est présenté comme `final`.

## 10. Ordre de commits recommandé

Les hashes seront ajoutés après réalisation. Chaque groupe doit rester atomique ; si le
diff devient large, créer plusieurs commits sans mélanger les catégories.

| Ordre | IDs principaux | Message indicatif |
|---|---|---|
| 1 | `GOV-001`, `GOV-002` | `test: freeze audited strategy baselines and economic prefixes` |
| 2 | `BST-001`, `QA-001` | `fix: select boosting portfolios before realized return joins` |
| 3 | `BST-002`, `BST-003`, `SIM-001` | `fix: model terminal returns and target censoring explicitly` |
| 4 | `UNI-001`, `UNI-002`, `UNI-003` | `fix: build point-in-time constituent membership` |
| 5 | `UNI-004`, `LEG-001` | `fix: use point-in-time sector classifications` |
| 6 | `FND-001` à `FND-004` | `fix: enforce filing availability and fundamental coverage policy` |
| 7 | `SIM-002`, `SIM-003`, `SIM-004` | `fix: use drifted turnover and causal execution costs` |
| 8 | `GOV-003` à `GOV-005`, `BST-005`, `QA-002` | `feat: seal reproducible immutable research runs` |
| 9 | `DASH-001`, `DASH-002` | `fix: reconcile IBKR cashflows and position history` |
| 10 | `PRC-003`, `DASH-003`, `DASH-004`, `DASH-006` | `fix: freeze dashboard vintages and expose degraded data` |
| 11 | `DASH-005` | `refactor: split portfolio analytics engine without economic changes` |
| 12 | `QA-003`, `DOC-001`, `DOC-002` | `docs: publish methodology identity and validation gates` |

## 11. Journal de suivi

Ajouter une ligne à chaque changement de statut. Ne pas modifier les anciennes lignes.

| Date UTC | ID | Responsable | Cible | Ancien statut | Nouveau statut | Commit | Preuve / commande / artefact | Note |
|---|---|---|---|---|---|---|---|---|
| 2026-08-17 | `ROADMAP` | — | Initialisation | — | Créée | `bafe06ba1afbbebb6e64657fae85db4422d5abc9` | Audit code, données, artefacts et tests des deux dépôts | Première version ; aucune remédiation déclarée réalisée |
| 2026-08-17 | `BST-001` | Codex | Gate G1 | À faire | Implémenté | `36ec9be79a7faa70c1ae4abb93a5bad60f766247` | Test ciblé 13/13 ; suite complète 256/256 | Sélection indépendante de la disponibilité des rendements futurs ; replay `v2` restant |
| 2026-08-17 | `SIM-001` | Codex | Gate G1 | À faire | Implémenté | `0cf79a9c5c19f2a83b47508e09b27aa885e402ac` | Tests ciblés 16/16 ; suite complète 257/257 ; parité Legacy max `2.08e-16` | Défaut fail-closed ; renormalisation uniquement explicite pour baseline historique |
| 2026-08-17 | `PRC-002` | Codex | Prix et vintages | À faire | Implémenté | `30904d777eef48abcea662d1f12c34505fe46de5` | 25 tests prix ciblés ; suite complète 265/265 ; validation documentation réussie | Nouveau vintage obligatoire, preuve datée et diff SHA-256 ; aucun package de production promu |
| 2026-08-17 | `BST-002` | Codex | Gate G1 | À faire | Implémenté | `a6feee33fd8fffcfa5f2255c5d55a3ddb079773b` | 22 tests portefeuille ciblés ; suite complète 265/265 ; replay mécanique Legacy/Boosting dans `1e-12` | Rendements terminaux après sélection ; cas non résolu conservé et fail-closed ; replay causal `v2` restant |
| 2026-08-17 | `GOV-001` | Codex | Gate G0 | À faire | Validé | `f526a11ff1aab53e39edbfdd7c99f309e0f8d3b4` | `outputs/methodology_baselines/v1-audited-biased` ; manifeste `d8e0273e69bd588a971d1ed2c28438245d262357eb4314eba9649abaa3ec79cf` ; inventaire `ec75cc709b923cdf1292638b569ec66dcab536dbc6bca97e21967afee16227d9` ; 269 tests | Baseline auditable et immuable, explicitement non causale |
| 2026-08-17 | `GOV-002` | Codex | Gate G0 | À faire | Validé | `3f2f8aa235329197b759f4a7d84fcc0e2700adf9` | Baseline comparée jusqu'à 2026-07 : 5 215 positions (`bdc9ddc47b48f32b2a0079c3a97b0eb2f9e7e8ee87795f8cee3ecdf8d3b3be48`) et 720 mois-stratégies (`21e6427aba921d060f3c7354d2215ae34b2107c679f249b81ca705b7b41dd833`) ; écarts maximaux nuls ; suite 273 tests | Extension future autorisée ; tolérance positive plafonnée à `1e-12` et obligatoirement justifiée |
| 2026-08-17 | `GOV-003` | Codex | Gate G0 | À faire | Validé | `2f89e39b519355a51be569aaf118a50f9fc46d31` | 26 tests ciblés puis suite 275/275 ; capture réelle sale : diff `fc317404697bea44904aff78973db68f68196e190166b438783c2d5405e3a85c`, dépendances `507eedd23191425cd08930feb4960ffa4a0909ae70aa54d50f210344c8df6b61`, bundle `80fa8815ff21d6f0550c39349a41b55af7d526dcdd81a986169e796911612b6b` | Gate G0 franchie ; R&D sale attribuable, promotion toujours réservée à un commit propre |
| 2026-08-17 | `BST-003` | Codex | Gate G1 | À faire | Implémenté | `34df93c7f99ff1f3961bc36b1d1b4f9422e38ce1` | Artefact audité reclassé : 84 443 évaluables, 3 008 horizons en attente, 1 497 cibles titre indisponibles ; test ciblé 25/25 ; suite 276/276 | Plus aucun drop générique : census par fold, entraînement fail-closed sur toute cible mature benchmark/titre/terminale non résolue ; replay v2 restant |
| 2026-08-17 | `SIM-004` | Codex | Gate G1 | À faire | Implémenté | `5b96074747c94058693607bdd7b5ef828aaa4804` | 46 tests ciblés ; suite 277/277 ; test signal/exécution/première observation et événement terminal résolu | Le défaut exige cinq timestamps causaux ; tous les replays historiques déclarent explicitement le mode `legacy_month_only` non promouvable ; replay v2 restant |
| 2026-08-17 | `QA-001` | Codex | Gate G1 | À faire | Implémenté | `68a1f557aeb146b5a1f031570c67086ef86d5365` | Mutation séparée cible/prix/membership/secteur/filing ; test d'acceptation 1/1 ; suite 278/278 | Garde transversal et join PIT auditable ajoutés ; validation finale dépend du branchement de toutes les sources UNI/FND et du replay v2 |
| 2026-08-17 | `UNI-001` | Codex | Gate G2 | À faire | Implémenté | `0acf0963ec94e69a234f1f82c08055027f16eecd` | VEEV, MRVL, FLEX, EA et FERG testés juste avant/à l'instant effectif ; 4 tests ciblés ; suite 279/279 | Heure explicite ou minuit New York ; snapshots mensuels rapprochés à la décision de fin de mois ; replay v2 restant |
| 2026-08-17 | `UNI-002` | Codex | Gate G2 | À faire | Implémenté | `1c67a9801a7ccb606a6ede2107bc57fe0772f6e6` | Fixture d'acceptation : 214 groupes résolus et audités ; fichier actif : 1 067 groupes datés, 225 620 sorties, zéro clé dupliquée ; 4 tests ciblés ; suite 280/280 | Nom normalisé le plus fréquent, égalité lexicographique ; aucune déduplication silencieuse ; replay v2 restant |
| 2026-08-17 | `UNI-003` | Codex | Gate G2 | À faire | Implémenté | `3b816e4ca494d27e49c5f0bed23655abaa869ccb` | Registre actif : 10/10 événements, 17 opérations, provenance complète ; test d'acceptation fail-closed ; 5 tests ciblés ; suite 281/281 | Date source sans heure placée à 23:59:59 New York ; audit manifeste/HTML complet ; replay v2 restant |
| 2026-08-17 | `UNI-004` | Codex | Gate G2 | À faire | Implémenté | `3690fa93ba8e6bbce51c528de4e12f8e5ca35ee4` | Changement Technology vers Health Care testé avant/après ; mutation future sans effet passé ; couverture partielle désactive toute la date ; 2 tests ciblés ; suite 282/282 | Aucun fallback vers le secteur courant ; branchement Legacy atomique sous `LEG-001` |
| 2026-08-17 | `FND-001` | Codex | Gate G2 | À faire | Implémenté | `89ac75cce09db0c7f89691b21dd1e469aea196ab` | Mutation des rendements futurs et de la survie sans effet sur l'éligibilité ; rapport annuel contrôlé ; test ciblé 1/1 ; suite 283/283 | Politique versionnée `sec-only-exclude-ex-ante-v1` ; 3 023 ticker-mois et 62 tickers à remesurer au replay v2 |
| 2026-08-17 | `FND-002` | Codex | Gate G2 | À faire | Implémenté | `33f9c47fc5a619ec769d0e670c6fada4b2c319e0` | Aucun TTM sur 1 à 3 trimestres ni avec un trimestre manquant ; sommes revenue 100, FCF 10 et EPS 1 contrôlées ; 2 tests ciblés ; suite 284/284 | Moyennes de bilan soumises à la même fenêtre complète ; nouvelle baseline v2 requise |
| 2026-08-17 | `FND-003` | Codex | Gate G2 | À faire | Implémenté | `c4600c55377491c7853ba63a7ac40ad5b02c9597` | Dix groupes dupliqués résolus, zéro clé restante, inversion complète de l'entrée sans effet ; 30 tests ciblés ; suite 285/285 | Priorité dépôt post-période, délai, date, accession ; lineage des candidats conservé |

## 12. Registre des baselines et publications

| Version | Statut | Méthode | Vintage données | Commit code | Artefact / manifest | Période | Motif de statut |
|---|---|---|---|---|---|---|---|
| `v1-audited-biased` | Validé | Legacy + Boosting audités | `20260816_142812` | `f526a11ff1aab53e39edbfdd7c99f309e0f8d3b4` pour le scellement ; code historique partiellement attribuable | `outputs/methodology_baselines/v1-audited-biased/baseline_manifest.json` (`d8e0273e...`) | Août 2011 à juillet 2026 pour la comparaison commune | Référence immuable de rapprochement, non preuve causale ; sera supersédée après promotion de `v2` |
| `v2-causal` | À faire | Legacy + Boosting corrigés | À sceller | — | — | Même période plus extension éventuelle | Promotion après gates G0 à G6 |

Les métriques de `v1-audited-biased` peuvent être conservées pour réconciliation, mais
ne doivent pas être utilisées comme preuve de supériorité du Boosting après confirmation
de la fuite future. La publication `v2-causal` doit montrer au minimum : rendement brut,
rendement net, CAGR, volatilité, Sharpe, drawdown, turnover, concentration, couverture,
nombre de titres sans rendement, et différences appariées contre Legacy et benchmark.

## 13. Décisions qui nécessitent une validation humaine

Ces décisions ne doivent pas être prises implicitement dans le code :

- fournisseur et règle de valorisation des radiations/acquisitions sans prix terminal ;
- convention canonique d'exécution : prochaine ouverture, VWAP ou autre prix réalisable ;
- délai opérationnel appliqué après publication des filings ;
- source historique des secteurs et traitement des périodes non couvertes ;
- modèle de coûts canonique et scénarios de sensibilité ;
- période scellée de confirmation et règle d'ouverture ;
- seuil de blocage du rapprochement ledger/positions ;
- tolérance numérique autorisée pour une migration déclarée sans effet économique.

Une décision validée doit être ajoutée au contrat méthodologique, couverte par un test et
référencée dans le journal de suivi avec son commit.

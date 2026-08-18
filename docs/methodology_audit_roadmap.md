# Roadmap de remédiation — données, Legacy, Boosting et portefeuille

- Dernière mise à jour : 2026-08-18
- Périmètre : dépôt `alpharank` et dashboard du dépôt frère `../portfolio`
- État : les 41 actions initiales sont traitées (3 validées, 38 implémentées) et les actions additionnelles `RUN-006`/`RUN-007` sont validées provisoirement ; `RUN-003` et `RUN-004` disposent désormais d'un replay complet et audité, tandis que la promotion finale et `RUN-005` restent bloquées par 7 événements terminaux sélectionnés à résoudre manuellement
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
8. **Gate G7 — Exécution causale v2** : `RUN-001` à `RUN-005`, puis revue
   humaine et promotion atomique via `GOV-005`.

Une gate n'est franchie que lorsque toutes ses tâches P0 et P1 sont `Validé`.

### Tableau de progression courant

| Catégorie | Total | P0 | P1 | P2 | Implémenté | Validé | Progression validée |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gouvernance | 5 | 2 | 3 | 0 | 2 | 3 | 60 % |
| Prix et vintages | 3 | 0 | 3 | 0 | 3 | 0 | 0 % |
| Univers et secteurs | 4 | 1 | 3 | 0 | 4 | 0 | 0 % |
| Fondamentaux | 4 | 1 | 1 | 2 | 4 | 0 | 0 % |
| Boosting | 6 | 3 | 3 | 0 | 6 | 0 | 0 % |
| Legacy | 4 | 0 | 3 | 1 | 4 | 0 | 0 % |
| Simulation | 4 | 2 | 1 | 1 | 4 | 0 | 0 % |
| Dashboard et IBKR | 6 | 1 | 4 | 1 | 6 | 0 | 0 % |
| Qualité et documentation | 5 | 2 | 1 | 2 | 5 | 0 | 0 % |
| Exécution causale v2 | 5 | 5 | 0 | 0 | 0 | 0 | 0 % |
| **Total** | **46** | **17** | **22** | **7** | **38** | **3** | **6,5 %** |

Mettre ce tableau à jour dans le commit documentaire de suivi immédiatement après
chaque commit d'action. Le total des criticités doit toujours égaler le total des tâches.

## 6. Roadmap détaillée

### A. Gouvernance, invariance et versionnement

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `GOV-001` | P0 | Figer les artefacts actuels Legacy, Boosting et benchmark en baseline `v1-audited-biased`; inclure entrées, sorties, configuration et rapport. | `test_baseline_package_is_immutable` : toute réécriture d'un fichier scellé échoue ; inventaire et SHA-256 complets. | Validé | `f526a11ff1aab53e39edbfdd7c99f309e0f8d3b4` | Aucun ; 266 fichiers / 297 256 217 octets conservés exactement |
| `GOV-002` | P0 | Ajouter un garde de préfixe économique commun aux migrations sans effet économique. Dépend de `GOV-001`. | `test_economic_prefix_is_bitwise_stable` : mêmes mois, tickers, poids, rendements bruts/nets et turnover ; écart maximal `0` ou tolérance explicitement justifiée pour la sérialisation. | Validé | `3f2f8aa235329197b759f4a7d84fcc0e2700adf9` | Identique sur la baseline : écarts maximaux nuls |
| `GOV-003` | P1 | Étendre les manifests avec commit Git, état dirty, diff hash, dépendances, interpréteur, configuration, code critique et identifiants de données. | `test_manifest_captures_complete_runtime_provenance` : échec si un champ requis est absent ou si `git_dirty` est faux alors que le worktree est sale. | Validé | `2f89e39b519355a51be569aaf118a50f9fc46d31` | Aucun ; manifests Legacy et Boosting enrichis |
| `GOV-004` | P1 | Rendre les répertoires de run uniques et atomiques ; interdire `exist_ok=True` sur un identifiant déjà utilisé. Dépend de `GOV-003`. | `test_run_directory_cannot_be_overwritten` : un second run avec le même ID échoue avant toute écriture. | Implémenté | `56882ad1b025cc81d3239e7f0d8a45f745a92ad5` | Aucun ; réservation atomique branchée sur Legacy, backtest et multihorizon |
| `GOV-005` | P1 | Définir promotion, rollback et supersession : pointeur canonique atomique, ancienne version conservée, motif et approbation enregistrés. | `test_promotion_is_atomic_and_reversible` : interruption simulée sans pointeur partiel ; rollback retrouve tous les hashes précédents. | Implémenté | `b3c8e3741185b507ae83dac266b68fb87cfc7d2b` | Aucun écrasement ; toutes les versions et leurs inventaires SHA-256 restent conservés |

### B. Prix, ajustements et vintages de données

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `PRC-001` | P1 | Recomposer le snapshot canonique avec le contrat de prix persistant récent et son registre de lineage, sans changer les octets de `US_Finalprice`. Dépend de `GOV-002`. | `test_price_registry_promotion_preserves_payload` : SHA-256 du prix, nombre de lignes, clés et séries économiques identiques avant/après. | Implémenté | `c7ed6acd29a26e630ed628fd37e088a843fc65a6` | Identique : octets, lignes, clés et séries contrôlés ; hash du registre inclus dans l'identité de composition |
| `PRC-002` | P1 | Formaliser les révisions de prix, splits, dividendes et corrections fournisseur par vintage et date de connaissance. | `test_price_revision_requires_new_vintage` : une valeur historique modifiée ne peut pas remplacer le vintage canonique sans nouveau package et rapport de diff. | Implémenté | `30904d777eef48abcea662d1f12c34505fe46de5` | Ancien vintage immuable ; nouveau résultat versionné ; package réel à produire pour validation |
| `PRC-003` | P1 | Construire un cache de prix du dashboard figé par date d'ingestion ; interdire qu'un appel réseau modifie implicitement une période historique déjà publiée. | `test_dashboard_history_is_stable_when_provider_changes` : deux réponses fournisseur différentes donnent le même historique pour un vintage scellé. | Implémenté | `9efb46f98c97e617f2e1cc0b178726c46df97c98` | Identique pour un vintage : première réponse scellée avec timestamp, hash et nombre de lignes ; appels ultérieurs rejoués localement |

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
| `FND-004` | P0 | Appliquer partout une disponibilité point-in-time stricte : `filing_date`, heure de publication, délai opérationnel et version du filing. | `test_feature_availability_precedes_decision` : pour chaque valeur de feature, `available_at <= decision_at`; mutations d'un filing futur sans effet sur le passé. | Implémenté | `00d5f1b32c485106cfdd8a04ea8eea486e449bb6` | Nouvelle baseline requise ; contrat prêt, matérialisation/replay v2 de toutes les features encore à produire |

### E. Algorithme Boosting

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `BST-001` | P0 | Classer tous les candidats éligibles avec le score disponible à la décision, sélectionner Top N, puis seulement joindre le rendement réalisé. Ne jamais filtrer sur `future_return_1m` avant sélection. | `test_boosting_selection_ignores_future_return_availability` : rendre nuls les rendements futurs du titre et du benchmark sans changer le Top N. | Implémenté | `36ec9be79a7faa70c1ae4abb93a5bad60f766247` | Nouvelle baseline Boosting obligatoire ; validation finale après replay `v2` |
| `BST-002` | P0 | Résoudre le rendement terminal total actionnaire pour radiations, faillites, acquisitions et changements de ticker. Dépend de `PRC-002`. | `test_terminal_return_is_included` : cas cash merger, échange d'actions, radiation à perte totale et pont de ticker ; aucun survivant implicite. | Implémenté | `a6feee33fd8fffcfa5f2255c5d55a3ddb079773b` | Nouvelle baseline Boosting obligatoire ; validation finale après replay `v2` |
| `BST-003` | P0 | Définir la censure de la cible d'entraînement : distinguer horizon non encore réalisé, donnée manquante et événement terminal ; supprimer toute exclusion corrélée à la survie. Dépend de `BST-002`. | `test_training_target_missingness_is_not_survival_filter` : les 1 497 observations sont classées par motif, sans drop générique ; rapport de censure par fold. | Implémenté | `34df93c7f99ff1f3961bc36b1d1b4f9422e38ce1` | Nouvelle baseline modèle obligatoire ; replay bloqué tant que les 1 497 cibles matures ne sont pas résolues |
| `BST-004` | P1 | Corriger ou déprécier `scripts/run_backtest.py`, dont sélection sparse et médiane sont actuellement calculées sur l'échantillon complet avant les folds. | `test_preprocessing_is_fit_inside_each_outer_fold` : mutation du futur sans effet sur features, colonnes retenues et imputations du passé. | Implémenté | `d09d79ce4a1a84cfcdb4b920587bd47f84ffbfa6` | Nouvelle baseline pour ce chemin R&D ; préprocesseur sérialisé par fold externe |
| `BST-005` | P1 | Sérialiser chaque modèle, préprocesseur, liste de features, seed et métadonnées de fold ; permettre un replay sans réentraînement. Dépend de `GOV-003`. | `test_serialized_model_reproduces_oos_predictions` : prédictions, rangs et portefeuille hors échantillon identiques après rechargement. | Implémenté | `59cc7bca89c81d758db8bd1a3d833198572713e4` | Modèle natif, hash, préprocesseur, features, seed, itérations et bornes du fold conservés ; replay `v2` du portefeuille restant |
| `BST-006` | P1 | Sceller un jeu de confirmation final et enregistrer toutes les variantes testées pour limiter le biais de sélection et le multiple testing. | `test_sealed_period_is_single_use` : toute lecture prématurée ou nouvelle optimisation après ouverture invalide la promotion ; registre des expériences complet. | Implémenté | `bb6b97aa9108e426f13b73147e132f409e212559` | Aucun recalage rétroactif : registre déclaré avant scellement, ouverture unique et promotion invalidée après toute optimisation tardive |

### F. Stratégie Legacy

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `LEG-001` | P1 | Brancher le cap sectoriel sur `UNI-004` et déclarer le comportement lorsque le secteur PIT manque. | `test_legacy_sector_cap_uses_pit_sector` : un reclassement futur ne modifie aucune décision passée ; cap vérifié avant l'ordre. | Implémenté | `e684fc6ea113d9b4f2039953efa5da7d42b234b2` | Nouvelle baseline Legacy : cap avant sélection si couverture PIT complète, sinon désactivation datée et motivée |
| `LEG-002` | P1 | Harmoniser l'éligibilité prix du titre et du benchmark, les ajustements dividendes/splits et le calcul d'excès de rendement. | `test_asset_and_benchmark_return_conventions_match` : même calendrier, même convention de total return, aucune interpolation asymétrique. | Implémenté | `227fbfa9c0693bc434bc10c4c90d7869729b0928` | Nouvelle baseline à mesurer : titre et SPY en `adjusted_close`, dates communes observées uniquement |
| `LEG-003` | P1 | Déclarer l'instant exact du signal et de l'exécution ; comparer clôture, prochaine ouverture et VWAP avec données réellement disponibles. | `test_order_price_occurs_after_signal_cutoff` : timestamp d'exécution strictement postérieur au cutoff ; rapport de sensibilité obligatoire. | Implémenté | `a5ee3b7814bfe1c67f456130f8f62c7a481637c4` | Nouvelle sensibilité obligatoire ; convention canonique `next_session_open_v1`, close non exécutable et VWAP seulement observé |
| `LEG-004` | P2 | Verrouiller le protocole Optuna et les ancres : espace de recherche, seeds, période de calibration, candidats rejetés et règle de choix. | `test_legacy_search_protocol_is_locked` : même manifeste, mêmes trials et même gagnant ; aucune donnée de validation finale utilisée pour choisir. | Implémenté | `a3a7dc16ec1d3bd9fe88ef97ee969e469fffc687` | Aucun retuning rétroactif : protocole `legacy-optuna-search-v1`, trials/candidats/rejets/gagnant persistés |

### G. Simulateur commun de portefeuille

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `SIM-001` | P0 | Remplacer `renormalize_available` comme défaut par un mode `raise` ou par un rendement terminal explicitement résolu. | `test_missing_selected_return_fails_closed_by_default` : un titre sélectionné sans rendement produit une erreur qualifiée ; l'ancien replay demande explicitement `renormalize_available`. | Implémenté | `0cf79a9c5c19f2a83b47508e09b27aa885e402ac` | Nouvelle baseline si des mois réalisés sont touchés ; l'ancienne baseline reste reproductible à `2.08e-16` |
| `SIM-002` | P1 | Calculer le turnover entre les poids dérivés après performance et les nouveaux poids cibles, avec gestion du cash et des entrées/sorties. | `test_turnover_uses_drifted_pretrade_weights` : exemples analytiques à deux actifs et comparaison indépendante. | Implémenté | `8cac9cf96385f737960fe4d0bff417ec2058195d` | Nouvelle baseline nette ; dérive, cash, entrées, sorties et pertes totales couverts par la même formule |
| `SIM-003` | P2 | Ajouter spreads, slippage, impact, minimum de frais, change et scénarios de coûts, séparés des rendements bruts. | `test_cost_model_is_monotonic_and_reconciled` : coût nul reproduit le brut ; coût croissant ne peut améliorer le net ; somme des coûts rapproche le P&L. | Implémenté | `0b28cdca7b4e23d68e5cb924a4dd4f985de13e86` | Séries nettes nouvelles ; brut inchangé et composants de coût exactement rapprochés |
| `SIM-004` | P0 | Imposer la frontière causale globale décision-exécution-rendement et intégrer les événements terminaux résolus par `BST-002`. | `test_holding_return_starts_after_trade` : première observation de rendement après exécution ; aucune donnée de la période détenue dans le signal. | Implémenté | `5b96074747c94058693607bdd7b5ef828aaa4804` | Nouvelle baseline requise ; `v1` demande désormais explicitement `legacy_month_only` |

### H. Ingestion IBKR et performance du dashboard

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `DASH-001` | P1 | Classer explicitement intérêts, dividendes, frais, taxes et transferts ; faire échouer ou mettre en quarantaine les descriptions cash inconnues au lieu de les transformer en dépôt/retrait. | `test_interest_cash_is_not_external_flow` : les trois exemples audités totalisant 41,45 EUR augmentent le rendement et ne sont pas neutralisés dans le TWR. | Implémenté | `c061d18be74fe4910017b0d4025f1b99bc023761` | Nouvelle baseline dashboard : 41,45 EUR d'intérêts deviennent du rendement interne ; inconnu bloqué par erreur typée |
| `DASH-002` | P0 | Rapprocher quotidiennement ledger, corporate actions et `positions_history.parquet`, qui reste la source de vérité ; bloquer la publication si l'écart inexpliqué dépasse la tolérance. | `test_ledger_reconciles_to_position_snapshots` : quantités par compte/ticker/date identiques, avec exceptions FX typées ; splits et transferts couverts. | Implémenté | `25701925d9849506a12185a5c54694c91f49bfd2` | Nouvelle baseline si écart : publication bloquée au-delà de `1e-8`, rapport conservé, exceptions FX et soldes d'ouverture typés |
| `DASH-003` | P1 | Exposer séparément fraîcheur positions, cash, transactions, prix, FX et valorisation ; avertissement visible et statut final/provisoire. | `test_freshness_contract_reports_each_source` : dates exactes, aucune date agrégée trompeuse, seuils de staleness testés. | Implémenté | `ffffe12dbeb97a48ffb25d6e7bf12cac358636ee` | Aucun ; chaque source conserve sa date, son seuil et son état, et toute source absente ou périmée rend le dashboard provisoire |
| `DASH-004` | P1 | Interdire la répétition d'un `mark_price` actuel sur tout l'historique lorsqu'un prix manque ; utiliser prix versionné ou état indisponible. Dépend de `PRC-003`. | `test_current_mark_is_never_backfilled_into_history` : fournisseur indisponible et historique partiel donnent une erreur/rupture documentée, pas une série plate artificielle. | Implémenté | `26826d7023051adc83dd1b41c77fde72862d9397` | Nouvelle baseline possible ; un historique absent reste vide même si une valorisation courante existe |
| `DASH-005` | P2 | Décomposer `backend/app/analytics/engine.py` en modules ingestion, pricing, positions, cashflows, performance, attribution et risque, sans changement économique. Dépend de `GOV-002`. | Tests de caractérisation plus `make test` : mêmes réponses API et hashes économiques sur fixtures avant/après. | Implémenté | `7545bae2d700061d6f424e6bbef06acbbd13e152` | Identique sur la fixture scellée : hash séries + métriques `56ee2d92…435c` conservé |
| `DASH-006` | P1 | Remplacer les `except` larges et fallbacks silencieux par erreurs typées, métriques et lineage de fallback dans l'API. | `test_fallback_is_visible_and_typed` : chaque panne simulée indique source, cause, fallback et dégradation ; aucun `except Exception: pass`. | Implémenté | `1758a2d9b497fe4d6e250db3db1fe0d1448b9f3e` | Aucun attendu ; les échecs prix, FX, cache et XML ne sont plus silencieux et conservent leur fallback explicite |

### I. Tests anti-biais, validation, CI et documentation

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `QA-001` | P0 | Créer une suite de tests sémantiques par mutation du futur : cible, prix futur, membership futur, secteur futur et filing futur. | `test_future_mutations_do_not_change_past_decisions` : scores et ordres antérieurs au cutoff restent identiques pour chaque mutation. | Implémenté | `68a1f557aeb146b5a1f031570c67086ef86d5365` | Aucun ; contrat générique prêt, branchement production UNI/FND et replay v2 restants |
| `QA-002` | P0 | Étendre les validateurs pour recalculer les sorties depuis le package, pas seulement vérifier quelques hashes de fichiers. Inclure tout le moteur commun et les règles d'éligibilité. | `test_replay_recomputes_outputs_from_sealed_inputs` : environnement neuf, sorties identiques ; échec à toute mutation de code, config, entrée ou modèle. | Implémenté | `c750254055a4fd740cf4213df879a8aae787b9cd` | Reproduit exactement la version scellée ; code moteur, éligibilité, configuration, entrée, modèle et sortie attendue sont inventoriés |
| `QA-003` | P1 | Ajouter une matrice CI des deux dépôts : tests unitaires, tests anti-look-ahead, replay court, validation documentation et build frontend. | Pipeline : AlphaRank complet, `make test`, `npm run build`, validateurs de docs et replay smoke tous verts sur commit propre. | Implémenté | `fcecf5ab0bb187e2f5ca9e9f44d0eee24c23ab26` | Aucun ; matrice sur checkouts propres AlphaRank/Portfolio avec référence Portfolio configurable |
| `DOC-001` | P2 | Mettre à jour les sources de vérité après chaque correction : contrat temporel, univers, prix, cible, exécution, coûts, limites et procédure de replay. | `test_documentation_structure.py` et revue croisée code/doc : chaque règle normative pointe vers son test et sa configuration. | Implémenté | `1ed8e66b9a8080536b20920b712ab17eceed6f13` | Aucun ; index normatif central relié aux propriétaires code, configurations et tests |
| `DOC-002` | P2 | Afficher dans les rapports et le dashboard version méthodologique, vintage de données, commit, statut `provisional/final/superseded` et avertissements connus. | `test_report_exposes_methodology_identity` : informations présentes et cohérentes avec le manifeste ; impossible de publier sans identité complète. | Implémenté | `52c93e272d12d8ceac9026bd3f1780af33f6f92e` | Aucun ; identité fail-closed exposée par l'API et bannière globale sur dashboard, documentation et rapports intégrés |

### J. Exécution, replay et rapprochement causal v2

| ID | Criticité | Changement et dépendances | Test associé et critère d'acceptation | Statut | Commit | Effet historique |
|---|---|---|---|---|---|---|
| `RUN-001` | P0 | Construire et sceller le snapshot causal `v2` réel depuis la dernière composition de production admissible : prix versionnés, univers historique, secteurs PIT, fondamentaux SEC disponibles à la décision, hashes et provenance complète. Dépend de `PRC-001`, `PRC-002`, `UNI-001` à `UNI-004` et `FND-001` à `FND-004`. | `test_causal_v2_snapshot_is_sealed_and_complete` : scope de production, sources autorisées, inventaire SHA-256, politiques PIT et registre de prix présents ; toute mutation invalide le package. | Validé | `e98478728a5396ae3f3683c33ba6d9c51479788d` | Aucun rendement calculé ; composition `2975e651…acb40b` byte-identique côté données, enrichie du registre persistant et scellée sans écraser le snapshot précédent |
| `RUN-002` | P0 | Exécuter Legacy `v2` depuis le snapshot `RUN-001` avec secteur PIT, benchmark total return, exécution `next_session_open_v1`, turnover dérivé et scénarios de coûts ; valider le package de replay strict. | `test_legacy_v2_run_is_replayable` : run terminé, manifeste complet, première sélection dérivée recalculable, replay strict et moteur commun verts. | Validé | `d8e29c8`, `ada5ff4`, `32fc41e` | Nouvelle série Legacy produite sans toucher `v1` : 7 776 holdings, 1 188 lignes stratégie/mois/scénario, deux stratégies, trois scénarios de coûts et erreur maximale de replay `2,78e-16` |
| `RUN-003` | P0 | Entraîner et rejouer Boosting `v2` en walk-forward strict depuis le même snapshot : préprocesseur par fold, modèles sérialisés, censure qualifiée et rendements terminaux résolus ou journalisés provisoirement. Dépend de `RUN-001`, `BST-001` à `BST-006` et `RUN-006`. | `test_boosting_v2_replay_is_serialized_and_causal` : folds complets, hashes modèle/préprocesseur, prédictions OOS reproductibles, zéro sélection filtrée par rendement réalisé et rapport exhaustif des cibles matures provisoires. | Validé provisoire — revue manuelle | `7fcb691`, `b17e456`, `73c6d9a`, `ef9fdda` | Run propre `boosting_v2_provisional_clean2` : 16 folds, 88 948 prédictions OOS, zéro cible mature non résolue au sens opérationnel, replay des scores exact ; 656 cibles uniques restent journalisées, dont 595 H6 / 110 tickers et 61 H1 / 61 tickers |
| `RUN-004` | P0 | Recalculer le portefeuille commun Legacy/Boosting/SPY avec le même snapshot, calendrier, exclusions, politique de rendement manquant, turnover, coûts et benchmark. Dépend de `RUN-002` et `RUN-003`. | `test_common_v2_replay_is_comparison_eligible` : hashes d'entrée et exclusions identiques, calendrier commun et rapprochement mensuel à `1e-12` ; tant que le journal terminal n'est pas résolu, `comparison_eligible=false` est obligatoire. | Validé provisoire — non promouvable | `3a6a2ec`, `293e835`, `8bf1680` | Run `common_v2_provisional5` : 6 305 holdings, 178 mois par stratégie, 712 lignes mensuelles et erreur maximale `3,33e-16` ; 193 candidats retirés par appartenance et 1 017 par absence d'ouverture à la première séance, avant Top-N et sans rendement futur |
| `RUN-005` | P0 | Produire le rapprochement économique scellé `v1-audited-biased` versus `v2-causal` : sélections, poids, rendements, turnover, coûts, CAGR, Sharpe et drawdown, avec attribution de chaque rupture aux corrections UNI/FND/BST/LEG/SIM. Dépend de `RUN-004`. | `test_v1_v2_reconciliation_is_complete` : chaque mois et chaque différence économique ont un statut et une cause ; métriques recalculées depuis les séries ; aucun mois divergent sans explication. | Implémenté — attente résolution terminale | `b236f78` | Le rapprochement final reste interdit tant que `RUN-004` porte `comparison_eligible=false` ; l'impact économique provisoire du seul groupe terminal est néanmoins publié séparément par `RUN-007` |
| `RUN-006` | P0 | Continuer un replay diagnostique lorsqu'une série de prix se termine après l'entrée : retenir la performance jusqu'à la dernière cotation, porter ensuite la valeur à rendement nul et journaliser chaque cible/position pour revue manuelle. L'entrée reste interdite sans ouverture à la première séance. | `test_mature_target_can_continue_with_logged_last_observation` et `test_early_final_quote_is_kept_and_flagged_for_manual_review` : date exacte conservée, aucune promotion du survivant suivant, statut `pending_manual_terminal_event_review`, politique fail-closed toujours disponible. | Validé provisoire | `73c6d9a4f70d8931c332a8994b5ec731019fb4a9`, `ef9fdda013fbfb97e386a6d199c58b1975af8386` | Le Boosting peut s'entraîner sans perdre 595 lignes H6 ; ces observations ne deviennent pas des événements terminaux définitifs et restent non promouvables jusqu'à revue sourcée |
| `RUN-007` | P0 | Attribuer séparément la part de performance des positions `provisional_last_observation` avec la décomposition canonique en log annualisé, le pont composé et l'impact CAGR marginal du groupe. Dépend de `RUN-004` et `RUN-006`. | `test_provisional_return_cagr_impact_is_reconciled_in_log_space` : somme log exacte, CAGR recomposé, journal positions et rapport lisible ; aucun point de CAGR présenté comme additif. | Validé provisoire | `fed216c3ec22b00eb38aaccdb55265571644b69f`, `a7238e86fd735ecb557e6e04eea9ddfe31c18609` | Sur 9 positions / 7 tickers : impact marginal provisoire de `-1,5678` pt pour Boosting Top 5, `-0,7985` pt pour Top 10 et `-0,1192` pt pour Legacy |

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
| 2026-08-17 | `FND-004` | Codex | Gate G2 | À faire | Implémenté | `00d5f1b32c485106cfdd8a04ea8eea486e449bb6` | Deux versions de filing avant/après décision ; `available_at <= decision_at` vérifié ; mutation future sans effet passé ; 2 tests ciblés ; suite 286/286 | Politique `sec-filing-availability-v1` : acceptance SEC sinon 23:59:59 New York, puis délai de 24 h ; replay v2 restant |
| 2026-08-17 | `GOV-004` | Codex | Gate G4 | À faire | Implémenté | `56882ad1b025cc81d3239e7f0d8a45f745a92ad5` | Collision simulée avant écriture, premier contenu intact ; 3 tests ciblés ; suite 287/287 | `exist_ok=False` centralisé pour les runners Legacy, backtest et multihorizon |
| 2026-08-17 | `GOV-005` | Codex | Gate G4 | À faire | Implémenté | `b3c8e3741185b507ae83dac266b68fb87cfc7d2b` | Interruption avant swap : pointeur bit-à-bit intact ; promotion v2 puis rollback v1 avec inventaire SHA-256 identique ; 2 tests ciblés ; suite 288/288 | Approbateur, motif, supersession et journal d'actions conservés ; aucune version supprimée |
| 2026-08-17 | `SIM-002` | Codex | Gate G3 | À faire | Implémenté | `8cac9cf96385f737960fe4d0bff417ec2058195d` | Dérive analytique 50/50 vers 2/3-1/3, rebalance 1/6 ; sortie/entrée 1/2 ; cash résiduel contrôlé ; 20 tests ciblés ; suite 289/289 | Premier mois depuis 100 % cash ; nouvelle baseline nette et replay v2 requis |
| 2026-08-17 | `SIM-003` | Codex | Gate G3 | À faire | Implémenté | `0b28cdca7b4e23d68e5cb924a4dd4f985de13e86` | Scénarios nul/faible/élevé ; spread, slippage, impact, commission/minimum et FX rapprochés ; 21 tests ciblés ; suite 290/290 | Coût croissant réduit le net, brut strictement invariant ; scénarios réels à publier avec v2 |
| 2026-08-17 | `BST-004` | Codex | Gate G4 | À faire | Implémenté | `d09d79ce4a1a84cfcdb4b920587bd47f84ffbfa6` | Colonne sparse exclue et médiane 2 apprises sur train uniquement ; mutation future sans effet sur le passé ; 2 tests ciblés ; suite 291/291 | Chaque `fold_XX/preprocessor.json` conserve features, exclusions et médianes |
| 2026-08-17 | `BST-005` | Codex | Gate G4 | À faire | Implémenté | `59cc7bca89c81d758db8bd1a3d833198572713e4` | Modèle XGBoost rechargé après vérification SHA-256 ; scores et rangs hors échantillon strictement identiques ; 2 tests ciblés ; suite 292/292 | Chaque fold conserve `model.ubj` et un manifeste complet ; replay portefeuille `v2` restant |
| 2026-08-17 | `BST-006` | Codex | Gate G6 | À faire | Implémenté | `bb6b97aa9108e426f13b73147e132f409e212559` | Ouverture prématurée et optimisation post-ouverture invalident la promotion ; registre exhaustif et dataset SHA-256 contrôlés ; 2 tests ciblés ; suite 293/293 | Période réelle et identifiants des expériences `v2` à sceller avant exécution |
| 2026-08-17 | `PRC-001` | Codex | Gate G2 | À faire | Implémenté | `c7ed6acd29a26e630ed628fd37e088a843fc65a6` | Payload source/promu identique octet par octet ; lignes, clés et hash canonique des séries contrôlés ; 5 tests ciblés ; suite 294/294 | Le registre persistant et son hash participent à l'identité du snapshot composé |
| 2026-08-18 | `PRC-003` | Codex | Gate G5 | À faire | Implémenté | `9efb46f98c97e617f2e1cc0b178726c46df97c98` | Deux réponses fournisseur divergentes rejouent le premier vintage ; 34 tests ciblés ; suite Portfolio 54/54 ; validation docs verte | Cache sous `data/cache/price_vintages`, manifeste atomique par fournisseur/symboles/période |
| 2026-08-18 | `LEG-001` | Codex | Gate G2 | À faire | Implémenté | `e684fc6ea113d9b4f2039953efa5da7d42b234b2` | Cap 1 sélectionne un titre par secteur connu ; mutation future sans effet passé ; table statique désactive le cap ; 3 tests ciblés ; suite 295/295 | Les sorties exposent état, raison, classification et timestamp sectoriels |
| 2026-08-18 | `LEG-002` | Codex | Gate G3 | À faire | Implémenté | `227fbfa9c0693bc434bc10c4c90d7869729b0928` | Dates communes strictes, `adjusted_close` des deux côtés et absence de forward-fill vérifiés ; 5 tests ciblés ; suite 296/296 | L'ancien index Legacy en `close` est remplacé par SPY total return ; replay `v2` requis |
| 2026-08-18 | `LEG-003` | Codex | Gate G3 | À faire | Implémenté | `a5ee3b7814bfe1c67f456130f8f62c7a481637c4` | Ouverture canonique strictement après cutoff ; close de référence non exécutable ; VWAP observé ; 17 tests ciblés ; suite 297/297 | Chaque run écrit la table de sensibilité et le manifeste `next_session_open_v1` |
| 2026-08-18 | `LEG-004` | Codex | Gate G6 | À faire | Implémenté | `a3a7dc16ec1d3bd9fe88ef97ee969e469fffc687` | Manifeste déterministe, trials triés, même gagnant et rejets motivés ; confirmation finale exclue ; 19 tests ciblés ; suite 298/298 | Smoke non conforme autorisé mais marqué non promouvable ; production verrouillée à 30 trials et `n_jobs=1` |
| 2026-08-18 | `DASH-001` | Codex | Gate G5 | À faire | Implémenté | `c061d18be74fe4910017b0d4025f1b99bc023761` | Trois intérêts = 41,45 EUR internes, zéro flux externe ; description inconnue bloquée ; 40 tests ciblés ; suite Portfolio 56/56 | Taxes prioritaires sur dividendes ; frais, intérêts et transferts ont des actions explicites |
| 2026-08-18 | `DASH-002` | Codex | Gate G5 | À faire | Implémenté | `25701925d9849506a12185a5c54694c91f49bfd2` | Ledger/snapshots identiques sur achats, split et transfert ; écart de 1 titre bloqué ; FX typé ; 8 tests ciblés ; suite Portfolio 57/57 | `positions_history.parquet` reste autoritaire ; rapport écrit avant toute publication courante |
| 2026-08-18 | `DASH-003` | Codex | Gate G5 | À faire | Implémenté | `ffffe12dbeb97a48ffb25d6e7bf12cac358636ee` | Six dates exactes et seuils indépendants contrôlés ; cash et valorisation périmés rendent le statut provisoire ; 35 tests ciblés ; suite Portfolio 58/58 ; build frontend et validation docs réussis | Endpoint et bannière exposent positions, cash, transactions, prix, FX et valorisation sans date agrégée |
| 2026-08-18 | `DASH-004` | Codex | Gate G5 | À faire | Implémenté | `26826d7023051adc83dd1b41c77fde72862d9397` | Fournisseurs local et distants indisponibles avec `mark_price` courant : historique vide ; 2 tests ciblés ; suite Portfolio 59/59 ; validation docs réussie | La valorisation courante reste autorisée pour l'instantané, jamais comme substitut d'une série historique |
| 2026-08-18 | `DASH-005` | Codex | Gate G5 | À faire | Implémenté | `7545bae2d700061d6f424e6bbef06acbbd13e152` | Sept modules extraits ; quatre tests ciblés ; empreinte économique `56ee2d92045c3ce7c91607f8d36756fe0f0308dc63791e2d389df0a64b04435c` ; suite Portfolio 60/60 ; validation docs réussie | Façade et réponses API inchangées ; calculs purs désormais testables indépendamment |
| 2026-08-18 | `DASH-006` | Codex | Gate G5 | À faire | Implémenté | `1758a2d9b497fe4d6e250db3db1fe0d1448b9f3e` | Panne `OSError` simulée : source, type, cause et fallback exacts ; 10 tests ciblés ; suite Portfolio 61/61 ; aucun `except Exception` restant sous `backend/app` ; validation docs réussie | Registre borné exposé par `/portfolio/fallbacks` avec statut nominal/dégradé et compteur |
| 2026-08-18 | `QA-002` | Codex | Gate G4 | À faire | Implémenté | `c750254055a4fd740cf4213df879a8aae787b9cd` | Replay propre recalculé exactement ; mutations code, config, entrée et modèle toutes rejetées ; test ciblé 1/1 ; suite AlphaRank 299/299 ; aide du validateur commun et validation docs réussies | Package autonome avec inventaire SHA-256, seal détaché, moteur commun complet et règles d'éligibilité ; replay causal v2 réel restant à produire |
| 2026-08-18 | `QA-003` | Codex | Gate G5 | À faire | Implémenté | `fcecf5ab0bb187e2f5ca9e9f44d0eee24c23ab26` | Workflow YAML contrôlé ; AlphaRank 300/300, Portfolio 61/61, liens Markdown suivis des deux dépôts et build frontend réussis | Matrice `alpharank/portfolio`, smoke anti-look-ahead et replay ; secret `CROSS_REPO_TOKEN` requis si Portfolio est privé |
| 2026-08-18 | `DOC-001` | Codex | Gate G6 | À faire | Implémenté | `1ed8e66b9a8080536b20920b712ab17eceed6f13` | Huit domaines normatifs reliés au code, aux politiques et aux tests ; 2 tests ciblés ; liens Markdown et validation documentaire réussis | `docs/research_governance.md` distingue explicitement implémentation, validation économique et promotion v2 |
| 2026-08-18 | `DOC-002` | Codex | Gate G6 | À faire | Implémenté | `52c93e272d12d8ceac9026bd3f1780af33f6f92e` | Identité complète acceptée, manifeste incomplet ou absent refusé ; 2 tests ciblés ; suite Portfolio 62/62 ; build frontend et validation docs réussis | Version `v2-causal-implementation`, vintage runtime, commit normatif, statut provisoire et trois avertissements affichés partout |
| 2026-08-18 | `RUN-001` | Codex | Gate G7 | À faire | Validé | `e98478728a5396ae3f3683c33ba6d9c51479788d` | Prix v2 : 3 709 695 lignes, 840 tickers, zéro clé retirée et hash prix `10777e4e…efd0` inchangé ; registre `ebc72122…45ec6` ; snapshot causal `v2-causal-2975e65156b0-20260818T085346Z`, manifeste `e55b56c2…f0601`, inventaire `6ed5c70d…26b54`, 29 fichiers ; 13 tests ciblés | Fondamentaux strictement SEC-only ; cap secteur désactivé faute d'historique PIT complet ; aucune performance calculée |
| 2026-08-18 | `RUN-002` | Codex | Gate G7 | À faire | Validé | `d8e29c8`, `ada5ff4`, `32fc41e` | Run `20260818_115710` terminé sur worktree propre ; 7 776 holdings, 1 188 lignes mensuelles, 2 stratégies, 3 coûts ; erreur replay `2,78e-16` ; validateur v2 et replay package `--strict-code` verts | Garde d'appartenance au mois détenu avant Top-N ; dates de prix texte normalisées ; les deux exécutions fail-closed antérieures restent conservées comme preuve |
| 2026-08-18 | `RUN-003` | Codex | Gate G7 | À faire | Bloqué — données terminales | `7fcb691`, `b17e456` | Test sérialisation causal vert ; exécution propre arrêtée avant entraînement. Correction du biais d'univers futur : fold 1 de 179 à 48 cibles manquantes ; panel complet encore à 595 lignes H6 / 110 tickers sans terminal sourcé | 106 tickers ont un `DelistedDate` EODHD mais la considération finale n'est pas scellée ; `ANSS`/`APC` demandent une identité historique, `BK`/`CTRA` exposent aussi une fin de prix prématurée ; aucune imputation autorisée |
| 2026-08-18 | `RUN-004` | Codex | Gate G7 | À faire | Implémenté — bloqué | `3a6a2ec` | `test_common_v2_replay_is_comparison_eligible` vert ; builder et validateur communs fail-closed prêts | Exécution interdite sans artefacts Boosting `RUN-003` valides |
| 2026-08-18 | `RUN-005` | Codex | Gate G7 | À faire | Implémenté — bloqué | `b236f78` | `test_v1_v2_reconciliation_is_complete` vert ; scellement, recalcul métrique et taxonomie exhaustive implémentés | Exécution interdite sans replay commun `RUN-004` ; aucune conclusion économique provisoire fabriquée |
| 2026-08-18 | `RUN-006` | Codex | Gate G7 | À faire | Validé provisoire | `73c6d9a`, `ef9fdda` | Run propre Boosting : 16 folds, 88 948 lignes OOS, 656 cibles journalisées, zéro cible mature bloquante, erreur de replay modèle `0` ; tests ciblés verts | Dernière observation portée à cash plat jusqu'à l'horizon ; 595 H6 / 110 tickers et 61 H1 / 61 tickers restent en revue manuelle, sans promotion finale |
| 2026-08-18 | `RUN-004` | Codex | Gate G7 | Implémenté — bloqué | Validé provisoire | `293e835`, `8bf1680` | Replay commun `common_v2_provisional5` : 6 305 holdings, 178 mois, 4 stratégies, erreur `3,33e-16` ; 193 lignes retirées par appartenance et 1 017 par ouverture d'exécution avant classement | `comparison_eligible=false` tant que 9 positions / 7 tickers du journal ne sont pas sourcées ; le validateur historique distinct conserve la parité Legacy (`2,08e-16`) mais signale sur son ancien artefact Alpha un écart turnover `0,131786` / net `0,000131786`, à réconcilier séparément |
| 2026-08-18 | `RUN-007` | Codex | Gate G7 | À faire | Validé provisoire | `fed216c`, `a7238e8` | Rapport `provisional_terminal_impact_report.md` réconcilié : Top 5 `14,4547 %` dont impact marginal provisoire `-1,5678` pt ; Top 10 `18,2535 %`, `-0,7985` pt ; Legacy `18,8508 %`, `-0,1192` pt | Contributions log annualisées additives et pont composé explicite ; 313 tests passent, seul le contrôle documentaire préexistant `data/production/` échoue |

## 12. Registre des baselines et publications

| Version | Statut | Méthode | Vintage données | Commit code | Artefact / manifest | Période | Motif de statut |
|---|---|---|---|---|---|---|---|
| `v1-audited-biased` | Validé | Legacy + Boosting audités | `20260816_142812` | `f526a11ff1aab53e39edbfdd7c99f309e0f8d3b4` pour le scellement ; code historique partiellement attribuable | `outputs/methodology_baselines/v1-audited-biased/baseline_manifest.json` (`d8e0273e...`) | Août 2011 à juillet 2026 pour la comparaison commune | Référence immuable de rapprochement, non preuve causale ; sera supersédée après promotion de `v2` |
| `v2-causal-provisional` | Rejoué — non promouvable | Legacy + Boosting corrigés, fins de cotation provisoires | `v2-causal-2975e65156b0-20260818T085346Z` | Boosting `fed216c` ; exécution commune `8bf1680` ; attribution `a7238e8` | Boosting `outputs/methodology_v2/run_003/boosting_v2_provisional_clean2/manifest.json` ; commun `outputs/methodology_v2/run_004/common_v2_provisional5/manifest.json` | Août 2011 à juillet 2026, 178 mois communs | Replay complet et réconcilié, mais `comparison_eligible=false` jusqu'à résolution manuelle des 9 positions / 7 tickers ; les 656 cibles d'apprentissage provisoires restent journalisées |

Les métriques de `v1-audited-biased` peuvent être conservées pour réconciliation, mais
ne doivent pas être utilisées comme preuve de supériorité du Boosting après confirmation
de la fuite future. La publication `v2-causal` doit montrer au minimum : rendement brut,
rendement net, CAGR, volatilité, Sharpe, drawdown, turnover, concentration, couverture,
nombre de titres sans rendement, et différences appariées contre Legacy et benchmark.

## 13. Décisions humaines approuvées

Les arbitrages suivants ont été approuvés et sont désormais matérialisés dans le
contrat méthodologique et les tests :

- rendement terminal : échec fermé pour toute promotion ; un replay diagnostique peut
  continuer avec la dernière cotation observée, portée ensuite à cash plat, seulement
  si chaque occurrence est journalisée et marquée pour revue manuelle ;
- exécution canonique : prochaine ouverture observée strictement après le cutoff ;
  VWAP seulement observé comme scénario de sensibilité ;
- disponibilité des filings : timestamp SEC, ou fin de journée New York en repli,
  puis délai opérationnel de 24 heures ;
- secteurs : classification point-in-time et désactivation du cap pour tout mois à
  couverture incomplète ;
- coûts : scénario nommé et décomposé, paramètres et sensibilités scellés dans le run ;
- confirmation : période et registre d'expériences scellés avant une ouverture unique ;
- rapprochement ledger/positions : publication bloquée au-delà de `1e-8` ;
- migration sans effet économique : tolérance numérique maximale `1e-12`, et
  identité SHA-256 pour les fichiers seulement transportés.

Ces décisions sont implémentées. Leur statut ne devient `Validé` qu'avec le replay
causal `v2`, son rapprochement économique et sa promotion atomique.

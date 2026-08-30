# Roadmap maître AlphaRank

**Dernière mise à jour : 2026-08-30.**

**Statut : seule source des priorités actives.**

Ce fichier est la seule roadmap active du dépôt. Il ordonne les travaux de
méthodologie, qualité, code, données, documentation et exploitation.

## 1. Pourquoi un second fichier méthodologique existe

[`METHODOLOGY_AUDIT_ROADMAP.md`](METHODOLOGY_AUDIT_ROADMAP.md) n'est plus une
seconde liste de priorités. C'est un **registre détaillé et append-only** qui
conserve les 41 actions d'audit initiales, leurs preuves, commits, résultats et
impacts économiques, plus le détail des incidents LIVE.

Règle :

- `ROADMAP.md` répond à « que fait-on maintenant et dans quel ordre ? » ;
- `METHODOLOGY_AUDIT_ROADMAP.md` répond à « qu'a-t-on décidé, implémenté et
  observé dans l'audit méthodologique ? » ;
- toute action méthodologique encore ouverte est représentée ici par un
  `TASK-ID` maître et renvoie vers son identifiant détaillé ;
- aucun nouveau travail ne démarre uniquement parce qu'une ligne ancienne du
  registre détaillé porte encore un statut historique.

Le registre n'est pas fusionné dans ce fichier parce que ses centaines de lignes
de preuves rendraient à nouveau l'onboarding illisible. Il n'est pas supprimé,
car il constitue l'historique d'audit demandé.

Son nom de fichier conserve encore le mot `ROADMAP` pour ne pas casser les liens
externes et les références historiques. `DOC-010` a découplé les tests du chemin
précédent sous `docs/` puis supprimé ce pointeur de compatibilité ; le registre
racine est désormais l'unique fichier actif pour ce contenu.

## 2. Priorités actives

| Ordre | TASK-ID maître | Travail | Détail lié | Statut |
| ---: | --- | --- | --- | --- |
| 1 | `GIT-001` | publier chaque commit directement depuis `main` et conserver `main-save` | lot GIT ci-dessous | fait |
| 2 | `DOC-014` | consolider `AGENTS.md` comme source unique et réduire `AGENT.md` à un pointeur | — | fait |
| 3 | `DOC-015` | imposer le contrat Git une tâche = un commit documenté | — | fait |
| 4 | `DOC-009` | découper le catalogue Boosting de 3 899 lignes | lot DOC ci-dessous | fait |
| 5 | `DOC-010` | découpler les tests des chemins documentaires historiques | lot DOC ci-dessous | fait |
| 6 | `QUAL-002` | configurer Ruff sans reformatage global | lot QUAL ci-dessous | fait |
| 7 | `METH-001` | matérialiser la clôture comme convention runtime canonique | `LEG-005` | fait |
| 8 | `METH-002` | séparer les deux identités SNDK et reconstruire les résultats | `LIVE-022` | fait |
| 9 | `CODEORG-001` | imposer un plafond de modules Python et inventorier les violations | lot CODEORG ci-dessous | fait |
| 10 | `CODEORG-002` | ranger les responsabilités data sans changer les API économiques | lot CODEORG ci-dessous | fait |
| 11 | `CODEORG-003` | classer les scripts par responsabilité et conserver les commandes publiques | lot CODEORG ci-dessous | fait |
| 12 | `CODEORG-006` | restaurer le mode exécutable perdu pendant le déplacement d'une commande | lot CODEORG ci-dessous | fait |
| 13 | `CODEORG-004` | classer les tests unitaires et d'intégration par domaine | lot CODEORG ci-dessous | fait |
| 14 | `CODEORG-005` | activer le plafond en CI après zéro violation | lot CODEORG ci-dessous | fait |
| 15 | `DOC-018` | ouvrir le lot de qualité résiduelle mesuré le 24 août | lots ci-dessous | fait |
| 16 | `METH-003` | compléter les métriques communes sans changer les rendements | lot METH ci-dessous | fait |
| 17 | `DATA-011` | reconstruire la couverture SEC historique sans promotion implicite | lot DATA ci-dessous | fait |
| 18 | `CODE-012` | extraire la construction du replay commun hors du script public | lot CODE ci-dessous | fait |
| 19 | `METH-004` | mesurer Boosting sur l'univers de valorisation Legacy causal | lot METH ci-dessous | fait |
| 20 | `CODEORG-007` | retirer le dashboard applicatif du dépôt AlphaRank | lot CODEORG ci-dessous | fait |
| 21 | `QUAL-011` | supprimer les erreurs statiques pouvant casser au runtime | lot QUAL ci-dessous | fait |
| 22 | `QUAL-012` | rendre toute la suite autonome dans un checkout propre | lot QUAL ci-dessous | fait |
| 23 | `QUAL-013` | bloquer toute nouvelle dette de taille ou de complexité | lot QUAL ci-dessous | fait |
| 24 | `QUAL-014` | étendre le typage strict à un package métier | lot QUAL ci-dessous | fait |
| 25 | `DOC-019` | rafraîchir les preuves chiffrées de la roadmap | lot DOC ci-dessous | fait |
| 26 | `DOC-020` | rendre le replay après refresh obligatoire et canonique | lot DOC ci-dessous | fait |
| 27 | `REPLAY-001` | attribuer tout drift data jusqu'aux deux portefeuilles | lot REPLAY ci-dessous | fait |
| 28 | `DATA-012` | exécuter le refresh complet et les deux replays scellés | lot DATA ci-dessous | fait |
| 29 | `REPLAY-002` | détailler les écarts de provenance sans faux drift de chemin | lot REPLAY ci-dessous | fait |
| 30 | `REPLAY-003` | rendre le statut de chaque source explicite après un arrêt amont | lot REPLAY ci-dessous | fait |
| 31 | `GIT-002` | versionner et publier la preuve de chaque run important | lot GIT ci-dessous | fait |
| 32 | `DATA-013` | publier la preuve du refresh et des replays du 25 août | lot DATA ci-dessous | fait |
| 33 | `DATA-014` | terminer les acquisitions avant la décision de publication prix | lot DATA ci-dessous | fait |
| 34 | `DATA-015` | conserver l'historique prix validé et n'ajouter que les nouveaux rendements | lot DATA ci-dessous | fait |
| 35 | `REPLAY-004` | rendre le drift d'un refresh lisible dans un rapport HTML causal | lot REPLAY ci-dessous | fait |
| 36 | `DATA-027` | rendre chaque téléchargement SEC explorable par entreprise et trimestre | lot DATA ci-dessous | fait |
| 37 | `METH-005` | rendre la désactivation des fondamentaux Legacy explicite et reproductible | lot METH ci-dessous | fait |
| 38 | `DATA-028` | bloquer les entrées post-fusion révélées par l'univers sans SEC | lot DATA ci-dessous | fait |
| 39 | `REPLAY-005` | promouvoir la politique sans SEC après replay commun strict | lot REPLAY ci-dessous | fait |
| 40 | `METH-006` | filtrer causalement les candidats Boosting par tendance avant classement | lot METH ci-dessous | fait |
| 41 | `METH-007` | rejouer et publier la variante Boosting filtrée par tendance | lot METH ci-dessous | fait |
| 42 | `DOC-021` | publier dans le site le guide complet des méthodes et de leurs pseudo-codes | lot DOC ci-dessous | fait |
| 43 | `DATA-029` | relier les prix SATS et ECHO sans valeur manuelle ni réécriture | lot DATA ci-dessous | fait |
| 44 | `REPLAY-006` | reconstruire le snapshot et rejouer les méthodes après correction SATS | lot REPLAY ci-dessous | fait |
| 45 | `METH-008` | construire une poche Boosting excluant causalement les titres Legacy | lot METH ci-dessous | à faire |
| 46 | `METH-009` | rejouer le portefeuille combiné comme alternative de diversification | lot METH ci-dessous | à faire |
| 47 | `QUAL-015` | enregistrer le schéma strict de la politique de transition ticker | lot QUAL ci-dessous | fait |
| 48 | `DOC-022` | rafraîchir l'inventaire data après les derniers lecteurs et runs | lot DOC ci-dessous | fait |
| 49 | `REPORT-001` | imposer un rapport de backtest interactif commun à toutes les méthodes | lot REPORT ci-dessous | fait |
| 50 | `REPORT-002` | générer et publier le rapport du replay SATS/ECHO dans le site | lot REPORT ci-dessous | fait |
| 51 | `REPORT-003` | comparer toutes les stratégies et borner les model cards par la fenêtre | lot REPORT ci-dessous | fait |
| 52 | `REPORT-004` | régénérer et republier le rapport comparatif SATS/ECHO | lot REPORT ci-dessous | fait |
| 53 | `REPORT-005` | faire piloter toutes les vues par les courbes affichées | lot REPORT ci-dessous | fait |
| 54 | `REPORT-006` | ajouter un laboratoire de portefeuille multi-stratégie | lot REPORT ci-dessous | fait |
| 55 | `REPORT-007` | republier le rapport corrigé et son laboratoire | lot REPORT ci-dessous | fait |
| 56 | `REPORT-008` | mesurer les corrélations et la richesse relative du portefeuille composé | lot REPORT ci-dessous | fait |
| 57 | `REPORT-009` | republier le rapport enrichi de diversification | lot REPORT ci-dessous | fait |
| 58 | `REPORT-010` | séparer le portefeuille en vigueur du dernier mois de performance réalisé | lot REPORT ci-dessous | fait |
| 59 | `REPORT-011` | republier le rapport avec le portefeuille en vigueur au 28 août | lot REPORT ci-dessous | fait |

Une tâche `prêt à committer` est implémentée dans le worktree mais n'est pas
`faite` tant que son unique commit n'existe pas.

Transition achevée : les travaux `DOC-001` à `DOC-013` et `QUAL-001`, commencés
avant l'activation du contrat Git, ont été redécoupés puis committés sous leurs
identifiants propres. Aucun commit global de nettoyage ne les a absorbés.

Le lot de fiabilité ouvert le 24 août 2026 est exécuté : le diagnostic
`REPLAY-001` a été appliqué au refresh intégral `DATA-012`, puis les faux drifts
de provenance et les statuts de sources ont été durcis par `REPLAY-002/003`.
La fenêtre d'observation `DATA-010` reste inchangée jusqu'au 19 septembre 2026 ;
aucune archive physique ne sera déclenchée avant sa revue.

## 3. Règles de conduite

- Aucun résultat historique, fichier de données ou preuve de run n'est supprimé
  pendant le rangement.
- Une tâche correspond à un commit et le message de commit contient son
  identifiant.
- Le contrat complet de découpage, message, preuves et staging est
  [`docs/standards/git.md`](docs/standards/git.md).
- Une tâche ne passe à `fait` que dans son commit ; si elle est trop grosse pour
  un commit atomique, elle est divisée ici avant modification.
- Un déplacement est précédé d'un inventaire des lecteurs, commandes, liens et
  hashes concernés.
- Une modification d'organisation ne change pas en même temps la méthodologie
  ou les résultats économiques.
- Les migrations de données sont d'abord simulées, puis comparées par clés,
  valeurs et hashes avant promotion.
- Les dossiers actuels restent lisibles tant qu'un remplacement n'est pas
  démontré et documenté.

## 4. État des lieux au 20 août 2026

| Zone | Constat mesuré | Conséquence |
| --- | --- | --- |
| Bibliothèque | 134 fichiers Python ; 9 dépassent 1 000 lignes | responsabilités difficiles à isoler et à tester |
| Scripts | 114 fichiers Python ; 35 directement à la racine de `scripts/` ; 6 dépassent 1 000 lignes | les points d'entrée et la logique métier se confondent |
| Plus gros fichier | `src/alpharank/data/ingestion/orchestration.py`, 3 608 lignes | risque élevé pour toute modification d'ingestion |
| Imports | 35 fichiers modifient `sys.path` | le comportement dépend du dossier depuis lequel la commande est lancée |
| Gestion d'erreur | 4 `except:` nus et 104 captures générales de `Exception` dans `src/` et `scripts/` | des échecs peuvent être masqués ou mal expliqués |
| Sorties console | 580 appels à `print()` hors tests | journaux hétérogènes et difficiles à relier à un run |
| Tests | 93 fichiers `test_*.py` directement à la racine de `tests/` | aucun découpage visible par domaine ou niveau |
| Modules transverses | huit modules Python directement sous `src/alpharank/`, dont plusieurs suffixés `_v2` | propriété et durée de vie peu évidentes |
| Configurations | 16 fichiers JSON sans schéma commun déclaré | une faute de clé peut être découverte tardivement |
| Documentation | 33 fichiers Markdown sous `docs/` ; un catalogue de 3 899 lignes et un handoff de 1 107 lignes | état courant, journal et archives sont mélangés |
| README racine | 552 lignes avant cette passe | impossible d'obtenir une vue d'ensemble en cinq minutes |
| Données | 34 Go et trois générations de rangement concurrentes | impossible de savoir visuellement quelle branche est canonique |
| Résultats | 31 Go et 346 dossiers directement sous `outputs/` | recherche d'un run lente et conventions de nommage incohérentes |
| Qualité | Ruff et mypy sont déclarés en dépendances optionnelles mais non configurés | les règles ne sont ni partagées ni contrôlées |
| Dépendances | dépendances répétées dans `pyproject.toml`, `requirements.txt` et `environment.yml` | risque de dérive entre installations |
| Intégration continue | le seul workflow AlphaRank lance aussi une matrice dépendante du dépôt Portfolio | validation AlphaRank et contrôle inter-projets sont mélangés |

Ces nombres sont un diagnostic initial, pas un jugement fichier par fichier. Un
gros fichier n'est pas supprimé ou découpé uniquement parce qu'il est gros.

## 4 bis. État courant mesuré au 24 août 2026

| Contrôle canonique | État courant |
| --- | --- |
| Graphe Python maintenu | 194 modules de bibliothèque, 159 scripts, 353 fichiers de code et 152 points d'entrée actifs |
| Dossiers Python | 481 fichiers suivis dans 67 dossiers ; zéro dossier au-dessus du plafond de 20 et zéro dérogation |
| Tests | 125 fichiers, 479 cas collectés et exécutés, zéro échec ; la suite ne dépend plus d'un `outputs/` local |
| Configurations | 17 familles, 22 fichiers JSON et zéro erreur de schéma |
| Ruff | 276 alertes historiques restantes ; zéro `F821`, `F403` ou `F405`, et aucune nouvelle régression autorisée |
| Taille et complexité | 474 fichiers maintenus mesurés ; 349 dépassements historiques : 71 modules, 210 fonctions et 68 complexités, tous bloqués contre l'aggravation |
| Mypy strict | 32 modules couverts dans `alpharank.quality` et tout `alpharank.portfolio`, zéro erreur |
| Données | 35 emplacements, 279 arêtes statiques lecteur/emplacement et 162 arêtes de migration classées, zéro non classée |
| Interface | zéro dashboard applicatif AlphaRank ; Portfolio possède l'interface interactive, AlphaRank conserve seulement les rapports d'audit statiques |

Ces nombres proviennent des inventaires versionnés sous `docs/architecture/`,
de la baseline de taille sous `configs/quality/` et des commandes de validation,
pas d'une recopie de l'état des lieux initial.

## 5. Cible lisible

```text
racine       onboarding, normes et roadmaps
docs/        documentation courante, recherche puis archives séparées
src/         logique Python réutilisable
scripts/     commandes minces regroupées par usage
tests/       tests classés par niveau et domaine
configs/     décisions versionnées
data/        raw -> stg -> def -> mart, puis snapshots immuables
outputs/     runs indexés par famille et identifiant
logs/        journaux reliés aux identifiants de run
```

La carte détaillée se trouve dans
[`docs/architecture/repository_map.md`](docs/architecture/repository_map.md).

## 6. Lot DOC — rendre le dépôt compréhensible

| ID | Action | Statut | Preuve attendue |
| --- | --- | --- | --- |
| `DOC-001` | inventorier tout le dépôt et mesurer les zones de confusion | fait | état des lieux ci-dessus, relevé le 2026-08-20 |
| `DOC-002` | écrire un point d'entrée unique pour les normes de développement | fait | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| `DOC-003` | réduire le README racine à un onboarding de cinq minutes | fait | parcours court et ancien README archivé |
| `DOC-004` | placer les rapports datés, le handoff et les anciennes pages dans `docs/archive/` | fait | rapports, handoff et pages historiques classés et indexés |
| `DOC-005` | rendre le registre méthodologique visible à la racine sans perdre son historique | fait | registre détaillé conservé à la racine et pointeur historique maintenu |
| `DOC-006` | documenter la réalité et la cible des données | fait | carte du dépôt, cycle de vie et pointeur de snapshot documentés |
| `DOC-007` | supprimer le premier doublon de rôle entre `AGENT.md` et `AGENTS.md` sans perdre le guide historique | fait | ancien guide singulier archivé |
| `DOC-008` | vérifier tous les README locaux et liens | fait | validateur documentaire et test de structure verts |
| `DOC-009` | découper le catalogue Boosting en synthèse courante et journaux datés | fait | synthèse de 98 lignes, quatre journaux et reconstruction SHA-256 exacte |
| `DOC-010` | retirer les chemins documentaires de la logique des tests avant le classement final par thème | fait | découverte par contenu et ancien pointeur méthodologique supprimé |
| `DOC-011` | formaliser le standard Python détaillé | fait | règles et checklist dans `docs/standards/python.md` |
| `DOC-012` | formaliser le standard data détaillé | fait | contrats et checklist dans `docs/standards/data.md` |
| `DOC-013` | formaliser l'organisation cible et les dépendances autorisées | fait | règles dans `docs/standards/repository.md` |
| `DOC-014` | consolider les instructions agents dans le seul fichier réellement chargé | fait | `AGENTS.md` court, ancien contenu archivé, `AGENT.md` simple pointeur |
| `DOC-015` | formaliser le contrat Git tâche/commit/preuves | fait | `docs/standards/git.md` et liens normatifs |
| `DOC-016` | clarifier roadmap maître et registre méthodologique | fait | une seule liste de priorités actives, historique intégral conservé |
| `DOC-017` | supprimer le pointeur singulier `AGENT.md` après audit de ses lecteurs | fait | aucun lecteur actif, `AGENTS.md` seule source normative, ancien guide conservé dans l'archive |
| `DOC-018` | ouvrir le lot de qualité résiduelle après audit du commit propre et du worktree | fait | onze tâches ordonnées, chacune isolée par son identifiant et son commit |
| `DOC-019` | rafraîchir les compteurs de configurations, tests, code et dossiers après le lot | fait | état courant séparé du diagnostic initial, inventaires canoniques régénérés et 479 tests verts |
| `DOC-020` | formaliser l'invariant refresh, replay et attribution du drift | fait | règle inscrite dans les normes agents/développement, contrat canonique indexé et tâches d'exécution séparées |
| `DOC-021` | publier dans le site le guide complet des méthodes et de leurs pseudo-codes | fait | projection site synchronisée depuis `docs/site_repository_guide.md`, six tableaux et neuf pseudo-codes rendus ; statut des variantes, cas SATS et proposition de diversification explicités |
| `DOC-022` | réaligner l'inventaire data sur les lecteurs suivis et les volumes observés le 29 août | fait | 35 emplacements, 288 arêtes lecteur/emplacement, déplacements de lecteurs reflétés, volumes observés rafraîchis et validateur vert |

`DOC-010` a retiré des tests les chemins documentaires historiques. Les contrats
restent directement sous `docs/` pour leurs lecteurs humains et liens publics ;
un classement ultérieur exigera son propre inventaire de lecteurs.

## 7. Lot GIT — publier une histoire lisible en continu

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `GIT-001` | travailler sur `main`, pousser chaque commit et préserver l'ancien `main` sous `main-save` | fait | `main-save` = `c1113ab…`, `main` intégré par fast-forward, publication immédiate inscrite dans les documents normatifs |
| `GIT-002` | versionner et publier la preuve de chaque refresh, replay, backtest ou run de production important | fait | les artefacts lourds restent ignorés mais une tâche, une preuve canonique, ses hashes, son commit et son push deviennent obligatoires |

## 8. Lot QUAL — rendre les normes contrôlables

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `QUAL-001` | activer les choix de `CONTRIBUTING.md` pour le nouveau code et les nouvelles données | fait | standard version 1 documenté et indexé |
| `QUAL-002` | configurer Ruff dans `pyproject.toml` sans reformater le dépôt | fait | configuration partagée et commandes ciblées documentées |
| `QUAL-003` | établir une baseline des alertes et empêcher seulement les nouvelles régressions | fait | baseline Ruff déterministe, rapport reproductible et CI différentielle |
| `QUAL-004` | activer le typage progressivement par package | fait | périmètre Mypy strict explicite, initialisé sur `alpharank.quality` et exécuté en CI |
| `QUAL-005` | classer les tests en unitaires, intégration, replay, réseau et production | fait | politique ordonnée, marqueurs automatiques et cinq commandes séparées |
| `QUAL-006` | ajouter les contrôles de documentation, lint et tests ciblés en CI | fait | six groupes CI appelables par la même commande locale |
| `QUAL-007` | choisir `pyproject.toml` comme source des dépendances et générer ou vérifier les autres fichiers | fait | vues pip et Conda déterministes, vérifiées dans la gate statique |
| `QUAL-008` | rendre la CI AlphaRank autonome et isoler le contrôle Portfolio dans un job inter-projets explicite | fait | job `alpharank` autonome et job `portfolio-integration` séparé |
| `QUAL-009` | définir et valider un schéma versionné pour chaque famille de configuration JSON | fait | 17 familles et 21 fichiers contrôlés récursivement avant les tests |
| `QUAL-010` | corriger la gate CI pour un checkout sans artefacts locaux | fait | `--group ci` statique et 48 tests ciblés verts depuis un dépôt temporaire propre |
| `QUAL-011` | supprimer en priorité `F821`, `F403` et `F405` de la baseline Ruff | fait | zéro `F821`, `F403` ou `F405` ; branches multihorizon, service EODHD et refresh de référence couvertes |
| `QUAL-012` | rendre la suite Pytest complète autonome dans un checkout sans `outputs/` locaux | fait | fixtures synthétiques à la place des `outputs/` locaux et 479 tests verts sans masquage |
| `QUAL-013` | mesurer taille de module, longueur de fonction et complexité puis bloquer toute régression | fait | baseline versionnée sur 800/250/80/10 et gate différentielle sans exception implicite |
| `QUAL-014` | étendre Mypy strict au prochain package métier compatible | fait | package `alpharank.portfolio` entier ajouté, zéro erreur et périmètre verrouillé par test |
| `QUAL-015` | enregistrer la politique de transition ticker dans le registre des schémas JSON | fait | famille `price_ticker_transition_policy`, schéma strict récursif, 18 familles et 23 fichiers classés sans ambiguïté |

## 9. Lot CODE — découper sans changer les résultats

Priorité donnée aux composants de production et d'ingestion, pas aux fichiers
les plus faciles à déplacer.

| ID | Action | Statut | Contrôle obligatoire |
| --- | --- | --- | --- |
| `CODE-001` | cartographier les appels entre scripts et bibliothèque | fait | inventaire versionné des entrées actives, imports, commandes et lecteurs inverses |
| `CODE-002` | retirer les 35 injections de `sys.path` | fait | zéro injection restante et commandes représentatives exécutables hors dépôt |
| `CODE-003` | découper `data/open_source/ingestion.py` par étape du pipeline | fait | orchestration, schémas, prix et référentiels séparés ; 63 tests de caractérisation verts |
| `CODE-004` | alléger `scripts/run_legacy.py` en déplaçant la logique testable dans `src/` | fait | commande ramenée à moins de 350 lignes, moteur et hash de provenance sous `src/` |
| `CODE-005` | découper `strategy/legacy.py` par agrégation, sélection et artefacts | fait | façades inchangées et sorties économiques caractérisées à l'identique |
| `CODE-006` | découper `governance.py` par contrat de validation | fait | façade stable, six propriétaires documentés et 13 refus/messages inchangés |
| `CODE-007` | séparer calcul et rendu dans les dashboards de plus de 1 000 lignes | fait | deux scripts suivis sous 1 000 lignes, calculs et HTML testés séparément |
| `CODE-008` | centraliser les commandes de comparaison aujourd'hui dupliquées | fait | grilles temporelles et années de départ déléguées au moteur économique partagé |
| `CODE-009` | remplacer les captures générales et `print()` par des erreurs et journaux explicites | fait | zéro `print()` bibliothèque, zéro capture nue ou générale hors frontière journalisée ; contexte de run structuré |
| `CODE-010` | déplacer les scripts réellement obsolètes après audit des lecteurs | fait | sept scripts archivés avec hashes et zéro lecteur actif ; candidat SEC encore lu conservé |
| `CODE-011` | attribuer ou déplacer les modules transverses `*_v2`, gouvernance et replay dans des packages nommés | fait | six implémentations sous `replay/`, façades compatibles et registre d'API ; gouvernance attribuée |
| `CODE-012` | déplacer la construction testable du replay commun depuis le script public vers `src/alpharank/replay/` | fait | script ramené à une façade de 54 lignes, API publique testée et Top 5/10 natifs caractérisés sans changement de sélection |

Chaque découpage commence par un test de caractérisation. Aucune valeur de
portefeuille, de KPI ou de sélection ne doit changer dans ce lot.

## 10. Lot CODEORG — empêcher les dossiers Python de redevenir des piles

Le plafond porte sur les fichiers directement présents dans un dossier : un
sous-package n'est acceptable que s'il possède une responsabilité et une
frontière de dépendance compréhensibles. Aucun résultat data ou économique ne
change dans ce lot.

| ID | Action | Statut | Contrôle obligatoire |
| --- | --- | --- | --- |
| `CODEORG-001` | fixer le plafond à 20 fichiers `.py`, dérogation uniquement approuvée par le propriétaire, et inventorier les violations courantes | fait | politique versionnée, validateur testé ; 481 fichiers dans 67 dossiers, zéro violation et zéro dérogation |
| `CODEORG-002` | répartir `src/alpharank/data` et l'empilement `data/open_source` selon contrats, sources, entrepôt, lignée, qualité et publication | fait | 48 déplacements inventoriés ; dossiers data entre 4 et 13 modules, collecte complète et tests data verts, aucun calcul modifié |
| `CODEORG-003` | répartir la racine des scripts, `scripts/open_source` et `scripts/experiments` par usage durable | fait | 90 implémentations déplacées et 32 façades stables inventoriées ; racines à 15, 19 et 16 fichiers avec le worktree courant ; 6 commandes exécutées hors dépôt, 32 façades importées et 26 tests ciblés verts |
| `CODEORG-006` | restaurer le mode exécutable de `build_sec_output_package_with_backfill.py` après son déplacement | fait | contenu inchangé, mode indexé `100755`, compilation et chargement hors dépôt validés sans lancer la publication |
| `CODEORG-004` | répartir `tests/unit` et `tests/integration` par domaine miroir | fait | 100 déplacements ; signatures identiques pour 339 tests et 1 105 assertions ; collecte de 465 scénarios sans retrait, les 3 ajouts provenant de CODEORG-001/003 ; zéro dossier au-dessus du plafond |
| `CODEORG-005` | activer le contrôle bloquant dans la gate statique | fait | gate `--enforce-limit` verte sur 481 fichiers et 67 dossiers, zéro violation, zéro exception et 48 tests CI ciblés verts |
| `CODEORG-007` | retirer d'AlphaRank le dashboard interactif et conserver seulement calculs, contrats et artefacts machine-lisibles | fait | quatre propriétaires applicatifs et deux tests dédiés retirés ; rapports d'audit statiques conservés, Portfolio déclaré propriétaire de l'interface et inventaires régénérés |

## 11. Lot TEST — rendre les preuves navigables

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `TESTORG-001` | cataloguer les tests par domaine, niveau, réseau et durée | fait | catalogue courant : 125 fichiers et 479 cas verts, sans dépendance implicite à des artefacts locaux |
| `TESTORG-002` | créer une arborescence `unit`, `integration`, `replay` et `production` sans changer la découverte Pytest | fait | 419 identifiants canoniques identiques avant/après ; réseau isolé sous `integration/network` |
| `TESTORG-003` | découper les deux fichiers de tests de plus de 1 000 lignes | fait | 42 scénarios et 127 assertions préservés bit à bit au niveau AST ; aucun module de test au-dessus de 1 000 lignes |
| `TESTORG-004` | centraliser uniquement les fixtures réellement partagées | fait | une seule fixture racine isole le contexte de run entre suites ; helpers métier et fixtures Pytest restent locaux ou natifs |

## 12. Lot DATA — converger sans retélécharger ni perdre une révision

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `DATA-001` | figer le dictionnaire des emplacements actuels et de leurs lecteurs | fait | 35 fichiers/packages, 58 références statiques et 279 arêtes lecteur-emplacement dans un inventaire machine-lisible |
| `DATA-002` | déclarer `warehouse/raw` comme cible de toutes les sources brutes, EODHD inclus | fait | huit fournisseurs ont une racine RAW unique, des datasets déclarés et des contrats communs de reçu/manifeste |
| `DATA-003` | enregistrer chaque téléchargement par reçu et hash ; réutiliser le payload s'il est identique | fait | reçu immuable par tentative, échecs inclus ; objet adressé par SHA-256 et manifeste fournisseur vérifié |
| `DATA-004` | normaliser uniquement dans `stg` | fait | contrat STG sans priorité ni sélection ; conflits fournisseur conservés comme observations distinctes |
| `DATA-005` | rendre `def` responsable du choix de valeur et de sa provenance | fait | règle versionnée, cutoff causal, reçu choisi et motif explicite pour chaque clé résolue ou non |
| `DATA-006` | construire les entrées AlphaRank uniquement depuis `mart` | fait | Legacy résout par défaut un mart canonique ; neuf hashes DEF, mart et snapshot source sont identiques |
| `DATA-007` | définir le snapshot comme publication immuable du mart, pas comme couche concurrente | fait | publication par référence sans copie, inventaire exhaustif hashé et pointeur atomique validé |
| `DATA-008` | migrer les racines historiques par référence/hash avant toute copie | fait | 18 racines, 13 979 fichiers et 31,75 Go référencés par SHA-256 ; zéro téléchargement et zéro copie |
| `DATA-009` | basculer les lecteurs un par un vers les emplacements canoniques | fait | 162 arêtes lecteur/emplacement classées, zéro non classée ; 10 chemins ancien/MART comparés par SHA-256, deux entrées par défaut basculées |
| `DATA-010` | rendre les anciennes racines en lecture seule puis les archiver | fait | gel contractuel de 18 racines, observation 2026-08-20 au 2026-09-19, archive par référence et retour arrière hashé |
| `DATA-011` | étendre le bridge ticker/CIK historique et fournir une reconstruction SEC candidate | fait | bridge versionné porté à 75 identités dont 67 ajouts audités ; candidat hashé et bloqué, fallback filing-level tracé, snapshot courant inchangé et tests de réutilisation de symbole |
| `DATA-012` | retélécharger les sources rafraîchissables, reconstruire un candidat sans promotion et rejouer Legacy puis Boosting | fait | bootstrap `20260824_214818` bloqué avant fondamentaux sur 44 révisions Yahoo ; chaque source classée ; snapshot inchangé puis 7 994 holdings Legacy, 88 948 prédictions Boosting et 6 395 holdings communs reproduits sans drift matériel |
| `DATA-013` | versionner et publier la preuve du refresh et des deux replays exécutés le 25 août 2026 | fait | run `20260825_001501` bloqué avant fondamentaux sur 45 révisions Yahoo ; snapshot `9a2058c9…425ad` inchangé, Legacy et Boosting recalculés puis cinq étages comparés sans position ni poids modifié |
| `DATA-014` | séparer l'acquisition de la décision de publication pour qu'une révision prix ne coupe plus SEC et les fallbacks | fait | la gate prix est appliquée après toutes les sources déclarées ; `acquisition_status.json` distingue téléchargement, échec fournisseur et quarantaine avant toute publication |
| `DATA-015` | empêcher un refresh Yahoo de recopier ou d'écraser l'historique prix validé | fait | l'observation fournisseur reste dans l'archive RAW différentielle ; le canonique conserve chaque clé validée et ajoute seulement les dates nouvelles via leurs rendements ancrés ; diagnostic et cause de non-remplacement sont persistés |
| `DATA-016` | limiter la gate des mouvements prix aux nouvelles clés canoniques et différer sa décision jusqu'à la publication | fait | smoke réel sur le run `20260826_224908` : 2 500 nouvelles clés contrôlées avec leur ancre, zéro anomalie et aucun des 76 mouvements anciens requalifié ; gate combinée appliquée après acquisition ; 500 tests et gates statiques/documentaires verts |
| `DATA-017` | rafraîchir toutes les sources au 27 août 2026 et expliquer le drift des deux backtests | fait | acquisition `20260827_070654` complète jusqu'à la séance du 26 août ; contrôle same-code puis ablation prix/SEC : le candidat SEC suffit à faire entrer `CVC.US` au rang 8 du Top 10 Boosting de juillet 2016 ; replay commun bloqué, rapport `16608074…f014`, aucune promotion |
| `DATA-018` | borner le transport SimFin et reprendre en IPv4 quand IPv6 reste bloqué | fait | connexion et lecture bornées ; un échec initial est retenté une fois en IPv4, puis classé explicitement sans bloquer indéfiniment l'ingestion ; archive installée atomiquement et tests de non-régression verts |
| `DATA-019` | autoriser l'ancre DEF conservée pour prolonger un ticker dont le payload frais commence après la dernière séance validée | fait | l'ancre garde son ancien `ingestion_run_id`, seules les dates nouvelles sont ajoutées et auditées ; test de non-régression fidèle au blocage EQR/AVB du 27 août |
| `DATA-020` | enregistrer la fusion AVB/EQR et reconstruire l'univers S&P courant avant le refresh prix | fait | l'événement officiel du 18 août retire AVB, ajoute RDDT, renomme EQR en VMRK et conserve séparément la contrepartie actionnaire AVB ; aucun symbole n'est prolongé comme s'il désignait le même titre ; 504 tests et gates documentaires verts |
| `DATA-021` | qualifier le mouvement RDDT du 30 octobre 2024 sans relâcher la gate prix | fait | hausse de 41,97 % bornée aux deux prix observés et reliée aux résultats T3 officiels publiés après la clôture précédente ; toute autre valeur reste bloquante et test de non-régression vert |
| `DATA-022` | republier un run entièrement acquis après revue sans retélécharger les mêmes sources | fait | commande dédiée sans client réseau, package candidat lié par hash au statut, au contrat, à la gate originale et au registre de revue ; ancien script ramené de 336 à 109 lignes, aucune régression de taille/complexité, 27 tests ciblés et 48 gates CI verts |
| `DATA-023` | reproduire la réconciliation canonique lors d'une republication différée | fait | la republication conserve chaque rendement historique validé, ajoute uniquement les nouveaux rendements et lie par hash les preuves de réconciliation ; 11 tests ciblés, lint et package réel sans réseau verts |
| `DATA-024` | transmettre les arguments de la façade CLI du package SEC | fait | la commande publique parse désormais les chemins et options avant d'appeler l'implémentation ; `--help` est couvert hors du dépôt, 5 tests ciblés, lint et docs verts, sans changement des transformations SEC |
| `DATA-025` | auditer un refresh dont le replay commun échoue sur une gate | fait | snapshot, Legacy et signaux Boosting restent comparés ; la raison exacte reçoit le statut `common_replay_blocked`, aucune table commune n'est inventée et la promotion reste interdite ; 11 tests ciblés, lint et docs verts |
| `DATA-026` | séparer les identifiants de données des paramètres dans la comparaison de provenance | fait | les hashes d'entrée restent comparés comme données sans créer un faux drift de configuration ; les politiques, seeds, code et runtime conservent leurs contrôles indépendants ; 11 tests ciblés, lint et docs verts |
| `DATA-027` | produire un explorateur SEC autonome par entreprise depuis un run RAW explicite | fait | run `20260827_070654` : 824 sociétés et 638 809 lignes SEC ; rapport `d8285970…20be`, payload `959a922c…8961` ; versions, quarters, lignes brutes, statuts et six hashes sources visibles ; tests, JavaScript, lint, plafond de dossier et docs verts ; gate de taille globale encore rouge sur un fichier non modifié |
| `DATA-028` | versionner les quatre événements terminaux révélés par le replay sans fondamentaux | fait | registre différentiel v2 lié par hash au v1 ; RX, TSS, TWTR et ABMD bloqués uniquement après leur dernière séance primaire ; quatre pièces SEC refetchées au même SHA-256, sans valoriser une contrepartie actionnaire ni réintroduire un facteur fondamental |
| `DATA-029` | prolonger un ancien ticker depuis les rendements d'un alias fournisseur de la même sécurité | fait | 24 séances SATS dérivées des rendements ECHO, zéro ligne antérieure modifiée et zéro valeur manuelle ; package réel et snapshot `1e6d5367…842a9` reconstruits sans réseau, 21 tests ciblés et validations documentaires verts |

Aucune suppression physique de données n'est autorisée par ce lot. Une éventuelle
politique de rétention fera l'objet d'une décision séparée après mesure des
doublons exacts et preuve de récupération.

### Détail de `DATA-027`

- **Objectif** : filtrer une société et auditer tous ses faits SEC téléchargés,
  avec graphiques fiscaux trimestriels et preuve ligne à ligne.
- **Périmètre** : six Parquet SEC/référentiel, statut d'acquisition, générateur
  statique, tests et contrat SEC.
- **Hors périmètre** : sélection DEF, package modèle, prix, Portfolio et
  promotion de `latest.json`.
- **Acceptation** : run RAW obligatoire, toutes les versions conservées, HTML
  autonome sans asset réseau, filtre société, export CSV et manifeste hashé.
- **Validations** : tests unitaires du payload et des refus, syntaxe JavaScript,
  Ruff, seuils Python et validation documentaire.
- **Impact** : aucun changement data ou économique ; nouvelle vue d'audit
  régénérable sur des fichiers immuables.
- **Rollback** : revenir au générateur précédent ; aucun dataset ni pointeur
  n'est modifié par la commande.

### Détail de `DATA-029`

- **Objectif** : récupérer les séances SATS présentes sous la clé fournisseur
  ECHO sans saisir un prix ou un rendement et sans fusionner leurs anciens
  historiques homonymes.
- **Périmètre** : registre versionné, contrôle de cinq rendements communs et de
  l'ancre du 24 avril, extension du 25 avril au 31 mai, lignée, manifeste,
  publication prix et tests.
- **Hors périmètre** : modification d'une ligne déjà publiée, fusion globale de
  SATS/ECHO, changement du calendrier de l'univers ou promotion d'un snapshot.
- **Acceptation** : même CIK et CUSIP inchangé sourcés ; anciennes clés
  identiques ; chaque nouvelle ligne dérivée du rendement quotidien ECHO depuis
  l'ancre SATS ; zéro valeur manuelle ; overlay idempotent et hashé.
- **Validations** : 21 tests du recouvrement, du refus, de l'idempotence, de la
  réconciliation et de la composition ; package et snapshot réels reconstruits
  sans réseau ; Ruff et documentation verts ; aucune nouvelle dette Python, la
  gate globale restant rouge sur deux fichiers non modifiés.
- **Impact** : les observations fin avril et mai deviennent calculables sous
  SATS ; l'impact sur les signaux et portefeuilles reste inconnu avant
  `REPLAY-006`.
- **Rollback** : omettre le nouveau package ; le snapshot précédent reste
  immuable et résolvable par son manifeste.

## 12 bis. Lot METH — preuves économiques complémentaires

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `METH-003` | exposer rendement excédentaire annualisé, tracking error, asymétrie et kurtosis dans le moteur commun | fait | formules centralisées dans `portfolio/performance.py`, cas sans benchmark et vide testés ; aucun rendement mensuel modifié |
| `METH-004` | filtrer les prédictions Boosting par l'éligibilité PE point-in-time de Legacy avant classement | fait | registre causal de 88 948 ticker-mois, variantes Top 5/10/15/20 natives et appariées sur le même snapshot, bootstrap de 50 000 tirages et aucune promotion de Boosting |
| `METH-005` | exposer une politique Legacy sans fondamentaux SEC sans la promouvoir avant replay strict | fait | `no_sec_fundamentals_v1` construit l'univers depuis prix et membership uniquement ; CLI, manifeste et provenance enregistrent le choix ; trois tests de contrat verts ; promotion traitée séparément par `REPLAY-005` |
| `METH-006` | rendre disponible un filtre causal de tendance avant le classement Boosting | fait | registre exact de 88 950 clés, dont 40 135 éligibles ; 16 replays OOS vérifiés par hash, majorité stricte orientée, couverture complète, CLI et artefacts ; cinq tests de politique, variante non promue et natif inchangé |
| `METH-007` | exécuter le replay commun de la variante Boosting filtrée par tendance | fait | Top 5/10 sur 180 mois : CAGR 23,63 %/21,95 %, mais IC bootstrap rendement et Sharpe traversent zéro ; essai Top 15/20 bloqué causalement sur le prix manquant de SATS ; dernier Top 10, hashes et statut non promu publiés |
| `METH-008` | construire une politique d'allocation Boosting complémentaire à Legacy | à faire | à chaque décision, les titres retenus causalement par Legacy sont exclus avant le classement Boosting ; scores et modèle restent inchangés ; test de causalité et audit mensuel de l'exclusion |
| `METH-009` | rejouer le portefeuille combiné Legacy et poche Boosting complémentaire | à faire | poids de poche préenregistrés, même snapshot et même moteur ; mesure du nombre de titres, recouvrement, concentration, corrélation, risque, coûts et performance combinée sans exiger que Boosting batte Legacy seul |

### Détail de `METH-006`

- **Objectif** : tester si le Boosting devient réellement trend-following lorsque
  son classement mensuel est limité aux titres dont une majorité stricte des
  signaux EMA relatifs causaux indique une tendance positive.
- **Périmètre** : registre ticker-mois construit depuis les paires gagnantes de
  chaque fold, filtre optionnel du replay commun, CLI, manifeste, tests et
  contrats méthodologiques.
- **Hors périmètre** : réapprentissage du modèle, nouvelle feature, lecture de
  SHAP ou de rendement futur pendant l'allocation, changement du profil public
  natif et promotion en production.
- **Acceptation** : chaque prédiction correspond exactement à une ligne du
  registre ; une paire manquante rend le titre inéligible ; les paires inversées
  conservent le bon sens économique ; la majorité est stricte et calculée avant
  le Top-N.
- **Validations** : tests de sens EMA, couverture, majorité, absence de
  dépendance aux cibles futures, CLI, Ruff, taille Python et documentation.
- **Impact** : nouvelle variante R&D explicite ; aucun score, univers natif,
  poids Legacy ou pointeur de production n'est modifié.
- **Rollback** : omettre l'option du replay ; les sorties natives restent
  identiques et la variante disparaît sans migration de donnée.

### Détail de `METH-007`

- **Objectif** : mesurer la variante sur tout l'historique commun puis figer sa
  performance, sa stabilité et son portefeuille le plus récent.
- **Périmètre** : tentative commune Top 5/10/15/20 puis replay exploitable
  Top 5/10 depuis le run sans SEC du 28 août 2026, rapport de preuve et mise à
  jour de la roadmap.
- **Hors périmètre** : tuning sur le résultat, modification des données, choix
  d'un autre snapshot ou remplacement du Boosting natif.
- **Acceptation** : code Git propre, mêmes 180 mois, snapshot, terminal gate,
  coûts et benchmark ; toute taille sélectionnant un rendement manquant
  s'arrête ; artefacts Top 5/10 hashés ; comparaison native/filtrée ; dernier
  Top 10 détaillé ; statut de promotion explicite.
- **Validations** : replay complet, manifeste, calendrier exact, absence de
  rendement censuré sélectionné, validations documentaires et Git.
- **Impact** : preuve économique supplémentaire seulement ; l'idée du filtre
  ayant été motivée après lecture des SHAP récents, le replay reste post-hoc et
  ne constitue pas à lui seul une validation indépendante.
- **Rollback** : conserver le run comme preuve négative ou exploratoire et ne
  changer aucun profil public ni portefeuille canonique.

### Détail de `METH-008`

- **Objectif** : produire une poche Boosting dont la sélection mensuelle ajoute
  des titres absents du portefeuille Legacy connu à la même date.
- **Périmètre** : allocation après scores OOS, exclusion causalement datée des
  titres Legacy, plafonds sectoriels optionnels et journal d'éligibilité.
- **Hors périmètre** : réentraînement XGBoost, nouvelle cible, usage des SHAP,
  choix de poids entre les deux poches et promotion en production.
- **Acceptation** : aucun titre de la poche complémentaire n'appartient au
  Legacy du même mois ; modifier un Legacy futur ne change aucune sélection
  passée ; les scores natifs restent bit-à-bit identiques.
- **Validations** : tests unitaires de sélection, test de causalité par mutation
  du futur et parité des prédictions natives.
- **Impact** : nouvelle allocation R&D, aucun changement de la production
  Legacy ni du modèle Boosting.
- **Rollback** : retirer la politique optionnelle et conserver les scores et
  allocations natives inchangés.

### Détail de `METH-009`

- **Objectif** : mesurer si une poche Boosting complémentaire améliore la
  diversification du portefeuille total, sans imposer une supériorité de sa
  performance autonome.
- **Périmètre** : poids de poche figés avant replay, moteur commun, coûts,
  recouvrement, nombre effectif de positions, concentration sectorielle,
  corrélation, risque et performance du portefeuille combiné.
- **Hors périmètre** : optimisation des poids sur les résultats finaux,
  changement de Legacy, promotion automatique ou recommandation d'achat.
- **Acceptation** : comparer les poids de poche préenregistrés de 10 %, 20 % et
  30 % sur le même snapshot et calendrier ; publier les métriques de
  diversification et les compromis économiques, y compris un résultat négatif.
- **Validations** : replay commun strict, bootstrap temporel apparié, contrôle
  des coûts et rapport exhaustif des mois bloqués ou non évaluables.
- **Impact** : preuve économique R&D d'un portefeuille combiné ; aucun
  changement de production sans nouvelle porte de promotion.
- **Rollback** : conserver les deux méthodes séparées et ignorer le portefeuille
  combiné.

## 12 ter. Lot REPLAY — relier chaque refresh aux portefeuilles

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `REPLAY-001` | comparer baseline et candidat depuis les entrées jusqu'aux holdings Legacy et Boosting | fait | audit causal au cutoff, preuves Parquet par clé, comparaison du code/config/runtime et code retour bloquant pour tout statut autre que l'identité historique |
| `REPLAY-002` | rendre chaque écart code, configuration et runtime directement explicable | fait | chemins de run neutralisés ; valeurs avant/après listées par chemin JSON pour Git, fichiers critiques, paramètres, dépendances et seeds |
| `REPLAY-003` | conserver un statut machine-lisible pour chaque source après une gate amont | fait | prix Yahoo téléchargés/quarantinés, historiques gelés conservés et acquisitions fondamentales non démarrées distingués explicitement |
| `REPLAY-004` | produire un rapport HTML autonome qui sépare drift prix, SEC, Legacy et Boosting | fait | rapport réel `8881cac6…b971`, payload `1aaac44b…cdb1`, ablations prix/SEC, scores, Top-N, CVC, gate et hashes réunis ; 13 tests ciblés, typage, lint, navigation HTML et absence d'asset externe validés |
| `REPLAY-005` | rejouer les deux méthodes sans SEC et promouvoir la politique si les gates communes passent | fait | données fraîches au 26 août ; Legacy strict, Boosting EMA-only et replay commun sur 180 mois verts ; 7/7 hashes identiques, 8 entrées terminales bloquées, zéro rendement censuré sélectionné, `publication_eligible=true` ; `no_sec_fundamentals_v1` devient le défaut Legacy |
| `REPLAY-006` | rejouer Legacy, Boosting et la variante tendance après l'overlay SATS/ECHO | fait | snapshot `bb1f90a9…8375` ; SATS reste rang 14 sans drift de score, son rendement mai devient +4,9131 % ; Legacy et Top 5/10 inchangés, Top 15/20 tendance calculables ; rapport HTML `28c67752…b291` |

### Détail de `REPLAY-005`

- **Objectif** : retirer les fondamentaux SEC de la sélection canonique Legacy
  après un backtest complet de Legacy et Boosting sur les mêmes données fraîches.
- **Périmètre** : défaut CLI/pipeline Legacy, replay frais du 28 août 2026,
  documentation méthodologique, runbook et tests de contrat du défaut.
- **Hors périmètre** : suppression des archives SEC, modification des prix,
  promotion de Boosting en production ou modification de ses variables EMA-only.
- **Acceptation** : package Legacy strict valide ; profil Boosting public exact ;
  replay commun `comparison_eligible=true` et `publication_eligible=true` ; mêmes
  hashes d'entrée, calendrier et coûts ; aucun rendement censuré sélectionné.
- **Validations** : 30 essais sur 17 fenêtres et quatre trajectoires Legacy ;
  16 folds Boosting ; 180 mois communs d'août 2011 à juillet 2026 ; tests ciblés,
  Ruff et validations documentaires.
- **Impact** : Legacy ne filtre plus l'univers sur le market cap ou
  `0 < PE < 100` ; Boosting reste sans feature SEC mais est réappris depuis les
  EMA gagnantes du nouveau Legacy ; les archives SEC restent disponibles pour
  audit et recherche.
- **Rollback** : passer explicitement
  `--fundamental-eligibility-policy-id legacy_pe_market_cap_v1` dans un replay
  de compatibilité ; ne pas restaurer silencieusement ce filtre comme défaut.

### Détail de `REPLAY-006`

- **Objectif** : mesurer l'effet exact de la continuité SATS/ECHO après avoir
  reconstruit toutes les features, cibles et simulations, sans injecter le
  rendement de mai dans le moteur.
- **Périmètre** : snapshot composé `bb1f90a9…8375`, Legacy 30 essais sur quatre
  trajectoires de 17 fenêtres, Boosting EMA-only 16 folds, replays communs natif
  et tendance Top 5/10/15/20, rapport JSON/HTML autonome et hashé.
- **Hors périmètre** : déplacement de `data/model_inputs/manifests/latest.json`,
  promotion de la variante tendance et modification d'un snapshot antérieur.
- **Acceptation** : 24 séances SATS ajoutées et aucune ligne antérieure changée ;
  score et rang SATS identiques ; rendement avril-vers-mai évaluable ; holdings
  Legacy identiques ; Top 15/20 tendance terminés avec les mêmes données, coûts
  et moteur que Top 5/10.
- **Validations** : 5 tests prix/rapport ciblés, Ruff, format, documentation et
  liens ; deux manifestes communs `comparison_eligible=true`, mêmes sept hashes,
  huit entrées terminales bloquées et zéro rendement censuré sélectionné.
- **Impact** : SATS reste rang 14 de la décision d'avril et réalise +4,9131 % en
  mai ; le signal Boosting ne change pas, Legacy ne détenait pas SATS et ses
  holdings/performance sont identiques ; Top 15/20 ne sont plus invalides.
- **Rollback** : résoudre le snapshot précédent par son manifeste immuable ; le
  pointeur canonique n'a pas été déplacé par cette preuve.

## 12 quater. Lot REPORT — standardiser la lecture des performances

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `REPORT-001` | centraliser le rapport HTML complet et ses filtres temporels | fait | 33 KPI de chaque fenêtre annuelle calculés par le moteur commun, 11 séries dont SPY, model cards CAGR/volatilité/drawdown en Viridis, holdings exhaustifs, méthodologies, lignée et tests sans asset réseau |
| `REPORT-002` | générer le rapport sur le replay SATS/ECHO et le synchroniser vers Portfolio | fait | HTML et manifeste hashés depuis le snapshot `bb1f90a9…8375`, preuve datée versionnée, copie byte-identique dans le site au commit Portfolio `7e66fa5`, build Vite et routes HTTP validés |
| `REPORT-003` | rendre la comparaison multi-stratégie explicite dans chaque vue | fait | KPI des 11 séries côte à côte avec surperformance SPY visible, multisélection des courbes, matrices cumulées bornées par début/fin et matrices annuelles incrémentales sans nouveau calcul navigateur |
| `REPORT-004` | publier une nouvelle instance SATS/ECHO du standard enrichi | fait | HTML et manifeste hashés, preuve datée, copie site byte-identique au commit Portfolio `71a73c5`, build Vite et contrôles interactifs 11 stratégies/2015–2019 validés |
| `REPORT-005` | aligner toutes les vues sur la sélection globale et rendre le drawdown lisible | fait | cartes, tableau KPI et matrices limités aux courbes cochées ; richesse puis drawdown en graphiques pleine largeur superposés |
| `REPORT-006` | comparer une combinaison équipondérée de stratégies à SPY | fait | nouvel onglet, sélection des poches, rendements mensuels et KPI de chaque combinaison pré-calculés par `alpharank.portfolio`, règle de coûts et rééquilibrage documentée |
| `REPORT-007` | publier la correction et le laboratoire SATS/ECHO | fait | nouvel artefact hashé, copie Portfolio byte-identique au commit `4f16576`, build et QA des filtres, du drawdown et du portefeuille composé |
| `REPORT-008` | distinguer corrélation et surperformance relative dans le laboratoire | fait | corrélation mensuelle du portefeuille au SPY, matrice entre poches cochées et richesse composée divisée par la richesse SPY, toutes bornées par la fenêtre active |
| `REPORT-009` | publier les diagnostics de corrélation SATS/ECHO | fait | nouvel artefact hashé, copie Portfolio byte-identique au commit `4fdc1b5`, build et QA des corrélations et de la richesse relative |
| `REPORT-010` | afficher le portefeuille en vigueur après le dernier mois de performance réalisé | fait | panier Legacy et Boosting du mois courant exposé séparément, date de marché explicite, rendement non réalisé visible et calendrier des KPI inchangé |
| `REPORT-011` | publier le portefeuille en vigueur au 28 août dans Portfolio | fait | artefact SATS/ECHO régénéré avec preuve de marché du 28 août, copie site byte-identique, build et QA du panier courant ; preuve `docs/research/backtest_performance_report_20260830_current_portfolio.md` |

### Détail de `REPORT-001`

- **Objectif** : disposer d'une page unique pour approfondir toute performance,
  période, méthode et position historique sans créer un second moteur de KPI.
- **Périmètre** : KPI communs, toutes les fenêtres bornées par année, graphiques,
  matrices par année de départ, holdings, pseudo-codes, lignée, générateur et
  contrat canonique.
- **Hors périmètre** : recalcul des signaux, modification d'un rendement,
  promotion d'une variante, recommandation d'achat et donnée Portfolio/IBKR.
- **Acceptation** : Legacy Frequency/Equal, Boosting natif et tendance Top
  5/10/15/20 et SPY partagent 180 mois ; tout KPI affiché provient du package
  `alpharank.portfolio` ; 2011 reste partiel ; aucun asset réseau.
- **Validations** : tests unitaires moteur/payload/HTML, syntaxe JavaScript,
  Ruff, mypy strict Portfolio, plafonds Python et documentation.
- **Impact** : reporting seulement ; zéro changement de score, sélection,
  poids, rendement, snapshot ou statut de promotion.
- **Rollback** : retirer le générateur et conserver les artefacts communs ; les
  backtests et leurs hashes restent inchangés.

### Détail de `REPORT-002`

- **Objectif** : produire la première instance du standard depuis le replay
  corrigé SATS/ECHO et la rendre accessible dans le portail Portfolio.
- **Périmètre** : run explicite, manifeste du rapport, preuve datée, copie HTML
  et manifeste, navigation du site et build frontend.
- **Hors périmètre** : déplacement de `latest.json`, recomputation dans
  Portfolio, changement du dashboard IBKR ou déploiement externe.
- **Acceptation** : rapport régénéré depuis un commit AlphaRank propre, hashes
  des six entrées, statut candidat/non promu visible, tous les mois et holdings
  retrouvables, route du site fonctionnelle.
- **Validations** : génération réelle, cohérence des nombres de lignes,
  manifeste SHA-256, validation documentaire AlphaRank et build Portfolio.
- **Impact** : publication d'une preuve de lecture ; aucune modification
  économique ni garantie supplémentaire de promotion.
- **Rollback** : retirer l'onglet et la copie publique ; le rapport source reste
  reconstructible depuis ses chemins et hashes.

### Détail de `REPORT-003`

- **Objectif** : lire immédiatement chaque KPI pour toutes les stratégies,
  choisir librement les courbes et distinguer performance cumulée et année
  isolée dans la fenêtre sélectionnée.
- **Périmètre** : comparaison KPI par stratégie, référence SPY, multisélection,
  filtrage des deux matrices Viridis par début/fin et projection annuelle
  incrémentale depuis le cube de KPI existant.
- **Hors périmètre** : nouvelle formule financière, changement de rendement,
  signal, sélection, modèle, snapshot ou statut de promotion.
- **Acceptation** : les 11 séries sont visibles dans les colonnes KPI ; les
  cellules comparables indiquent surperformance/sous-performance contre SPY ;
  toute combinaison de courbes est disponible ; les deux matrices ne dépassent
  jamais les bornes choisies et utilisent exclusivement `metric_windows`.
- **Validations** : tests du payload/HTML, test des fenêtres annuelles, syntaxe
  JavaScript, Ruff, taille Python, documentation et revue navigateur.
- **Impact** : interaction et lecture seulement ; aucun artefact économique
  source n'est modifié.
- **Rollback** : republier l'HTML `REPORT-002` ; ses entrées et hashes restent
  conservés dans la preuve du 29 août.

### Détail de `REPORT-004`

- **Objectif** : rendre la version comparative accessible dans le portail
  Portfolio avec une preuve exacte de l'artefact servi.
- **Périmètre** : génération SATS/ECHO, manifeste, copie HTML, navigation déjà
  existante, build du site, QA interactive et rapport daté.
- **Hors périmètre** : déploiement Cloudflare, calcul frontend, promotion de
  modèle et modification du dashboard IBKR.
- **Acceptation** : source, copie publique et build ont les mêmes hashes ; le
  multiselect, les colonnes KPI et les matrices bornées sont vérifiés sur une
  fenêtre réduite ; le manifeste conserve le statut candidat/non promu.
- **Validations** : génération réelle, hashes, contrat du manifeste, build
  Vite, routes HTTP, absence d'erreur console et documentation AlphaRank.
- **Impact** : remplacement d'une projection statique du même replay ; zéro
  changement de portefeuille ou de performance.
- **Rollback** : restaurer la copie site de `REPORT-002` depuis le commit
  Portfolio `7e66fa5` sans toucher aux runs AlphaRank.

### Détail de `REPORT-005`

- **Objectif** : faire du multiselect l'unique filtre de stratégies visible et
  rendre le drawdown aussi lisible que la croissance composée.
- **Périmètre** : cartes de synthèse, tableau des 33 KPI, matrices cumulées et
  annuelles, mise en page des deux graphiques et documentation du filtre.
- **Hors périmètre** : calcul de KPI, rendement, signal, portefeuille source,
  snapshot, benchmark et statut de promotion.
- **Acceptation** : avec deux courbes cochées, aucune troisième stratégie
  n'apparaît dans les cartes, le tableau, les légendes ou les matrices ; le
  drawdown occupe une ligne pleine largeur sous la croissance composée.
- **Validations** : tests HTML, syntaxe JavaScript, Ruff, documentation et QA
  navigateur sur une sélection de deux courbes.
- **Impact** : correction d'affichage uniquement ; aucun chiffre source ne
  change.
- **Rollback** : restaurer le rendu de `REPORT-003`, sans toucher au payload.

### Détail de `REPORT-006`

- **Objectif** : mesurer si combiner plusieurs méthodes diversifie réellement
  la volatilité et le drawdown face à SPY.
- **Périmètre** : toutes les combinaisons équipondérées des dix stratégies hors
  SPY, rééquilibrage mensuel, KPI et rendements pré-calculés, onglet interactif
  et méthodologie visible.
- **Hors périmètre** : poids libres, optimisation de poids, coût supplémentaire
  entre poches, fusion des holdings sous-jacents et promotion d'une combinaison.
- **Acceptation** : l'utilisateur coche les poches, voit leur poids égal, la
  courbe composée, le drawdown, CAGR, rendement, volatilité, Sharpe, Sortino et
  max drawdown contre SPY sur la fenêtre active ; aucun KPI n'est calculé en
  JavaScript.
- **Validations** : test de parité d'une combinaison contre le moteur commun,
  test du payload/HTML, taille de l'artefact, syntaxe JavaScript et QA navigateur.
- **Impact** : nouveau diagnostic post-hoc ; rendements sources et méthodes
  restent inchangés.
- **Rollback** : retirer l'onglet et le payload de combinaisons ; les séries
  individuelles restent identiques.

### Détail de `REPORT-007`

- **Objectif** : remplacer dans Portfolio le rapport mal interprété par la
  version corrigée et conserver une preuve de l'artefact exact.
- **Périmètre** : génération SATS/ECHO, hashes, copie statique, build Portfolio,
  QA et preuve datée.
- **Hors périmètre** : dashboard IBKR, déploiement Cloudflare, donnée ou modèle.
- **Acceptation** : source, copie publique et build sont byte-identiques ; les
  critères `REPORT-005/006` sont rejoués sur le fichier réellement servi.
- **Validations** : manifeste, hashes, build Vite, routes locales, console et
  documentation.
- **Impact** : publication d'une nouvelle projection du même replay.
- **Rollback** : restaurer la copie Portfolio du commit `71a73c5`.

### Détail de `REPORT-008`

- **Objectif** : séparer clairement la dépendance statistique entre stratégies
  de leur surperformance composée face au SPY.
- **Périmètre** : corrélations de Pearson des rendements mensuels entre les
  poches cochées, corrélation du portefeuille composé avec SPY et richesse
  relative `richesse portefeuille / richesse SPY` sur la fenêtre active.
- **Hors périmètre** : corrélation des niveaux de prix, optimisation des poids,
  interprétation causale de la corrélation et promotion d'une combinaison.
- **Acceptation** : la matrice contient uniquement les poches cochées ; le KPI
  de corrélation au SPY provient du moteur commun ; une courbe relative au-dessus
  de 1 indique que le portefeuille a davantage composé que SPY depuis le début
  de la fenêtre, sans être appelée corrélation.
- **Validations** : test de parité des corrélations, payload/HTML, syntaxe
  JavaScript, génération réelle et QA navigateur sur deux poches Boosting.
- **Impact** : nouvelles lectures d'un même replay ; aucun rendement, poids,
  signal ou portefeuille source ne change.
- **Rollback** : retirer la matrice, le KPI et la courbe relative sans toucher
  aux six KPI et aux deux graphiques existants.

### Détail de `REPORT-009`

- **Objectif** : publier les nouveaux diagnostics dans Portfolio avec leur
  preuve exacte.
- **Périmètre** : génération SATS/ECHO, hashes, copie statique, build Portfolio,
  QA de deux poches Boosting et preuve datée.
- **Hors périmètre** : dashboard IBKR, déploiement externe, donnée ou modèle.
- **Acceptation** : source, copie publique et build sont byte-identiques ; la
  corrélation entre poches, la corrélation au SPY et la richesse relative
  réagissent à la sélection et à la fenêtre sur le fichier réellement servi.
- **Validations** : manifeste, hashes, build Vite, route locale, interactions et
  documentation.
- **Impact** : publication d'une nouvelle projection du même replay.
- **Rollback** : restaurer la copie Portfolio du commit `4f16576`.

### Détail de `REPORT-010`

- **Objectif** : ne plus masquer le portefeuille mensuel encore en vigueur
  lorsque son rendement complet n'appartient pas encore au backtest réalisé.
- **Périmètre** : holdings live Boosting, agrégations Legacy courantes, preuve
  explicite de la dernière séance observée, payload, rendu HTML, tests et
  contrat de reporting.
- **Hors périmètre** : ajout d'un rendement mensuel incomplet aux KPI,
  recalcul de signal au milieu du mois, promotion d'une stratégie ou mutation
  d'un run source.
- **Acceptation** : les courbes et KPI restent arrêtés en juillet ; le panier
  détenu en août est visible pour les dix stratégies avec décision, détention,
  date de marché et statut « rendement non réalisé » distincts.
- **Validations** : test de non-régression payload/HTML, syntaxe JavaScript,
  Ruff, plafonds Python et documentation.
- **Impact** : correction de lecture uniquement ; scores, sélections, poids,
  rendements réalisés et hashes des replays restent inchangés.
- **Rollback** : retirer la vue courante et conserver la table historique
  bornée au dernier mois réalisé.

### Détail de `REPORT-011`

- **Objectif** : rendre la correction accessible dans le portail Portfolio
  avec une preuve exacte du portefeuille en vigueur au 28 août 2026.
- **Périmètre** : génération SATS/ECHO, évidence canonique de prolongation prix
  au 28 août, hashes, copie statique, build Portfolio, QA et preuve datée.
- **Hors périmètre** : nouveau signal fondé sur le mois d'août incomplet,
  dashboard IBKR, déploiement externe, donnée ou modèle.
- **Acceptation** : source et copie publique sont byte-identiques ; juillet
  reste la fin des performances et août apparaît comme portefeuille courant
  valorisable au 28 août, sans rendement mensuel inventé.
- **Validations** : manifeste, hashes, build Vite, route locale, interactions,
  inspection du payload et documentation.
- **Impact** : publication d'une nouvelle projection du même replay ; aucun
  changement économique des stratégies.
- **Rollback** : restaurer la copie Portfolio du commit `4fdc1b5`.

## 13. Lot RUN — remettre de l'ordre dans résultats et journaux

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `RUNORG-001` | inventorier les 346 dossiers racine de `outputs/` par famille, date, statut et taille | fait | registre de 346 racines, 33,82 Go et 17 familles, sans inférence de statut depuis le nom |
| `RUNORG-002` | définir un chemin unique `outputs/<famille>/<run_id>/` | fait | convention `lower_snake_case` + identifiant UTC documentée et validée à exactement deux niveaux |
| `RUNORG-003` | séparer `candidate`, `validated`, `published` et `failed` dans le manifeste, pas dans des noms libres | fait | manifeste obligatoire dès l'initialisation, historique et transitions contrôlés, statut interdit dans le chemin |
| `RUNORG-004` | relier chaque journal au manifeste du run | fait | nouveaux journaux hashés dans le manifeste avec sidecar de retour ; 74 journaux historiques préservés sans appariement inventé |
| `RUNORG-005` | produire des pointeurs `latest` atomiques sans copier les résultats | fait | pointeur atomique vers un run publié, manifeste et arbre hashés, copie immuable du pointeur, zéro copie de résultat |
| `RUNORG-006` | mesurer les doublons exacts et proposer une rétention réversible | fait | 3 866 groupes SHA-256, 10 404 copies et 11,32 Go récupérables mesurés ; proposition réversible, zéro suppression |

## 14. Ordre d'exécution et portes de décision

1. **Documentation** : terminer `DOC-*`, sans code ni donnée.
2. **Validation des normes** : le propriétaire approuve ou modifie
   `CONTRIBUTING.md`.
3. **Garde-fous** : exécuter `QUAL-*` avant les gros déplacements.
4. **Code et tests** : traiter `CODE-*` et `TESTORG-*` par risque, avec parité
   fonctionnelle et même collecte de tests.
5. **Données** : traiter `DATA-*` avec inventaires, hashes et retour arrière.
6. **Résultats** : traiter `RUNORG-*` une fois les nouveaux contrats stables.

La remise en ordre est terminée seulement lorsqu'un nouvel humain peut :

- trouver en moins de cinq minutes comment lancer Legacy ou Boosting ;
- identifier sans ambiguïté la donnée de production courante ;
- expliquer le trajet d'une valeur de sa source au portefeuille ;
- retrouver un run et ses journaux par un seul identifiant ;
- modifier un composant sans lire un fichier monolithique sans rapport avec sa
  tâche ;
- exécuter localement les mêmes contrôles que la CI.

# Roadmap maître AlphaRank

**Dernière mise à jour : 2026-08-25.**

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
| Configurations | 17 familles, 21 fichiers JSON et zéro erreur de schéma |
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

Aucune suppression physique de données n'est autorisée par ce lot. Une éventuelle
politique de rétention fera l'objet d'une décision séparée après mesure des
doublons exacts et preuve de récupération.

## 12 bis. Lot METH — preuves économiques complémentaires

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `METH-003` | exposer rendement excédentaire annualisé, tracking error, asymétrie et kurtosis dans le moteur commun | fait | formules centralisées dans `portfolio/performance.py`, cas sans benchmark et vide testés ; aucun rendement mensuel modifié |
| `METH-004` | filtrer les prédictions Boosting par l'éligibilité PE point-in-time de Legacy avant classement | fait | registre causal de 88 948 ticker-mois, variantes Top 5/10/15/20 natives et appariées sur le même snapshot, bootstrap de 50 000 tirages et aucune promotion de Boosting |

## 12 ter. Lot REPLAY — relier chaque refresh aux portefeuilles

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `REPLAY-001` | comparer baseline et candidat depuis les entrées jusqu'aux holdings Legacy et Boosting | fait | audit causal au cutoff, preuves Parquet par clé, comparaison du code/config/runtime et code retour bloquant pour tout statut autre que l'identité historique |
| `REPLAY-002` | rendre chaque écart code, configuration et runtime directement explicable | fait | chemins de run neutralisés ; valeurs avant/après listées par chemin JSON pour Git, fichiers critiques, paramètres, dépendances et seeds |
| `REPLAY-003` | conserver un statut machine-lisible pour chaque source après une gate amont | fait | prix Yahoo téléchargés/quarantinés, historiques gelés conservés et acquisitions fondamentales non démarrées distingués explicitement |
| `REPLAY-004` | produire un rapport HTML autonome qui sépare drift prix, SEC, Legacy et Boosting | fait | rapport réel `8881cac6…b971`, payload `1aaac44b…cdb1`, ablations prix/SEC, scores, Top-N, CVC, gate et hashes réunis ; 13 tests ciblés, typage, lint, navigation HTML et absence d'asset externe validés |

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

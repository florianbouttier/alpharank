# Roadmap maître AlphaRank

**Dernière mise à jour : 2026-08-20.**

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

Une tâche `prêt à committer` est implémentée dans le worktree mais n'est pas
`faite` tant que son unique commit n'existe pas.

Transition achevée : les travaux `DOC-001` à `DOC-013` et `QUAL-001`, commencés
avant l'activation du contrat Git, ont été redécoupés puis committés sous leurs
identifiants propres. Aucun commit global de nettoyage ne les a absorbés.

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
| Plus gros fichier | `src/alpharank/data/open_source/ingestion.py`, 3 608 lignes | risque élevé pour toute modification d'ingestion |
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

`DOC-010` a retiré des tests les chemins documentaires historiques. Les contrats
restent directement sous `docs/` pour leurs lecteurs humains et liens publics ;
un classement ultérieur exigera son propre inventaire de lecteurs.

## 7. Lot GIT — publier une histoire lisible en continu

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `GIT-001` | travailler sur `main`, pousser chaque commit et préserver l'ancien `main` sous `main-save` | fait | `main-save` = `c1113ab…`, `main` intégré par fast-forward, publication immédiate inscrite dans les documents normatifs |

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
| `QUAL-009` | définir et valider un schéma versionné pour chaque famille de configuration JSON | fait | 14 familles et 18 fichiers contrôlés récursivement avant les tests |
| `QUAL-010` | corriger la gate CI pour un checkout sans artefacts locaux | fait | `--group ci` statique et huit tests ciblés verts depuis un dépôt temporaire propre |

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

Chaque découpage commence par un test de caractérisation. Aucune valeur de
portefeuille, de KPI ou de sélection ne doit changer dans ce lot.

## 10. Lot TEST — rendre les preuves navigables

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `TESTORG-001` | cataloguer les tests par domaine, niveau, réseau et durée | fait | catalogue courant : 111 fichiers et 421 cas mesurés, dont trois dépendances locales explicites |
| `TESTORG-002` | créer une arborescence `unit`, `integration`, `replay` et `production` sans changer la découverte Pytest | fait | 419 identifiants canoniques identiques avant/après ; réseau isolé sous `integration/network` |
| `TESTORG-003` | découper les deux fichiers de tests de plus de 1 000 lignes | fait | 42 scénarios et 127 assertions préservés bit à bit au niveau AST ; aucun module de test au-dessus de 1 000 lignes |
| `TESTORG-004` | centraliser uniquement les fixtures réellement partagées | fait | une seule fixture racine isole le contexte de run entre suites ; helpers métier et fixtures Pytest restent locaux ou natifs |

## 11. Lot DATA — converger sans retélécharger ni perdre une révision

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `DATA-001` | figer le dictionnaire des emplacements actuels et de leurs lecteurs | fait | 35 fichiers/packages, 58 références statiques et 261 arêtes lecteur-emplacement dans un inventaire machine-lisible |
| `DATA-002` | déclarer `warehouse/raw` comme cible de toutes les sources brutes, EODHD inclus | fait | huit fournisseurs ont une racine RAW unique, des datasets déclarés et des contrats communs de reçu/manifeste |
| `DATA-003` | enregistrer chaque téléchargement par reçu et hash ; réutiliser le payload s'il est identique | fait | reçu immuable par tentative, échecs inclus ; objet adressé par SHA-256 et manifeste fournisseur vérifié |
| `DATA-004` | normaliser uniquement dans `stg` | à faire | aucune règle de préférence fournisseur dans STG |
| `DATA-005` | rendre `def` responsable du choix de valeur et de sa provenance | à faire | une décision expliquée par clé et date de connaissance |
| `DATA-006` | construire les entrées AlphaRank uniquement depuis `mart` | à faire | parité exacte avec un snapshot validé |
| `DATA-007` | définir le snapshot comme publication immuable du mart, pas comme couche concurrente | à faire | manifeste et hashes complets |
| `DATA-008` | migrer les racines historiques par référence/hash avant toute copie | à faire | aucun nouveau téléchargement, aucune clé perdue |
| `DATA-009` | basculer les lecteurs un par un vers les emplacements canoniques | à faire | ancien et nouveau chemins comparés |
| `DATA-010` | rendre les anciennes racines en lecture seule puis les archiver | à faire | période d'observation et procédure de retour arrière |

Aucune suppression physique de données n'est autorisée par ce lot. Une éventuelle
politique de rétention fera l'objet d'une décision séparée après mesure des
doublons exacts et preuve de récupération.

## 12. Lot RUN — remettre de l'ordre dans résultats et journaux

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `RUNORG-001` | inventorier les 346 dossiers racine de `outputs/` par famille, date, statut et taille | à faire | registre consultable sans ouvrir chaque dossier |
| `RUNORG-002` | définir un chemin unique `outputs/<famille>/<run_id>/` | à faire | convention documentée et validée |
| `RUNORG-003` | séparer `candidate`, `validated`, `published` et `failed` dans le manifeste, pas dans des noms libres | à faire | statut explicite de chaque nouveau run |
| `RUNORG-004` | relier chaque journal au manifeste du run | à faire | navigation dans les deux sens |
| `RUNORG-005` | produire des pointeurs `latest` atomiques sans copier les résultats | à faire | cible et hash vérifiés |
| `RUNORG-006` | mesurer les doublons exacts et proposer une rétention réversible | à faire | rapport d'espace, aucune suppression automatique |

## 13. Ordre d'exécution et portes de décision

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

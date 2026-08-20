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
et contrôles historiques. `DOC-010` pourra le renommer en registre après avoir
découplé ces lecteurs ; son rôle est déjà clarifié dès maintenant.

## 2. Priorités actives

| Ordre | TASK-ID maître | Travail | Détail lié | Statut |
| ---: | --- | --- | --- | --- |
| 1 | `GIT-001` | publier chaque commit directement depuis `main` et conserver `main-save` | lot GIT ci-dessous | en cours |
| 2 | `DOC-014` | consolider `AGENTS.md` comme source unique et réduire `AGENT.md` à un pointeur | — | prêt à committer |
| 3 | `DOC-015` | imposer le contrat Git une tâche = un commit documenté | — | prêt à committer |
| 4 | `DOC-009` | découper le catalogue Boosting de 3 899 lignes | lot DOC ci-dessous | à faire |
| 5 | `DOC-010` | découpler les tests des chemins documentaires historiques | lot DOC ci-dessous | à faire |
| 6 | `QUAL-002` | configurer Ruff sans reformatage global | lot QUAL ci-dessous | à faire |
| 7 | `METH-001` | matérialiser la clôture comme convention runtime canonique | `LEG-005` | à faire |
| 8 | `METH-002` | séparer les deux identités SNDK et reconstruire les résultats | `LIVE-022` | à faire — bloque la publication économique |

Une tâche `prêt à committer` est implémentée dans le worktree mais n'est pas
`faite` tant que son unique commit n'existe pas.

Transition : `DOC-001` à `DOC-013` et `QUAL-001` ont été réalisés dans le
worktree avant l'activation du contrat Git, sans commit. Leur statut ci-dessous
est donc corrigé en `prêt à committer`. Ils devront chacun recevoir leur commit
atomique ou être redécoupés avant tout nettoyage de code ; aucun commit global
« documentation cleanup » ne les absorbera silencieusement.

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
| `DOC-001` | inventorier tout le dépôt et mesurer les zones de confusion | prêt à committer | état des lieux ci-dessus |
| `DOC-002` | écrire un point d'entrée unique pour les normes de développement | prêt à committer | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| `DOC-003` | réduire le README racine à un onboarding de cinq minutes | prêt à committer | parcours court et ancien README archivé |
| `DOC-004` | placer les rapports datés, le handoff et les anciennes pages dans `docs/archive/` | prêt à committer | aucun rapport historique mélangé aux contrats courants |
| `DOC-005` | rendre le registre méthodologique visible à la racine sans perdre son historique | prêt à committer | registre détaillé conservé et compatibilité temporaire documentée |
| `DOC-006` | documenter la réalité et la cible des données | prêt à committer | carte du dépôt et cycle de vie des données |
| `DOC-007` | supprimer le premier doublon de rôle entre `AGENT.md` et `AGENTS.md` sans perdre le guide historique | prêt à committer | ancien guide singulier archivé |
| `DOC-008` | vérifier tous les README locaux et liens | prêt à committer | validateur documentaire et tests de structure verts |
| `DOC-009` | découper le catalogue Boosting en synthèse courante et journaux datés | à faire | page courante courte, entrées historiques intactes |
| `DOC-010` | retirer les chemins documentaires de la logique des tests avant le classement final par thème | à faire | les tests valident le contenu, pas un ancien emplacement |
| `DOC-011` | formaliser le standard Python détaillé | prêt à committer | règles et checklist dans `docs/standards/python.md` |
| `DOC-012` | formaliser le standard data détaillé | prêt à committer | contrats et checklist dans `docs/standards/data.md` |
| `DOC-013` | formaliser l'organisation cible et les dépendances autorisées | prêt à committer | règles dans `docs/standards/repository.md` |
| `DOC-014` | consolider les instructions agents dans le seul fichier réellement chargé | prêt à committer | `AGENTS.md` court, ancien contenu archivé, `AGENT.md` simple pointeur |
| `DOC-015` | formaliser le contrat Git tâche/commit/preuves | prêt à committer | `docs/standards/git.md` et liens normatifs |
| `DOC-016` | clarifier roadmap maître et registre méthodologique | fait | une seule liste de priorités actives, historique intégral conservé |

`DOC-010` explique pourquoi plusieurs contrats restent temporairement à la
racine de `docs/` : le code de test verrouille aujourd'hui leurs chemins et
cette passe a reçu l'interdiction explicite de modifier du code.

## 7. Lot GIT — publier une histoire lisible en continu

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `GIT-001` | travailler sur `main`, pousser chaque commit et préserver l'ancien `main` sous `main-save` | en cours | sauvegarde distante vérifiée, `main` intégré sans réécriture et règle inscrite dans les documents normatifs |

## 8. Lot QUAL — rendre les normes contrôlables

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `QUAL-001` | activer les choix de `CONTRIBUTING.md` pour le nouveau code et les nouvelles données | prêt à committer | standard version 1 documenté et indexé |
| `QUAL-002` | configurer Ruff dans `pyproject.toml` sans reformater le dépôt | à faire | configuration et commande documentées |
| `QUAL-003` | établir une baseline des alertes et empêcher seulement les nouvelles régressions | à faire | rapport reproductible, CI différentielle |
| `QUAL-004` | activer le typage progressivement par package | à faire | périmètre mypy explicite et croissant |
| `QUAL-005` | classer les tests en unitaires, intégration, replay, réseau et production | à faire | marqueurs et commandes séparées |
| `QUAL-006` | ajouter les contrôles de documentation, lint et tests ciblés en CI | à faire | contrôles reproductibles localement |
| `QUAL-007` | choisir `pyproject.toml` comme source des dépendances et générer ou vérifier les autres fichiers | à faire | absence de dérive entre les trois installations |
| `QUAL-008` | rendre la CI AlphaRank autonome et isoler le contrôle Portfolio dans un job inter-projets explicite | à faire | AlphaRank validable sans checkout Portfolio |
| `QUAL-009` | définir et valider un schéma versionné pour chaque famille de configuration JSON | à faire | erreurs de clé refusées avant le run |

## 9. Lot CODE — découper sans changer les résultats

Priorité donnée aux composants de production et d'ingestion, pas aux fichiers
les plus faciles à déplacer.

| ID | Action | Statut | Contrôle obligatoire |
| --- | --- | --- | --- |
| `CODE-001` | cartographier les appels entre scripts et bibliothèque | à faire | graphe des entrées actives et lecteurs |
| `CODE-002` | retirer les 35 injections de `sys.path` | à faire | commandes exécutables depuis un autre dossier |
| `CODE-003` | découper `data/open_source/ingestion.py` par étape du pipeline | à faire | parité des sorties et tests d'ingestion |
| `CODE-004` | alléger `scripts/run_legacy.py` en déplaçant la logique testable dans `src/` | à faire | replay Legacy strict inchangé |
| `CODE-005` | découper `strategy/legacy.py` par agrégation, sélection et artefacts | à faire | décisions mensuelles identiques |
| `CODE-006` | découper `governance.py` par contrat de validation | à faire | mêmes refus et mêmes messages structurés |
| `CODE-007` | séparer calcul et rendu dans les dashboards de plus de 1 000 lignes | à faire | données et HTML comparés séparément |
| `CODE-008` | centraliser les commandes de comparaison aujourd'hui dupliquées | à faire | un seul moteur économique partagé |
| `CODE-009` | remplacer les captures générales et `print()` par des erreurs et journaux explicites | à faire | aucun échec silencieux, run_id présent |
| `CODE-010` | déplacer les scripts réellement obsolètes après audit des lecteurs | à faire | zéro import ou appel actif avant archivage |
| `CODE-011` | attribuer ou déplacer les modules transverses `*_v2`, gouvernance et replay dans des packages nommés | à faire | propriétaire et API publique documentés |

Chaque découpage commence par un test de caractérisation. Aucune valeur de
portefeuille, de KPI ou de sélection ne doit changer dans ce lot.

## 10. Lot TEST — rendre les preuves navigables

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `TESTORG-001` | cataloguer les 93 tests par domaine, niveau, réseau et durée | à faire | registre sans déplacement initial |
| `TESTORG-002` | créer une arborescence `unit`, `integration`, `replay` et `production` sans changer la découverte Pytest | à faire | même liste de tests collectés avant/après |
| `TESTORG-003` | découper les deux fichiers de tests de plus de 1 000 lignes | à faire | mêmes scénarios et assertions |
| `TESTORG-004` | centraliser uniquement les fixtures réellement partagées | à faire | dépendances de fixture lisibles et locales par défaut |

## 11. Lot DATA — converger sans retélécharger ni perdre une révision

| ID | Action | Statut | Critère de fin |
| --- | --- | --- | --- |
| `DATA-001` | figer le dictionnaire des emplacements actuels et de leurs lecteurs | à faire | inventaire machine-lisible par fichier/package |
| `DATA-002` | déclarer `warehouse/raw` comme cible de toutes les sources brutes, EODHD inclus | à faire | contrat par fournisseur et manifestes |
| `DATA-003` | enregistrer chaque téléchargement par reçu et hash ; réutiliser le payload s'il est identique | à faire | tentative tracée sans duplication physique inutile |
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

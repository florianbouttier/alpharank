# Normes de développement AlphaRank

**Statut : norme active pour tout nouveau code et toute nouvelle donnée — version 1.0, 20 août 2026.**

Ce document est le point d'entrée. Les règles détaillées sont séparées pour
rester consultables :

- [`docs/standards/python.md`](docs/standards/python.md) : style Python, API,
  typage, erreurs, calcul tabulaire et tests ;
- [`docs/standards/data.md`](docs/standards/data.md) : contrats des couches,
  noms, clés, temps, lignée, qualité et publication ;
- [`docs/standards/repository.md`](docs/standards/repository.md) : organisation
  du dépôt, dépendances entre packages, scripts, tests, configurations et
  sorties.
- [`docs/standards/git.md`](docs/standards/git.md) : relation stricte entre
  tâche de roadmap, commit, preuves, staging et publication.

Le dépôt existant ne respecte pas encore intégralement ces règles. Elles
s'appliquent immédiatement au code nouveau et aux lignes substantiellement
modifiées. La dette existante est migrée tâche par tâche selon
[`ROADMAP.md`](ROADMAP.md), sans reformatage massif et avec un commit distinct
par tâche.

### Niveaux d'obligation

- **OBLIGATOIRE** : aucune nouvelle violation ; une exception bloque la revue.
- **RECOMMANDÉ** : règle normale, écart accepté seulement avec une raison écrite.
- **AUTORISÉ** : choix possible dans le cadre indiqué.
- **INTERDIT** : ne doit pas apparaître dans du code ou une donnée nouvelle.

En cas de divergence, le document détaillé du domaine prévaut sur le résumé de
ce fichier. Une exception durable est enregistrée dans la roadmap ou dans une
décision d'architecture ; elle n'est pas cachée dans un commentaire.

## 1. Principes non négociables

1. Un emplacement a une seule responsabilité compréhensible.
2. Une règle métier n'a qu'une seule implémentation active.
3. Un script assemble des composants ; il ne contient pas un second moteur
   métier.
4. Toute décision temporelle ou financière est explicite, testable et présente
   dans le manifeste du run.
5. Une donnée brute n'est jamais réécrite silencieusement.
6. Une erreur de couverture ne doit pas être transformée en résultat plausible
   par un repli implicite.
7. Une expérience reste séparée de la production tant qu'elle n'a pas franchi
   ses contrôles de promotion.
8. Une documentation courante décrit l'état courant ; une preuve datée va dans
   les archives.

## 2. Langue et vocabulaire

- Les noms Python, clés de manifeste, arguments de commande et messages
  techniques restent en anglais.
- Les documents d'onboarding et d'exploitation destinés au propriétaire du
  projet sont rédigés en français clair. Les termes anglais imposés par une API
  peuvent être conservés et expliqués à leur première apparition.
- Les dates utilisent le format ISO `YYYY-MM-DD` et les heures incluent le
  fuseau lorsqu'elles ont une portée opérationnelle.
- Les mots `Legacy`, `Boosting`, `raw`, `stg`, `def`, `mart`, `snapshot` et
  `run` gardent une définition unique, donnée dans
  [`docs/architecture/data_lifecycle.md`](docs/architecture/data_lifecycle.md).

## 3. Organisation cible du dépôt

```text
alpharank/
├── README.md                 # onboarding humain en cinq minutes
├── ROADMAP.md                # travaux actifs de remise en ordre
├── CONTRIBUTING.md           # normes de développement
├── METHODOLOGY_AUDIT_ROADMAP.md
├── AGENTS.md                 # contraintes destinées aux agents
├── configs/                  # décisions versionnées, sans logique Python
├── data/                     # données locales et manifestes de lignée
├── docs/                     # contrats courants, recherche et archives
├── scripts/                  # commandes minces et orchestration
├── src/alpharank/            # seule bibliothèque Python active
├── tests/                    # tests organisés par niveau et contrat
├── outputs/                  # résultats de runs, jamais du code source
└── logs/                     # journaux d'exécution
```

### Responsabilité des dossiers

- `src/alpharank/` contient toute logique réutilisable.
- `scripts/` contient les interfaces en ligne de commande et l'orchestration.
- `tests/` ne contient ni donnée de production ni logique appelée en
  production.
- `configs/` contient des choix versionnés et relus ; jamais des sorties de
  run.
- `docs/` ne contient pas de copie d'un résultat volumineux lorsque le manifeste
  du run suffit à le retrouver.
- `outputs/` et `logs/` sont générés. Ils doivent être indexés par des manifestes
  mais ne sont pas importables par le code.

Chaque dossier maintenu possède un `README.md` court avec exactement cinq
rubriques : **responsabilité**, **entrées**, **sorties**, **dossiers enfants** et
**ce qui est interdit ici**. Les dossiers générés n'ont pas besoin d'un README
par run ; leur dossier parent explique le contrat commun.

## 4. Style Python proposé

| Sujet | Règle proposée |
| --- | --- |
| Versions | conserver `Python >= 3.10` tant qu'une migration dédiée n'est pas décidée |
| Formatage | `ruff format`, longueur de ligne 100 |
| Contrôle statique | `ruff check`, règles activées progressivement et inscrites dans `pyproject.toml` |
| Typage | annotations obligatoires sur toute API publique ; `mypy` renforcé package par package |
| Imports | imports absolus `alpharank.*` ; aucune modification de `sys.path` |
| Chemins | `pathlib.Path`, jamais de chemin de production codé en dur |
| Noms | modules/fonctions en `snake_case`, classes en `PascalCase`, constantes en `UPPER_SNAKE_CASE` |
| Guillemets | choix laissé au formateur ; aucune retouche manuelle purement cosmétique |
| Docstrings | contrat, unités, calendrier et erreurs d'une API publique ; pas de paraphrase du code |
| Commentaires | expliquer le pourquoi, le risque causal ou la provenance, pas la syntaxe |

### Taille et complexité

Ces seuils sont des garde-fous, pas une invitation à découper artificiellement :

- fonction : cible de 50 lignes, revue obligatoire au-delà de 80 ;
- module de bibliothèque : cible de 500 lignes, plan de découpage obligatoire
  au-delà de 800 ;
- script : cible de 250 lignes ; au-delà, déplacer la logique dans
  `src/alpharank/` ;
- classe : une responsabilité métier et pas de méthode cachant un pipeline
  complet ;
- pas de nouveau fichier de plus de 800 lignes sans justification écrite dans
  la revue.

Les fichiers existants qui dépassent ces seuils ne seront pas coupés en bloc.
Ils seront traités par risque et couverts par des tests de comportement avant
tout déplacement.

## 5. Interfaces et types

- Une fonction publique indique les types, unités et conventions temporelles de
  ses entrées et sorties.
- Les manifestes stables utilisent des structures nommées et versionnées
  (`dataclass`, `TypedDict` ou modèle équivalent), pas des dictionnaires dont les
  clés apparaissent au fil du code.
- Un schéma de données possède une version et un validateur à sa frontière.
- Une colonne monétaire précise sa devise ; un rendement précise s'il est brut
  ou net et sa période ; un prix précise `open`, `close`, `adjusted_close` ou une
  autre convention exacte.
- Les fonctions de calcul ne lisent pas l'heure courante ni un pointeur
  `latest` caché. L'instant, le snapshot et la configuration sont injectés.
- Les états métiers utilisent des valeurs explicites plutôt qu'un booléen
  ambigu, par exemple `evaluable`, `horizon_pending` ou
  `ticker_target_unavailable`.

## 6. Erreurs, replis et journaux

- Attraper une exception précise. `except Exception` n'est acceptable qu'à une
  frontière de processus, avec contexte, journalisation et nouvel échec ou
  statut explicite.
- Aucun `except:` nu dans du code maintenu.
- Aucun `print()` dans la bibliothèque. Les scripts utilisent un journal avec
  niveau et contexte ; la sortie utilisateur reste courte.
- Tout journal de production porte au minimum `run_id`, `snapshot_id`, étape et
  résultat.
- Un repli vers une autre source, une autre date ou une ancienne valeur est une
  décision métier. Il doit être nommé, enregistré et testé. Il n'est jamais
  déclenché silencieusement parce qu'une valeur manque.
- Une validation de production échoue fermement lorsque la provenance, le
  calendrier ou le schéma ne sont pas démontrés.

## 7. Séparation données / calcul / publication

Le contrat cible est :

```text
raw -> stg -> def -> mart -> snapshot publié -> run -> rapport
```

- `raw` conserve ce que la source a réellement fourni et l'historique de ses
  changements.
- `stg` normalise les noms, types, identifiants et calendriers sans arbitrer la
  valeur économique.
- `def` choisit la valeur retenue pour une clé et explique ce choix.
- `mart` prépare un jeu de données pour un consommateur précis.
- un `snapshot` fige un mart, sa configuration, ses sources et ses hashes ; ce
  n'est pas une cinquième transformation.
- un `run` consomme un snapshot immuable et produit des résultats dans
  `outputs/`.

Les règles détaillées, y compris le rôle d'EODHD et des répertoires historiques,
sont dans [`docs/architecture/data_lifecycle.md`](docs/architecture/data_lifecycle.md).

## 8. Scripts et commandes

- Une commande expose `main()` et peut être appelée sans effet de bord lors de
  son import.
- L'analyse des arguments, l'appel des services et l'écriture finale restent
  dans le script ; les calculs vivent dans la bibliothèque.
- Toute commande qui publie ou promeut une donnée propose d'abord un mode de
  vérification sans mutation.
- Une commande de production affiche les sources résolues avant de commencer
  et écrit un manifeste même en cas d'échec contrôlé.
- Les commandes actives sont listées dans `scripts/README.md`. Une expérience
  ponctuelle va sous `scripts/experiments/`; un script abandonné va dans
  `_old/` seulement après preuve qu'aucun workflow actif ne l'appelle.

## 9. Tests

- Nom : `test_<comportement>_<condition>_<résultat>()`, sans reprendre seulement
  le nom de la fonction.
- Un correctif de bug commence par un test qui reproduit le défaut.
- Les tests unitaires n'accèdent ni au réseau ni aux pointeurs de production.
- Les tests d'intégration utilisent des fixtures minimales et explicites.
- Les tests de replay comparent les hashes, le calendrier, le snapshot et les
  sorties économiques ; ils ne se limitent pas à vérifier qu'un fichier existe.
- Les tests de causalité mutent volontairement le futur et vérifient que le
  passé ne change pas.
- Les tests lents, réseau et production sont identifiés séparément. Une suite
  standard ne doit pas les exécuter par accident.
- Une modification purement documentaire exécute au minimum le validateur de
  liens et de structure documentaire.

## 10. Documentation

Chaque document commence par son rôle :

- **canonique** : règle active et unique ;
- **runbook** : procédure opératoire ;
- **roadmap** : tâches et preuves attendues ;
- **rapport daté** : observation historique non normative ;
- **archive** : conservée pour la traçabilité.

Une performance publiée cite toujours le run, le snapshot, le commit, la
période demandée et effective, la convention d'achat, les frais et le benchmark.
Un document daté ne doit jamais redevenir implicitement « courant » parce qu'il
reste proche de la racine.

## 11. Tâches, branches et commits

Le contrat complet est dans [`docs/standards/git.md`](docs/standards/git.md).

- Une tâche de roadmap a un identifiant stable, par exemple `CODE-003`.
- Relation stricte : une tâche donne exactement un commit et un commit ne traite
  qu'une tâche. Si le changement est trop grand, découper la roadmap avant de
  coder.
- Un commit référence la tâche :
  `refactor(CODE-003): remove runtime path injection`.
- Le commit inclut implémentation, test ou preuve, documentation et passage de
  la tâche à `fait`.
- Le dépôt étant maintenu par une seule personne, la branche normale de travail
  et de publication est `main`. Une tâche autorisée est committée sur `main`,
  puis poussée immédiatement avec `git push origin main` ; aucune confirmation
  de push supplémentaire n'est attendue.
- `main-save` conserve l'état de `main` antérieur à l'intégration du 20 août
  2026, au commit `c1113ab0613c06c8e3deb27e7a7f35d892e80bca`. Cette branche
  est une sauvegarde immuable, pas une seconde branche de développement.
- Avant le commit, vérifier les références distantes. Après le push, vérifier
  que les hashes de `main` et `origin/main` sont identiques. En cas de
  divergence, ne jamais utiliser un force-push pour la masquer.
- Une remise en forme globale, un déplacement et un changement métier ne sont
  jamais mélangés dans le même commit.
- Les artefacts générés, données brutes, secrets, caches et gros résultats ne
  sont pas ajoutés à Git sans demande explicite.
- Aucun historique n'est réécrit pour rendre une ancienne organisation plus
  jolie.

## 12. Contrôles proposés avant publication

Après adoption, les contrôles minimaux seront :

1. formatage vérifié sans réécriture automatique en CI ;
2. lint sur les fichiers modifiés, puis extension progressive au dépôt ;
3. typage sur les packages déjà migrés ;
4. tests ciblés selon le périmètre ;
5. validation de la documentation ;
6. contrôles de replay et de lignée pour toute modification de production.

L'activation de ces contrôles est volontairement reportée aux tâches `QUAL-*`
de la roadmap. Aucun outil ne doit reformater d'un coup le dépôt actuel : cela
rendrait les changements métier impossibles à relire.

## 13. Décision active et migration

La norme retenue est donc :

- garder Python 3.10 comme compatibilité minimale pour l'instant ;
- adopter Ruff avec une ligne de 100 caractères ;
- introduire le typage et les seuils de taille progressivement ;
- interdire immédiatement dans le nouveau code les injections de `sys.path`,
  les replis silencieux et la logique métier dans les scripts ;
- conserver le contrat `raw -> stg -> def -> mart`, où `def` reste nécessaire
  pour arbitrer et expliquer les valeurs ;
- appliquer « une tâche, un commit, une preuve » à toute la remise en ordre.

Ces règles sont plus précisément définies dans
[`docs/standards/`](docs/standards/README.md). Leur automatisation progressive
est suivie par les tâches `QUAL-*` ; l'absence temporaire d'un contrôle Ruff ou
mypy ne suspend pas la règle pour le nouveau code.

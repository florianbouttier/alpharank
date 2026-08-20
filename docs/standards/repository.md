# Standard d'organisation du dépôt AlphaRank

**Rôle : norme de placement, propriété et dépendances entre dossiers.**

L'objectif n'est pas d'obtenir une arborescence esthétique. Il est de permettre
à un humain de répondre rapidement à trois questions : où modifier une règle,
qui peut dépendre de quoi, et où retrouver la preuve d'un run.

## 1. Racine du dépôt

La racine est une page d'accueil, pas un espace de stockage.

### Fichiers autorisés

- `README.md` : onboarding court ;
- `ROADMAP.md` et roadmaps majeures explicitement nommées ;
- `CONTRIBUTING.md` et l'unique guide agent `AGENTS.md` ;
- `pyproject.toml`, fichier d'environnement et fichiers de dépendances ;
- `.gitignore`, `.editorconfig` et configurations d'outils standard ;
- fichiers légaux éventuels.

### Dossiers autorisés

`configs/`, `data/`, `docs/`, `logs/`, `outputs/`, `scripts/`, `src/`, `tests/`
et dossiers techniques cachés nécessaires à Git ou à l'éditeur.

### Interdit à la racine

- rapport daté ;
- dataset ou export ;
- script ponctuel ;
- notebook ;
- HTML de dashboard ;
- fichier `final`, `new`, `copy`, `tmp` ;
- documentation d'une seule expérience ;
- environnement virtuel destiné à être versionné.

## 2. Arborescence cible

```text
alpharank/
├── README.md
├── ROADMAP.md
├── CONTRIBUTING.md
├── AGENTS.md
├── configs/
│   ├── data_contracts/
│   ├── data_quality/
│   └── research/
├── data/
│   ├── warehouse/{raw,stg,def,mart}/
│   ├── model_inputs/{history,manifests}/
│   └── legacy/                 # cible transitoire explicite
├── docs/
│   ├── architecture/
│   ├── standards/
│   ├── research/
│   └── archive/
├── scripts/
│   ├── production/
│   ├── data/
│   ├── research/
│   ├── validation/
│   └── maintenance/
├── src/alpharank/
│   ├── data/
│   ├── features/
│   ├── signals/{legacy,boosting}/
│   ├── portfolio/
│   ├── reporting/
│   └── governance/
├── tests/{unit,integration,replay,production}/
├── outputs/
└── logs/
```

Cette arborescence est une cible. Les noms actuels `strategy/`, `multihorizon/`
et `backtest/` ne sont pas déplacés avant la cartographie `CODE-001` et des tests
de parité. Aucun grand renommage n'est implicite dans ce document.

## 3. Dépendances autorisées

Le sens général est :

```text
data -> features -> signals
                  portfolio
signals --------> portfolio adapters
portfolio ------> reporting
data/portfolio -> governance validators
scripts --------> tous les packages publics nécessaires
```

Règles :

- `data` ne dépend pas des signaux, du portefeuille ou des rapports.
- `features` dépend des contrats data, pas des scripts ni des rapports.
- Legacy et Boosting peuvent dépendre de `data` et `features`, jamais l'un de
  l'autre.
- `portfolio` reste méthodologiquement neutre et ne dépend pas de l'algorithme
  Legacy ou Boosting ; des adaptateurs convertissent leurs décisions.
- `reporting` consomme des artefacts publics ; il ne recalcule pas les KPI ni les
  portefeuilles.
- `governance` valide des contrats ; il ne devient pas une seconde
  implémentation des calculs.
- `scripts` est la couche la plus haute et peut orchestrer ; aucun package ne
  l'importe.
- `tests` peut importer le code ; le code ne peut jamais importer `tests`.

Une dépendance circulaire est bloquante. La résoudre par un import local caché
ou par `TYPE_CHECKING` sans corriger la frontière n'est pas acceptable.

## 4. Organisation de `src/alpharank`

### `data/`

Propriété : contrats, providers, ingestion, transformations d'entrepôt, lignée
et publication.

Cible interne :

```text
data/
├── contracts/       schémas et types communs
├── sources/         un adaptateur par fournisseur
├── warehouse/       transformations stg/def/mart
├── lineage/         hashes, manifestes, provenance
├── quality/         contrôles sans politique de modèle
└── publishing/      snapshots et promotion atomique
```

Le mot `open_source` peut décrire une famille de fournisseurs, mais ne doit pas
rester le propriétaire de toute l'ingestion, de la SEC, des prix, des audits et
de la publication dans un même module.

### `features/`

Transformations réutilisables et causales à partir de données gouvernées. Une
feature expose son calendrier, ses colonnes d'entrée, son type de sortie et ses
besoins de warm-up.

### `signals/legacy/` et `signals/boosting/`

Génération des scores et décisions avant simulation. Les deux packages ne
partagent pas un calcul uniquement pour forcer leur rapprochement ; seuls les
primitives réellement communes vont dans `features` ou un contrat neutre.

### `portfolio/`

Holdings, pondération, exécution, coûts, rendements, benchmark, attribution et
KPI communs. Toute nouvelle implémentation locale de CAGR, Sharpe, drawdown,
turnover ou attribution est interdite.

### `reporting/`

Préparation de modèles de vue et rendu HTML/JSON. Le calcul économique est déjà
terminé avant l'entrée dans ce package.

### `governance/`

Validations de snapshot, calendrier, runtime, promotion et replay. Une règle
volumineuse est placée dans un module nommé par contrat plutôt que dans un unique
`governance.py`.

### `utils/`

`utils` n'est pas un dossier d'attente. Un utilitaire qui n'a qu'un consommateur
reste dans son domaine. Un module commun est accepté seulement s'il est :

- sans dépendance métier inverse ;
- utilisé par au moins deux domaines ;
- nommé précisément (`hashing.py`, `paths.py`), pas `helpers.py` ;
- couvert par des tests propres.

## 5. Modules et API publiques

- Chaque package expose son API dans `__init__.py` avec une liste courte et
  intentionnelle.
- Un consommateur n'importe pas un sous-module privé d'un autre domaine.
- Préfixe `_` pour une implémentation privée au package.
- Un module porte le nom de son contrat principal, pas celui d'une personne ou
  d'une date.
- Un fichier ne combine pas provider réseau, transformation, modèle, simulation
  et rendu.
- Les types partagés vivent au niveau le plus bas qui les possède, pas dans un
  module `common_v2.py` générique.
- Une nouvelle version incompatible coexiste par protocole/configuration ou
  package explicitement versionné pendant la migration, puis l'ancienne est
  dépréciée ; ne pas accumuler `v2`, `v3`, `final` dans les noms.

## 6. Organisation de `scripts`

### Points d'entrée publics

Les commandes réellement utilisées par un humain peuvent conserver un petit
wrapper stable directement sous `scripts/`, par exemple `run_legacy.py`. La
liste est explicitement maintenue dans `scripts/README.md`.

### Catégories cibles

| Dossier | Contenu |
| --- | --- |
| `production/` | lancements mensuels et publication |
| `data/` | ingestion, reconstruction et migration |
| `research/` | expériences reproductibles, jamais production implicite |
| `validation/` | contrôles sans mutation |
| `maintenance/` | compaction, indexation et opérations réversibles |
| `_archive/` | scripts sans lecteur actif, avec provenance et date d'archivage |

Règles :

- maximum 250 lignes ;
- pas d'import depuis un autre script ; extraire une bibliothèque ;
- aucun `sys.path` ;
- `main()` sans effet à l'import ;
- arguments et artefacts documentés ;
- commande destructive séparée d'une commande de diagnostic ;
- une expérience promue devient un service/package testé avant d'être utilisée
  en production.

## 7. Organisation de `tests`

```text
tests/
├── unit/          fonctions et classes isolées, sans réseau
├── integration/   plusieurs composants et petits fichiers contrôlés
├── replay/        snapshots immuables, lignée et parité économique
├── production/    contrôles lents ou dépendants d'un package de production
└── fixtures/      petites fixtures communes documentées
```

- L'arborescence miroir le domaine testé, pas le nom du développeur.
- Les fichiers ne dépassent pas 500 lignes sans plan de découpage.
- Les fixtures propres à un domaine restent proches de ses tests.
- Le réseau est marqué et désactivé par défaut.
- Les tests production ne sont pas nécessaires à chaque boucle locale mais sont
  obligatoires avant une promotion concernée.
- Le déplacement des 93 tests actuels conserve exactement la collecte Pytest et
  se fait indépendamment d'un changement de comportement.

## 8. Organisation de `configs`

- Une config modifie un comportement ; elle n'est ni une donnée source ni un
  résultat.
- Format lisible, schéma versionné et clés inconnues refusées.
- Un fichier par décision cohérente, avec `policy_id`, version, description,
  date d'effet et provenance lorsque nécessaire.
- Une configuration déjà consommée par un run publié est immuable. Toute
  modification crée une nouvelle version.
- Les secrets et chemins locaux sont interdits.
- `data_contracts/` décrit les schémas ; `data_quality/` contient des décisions
  sourcées ; `research/` contient des profils d'expérience figés.

## 9. Organisation de `data`

La norme détaillée est [`data.md`](data.md). Physiquement :

- `warehouse/raw`, `stg`, `def`, `mart` sont les couches de transformation ;
- `model_inputs/history` contient les snapshots immuables publiés ;
- `model_inputs/manifests/latest.json` est le pointeur courant ;
- les anciennes racines sont inventoriées puis regroupées sous une zone legacy
  explicite seulement après bascule des lecteurs ;
- aucun déplacement de données ne dépend d'un nouveau téléchargement ;
- les caches restent séparés et jetables.

Le code source ne vit jamais sous `data/`. Un README et les manifestes peuvent y
être versionnés ; les payloads volumineux restent hors Git.

## 10. Organisation de `outputs` et `logs`

Cible :

```text
outputs/<family>/<run_id>/
logs/<family>/<run_id>.log
```

Chaque run contient :

- `manifest.json` ;
- configuration résolue ;
- références/hashes d'entrée ;
- statut ;
- résultats structurés ;
- rapports dérivés ;
- validation et erreurs éventuelles.

Le statut appartient au manifeste (`candidate`, `validated`, `published`,
`failed`, `quarantined`), pas à un suffixe libre du dossier. Les pointeurs
`latest` sont petits, atomiques et vérifiables.

Un dashboard lit un manifeste ou un index de runs. Il ne parcourt pas 346
dossiers en devinant lequel est courant.

## 11. Organisation de `docs`

| Zone | Rôle |
| --- | --- |
| `docs/architecture/` | responsabilités et flux durables |
| `docs/standards/` | règles de développement |
| `docs/research/` | expériences reproductibles et catalogue R&D |
| `docs/archive/` | rapports et journaux historiques non normatifs |
| racine de `docs/` | contrats courants temporairement path-coupled |

Après `DOC-010`, les contrats courants pourront être classés par thème sans que
les tests connaissent leurs chemins. Un document daté ne revient jamais à la
racine pour gagner en visibilité.

## 12. README local obligatoire

Chaque dossier maintenu comporte un README court avec :

1. **Responsabilité** : une phrase ;
2. **Entrées** : APIs, tables ou fichiers reçus ;
3. **Sorties** : APIs, tables ou artefacts produits ;
4. **Dossiers enfants** : rôle de chacun ;
5. **Interdit ici** : frontière d'ownership.

Le README pointe vers le contrat canonique et n'en recopie pas plusieurs pages.
Les dossiers générés par run sont couverts par le README de leur parent et leur
manifeste.

## 13. Notebooks et exploration

- Notebook autorisé pour exploration reproductible, jamais comme seule
  implémentation d'une règle publiée.
- Entrées et seed figées ; aucune cellule dépendante d'un état caché.
- Le code promu quitte le notebook, entre dans `src/` et reçoit des tests.
- Les gros outputs de cellule ne sont pas versionnés.
- Les conclusions sont enregistrées dans le catalogue de recherche avec run et
  snapshot.

## 14. Dépréciation et archive

Avant d'archiver un module, script, config ou chemin :

1. inventorier imports, appels, docs, launch agents et CI ;
2. annoncer le remplacement et sa date ;
3. faire passer les consommateurs ;
4. démontrer la parité ou documenter la différence ;
5. conserver un chemin de retour arrière ;
6. archiver dans un commit distinct ;
7. supprimer plus tard seulement sur décision explicite.

Un dossier `_old` sans inventaire ne constitue pas une politique de
dépréciation.

## 15. Dépendances et environnements

- `pyproject.toml` devient la source canonique des dépendances Python.
- `requirements.txt` et `environment.yml` sont générés ou vérifiés contre cette
  source ; ils ne divergent pas manuellement.
- Dépendances runtime séparées des outils de développement et des extras
  réellement optionnels.
- Version Python identique en local documenté et CI principale.
- Environnement virtuel, caches et outils téléchargés ne sont pas versionnés.

## 16. Commits et déplacements

Le contrat normatif complet est [`git.md`](git.md).

- Une tâche de roadmap par commit, identifiant dans le message.
- Un commit de déplacement ne change ni calcul ni formatage.
- Un commit de formatage ne change pas le comportement.
- Un changement métier inclut ses tests et sa documentation, mais pas un
  rangement sans rapport.
- Un gros déplacement fournit une table ancien chemin -> nouveau chemin et la
  preuve que les lecteurs ont été mis à jour.
- Aucun fichier historique ou dataset supprimé pendant une migration
  d'organisation sans décision dédiée.

## 17. Checklist d'organisation

- [ ] une seule responsabilité et un propriétaire clair ;
- [ ] dépendance dans le sens autorisé ;
- [ ] aucune logique importée depuis `scripts` ou `tests` ;
- [ ] API publique courte et imports privés évités ;
- [ ] README local à jour ;
- [ ] nom durable sans `final/new/v2` opportuniste ;
- [ ] test dans le bon niveau ;
- [ ] config, donnée, output et documentation dans des zones distinctes ;
- [ ] déplacement isolé d'un changement économique ;
- [ ] roadmap, liens et validateurs mis à jour.

# Carte du dépôt AlphaRank

**Rôle : document canonique d'architecture du dépôt.**

**État observé : 2026-08-20.**

## 1. Ce qu'il faut utiliser aujourd'hui

Pour un run mensuel de production, la source de données courante ne se choisit
pas en parcourant les dossiers. Elle est désignée par :

```text
data/model_inputs/manifests/latest.json
```

Ce pointeur mène à un snapshot immuable. `data/open_source/output/`,
`data/sec/output/`, les Parquet directement sous `data/` et les nombreux
dossiers sous `outputs/` ne doivent pas être choisis comme remplacements au gré
des fichiers disponibles.

## 2. Carte simple

```text
README.md / ROADMAP.md / CONTRIBUTING.md
    orientation humaine, travaux et normes

configs/
    décisions versionnées qui modifient un run

src/alpharank/
    logique Python réutilisable
    ├── strategy/       signal Legacy
    ├── multihorizon/   signal Boosting actuel
    ├── portfolio/      simulation et KPI partagés
    ├── data/           chargement, lignée et ingestion
    ├── replay/         contrats causaux et replays recalculables
    ├── governance_contracts/ validations de promotion et provenance
    └── backtest/       pipeline de recherche historique

scripts/
    commandes et orchestration ; pas une seconde bibliothèque

data/
    données sources, transformations et snapshots immuables

outputs/
    résultats de calcul et rapports de runs

logs/
    journaux d'exécution reliés aux runs

docs/
    règles courantes, recherche reproductible et archives séparées
```

## 3. Pourquoi `data/` semble incohérent

Trois générations se superposent actuellement :

| Génération | Emplacements principaux | Rôle réel aujourd'hui |
| --- | --- | --- |
| historique | fichiers `data/*.parquet`, `data/eodhd/`, `data/_snapshots/` | anciennes interfaces et archive EODHD, encore lues par certains replays |
| migration open source | `data/open_source/`, `data/sec/` | acquisition, audits et anciens packages publiés |
| entrepôt cible | `data/warehouse/raw`, `stg`, `def`, `mart` | structure destinée à devenir l'unique trajet de transformation |

À cela s'ajoute `data/model_inputs/`, qui ne représente pas une quatrième
méthode de transformation : il conserve les **releases immuables** effectivement
consommées par les modèles.

Le problème n'est donc pas l'existence de `raw/stg/def/mart`. Le problème est
que la migration n'est pas achevée et qu'aucune carte visible n'expliquait les
anciennes branches encore nécessaires.

L'état observé détaillé est figé dans
[`data_location_inventory_v1.json`](data_location_inventory_v1.json) : chaque
fichier ou package y porte son rôle, sa cible, ses volumes et ses lecteurs de
code actifs.

## 4. Responsabilité des grandes zones

| Chemin | Contient | Ne doit pas contenir |
| --- | --- | --- |
| `configs/` | exclusions, événements sourcés, profils de recherche | sorties calculées ou secrets |
| `src/alpharank/` | calculs, contrats, validateurs réutilisables | arguments de commande ou chemins locaux implicites |
| `scripts/` | lecture d'arguments, orchestration, écriture d'artefacts | moteurs de calcul dupliqués |
| `tests/` | preuves de comportement et de replay | logique importée en production |
| `data/` | données et manifestes de lignée | rapports HTML ou code Python métier |
| `outputs/` | artefacts d'un run identifié | source de vérité d'entrée non manifestée |
| `logs/` | événements d'exécution | résultats économiques canoniques |
| `docs/` | règles, procédures, décisions et archives | copie manuelle d'une donnée destinée au modèle |

## 5. Source, snapshot et run ne sont pas synonymes

```text
source fournisseur
        |
        v
raw -> stg -> def -> mart
                       |
                       v
             snapshot immuable
                       |
                       v
                 run de modèle
                       |
                       v
                rapport / site
```

- La **source** est l'observation reçue d'un fournisseur.
- Le **mart** est une table prête pour un besoin défini.
- Le **snapshot** fige un ensemble de marts avec leurs preuves.
- Le **run** est un calcul qui consomme ce snapshot.
- Le **rapport** expose les résultats du run ; il ne devient pas une source de
  données par sa simple présence dans `outputs/`.

## 6. Cible de rangement

La cible n'est pas de déplacer brutalement 65 Go. Elle est de faire converger
les lecteurs :

1. cataloguer chaque ancien emplacement et ses consommateurs ;
2. référencer ou importer son contenu dans la couche correcte sans nouveau
   téléchargement ;
3. démontrer l'identité par clés, valeurs et hashes ;
4. basculer un lecteur à la fois ;
5. conserver l'ancien emplacement en lecture seule pendant une période
   d'observation ;
6. décider séparément d'une éventuelle déduplication physique.

Le registre
[`data_reader_migration_v1.json`](data_reader_migration_v1.json) applique cette
règle aux lecteurs Legacy actifs. Les commandes `run_legacy.py` et
`run_backtest.py` résolvent désormais le MART canonique par défaut. Un replay,
un audit ou une transition qui choisit une ancienne source doit encore la
nommer explicitement : les hashes prouvent que huit des neuf fichiers modèle
Legacy ne sont pas identiques au MART courant, donc une substitution silencieuse
modifierait les données économiques.

Les anciennes racines cataloguées sont gelées pour le code gouverné par
[`legacy_data_archive_policy_v1.json`](legacy_data_archive_policy_v1.json).
La période d'observation court du 20 août au 19 septembre 2026. L'archive reste
une référence vers les octets hashés : aucune permission système n'est modifiée,
aucun payload n'est déplacé et aucune suppression automatique n'est autorisée.
Le même contrat décrit le retour arrière lecteur par lecteur si un replay
explicite ne peut plus résoudre sa source.

Les 346 dossiers directement sous `outputs/` restent des résultats historiques,
mais ils sont maintenant consultables dans
[`run_root_inventory_v1.json`](run_root_inventory_v1.json) par famille, date,
statut manifesté et volume. Le registre n'invente pas un statut à partir d'un
nom libre : l'absence de manifeste conforme reste `legacy_unclassified`.

Tout nouveau résultat suit le contrat
[`../run_organization.md`](../run_organization.md) : exactement
`outputs/<famille>/<run_id>/`, avec une famille `lower_snake_case` et un
identifiant UTC immuable `YYYYMMDDTHHMMSSZ_<slug>`. Les racines historiques ne
sont pas déplacées implicitement.

Le statut d'un nouveau run vit uniquement dans son `manifest.json`. Il commence
à `candidate`, conserve chaque transition et ne peut atteindre `published`
qu'après `validated`. Un suffixe libre dans le nom n'a aucun effet de promotion.

Chaque nouveau journal suit `logs/<famille>/<run_id>/*.log`. Son hash et son
sidecar figurent dans le manifeste du run ; le sidecar renvoie vers ce même
manifeste. Les 74 journaux historiques ne sont pas appariés par supposition.

Le `latest.json` d'une famille référence uniquement un run `published`, avec le
hash du manifeste et de tout l'arbre. Son remplacement est atomique et ne copie
aucun résultat ; la version immuable du pointeur permet de retrouver la cible.

La rétention reste une proposition : le rapport
[`run_retention_report_v1.json`](run_retention_report_v1.json) mesure uniquement
les doublons SHA-256 exacts, désigne une source de récupération et interdit
toute suppression automatique.

Le détail et les tâches sont dans [`../../ROADMAP.md`](../../ROADMAP.md). Le
contrat des couches est dans [`data_lifecycle.md`](data_lifecycle.md) et les
règles de placement sont dans
[`../standards/repository.md`](../standards/repository.md).

## 7. Graphe exécutable et lecteurs

Le registre
[`code_dependency_inventory_v1.json`](code_dependency_inventory_v1.json)
énumère les fichiers Python suivis, leurs imports internes, les commandes de
scripts détectables statiquement et les lecteurs inverses. Il distingue les
entrées actives de `scripts/_archive/` et fige les six commandes publiques de
`scripts/README.md`. L'audit de déplacement et les hashes antérieurs sont
conservés dans
[`script_archival_audit_v1.json`](script_archival_audit_v1.json). Toute
modification du graphe doit régénérer ce registre explicitement et rester
contrôlée par la CI.

Les quelques modules encore directement sous `src/alpharank/` sont attribués
dans [`root_module_ownership_v1.json`](root_module_ownership_v1.json). Les six
anciens noms de replay y sont déclarés comme façades ; les implémentations et
l'API canonique vivent sous `alpharank.replay`.

Le registre [`test_catalog_v1.json`](test_catalog_v1.json) fige les fichiers de
test suivis, leur domaine, leur suite logique, leur frontière réseau et leur
durée observée. La collecte canonique est conservée dans
[`test_collection_v1.json`](test_collection_v1.json), et la découpe des anciens
modules monolithiques est protégée par les empreintes AST de
[`test_split_audit_v1.json`](test_split_audit_v1.json).

# Données locales

Ce dossier fait environ 34 Go au 20 août 2026. Plusieurs générations de stockage
y coexistent encore. Le pointeur de production actuel est uniquement :

```text
model_inputs/manifests/latest.json
```

Ne pas choisir un fichier de remplacement en fonction de sa date ou de son nom.

## Pourquoi plusieurs structures coexistent

| Zone | Rôle actuel | Statut |
| --- | --- | --- |
| fichiers `US_*.parquet`, `SP500*` et `latest_snapshot.json` | ancienne interface Legacy | compatibilité, à inventorier avant migration |
| `_snapshots/` | anciennes captures locales | historique non canonique |
| `eodhd/` | archive EODHD figée, importante pour les titres inactifs | source historique à conserver |
| `open_source/` | acquisition multi-source, audits, transactions et anciens outputs | génération intermédiaire encore utilisée par certains outils |
| `sec/` | anciennes générations SEC-only | migration inachevée |
| `warehouse/` | nouvelle structure `raw -> stg -> def -> mart` | cible de transformation |
| `model_inputs/` | snapshots composés immuables réellement consommables | publication courante |
| `production/` | petits pointeurs de promotion | contrôle, pas contenu modèle |
| `outputs/` | anciens checkpoints de données | ne pas confondre avec `/outputs` à la racine |

Cette coexistence est documentée honnêtement ; elle ne signifie pas que chaque
zone est équivalente ou qu'un agent peut choisir la plus récente à la main.
L'inventaire machine-lisible des 35 emplacements et de leurs lecteurs est
[`../docs/architecture/data_location_inventory_v1.json`](../docs/architecture/data_location_inventory_v1.json).

## Contrat cible

```text
warehouse/raw -> warehouse/stg -> warehouse/def -> warehouse/mart
                                                        |
                                                        v
                                      model_inputs/history/<snapshot>
```

- `raw/` conserve les observations fournisseur et les changements. Un
  téléchargement identique garde un nouveau reçu mais peut référencer le même
  payload par hash, sans recopier les mêmes octets.
- `stg/` normalise les formes et identifiants sans choisir entre deux valeurs.
- `def/` retient une valeur selon une règle versionnée et en conserve la
  provenance. Cette couche reste nécessaire même lorsque les doublons exacts
  ont été éliminés dans `raw`.
- `mart/` assemble les tables destinées à Legacy, Boosting ou un autre
  consommateur précis.
- `model_inputs/history/` conserve les snapshots immuables publiés à partir des
  marts. Un snapshot est une release, pas une transformation concurrente.

Les huit racines fournisseur autorisées et leurs manifestes sont définis dans
[`../configs/data_contracts/raw_provider_contracts_v1.json`](../configs/data_contracts/raw_provider_contracts_v1.json).

Le contrat complet est dans
[`../docs/architecture/data_lifecycle.md`](../docs/architecture/data_lifecycle.md).

## Règles de sécurité

- Ne jamais réécrire EODHD ou un snapshot publié en place.
- Ne jamais supprimer une ancienne racine avant inventaire de tous ses lecteurs
  et preuve de parité par clés, valeurs et hashes.
- Ne pas retélécharger une histoire déjà prouvée localement pour la seule raison
  qu'elle change de dossier.
- Conserver chaque révision fournisseur et chaque correction sourcée.
- Un cache de transport n'est pas une archive brute.
- Tout package publiable embarque un manifeste de lignée.

Les dossiers horodatés ou générés n'ont pas chacun un README ; leur parent
définit le contrat et chaque package doit s'identifier par son manifeste. La
migration progressive est suivie par `DATA-001` à `DATA-010` dans
[`../ROADMAP.md`](../ROADMAP.md).

# Architecture du dépôt

Ce dossier explique l'organisation du projet sans raconter l'historique des
runs ni redécrire les deux modèles.

- [`repository_map.md`](repository_map.md) : où se trouve chaque responsabilité,
  aujourd'hui et dans la cible.
- [`data_lifecycle.md`](data_lifecycle.md) : rôle précis de `raw`, `stg`, `def`,
  `mart`, des snapshots et des sorties.

Les migrations nécessaires sont suivies dans [`../../ROADMAP.md`](../../ROADMAP.md).
Les règles applicables aux nouveaux changements sont sous
[`../standards/`](../standards/).

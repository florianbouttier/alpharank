# Refresh intégral et replay historique du 24-25 août 2026

**Tâche : `DATA-012`. Statut : exécutée, candidat data bloqué, snapshot publié
inchangé, replays du snapshot valide identiques.**

## Conclusion

Le protocole obligatoire produit bien l'une des conclusions prévues, sans
ambiguïté entre données et calcul :

- le refresh bootstrap `20260824_214818` est
  `blocked_before_replay` à cause de révisions historiques Yahoo non revues ;
- aucun snapshot candidat n'a été publié et aucun modèle n'a consommé ces
  données ;
- Legacy et Boosting ont ensuite été recalculés intégralement sur le dernier
  snapshot publié valide ;
- leurs positions et poids historiques sont identiques à la baseline au cutoff
  commun du 1er juillet 2026.

Cette exécution ne valide donc pas les nouvelles valeurs Yahoo. Elle valide le
comportement attendu : une donnée douteuse est isolée avec son explication,
tandis que le calcul reste reproductible sur la dernière donnée recevable.

## Baseline scellée

- snapshot publié : composition
  `9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad` ;
- pointeur : `data/model_inputs/manifests/latest.json`, SHA-256
  `5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4` ;
- Legacy : `outputs/production_refresh_20260820/legacy_runs_live021/2026-08-20/runs/20260820_055245` ;
- Boosting : `outputs/production_refresh_20260820/boosting_latest_common_live021_final_r2` ;
- replay commun : `outputs/production_refresh_20260820/common_replay_live021_final` ;
- cutoff historique commun : `2026-07-01`.

Le hash du pointeur est resté identique après le refresh échoué.

## Acquisition réellement exécutée

La commande demandait un bootstrap depuis 2005 et 22 années de fallbacks
financiers, sans aucune option autorisant les révisions :

```bash
./.venv/bin/python scripts/open_source/ingestion/run_ingestion.py \
  --mode bootstrap \
  --start-date 2005-01-01 \
  --financial-lookback-years 22 \
  --live-dir data/open_source/official \
  --reference-data-dir data \
  --eodhd-price-seed-path data/eodhd/output/US_Finalprice.parquet
```

La gate prix est placée avant les acquisitions fondamentales. Les statuts
suivants sont conservés dans le rapport machine-lisible :

| Source | Statut | Preuve ou conséquence |
| --- | --- | --- |
| prix Yahoo | `downloaded_quarantined` | 502 tickers rafraîchissables, 2 480 115 lignes actives résolues ; archive RAW hashée |
| seed prix EODHD | `retained_not_redownloadable` | 1 232 112 lignes et 338 tickers historiques conservés |
| historique ouvert validé précédent | `retained_by_vintage` | 5 423 lignes open source only ; 5 440 lignes EQR reportées et auditées |
| registre S&P 500 | `retained_reference_input` | registre validé consommé comme référence, pas reconstruit par cet entrypoint |
| métadonnées Yahoo | `not_started_blocked_upstream` | arrêt avant cette phase |
| SEC Companyfacts | `not_started_blocked_upstream` | arrêt avant cette phase |
| SEC Submissions | `not_started_blocked_upstream` | arrêt avant cette phase |
| documents de filing SEC | `not_started_blocked_upstream` | arrêt avant cette phase |
| fondamentaux SimFin | `not_started_blocked_upstream` | arrêt avant cette phase |
| fallback fondamental yfinance | `not_started_blocked_upstream` | arrêt avant cette phase |

L'archive Yahoo est
`data/warehouse/raw/yahoo/prices/runs/20260824_214818/manifest.json`, SHA-256
`02c137533cce0fc06418312e6baf9ae960131980fcc6f934f13bd4364f98bd04`.
Son Parquet d'événements porte le SHA-256
`b0fcddee938d8b2cf12bb2e80a540b7824cac102bc27a212ba4f7df0344dbdf6`.

## Drift data bloquant

La gate `price_revision_guard` a refusé le candidat avec la raison
`unreviewed_historical_return_revisions` :

- 3 712 227 lignes candidates et 840 tickers ;
- 44 rendements journaliers historiques modifiés de plus de 1 bp, répartis sur
  30 tickers après exclusion de la fenêtre récente mutable de sept jours ;
- dix de ces révisions sont antérieures au cutoff modèle commun, toutes sur
  `AVB.US`, entre 2007 et 2019 ;
- six révisions `MNST.US` entre le 20 juillet et le 7 août 2026 oscillent
  jusqu'à `+97,79` points de rendement journalier ;
- d'autres révisions du 12 août touchent notamment `PSX.US`, `ARES.US`,
  `Q.US`, `KEYS.US`, `FOXA.US`, `IR.US` et `FOX.US` ;
- zéro clé historique retirée, zéro changement de disponibilité du rendement
  et zéro anomalie de facteur de transition.

La preuve canonique est
`data/open_source/official/runs/20260824_214818/price_revision_guard.json`,
SHA-256
`ac8856ec4449c18827288634455b2a11a332d1add65cc0fa4e7a1c8fb39774d4`.
Le détail par clé est dans
`price_daily_return_revisions.parquet` sous la même racine. Aucun impact de
portefeuille candidat n'est inventé : les modèles n'ont pas été lancés sur une
donnée qui a échoué.

## Replays du dernier snapshot valide

Le Legacy complet a été relancé avec 30 essais, un seul job, le premier mois
`2010-01`, les mêmes exclusions et le même snapshot :

`outputs/data_refresh_replay_20260824/legacy/2026-08-25/runs/20260825_000250`.

Le Boosting a consommé exactement l'`input_snapshot/`, le détail Legacy et les
rendements mensuels de ce nouveau run :

`outputs/data_refresh_replay_20260824/boosting`.

Le replay commun reproduit le profil scellé avec 10 bps, Top 5/10 et univers
natif :

```bash
./.venv/bin/python scripts/build_common_legacy_boosting_replay.py \
  --legacy-run-dir outputs/data_refresh_replay_20260824/legacy/2026-08-25/runs/20260825_000250 \
  --boosting-run-dir outputs/data_refresh_replay_20260824/boosting \
  --output-dir outputs/data_refresh_replay_20260824/common_baseline_profile \
  --transaction-cost-bps 10 \
  --top-n 5 10 \
  --native-only
```

Un premier appel sans les deux derniers paramètres a correctement échoué : le
nouveau défaut ajoute Top 15/20 et l'univers PE Legacy. Il sélectionnait alors
`SATS.US`, 17e en avril 2026, malgré une cible un mois censurée. Les prédictions
n'avaient pas changé ; la configuration ne correspondait simplement pas à la
baseline. Cet essai n'est pas utilisé dans la comparaison finale.

## Résultat exact au cutoff

| Étage | Lignes comparées | Ajouts | Retraits | Valeurs matérielles modifiées | Écart numérique maximal |
| --- | ---: | ---: | ---: | ---: | ---: |
| holdings Legacy | 7 994 | 0 | 0 | 0 | `2,78e-17` |
| simulation Legacy | 594 | 0 | 0 | 0 | `2,78e-16` |
| prédictions Boosting | 88 948 | 0 | 0 | 0 | `0` |
| holdings communs | 6 395 | 0 | 0 | 0 | `2,78e-17` |
| simulation commune | 720 | 0 | 0 | 0 | `2,22e-16` |

Les huit tables d'entrée du snapshot ont aussi zéro clé et zéro valeur
différente. Les résidus inférieurs à `1e-12` sont du bruit de représentation
flottante ; ils ne changent aucun titre ni poids. Le Parquet des prédictions
Boosting est physiquement identique à la baseline, SHA-256
`9d82b0e2c72ccb73ef8813d42e85c4cadf27f9beda13f1485b5313b7a48ad8e6`.

Le code et certaines dépendances ont évolué depuis le commit baseline
`496ae0c`, et le Legacy déclare maintenant ses scénarios d'exécution. Ces écarts
de provenance sont listés valeur avant/après dans le rapport ; ils n'ont pas
modifié les portefeuilles observés.

## Rapports et décision

- refresh bloqué :
  `outputs/data_refresh_replay_20260824/blocked_bootstrap_20260824_214818/refresh_replay_report.json`,
  SHA-256
  `c772d969a1e6691de98132aad097876e12a71a83b751aaf1c194ad29785f85f6` ;
- replay complet valide :
  `outputs/data_refresh_replay_20260824/complete_canonical_replay/refresh_replay_report.json`,
  SHA-256
  `f6a3ed299d31bcffe4d3917c0bc64e7af1e5b4548090d29c73c5279328cf40ab` ;
- commit propre de l'auditeur dans les deux rapports : `3b7a875` ;
- commit des deux recalculs modèles : `0f91dcf`.

Décision : aucune promotion data. Le snapshot `9a2058c9…425ad` reste la seule
entrée de production. La reproductibilité des portefeuilles est démontrée sur
ce snapshot ; le prochain refresh doit reprendre toute la procédure et ne peut
avancer que lorsque la gate prix puis les acquisitions suivantes aboutissent.

# Candidat data complet du 8 septembre 2026

## Verdict data

Le run réseau `20260908_002341` a acquis toutes les sources déclarées puis a
été arrêté par la seule fraîcheur du calendrier S&P 500, encore borné au
1er août. Après extension déterministe du calendrier au 1er septembre, les 503
membres attendus correspondent exactement aux 503 titres actifs téléchargés.
Le candidat composé `446f06e0365b0cb7be3bb70cdcca82d7d700accd437d0e1d6608d37feb27298f`
passe les gates data et reste non promu jusqu'au replay `REPLAY-007`.

Ce verdict ne dit encore rien sur l'identité des portefeuilles historiques :
Legacy et Boosting doivent être recalculés sur la publication et le candidat
avec le même code, les mêmes paramètres et le même cutoff. Le pointeur
`data/model_inputs/manifests/latest.json` reste donc byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`,
sur la composition publiée
`9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad`.

## Acquisition et durée observée

Le nightly a démarré le 8 septembre à 02:23:41 heure de Paris. Les derniers
artefacts du run ont été écrits à 21:45:51 et l'erreur finale de fraîcheur à
21:45:53, soit environ 19 h 22. Cette durée élevée vient surtout des appels SEC
et de plusieurs attentes réseau ; elle constitue l'estimation empirique à
retenir pour un nouveau téléchargement intégral dans les mêmes conditions.

Une seconde invocation accidentelle de la façade `nightly_ingestion.py` avec
`--help` a démarré un run au lieu d'afficher l'aide. Le run
`20260908_210632` a été interrompu après 56 secondes, avant toute acquisition
utile. Il n'est utilisé par aucun package et le run complet
`20260908_002341` reste la seule source du candidat.

Statuts du run complet :

| Source | Statut | Lignes | Échecs | Interprétation |
| --- | --- | ---: | ---: | --- |
| Yahoo prix | `downloaded_revisions_reconciled` | 2 484 499 | 0 | 503/503 titres actifs observés |
| Yahoo metadata | `downloaded` | 4 | 0 | diagnostic |
| yfinance earnings | `downloaded` | 43 475 | 0 | fallback diagnostic |
| SEC submissions | `downloaded` | 3 903 | 0 | réseau actif complet |
| SEC companyfacts | `downloaded_with_failures` | 503 912 | 7 | sept 404 historiques/inactifs, aucun échec actif |
| documents SEC | `downloaded` | 17 | 0 | fallback filing ciblé |
| SimFin | `downloaded_with_failures` | 9 656 | 2 | fallback diagnostic, jamais valeur officielle |
| yfinance fundamentals | `downloaded` | 26 603 | 0 | fallback diagnostic |

Les sept réponses SEC 404 concernent `ABS`, `CFC`, `GLK`, `PLL`, `RX`, `SBNY`
et `TMC`, tous hors univers actif. Le manifeste d'acquisition est
`data/open_source/official/runs/20260908_002341/acquisition_status.json`,
SHA-256 `16f6579d67be0cf61506ad202d355ad75b2ef3010099a7d872dce6171cbd7f59`.

## Calendrier de septembre

La commande suivante a d'abord été exécutée en mode simulation puis réellement :

```bash
./.venv/bin/python scripts/open_source/refresh_sp500_constituents.py \
  --target-month 2026-09-01
```

Elle produit 228 056 lignes mensuelles, 503 membres en septembre et aucun
événement de composition entre août et septembre. La comparaison exacte des
symboles donne zéro membre de septembre absent du téléchargement et zéro titre
actif téléchargé hors de cette photographie. Le fichier source a le SHA-256
`b8c7b8a618e9c8d00a25cc9d56b8b5a6e1f14049c63b5e8ff1d380751135e209`.

## Prix : historique figé, nouvelles séances ajoutées

Le package a été reconstruit sans réseau :

```bash
./.venv/bin/python scripts/open_source/build_acquired_price_package.py \
  --acquisition-run-dir data/open_source/official/runs/20260908_002341 \
  --sec-package-dir outputs/data_refresh_replay_20260908/sec_candidate \
  --constituents-source data/SP500_Constituents.csv \
  --eodhd-seed data/eodhd/output/US_Finalprice.parquet \
  --output-dir outputs/data_refresh_replay_20260908/price_candidate \
  --expected-through 2026-09-08
```

Le manifeste
`outputs/data_refresh_replay_20260908/price_candidate/lineage/manifest.json`,
SHA-256 `bc63f7fc42401fa7517820a3dc26ec3e3780e7ddf6a22b1d4b966c64ff740de9`,
prouve :

- 3 727 479 lignes et 843 tickers dans le package canonique ;
- prix des 503 membres et SPY jusqu'à la séance close du 4 septembre ;
- zéro membre actif manquant, zéro ligne active portée sans observation du run ;
- zéro ancienne ligne publiée modifiée et zéro clé historique supprimée ;
- 6 000 prolongements de rendement sur 500 titres et 10 263 lignes d'historique
  pour de nouveaux tickers ;
- zéro révision de rendement historique au-dessus de 1 point de base dans le
  package canonique, malgré les 41 révisions provider conservées dans le RAW ;
- gates de révision, mouvement extrême, identité et publication vertes.

Les hashes de sortie sont
`dfe486e8fab4b1d5cf67316ce968668957afa20eadfdf0a3ee6486dfdb3301ff`
pour `US_Finalprice.parquet` et
`e05853299f0be533c5360bfc2679834c99cfeb339aefc24dc61afefbcc774e6d`
pour SPY.

## SEC et snapshot composé

Le package SEC-only contient 448 889 lignes financières et 45 697 lignes
earnings pour 723 tickers. Son manifeste, SHA-256
`63d5064096ab243a075f4f80f2dd6a322cd2827f1ee5858a43abdea20c9ed29e`,
conserve les révisions historiques de chaque table et la revue explicite qui
autorise ce candidat de diagnostic, sans autoriser sa promotion. Depuis
`REPLAY-005`, Legacy utilise `no_sec_fundamentals_v1` : ces tables restent dans
le snapshot pour audit mais ne déterminent ni son univers ni son signal.

La composition a été créée avec un pointeur local :

```bash
./.venv/bin/python scripts/open_source/build_composed_model_snapshot.py \
  --price-package-dir outputs/data_refresh_replay_20260908/price_candidate \
  --sec-package-dir outputs/data_refresh_replay_20260908/sec_candidate \
  --history-root outputs/data_refresh_replay_20260908/composed_history \
  --latest-manifest outputs/data_refresh_replay_20260908/candidate_latest.json \
  --expected-through 2026-09-08
```

Le snapshot est
`outputs/data_refresh_replay_20260908/composed_history/alpharank_input_20260908_211605_446f06e0365b`.
Son manifeste, SHA-256
`896c38e37384d2b994ef21f32930d1fbec7e6fe2dea3c447a846b87424c1d152`,
valide neuf fichiers, le payload prix exact, les identités, le registre des
historiques persistants et l'usage du même snapshot pour Legacy et Boosting.

## Gate suivante

`REPLAY-007` doit maintenant exécuter les deux méthodes sur la publication et
le candidat, au cutoff commun, puis comparer toutes les positions et tous les
poids historiques. Les décisions de fin juillet et fin août feront l'objet
d'une extraction explicite : juillet doit être comparé au portefeuille conservé
avant ce refresh ; août doit être séparé entre signal de fin août et rendement
de septembre encore immature. Toute divergence inexpliquée bloque la promotion.

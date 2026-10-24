# Candidat data du 24 octobre 2026

## Verdict data

Le refresh complet `20261024_140104` a reçu toutes les sources déclarées après
la correction point-in-time `PSKY` vers `SKYD` de `DATA-042`. Le snapshot
candidat composé porte l'identifiant
`c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e`
et valide ses neuf fichiers. Il reste volontairement local jusqu'au verdict de
`REPLAY-012`.

Le pointeur canonique `data/model_inputs/manifests/latest.json` reste
byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`,
sur la composition publiée
`9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad`.

## Deux arrêts utiles, sans perte du téléchargement

Le premier run automatique du jour, `20261024_001501`, a refusé la publication
parce que le registre demandait encore `PSKY`, dont la série fournisseur ne
franchissait pas la gate de fraîcheur. L'enquête a relié cette fin de série au
transfert Nasdaq vers NYSE et au changement officiel de symbole vers `SKYD` le
6 octobre. `DATA-042` a versionné cet événement avant toute nouvelle
acquisition.

Le run corrigé a duré 49 minutes 24 secondes, de 16:01:04 à 16:50:28 heure de
Paris. Son statut final reste `failed` uniquement parce que la publication
automatique refuse les révisions fondamentales historiques non revues. Les RAW
ont tous été conservés sous
`data/open_source/official/runs/20261024_140104`, puis réutilisés sans réseau
pour construire le candidat diagnostic explicitement non promouvable.

## Acquisition

| Source | Statut | Lignes | Échecs | Lecture |
| --- | --- | ---: | ---: | --- |
| Yahoo prix | `downloaded_revisions_reconciled` | 2 519 731 | 0 | 503 actifs et six passerelles de maturité |
| Yahoo metadata | `downloaded` | 9 | 0 | diagnostic |
| yfinance earnings | `downloaded` | 41 348 | 0 | fallback diagnostic |
| SEC submissions | `downloaded` | 4 034 | 0 | univers actif complet |
| SEC Companyfacts | `downloaded_with_failures` | 506 860 | 7 | anciens symboles 404 hors univers actif |
| documents SEC | `downloaded` | 17 | 0 | fallback filing ciblé |
| SimFin | `downloaded_with_failures` | 13 042 | 2 | fallback diagnostic uniquement |
| yfinance fondamentaux | `downloaded` | 26 587 | 0 | fallback diagnostic uniquement |

Les sept 404 Companyfacts concernent `ABS`, `CFC`, `GLK`, `PLL`, `RX`, `SBNY`
et `TMC`. Aucun membre actif ne manque dans Companyfacts ou submissions.

## Prix réconciliés

Le package prix contient 3 757 277 lignes pour 848 tickers. Les 503 membres
courants ont une dernière séance commune au 23 octobre, tout comme SPY. `SKYD`
est membre courant ; `PSKY` reste seulement une passerelle de maturité afin de
ne pas casser un portefeuille formé avant le changement de symbole.

La réconciliation conserve les 3 711 192 lignes du snapshot publié sans en
modifier une seule, ajoute 22 973 observations par rendements fournisseurs sur
501 tickers et ne laisse aucun ticker non résolu. Les révisions Yahoo restent
dans le RAW et l'audit ; elles ne remplacent pas l'historique canonique. Le
manifeste prix a le SHA-256
`b909cae03cb34bc1a1249ac77adeb275dd2ee4896313c97813738349a8acfe48`.

## SEC cumulatif et candidat strict

Le rebuild cumulatif conserve les versions précédentes et ajoute le delta du
jour : 514 118 faits Companyfacts, 145 650 faits filing, 58 147 calendriers,
39 644 actuals et 1 657 références de lignée. Son manifeste a le SHA-256
`8dc66929118de5eb4cdd11123d12babe80c34ed284a5e4fbc9f2bd24e90ea161`.

Le package SEC-only produit 45 145 lignes income, 45 109 balance, 33 512
cash-flow, 43 302 earnings, 43 404 shares et 848 références générales. Les cinq
familles portent une revue explicite pour ce candidat de diagnostic ; cette
revue n'autorise aucune promotion. Un second build depuis le même RAW reproduit
byte pour byte les sept exports. Le manifeste SEC a le SHA-256
`efdf5da36d91039d538a887ecd1933d99adbf513a2c3e09c5d087fea63212537`.

## Snapshot candidat et gate suivante

Le snapshot se trouve dans
`outputs/data_refresh_replay_20261024/composed_history/alpharank_input_20261024_145316_c340a9b45509`.
Son manifeste, SHA-256
`88988103332b851a97c9625102902dc0755f11a432b2c6002c9e4c38d6eb3bc8`,
confirme un même payload pour Legacy et Boosting, des fondamentaux strictement
SEC-only et la conservation exacte du package prix.

`REPLAY-012` doit maintenant recalculer Legacy, Boosting et la variante tendance
sur ce candidat, comparer automatiquement le dernier portefeuille déjà formé à
sa date exacte, puis rapprocher tout l'historique commun. Tout titre ou poids
passé modifié doit être listé et expliqué ; une différence inexpliquée bloque
la promotion.

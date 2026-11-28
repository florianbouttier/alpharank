# Candidat data du 31 octobre 2026

## Verdict data

Le refresh complet `20261031_151756` a acquis toutes les sources déclarées et
conserve ses RAW immuables. La publication automatique s'est arrêtée, comme
prévu, devant les révisions historiques SEC des cinq familles fondamentales.
Aucun pointeur de production n'a été déplacé.

Le candidat diagnostic composé porte l'identifiant
`e0e6045a5b9b2702840a4f96c94618aecc708272adacfea8a2c2e843790ef95a`.
Il valide ses neuf fichiers et reste local jusqu'au verdict du replay
inter-vintages. Le pointeur canonique
`data/model_inputs/manifests/latest.json` reste byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`.

## Acquisition

Le run a duré environ 51 minutes. Les 509 symboles prix demandés ont tous été
reçus : 503 membres courants et six passerelles de maturité. Companyfacts et
submissions sont complets pour l'univers actif ; les sept 404 Companyfacts
concernent les anciens symboles `ABS`, `CFC`, `GLK`, `PLL`, `RX`, `SBNY` et
`TMC`. SimFin a échoué deux fois et reste un fallback diagnostic, jamais une
valeur fondamentale finale.

| Source | Statut | Lignes | Échecs |
| --- | --- | ---: | ---: |
| Yahoo prix | `downloaded_revisions_reconciled` | 2 522 267 | 0 |
| Yahoo metadata | `downloaded` | 9 | 0 |
| yfinance earnings | `downloaded` | 41 725 | 0 |
| SEC submissions | `downloaded` | 4 233 | 0 |
| SEC Companyfacts | `downloaded_with_failures` | 509 135 | 7 anciens symboles |
| documents SEC | `downloaded` | 17 | 0 |
| SimFin | `downloaded_with_failures` | 10 239 | 2 |
| yfinance fondamentaux | `downloaded` | 26 548 | 0 |

## Prix réconciliés

Le package prix contient 3 759 812 lignes pour 848 tickers. Les 503 membres
courants et SPY atteignent tous la séance du 30 octobre 2026. La réconciliation
conserve exactement les 3 711 192 observations précédemment validées : zéro
rendement historique modifié, zéro clé supprimée. Elle ajoute 25 468 séances
par prolongation des rendements fournisseur sur 501 tickers ; les révisions
brutes du fournisseur restent dans les audits du run.

Le manifeste prix a le SHA-256
`15a5e014019c3046494c15dd70f07def3c92224fc26b2c878000259193d78ca2`.
Les gates de révision, mouvements extrêmes, identité de titre et benchmark
passent toutes.

## SEC cumulatif

Le dossier historique `data/open_source/official/raw` date du 27 août et ne
contient que quatre tickers dans son calendrier SEC ; il n'est donc pas un
parent cumulatif valide. Le rebuild lie explicitement le candidat immuable du
24 octobre au delta RAW du run courant. Il produit 516 393 faits Companyfacts,
145 650 faits filing, 58 353 calendriers, 39 876 actuals et 1 657 références de
lignée, sans perdre l'historique antérieur.

Le package SEC-only contient 45 363 lignes income, 45 322 balance, 33 704
cash-flow, 43 524 earnings, 43 616 shares et 848 références générales. Son
manifeste, SHA-256
`4001cda622ebd150c68301a4d8fef364f2175826f9a284083ed65e7918f59b1d`,
porte la revue explicite des révisions pour ce candidat diagnostic. Cette revue
n'autorise aucune promotion.

## Snapshot et gate suivante

Le snapshot se trouve dans
`outputs/data_refresh_replay_20261031/composed_history/alpharank_input_20261031_161144_e0e6045a5b9b`.
Son manifeste, SHA-256
`e2d9381fc08a0f5510df508e0a9a24b4e6af24b27980cfef04da3f4f671b1d58`,
confirme le même payload pour Legacy et Boosting, les fondamentaux SEC-only et
la conservation exacte du package prix.

Le replay suivant doit recalculer Boosting Top 5 et Boosting tendance Top 5,
comparer le dernier portefeuille déjà formé à sa date exacte, rapprocher tout
l'historique commun, puis seulement calculer la poche 50/50 jusqu'à octobre.
Tout drift passé inexpliqué bloque la conclusion et la promotion.

## 28 novembre 2026 — membres-mois sans cours (`DATA-060`)

Ce diagnostic lit les audits prix des deux runs Legacy de
`outputs/data_refresh_replay_20261031/replay_head3b17eb7/`, le témoin
`baseline/legacy/2026-10-31/runs/20261031_175050` sur la composition
`c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e` et
`candidate/legacy/2026-10-31/runs/20261031_175053` sur `e0e6045a…ef95a`.
La fenêtre couvre les 183 mois de décision Boosting de juillet 2011 à
septembre 2026. Les dates sont des mois de décision, avant la détention.

Les deux `input_snapshot/SP500_Constituents.csv` ont le même SHA-256
`b506faeff7a7291a2dff333bd225b4ed4386d52a310ce77ed2e1e2361fa4d993`,
vérifié contre chaque `data_input_manifest.json`. La normalisation
`prepare_constituents_monthly` lit `Date` et `Ticker`, produit la clé unique
`(ticker, decision_month)` et reprend les exclusions `run_config.excluded_tickers`.
Sur les clés restantes, une anti-jointure aux lignes `price_observations > 0`
de `monthly_price_eligibility.parquet` dénombre les mois sans observation.
Le champ `price_eligible` sépare les mois observés qui échouent au filtre de
nombre de séances, liquidité ou qualité OHLC.

| Résultat sur la même fenêtre | Témoin | Candidat |
| --- | ---: | ---: |
| Clés du CSV de membres | 92 807 | 92 807 |
| Clés de symboles en quarantaine | 430 | 430 |
| Clés après quarantaine | 92 377 | 92 377 |
| Sans observation de cours | 2 328 (2,52 %, 66 symboles) | 2 328 (2,52 %, 66 symboles) |
| Prix observé, gate échouée | 96 | 96 |
| Prix observé, gate passée | 89 953 | 89 953 |

Les ensembles de clés sans cours sont identiques entre les deux runs. Pour
le candidat, les 89 953 clés passant la gate correspondent exactement aux
89 953 clés du fichier H06 `boosting_end_2026_09/classification_h06/predictions.parquet`
avant le filtre terminal du moteur commun, avec les deux anti-jointures vides.

Les absences se répartissent autour de la plage observée de chaque symbole
dans l'audit prix tronqué à septembre 2026. On compte 1 734 clés avant son
premier mois coté (41 symboles), 549 sans série sur l'audit (10), et 45 après
son dernier mois observé (15). Les plus grands groupes sont `MBC` (137),
`STI` (102), `LLL` (97), `FOXA` (92), `CA` (89) et `DAY` (81). Ce sont des
écarts au calendrier historique du CSV, dont la certification reste suivie
dans `DATA-053`. Les dates et noms ne suffisent pas à prouver la continuité
d'un instrument.

Quelques cas permettent de distinguer la nature des preuves disponibles.

- **Identité ancienne certifiée, prix absents.** Le registre
  `security_identity_intervals_v1` fixe `SNDK_OLD` (CIK `0001000180`) jusqu'au
  12 mai 2016 et `SNDK` (CIK `0002023554`) à partir du 24 février 2025.
  L'ancien titre garde 59 mois sans cours entre juillet 2011 et mai 2016.
  [Registre d'identité](../../src/alpharank/data/open_source/reference/security_identity_registry.csv)
- **Périodes et compagnies à rapprocher.** Les mois manquants de `FOXA`
  portent le nom 21st Century Fox, alors que la série candidate débute en mars
  2019 et que le calendrier utilise ensuite Fox Corporation. Les mois
  manquants de `DOW` portent Dow Chemical Company, distinct de Dow Inc. dans
  le CSV. Le registre de prix persistant date le début du prix `MBC` au
  9 décembre 2022 et celui de `STI` au 2 mai 2022. Le pont SEC de `STI` borne
  l'ancien SunTrust au 31 décembre 2019. Aucune clé manquante ne trouve une
  série mensuelle par une simple substitution point vers tiret.
  [Pont SEC historique](../../src/alpharank/data/open_source/reference/sec_historical_ticker_bridge.csv)
- **Alias déjà vérifié.** Le manifeste du candidat lie le cas SATS/ECHO à un
  CUSIP inchangé, une ancre validée et cinq rendements communs. Il enregistre
  24 séances dérivées, zéro ligne antérieure changée et zéro valeur manuelle.
  Cette preuve fixe les seules dates couvertes par cet overlay.
  [Replay SATS/ECHO](sats_echo_replay_20260829.md)
- **Dernier mois manquant.** `WBD` est le seul cas hors quarantaine en
  septembre 2026. La sortie d'indice est datée du 6 octobre. Le fichier prix
  du candidat contient 13 séances d'août, jusqu'au 19, aucune en septembre
  et une ligne le 5 octobre avec volume 40. Ce constat ne tranche ni la
  cause du trou, ni l'exécutabilité de la ligne d'octobre.
  [Calendrier S&P d'octobre](sp500_constituents_20261008.md#sources-et-événements)

| Artefact contrôlé | SHA-256 |
| --- | --- |
| `monthly_price_eligibility.parquet`, témoin | `c81cece22f3e0032f606244d49f46c6fa9e935dff11c998bc75b38dadd68736e` |
| `monthly_price_eligibility.parquet`, candidat | `3de982b076d514f4f957c2464fe0a01242c57c05ced74b6c479414d683eb57b7` |
| `predictions.parquet`, H06 candidat | `838e0246206940019e137c4e6b01a04fb38396f7949cbdbe370a9281a977da5f` |
| `price_candidate/lineage/persistent_price_history_registry.parquet` | `093da153a20b9f38d44d924074ea02fb2476b5f3d4fc6f2214a020f37a7f1597` |

Le préflight du 20 novembre contrôle une autre population. Il rapporte 503
membres courants avec prix, ce qui renseigne la fraîcheur à cette date. Son
reçu `recovery_20261125_cutoff20/price_preflight_validation.json` du refresh
du 23 novembre porte le SHA-256
`949c0311402996d9e485c85f4628c8925cd6efc7ef4d10a3763190c6d3dcdcab`.

L'audit EODHD du 14 août conserve son inventaire par symbole et dates dans
`outputs/eodhd_price_seed_audit_20260814/missing_eodhd_price_tickers.parquet`.
Il portait sur l'ancien package ouvert et l'archive EODHD au SHA-256
`0ee4b6d9766fef6942f12bb1591426302b29e19347fb32c87cb04e6777b3f8f5`.
Il sert à orienter une recherche de sources historique par instrument.
[Contrat de prix historiques](../open_source_ingestion_architecture.md#reused-ticker-identities)

`DATA-054` reste ouvert. Le prochain rapprochement local porte sur les
séances `WBD` d'août à octobre 2026 dans les RAW et la lignée déjà scellés,
puis sur l'ancien `SNDK` avant mai 2016 dans l'archive EODHD, avec son intervalle
et son identifiant d'instrument. Pour chaque séance, le relevé devra lier
instrument, date, source, vintage et hash, ou conserver le statut non résolu.
Ce lot est une lecture des fichiers existants, sans nouvel artefact de prix
ni changement de snapshot.

## 28 novembre 2026 — origine locale du trou WBD (`DATA-062`)

Les fichiers `raw/prices_yfinance.parquet` des acquisitions `20261024_140104`
et `20261031_151756` contiennent pour `WBD.US` les mêmes 14 dates d'août à
octobre. Les 13 séances du 3 au 19 août proviennent du vintage
`20260820_011146` et sont conservées par reprise du préfixe. Septembre est vide
dans les deux fichiers. La seule ligne d'octobre est le 5, avec un cours de
clôture et un cours ajusté de 30,95 USD et un volume de 40. Elle porte le
vintage de l'acquisition correspondante.

Dans `def_price_selection_audit.parquet` du 31 octobre, les 21 séances de
septembre portent `unresolved_invalid_current_raw`, avec cours courant et
source antérieure nuls. La lignée du candidat conserve les mêmes 14 dates.
Son journal de prolongation lie le 5 octobre au 19 août avec un rendement
ajusté de `+8,4823 %` sur l'intervalle couvert par ces deux observations.
L'absence de séances entre elles est déjà présente au stade RAW local. La
cause de l'absence dans l'acquisition et l'exécutabilité de la ligne du
5 octobre restent à établir. L'appartenance S&P est confirmée en septembre,
la sortie intervenant le 6 octobre. `DATA-054` garde ce cas non résolu.
[Calendrier officiel archivé](sp500_constituents_20261008.md#sources-et-événements)

| Artefact lu en filtrant `WBD.US` du 1er août au 31 octobre | SHA-256 |
| --- | --- |
| `raw/prices_yfinance.parquet`, run du 24 octobre | `0b41b0fde1cc4b387d3edc8afd8cffed9aa87420b79e93067dfb429431494f98` |
| `raw/prices_yfinance.parquet`, run du 31 octobre | `72d6e4cbe63347d0b17d93fc414aa7d2feba6438ada40641cd9cda8433f1ef15` |
| `def_price_selection_audit.parquet`, run du 31 octobre | `a7a770668b0b1699d31e7f35745b774690eac99a057cf4e55bebfff4e1b4fca1` |
| `price_return_extension_audit.parquet`, run du 31 octobre | `d1954f09d62ff0fd402b7a366c6698f5231579f8dcd9c374f26de43ca732dae9` |
| `price_candidate/lineage/prices_open_source_lineage.parquet` | `2e0312040524a19f20a6da6b41aac8297cefd584b084583a72fb60007b653158` |

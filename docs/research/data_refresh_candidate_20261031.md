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

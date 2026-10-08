# Candidat data final du 8 octobre 2026

## Verdict data

Le run réseau `20261008_095842` a retéléchargé toutes les sources déclarées
après l'ajout des passerelles de maturité de `DATA-039`. Le candidat composé
`dbaac67b89e67f95a8a7040c07ae560fc13be574c8c463b3f07bcb5b0068e598`
contient les 503 membres actifs et les cinq sortants récents encore nécessaires
aux rendements de portefeuille : `BLDR`, `CTVA`, `TAP`, `TTD` et `WBD`.

La publication automatique s'est arrêtée comme prévu devant les révisions
fondamentales historiques. Les réponses reçues sont conservées, puis un package
diagnostic explicitement autorisé pour le replay a été construit sans déplacer
la vérité de production. Le pointeur `data/model_inputs/manifests/latest.json`
reste byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`,
sur la composition
`9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad`.

## Acquisition et durée

L'acquisition s'est déroulée de 11:58:42 à 12:53:12, heure de Paris, soit
54 minutes 30 secondes. Le statut final `failed` décrit uniquement le refus de
publication des révisions SEC ; il ne signifie pas que les téléchargements ont
été perdus. Leur RAW est conservé sous
`data/open_source/official/runs/20261008_095842`.

| Source | Statut | Lignes | Échecs | Lecture |
| --- | --- | ---: | ---: | --- |
| Yahoo prix | `downloaded_revisions_reconciled` | 2 508 398 | 0 | 508 historiques prix observés |
| Yahoo metadata | `downloaded` | 8 | 0 | diagnostic |
| yfinance earnings | `downloaded` | 42 333 | 0 | fallback diagnostic |
| SEC submissions | `downloaded` | 3 952 | 0 | univers actif complet |
| SEC Companyfacts | `downloaded_with_failures` | 505 878 | 7 | anciens symboles 404 hors univers actif |
| documents SEC | `downloaded` | 17 | 0 | fallback filing ciblé |
| SimFin | `downloaded_with_failures` | 14 090 | 2 | fallback diagnostic uniquement |
| yfinance fondamentaux | `downloaded` | 26 575 | 0 | fallback diagnostic uniquement |

Les sept réponses Companyfacts 404 sont `ABS`, `CFC`, `GLK`, `PLL`, `RX`,
`SBNY` et `TMC`. Aucun membre actif ne manque dans Companyfacts ou submissions.

## Prix : historique validé conservé

Le package prix final contient 3 745 944 lignes pour 847 tickers. Les 503
membres actifs et les cinq passerelles ont été demandés au fournisseur ; leur
dernière séance close est le 7 octobre 2026. La réconciliation valide :

- zéro ancienne ligne canonique modifiée et zéro clé historique supprimée ;
- 1 200 483 clés EODHD attendues et zéro manque ;
- 16 978 lignes d'extension par rendement pour 501 tickers ;
- cinq mouvements extrêmes approuvés et aucune alerte non revue ;
- 34 séances nouvelles par passerelle, sauf `WBD` déjà presque complet.

Le manifeste prix a le SHA-256
`763176006a5b7992222377bb9e899081351b77ad8b939a6a0265ca44c94c0be2`.
Le RAW fournisseur et l'audit des variations restent séparés du package
réconcilié ; une variation fine observée ne réécrit donc pas une clé déjà
validée.

## SEC cumulatif et déterminisme

Le delta du run a été rejoué sur le RAW point-in-time retenu. La version
cumulative contient 513 136 faits Companyfacts, 145 650 faits filing, 58 065
calendriers, 39 569 actuals et 1 656 références. Son manifeste a le SHA-256
`8bcf91a799b21fd40102f62cda9b51478899a449bfdcb5261e0e8a2d755ce1a0`.

Le package diagnostic produit 45 069 lignes income, 45 026 balance, 33 441
cash-flow, 43 228 earnings, 43 323 shares et 847 références générales. Son
manifeste a le SHA-256
`6fc7ecc225a8d73c8d6d92e8bb55d38910f85e7fb992bd0de2644284dde82f22`.
Un second build strict depuis exactement le même RAW reproduit byte pour byte
les sept exports et les quatre preuves de lignée : le transformateur courant
est déterministe. Les différences avec un ancien package restent donc un écart
de vintage/provenance à expliquer par le replay, pas une instabilité du build.

## Snapshot candidat commun

Le snapshot final se trouve dans
`outputs/data_refresh_replay_20261008_095842/composed_history/alpharank_input_20261008_105551_dbaac67b89e6`.
Son manifeste, SHA-256
`4232b271bcc1b78f99d3926886fe9ac3fb7731588167a528ab8a5f446b9b4f14`,
valide les neuf fichiers et leur lignée. Le pointeur candidat local a le
SHA-256
`d3a8c59a8ea7dafa1238dc0edc9eaee86078913f6de5d2349d85fe1ce0f998a0`.

Le premier candidat du jour, issu du run `20261008_072222` et de la composition
`eac32074…e1fe6`, reste conservé comme preuve. Il est remplacé pour le replay
par ce candidat final parce qu'il ne téléchargeait pas encore les cinq
passerelles de maturité.

## Incidents et gate suivante

La commande historique `nightly_ingestion.py --help` ignore actuellement cet
argument et exécute le refresh par défaut. Cette invocation a bien réalisé le
téléchargement demandé, mais ce comportement CLI doit être corrigé séparément ;
il ne change aucun hash ci-dessus.

`REPLAY-010` doit exécuter Legacy, Boosting et le moteur commun avec le même
code sur le vintage précédent et ce candidat. Il compare automatiquement le
dernier portefeuille déjà formé à la même date, puis attribue chaque différence
par ablations prix-seuls et SEC-seuls. Aucune promotion n'est permise avant ce
verdict.

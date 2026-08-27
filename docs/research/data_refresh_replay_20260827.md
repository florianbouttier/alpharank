# Refresh intégral et attribution du drift du 27 août 2026

**Tâche : `DATA-017`. Statut : acquisition complète, deux backtests exécutés,
drift attribué, replay commun bloqué, snapshot publié inchangé.**

## Conclusion

Les données rafraîchissables ont été retéléchargées jusqu'au 27 août 2026. La
dernière séance fermée effectivement disponible est le 26 août pour les prix,
SPY et les dépôts SEC ; les 503 membres courants sont couverts.

Le candidat ne reproduit pas les portefeuilles historiques. Un contrôle exécuté
sur le même code, les mêmes paramètres et le même runtime montre que le premier
écart se situe dans le snapshot data, puis modifie 4 819 positions Legacy
communes et 88 944 prédictions Boosting communes avant le cutoff du 1er juillet
2026. Le replay commun refuse ensuite `CVC.US`, sélectionné dans le Top 10
Boosting de juillet 2016 alors que son rendement futur est censuré par une fin
de cotation.

Deux replays croisés isolent la cause : les prix/univers rafraîchis avec les
fondamentaux antérieurs conservent `CVC.US` au rang 35 et passent la gate ; les
fondamentaux SEC rafraîchis avec les prix antérieurs placent `CVC.US` au rang 8
et reproduisent exactement le blocage. Le téléchargement est donc conservé,
mais la composition candidate n'est pas promue.

## Règle de fiabilité appliquée

Après chaque refresh complet, AlphaRank doit produire l'une des deux preuves
suivantes avant toute promotion :

1. les portefeuilles historiques Legacy et Boosting sont identiques sur le
   même cutoff ;
2. le rapport relie précisément les tables data modifiées aux premiers signaux
   ou positions divergents, avec code, configuration et runtime identiques.

Une variation fournisseur ne bloque jamais le téléchargement et n'efface
jamais l'observation reçue. Elle peut en revanche bloquer la promotion. Les prix
publiés ne sont pas remplacés : le RAW différentiel conserve les observations
provider, tandis que le candidat canonique garde les rendements journaliers
validés et ajoute les nouvelles séances depuis la dernière ancre.

Cette règle est désormais explicite dans
`docs/monthly_portfolio_runbook.md`, section « Historical replay gate after
every full refresh ».

## Acquisition exécutée

La commande de bootstrap a demandé tout l'historique depuis 2005 et toutes les
sources déclarées, sans interrompre SEC ou les fallbacks sur une anomalie prix :

```bash
./.venv/bin/python scripts/open_source/run_ingestion.py \
  --mode bootstrap \
  --start-date 2005-01-01 \
  --end-date 2026-08-27 \
  --financial-lookback-years 2 \
  --audit-years 2025 \
  --live-dir data/open_source/official \
  --reference-data-dir data \
  --eodhd-price-seed-path data/eodhd/output/US_Finalprice.parquet
```

- run : `data/open_source/official/runs/20260827_070654` ;
- durée : environ 4 h 58 ;
- statut d'acquisition SHA-256 :
  `11f95ec390d0496e8d7bbb1b7c555c5cea4dc01ca6dd2bbfa623ef97cf7a685a` ;
- contrat de refresh SHA-256 :
  `23ec57a17daf9d4a9993396aa4b692b9309c98de2664400eca48352a76be86b0` ;
- manifeste RAW Yahoo SHA-256 :
  `9e1768d05ec436524d2fb2b2fe833c7340574ea6b40f6612b97ba8e48a351402` ;
- snapshot logique RAW Yahoo SHA-256 :
  `0f50c9010dbe7ffe65260995d1fd7b7f901ab3ec127a0a46086bfc6991023f42`.

| Source | Statut | Lignes | Diagnostic |
| --- | --- | ---: | --- |
| prix Yahoo | `downloaded_quarantined` | 2 480 977 | observation complète des 503 tickers actifs ; décision prix différée |
| métadonnées Yahoo | `downloaded` | 503 | aucune erreur |
| earnings yfinance | `downloaded` | 43 461 | aucune erreur |
| SEC Submissions | `downloaded` | 58 770 | couverture active complète |
| SEC Companyfacts | `downloaded_with_failures` | 538 778 | 19 réponses 404, toutes sur des symboles historiques inactifs ; zéro échec actif |
| documents de filing SEC | `downloaded` | 17 | aucun échec |
| SimFin | `downloaded_with_failures` | 92 163 | 22 échecs bornés HTTP/timeout ; fallback diagnostic, jamais valeur fondamentale finale |
| fondamentaux yfinance | `downloaded` | 26 589 | fallback diagnostic uniquement |

La sortie initiale est non nulle parce que le mouvement `RDDT.US` du 30 octobre
2024 n'était pas encore inscrit dans le registre de revue. Toutes les sources
avaient néanmoins été tentées et enregistrées. Après qualification officielle
du mouvement, la republication différée a recalculé les gates sans accès réseau
et produit le package prix suivant :

- `outputs/data_refresh_replay_20260827/price_candidate_reconciled` ;
- manifeste SHA-256 :
  `1d150e7c5cf0894a25b9b4ed5ea1a33f8e38506e0f3628343062e89f3e1d7b42` ;
- `US_Finalprice.parquet` : 3 723 934 clés uniques, 843 tickers, jusqu'au
  26 août, SHA-256
  `b0f6fe459e229f5838ac13424304188a2f5ec03dbdf620bff4eb547840ce9d80` ;
- SPY jusqu'au 26 août, SHA-256
  `9b9242c7874c8fbad7858c8f12f1a4bc7aebf447439b2e43a8f758a2469b2463` ;
- les gates de révision et de mouvement extrême passent après revue.

Au cutoff historique, aucune valeur prix commune n'est remplacée : 10 125 clés
sont ajoutées et zéro clé commune ne change. Les six retraits sont exclusivement
les séances du 13 au 21 février 2025 attribuées à tort à `SNDK.US` avant la
séparation de l'ancien symbole et du nouvel instrument. Ce sont des corrections
d'identité déjà sourcées, pas des rendements provider écrasés.

## Package SEC et composition candidate

Le package officiel SEC est
`outputs/data_refresh_replay_20260827/sec_candidate`, manifeste SHA-256
`79b2d8e04b83c4b608a9fe816f3ba09a38d8c303ce9701693eb64f89b3e7c9b8`.
Il contient 472 325 lignes financières sur 815 tickers et 57 351 lignes
earnings sur 822 tickers. La dernière date de dépôt SEC est le 26 août 2026.

Le guard compare les périodes antérieures au 27 août 2024 avec l'ancien package
SEC :

| Table SEC | Anciennes lignes | Candidat | Ajouts | Retraits | Communes modifiées |
| --- | ---: | ---: | ---: | ---: | ---: |
| income statement | 38 088 | 41 387 | 6 853 | 3 554 | 2 155 |
| balance sheet | 39 391 | 41 204 | 3 815 | 2 002 | 28 499 |
| cash flow | 16 898 | 31 311 | 16 902 | 2 489 | 425 |
| shares | 35 302 | 39 015 | 4 820 | 1 107 | 24 942 |
| earnings | 37 630 | 39 082 | 3 466 | 2 014 | 6 634 |

Ces écarts sont conservés comme révisions. Le candidat a reçu une revue
explicite pour pouvoir être rejoué, liée aux corrections de sélection
trimestrielle, de durée, de point-in-time et d'identité déjà versionnées. Cette
revue autorise le diagnostic, jamais la promotion automatique.

La composition candidate immuable est :

- composition id :
  `5bfbc1d3cb04a80e25e7521f4f148a71224689206095e288652c28325e22eabb` ;
- dossier :
  `outputs/data_refresh_replay_20260827/composed_history/alpharank_input_20260827_122648_5bfbc1d3cb04` ;
- manifeste SHA-256 :
  `8a1de5b1a35ed15d2405088d6d067bef3672e7da8f9ff8438bb20a4bea074728` ;
- fondamentaux strictement SEC, payload prix exact, politique d'identité
  appliquée, validation de composition verte ;
- prix et SPY au 26 août, calendrier S&P au 1er août, dépôts SEC au 26 août.

Le mois d'août est volontairement partiel pour les modèles. La dernière
décision complète rejouée reste donc juillet 2026.

## Replays same-code

Baseline et candidat ont été recalculés sur le commit modèle `55bf5a3`, avec un
worktree propre, 30 essais Legacy, un job, un premier mois `2010-01`, puis le
profil Boosting commun scellé. Les manifestes prouvent les mêmes fichiers de
code critiques, paramètres, seeds, dépendances et runtime.

| Rôle | Legacy | Boosting | Replay commun |
| --- | --- | --- | --- |
| baseline publiée | `same_code_baseline/legacy/2026-08-27/runs/20260827_150224` | `same_code_baseline/boosting` | `same_code_baseline/common` : passe |
| candidate refresh | `same_code_candidate/legacy/2026-08-27/runs/20260827_150224` | `same_code_candidate/boosting` | arrêt exact sur `CVC.US` |

Toutes ces racines sont sous
`outputs/data_refresh_replay_20260827/`. Le rapport final est
`same_code_audit_data_only/refresh_replay_report.json`, SHA-256
`16608074038c363a6a079572735eebc3bac913f8f546e6fa98dd6a87d286f014`.
Il conclut `common_replay_blocked`, interdit la promotion et confirme :

- `all_code_identical: true` ;
- `all_config_identical: true` ;
- `all_runtime_identical: true` ;
- premier étage divergent : `snapshot`.

### Drift au cutoff du 1er juillet 2026

| Table d'entrée | Baseline | Candidat | Ajouts | Retraits | Communes modifiées |
| --- | ---: | ---: | ---: | ---: | ---: |
| prix titres | 3 690 557 | 3 700 676 | 10 125 | 6 | 0 |
| prix SPY | 5 407 | 5 407 | 0 | 0 | 4 298 |
| général | 835 | 503 | 7 | 339 | 7 |
| income statement | 44 274 | 45 892 | 2 470 | 852 | 1 491 |
| balance sheet | 44 185 | 45 794 | 1 977 | 368 | 10 159 |
| cash flow | 32 756 | 34 011 | 1 489 | 234 | 61 |
| earnings | 42 477 | 43 324 | 1 829 | 982 | 1 810 |
| calendrier S&P | 225 881 | 225 880 | 324 | 325 | 0 |

| Sortie modèle | Baseline | Candidat | Ajouts | Retraits | Communes modifiées |
| --- | ---: | ---: | ---: | ---: | ---: |
| positions Legacy | 7 994 | 6 732 | 1 612 | 2 874 | 4 819 |
| mois Legacy | 594 | 594 | 0 | 0 | 572 |
| prédictions Boosting | 88 948 | 88 950 | 6 | 4 | 88 944 |

Le candidat ne publie aucune table commune : le moteur s'arrête avant de les
écrire sur la position censurée. Le rapport compare les étages disponibles et
n'invente pas de holdings pour contourner la gate.

## Attribution causale du blocage CVC

Deux snapshots diagnostics ont été construits sans modifier les packages
immuables, puis Legacy et Boosting ont été réentraînés intégralement avec les
mêmes paramètres :

Ces ablations ont été lancées au commit `cf6d403`, qui ne change que le
classement data/config de l'auditeur. Les hashes de tous les fichiers critiques
Legacy et Boosting sont identiques à ceux du contrôle `55bf5a3`.

| Données du scénario | Rang CVC en juin 2016 | Score | Résultat commun |
| --- | ---: | ---: | --- |
| prix baseline + SEC baseline | 35 | `0,10331903` | passe |
| prix candidats + SEC baseline | 35 | `0,10331903` | passe |
| prix baseline + SEC candidat | 8 | `0,11756054` | bloqué sur CVC |
| prix candidats + SEC candidat | 8 | `0,11756054` | bloqué sur CVC |

Le package SEC candidat est donc suffisant et nécessaire, dans cette ablation,
pour reproduire le blocage ; le package prix candidat n'a aucun effet sur le
rang ou le score de CVC.

La série prix CVC est strictement identique dans les deux snapshots : 2 886
séances du 3 janvier 2005 au 20 juin 2016. Le snapshot publié ne contient aucune
ligne fondamentale CVC ; le candidat ajoute 29 lignes income, 30 balance, 27
cash-flow et 24 earnings officielles, soit 110 observations, connues au plus
tard le 5 mai 2016.

Boosting n'utilise pas directement ces fondamentaux dans ce profil : ses
features sont des EMA prix choisies à partir des gagnants Legacy. Le refresh SEC
modifie ces gagnants et donc les couples EMA du fold 5. Les tailles restent
strictement identiques — 49 501 lignes d'entraînement, 2 861 de validation et
5 823 de test — mais seulement trois des dix couples EMA sont conservés. Le
modèle global réentraîné fait alors passer CVC du rang 35 au rang 8. L'ablation
attribue précisément l'effet à la famille SEC ; elle ne prétend pas qu'une des
110 lignes CVC, isolément, explique à elle seule le réentraînement global.

Preuves des ablations :

- prix candidats + SEC baseline : Legacy manifeste
  `8ac58b1c510a150de55ef11a5456a6fd205b4c423764997b6d173058ad2a3bec`,
  Boosting `b8ea95f3123d80635f794960f3307e93bc14c3c4b4b72fb52300b955126c3ef6`,
  commun passant
  `39c1b90bc27f82092deb4b9cd14c8b980b331292ca32461b64fb5a11fd78999a` ;
- prix baseline + SEC candidat : Legacy manifeste
  `a58e4a6133d1202ff9fcec0b2281b7879c51aa60d3fc1237b91d0f83f799afa6`,
  Boosting `9ce9c820555e9e4fbee3cd72e3cda4059e8b09ab4877e284558ce2b141162673`,
  puis erreur commune CVC identique au candidat complet.

## Décision et retour arrière

Le pointeur `data/model_inputs/manifests/latest.json` reste byte-identique,
SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`,
sur la composition publiée
`9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad`.

Aucun rollback data n'est nécessaire : aucune publication n'a eu lieu. Les
payloads RAW, candidats, snapshots diagnostics, replays et rapports restent
conservés pour poursuivre l'analyse SEC/CVC. La prochaine promotion devra soit
corriger la politique terminale de manière sourcée, soit produire un candidat
qui passe le replay commun ; masquer CVC, remplacer son rendement par zéro ou
ignorer la gate est interdit.

# Replay du candidat data du 8 octobre 2026

## Verdict

Le refresh complet `20261008_095842`, le replay Legacy, le replay Boosting et le
moteur commun sont terminés. Le portefeuille formé fin juillet est strictement
identique au run publié : 80 lignes sur 80, mêmes titres et mêmes poids. Le
contrôle obligatoire du dernier vintage connu échoue en revanche pour la
décision de fin août : sur 78 lignes, `WDC.US` sort et `CHTR.US` entre uniquement
dans `Boosting · Top 10`, sans changement de poids sur les 77 lignes communes.

Le drift est exhaustivement attribué au vintage prix, mais cette explication ne
rend pas le portefeuille identique. La promotion reste donc bloquée et
`data/model_inputs/manifests/latest.json` reste byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`,
sur la composition de production
`9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad`.

## Comparaisons temporelles obligatoires

| Baseline | Cutoff historique | Décision contrôlée | Résultat |
| --- | --- | --- | --- |
| production publiée | juin 2026 | fin juillet, détention août | exact, 80/80 lignes |
| candidat du 8 septembre | juillet 2026 | fin août, détention septembre | échec, 78/78 avec 1 ajout et 1 retrait |

La seconde comparaison est dérivée automatiquement du dernier portefeuille du
run précédent. Elle ne peut pas être remplacée par une date plus ancienne qui
passerait. Elle montre exactement : ajout `CHTR.US`, retrait `WDC.US`, stratégie
`Boosting · Top 10`, zéro colonne ou poids commun modifié et écart numérique
maximum nul sur les lignes communes.

Jusqu'au cutoff historique de juillet, les 6 716 positions Legacy restent
exactes dans tous les scénarios. Les 88 951 scores Boosting communs restent eux
aussi exacts. Les 1 003 lignes Boosting signalées comme différentes concernent
les cibles ou rendements descriptifs devenus observables, pas les scores. Les
49 ajouts et 49 retraits dus aux révisions SEC sont cantonnés à la variante R&D
`Boosting Top N | Legacy PE universe` et n'affectent aucun portefeuille
canonique.

## Attribution WDC vers CHTR

Les quatre scénarios passent techniquement : baseline complète, prix candidat
avec SEC baseline, prix baseline avec SEC candidat et candidat complet.
L'ablation donne un résultat sans ambiguïté :

| Scénario à fin août | Portefeuille versus vintage précédent |
| --- | --- |
| SEC candidat seulement | exact, 78/78 |
| prix candidat seulement | `WDC.US` remplacé par `CHTR.US` |
| prix et SEC candidats | `WDC.US` remplacé par `CHTR.US` |

Les 16 hashes de modèles, les manifestes de features, le code, la configuration
et le runtime sont identiques. Le score WDC passe de `0,2937767804` à
`0,2891675234`, tandis que CHTR reste à `0,2937456965` et franchit donc la
frontière Top 10.

La cause en entrée est une révision Yahoo extrêmement fine de 12 lignes WDC du
20 août au 4 septembre 2026 : écart maximal d'`adjusted_close` de
`0,0000183544` dollar et écart maximal de rendement journalier de
`0,0000000545`. Le préfixe prix déjà publié reste strictement protégé ; le trou
de contrat concerne la frontière récente, déjà utilisée par le candidat de
septembre mais jamais promue, qui a été retéléchargée en octobre. Ce n'est pas
une lecture de rendement futur par le classement, mais ce n'est pas compatible
avec l'exigence d'identité inter-vintages. `DATA-041` doit donc figer la première
observation canonique de cette frontière tout en conservant chaque RAW révisé.

## Portefeuilles candidats au 7 octobre

Ces portefeuilles sont recalculés sur le candidat non promu. La décision est
datée de septembre, la détention d'octobre et son rendement est encore en cours.

| Portefeuille | Titres et poids |
| --- | --- |
| Legacy Frequency | DELL, HPE, MRNA à 12,5 % ; AMD, CRWD, MU, SNDK à 9,375 % ; MPC, VLO à 6,25 % ; ILMN, MRVL, P, STX à 3,125 % |
| Boosting Top 5 | MRNA, DELL, LULU, CHTR, PODD à 20 % |
| Boosting tendance Top 5 | MRNA, DELL, HPE, MU, STX à 20 % |

Le rapport de performance expose aussi les Top 10, 15 et 20, leurs variantes
tendance et toutes les positions historiques. Le moteur commun a 182 mois de
rendements réalisés d'août 2011 à septembre 2026 ; les 126 lignes du portefeuille
d'octobre sont séparées avec le statut `in_force_return_pending`.

## Performance et risque

Les années complètes utilisées pour le nombre d'années battant SPY, la pire
année et la pire année relative sont 2012 à 2025, soit 14 années. Ces chiffres
sont un diagnostic post-hoc sur le candidat et ne constituent pas une promotion.

| Stratégie | CAGR | Volatilité | Max drawdown | Années > SPY | Pire année | Pire année vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy Frequency | 21,14 % | 26,76 % | -26,49 % | 8/14 | 2018 : -11,13 % | 2023 : -16,41 pp |
| Legacy Equal | 21,89 % | 25,16 % | -25,23 % | 9/14 | 2018 : -8,94 % | 2023 : -12,66 pp |
| Boosting Top 5 | 19,90 % | 34,74 % | -31,37 % | 9/14 | 2024 : -6,58 % | 2024 : -31,47 pp |
| Boosting tendance Top 5 | 25,11 % | 31,76 % | -27,37 % | 8/14 | 2018 : -7,14 % | 2021 : -28,99 pp |
| Boosting tendance Top 10 | 22,80 % | 26,34 % | -25,69 % | 6/14 | 2018 : -8,26 % | 2021 : -14,72 pp |
| SPY total return | 14,40 % | 14,23 % | -23,93 % | référence | — | — |

## Artefacts et lignée

- candidat : `dbaac67b89e67f95a8a7040c07ae560fc13be574c8c463b3f07bcb5b0068e598` ;
- snapshot : `outputs/data_refresh_replay_20261008_095842/composed_history/alpharank_input_20261008_105551_dbaac67b89e6` ;
- manifeste snapshot : `4232b271bcc1b78f99d3926886fe9ac3fb7731588167a528ab8a5f446b9b4f14` ;
- manifeste moteur commun : `3e877be7bef9753556229d2275822879924797342887294cd62d8a5e609ff194` ;
- audit JSON : `outputs/data_refresh_replay_20261008_095842/replay_head7cc22c7/audit_previous_vintage_cutoff_2026_07/refresh_replay_attribution.json`, SHA-256 `79a88df06e436e5d5609478a0ffe56e7349228109fcb03e9f2e580d234f77892` ;
- rapport HTML drift : même dossier, `refresh_replay_report.html`, SHA-256 `eb9d675ca929c327f8864f7914416be0cc8ca6c167f8661b47f92e66b877f6cc` ;
- clés ajoutées : `latest_portfolio_diffs/latest_portfolios_added_keys.parquet`, SHA-256 `f923537d95684bdf4427d5378e1952b54b81808a1fcf4ad261fee624c6a2737f` ;
- clés retirées : `latest_portfolio_diffs/latest_portfolios_removed_keys.parquet`, SHA-256 `1c05f6fee55506eaf22cf2913149aefba255dcc093aa0731cac9b47143ee66f7` ;
- rapport HTML performance : `outputs/performance_reports/data_refresh_20261008_095842/backtest_performance_report.html`, SHA-256 `e4e9dfd292e8ac04f60ba14c8a7355d41c76f35960f0bf1866e669b757b41a92` ;
- manifeste du rapport performance : SHA-256 `1e49f364e4e264261b0ad946421737b8536a4757401f6a9bedb0328ac0ef8766`.

L'acquisition réseau a duré 54 minutes 30 secondes. La campagne de replay et
d'ablations s'est étendue sur environ 5 h 16 ; l'ablation SEC seule a consommé
environ quatre heures à cause de contention mémoire locale, sans changer son
résultat ni ses hashes.

## Statut de validation

Les quatre scénarios communs sont exécutables et l'attribution machine-lisible
porte le statut `explained_data_drift`. Le rapport performance est
`comparison_eligible=true`, `publication_eligible=false` et
`post_hoc_research_diagnostic`. La stabilité de juillet est prouvée ; la
stabilité d'août est réfutée et bloque volontairement la promotion jusqu'à
`DATA-041`. Les 48 tests replay et les 2 tests reporting passent ; les deux HTML
réels sont parseables, leur JavaScript est syntaxiquement valide et ils ne
chargent aucun asset réseau. La documentation et ses liens locaux sont valides.

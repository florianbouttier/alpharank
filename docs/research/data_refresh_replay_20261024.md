# Replay du candidat data du 24 octobre 2026

## Verdict

Le refresh complet du 24 octobre, Legacy, Boosting, la variante tendance et le
moteur commun sont recalculés avec le même code, la même configuration et le
même runtime que le candidat du 8 octobre. Le portefeuille formé fin septembre
et détenu en octobre est strictement identique : 126 lignes sur 126, mêmes
titres et mêmes poids à une tolérance de `1e-12`.

Les 6 744 positions Legacy historiques sont exactes. Les 89 453 scores Boosting
sont exacts ; 138 lignes ont seulement des champs descriptifs de rendement ou
de cible devenus observables. Les portefeuilles Boosting natifs et tendance
restent exacts. Le seul drift de positions porte sur la variante diagnostique
`Boosting Top N | Legacy PE universe` : 23 ajouts et 23 retraits répartis entre
2012, 2017 et 2024.

Les ablations attribuent exhaustivement ce drift au vintage SEC : le scénario
prix candidat avec SEC baseline conserve les 30 525 positions communes, tandis
que SEC candidat avec prix baseline reproduit les 23 ajouts et 23 retraits du
candidat complet. Les nouveaux prix ne modifient aucun score Boosting, aucun
Top 5/10 brut et aucune position. Le statut machine-lisible devient
`explained_data_drift`, mais l'identité historique globale est réfutée ; la
promotion reste bloquée et `data/model_inputs/manifests/latest.json` reste
byte-identique.

## Contrôle inter-vintages obligatoire

| Baseline | Candidat | Décision dérivée | Détention | Résultat |
| --- | --- | --- | --- | --- |
| candidat du 8 octobre | candidat du 24 octobre | septembre 2026 | octobre 2026 | exact, 126/126 |

Ce contrôle utilise le dernier mois de décision déjà formé dans le run
précédent. Octobre 2026 n'est pas utilisé comme mois de décision car il est
encore incomplet au 24 octobre. La dernière preuve marché du portefeuille en
vigueur est datée du 23 octobre 2026.

## Drift historique expliqué

Les 46 clés différentes appartiennent toutes à l'univers diagnostique Legacy
PE. Elles incluent notamment `LEN.US` en 2012, `MOS.US` contre `FCX.US` en août
2017 et `GE.US` sur plusieurs Top N de mars à juin 2024. Les prix de `GE.US`
sont strictement identiques dans le focus d'avril 2024, tandis que le package
SEC contient deux lignes de compte de résultat supplémentaires disponibles
dans le candidat.

Cette attribution ne transforme pas le drift en identité. La variante PE reste
R&D et non promue, mais la gate globale demeure volontairement rouge tant que
le contrat exige l'identité de toutes les variantes incluses dans le replay.

## Portefeuilles en vigueur

La décision de septembre est inchangée par rapport au run du 8 octobre ; il n'y
a donc aucune nouvelle action à acheter du seul fait de ce refresh.

| Portefeuille | Titres et poids |
| --- | --- |
| Legacy Frequency | DELL, HPE, MRNA à 12,5 % ; AMD, CRWD, MU, SNDK à 9,375 % ; MPC, VLO à 6,25 % ; ILMN, MRVL, P, STX à 3,125 % |
| Boosting Top 5 | MRNA, DELL, LULU, CHTR, PODD à 20 % |
| Boosting tendance Top 5 | MRNA, DELL, HPE, MU, STX à 20 % |

Les Top 10, 15 et 20, leurs variantes tendance et les 126 positions exactes
sont consultables dans le rapport de performance. Le rendement d'octobre reste
`in_force_return_pending` et n'est pas injecté dans les KPI.

## Performance et risque

La fenêtre comparable va d'août 2011 à septembre 2026, soit 182 mois réalisés.
Ces chiffres sont des diagnostics post-hoc sur le candidat non promu.

| Stratégie | CAGR | Volatilité | Max drawdown |
| --- | ---: | ---: | ---: |
| Legacy Frequency | 21,14 % | 26,76 % | -26,49 % |
| Legacy Equal | 21,89 % | 25,16 % | -25,23 % |
| Boosting Top 5 | 19,90 % | 34,74 % | -31,37 % |
| Boosting tendance Top 5 | 25,11 % | 31,76 % | -27,37 % |
| Boosting tendance Top 10 | 22,80 % | 26,34 % | -25,69 % |
| SPY total return | 14,40 % | 14,23 % | -23,93 % |

## Artefacts et lignée

- composition candidate : `c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e` ;
- manifeste snapshot : `88988103…3bc8` ;
- manifeste Legacy candidat : `4ea2b90a…df0f` ;
- manifeste Boosting candidat : `f61d6390…332c` ;
- manifeste commun candidat : `f60fcfb9…b82f` ;
- audit brut : `outputs/data_refresh_replay_20261024/replay_head5545e2e/audit_previous_vintage_cutoff_2026_08_clean/refresh_replay_report.json`, SHA-256 `c755e00a…e11` ;
- attribution : même dossier, `refresh_replay_attribution.json`, SHA-256 `642a626e…8bd` ;
- rapport HTML de drift : même dossier, `refresh_replay_report.html`, SHA-256 `56a82a1a…d91` ;
- rapport performance : `outputs/performance_reports/data_refresh_20261024/backtest_performance_report.html`, SHA-256 `d7b0a8c8…105` ;
- manifeste performance : même dossier, `backtest_performance_report_manifest.json`, SHA-256 `01858deb…c2b` ;
- pointeur production inchangé : SHA-256 `5c2d0ec0…642e`, composition `9a2058c9…25ad`.

Les outputs restent ignorés par Git ; ce document et `ROADMAP.md` en portent la
preuve durable. Le premier audit a correctement refusé une comparaison dont le
baseline portait seulement la modification roadmap non committée ; les deux
côtés ont été rejoués dans un worktree propre avant le verdict data-only.

## Statut de validation

Les quatre scénarios — baseline, prix seuls, SEC seul et candidat complet —
passent le moteur commun. L'attribution est exhaustive et les trois comparaisons
code/config/runtime sont identiques. Le rapport performance est
`comparison_eligible=true`, `publication_eligible=false` et
`post_hoc_research_diagnostic`. Le pointeur de production n'est pas déplacé.

Les validations finales comptent 59 tests replay/reporting verts, les
validateurs de documentation et de liens Markdown verts, ainsi que le parse
HTML, la syntaxe JavaScript et l'absence de ressource réseau vérifiés sur les
deux rapports réels.

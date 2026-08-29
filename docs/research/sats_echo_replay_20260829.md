# Replay SATS/ECHO du 29 août 2026

## Verdict

Le prix économique de SATS existait bien en mai 2026 sous la clé fournisseur
`ECHO.US`. La politique `price_ticker_transition_return_overlay_v1` a ajouté 24
séances du 27 avril au 29 mai sous `SATS.US`, à partir des seuls rendements ECHO,
sans modifier une ligne SATS antérieure et sans saisir un prix ou rendement.

Le replay complet passe. SATS conserve exactement son score Boosting
`0,1067675874` et son rang 14 dans l'univers tendance d'avril ; sa cible un mois
passe de `approved_censored_last_observation` avec zéro de compatibilité à
`evaluable` avec un rendement réalisé de `+4,9131095 %` et un excès sur SPY de
`-0,3319799 %`. Top 15 et Top 20 peuvent donc valoriser mai sans remplacer SATS
par un titre moins bien classé.

## Entrées et runs

```text
baseline Legacy : outputs/no_sec_fresh_replay_20260828/legacy/2026-08-28/runs/20260828_184601
baseline Boosting : outputs/no_sec_fresh_replay_20260828/boosting
snapshot candidat : outputs/sats_echo_replay_20260829/composed_history/alpharank_input_20260829_123853_bb1f90a907bb
composition_id : bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375
Legacy candidat : outputs/sats_echo_replay_20260829/legacy/2026-08-29/runs/20260829_143904
Boosting candidat : outputs/sats_echo_replay_20260829/boosting
replay natif : outputs/sats_echo_replay_20260829/common_replay
replay tendance : outputs/sats_echo_replay_20260829/common_replay_causal_trend
```

Le snapshot contient des prix jusqu'au 26 août 2026 et des fondamentaux SEC
acquis jusqu'au 26 août. Il est immuable mais reste un candidat local :
`data/model_inputs/manifests/latest.json` n'a pas été déplacé par ce replay.

Legacy a exécuté 30 essais sur chacune des quatre trajectoires de 17 fenêtres.
Boosting a repris le profil public
`legacy_ema_latest_common_v2_approved_censoring`, 16 folds, seed 42, 100 tours,
features `legacy_winners_pit_ema_only`, sans tuning additionnel ni échantillonnage
SHAP. Les deux replays communs utilisent 180 détentions d'août 2011 à juillet
2026, SPY total return et 10 points de base multipliés par le turnover.

## Attribution avant/après

| Niveau | Résultat |
| --- | --- |
| prix SATS | 24 lignes ajoutées, 0 retirée, 0 ligne commune changée |
| prédictions Boosting | 88 950 → 88 951 lignes ; une ligne SATS mai ajoutée |
| scores/probabilités communs | 0 changement |
| cible SATS d'avril | 0 censuré → +4,9131095 % évaluable |
| classement SATS d'avril | rang 14 sur 216 éligibles, inchangé |
| holdings Legacy | 6 686 lignes, clés et rendements identiques ; SATS jamais détenu |
| performance Legacy | identité exacte des CAGR, Sharpe et drawdowns |
| Top 5/10 natifs et tendance | identité exacte avec le replay du 28 août |
| Top 15/20 tendance | replay désormais complet, zéro rendement censuré sélectionné |

SATS contribue `+0,327541 %` au rendement brut de Top 15 en mai et
`+0,245655 %` à Top 20. Le mois de détention réalise respectivement `+18,3627 %`
net pour Top 15 tendance et `+14,6553 %` net pour Top 20 tendance, contre
`+5,2626 %` pour SPY.

| Stratégie tendance | CAGR | Volatilité | Sharpe | Drawdown max | Turnover moyen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boosting Top 15 | 21,1093 % | 23,9306 % | 0,7985 | -35,3611 % | 48,2660 % |
| Boosting Top 20 | 18,3883 % | 21,7840 % | 0,7523 | -33,8463 % | 47,7861 % |

Ces chiffres complètent la preuve Top 5/10 ; ils ne promeuvent pas la variante.
La validation temporelle de son avantage reste celle déjà documentée : les
intervalles Top 5/10 croisent zéro.

## Rapport et empreintes

Le rapport humain autonome est
`outputs/sats_echo_replay_20260829/ticker_transition_replay_report.html`, SHA-256
`28c677526f652e7580ba6285ae85eed3dfd3145ba0128b4aa63ad7d850eab291`.
Son payload JSON a le SHA-256
`7233ef207d8712884358562c9c960e823ebba6740d19403bdcd5226462ae4aa4`.
Le manifeste de rapport lie ces deux fichiers aux preuves suivantes :

| Artefact | SHA-256 |
| --- | --- |
| snapshot manifest | `1af2779bfd2f5156d83a4a9aad6eea2583bdaed746ee2754867c0963ed2158b9` |
| holdings Legacy | `4f566fba8b553b56088927effd65bcd4e9afefc110c07827238581b861c2f3e1` |
| prédictions Boosting | `bdc0d53a1430a3a6abddaa2d0b9a850c08ba087f100a50d3a27a7e96582581cd` |
| manifeste commun natif | `8e676963554b0b5eeee3b2beee958179b3a409ec1356a47b1124ed33a5b371d8` |
| manifeste commun tendance | `36a17e417c49bc53ddaf1e060d22dc331aace6a58e058eb98cc589b0530a5155` |
| performance commune tendance | `972d41522242152c545e7a066c2c34c3f7dd256fb87e9419d7f0c8982f2fa268` |

L'annonce officielle [EchoStar du passage de SATS à ECHO](https://ir.echostar.com/news-releases/news-release-details/echostar-changing-stocker-ticker-sats-echo-marking-companys-next)
confirme une prise d'effet le 24 juin 2026, avec CUSIP, capital et droits des
actionnaires inchangés. Cette preuve justifie l'identité de sécurité ; les 24
valeurs restent dérivées exclusivement des rendements du vintage fournisseur
`20260820_011146` et sont auditées ligne par ligne.

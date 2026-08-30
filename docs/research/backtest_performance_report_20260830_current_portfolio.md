# Portefeuille en vigueur au 28 août 2026

## Verdict

Les tâches `REPORT-010`, `REPORT-011` et `SITE-007` corrigent l'ambiguïté du
rapport précédent : juillet 2026 reste le dernier mois dont le rendement est
entièrement réalisé, mais le portefeuille effectivement détenu en août est
maintenant affiché dans une section séparée **Portefeuille en vigueur**.

La vue indique explicitement :

- dernière séance observée : **28 août 2026** ;
- mois de décision : **juillet 2026** ;
- mois de détention : **août 2026** ;
- statut : **rendement mensuel non réalisé**.

Août n'est ajouté ni aux courbes, ni aux KPI, ni aux model cards. Le rapport ne
fabrique donc pas un rendement mensuel à partir d'un mois incomplet.

## Portefeuilles exposés

| Stratégie | Positions | Somme des poids |
| --- | ---: | ---: |
| Legacy · Frequency | 15 | 100 % |
| Legacy · Equal | 15 | 100 % |
| Boosting · Top 5 | 5 | 100 % |
| Boosting · Top 10 | 10 | 100 % |
| Boosting · Top 15 | 15 | 100 % |
| Boosting · Top 20 | 20 | 100 % |
| Boosting tendance · Top 5 | 5 | 100 % |
| Boosting tendance · Top 10 | 10 | 100 % |
| Boosting tendance · Top 15 | 15 | 100 % |
| Boosting tendance · Top 20 | 20 | 100 % |

Le payload contient 130 lignes. À titre de contrôle lisible, le Top 5 Boosting
natif est `BSX.US`, `CHTR.US`, `INTU.US`, `LULU.US`, `ZTS.US` ; le Top 5
Boosting tendance est `DELL.US`, `MU.US`, `SNDK.US`, `STX.US`, `WDC.US`.

## Pourquoi le 28 août est légitime

Le portefeuille d'août porte `decision_month=2026-07-01` et
`holding_month=2026-08-01`. Son signal ne dépend donc pas d'un mois d'août
terminé. Le snapshot du replay contient des prix au plus tard au 26 août et la
preuve de prolongation prix explicitement fournie atteint le 28 août ; elle
sert uniquement à dater le panier encore en vigueur, jamais à recalculer le
signal ou les KPI.

Le replay exécuté le 28 août publiait seulement Legacy et les variantes
Boosting Top 5/Top 10 demandées à ce run. Les 30 lignes Boosting communes et les
30 lignes Legacy courantes sont identiques à celles du replay utilisé par le
rapport. Les Top 15/Top 20 ont été matérialisés dans le replay du 29 août, mais
à partir du même mois de décision de juillet et d'un snapshot dont la date de
marché maximale est le 26 août : aucune information du 29 août n'entre dans
leurs sélections.

## Lignée

```text
snapshot : outputs/sats_echo_replay_20260829/composed_history/alpharank_input_20260829_123853_bb1f90a907bb
composition_id : bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375
Legacy : outputs/sats_echo_replay_20260829/legacy/2026-08-29/runs/20260829_143904
holdings Boosting : outputs/sats_echo_replay_20260829/common_replay_causal_trend/boosting_live_score_holdings.parquet
preuve au 28 août : data/open_source/official/runs/20260830_001504/price_return_extension_audit.parquet
run Legacy du 28 août comparé : outputs/no_sec_fresh_replay_20260828/legacy/2026-08-28/runs/20260828_184601
run Boosting du 28 août comparé : outputs/no_sec_fresh_replay_20260828/common_replay_causal_trend_top5_top10
implémentation AlphaRank : 346359331cefd3cbdabb30690500b16cc6d9dbe2
publication Portfolio : 1d42a2c21b726d4190f0ce9e3d3d27620525f25c
```

Le manifeste de schéma 4 hash les holdings live Boosting, les holdings détaillés
Legacy, la preuve de marché et les sources historiques déjà présentes.

## Artefacts publiés

```text
rapport source : outputs/performance_reports/sats_echo_20260830_current_portfolio/backtest_performance_report.html
manifeste source : outputs/performance_reports/sats_echo_20260830_current_portfolio/backtest_performance_report_manifest.json
copie site : ../portfolio/frontend/public/research/backtest_performance_report.html
copie build : ../portfolio/frontend/dist/research/backtest_performance_report.html
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| HTML source, copie publique et build | `51675425c854eb9f227c74756a049f380963d2656195de6d8e6027599425a871` | 14 584 478 octets |
| manifeste source, copie publique et build | `a2ca6d192e8a4906e93e729c7b1b9ff4e868060f00d90b99ecfbc62bc41cc1e5` | 9 956 octets |

## Validations exécutées

- test ciblé du payload et du HTML : 2 tests passés ;
- Ruff et format des fichiers Python touchés : passés ;
- syntaxe JavaScript embarquée : passée ;
- validation documentaire et liens Markdown : passés ;
- génération réelle : 180 mois réalisés jusqu'en juillet, 130 positions
  courantes sur dix stratégies, poids total de 100 % pour chacune ;
- parité du run du 28 août : Legacy identique et quatre variantes Boosting
  Top 5/Top 10 identiques ;
- source, copie publique et copie build byte-identiques ;
- build Vite : passé ;
- QA navigateur : titre et quatre dates/statuts visibles, changement vers
  Boosting tendance Top 5 validé à cinq lignes et 100 %, aucune erreur console ;
- pushes vérifiés : `origin/main=3463593` et `origin/master=1d42a2c`.

Le contrôle global de taille Python reste rouge sur deux régressions SEC
préexistantes et hors périmètre. Aucun fichier modifié par `REPORT-010` ou
`REPORT-011` n'est signalé par ce contrôle.

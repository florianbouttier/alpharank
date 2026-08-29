# Rapport comparatif SATS/ECHO du 30 août 2026

## Verdict

Les tâches [`REPORT-003`](../performance_reporting_standard.md) et
`REPORT-004` ont rendu le rapport SATS/ECHO directement comparable : chaque KPI
affiche les onze stratégies à côté de SPY, les courbes sont multisélectionnables
et les model cards cumulées et annuelles respectent les deux bornes de la
fenêtre active.

Le contenu économique est inchangé par rapport à la preuve du
[29 août](backtest_performance_report_20260829.md) : 180 mois d'août 2011 à
juillet 2026, 11 séries, 33 KPI, 136 fenêtres annuelles, 1 980 observations
mensuelles et 24 392 lignes de holdings. La comparaison mécanique reste valide,
mais aucun modèle n'est promu : `comparison_eligible=true`,
`publication_eligible=false` et
`methodology_status=post_hoc_research_diagnostic`.

## Artefacts publiés

```text
rapport source : outputs/performance_reports/sats_echo_20260830/backtest_performance_report.html
manifeste source : outputs/performance_reports/sats_echo_20260830/backtest_performance_report_manifest.json
copie site : ../portfolio/frontend/public/research/backtest_performance_report.html
copie manifeste site : ../portfolio/frontend/public/research/backtest_performance_report_manifest.json
générateur AlphaRank : 34d24b4e9d6eb73730d27a1180a67587d03b866d
publication Portfolio : 71a73c5ac569612013de2a92051bb29571000d7f
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| HTML source, copie site et build | `6f4dd8961d7e89531dbc952aa58946feb1091c2ef7fa5cda9e142a1b19354d90` | 1 001 003 octets |
| manifeste source, copie site et build | `7cd305f108b54e1cc6ef606ec8abb2750a617ee3dd88ac62b6724127874f99f8` | 8 538 octets |

Portfolio sert les deux fichiers sans recalcul. Le HTML reste autonome, sans
asset réseau, et son manifeste désigne le snapshot candidat exact.

## Contrat d'interaction vérifié

- les six cartes principales affichent chacune les onze stratégies et signalent
  la surperformance ou sous-performance par rapport à SPY selon le sens
  économique du KPI ;
- le tableau complet possède 13 colonnes : KPI, onze stratégies et définition ;
- le multiselect affiche librement les onze courbes de richesse et de drawdown ;
- avec `début=2015` et `fin=2019`, les matrices cumulée et annuelle contiennent
  exactement les colonnes 2015, 2016, 2017, 2018 et 2019 ;
- l'onglet annuel CAGR affiche le rendement composé de l'année isolée ; une
  année de bord partielle reste explicitement partielle ;
- les valeurs viennent exclusivement de `metric_windows` produit par
  `alpharank.portfolio.performance.portfolio_period_statistics()`.

## Lignée inchangée

```text
snapshot : outputs/sats_echo_replay_20260829/composed_history/alpharank_input_20260829_123853_bb1f90a907bb
composition_id : bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375
Legacy : outputs/sats_echo_replay_20260829/legacy/2026-08-29/runs/20260829_143904
replay commun : outputs/sats_echo_replay_20260829/common_replay_causal_trend
commit capturé par le replay : e21f90800c63b75e2ed8276d1bab8959e493bae3
profil : legacy_ema_latest_common_v2_approved_censoring
```

Les six hashes d'entrée sont identiques à la preuve du 29 août. `REPORT-003`
ne change aucun signal, rendement, portefeuille, snapshot, coût, benchmark ou
statut de promotion.

## Validations exécutées

- suite AlphaRank complète : 533 tests passés ;
- tests reporting et fenêtres annuelles : 5 passés ;
- Ruff, format, mypy ciblé et syntaxe JavaScript : passés ;
- documentation, liens Markdown et inventaires code/dossiers : passés ;
- build Vite Portfolio : passé, avec copies `public/` et `dist/` byte-identiques ;
- QA navigateur du fichier servi : 11 courbes, 13 colonnes, deux matrices
  2015–2019 et aucun avertissement ni erreur console ;
- pushes vérifiés : `origin/main=34d24b4` pour le standard puis
  `origin/master=71a73c5` pour la copie site.

Le contrôle global de taille AlphaRank conserve deux écarts antérieurs dans des
fichiers SEC étrangers à ce lot. Le validateur documentaire Portfolio reste
rouge sur le dossier utilisateur non suivi
`frontend/public/research/downloads/`; ce lot ne l'a ni modifié ni indexé.

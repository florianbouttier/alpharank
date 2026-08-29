# Rapport de performance SATS/ECHO du 29 août 2026

## Verdict

Le standard [`REPORT-001`](../performance_reporting_standard.md) a été exécuté
sur le replay SATS/ECHO complet puis publié dans le portail Portfolio. Le rapport
couvre 180 mois d'août 2011 à juillet 2026, 11 séries, 33 KPI, 136 fenêtres
annuelles, 1 980 observations mensuelles et 24 392 lignes de holdings.

La comparaison mécanique est valide, mais cette publication ne promeut aucun
modèle : le snapshot reste candidat, `publication_eligible=false` et les
variantes Boosting tendance restent des diagnostics post-hoc de R&D.

## Artefacts publiés

```text
rapport source : outputs/performance_reports/sats_echo_20260829/backtest_performance_report.html
manifeste source : outputs/performance_reports/sats_echo_20260829/backtest_performance_report_manifest.json
copie site : ../portfolio/frontend/public/research/backtest_performance_report.html
copie manifeste site : ../portfolio/frontend/public/research/backtest_performance_report_manifest.json
générateur AlphaRank : 4a6ecb5dc15f2d1aaff93a4129f5b722ef3832a9
publication Portfolio : 7e66fa51cb168a856d2d0d643dbc9a3947895ef3
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| HTML source et copie site | `2e66021c549136299a08bc046fab5c808088b8ec1d6ea96796432d45414c80b3` | 989 729 octets |
| manifeste source et copie site | `bc5ddfdf5b5fdeddd027465fabb005382efe159b9e631d3c05660afac5e1159b` | 8 537 octets |

Le HTML est autonome : données, styles et JavaScript sont embarqués sans asset
réseau. Portfolio ne recalcule aucun KPI ; le site copie les deux artefacts et
ajoute seulement l'onglet de navigation.

## Contenu fonctionnel

- sélection de chaque stratégie ou de toutes les stratégies ;
- fenêtre temporelle de 2011 à 2026, avec 2011 explicitement partiel ;
- 33 KPI précalculés par `alpharank.portfolio` pour chaque fenêtre annuelle ;
- courbes de richesse et de drawdown issues des rendements mensuels communs ;
- matrices Viridis CAGR, volatilité et drawdown par année de départ ;
- consultation et export des 24 392 positions historiques par stratégie, mois
  et ticker ;
- cartes méthodologiques, pseudo-codes, statuts et lignée complète.

Les holdings sources ne portent aucun secteur observable. Les deux KPI de
concentration sectorielle sont donc rendus indisponibles, jamais convertis en
une concentration artificielle de 100 % sur une catégorie inconnue.

## Performance sur la fenêtre commune complète

| Série | CAGR | Volatilité | Drawdown max |
| --- | ---: | ---: | ---: |
| Legacy Frequency | 19,6430 % | 26,5670 % | -26,4931 % |
| Legacy Equal | 20,6690 % | 25,0445 % | -25,2320 % |
| Boosting Top 5 | 19,9416 % | 35,9188 % | -37,2782 % |
| Boosting Top 10 | 19,9271 % | 31,4932 % | -31,3583 % |
| Boosting Top 15 | 18,8992 % | 29,7037 % | -32,7137 % |
| Boosting Top 20 | 17,8061 % | 28,6770 % | -31,8439 % |
| Boosting tendance Top 5 | 23,6320 % | 31,7346 % | -27,3678 % |
| Boosting tendance Top 10 | 21,9512 % | 26,1454 % | -24,7904 % |
| Boosting tendance Top 15 | 21,1093 % | 23,9306 % | -35,3611 % |
| Boosting tendance Top 20 | 18,3883 % | 21,7840 % | -33,8463 % |
| SPY total return | 14,3975 % | 14,3014 % | -23,9272 % |

Ces valeurs décrivent le replay commun depuis août 2011. Elles ne constituent
ni une recommandation d'achat ni une preuve de stabilité hors échantillon des
variantes post-hoc.

## Lignée des entrées

```text
snapshot : outputs/sats_echo_replay_20260829/composed_history/alpharank_input_20260829_123853_bb1f90a907bb
composition_id : bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375
Legacy : outputs/sats_echo_replay_20260829/legacy/2026-08-29/runs/20260829_143904
replay commun : outputs/sats_echo_replay_20260829/common_replay_causal_trend
commit capturé par le replay : e21f90800c63b75e2ed8276d1bab8959e493bae3
profil : legacy_ema_latest_common_v2_approved_censoring
```

| Entrée | SHA-256 |
| --- | --- |
| manifeste commun | `36a17e417c49bc53ddaf1e060d22dc331aace6a58e058eb98cc589b0530a5155` |
| performances mensuelles communes | `e521d159329e890091f01c46d33c5371b3058a5d483188dcaa4e328d12f76700` |
| holdings communs | `422b87b4f052e2ed7e650f7554b65f6f817372c10d0d9195ca43c3723bea3fc2` |
| performances mensuelles Legacy | `936a609c9a04db6f4138ef7e7c0888efd9e03331e886b59fdd1821f36df342ba` |
| holdings Legacy | `4f566fba8b553b56088927effd65bcd4e9afefc110c07827238581b861c2f3e1` |
| manifeste du snapshot | `1af2779bfd2f5156d83a4a9aad6eea2583bdaed746ee2754867c0963ed2158b9` |

Le CAGR Legacy Frequency du rapport vaut exactement
`0,19642986941908291`, comme `comparison_common_performance.csv` du replay.

## Validations exécutées

- suite AlphaRank complète : 533 tests passés ;
- tests catalogue après régénération : 4 passés ;
- Ruff et format sur les fichiers modifiés : passés ;
- mypy strict sur les modules Portfolio modifiés : passé ;
- syntaxe JavaScript embarquée : passée ;
- documentation, liens, inventaires de code et de dossiers : passés ;
- revue visuelle du rapport autonome : chargement, filtres, matrices, holdings et
  absence d'erreur console vérifiés ;
- build Vite Portfolio : passé ;
- routes locales du site et du rapport : HTTP 200 ;
- copie Portfolio : hashes identiques à la source.

Le contrôle global de taille conserve deux écarts antérieurs hors de ce lot,
et le mypy global conserve une erreur antérieure dans
`terminal_event_registry.py`. Aucun nouveau fichier ou fonction de ce rapport
ne dépasse les plafonds.

# Corrélations et richesse relative SATS/ECHO du 30 août 2026

## Verdict

Les tâches `REPORT-008` et `REPORT-009` distinguent désormais deux notions dans
le laboratoire de portefeuille composé :

- la corrélation de Pearson des rendements mensuels mesure la dépendance entre
  les poches et entre le portefeuille composé et SPY ;
- la richesse relative divise la richesse composée du portefeuille par celle
  du SPY depuis le début de la fenêtre. Elle ne divise jamais deux pourcentages
  de performance et ne s'appelle pas corrélation.

La matrice contient uniquement les stratégies cochées et change avec la fenêtre
active. Le KPI de corrélation du portefeuille au SPY est pré-calculé pour les
1 023 combinaisons et les 136 fenêtres. La courbe de richesse relative est une
projection des rendements mensuels déjà publiés.

## Résultat des deux Boosting Top 5

Le preset `Boosting Top 5 + tendance` combine à 50/50 Boosting Top 5 natif et
Boosting tendance Top 5.

| Mesure, août 2011 à juillet 2026 | Valeur |
| --- | ---: |
| Corrélation mensuelle entre les deux poches | 0,376 |
| Corrélation mensuelle du portefeuille au SPY | 0,671 |
| CAGR du portefeuille | 23,63 % |
| Volatilité annualisée | 28,08 % |
| Max drawdown | -24,63 % |
| Sharpe | 0,770 |
| Sortino | 1,446 |
| Richesse finale du portefeuille / richesse finale du SPY | 3,204× |

Depuis janvier 2020, la corrélation entre les deux poches vaut 0,330 et celle du
portefeuille au SPY 0,689. La corrélation inter-poches est donc modérée à faible
sur les deux fenêtres observées, sans constituer une garantie future ni suffire
à promouvoir la combinaison.

## Artefacts publiés

```text
rapport source : outputs/performance_reports/sats_echo_20260830_correlation/backtest_performance_report.html
manifeste source : outputs/performance_reports/sats_echo_20260830_correlation/backtest_performance_report_manifest.json
copie site : ../portfolio/frontend/public/research/backtest_performance_report.html
copie build : ../portfolio/frontend/dist/research/backtest_performance_report.html
implémentation AlphaRank : d87802db3995229b90f1f857d879b61392b11c3f
publication Portfolio : 4fdc1b58a15747ac4554019c484bc9bd2891f118
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| HTML source, copie publique et build | `6f14edff128936044e8e2b465eee242d2c4f1b4056c6797d0e04825578b17c47` | 14 577 529 octets |
| manifeste source, copie publique et build | `5a0b0029c21a00acee0fdc56f8ee124b5db4b61c814f084a2a8402449f72e344` | 8 866 octets |

Le schéma 3 du manifeste enregistre sept KPI de combinaison, dont la
corrélation au SPY, et 136 matrices de corrélation entre les dix poches.

## Lignée économique inchangée

```text
snapshot : outputs/sats_echo_replay_20260829/composed_history/alpharank_input_20260829_123853_bb1f90a907bb
composition_id : bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375
Legacy : outputs/sats_echo_replay_20260829/legacy/2026-08-29/runs/20260829_143904
replay commun : outputs/sats_echo_replay_20260829/common_replay_causal_trend
commit capturé par le replay : e21f90800c63b75e2ed8276d1bab8959e493bae3
```

Cette publication ne change aucun signal, score, holding, rendement source,
snapshot, benchmark, coût ou statut de promotion.

## Validations exécutées

- suite AlphaRank complète : 533 tests passés ;
- tests ciblés moteur, payload, HTML, inventaires et limites de dossiers : 10
  passés ;
- Ruff, mypy ciblé, syntaxe JavaScript, documentation et liens : passés ;
- génération réelle : 180 mois, 1 023 combinaisons, sept KPI et 136 fenêtres de
  corrélation ;
- build Vite Portfolio : passé ; source, copie publique et build
  byte-identiques ;
- QA du fichier servi : preset des deux Boosting, matrice 2×2, KPI de corrélation
  au SPY, courbe relative et changement de fenêtre 2020–2026 validés ;
- pushes vérifiés : `origin/main=d87802d` pour l'implémentation et
  `origin/master=4fdc1b5` pour la copie site.

Le contrôle global de taille AlphaRank conserve deux écarts SEC antérieurs. Le
validateur documentaire Portfolio reste rouge uniquement sur le dossier
utilisateur non suivi `frontend/public/research/downloads/` ; aucun de ces
éléments n'a été modifié ou indexé par cette publication.

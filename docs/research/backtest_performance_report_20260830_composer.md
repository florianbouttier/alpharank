# Rapport SATS/ECHO corrigé et portefeuille composé du 30 août 2026

## Verdict

Les tâches `REPORT-005`, `REPORT-006` et `REPORT-007` corrigent puis republient
le rapport demandé. Les cartes, le tableau KPI, les deux graphiques et les deux
model cards montrent uniquement les courbes cochées. Le drawdown est un
graphique pleine largeur placé sous la croissance composée.

Un nouvel onglet permet de combiner les dix stratégies hors SPY. Le portefeuille
attribue chaque mois `1 / N` à chacune des poches cochées ; ses six KPI et ses
rendements mensuels sont sélectionnés parmi 1 023 combinaisons pré-calculées par
le moteur commun. Il s'agit d'un diagnostic post-hoc non promu, sans poids libre,
sans optimisation et sans coût inter-poche ajouté. Les rendements de chaque
poche restent nets de leurs frais propres et les titres communs à plusieurs
stratégies restent exposés dans chacune de ces poches.

## Artefacts publiés

```text
rapport source : outputs/performance_reports/sats_echo_20260830_composer/backtest_performance_report.html
manifeste source : outputs/performance_reports/sats_echo_20260830_composer/backtest_performance_report_manifest.json
copie site : ../portfolio/frontend/public/research/backtest_performance_report.html
copie build : ../portfolio/frontend/dist/research/backtest_performance_report.html
implémentation AlphaRank : 9448bfb6e59dbc7e5dfad545df4dcd6e448caa73
publication Portfolio : 4f16576d745eb85b8ac041f709d6214ac61f5aab
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| HTML source, copie publique et build | `2ab1f46dc60ebfa026c5b7240a84ce58e3aa7ab267df7ced569704bd55e82123` | 12 857 781 octets |
| manifeste source, copie publique et build | `37b60de430f46036bad3b228cf8073309ba82fde443d648708cf7b20625a7eef` | 8 810 octets |

Le manifeste utilise le schéma 2 et enregistre explicitement la méthode
`monthly_equal_weight_strategy_sleeves`, les 1 023 combinaisons et les six KPI
du laboratoire : rendement total, CAGR, volatilité, Sharpe, max drawdown et
Sortino.

## Contrat d'interaction vérifié

- avec le preset `Legacy + SPY`, les six cartes contiennent 12 lignes au total,
  le tableau possède quatre colonnes, chaque model card possède deux lignes et
  les deux graphiques possèdent deux légendes ;
- la croissance composée et le drawdown sont deux panneaux pleine largeur
  superposés ;
- avec Legacy Frequency, Legacy Equal et Boosting tendance Top 5 cochés, le
  laboratoire affiche trois poches à 33,33 %, six KPI et deux courbes contre
  SPY ;
- le passage de la fenêtre complète à 2020–2026 change les valeurs du
  laboratoire en sélectionnant le cube canonique correspondant ;
- le fichier réellement servi depuis Portfolio a été utilisé pour ces contrôles
  et s'est chargé sans écran d'erreur.

## Lignée économique inchangée

```text
snapshot : outputs/sats_echo_replay_20260829/composed_history/alpharank_input_20260829_123853_bb1f90a907bb
composition_id : bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375
Legacy : outputs/sats_echo_replay_20260829/legacy/2026-08-29/runs/20260829_143904
replay commun : outputs/sats_echo_replay_20260829/common_replay_causal_trend
commit capturé par le replay : e21f90800c63b75e2ed8276d1bab8959e493bae3
```

Cette publication ne change aucun signal, score, holding, rendement source,
snapshot, benchmark, coût propre aux stratégies ou statut de promotion. Elle
ajoute une projection de diversification entre séries déjà produites.

## Validations exécutées

- suite AlphaRank complète : 533 tests passés ;
- tests ciblés portefeuille, payload, HTML, inventaire et limites de dossiers :
  10 passés ;
- Ruff, mypy ciblé, syntaxe JavaScript, documentation, liens Markdown et
  inventaires : passés ;
- génération SATS/ECHO : 180 mois, 11 séries individuelles, 10 poches et 1 023
  combinaisons ;
- build Vite Portfolio : passé ; source, copie publique et build
  byte-identiques ;
- QA navigateur du fichier servi : sélection globale à deux courbes,
  drawdown séparé, combinaison à trois poches et fenêtre 2020–2026 validés ;
- pushes vérifiés : `origin/main=9448bfb` pour l'implémentation et
  `origin/master=4f16576` pour la copie site.

Le contrôle global de taille AlphaRank conserve deux écarts SEC antérieurs et
le mypy global un écart antérieur dans `terminal_event_registry.py`. Le
validateur documentaire Portfolio reste rouge uniquement sur le dossier
utilisateur non suivi `frontend/public/research/downloads/` ; aucun de ces
éléments n'a été modifié ou indexé par cette publication.

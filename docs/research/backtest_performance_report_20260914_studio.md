# Studio de comparaison unifié du 14 septembre 2026

## Verdict

Les tâches `REPORT-012` et `REPORT-013` réunissent dans un même panneau la
comparaison de stratégies et la composition équipondérée de plusieurs poches.
La période, les six cartes de synthèse, le graphique principal, le tableau KPI
et les model cards restent synchronisés quand le mode change. SPY total return
est ajouté automatiquement et ne peut pas être retiré.

La publication modifie uniquement le rendu. Le calendrier, les rendements
mensuels, les holdings réalisés, le portefeuille en vigueur, les KPI, les 1 023
combinaisons, la lignée, les méthodologies et le statut sont identiques au
rapport précédent après décodage. Leur sérialisation canonique porte le hash
`a91a6ffc09976f0a7a4ea1957b4bf6d0062546495e384b060767bdf0247b7ad3`.

## Interaction publiée

- mode **Stratégies** : multisélection des dix stratégies investissables, avec
  SPY permanent dans les cartes, courbes, KPI et model cards ;
- mode **Portefeuille composé** : sélection de plusieurs poches équipondérées,
  comparées à SPY dans exactement le même panneau ;
- vues **Performance**, **Drawdown** et **Vs SPY** sur le même canvas ;
- écarts numériques au SPY et états visuels vert/rouge selon le sens économique
  du KPI ;
- résumé dynamique du leader ou de la composition et nombre de KPI de synthèse
  qui surpassent SPY ;
- tableau complet, model cards cumulées/incrémentales et corrélations accessibles
  dans des tiroirs repliables, sans sections dupliquées plus bas.

Le portefeuille composé reste un laboratoire post-hoc non promu. Les poches
sont équipondérées et rééquilibrées mensuellement ; le rapport n'optimise pas
leurs poids.

## Lignée et artefacts

```text
snapshot : outputs/data_refresh_replay_20260908/composed_history_rollforward/alpharank_input_20260908_233538_35f0244f39cc
composition_id : 35f0244f39cc8afffaf1af08886d7f3c1ab8a2fdf4ba5043782118eba7978421
replay commun : outputs/data_refresh_replay_20260908/replay_20260909_head424a3b4/candidate/common_end_august_causal_trend
run Legacy : outputs/data_refresh_replay_20260908/replay_20260909_head424a3b4/candidate/legacy/2026-09-09/runs/20260909_020319
rapport source : outputs/performance_reports/data_refresh_20260914_studio/backtest_performance_report.html
copie publique : ../portfolio/frontend/public/research/backtest_performance_report.html
copie build : ../portfolio/frontend/dist/research/backtest_performance_report.html
implémentation AlphaRank : 63b1084849bfe17cd6ada6819d630a92aec515a2
publication Portfolio : b6f3900252a44195e1cfb2f3a9e89df2f88673a6
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| HTML source, copie publique et build | `46db35c3b6963c41c46fbcc35826415234224cc8a6f6e5224e414c73b8f15bba` | 14 606 051 octets |
| manifeste source et copie publique | `4c6583e4bcb13e48226ece0e965337f94ef91999d97d769fb8757aa5e748e803` | 10 315 octets |

## Validations exécutées

- 18 tests du package reporting passés ;
- Ruff, format, syntaxe JavaScript et liens Markdown passés ;
- génération réelle passée sur 181 mois, d'août 2011 à août 2026 ;
- comparaison avant/après des sections économiques : toutes identiques ;
- runtime DOM local : six cartes, bascule simple/composé, trois vues, matrice de
  corrélation et filtre janvier 2018–décembre 2025 de 96 mois validés ;
- build Vite : passé, avec copie publique et copie `dist` byte-identiques ;
- pushes vérifiés : `origin/main=63b1084` et `origin/master=b6f3900`.

L'outil de navigateur a refusé la navigation automatisée vers l'URL locale
`file://` pour raison de sécurité. La QA visuelle automatisée n'est donc pas
revendiquée ; elle est remplacée ici par le test du runtime DOM, la syntaxe JS,
le build Vite et la parité byte-à-byte de l'artefact servi. Le validateur de
documentation Portfolio reste rouge à cause du dossier utilisateur non suivi
`frontend/public/research/downloads/`, hors commit de publication.

Le gate global de taille Python conserve deux alertes SEC préexistantes hors
diff. Les trois nouveaux modules du studio mesurent 144, 189 et 103 lignes, et
le dossier reporting contient 15 fichiers Python, dans la cible du standard.

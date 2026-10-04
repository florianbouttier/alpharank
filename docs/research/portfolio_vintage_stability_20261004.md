# Stabilité inter-vintages vérifiée le 4 octobre 2026

## Réponse directe

Oui, le dernier contrôle disponible a fonctionné : le nouveau replay reproduit
exactement le portefeuille que le run précédent avait formé à la même date.
La date n'est plus choisie manuellement par l'opérateur ; elle est dérivée des
derniers portefeuilles Legacy et Boosting de la baseline, dont les dates doivent
être alignées.

Il n'existe toutefois aucun nouveau refresh ou backtest après le 9 septembre
dans le dépôt. Cette preuve du 4 octobre réexécute donc l'audit sur les artefacts
immuables du dernier couple baseline/candidat ; elle ne prétend pas comparer un
run de septembre à un run d'octobre inexistant.

## Résultat du test par date

| Champ | Résultat |
| --- | --- |
| source de la date | dernier portefeuille live du run précédent |
| mois de décision dérivé | `2026-07-01`, portefeuille formé fin juillet |
| mois de détention | `2026-08-01` |
| lignes run précédent | 80 |
| lignes nouveau replay à la même date | 80 |
| ajouts | 0 |
| retraits | 0 |
| poids modifiés | 0 |
| différence numérique maximale | 0 |
| verdict | `passed`, strictement identique |

Les 80 lignes couvrent 30 positions Legacy et 50 positions Boosting. Le test ne
lit pas le rendement réalisé d'août pour sélectionner les titres : il compare
uniquement les stratégies, dates de décision/détention, tickers et poids cibles
déjà formés.

## Règle désormais non contournable

`REPLAY-009` transforme la preuve de `REPLAY-008` en invariant permanent :

1. chaque audit complet résout les derniers mois Legacy et Boosting du run
   baseline et échoue s'ils ne sont pas alignés ;
2. le candidat doit être rejoué à cette date exacte ;
3. le JSON contient toujours `vintage_portfolio_stability` avec le statut,
   la date, le mois détenu et les écarts ;
4. le HTML expose « Stabilité par date » dans sa navigation ;
5. toute différence de titre ou poids classe le contrôle `failed` et bloque ;
6. si le replay commun est bloqué, le rapport dit explicitement
   `not_evaluable_common_replay_blocked` au lieu d'omettre la preuve ;
7. l'ancien argument `--latest-decision-month` ne peut plus sélectionner une
   date plus ancienne : s'il est fourni, il doit égaler la date automatique.

## Artefacts

```text
racine : outputs/data_refresh_replay_20260908/replay_20260909_head424a3b4/audit_vintage_stability_20261004
audit machine : refresh_replay_report.json
attribution finale : refresh_replay_attribution.json
rapport humain : refresh_replay_report.html
```

| Artefact | SHA-256 | Taille |
| --- | --- | ---: |
| audit machine | `e97a3bbd84bfed2432982633ff5643f8e2d1433f29f9def523edd081b6e35f7b` | 21 702 octets |
| attribution finale | `3037b5ea3833b8b37a762ec062854038065a271ef575613daa407fe0ca677d47` | 26 888 octets |
| rapport HTML | `916111044e6545c3fc906b5d61303f36d2b6a57dd17a93abcf0f2403d6cc4747` | 26 519 octets |

Le statut brut de l'audit historique reste `unexplained_portfolio_drift`, puis
les ablations déjà validées concluent `explained_data_drift`. Cette question
est distincte du contrôle inter-vintages, qui passe exactement. La promotion
reste refusée pour la revue data déjà documentée ; le pointeur de production
est inchangé au hash
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`.

## Validations du changement

- 20 tests ciblés de drift et de rendu HTML passent ;
- la suite complète passe avec 544 tests et zéro échec ;
- Ruff ciblé et différentiel, inventaires, documentation et liens passent ;
- le contrôle global de taille conserve deux alertes SEC préexistantes hors
  périmètre (`_sec_explorer_html.py` et `build_sec_output_package.py`) ; aucun
  fichier modifié par `REPLAY-009` n'ajoute de dépassement.

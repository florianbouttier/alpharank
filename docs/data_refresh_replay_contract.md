# Contrat de refresh et de reproductibilité des portefeuilles

**Rôle : contrat canonique obligatoire avant promotion d'un nouveau snapshot.**

## 1. Règle observable

Après un refresh complet, AlphaRank doit aboutir à l'un des deux résultats
suivants sur la période historique commune :

1. les portefeuilles Legacy et Boosting sont identiques à ceux du snapshot
   publié précédent ;
2. chaque différence de titre, poids ou mois est reliée à une différence data
   précise, sourcée et datée.

Une différence seulement constatée dans un CAGR, un score final ou un hash ne
constitue pas une explication. Un écart non attribué bloque la promotion : en
cas de doute, on arrête.

## 2. Sens opérationnel de « retélécharger toutes les données »

Le refresh interroge de nouveau toutes les sources que leur contrat autorise à
rafraîchir : historique Yahoo complet de l'univers actif et de SPY depuis le
cutoff demandé, Companyfacts SEC complet, Submissions SEC, documents de filing
nécessaires et fallbacks déclarés disponibles. Chaque requête produit un reçu,
y compris en cas d'échec ou de contenu identique.

Les historiques inactifs ou delistés qu'un fournisseur ne sert plus ne sont pas
inventés ni supprimés. Leur preuve EODHD ou leur dernière observation raw
validée est conservée par hash, avec le statut `retained_not_redownloadable`.
Une source sans credential ou indisponible reçoit un statut explicite ; elle ne
disparaît pas silencieusement du périmètre annoncé.

Le téléchargement construit un candidat isolé. Il ne déplace jamais
`data/model_inputs/manifests/latest.json` avant la fin de tous les contrôles.

## 3. Baseline et cutoff commun

Avant le réseau, le run fige :

- le manifeste du snapshot publié et tous ses hashes ;
- les runs Legacy et Boosting de référence ;
- le commit, les hashes de code critique, les dépendances et les seeds ;
- les configurations, exclusions, politiques de prix et conventions
  d'exécution ;
- le dernier mois de décision et de détention complet dans les deux replays.

La comparaison historique s'arrête au minimum de ces cutoffs. Les nouvelles
dates ajoutées par le refresh sont rapportées séparément et ne peuvent pas
faire apparaître artificiellement un drift du passé.

## 4. Chaîne de comparaison obligatoire

Le rapport rapproche dans cet ordre :

1. **acquisition** : requête, statut, hash du payload, plage et couverture ;
2. **raw/stg/def/mart** : schéma, clés ajoutées/retirées, valeurs modifiées,
   source choisie, `available_at` et règle de sélection ;
3. **snapshot** : composition, hashes physiques et logiques ;
4. **univers** : identité de titre, composition S&P historique, exclusions,
   liquidité/OHLC et éligibilité fondamentale point-in-time ;
5. **signal Legacy** : paramètres annuels, classement EMA et votes finaux ;
6. **signal Boosting** : folds, catalogue de features, cibles, scores et rangs ;
7. **portefeuille** : titre, mois de décision, mois de détention, poids et
   rendement avec la clé
   `(strategy, decision_month, holding_month, ticker)` ;
8. **simulation** : rendement brut/net, turnover, coûts et benchmark commun.

Pour chaque position différente, le rapport donne le premier étage où la
divergence apparaît et les clés data candidates qui l'expliquent. Une révision
postérieure au cutoff de connaissance d'une décision passée est signalée comme
risque causal bloquant, même si l'impact économique paraît faible.

## 5. Classification des résultats

| Statut | Signification | Promotion |
| --- | --- | --- |
| `identical_historical_portfolios` | mêmes positions et poids au cutoff commun | possible après les autres gates |
| `explained_data_drift` | différences exhaustivement reliées à des révisions data recevables | revue humaine obligatoire |
| `blocked_before_replay` | acquisition ou candidat invalide avant les modèles | interdite |
| `code_config_runtime_drift` | data identiques mais environnement de calcul différent | interdite jusqu'à rapprochement |
| `unexplained_portfolio_drift` | au moins une position différente sans cause démontrée | interdite |

Le rapport ne transforme jamais automatiquement `explained_data_drift` en
approbation. Une correction ou une révision recevable crée une nouvelle version
ou un overlay sourcé ; aucun snapshot publié n'est réécrit.

## 6. Artefacts minimaux

Un run conserve sous une seule racine identifiée :

- `refresh_manifest.json` et les reçus réseau ;
- `data_drift_summary.json` et les différences tabulaires détaillées ;
- les manifestes des deux snapshots et des quatre runs comparés ;
- les différences d'univers, scores, positions, poids et rendements ;
- `refresh_replay_report.json`, conclusion machine-lisible unique ;
- la commande, le runtime, le commit et les hashes du code critique.

Les gros Parquet restent dans les sorties de run, hors Git. Une synthèse datée
peut être conservée sous `docs/research/`, avec les identifiants et hashes
nécessaires pour retrouver la preuve complète.

## 7. Procédure

1. résoudre et sceller la baseline publiée ;
2. exécuter le refresh complet vers un candidat non promu ;
3. arrêter et produire `blocked_before_replay` si une gate data échoue ;
4. sinon construire le snapshot candidat et rejouer Legacy avec sa
   configuration scellée ;
5. faire consommer au Boosting exactement l'`input_snapshot/` et les holdings
   Legacy du nouveau run ;
6. exécuter le replay commun et l'audit baseline/candidat ;
7. refuser la promotion tant que le statut n'est pas recevable et toutes les
   preuves présentes.

Le fait d'arrêter avant les modèles lorsqu'un candidat est corrompu est une
preuve de fiabilité, pas un replay réussi. Le rapport doit alors nommer les
sources, clés et seuils qui ont empêché les deux backtests de consommer cette
donnée.

## 8. Commande canonique d'audit

Après un candidat valide et les deux replays, la comparaison s'exécute avec :

```bash
python scripts/validation/audit_refresh_replay.py \
  --baseline-snapshot <snapshot-publie> \
  --candidate-snapshot <snapshot-candidat> \
  --baseline-legacy <run-legacy-publie> \
  --candidate-legacy <run-legacy-candidat> \
  --baseline-boosting <run-boosting-publie> \
  --candidate-boosting <run-boosting-candidat> \
  --baseline-common <replay-commun-publie> \
  --candidate-common <replay-commun-candidat> \
  --historical-cutoff YYYY-MM-DD \
  --output-dir <racine-audit>
```

Si une gate data arrête le candidat avant les modèles, le même outil produit
la conclusion obligatoire sans lancer les backtests sur une donnée invalide :

```bash
python scripts/validation/audit_refresh_replay.py \
  --baseline-snapshot <snapshot-publie> \
  --failed-refresh-run <data/open_source/official/runs/RUN_ID> \
  --output-dir <racine-audit>
```

Le code retour vaut zéro seulement pour
`identical_historical_portfolios`. Toute autre classification retourne `2`,
conserve les clés de divergence en Parquet et interdit la promotion jusqu'à la
revue prévue par ce contrat.

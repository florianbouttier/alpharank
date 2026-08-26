# AlphaRank — instructions agents

Ce fichier est l'unique source d'instructions automatiquement normative pour
les agents travaillant dans ce dépôt. Ne jamais maintenir une seconde copie des
règles sous un nom voisin.

## 1. Ordre de lecture

Avant toute modification :

1. lire `README.md` pour le contexte ;
2. lire `ROADMAP.md`, seule source des priorités actives ;
3. lire `CONTRIBUTING.md` et le standard concerné sous `docs/standards/` ;
4. lire le contrat métier ou le runbook indiqué dans `docs/README.md` ;
5. inspecter le code, les tests, les données et le worktree réels avant de
   proposer ou modifier.

`docs/archive/` et `METHODOLOGY_AUDIT_ROADMAP.md` sont des preuves historiques
détaillées. Ils peuvent expliquer une décision passée mais ne définissent pas
seuls la prochaine priorité.

## 2. Périmètre et source de vérité

- AlphaRank est le dépôt principal pour la méthodologie, les données, les
  backtests et les portefeuilles produits par AlphaRank.
- Le dépôt Portfolio est un consommateur/suivi séparé. Ne pas y placer une
  correction AlphaRank et ne pas utiliser IBKR comme gate de production
  AlphaRank sauf demande explicite.
- Une information de chat ou un ancien rapport ne remplace jamais le code, le
  manifeste, le snapshot ou le contrat canonique courant.
- En cas d'ambiguïté, vérifier les artefacts et signaler clairement ce qui est
  observé, supposé ou historique.

## 3. Autonomie

- Exécuter proactivement les inspections, tests et validations utiles.
- Prendre la plus petite décision réversible compatible avec la tâche.
- Ne demander une clarification que si un choix change réellement la méthode,
  les données publiées, les résultats économiques ou une action externe.
- Si une expérience invalide l'hypothèse, documenter le résultat et passer au
  prochain diagnostic prévu plutôt que masquer l'échec.
- Ne jamais élargir silencieusement le périmètre autorisé.

## 4. Sécurité du worktree et des fichiers

- Le worktree peut contenir des changements appartenant à l'utilisateur ou à
  un autre travail. Les préserver et isoler strictement la tâche courante.
- Ne jamais utiliser `git reset --hard`, `git checkout --`, réécrire l'historique
  ou supprimer un fichier pour obtenir artificiellement un worktree propre.
- Ne pas supprimer de données, snapshots, sorties, logs, secrets ou documents
  sans demande explicite et procédure de récupération.
- Avant un déplacement, inventorier les lecteurs, liens, commandes et hashes.
- Préférer des diffs minimaux et les corrections de cause racine.

## 5. Contrat Git obligatoire

Lire `docs/standards/git.md` avant tout commit.

- Aucun commit sans autorisation explicite de l'utilisateur. Une demande
  explicite d'exécuter un lot de roadmap autorise les commits unitaires de ce
  lot jusqu'à changement de direction.
- Une demande d'exécuter un refresh, un replay, un backtest ou un run de
  production important exige une preuve suivie dans la roadmap et le document
  canonique concerné, puis son commit et son push, même lorsque les données,
  sorties et journaux générés restent volontairement ignorés par Git.
- Relation stricte **1 tâche de roadmap = 1 commit = 1 identifiant**.
- Si une tâche nécessite plusieurs commits, la découper dans la roadmap avant
  de coder. Si deux tâches semblent tenir dans un commit, les séparer.
- Message : `<type>(<TASK-ID>): <résumé impératif>`.
- Le même commit contient l'implémentation, les tests, la documentation et le
  passage de la tâche à `fait`.
- Le corps du commit documente pourquoi, quoi, preuves exécutées, impact
  économique/data et risques restants.
- Ne jamais mélanger reformatage, déplacement et changement métier.
- Vérifier le diff indexé avant le commit ; ne jamais stage des fichiers sans
  rapport, données brutes, snapshots, caches, secrets ou gros outputs.
- Le propriétaire étant seul sur ce dépôt, travailler et committer directement
  sur `main` par défaut. Une branche séparée n'est créée que sur demande
  explicite ou lorsqu'un travail parallèle ne peut pas être isolé autrement.
- L'autorisation d'exécuter une tâche de roadmap inclut son unique commit et son
  push normal. Après chaque commit validé, exécuter immédiatement
  `git push origin main`, puis vérifier que `origin/main` porte le même hash.
- Avant de committer, récupérer les références distantes et vérifier que
  `main` n'a pas divergé de `origin/main`. En cas de divergence ou de rejet du
  push, ne jamais forcer : inspecter et réconcilier explicitement.
- `main-save` est la sauvegarde immuable de l'ancien `main` au commit
  `c1113ab0613c06c8e3deb27e7a7f35d892e80bca`. Ne jamais l'avancer, la fusionner
  ou la supprimer sans demande explicite du propriétaire.
- Aucun force-push. Une éventuelle exception exige une demande portant
  explicitement sur cette opération et une revue de l'impact distant.
- Ne pas amender ou rebaser un commit déjà créé sans demande explicite. Une
  correction ultérieure reçoit une nouvelle tâche et un nouveau commit.

Le hash du commit n'est pas recopié dans ce même commit, car son hash n'existe
qu'après création. Le lien durable est le `TASK-ID`, présent dans la roadmap et
dans le trailer `Roadmap-Task`; le hash se résout avec Git.

## 6. Standards de code et de données

- Tout nouveau code Python suit `docs/standards/python.md`.
- Toute nouvelle donnée, table, transformation ou publication suit
  `docs/standards/data.md`.
- Tout déplacement ou nouveau dossier suit `docs/standards/repository.md`.
- Les standards s'appliquent immédiatement aux nouvelles lignes et aux parties
  significativement modifiées.
- Ne pas reformater ou migrer globalement l'ancien code dans une tâche sans
  rapport. Créer une tâche dédiée avec test de caractérisation.
- La logique réutilisable vit sous `src/alpharank/`; les scripts restent minces.
- Aucun nouveau `sys.path`, repli silencieux, chemin utilisateur codé en dur,
  moteur KPI local ou fichier `final_v2_fixed`.

## 7. Documentation

- `README.md` est l'onboarding court ; `docs/README.md` est l'index canonique.
- Une seule source de vérité par sujet. Mettre à jour le document canonique dans
  le même commit que le changement.
- Ne pas créer de note isolée si un contrat, catalogue ou runbook existe.
- Les preuves datées vont sous `docs/research/` ou `docs/archive/`.
- `METHODOLOGY_AUDIT_ROADMAP.md` conserve le détail historique ; les tâches
  actives et leur ordre vivent dans `ROADMAP.md`.
- Chaque dossier maintenu possède un README court : responsabilité, entrées,
  sorties, enfants et contenu interdit.
- Après changement documentaire ou structurel, exécuter
  `scripts/validate_documentation.py` et le contrôle des liens Markdown.

## 8. Production mensuelle

- Le portefeuille mensuel canonique Legacy est lancé par
  `scripts/run_legacy.py`.
- `scripts/run_backtest.py` et le Boosting restent R&D sauf demande ou promotion
  explicite.
- Lire `docs/monthly_portfolio_runbook.md` avant tout run, replay ou
  rééquilibrage mensuel.
- La donnée de production est le snapshot immuable désigné par
  `data/model_inputs/manifests/latest.json`.
- Ne pas utiliser `data/open_source/output`, `data/sec/output`, les Parquet
  racine ou un dossier `outputs/` comme substitut choisi manuellement.
- Un run résout sa cible au démarrage et conserve le snapshot, manifeste,
  configuration, code/runtime, journaux et artefacts exigés par le runbook.
- Un package diagnostic, quarantiné, partiel ou de réparation n'est pas une
  vérité de production.

## 9. Invariants data

- Trajet cible : `raw -> stg -> def -> mart -> snapshot -> run`.
- Raw et snapshots publiés sont immuables. Une correction crée une nouvelle
  version ou un overlay sourcé.
- EODHD reste la preuve historique prix pour les titres inactifs/delistés ; les
  refreshs ouverts ne doivent jamais effacer ce préfixe.
- Les fondamentaux officiels de production sont SEC/GAAP uniquement. Yahoo,
  SimFin, StockAnalysis ou EODHD peuvent servir au diagnostic ou au mapping,
  jamais fournir silencieusement la valeur fondamentale finale.
- Ticker, instrument et émetteur sont des identités différentes ; une
  réutilisation de symbole ne crée aucune continuité automatique.
- Grain, clé, temps de connaissance, source, politique et hash sont obligatoires
  pour toute donnée publiable.
- Zéro, null, confidentialité, non-applicable, horizon immature et événement
  terminal ne sont jamais confondus.
- Une anomalie conserve les données reçues et bloque la promotion ; elle ne les
  supprime pas pour faire passer un contrôle.
- Tout refresh complet compare le snapshot publié et le candidat au même cutoff,
  rejoue Legacy puis Boosting, et rapproche entrées, univers, scores, positions,
  poids et rendements. Si les portefeuilles historiques changent, chaque écart
  doit remonter à une révision data sourcée ; un écart inexpliqué bloque.

## 10. Invariants méthodologiques

- Legacy et Boosting génèrent leurs signaux séparément.
- La sélection est faite avec l'information disponible à la décision, avant de
  regarder la disponibilité ou la valeur du rendement réalisé.
- Toute nouvelle simulation échoue sur un rendement sélectionné manquant, sauf
  compatibilité historique explicitement nommée et non promouvable.
- Toute comparaison publique utilise le même snapshot, les mêmes exclusions,
  la même convention d'exécution, les mêmes coûts et le moteur commun
  `src/alpharank/portfolio/`.
- Une parité mécanique sur deux snapshots différents ne rend pas les stratégies
  comparables.
- Le benchmark standard est SPY total return depuis `adjusted_close`; l'ancien
  SP500 prix n'est pas substituable.
- Le calendrier de maturité des cibles modèle reste distinct de celui des
  rendements portefeuille.
- Une vue SHAP mensuelle indique son nombre de lignes et son statut exhaustif ou
  échantillonné.
- Les KPI, conventions et attributions sont définis dans
  `docs/common_portfolio_backtest_engine.md`; ne pas les réimplémenter dans un
  dashboard ou un script.

## 11. Vérifications

- Exécuter le test le plus étroit qui reproduit le contrat modifié, puis les
  validations transverses proportionnées au risque.
- Une correction de bug ajoute d'abord un test de non-régression.
- Une modification de code/data de production exécute les gates du runbook et
  les validateurs de replay concernés.
- Une suite unitaire verte ne suffit jamais à revendiquer absence de fuite,
  comparabilité ou readiness production : nommer snapshot, cutoff, calendrier,
  univers, sources et risques de couverture.
- Avant commit, vérifier le diff, les tests, la documentation, le statut roadmap
  et l'absence d'artefact généré indexé.

## 12. Handoff obligatoire

Le compte rendu final indique :

- tâche de roadmap et commit éventuel ;
- fichiers réellement modifiés ;
- tests/validateurs exécutés et résultat ;
- snapshot/run concernés lorsqu'il y en a ;
- changement économique ou garantie de parité ;
- risques, dette ou travail restant ;
- hash local et hash distant après le push, ou raison précise d'un push en
  attente.

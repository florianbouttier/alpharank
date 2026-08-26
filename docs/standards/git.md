# Standard Git AlphaRank

**Rôle : norme des tâches, branches, commits, preuves et publication Git.**

## 1. Règle centrale

```text
1 ligne de roadmap avec un TASK-ID = 1 changement atomique = 1 commit
```

Cette relation est stricte dans les deux sens :

- un commit ne traite jamais deux `TASK-ID` ;
- une tâche ne produit jamais deux commits ;
- si une tâche est trop grande, elle est découpée dans la roadmap **avant** de
  modifier le code ;
- si un correctif devient nécessaire après le commit, il reçoit un nouveau
  `TASK-ID` qui référence le premier ;
- aucun commit `WIP`, `fixup`, « suite » ou « finalisation » sur une tâche déjà
  committée.

Une checklist à l'intérieur d'une tâche décrit ses critères d'acceptation ; ses
cases ne sont pas des tâches indépendantes. Si une case peut être livrée,
testée ou annulée séparément, elle doit devenir une ligne de roadmap avec son
propre identifiant.

## 2. Une bonne tâche de roadmap

Avant de coder, chaque ligne possède :

| Champ | Contenu obligatoire |
| --- | --- |
| `TASK-ID` | identifiant unique et stable, par exemple `CODE-003` |
| objectif | un résultat observable, formulé avec un verbe |
| périmètre | dossiers, composants ou datasets autorisés |
| hors périmètre | ce qui ne doit pas changer |
| critères d'acceptation | comportement et preuves mesurables |
| validations | tests, replays, hashes ou contrôles attendus |
| impact | code, data, économique, historique ou aucun |
| rollback | manière de revenir au comportement précédent |
| statut | `à faire`, `en cours`, `bloqué` ou `fait` |

Une tâche qui contient « et » entre deux résultats indépendants doit
probablement être découpée.

Exemple correct :

```text
CODE-003 — Extraire la normalisation Yahoo de ingestion.py
Périmètre : normalisation et tests associés.
Hors périmètre : téléchargement, préférence fournisseur, publication.
Acceptation : mêmes clés, valeurs et hashes logiques avant/après.
```

Exemple incorrect :

```text
CODE-003 — Nettoyer l'ingestion, corriger les données et refaire le dashboard.
```

## 3. Cycle d'une tâche

1. Lire la tâche, les standards et les contrats concernés.
2. Inspecter le worktree et identifier les modifications déjà présentes.
3. Passer la tâche à `en cours` dans le worktree, sans la déclarer terminée.
4. Écrire ou identifier le test de caractérisation.
5. Réaliser uniquement le périmètre annoncé.
6. Exécuter les validations prévues.
7. Relire le diff complet et le diff qui sera indexé.
8. Mettre la tâche à `fait` dans le même ensemble indexé que le code, les tests
   et la documentation.
9. Créer l'unique commit.
10. Vérifier le commit créé avec `git show --stat` et `git show`.
11. Pousser immédiatement avec `git push origin main`.
12. Vérifier après récupération des références que `main` et `origin/main`
    désignent le même commit.

Un refresh, replay, backtest ou run de production important demandé par le
propriétaire ne se termine pas seulement par des artefacts ignorés sous
`data/`, `outputs/` ou `logs/`. Il crée une tâche atomique et met à jour une
preuve suivie dans le document canonique du run ; cette preuve, son statut
roadmap et ses hashes sont commités puis poussés selon le cycle ci-dessus.

Une tâche n'est `fait` que lorsque le commit existe et que ses validations sont
documentées. Pour le suivi visible à distance, elle n'est publiée comme `faite`
que lorsque ce commit est présent sur `origin/main`. Du code présent uniquement
dans le worktree reste `en cours` ou `prêt à committer`. Si le commit local
existe mais que le push échoue, le handoff porte explicitement `push en attente`
et le hash concerné ; pousser ce même commit reste l'action suivante.

## 4. Autorisation

- Aucun commit sans autorisation explicite de l'utilisateur.
- Une demande explicite d'exécuter un ensemble de tâches de roadmap autorise
  les commits unitaires nécessaires à cet ensemble.
- Dans ce dépôt personnel, cette autorisation inclut le push normal immédiat de
  chaque commit sur `origin/main`. Elle ne permet ni suppression, ni
  réécriture d'historique, ni force-push.
- Une demande explicite d'exécution d'un run important autorise également le
  commit et le push de sa preuve suivie, sans autoriser l'indexation des données
  brutes, modèles, sorties volumineuses ou journaux ignorés.
- En présence d'un worktree sale, ne pas demander automatiquement de le nettoyer
  ou de le stasher : isoler la tâche par staging sélectif.

## 5. Branches

Le propriétaire étant le seul contributeur, le flux normal est :

```text
main local -> origin/main
```

- Travailler, committer et pousser directement depuis `main`.
- Une branche `codex/<task-id-lowercase>-<slug-court>` n'est utilisée que sur
  demande explicite ou pour un travail parallèle réellement isolé.
- Une branche temporaire est réintégrée sans masquer les commits unitaires et
  sans merge commit parasite lorsqu'un fast-forward est possible.
- `main-save` désigne exclusivement l'ancien `main` distant au commit
  `c1113ab0613c06c8e3deb27e7a7f35d892e80bca`. Elle est immuable : ne pas y
  committer, la fusionner, la rebaser, la supprimer ou la faire avancer sans
  demande explicite.
- La présence d'un seul contributeur ne dispense pas de récupérer les
  références distantes : une automatisation ou une modification depuis un
  autre appareil peut avoir avancé `origin/main`.

## 6. Format du message

Sujet obligatoire :

```text
<type>(<TASK-ID>): <résumé impératif>
```

Types autorisés :

| Type | Usage |
| --- | --- |
| `feat` | capacité nouvelle |
| `fix` | correction d'un comportement erroné |
| `refactor` | structure changée, comportement conservé |
| `data` | contrat, ingestion ou migration de données |
| `docs` | documentation uniquement |
| `test` | amélioration des preuves sans changement métier |
| `perf` | optimisation avec parité démontrée |
| `ci` | intégration continue et contrôles |
| `build` | dépendances ou packaging |
| `chore` | maintenance qui ne rentre pas dans les catégories précédentes |

Règles du sujet :

- `TASK-ID` en majuscules et identique à la roadmap ;
- verbe impératif, formulation précise ;
- maximum 72 caractères recommandé ;
- pas de point final ;
- pas de « update », « changes », « fixes » ou « cleanup » sans objet précis.

Exemples :

```text
refactor(CODE-003): extract Yahoo normalization stage
data(DATA-003): deduplicate raw payloads by SHA-256
docs(DOC-015): define the Git task and commit contract
```

## 7. Corps du commit

Tout commit non trivial possède ce corps :

```text
Why:
- problème et risque corrigés

Changes:
- modifications concrètes et frontières conservées

Validation:
- commandes réellement exécutées et résultats

Impact:
- impact économique/data/historique, ou "none"

Risks:
- limites et travail restant, ou "none"

Roadmap-Task: CODE-003
```

Règles :

- documenter les commandes réellement exécutées, jamais celles seulement
  recommandées ;
- préciser le snapshot, run ou dataset lorsqu'il est concerné ;
- écrire explicitement `Economic-Impact: none` pour un refactor à parité ;
- indiquer les nombres de tests sans recopier un ancien compteur ;
- ne pas déclarer « no leakage » depuis des tests unitaires seuls ;
- ne pas mettre de secret, chemin personnel inutile ou payload brut dans le
  message.

Trailers recommandés :

```text
Roadmap-Task: CODE-003
Economic-Impact: none
Data-Impact: none
```

## 8. Lien entre commit et roadmap

Le même commit contient :

- la modification ;
- ses tests ;
- la documentation canonique ;
- le statut `fait` et les preuves de la tâche.

Le hash exact d'un commit ne peut pas être écrit à l'intérieur de ce même
commit : son contenu participe au calcul du hash. Le lien canonique est donc le
`TASK-ID`, présent dans le sujet, le trailer et la roadmap.

Résolution :

```bash
git log --all --grep='Roadmap-Task: CODE-003' --format='%H %s'
```

Un registre de release peut recopier les hashes après coup, mais ce registre est
un artefact généré ou une tâche distincte ; il ne justifie pas d'amender le
commit original.

## 9. Staging dans un worktree sale

Avant le commit :

1. lire `git status --short` ;
2. distinguer les changements de la tâche et ceux déjà présents ;
3. indexer uniquement les fichiers ou hunks autorisés ;
4. contrôler `git diff --cached --stat` ;
5. lire entièrement `git diff --cached` ;
6. vérifier que le diff non indexé reste intact ;
7. rechercher secrets, données volumineuses et artefacts générés.

Interdits :

- `git add .` ou `git add -A` sans revue dans un worktree sale ;
- stage d'un fichier entier lorsque seules quelques lignes appartiennent à la
  tâche et que d'autres modifications sont présentes ;
- restauration ou écrasement d'un changement tiers pour simplifier le commit ;
- commit de fichiers dont le rôle n'est pas expliqué par le `TASK-ID`.

## 10. Contenu interdit dans Git par défaut

- `.env`, tokens, credentials et clés privées ;
- données raw, snapshots, caches et environnements ;
- gros outputs HTML/Parquet/CSV de runs ;
- logs d'exécution ;
- modèles entraînés ;
- fichiers temporaires ou éditeur ;
- chemins absolus utilisateur dans un artefact reproductible.

Une exception exige une demande explicite et une justification de versionner
l'artefact plutôt que son manifeste/hash.

## 11. Validations avant commit

Minimum pour toute tâche :

- test ou validateur le plus étroit ;
- contrôle documentaire si liens/structure changent ;
- inspection du diff indexé ;
- statut roadmap cohérent.

En plus selon le risque :

- suite du package ;
- test causal de mutation du futur ;
- comparaison de clés/valeurs/hashes ;
- replay strict Legacy ;
- replay commun Legacy/Boosting ;
- audit du snapshot précédent ;
- rendu du dashboard dans un navigateur.

Un test échoué n'est pas omis du message. Corriger dans la même tâche avant le
commit ou laisser la tâche non committée et documenter le blocage.

## 12. Commits de refactor et données

### Refactor

- comportement, sélections et sorties inchangés ;
- test de caractérisation avant déplacement ;
- parité chiffrée dans le commit ;
- aucun formatage global ajouté au diff.

### Data

- anciennes données et snapshots conservés ;
- clés ajoutées/modifiées/supprimées comptées ;
- hashes avant/après ;
- règle de migration et rollback ;
- impact économique évalué ou explicitement non applicable.

### Documentation

- aucun code ou donnée ;
- liens et index validés ;
- déplacement historique conservé ou redirection documentée.

## 13. Correction après commit

Par défaut :

- ne pas amender ;
- ne pas rebaser ;
- ne pas forcer le push ;
- créer une nouvelle tâche `FIX-*` ou adaptée, liée à la tâche initiale ;
- expliquer pourquoi la preuve précédente était insuffisante.

Un `git revert` est préféré pour annuler proprement un commit publié. La roadmap
conserve le statut historique et référence la tâche de revert.

## 14. Publication directe sur `main`

Chaque commit validé est publié immédiatement, afin que l'historique soit
consultable depuis les autres appareils du propriétaire.

Cycle obligatoire :

1. vérifier que la branche courante est `main` ;
2. exécuter `git fetch origin --prune` ;
3. vérifier que `origin/main` n'a pas de commit absent de `main` ;
4. exécuter les validations et relire le diff indexé ;
5. créer l'unique commit de la tâche ;
6. exécuter `git push origin main` sans délai ;
7. récupérer à nouveau les références et comparer les hashes de `main` et
   `origin/main`.

Une pull request n'est pas exigée dans ce dépôt personnel. Si le push est
rejeté ou si les historiques divergent :

- ne pas forcer le push ;
- ne pas rebaser, fusionner ou écraser automatiquement ;
- inspecter les commits distants et les changements locaux ;
- réconcilier explicitement en conservant la relation tâche/commit ;
- signaler le hash local non publié et la cause dans le handoff.

Le force-push est interdit par défaut, y compris sur `main-save`. Une exception
exige une demande explicite nommant la branche et une revue de l'impact distant.

## 15. Checklist finale

- [ ] une seule ligne de roadmap et un seul `TASK-ID` ;
- [ ] critères d'acceptation remplis ;
- [ ] un seul objectif atomique ;
- [ ] code, tests, docs et statut roadmap réunis ;
- [ ] message et corps complets ;
- [ ] validations réellement exécutées ;
- [ ] impact économique/data déclaré ;
- [ ] diff indexé relu entièrement ;
- [ ] aucune modification tierce ou artefact généré ;
- [ ] branche courante `main` et absence de divergence distante ;
- [ ] commit poussé sur `origin/main` et égalité des hashes vérifiée.

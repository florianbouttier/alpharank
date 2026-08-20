# Référentiel de développement AlphaRank

**Rôle : index normatif.** Ces règles s'appliquent immédiatement au nouveau
code, aux nouveaux jeux de données et aux parties significativement modifiées.
Le code historique est mis en conformité progressivement par la roadmap.

## Documents

| Domaine | Source normative |
| --- | --- |
| Python, calculs et tests | [`python.md`](python.md) |
| Données et lignée | [`data.md`](data.md) |
| Organisation du dépôt | [`repository.md`](repository.md) |
| Git, tâches et commits | [`git.md`](git.md) |
| Résumé et processus de contribution | [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) |

## Interprétation

- **OBLIGATOIRE** : exigé pour toute nouvelle modification.
- **RECOMMANDÉ** : choix par défaut ; tout écart est expliqué dans la revue.
- **AUTORISÉ** : option admise dans le périmètre indiqué.
- **INTERDIT** : nouvelle occurrence refusée.

Les seuils de taille, typage et lint ne justifient pas une réécriture globale.
Lorsqu'un fichier historique non conforme doit recevoir un petit correctif :

1. ne pas reformater tout le fichier ;
2. rendre les nouvelles lignes conformes autant que possible ;
3. ajouter un test de comportement ;
4. enregistrer la dette structurelle dans la roadmap si elle empêche une
   correction propre ;
5. réserver le découpage à un commit distinct sans changement économique.

## Exception

Une exception doit préciser : règle concernée, fichier ou dataset, motif,
risque, contrôle compensatoire, propriétaire et date de retrait prévue. Une
mention « legacy » sans échéance ni contrôle n'est pas une exception valide.

## Évolution du standard

Une modification de ce référentiel :

- possède une tâche de roadmap ;
- explique son impact sur le code et les données existants ;
- ne reformate ni ne migre implicitement le dépôt ;
- met à jour les contrôles automatisés dans un commit séparé lorsque nécessaire.

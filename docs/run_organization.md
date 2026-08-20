# Organisation des résultats de runs

**Rôle : contrat canonique de chemin et de gouvernance des nouveaux résultats.**

## Chemin unique

Tout nouveau run écrit sous :

```text
outputs/<famille>/<run_id>/
```

- `<famille>` est en `lower_snake_case` et décrit le producteur, par exemple
  `monthly_legacy`, `boosting_research` ou `data_refresh` ;
- `<run_id>` suit `YYYYMMDDTHHMMSSZ_<slug>` ; l'horodatage est UTC et le suffixe
  reste en `lower_snake_case` ;
- le dossier du run est exactement à deux niveaux sous `outputs/` ;
- `latest`, `final`, `retry`, `candidate` ou `published` ne remplacent jamais
  l'identifiant immuable du run.

Le validateur exécutable `validate_canonical_run_dir` refuse les chemins libres,
les traversées hors de `outputs/` et les niveaux supplémentaires. Les 346
racines historiques inventoriées restent en place : ce contrat s'applique aux
nouveaux runs et ne déclenche aucun déplacement rétroactif.

## Statut dans le manifeste

Chaque nouveau dossier contient immédiatement un `manifest.json` conforme à
`alpharank_run_manifest_v1`. Il naît avec le statut `candidate`, puis les seules
transitions possibles sont :

```text
candidate -> validated -> published
         \-> failed    \-> failed
```

Chaque transition conserve son instant et sa raison dans `status_history`.
`published` et `failed` sont terminaux. Le statut est interdit dans la famille
ou le suffixe du `run_id` : renommer un dossier ne peut donc ni valider ni
publier un résultat.

## Journaux reliés

Un journal de nouveau run vit sous `logs/<famille>/<run_id>/*.log`. Le manifeste
du run enregistre son chemin, son rôle, sa taille, son SHA-256 et le chemin d'un
petit sidecar. Ce sidecar renvoie vers `outputs/<famille>/<run_id>/manifest.json`
et reprend l'identité hashée du journal. Le validateur parcourt donc les liens
dans les deux sens et détecte toute modification ultérieure des octets.

Les 74 journaux historiques restent intacts et ne reçoivent pas d'association
inventée après coup. Le contrat bidirectionnel est obligatoire pour chaque
journal créé avec le nouveau format de run.

## Pointeur `latest`

Après le statut `published`, `outputs/<famille>/latest.json` peut être remplacé
atomiquement. Ce petit pointeur contient le chemin du run, le hash du manifeste,
l'empreinte SHA-256 de l'inventaire complet de l'arbre, son nombre de fichiers
et sa taille. Une copie immuable du pointeur vit sous
`outputs/<famille>/pointers/<run_id>/manifest.json`.

Le pointeur ne contient et ne recopie aucun résultat (`result_copy_count: 0`).
Sa validation relit la cible et recalcule l'empreinte ; un run encore
`candidate` ou `validated`, un manifeste modifié ou un octet de résultat changé
font échouer la résolution.

## Rétention réversible

Le rapport `architecture/run_retention_report_v1.json` mesure les doublons par
taille puis par SHA-256 exact. Une taille identique ne suffit jamais. La
proposition conserve une source de récupération par groupe et exclut avant
toute action les chemins référencés par un manifeste, un pointeur ou un retour
arrière.

Cette proposition ne supprime rien. Une décision ultérieure devra fermer la
fenêtre d'observation, rehasher chaque paire, placer d'abord les candidats en
quarantaine récupérable et faire l'objet d'une tâche de roadmap séparée.

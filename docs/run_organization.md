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

Les statuts, journaux, pointeurs et règles de rétention sont ajoutés par les
tâches RUNORG suivantes dans ce même contrat.

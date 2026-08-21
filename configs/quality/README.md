# Quality baselines

This folder contains versioned development-quality decisions, not generated run
outputs. A baseline records accepted historical debt so CI can reject new debt
without hiding or rewriting the existing findings.

- `ruff_baseline_v1.json` is regenerated only through the reviewed
  `QUAL-003` workflow and has no timestamp-dependent fields.
- `test_suites_v1.json` assigns every test file to one ordered pytest suite;
  the first matching rule wins and the explicit default is `unit`.
- `python_directory_policy_v1.json` fixe le plafond absolu de fichiers Python
  par dossier. Une exception est interdite sans approbation explicite,
  nominative, datée et reliée à une tâche de roadmap. La gate statique exécute
  l'inventaire avec `--enforce-limit` et bloque tout dépassement.

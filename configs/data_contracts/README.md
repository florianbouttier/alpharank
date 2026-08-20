# Configuration schemas

This folder contains versioned contracts for configuration files. It is not a
source-data or run-output location.

- `config_schema_registry_v1.json` covers every maintained JSON family under
  `configs/data_quality`, `configs/quality` and `configs/research`.

Schemas reject undeclared keys recursively. Regeneration is an explicit roadmap
action because inferring again from a typo would otherwise legitimize it.

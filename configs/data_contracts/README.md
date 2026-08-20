# Configuration schemas

This folder contains versioned contracts for configuration files. It is not a
source-data or run-output location.

- `config_schema_registry_v1.json` covers every maintained JSON family under
  `configs/data_contracts`, `configs/data_quality`, `configs/quality` and
  `configs/research` (excluding the registry itself).
- `raw_provider_contracts_v1.json` assigns every retained or downloadable raw
  source to `data/warehouse/raw/<provider_id>` and declares receipt and provider
  manifest fields before any reader migration.

The executable writer is `record_raw_download` in
`src/alpharank/data/open_source/raw_archive.py`. It records successful and
failed attempts separately, stores payload bytes by SHA-256, and regenerates a
validated provider manifest. Legacy roots are migrated only by the later
`DATA-008` and `DATA-009` tasks.

Schemas reject undeclared keys recursively. Regeneration is an explicit roadmap
action because inferring again from a typo would otherwise legitimize it.

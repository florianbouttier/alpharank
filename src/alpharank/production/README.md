# Production workflows

Reusable orchestration for canonical AlphaRank production commands.

- `legacy_pipeline.py` owns the testable monthly Legacy workflow.
- `scripts/run_legacy.py` remains the stable human-facing command.

Without an explicit replay input, Legacy resolves
`data/model_inputs/manifests/latest.json` through the canonical MART resolver.
It stops if the target is outside `data/warehouse/mart`, if DEF/source parity is
not attested, or if any model-file hash differs.

For pointers using the snapshot-publication contract, the workflow also
revalidates the immutable publication manifest and the complete MART tree before
copying the run-local input package.

The runtime manifest hashes both the stable command wrapper and this owning
module so a replay cannot miss a production-logic change.

This package may coordinate data, signal, portfolio, governance and reporting
services; it does not redefine their domain rules.

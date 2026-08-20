# Canonical Model Inputs

This directory stores immutable, hash-verified input packages shared by Legacy
and Boosting.

- Resolve the current package through `manifests/latest.json`.
- Never point a production model directly at `data/open_source/output`; that is
  a mixed-source research package.
- Each folder under `history/` contains hybrid EODHD/open prices, strict
  SEC-only fundamentals, both source lineages, and `lineage/manifest.json`.
- Rebuild with `scripts/open_source/build_composed_model_snapshot.py` only after
  the independent price and SEC publication guards pass.
- Old history folders are immutable replay evidence and must not be edited.

The current package's `composition_id` is content-addressed from all nine model
input file hashes. Legacy and Boosting comparisons are eligible only when both
runs declare the same hashes and use the same retained Legacy `input_snapshot/`.

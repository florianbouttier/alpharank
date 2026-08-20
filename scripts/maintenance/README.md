# Maintenance scripts

Reversible repository inventories and maintenance operations live here. These
commands do not implement portfolio, signal or data-provider logic.

- `build_code_inventory.py` verifies the tracked Python entrypoint and reader
  graph, or regenerates it explicitly with `--write`.
- `build_data_location_inventory.py` snapshots current data files/packages,
  observed sizes and active Python reader references without reading payloads.
- `build_test_catalog.py` merges the suite policy with a JUnit report produced
  from a clean index checkout; it does not execute tests itself.
- `build_test_collection.py` records path-independent Pytest node identifiers
  so a physical move cannot silently add or remove a scenario.

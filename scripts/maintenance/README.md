# Maintenance scripts

Reversible repository inventories and maintenance operations live here. These
commands do not implement portfolio, signal or data-provider logic.

- `build_code_inventory.py` verifies the tracked Python entrypoint and reader
  graph, or regenerates it explicitly with `--write`.

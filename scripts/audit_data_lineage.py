from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    manifests = sorted((project_root / "outputs").glob("*/data_input_manifest.json"))

    if len(manifests) < 2:
        print("Need at least two run manifests in outputs/*/data_input_manifest.json")
        return

    prev_path, curr_path = manifests[-2], manifests[-1]
    prev = _load(prev_path)
    curr = _load(curr_path)

    print(f"Comparing:\n  prev={prev_path}\n  curr={curr_path}")
    print(f"prev snapshot_id={prev.get('snapshot_id')} generated_at={prev.get('generated_at')}")
    print(f"curr snapshot_id={curr.get('snapshot_id')} generated_at={curr.get('generated_at')}")

    changed = False
    all_names = sorted(set(prev.get("datasets", {})) | set(curr.get("datasets", {})))
    for name in all_names:
        p = prev.get("datasets", {}).get(name)
        c = curr.get("datasets", {}).get(name)
        if p is None or c is None:
            changed = True
            print(f"- {name}: present in only one manifest")
            continue

        diffs = []
        for field in ["sha256", "modified_at", "size_bytes"]:
            if p.get(field) != c.get(field):
                diffs.append(f"{field}: {p.get(field)} -> {c.get(field)}")

        p_summary = p.get("summary", {})
        c_summary = c.get("summary", {})
        if p_summary.get("max_temporal_values") != c_summary.get("max_temporal_values"):
            diffs.append(
                "max_temporal_values: "
                f"{p_summary.get('max_temporal_values')} -> {c_summary.get('max_temporal_values')}"
            )

        if diffs:
            changed = True
            print(f"- {name}")
            for diff in diffs:
                print(f"    {diff}")

    if not changed:
        print("No dataset-level differences detected between the two latest manifests.")


if __name__ == "__main__":
    main()

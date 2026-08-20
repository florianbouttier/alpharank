"""Generate and verify dependency views from canonical pyproject metadata."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]  # Python 3.10 dev dependency.


@dataclass(frozen=True, slots=True)
class DependencySource:
    """Canonical dependency metadata required by generated environment views."""

    project_name: str
    runtime_dependencies: tuple[str, ...]
    conda_environment_name: str
    conda_python: str


def load_dependency_source(pyproject_path: Path) -> DependencySource:
    """Load and validate the canonical dependency fields from pyproject.toml."""

    raw: object = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "pyproject")
    project = _require_mapping(root.get("project"), "project")
    tool = _require_mapping(root.get("tool"), "tool")
    alpharank = _require_mapping(tool.get("alpharank"), "tool.alpharank")
    dependency_policy = _require_mapping(
        alpharank.get("dependencies"),
        "tool.alpharank.dependencies",
    )
    return DependencySource(
        project_name=_require_string(project.get("name"), "project.name"),
        runtime_dependencies=_require_string_sequence(
            project.get("dependencies"),
            "project.dependencies",
        ),
        conda_environment_name=_require_string(
            dependency_policy.get("conda-environment-name"),
            "tool.alpharank.dependencies.conda-environment-name",
        ),
        conda_python=_require_string(
            dependency_policy.get("conda-python"),
            "tool.alpharank.dependencies.conda-python",
        ),
    )


def render_requirements(source: DependencySource) -> str:
    """Render the pip runtime compatibility view without resolving versions."""

    return "\n".join(source.runtime_dependencies) + "\n"


def render_conda_environment(source: DependencySource) -> str:
    """Render a minimal Conda bootstrap that delegates Python deps to pyproject."""

    return (
        f"name: {source.conda_environment_name}\n"
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        f"  - python={source.conda_python}\n"
        "  - pip\n"
        "  - pip:\n"
        "      - -e .[dev]\n"
    )


def dependency_sync_report(
    source: DependencySource,
    *,
    requirements_path: Path,
    environment_path: Path,
) -> dict[str, object]:
    """Compare generated views byte-for-byte and return an auditable report."""

    expected = {
        "requirements.txt": render_requirements(source),
        "environment.yml": render_conda_environment(source),
    }
    observed = {
        "requirements.txt": requirements_path.read_text(encoding="utf-8"),
        "environment.yml": environment_path.read_text(encoding="utf-8"),
    }
    files = {
        name: {
            "matches": observed[name] == content,
            "expected_sha256": _text_sha256(content),
            "observed_sha256": _text_sha256(observed[name]),
        }
        for name, content in expected.items()
    }
    return {
        "policy_id": "pyproject_dependency_source_v1",
        "passed": all(bool(row["matches"]) for row in files.values()),
        "runtime_dependency_count": len(source.runtime_dependencies),
        "conda_python": source.conda_python,
        "files": files,
    }


def write_dependency_views(
    source: DependencySource,
    *,
    requirements_path: Path,
    environment_path: Path,
) -> None:
    """Replace both generated views from one validated canonical source."""

    requirements_path.write_text(render_requirements(source), encoding="utf-8")
    environment_path.write_text(render_conda_environment(source), encoding="utf-8")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Dependency source {label} must be a string-keyed table")
    return value


def _require_string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Dependency source {label} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Dependency source {label} must contain non-empty strings")
    return tuple(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dependency source {label} must be a non-empty string")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

from __future__ import annotations

from pathlib import Path

from alpharank.quality.dependencies import (
    DependencySource,
    dependency_sync_report,
    load_dependency_source,
    render_conda_environment,
    render_requirements,
)

ROOT = Path(__file__).resolve().parents[3]


def test_dependency_views_are_generated_from_pyproject() -> None:
    source = load_dependency_source(ROOT / "pyproject.toml")

    report = dependency_sync_report(
        source,
        requirements_path=ROOT / "requirements.txt",
        environment_path=ROOT / "environment.yml",
    )

    assert report["passed"] is True
    assert report["runtime_dependency_count"] == len(source.runtime_dependencies)


def test_dependency_rendering_keeps_runtime_and_dev_roles_separate() -> None:
    source = DependencySource(
        project_name="sample",
        runtime_dependencies=("numpy>=1", "conditional; python_version < '3.11'"),
        conda_environment_name="sample-dev",
        conda_python="3.11",
    )

    assert render_requirements(source) == ("numpy>=1\nconditional; python_version < '3.11'\n")
    assert render_conda_environment(source) == (
        "name: sample-dev\n"
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.11\n"
        "  - pip\n"
        "  - pip:\n"
        "      - -e .[dev]\n"
    )

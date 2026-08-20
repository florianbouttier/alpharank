"""Deterministic inventory of tracked pytest files and measured runtimes."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence, TypedDict
from xml.etree import ElementTree

from alpharank.quality.test_suites import SuitePolicy, classify_test_path

TEST_CATALOG_SCHEMA_VERSION = 1
ALLOWED_TEST_DOMAINS = (
    "backtest",
    "boosting",
    "data",
    "legacy",
    "portfolio",
    "quality_tooling",
    "replay_governance",
    "reporting",
)


class TestMeasurement(TypedDict):
    """Aggregated JUnit measurement for one tracked test file."""

    test_case_count: int
    duration_seconds: float
    failure_count: int
    observed_outcome: str


class TestCatalogRow(TypedDict):
    """One reviewable test-catalog entry."""

    path: str
    domain: str
    suite: str
    network_requirement: str
    test_case_count: int
    duration_seconds: float
    duration_class: str
    observed_outcome: str
    failure_count: int


def tracked_test_paths(root: Path) -> list[str]:
    """Return the test files recorded by the current Git index."""

    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "tests/test_*.py",
            "tests/**/test_*.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(path for path in completed.stdout.decode().split("\0") if path)


def build_test_catalog(
    paths: Sequence[str],
    policy: SuitePolicy,
    *,
    junit_path: Path,
    measured_at: str,
    measurement_command: str,
) -> dict[str, object]:
    """Combine static ownership with per-file Pytest JUnit measurements."""

    measurements = _load_junit_measurements(junit_path)
    measurements_by_name = {Path(path).name: row for path, row in measurements.items()}
    rows: list[TestCatalogRow] = []
    for path in sorted(paths):
        measurement = measurements.get(
            path,
            measurements_by_name.get(Path(path).name, _empty_measurement()),
        )
        suite = classify_test_path(path, policy)
        duration_seconds = round(float(measurement["duration_seconds"]), 6)
        rows.append(
            TestCatalogRow(
                path=path,
                domain=classify_test_domain(path),
                suite=suite,
                network_requirement=(
                    "provider_boundary_disabled_by_default" if suite == "network" else "none"
                ),
                test_case_count=measurement["test_case_count"],
                duration_seconds=duration_seconds,
                duration_class=_duration_class(duration_seconds),
                observed_outcome=measurement["observed_outcome"],
                failure_count=measurement["failure_count"],
            )
        )

    domains = Counter(str(row["domain"]) for row in rows)
    suites = Counter(str(row["suite"]) for row in rows)
    outcomes = Counter(str(row["observed_outcome"]) for row in rows)
    return {
        "schema_version": TEST_CATALOG_SCHEMA_VERSION,
        "catalog_id": "alpharank_test_catalog_v1",
        "policy_id": policy.policy_id,
        "measurement": {
            "measured_at": measured_at,
            "command": measurement_command,
            "junit_source": "ephemeral_clean_index_checkout",
        },
        "summary": {
            "file_count": len(rows),
            "test_case_count": sum(int(row["test_case_count"]) for row in rows),
            "failure_count": sum(int(row["failure_count"]) for row in rows),
            "duration_seconds": round(
                sum(float(row["duration_seconds"]) for row in rows), 6
            ),
            "domain_counts": {
                domain: domains.get(domain, 0) for domain in ALLOWED_TEST_DOMAINS
            },
            "suite_counts": dict(sorted(suites.items())),
            "outcome_counts": dict(sorted(outcomes.items())),
        },
        "files": rows,
    }


def classify_test_domain(path: str) -> str:
    """Assign one stable business or tooling owner from the test filename."""

    name = Path(path).stem.removeprefix("test_")
    if name.startswith(("backtest_", "mlcraft_", "alpha_shap_")):
        return "backtest"
    if name.startswith("multihorizon_"):
        return "boosting"
    if name.startswith(("portfolio_", "terminal_")):
        return "portfolio"
    if name.startswith(("strategy_", "legacy_", "run_legacy_")):
        return "legacy"
    if name.startswith(
        (
            "boosting_v2_replay",
            "causal_snapshot",
            "common_v2_replay",
            "definitive_prices",
            "future_mutation_invariance",
            "governance_",
            "recomputable_replay",
            "reconciliation_v2",
            "replay_package_api",
            "snapshot_revision_audit",
        )
    ):
        return "replay_governance"
    if name.startswith(
        (
            "central_research_dashboard",
            "dashboard_boundaries",
            "latest_common_dashboard",
            "methodology_v2_study",
            "sec_core_kpi_",
            "sec_kpi_",
            "start_year_performance",
            "strategy_comparison_report",
        )
    ):
        return "reporting"
    if name.startswith(
        (
            "code_inventory",
            "config_schemas",
            "dependency_sync",
            "documentation_structure",
            "error_handling_policy",
            "methodology_ci",
            "methodology_documentation_contract",
            "observability",
            "root_module_ownership",
            "ruff_baseline",
            "script_",
            "test_catalog",
            "test_suite_classification",
        )
    ):
        return "quality_tooling"
    return "data"


def write_test_catalog(path: Path, catalog: dict[str, object]) -> None:
    """Write the reviewable catalog with stable JSON formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_junit_measurements(path: Path) -> dict[str, TestMeasurement]:
    grouped: dict[str, list[ElementTree.Element]] = defaultdict(list)
    root = ElementTree.parse(path).getroot()
    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "")
        if not classname.startswith("tests."):
            continue
        test_path = classname.replace(".", "/") + ".py"
        grouped[test_path].append(case)

    measurements: dict[str, TestMeasurement] = {}
    for test_path, cases in grouped.items():
        failures = [
            failure
            for case in cases
            if (failure := case.find("failure")) is not None
        ]
        missing_local_artifact = any(
            "FileNotFoundError" in (failure.text or "") for failure in failures
        )
        measurements[test_path] = TestMeasurement(
            test_case_count=len(cases),
            duration_seconds=sum(float(case.attrib.get("time", "0")) for case in cases),
            failure_count=len(failures),
            observed_outcome=(
                "passed"
                if not failures
                else "failed_missing_local_artifacts"
                if missing_local_artifact
                else "failed"
            ),
        )
    return measurements


def _empty_measurement() -> TestMeasurement:
    return TestMeasurement(
        test_case_count=0,
        duration_seconds=0.0,
        failure_count=0,
        observed_outcome="not_measured",
    )


def _duration_class(duration_seconds: float) -> str:
    if duration_seconds < 0.5:
        return "fast"
    if duration_seconds < 2.0:
        return "medium"
    return "slow"

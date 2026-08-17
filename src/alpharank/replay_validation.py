"""Sealed, recomputable replay packages for methodology validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import polars as pl

from alpharank.portfolio.simulation import simulate_weighted_portfolio


REPLAY_CONTRACT_VERSION = "recomputable-common-portfolio-v1"
REPLAY_MANIFEST_NAME = "replay_manifest.json"
REPLAY_SEAL_NAME = "replay_manifest.sha256"


class ReplayValidationError(RuntimeError):
    """Raised when a sealed replay cannot be trusted or reproduced."""


@dataclass(frozen=True)
class ReplayArtifact:
    role: str
    path: str
    sha256: str
    size_bytes: int


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_frame_sha256(frame: pl.DataFrame) -> str:
    ordered_columns = sorted(frame.columns)
    ordered = frame.select(ordered_columns)
    sort_columns = [
        column
        for column in ("strategy", "decision_month", "holding_month", "ticker")
        if column in ordered.columns
    ]
    if sort_columns:
        ordered = ordered.sort(sort_columns)
    payload = json.dumps(
        ordered.to_dicts(),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported replay JSON value: {type(value).__name__}")


def default_replay_code_paths(project_root: Path) -> tuple[Path, ...]:
    """Return the complete common engine plus its price-eligibility policy."""

    portfolio_root = project_root / "src" / "alpharank" / "portfolio"
    code_paths = sorted(portfolio_root.rglob("*.py"))
    code_paths.append(project_root / "src" / "alpharank" / "data" / "price_eligibility.py")
    missing = [str(path) for path in code_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replay code files: " + ", ".join(missing))
    return tuple(code_paths)


def create_recomputable_replay_package(
    package_dir: Path,
    *,
    holdings: pl.DataFrame,
    config: dict[str, Any],
    model_path: Path,
    project_root: Path,
    code_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Seal inputs, config, model, engine code and recomputed expected output."""

    package_dir = package_dir.resolve()
    if package_dir.exists():
        raise FileExistsError(f"Replay package already exists: {package_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    payload_dir = package_dir / "payload"
    payload_dir.mkdir(parents=True)
    artifacts: list[ReplayArtifact] = []

    holdings_path = payload_dir / "input" / "holdings.parquet"
    holdings_path.parent.mkdir(parents=True)
    holdings.write_parquet(holdings_path)
    artifacts.append(_artifact(package_dir, holdings_path, "input"))

    config_path = payload_dir / "config" / "replay_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.append(_artifact(package_dir, config_path, "config"))

    sealed_model_path = payload_dir / "model" / model_path.name
    sealed_model_path.parent.mkdir(parents=True)
    shutil.copy2(model_path, sealed_model_path)
    artifacts.append(_artifact(package_dir, sealed_model_path, "model"))

    resolved_code_paths = tuple(code_paths or default_replay_code_paths(project_root))
    for source_path in resolved_code_paths:
        source_path = source_path.resolve()
        try:
            relative_source = source_path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Replay code must be inside project_root: {source_path}") from exc
        destination = payload_dir / "code" / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        artifacts.append(_artifact(package_dir, destination, "code"))

    output = _recompute(holdings_path, config)
    expected_path = payload_dir / "expected" / "monthly_returns.parquet"
    expected_path.parent.mkdir(parents=True)
    output.write_parquet(expected_path)
    artifacts.append(_artifact(package_dir, expected_path, "expected_output"))

    manifest = {
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "engine": "alpharank.portfolio.simulation.simulate_weighted_portfolio",
        "eligibility": "alpharank.portfolio.contracts.validate_holdings+validate_causal_timing",
        "artifacts": [artifact.__dict__ for artifact in artifacts],
        "expected_output_sha256": _canonical_frame_sha256(output),
    }
    manifest_path = package_dir / REPLAY_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (package_dir / REPLAY_SEAL_NAME).write_text(
        f"{_sha256_path(manifest_path)}  {REPLAY_MANIFEST_NAME}\n",
        encoding="utf-8",
    )
    return manifest


def validate_and_recompute_replay_package(package_dir: Path) -> dict[str, Any]:
    """Validate every sealed role, then recompute and compare economic output."""

    package_dir = package_dir.resolve()
    manifest_path = package_dir / REPLAY_MANIFEST_NAME
    seal_path = package_dir / REPLAY_SEAL_NAME
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ReplayValidationError("Replay manifest or detached seal is missing.")
    expected_manifest_hash = seal_path.read_text(encoding="utf-8").split()[0]
    if _sha256_path(manifest_path) != expected_manifest_hash:
        raise ReplayValidationError("Replay manifest SHA-256 mismatch.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("replay_contract_version") != REPLAY_CONTRACT_VERSION:
        raise ReplayValidationError("Unsupported replay contract version.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReplayValidationError("Replay artifact inventory is missing.")
    by_role: dict[str, list[Path]] = {}
    for row in artifacts:
        role = str(row.get("role", ""))
        relative_path = Path(str(row.get("path", "")))
        path = package_dir / relative_path
        if not path.is_file():
            raise ReplayValidationError(f"Missing {role} artifact: {relative_path}")
        if path.stat().st_size != int(row.get("size_bytes", -1)):
            raise ReplayValidationError(f"{role} artifact size mismatch: {relative_path}")
        if _sha256_path(path) != row.get("sha256"):
            raise ReplayValidationError(f"{role} artifact SHA-256 mismatch: {relative_path}")
        by_role.setdefault(role, []).append(path)

    required_singletons = ("input", "config", "model", "expected_output")
    for role in required_singletons:
        if len(by_role.get(role, [])) != 1:
            raise ReplayValidationError(f"Replay requires exactly one {role} artifact.")
    if not by_role.get("code"):
        raise ReplayValidationError("Replay requires sealed common-engine code.")

    config = json.loads(by_role["config"][0].read_text(encoding="utf-8"))
    recomputed = _recompute(by_role["input"][0], config)
    expected = pl.read_parquet(by_role["expected_output"][0])
    recomputed_hash = _canonical_frame_sha256(recomputed)
    expected_hash = _canonical_frame_sha256(expected)
    sealed_expected_hash = manifest.get("expected_output_sha256")
    if expected_hash != sealed_expected_hash or recomputed_hash != sealed_expected_hash:
        raise ReplayValidationError(
            "Recomputed output differs from the sealed expected economic output."
        )
    return {
        "passed": True,
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "artifact_count": len(artifacts),
        "code_file_count": len(by_role["code"]),
        "recomputed_output_sha256": recomputed_hash,
    }


def _artifact(package_dir: Path, path: Path, role: str) -> ReplayArtifact:
    return ReplayArtifact(
        role=role,
        path=path.relative_to(package_dir).as_posix(),
        sha256=_sha256_path(path),
        size_bytes=path.stat().st_size,
    )


def _recompute(holdings_path: Path, config: dict[str, Any]) -> pl.DataFrame:
    allowed = {
        "transaction_cost_bps",
        "missing_return_policy",
        "causal_timing_policy",
    }
    unexpected = sorted(set(config) - allowed)
    if unexpected:
        raise ReplayValidationError("Unsupported replay config keys: " + ", ".join(unexpected))
    try:
        return simulate_weighted_portfolio(pl.read_parquet(holdings_path), **config)
    except (OSError, TypeError, ValueError) as exc:
        raise ReplayValidationError(f"Replay recomputation failed: {exc}") from exc

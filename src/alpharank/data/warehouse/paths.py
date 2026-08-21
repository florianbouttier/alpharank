from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WarehousePaths:
    """Canonical AlphaRank data layers; generated contents stay outside Git."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def stg(self) -> Path:
        return self.root / "stg"

    @property
    def definitive(self) -> Path:
        return self.root / "def"

    @property
    def mart(self) -> Path:
        return self.root / "mart"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def raw_provider(self, provider_id: str) -> Path:
        """Return one lower-snake-case provider root under canonical RAW."""

        if re.fullmatch(r"[a-z][a-z0-9_]*", provider_id) is None:
            raise ValueError(f"Invalid RAW provider id: {provider_id!r}")
        return self.raw / provider_id

    def ensure(self) -> None:
        for path in (self.raw, self.stg, self.definitive, self.mart, self.manifests):
            path.mkdir(parents=True, exist_ok=True)

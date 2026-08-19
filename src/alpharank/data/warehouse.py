from __future__ import annotations

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

    def ensure(self) -> None:
        for path in (self.raw, self.stg, self.definitive, self.mart, self.manifests):
            path.mkdir(parents=True, exist_ok=True)

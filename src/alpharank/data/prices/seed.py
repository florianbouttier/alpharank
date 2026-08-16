from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import polars as pl

from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    EODHD_DATASET,
    EODHD_SOURCE,
    PRICE_LINEAGE_COLUMNS,
)


_TICKER_ALIASES = {
    "BF-B.US": "BF.B.US",
    "BRK-B.US": "BRK.B.US",
}


@dataclass(frozen=True)
class EodhdSeed:
    frame: pl.DataFrame
    path: Path
    sha256: str
    row_count: int
    ticker_count: int
    min_date: str | None
    max_date: str | None

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "row_count": self.row_count,
            "ticker_count": self.ticker_count,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "source": EODHD_SOURCE,
            "immutable": True,
        }


def load_eodhd_seed(path: Path | str, *, start_date: str | None = None) -> EodhdSeed:
    seed_path = Path(path).expanduser().resolve()
    if not seed_path.exists():
        raise FileNotFoundError(f"Frozen EODHD price seed not found: {seed_path}")
    digest = _sha256_file(seed_path)
    frame = _normalize_seed(pl.read_parquet(seed_path), digest=digest)
    if start_date is not None:
        frame = frame.filter(pl.col("date") >= start_date)
    dates = frame.select(pl.col("date").min().alias("min"), pl.col("date").max().alias("max")).row(0)
    return EodhdSeed(
        frame=frame,
        path=seed_path,
        sha256=digest,
        row_count=frame.height,
        ticker_count=frame.select(pl.col("ticker").n_unique()).item(),
        min_date=dates[0],
        max_date=dates[1],
    )


def _normalize_seed(frame: pl.DataFrame, *, digest: str) -> pl.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen EODHD seed is missing columns: {sorted(missing)}")
    normalized = frame.select(
        pl.col("date").cast(pl.Date, strict=False).dt.strftime("%Y-%m-%d"),
        *(pl.col(column).cast(pl.Float64, strict=False) for column in ("open", "high", "low", "close", "volume", "adjusted_close")),
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
    ).with_columns(
        pl.col("ticker").replace_strict(_TICKER_ALIASES, default=pl.col("ticker")),
        pl.lit(EODHD_SOURCE).alias("source"),
        pl.lit(EODHD_DATASET).alias("dataset"),
        pl.lit(f"eodhd_frozen_{digest[:12]}").alias("ingestion_run_id"),
        pl.lit("immutable").alias("ingested_at"),
        pl.lit(f"eodhd_frozen_{digest[:12]}").alias("source_vintage_id"),
        pl.lit(f"eodhd_frozen_{digest[:12]}").alias("return_source_vintage_id"),
        pl.lit(ADJUSTMENT_POLICY_VERSION).alias("adjustment_policy_version"),
        pl.lit(1.0).alias("adjustment_bridge_factor"),
        pl.lit(digest).alias("eodhd_seed_sha256"),
        pl.lit(None).cast(pl.String).alias("correction_overlay_id"),
    )
    duplicate_count = normalized.height - normalized.select(pl.struct(["ticker", "date"]).n_unique()).item()
    if duplicate_count:
        raise ValueError(f"Frozen EODHD seed has {duplicate_count} duplicate ticker/date keys")
    return normalized.select(PRICE_LINEAGE_COLUMNS).sort(["ticker", "date"])


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

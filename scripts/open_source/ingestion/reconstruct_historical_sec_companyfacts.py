#!/usr/bin/env python3
"""Build a quarantined SEC candidate for versioned historical identities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.sec_mapping import SEC_HISTORICAL_TICKER_BRIDGE_PATH
from alpharank.data.sources.sec_historical import (
    HistoricalSecReconstructionConfig,
    reconstruct_historical_sec_companyfacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, default=SEC_HISTORICAL_TICKER_BRIDGE_PATH)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument(
        "--user-agent",
        default="Florian Bouttier florianbouttier@example.com",
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    companyfacts = SecCompanyFactsClient(
        user_agent=args.user_agent,
        cache_dir=None,
        persist_cache=False,
        request_pause_seconds=0.25,
    )
    filings = SecFilingFactsClient(
        user_agent=args.user_agent,
        cache_dir=None,
        persist_metadata_cache=False,
        persist_filing_documents=False,
        request_pause_seconds=0.25,
    )
    manifest_path = reconstruct_historical_sec_companyfacts(
        HistoricalSecReconstructionConfig(
            output_dir=args.output_dir.resolve(),
            bridge_path=args.bridge.resolve(),
            retrieved_at=datetime.now(timezone.utc),
            tickers=tuple(args.tickers) if args.tickers else None,
            workers=args.workers,
        ),
        companyfacts_client=companyfacts,
        filing_client=filings,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

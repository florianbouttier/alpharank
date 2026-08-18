#!/usr/bin/env python3
"""Validate the reviewed terminal-event registry and optionally refetch evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.portfolio.terminal_event_registry import (
    DEFAULT_TERMINAL_EVENT_REGISTRY,
    load_terminal_event_registry,
    verify_terminal_event_source_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_TERMINAL_EVENT_REGISTRY,
    )
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Refetch every primary document and require its reviewed SHA-256.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    registry = load_terminal_event_registry(args.registry)
    sources = [
        source
        for event in registry.events
        for source in event["source_documents"]
    ]
    remote = (
        verify_terminal_event_source_hashes(
            registry,
            timeout_seconds=args.timeout_seconds,
            attempts=args.attempts,
        )
        if args.verify_source_hashes
        else []
    )
    print(
        json.dumps(
            {
                "passed": True,
                "registry_id": registry.payload["registry_id"],
                "registry_sha256": registry.sha256,
                "event_count": len(registry.events),
                "terminal_consideration_count": registry.terminal_consideration_events(
                    price_vintage_id="validation-vintage"
                ).height,
                "pre_execution_block_count": registry.pre_execution_blocks().height,
                "source_document_count": len(sources),
                "remote_source_hashes_verified": len(remote),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

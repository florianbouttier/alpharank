#!/usr/bin/env python3
"""Serve one captured HTML report and its read-only Python metric API on loopback."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alpharank.reporting.strategy_analysis.server import create_analysis_server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    server = create_analysis_server(args.report_dir, port=args.port)
    print(
        f"Rapport et moteur de KPI en lecture seule : http://127.0.0.1:{server.server_port}/",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

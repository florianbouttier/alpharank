"""Loopback-only read-only viewer: no directory listing, uploads or model execution."""

from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from alpharank.reporting.strategy_analysis.dataset import AnalysisDataset
from alpharank.reporting.strategy_analysis.report import load_analysis_report
from alpharank.reporting.strategy_analysis.service import AnalysisQuery, query_analysis

LOGGER = logging.getLogger(__name__)


class AnalysisHttpServer(ThreadingHTTPServer):
    """Serve exactly one captured report and its bounded local query service."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], dataset: AnalysisDataset, html: bytes) -> None:
        self.dataset = dataset
        self.html = html
        self.cached_query = lru_cache(maxsize=64)(self._query)
        super().__init__(address, AnalysisRequestHandler)

    def _query(self, query: AnalysisQuery) -> dict[str, Any]:
        return query_analysis(self.dataset, query)


class AnalysisRequestHandler(BaseHTTPRequestHandler):
    """Allow only same-origin GET/HEAD reads of the report or analysis API."""

    server: AnalysisHttpServer

    def do_GET(self) -> None:
        if self.headers.get("Host", "").split(":")[0] not in {"127.0.0.1", "localhost"}:
            self._json(403, {"error": "Only the loopback report origin is allowed."})
            return
        path = urlsplit(self.path)
        if path.path == "/api/analysis":
            self._analysis(path.query)
        elif path.path == "/api/health":
            self._json(200, {"read_only": True, "sources": list(self.server.dataset.ledgers)})
        elif path.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif path.path in {
            "/",
            "/index.html",
            "/backtest_performance_report.html",
        }:
            self._send(200, self.server.html, "text/html; charset=utf-8")
        elif path.path.startswith("/outputs/performance_reports/"):
            self.send_response(302)
            self.send_header("Location", "/backtest_performance_report.html")
            self.end_headers()
        else:
            self._json(404, {"error": "Unknown report endpoint."})

    def do_HEAD(self) -> None:
        self.do_GET()

    def _analysis(self, query_string: str) -> None:
        args = parse_qs(query_string, keep_blank_values=True)
        try:
            if set(args) - {"start", "end", "source", "sleeve", "frontier"}:
                raise ValueError("Unknown analysis parameter.")
            if any(len(values) != 1 for key, values in args.items() if key != "sleeve"):
                raise ValueError("Duplicate scalar analysis parameter.")
            query = AnalysisQuery(
                date.fromisoformat(args.get("start", [""])[0]),
                date.fromisoformat(args.get("end", [""])[0]),
                args.get("source", ["history"])[0],
                tuple(args.get("sleeve", [])),
                args.get("frontier", ["0"])[0] == "1",
            )
            result = self.server.cached_query(query)
        except ValueError as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, result)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            "application/json; charset=utf-8",
        )

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                LOGGER.debug("Viewer cancelled an analysis response.")

    def log_message(self, message: str, *args: object) -> None:
        LOGGER.info(message, *args)


def create_analysis_server(report_dir: Path, *, port: int = 8765) -> AnalysisHttpServer:
    """Capture verified report artifacts before opening a loopback-only listener."""
    dataset = load_analysis_report(report_dir)
    path = report_dir / "backtest_performance_report.html"
    return AnalysisHttpServer(("127.0.0.1", port), dataset, path.read_bytes())

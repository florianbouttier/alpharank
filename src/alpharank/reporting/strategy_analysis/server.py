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
from alpharank.reporting.strategy_analysis.service import AnalysisQuery, query_analysis
from alpharank.reporting.strategy_analysis.version_catalog import (
    AnalysisCollection,
    load_analysis_collection,
)
from alpharank.reporting.strategy_analysis.versions import compare_versions

LOGGER = logging.getLogger(__name__)


class AnalysisHttpServer(ThreadingHTTPServer):
    """Serve one explicit collection of captured versions and bounded read-only queries."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], collection: AnalysisCollection) -> None:
        self.collection = collection
        self.dataset = collection.datasets[collection.default_version_id]
        self.cached_query = lru_cache(maxsize=64)(self._query)
        self.cached_comparison = lru_cache(maxsize=16)(self._compare)
        super().__init__(address, AnalysisRequestHandler)

    def version(self, identity: str) -> AnalysisDataset:
        """Resolve only a version already captured by this server, never a supplied path."""
        if identity not in self.collection.datasets:
            raise ValueError("Unknown captured backtest version: " + identity)
        return self.collection.datasets[identity]

    def _query(self, identity: str, query: AnalysisQuery) -> dict[str, Any]:
        return query_analysis(self.version(identity), query) | {"version_id": identity}

    def _compare(self, left: str, right: str, query: AnalysisQuery) -> dict[str, Any]:
        return compare_versions(self.version(left), self.version(right), query) | {
            "left_version_id": left,
            "right_version_id": right,
        }


def _parameters(
    query: str, allowed: set[str], *, repeated: str | None = None
) -> dict[str, list[str]]:
    args = parse_qs(query, keep_blank_values=True)
    if set(args) - allowed:
        raise ValueError("Unknown analysis parameter.")
    if any(len(values) != 1 for key, values in args.items() if key != repeated):
        raise ValueError("Duplicate scalar analysis parameter.")
    return args


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
        elif path.path == "/api/version-comparison":
            self._comparison(path.query)
        elif path.path == "/api/versions":
            self._json(200, self.server.collection.metadata)
        elif path.path == "/api/health":
            self._json(200, {"read_only": True, "sources": list(self.server.dataset.ledgers)})
        elif path.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif path.path in {
            "/",
            "/index.html",
            "/backtest_performance_report.html",
        }:
            self._page(path.query)
        elif path.path.startswith("/outputs/performance_reports/"):
            self.send_response(302)
            self.send_header("Location", "/backtest_performance_report.html")
            self.end_headers()
        else:
            self._json(404, {"error": "Unknown report endpoint."})

    def do_HEAD(self) -> None:
        self.do_GET()

    def _analysis(self, query_string: str) -> None:
        try:
            args = _parameters(
                query_string,
                {"start", "end", "source", "sleeve", "frontier", "version"},
                repeated="sleeve",
            )
            query = AnalysisQuery(
                date.fromisoformat(args.get("start", [""])[0]),
                date.fromisoformat(args.get("end", [""])[0]),
                args.get("source", ["history"])[0],
                tuple(args.get("sleeve", [])),
                args.get("frontier", ["0"])[0] == "1",
            )
            identity = args.get("version", [self.server.collection.default_version_id])[0]
            result = self.server.cached_query(identity, query)
        except ValueError as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, result)

    def _page(self, query_string: str) -> None:
        try:
            args = _parameters(query_string, {"version", "report"})
            identity = args.get("version", [self.server.collection.default_version_id])[0]
            self.server.version(identity)
        except ValueError as error:
            self._json(400, {"error": str(error)})
            return
        self._send(200, self.server.collection.reports[identity], "text/html; charset=utf-8")

    def _comparison(self, query_string: str) -> None:
        try:
            args = _parameters(query_string, {"start", "end", "left", "right", "frequency"})
            frequency = args.get("frequency", ["daily"])[0]
            if frequency not in {"daily", "monthly"}:
                raise ValueError("Unknown comparison observation frequency.")
            query = AnalysisQuery(
                date.fromisoformat(args.get("start", [""])[0]),
                date.fromisoformat(args.get("end", [""])[0]),
                "monthly_reference" if frequency == "monthly" else "history",
            )
            result = self.server.cached_comparison(
                args.get("left", [""])[0], args.get("right", [""])[0], query
            )
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
    return AnalysisHttpServer(("127.0.0.1", port), load_analysis_collection(report_dir))

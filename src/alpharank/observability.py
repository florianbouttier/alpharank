"""Structured logging context shared by AlphaRank library components."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, MutableMapping, TextIO

__all__ = [
    "RunLogContext",
    "RunLoggerAdapter",
    "configure_run_logging",
    "current_run_log_context",
    "get_run_logger",
    "reset_run_log_context",
    "set_run_log_context",
]


@dataclass(frozen=True, slots=True)
class RunLogContext:
    """Identifiers attached to every library log record."""

    run_id: str = "not_applicable"
    snapshot_id: str = "not_applicable"
    component: str = "alpharank"
    step: str = "unspecified"


_RUN_LOG_CONTEXT: ContextVar[RunLogContext] = ContextVar(
    "alpharank_run_log_context",
    default=RunLogContext(),
)


class RunLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Inject the current run context without changing call signatures."""

    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        context = _RUN_LOG_CONTEXT.get()
        supplied_extra = kwargs.get("extra", {})
        if not isinstance(supplied_extra, dict):
            raise TypeError("logging extra must be a dictionary")
        kwargs["extra"] = {
            "run_id": context.run_id,
            "snapshot_id": context.snapshot_id,
            "component": (self.extra or {}).get("component", context.component),
            "step": context.step,
            **supplied_extra,
        }
        return msg, kwargs


class _RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _RUN_LOG_CONTEXT.get()
        defaults = {
            "run_id": context.run_id,
            "snapshot_id": context.snapshot_id,
            "component": context.component,
            "step": context.step,
            "result": "unspecified",
        }
        for field, value in defaults.items():
            if not hasattr(record, field):
                setattr(record, field, value)
        return True


def get_run_logger(component: str) -> RunLoggerAdapter:
    """Return a logger that always carries the AlphaRank run identifiers."""

    return RunLoggerAdapter(logging.getLogger(component), {"component": component})


def configure_run_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Handler:
    """Configure one idempotent structured handler at a process boundary."""

    package_logger = logging.getLogger("alpharank")
    for handler in package_logger.handlers:
        if getattr(handler, "_alpharank_run_handler", False):
            handler.setLevel(level)
            return handler

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.addFilter(_RunContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s run_id=%(run_id)s snapshot_id=%(snapshot_id)s "
            "component=%(component)s step=%(step)s result=%(result)s %(message)s"
        )
    )
    setattr(handler, "_alpharank_run_handler", True)
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False
    return handler


def set_run_log_context(
    *,
    run_id: str,
    snapshot_id: str | None = None,
    component: str | None = None,
    step: str | None = None,
) -> Token[RunLogContext]:
    """Bind durable run identifiers to subsequent log records in this context."""

    if not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    current = _RUN_LOG_CONTEXT.get()
    return _RUN_LOG_CONTEXT.set(
        replace(
            current,
            run_id=run_id,
            snapshot_id=snapshot_id or current.snapshot_id,
            component=component or current.component,
            step=step or current.step,
        )
    )


def reset_run_log_context(token: Token[RunLogContext]) -> None:
    """Restore the context that preceded :func:`set_run_log_context`."""

    _RUN_LOG_CONTEXT.reset(token)


def current_run_log_context() -> RunLogContext:
    """Expose the immutable context for tests and process-boundary formatters."""

    return _RUN_LOG_CONTEXT.get()

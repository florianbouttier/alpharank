from __future__ import annotations

import logging
from io import StringIO

import pytest

from alpharank.observability import (
    configure_run_logging,
    current_run_log_context,
    get_run_logger,
    reset_run_log_context,
    set_run_log_context,
)


def test_run_logger_injects_bound_identifiers(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_run_logger("alpharank.test")
    token = set_run_log_context(
        run_id="run-123",
        snapshot_id="snapshot-456",
        step="validation",
    )
    try:
        with caplog.at_level(logging.INFO, logger="alpharank.test"):
            logger.info("validated", extra={"result": "passed"})
    finally:
        reset_run_log_context(token)

    record = caplog.records[-1]
    assert record.run_id == "run-123"
    assert record.snapshot_id == "snapshot-456"
    assert record.component == "alpharank.test"
    assert record.step == "validation"
    assert record.result == "passed"


def test_run_log_context_requires_non_empty_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        set_run_log_context(run_id=" ")

    assert current_run_log_context().run_id == "not_applicable"


def test_configured_handler_renders_run_identifiers() -> None:
    package_logger = logging.getLogger("alpharank")
    prior_handlers = package_logger.handlers[:]
    prior_propagate = package_logger.propagate
    package_logger.handlers.clear()
    stream = StringIO()
    token = set_run_log_context(run_id="run-789", snapshot_id="snapshot-012")
    try:
        configure_run_logging(stream=stream)
        get_run_logger("alpharank.test").warning(
            "provider fallback",
            extra={"result": "fallback"},
        )
    finally:
        reset_run_log_context(token)
        package_logger.handlers[:] = prior_handlers
        package_logger.propagate = prior_propagate

    rendered = stream.getvalue()
    assert "run_id=run-789" in rendered
    assert "snapshot_id=snapshot-012" in rendered
    assert "result=fallback" in rendered

from __future__ import annotations

from types import SimpleNamespace

from alpharank.data.service import EODHDDataService


def test_service_forwards_tickers_when_processing_fundamentals() -> None:
    calls: list[tuple[list[dict], list[str], str]] = []
    service = object.__new__(EODHDDataService)
    service.fundamental_data = SimpleNamespace(
        process_fundamental_data=lambda rows, tickers, data_type: calls.append(
            (rows, tickers, data_type)
        )
    )
    rows = [{"General": {"Code": "AAA"}}]

    service.process_fundamental_data(rows, ["AAA.US"], "general")

    assert calls == [(rows, ["AAA.US"], "general")]

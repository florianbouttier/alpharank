from __future__ import annotations

import socket
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from alpharank.data.sources import simfin as simfin_source
from alpharank.data.sources import simfin_transport


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.is_redirect = False
        self.headers: dict[str, str] = {}

    def close(self) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.payload[index : index + chunk_size] for index in range(0, len(self.payload), chunk_size)]


def test_simfin_download_retries_ipv4_and_installs_atomically(tmp_path: Path, monkeypatch) -> None:
    target_path = tmp_path / "us-balance-quarterly.csv"
    download_path = tmp_path / "download" / "us-balance-quarterly.zip"
    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(target_path.name, "Ticker;Report Date\nAAPL;2026-06-30\n")

    calls: list[bool] = []
    original_getaddrinfo = socket.getaddrinfo

    def fake_get(*args: object, **kwargs: object) -> _Response:
        calls.append(socket.getaddrinfo is not original_getaddrinfo)
        if len(calls) == 1:
            raise requests.ReadTimeout("IPv6 stalled")
        return _Response(archive_buffer.getvalue())

    monkeypatch.setattr(simfin_transport, "_path_dataset", lambda **kwargs: str(target_path))
    monkeypatch.setattr(simfin_transport, "_path_download_dataset", lambda **kwargs: str(download_path))
    monkeypatch.setattr(simfin_transport, "_url_dataset", lambda **kwargs: "https://simfin.test/data")
    monkeypatch.setattr(simfin_transport, "_headers_dataset", lambda: {"Authorization": "test"})
    monkeypatch.setattr(simfin_transport.requests, "get", fake_get)

    refreshed = simfin_transport.refresh_simfin_dataset(
        dataset="balance",
        market="us",
        variant="quarterly",
        refresh_days=0,
    )

    assert refreshed is True
    assert calls == [False, True]
    assert target_path.read_text(encoding="utf-8") == "Ticker;Report Date\nAAPL;2026-06-30\n"
    assert not target_path.with_suffix(".csv.part").exists()


def test_simfin_source_records_bounded_transport_failure(monkeypatch) -> None:
    def fail_refresh(**kwargs: object) -> bool:
        raise OSError("bounded transport failure")

    monkeypatch.setattr(simfin_source, "refresh_simfin_dataset", fail_refresh)
    client = simfin_source.SimFinClient(api_key="configured")

    result = client._load_dataset_frame_safe("balance", {"AAPL"}, 2026)

    assert result.is_empty()
    assert client.last_fetch_failures == [
        {"dataset": "balance", "error": "bounded transport failure"}
    ]

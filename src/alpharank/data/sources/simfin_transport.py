from __future__ import annotations

import os
import socket
import threading
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

import requests
from simfin.download import _headers_dataset, _url_dataset
from simfin.paths import _path_dataset, _path_download_dataset

SIMFIN_TIMEOUT_SECONDS = (10, 60)
_RESOLVER_LOCK = threading.Lock()


def refresh_simfin_dataset(
    *,
    dataset: str,
    market: str,
    variant: str,
    refresh_days: int,
) -> bool:
    """Refresh one SimFin bulk dataset with bounded network waits."""
    target_path = Path(_path_dataset(dataset=dataset, market=market, variant=variant))
    if not _requires_refresh(target_path, refresh_days):
        return False

    download_path = Path(_path_download_dataset(dataset=dataset, market=market, variant=variant))
    url = _url_dataset(dataset=dataset, market=market, variant=variant)
    _download_with_ipv4_retry(url=url, headers=_headers_dataset(), destination=download_path)
    _install_download(download_path=download_path, target_path=target_path)
    return True


def _requires_refresh(path: Path, refresh_days: int) -> bool:
    if refresh_days == 0 or not path.exists():
        return True
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - modified_at).days
    return age_days >= refresh_days


def _download_with_ipv4_retry(*, url: str, headers: dict[str, str], destination: Path) -> None:
    try:
        _download(url=url, headers=headers, destination=destination)
    except requests.RequestException as first_error:
        try:
            with _ipv4_resolution():
                _download(url=url, headers=headers, destination=destination)
        except requests.RequestException as second_error:
            raise OSError(
                "SimFin bulk download failed with bounded default and IPv4 transports: "
                f"{type(first_error).__name__}; {type(second_error).__name__}"
            ) from second_error


def _download(*, url: str, headers: dict[str, str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(f"{destination.suffix}.part")
    response = _open_download_response(url=url, headers=headers)
    try:
        response.raise_for_status()
        with partial_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    output.write(chunk)
        os.replace(partial_path, destination)
    finally:
        response.close()
        partial_path.unlink(missing_ok=True)


def _open_download_response(*, url: str, headers: dict[str, str]) -> requests.Response:
    current_url = url
    current_headers = headers
    for _ in range(4):
        response = requests.get(
            current_url,
            headers=current_headers,
            stream=True,
            timeout=SIMFIN_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        if not response.is_redirect:
            return response
        location = response.headers.get("location")
        response.close()
        if not location:
            raise requests.TooManyRedirects("SimFin redirect has no location")
        redirected_url = urljoin(current_url, location)
        current_headers = headers if urlparse(redirected_url).hostname == urlparse(url).hostname else {}
        current_url = redirected_url
    raise requests.TooManyRedirects("SimFin bulk download exceeded four redirects")


def _install_download(*, download_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if download_path.suffix != ".zip":
        os.replace(download_path, target_path)
        return

    staged_path = target_path.with_suffix(f"{target_path.suffix}.part")
    try:
        with zipfile.ZipFile(download_path) as archive:
            member = _dataset_member(archive=archive, target_name=target_path.name)
            with archive.open(member) as source, staged_path.open("wb") as output:
                while chunk := source.read(64 * 1024):
                    output.write(chunk)
        os.replace(staged_path, target_path)
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise OSError(f"Invalid SimFin archive for {target_path.name}: {error}") from error
    finally:
        staged_path.unlink(missing_ok=True)


def _dataset_member(*, archive: zipfile.ZipFile, target_name: str) -> str:
    matches = [name for name in archive.namelist() if Path(name).name == target_name]
    if len(matches) != 1:
        raise KeyError(f"expected one {target_name!r} member, found {len(matches)}")
    return matches[0]


@contextmanager
def _ipv4_resolution() -> Iterator[None]:
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(
        host: str | bytes | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[object, ...]]]:
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    with _RESOLVER_LOCK:
        socket.getaddrinfo = ipv4_getaddrinfo
        try:
            yield
        finally:
            socket.getaddrinfo = original_getaddrinfo

#!/usr/bin/env python3
"""Portfolio allocation utility with live Yahoo Finance quotes.

Examples:
  python -c "from scripts.portfolio_allocation import main; main(amount=10000, currency='USD', tickers=('AAPL', 'MSFT', 'NVDA'))"
  python -c "from scripts.portfolio_allocation import main; main(amount=5000, currency='EUR', tickers=('AAPL', 'MSFT'), weights=(70, 30))"
  python -c "from scripts.portfolio_allocation import main; main(amount=3000, currency='EUR', tickers=('AAPL', 'TSLA'), watch=5)"
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List
from urllib.parse import quote
from urllib.request import Request, urlopen


YAHOO_QUOTE_URLS = [
    "https://query1.finance.yahoo.com/v7/finance/quote?symbols=",
    "https://query2.finance.yahoo.com/v7/finance/quote?symbols=",
]


@dataclass
class Quote:
    symbol: str
    price: float
    currency: str | None


def _request_json(url: str, timeout: int = 8) -> dict:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_quotes_yahoo(symbols: List[str], timeout: int = 8) -> Dict[str, Quote]:
    if not symbols:
        return {}
    payload = None
    symbols_query = quote(",".join(symbols))

    for base_url in YAHOO_QUOTE_URLS:
        url = base_url + symbols_query
        for attempt in range(3):
            try:
                payload = _request_json(url, timeout=timeout)
                break
            except Exception:
                if attempt == 2:
                    break
                time.sleep(0.6 * (attempt + 1))
        if payload is not None:
            break

    if payload is None:
        raise RuntimeError("Unable to fetch quotes from Yahoo Finance (rate limited or unavailable).")

    results = payload.get("quoteResponse", {}).get("result", [])
    out: Dict[str, Quote] = {}
    for item in results:
        symbol = item.get("symbol")
        price = item.get("regularMarketPrice")
        if symbol is None or price is None:
            continue
        out[symbol.upper()] = Quote(
            symbol=symbol.upper(),
            price=float(price),
            currency=item.get("currency"),
        )
    return out


def _fetch_stooq_price(symbol: str, timeout: int = 8) -> float | None:
    stooq_variants = [symbol.lower(), f"{symbol.lower()}.us"]
    for stooq_symbol in stooq_variants:
        url = f"https://stooq.com/q/l/?s={quote(stooq_symbol)}&i=5"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=timeout) as response:
                line = response.read().decode("utf-8").strip()
            # CSV format: SYMBOL,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOLUME,
            fields = [f.strip() for f in line.split(",")]
            if len(fields) >= 7 and fields[6] not in {"N/D", ""}:
                return float(fields[6])
        except Exception:
            continue
    return None


def fetch_quotes(symbols: List[str], timeout: int = 8) -> tuple[Dict[str, Quote], str]:
    try:
        return fetch_quotes_yahoo(symbols, timeout=timeout), "yahoo"
    except Exception:
        out: Dict[str, Quote] = {}
        for symbol in symbols:
            price = _fetch_stooq_price(symbol, timeout=timeout)
            if price is not None:
                out[symbol.upper()] = Quote(symbol=symbol.upper(), price=float(price), currency="USD")
        if out:
            return out, "stooq (fallback)"
        raise RuntimeError("Unable to fetch prices from Yahoo Finance and fallback source.")
def normalize_weights(raw_weights: List[float] | None, n: int) -> List[float]:
    if raw_weights is None:
        return [1.0 / n] * n
    if len(raw_weights) != n:
        raise ValueError("The number of weights must match the number of tickers.")
    if any(w < 0 for w in raw_weights):
        raise ValueError("Weights must be non-negative.")

    # If user passed percentages like 40 30 30, normalize directly anyway.
    total = sum(raw_weights)
    if total <= 0:
        raise ValueError("Weights sum must be positive.")
    return [w / total for w in raw_weights]


def get_fx_rates(portfolio_ccy: str, timeout: int = 8) -> tuple[float, float, str]:
    """Return (eurusd, usdeur)."""
    if portfolio_ccy not in {"USD", "EUR"}:
        raise ValueError("Unsupported currency")

    try:
        fx_quotes_yahoo = fetch_quotes_yahoo(["EURUSD=X", "USDEUR=X"], timeout=timeout)
        eurusd = fx_quotes_yahoo.get("EURUSD=X")
        usdeur = fx_quotes_yahoo.get("USDEUR=X")

        eurusd_rate = eurusd.price if eurusd else (1.0 / usdeur.price if usdeur else None)
        usdeur_rate = usdeur.price if usdeur else (1.0 / eurusd.price if eurusd else None)
        if eurusd_rate is None or usdeur_rate is None:
            raise RuntimeError("missing fx")
        return float(eurusd_rate), float(usdeur_rate), "yahoo"
    except Exception:
        # Fallback 1: exchangerate-api mirror without auth
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                payload = _request_json("https://open.er-api.com/v6/latest/USD", timeout=timeout)
                usdeur_rate = float(payload["rates"]["EUR"])
                eurusd_rate = 1.0 / usdeur_rate
                return eurusd_rate, usdeur_rate, "open.er-api (fallback)"
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        # Fallback 2: frankfurter.app
        for attempt in range(2):
            try:
                usd_to_eur = _request_json(
                    "https://api.frankfurter.app/latest?from=USD&to=EUR",
                    timeout=timeout,
                )
                usdeur_rate = float(usd_to_eur["rates"]["EUR"])
                eurusd_rate = 1.0 / usdeur_rate
                return eurusd_rate, usdeur_rate, "frankfurter (fallback)"
            except Exception as exc:
                last_exc = exc
                if attempt < 1:
                    time.sleep(0.5)
        raise RuntimeError(f"Unable to fetch EUR/USD exchange rates: {last_exc}")


def allocate_once(
    amount: float,
    portfolio_ccy: str,
    tickers: List[str],
    weights: List[float],
    timeout: int,
) -> None:
    symbols = [t.upper() for t in tickers]
    quotes, price_source = fetch_quotes(symbols, timeout=timeout)
    missing = [s for s in symbols if s not in quotes]
    if missing:
        raise RuntimeError(f"Missing quote(s): {', '.join(missing)}")

    eurusd, usdeur, fx_source = get_fx_rates(portfolio_ccy, timeout=timeout)

    if portfolio_ccy == "USD":
        amount_usd = amount
        amount_eur = amount * usdeur
    else:
        amount_eur = amount
        amount_usd = amount * eurusd

    print("=" * 90)
    print(
        f"Portfolio amount: {amount:,.2f} {portfolio_ccy} | "
        f"EURUSD={eurusd:.6f} USDEUR={usdeur:.6f}"
    )
    print(f"Price source: {price_source} | FX source: {fx_source}")
    print(f"Notional: {amount_usd:,.2f} USD / {amount_eur:,.2f} EUR")
    print("-" * 90)
    alloc_ccy_col = f"Alloc({portfolio_ccy})"
    alloc_usd_col = "AllocUSD(eqv)" if portfolio_ccy == "USD" else "Alloc(USD)"
    print(
        f"{'Ticker':<10} {'Price(USD)':>12} {'Weight(%)':>10} "
        f"{alloc_ccy_col:>14} {alloc_usd_col:>14} {'Est.Shares':>12}"
    )

    for symbol, weight in zip(symbols, weights):
        q = quotes[symbol]
        alloc_portfolio_ccy = amount * weight
        alloc_usd = amount_usd * weight
        est_shares = alloc_usd / q.price if q.price > 0 else 0.0

        print(
            f"{symbol:<10} {q.price:>12.4f} {weight * 100:>10.2f} "
            f"{alloc_portfolio_ccy:>14.2f} {alloc_usd:>14.2f} {est_shares:>12.4f}"
        )
    print("=" * 90)


def main(
    *,
    amount: float = 10000.0,
    currency: str = "USD",
    tickers: List[str] | tuple[str, ...] = ("AAPL", "MSFT", "NVDA"),
    weights: List[float] | tuple[float, ...] | None = None,
    watch: int = 0,
    timeout: int = 8,
) -> int:
    if amount <= 0:
        print("Error: --amount must be > 0.", file=sys.stderr)
        return 1

    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        print("Error: provide at least one ticker.", file=sys.stderr)
        return 1

    try:
        normalized_weights = normalize_weights(list(weights) if weights is not None else None, len(tickers))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    watch = max(watch, 0)
    try:
        if watch == 0:
            allocate_once(amount, currency, tickers, normalized_weights, timeout=timeout)
            return 0

        print(f"Live mode enabled. Refresh every {watch}s. Press Ctrl+C to stop.")
        while True:
            allocate_once(amount, currency, tickers, normalized_weights, timeout=timeout)
            time.sleep(watch)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Portfolio allocation utility with live Yahoo Finance quotes.

Examples:
  python -c "from scripts.portfolio_allocation import main; main(amount=10000, currency='USD', tickers=('AAPL', 'MSFT', 'NVDA'))"
  python -c "from scripts.portfolio_allocation import main; main(amount=5000, currency='EUR', tickers=('AAPL', 'MSFT'), weights=(70, 30))"
  python -c "from scripts.portfolio_allocation import main; main(amount=3000, currency='EUR', tickers=('AAPL', 'TSLA'), watch=5)"
  python -c "from scripts.portfolio_allocation import main; main(amount=57000, currency='EUR', tickers=('APA','CF','WDC','FIX','WBD','VRT','TER'), current_shares={'WDC': 21, 'FIX': 4, 'WBD': 186, 'GOOGL': 17, 'NVDA': 70}, keep_tickers=('GOOGL','NVDA'), buy_only_equal_weight=True)"
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from math import floor
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
    # Prefer the US symbol suffix first. Bare symbols like APA can resolve to
    # unrelated instruments and return bogus values such as 1.0.
    stooq_variants = [f"{symbol.lower()}.us", symbol.lower()]
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


def _normalize_symbols(tickers: List[str] | tuple[str, ...]) -> List[str]:
    return [ticker.strip().upper() for ticker in tickers if ticker.strip()]


def _normalize_current_shares(current_shares: Dict[str, float] | None) -> Dict[str, float]:
    if current_shares is None:
        return {}

    out: Dict[str, float] = {}
    for symbol, shares in current_shares.items():
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            continue
        numeric_shares = float(shares)
        if numeric_shares < 0:
            raise ValueError(f"Current shares must be non-negative: {normalized_symbol}")
        out[normalized_symbol] = numeric_shares
    return out


def _solve_buy_only_level(current_values: List[float], total_target_usd: float) -> float:
    if total_target_usd < 0:
        raise ValueError("Target bucket must be non-negative.")

    current_total = sum(current_values)
    if current_total > total_target_usd + 1e-9:
        raise ValueError(
            "Current positions inside the equal-weight bucket already exceed the target bucket. "
            "Increase the portfolio amount or allow sells."
        )

    lo, hi = 0.0, max(total_target_usd, max(current_values, default=0.0))
    for _ in range(200):
        mid = (lo + hi) / 2.0
        required = sum(max(value, mid) for value in current_values)
        if required > total_target_usd:
            hi = mid
        else:
            lo = mid
    return lo


def allocate_buy_only_equal_weight(
    amount: float,
    portfolio_ccy: str,
    tickers: List[str],
    current_shares: Dict[str, float] | None,
    keep_tickers: List[str] | tuple[str, ...] = (),
    amount_mode: str = "final_total",
    timeout: int = 8,
) -> None:
    target_symbols = _normalize_symbols(tickers)
    if not target_symbols:
        raise ValueError("Provide at least one target ticker.")

    current_shares_map = _normalize_current_shares(current_shares)
    keep_symbols = _normalize_symbols(list(keep_tickers))
    all_symbols = list(dict.fromkeys(target_symbols + list(current_shares_map.keys()) + keep_symbols))

    quotes, price_source = fetch_quotes(all_symbols, timeout=timeout)
    missing = [symbol for symbol in all_symbols if symbol not in quotes]
    if missing:
        raise RuntimeError(f"Missing quote(s): {', '.join(missing)}")

    current_values_usd = {
        symbol: current_shares_map.get(symbol, 0.0) * quotes[symbol].price
        for symbol in all_symbols
    }
    eurusd, usdeur, fx_source = get_fx_rates(portfolio_ccy, timeout=timeout)
    current_total_usd = sum(current_values_usd.values())
    current_total_eur = current_total_usd * usdeur

    if amount_mode == "final_total":
        if portfolio_ccy == "USD":
            amount_usd = amount
            amount_eur = amount * usdeur
        else:
            amount_eur = amount
            amount_usd = amount * eurusd
        cash_to_deploy_usd = amount_usd - current_total_usd
        cash_to_deploy_eur = amount_eur - current_total_eur
    elif amount_mode == "cash_to_deploy":
        if portfolio_ccy == "USD":
            cash_to_deploy_usd = amount
            cash_to_deploy_eur = amount * usdeur
        else:
            cash_to_deploy_eur = amount
            cash_to_deploy_usd = amount * eurusd
        amount_usd = current_total_usd + cash_to_deploy_usd
        amount_eur = current_total_eur + cash_to_deploy_eur
    else:
        raise ValueError("amount_mode must be 'final_total' or 'cash_to_deploy'.")

    if cash_to_deploy_usd < -1e-9:
        raise ValueError(
            "Target final total is below the current marked value. Increase the target or use cash_to_deploy mode."
        )

    frozen_symbols = [symbol for symbol in keep_symbols if symbol not in target_symbols]
    frozen_value_usd = sum(current_values_usd[symbol] for symbol in frozen_symbols)
    target_bucket_usd = amount_usd - frozen_value_usd
    if target_bucket_usd <= 0:
        raise ValueError("No budget left for the equal-weight target basket after frozen positions.")

    target_current_values = [current_values_usd[symbol] for symbol in target_symbols]
    target_level_usd = _solve_buy_only_level(target_current_values, target_bucket_usd)

    buy_shares: Dict[str, int] = {}
    final_values_usd: Dict[str, float] = {}
    for symbol in target_symbols:
        gap_usd = max(0.0, target_level_usd - current_values_usd[symbol])
        buy_shares[symbol] = floor(gap_usd / quotes[symbol].price) if quotes[symbol].price > 0 else 0
        final_values_usd[symbol] = current_values_usd[symbol] + buy_shares[symbol] * quotes[symbol].price

    cash_left_usd = target_bucket_usd - sum(final_values_usd.values())

    # Spend the residual cash where one extra share best reduces the
    # distance to the common target level, while keeping a buy-only policy.
    while True:
        best_symbol: str | None = None
        best_improvement = 0.0
        for symbol in target_symbols:
            price = quotes[symbol].price
            if price > cash_left_usd:
                continue
            before = (final_values_usd[symbol] - target_level_usd) ** 2
            after = (final_values_usd[symbol] + price - target_level_usd) ** 2
            improvement = before - after
            if improvement > best_improvement + 1e-9:
                best_improvement = improvement
                best_symbol = symbol
        if best_symbol is None:
            break
        price = quotes[best_symbol].price
        buy_shares[best_symbol] += 1
        final_values_usd[best_symbol] += price
        cash_left_usd -= price

    buy_cost_usd = sum(buy_shares[symbol] * quotes[symbol].price for symbol in target_symbols)
    final_total_usd = current_total_usd + buy_cost_usd

    print("=" * 128)
    print(
        f"Portfolio amount: {amount:,.2f} {portfolio_ccy} | "
        f"EURUSD={eurusd:.6f} USDEUR={usdeur:.6f}"
    )
    print(f"Price source: {price_source} | FX source: {fx_source}")
    print(f"Current marked value: {current_total_usd:,.2f} USD")
    if amount_mode == "cash_to_deploy":
        print(f"Cash to deploy:      {cash_to_deploy_usd:,.2f} USD / {cash_to_deploy_eur:,.2f} EUR")
    print(f"Target final value:   {amount_usd:,.2f} USD / {amount_eur:,.2f} EUR")
    print(f"Frozen value outside target basket: {frozen_value_usd:,.2f} USD")
    print(f"Buy-only equal-weight target per basket name: {target_level_usd:,.2f} USD")
    print("-" * 128)
    print(
        f"{'Ticker':<10} {'Price(USD)':>12} {'CurrentShr':>11} {'CurrentUSD':>12} "
        f"{'BuyShr':>8} {'BuyUSD':>12} {'FinalShr':>11} {'FinalUSD':>12} {'FinalWt(%)':>11}"
    )

    for symbol in target_symbols:
        price = quotes[symbol].price
        current_shares_value = current_shares_map.get(symbol, 0.0)
        current_usd = current_values_usd[symbol]
        buy_count = buy_shares[symbol]
        buy_usd = buy_count * price
        final_shares = current_shares_value + buy_count
        final_usd = final_values_usd[symbol]
        final_weight = (final_usd / amount_usd * 100.0) if amount_usd > 0 else 0.0
        print(
            f"{symbol:<10} {price:>12.4f} {current_shares_value:>11.2f} {current_usd:>12.2f} "
            f"{buy_count:>8d} {buy_usd:>12.2f} {final_shares:>11.2f} {final_usd:>12.2f} {final_weight:>11.2f}"
        )

    if frozen_symbols:
        print("-" * 128)
        print("Frozen / kept positions outside equal-weight basket:")
        print(f"{'Ticker':<10} {'Price(USD)':>12} {'Shares':>11} {'CurrentUSD':>12} {'FinalWt(%)':>11}")
        for symbol in frozen_symbols:
            price = quotes[symbol].price
            shares = current_shares_map.get(symbol, 0.0)
            current_usd = current_values_usd[symbol]
            final_weight = (current_usd / amount_usd * 100.0) if amount_usd > 0 else 0.0
            print(f"{symbol:<10} {price:>12.4f} {shares:>11.2f} {current_usd:>12.2f} {final_weight:>11.2f}")

    print("-" * 128)
    print(f"Planned buy cost: {buy_cost_usd:,.2f} USD / {buy_cost_usd * usdeur:,.2f} EUR")
    print(f"Residual cash:    {cash_left_usd:,.2f} USD / {cash_left_usd * usdeur:,.2f} EUR")
    print(f"Final marked total:{final_total_usd + cash_left_usd:,.2f} USD / {(final_total_usd + cash_left_usd) * usdeur:,.2f} EUR")
    print("=" * 128)


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
    current_shares: Dict[str, float] | None = None,
    keep_tickers: List[str] | tuple[str, ...] = (),
    buy_only_equal_weight: bool = False,
    amount_mode: str = "final_total",
    watch: int = 0,
    timeout: int = 8,
) -> int:
    if amount <= 0:
        print("Error: --amount must be > 0.", file=sys.stderr)
        return 1

    tickers = _normalize_symbols(tickers)
    if not tickers:
        print("Error: provide at least one ticker.", file=sys.stderr)
        return 1

    if buy_only_equal_weight:
        if watch > 0:
            print("Error: live watch mode is not supported with buy_only_equal_weight.", file=sys.stderr)
            return 1
        try:
            allocate_buy_only_equal_weight(
                amount=amount,
                portfolio_ccy=currency,
                tickers=tickers,
                current_shares=current_shares,
                keep_tickers=keep_tickers,
                amount_mode=amount_mode,
                timeout=timeout,
            )
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
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

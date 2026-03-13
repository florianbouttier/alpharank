from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    statement: str
    metric: str
    eodhd_column: str
    sec_tags: tuple[str, ...]
    yfinance_rows: tuple[str, ...]


PILOT_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "META", "AMZN")

PRICE_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "ticker",
)

GENERAL_COLUMNS: tuple[str, ...] = ("ticker", "name", "exchange", "cik", "source")

METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        statement="income_statement",
        metric="revenue",
        eodhd_column="totalRevenue",
        sec_tags=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
        yfinance_rows=("Total Revenue", "Operating Revenue"),
    ),
    MetricSpec(
        statement="income_statement",
        metric="gross_profit",
        eodhd_column="grossProfit",
        sec_tags=("GrossProfit",),
        yfinance_rows=("Gross Profit",),
    ),
    MetricSpec(
        statement="income_statement",
        metric="operating_income",
        eodhd_column="operatingIncome",
        sec_tags=("OperatingIncomeLoss",),
        yfinance_rows=("Operating Income", "Total Operating Income As Reported"),
    ),
    MetricSpec(
        statement="income_statement",
        metric="net_income",
        eodhd_column="netIncome",
        sec_tags=("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
        yfinance_rows=("Net Income", "Net Income Common Stockholders"),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="total_assets",
        eodhd_column="totalAssets",
        sec_tags=("Assets",),
        yfinance_rows=("Total Assets",),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="total_liabilities",
        eodhd_column="totalLiab",
        sec_tags=("Liabilities", "LiabilitiesCurrentAndNoncurrent"),
        yfinance_rows=("Total Liabilities Net Minority Interest", "Total Liabilities"),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="stockholders_equity",
        eodhd_column="totalStockholderEquity",
        sec_tags=(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        yfinance_rows=("Stockholders Equity", "Common Stock Equity"),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="cash_and_equivalents",
        eodhd_column="cashAndEquivalents",
        sec_tags=("CashAndCashEquivalentsAtCarryingValue",),
        yfinance_rows=(),
    ),
    MetricSpec(
        statement="cash_flow",
        metric="operating_cash_flow",
        eodhd_column="totalCashFromOperatingActivities",
        sec_tags=("NetCashProvidedByUsedInOperatingActivities",),
        yfinance_rows=("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
    ),
    MetricSpec(
        statement="cash_flow",
        metric="capital_expenditures",
        eodhd_column="capitalExpenditures",
        sec_tags=("PaymentsToAcquirePropertyPlantAndEquipment",),
        yfinance_rows=("Capital Expenditure",),
    ),
    MetricSpec(
        statement="cash_flow",
        metric="free_cash_flow",
        eodhd_column="freeCashFlow",
        sec_tags=(),
        yfinance_rows=("Free Cash Flow",),
    ),
)


def specs_for_statement(statement: str) -> tuple[MetricSpec, ...]:
    return tuple(spec for spec in METRIC_SPECS if spec.statement == statement)

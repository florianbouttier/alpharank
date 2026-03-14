from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    statement: str
    metric: str
    eodhd_column: str
    eodhd_path: str
    sec_fact_roots: tuple[str, ...]
    sec_tags: tuple[str, ...]
    yfinance_rows: tuple[str, ...]
    simfin_datasets: tuple[str, ...] = ()
    simfin_columns: tuple[str, ...] = ()


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
        eodhd_path="US_Income_statement.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
        yfinance_rows=("Total Revenue", "Operating Revenue"),
        simfin_datasets=("income_statement",),
        simfin_columns=("Revenue", "Sales & Services Revenue"),
    ),
    MetricSpec(
        statement="income_statement",
        metric="gross_profit",
        eodhd_column="grossProfit",
        eodhd_path="US_Income_statement.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("GrossProfit",),
        yfinance_rows=("Gross Profit",),
        simfin_datasets=("income_statement",),
        simfin_columns=("Gross Profit",),
    ),
    MetricSpec(
        statement="income_statement",
        metric="operating_income",
        eodhd_column="operatingIncome",
        eodhd_path="US_Income_statement.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("OperatingIncomeLoss",),
        yfinance_rows=("Operating Income", "Total Operating Income As Reported"),
        simfin_datasets=("income_statement",),
        simfin_columns=("Operating Income (Loss)",),
    ),
    MetricSpec(
        statement="income_statement",
        metric="net_income",
        eodhd_column="netIncome",
        eodhd_path="US_Income_statement.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
        yfinance_rows=("Net Income", "Net Income Common Stockholders"),
        simfin_datasets=("income_statement",),
        simfin_columns=("Net Income", "Net Income (Common)"),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="total_assets",
        eodhd_column="totalAssets",
        eodhd_path="US_Balance_sheet.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("Assets",),
        yfinance_rows=("Total Assets",),
        simfin_datasets=("balance_sheet",),
        simfin_columns=("Total Assets",),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="total_liabilities",
        eodhd_column="totalLiab",
        eodhd_path="US_Balance_sheet.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("Liabilities", "LiabilitiesCurrentAndNoncurrent"),
        yfinance_rows=("Total Liabilities Net Minority Interest", "Total Liabilities"),
        simfin_datasets=("balance_sheet",),
        simfin_columns=("Total Liabilities",),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="stockholders_equity",
        eodhd_column="totalStockholderEquity",
        eodhd_path="US_Balance_sheet.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        yfinance_rows=("Stockholders Equity", "Common Stock Equity"),
        simfin_datasets=("balance_sheet",),
        simfin_columns=("Total Equity",),
    ),
    MetricSpec(
        statement="balance_sheet",
        metric="cash_and_equivalents",
        eodhd_column="cashAndEquivalents",
        eodhd_path="US_Balance_sheet.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("CashAndCashEquivalentsAtCarryingValue",),
        yfinance_rows=(),
        simfin_datasets=("balance_sheet",),
        simfin_columns=("Cash, Cash Equivalents & Short Term Investments",),
    ),
    MetricSpec(
        statement="cash_flow",
        metric="operating_cash_flow",
        eodhd_column="totalCashFromOperatingActivities",
        eodhd_path="US_Cash_flow.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("NetCashProvidedByUsedInOperatingActivities",),
        yfinance_rows=("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
        simfin_datasets=("cash_flow",),
        simfin_columns=("Net Cash from Operating Activities",),
    ),
    MetricSpec(
        statement="cash_flow",
        metric="capital_expenditures",
        eodhd_column="capitalExpenditures",
        eodhd_path="US_Cash_flow.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=("PaymentsToAcquirePropertyPlantAndEquipment",),
        yfinance_rows=("Capital Expenditure",),
        simfin_datasets=("cash_flow",),
        simfin_columns=("Change in Fixed Assets & Intangibles",),
    ),
    MetricSpec(
        statement="cash_flow",
        metric="free_cash_flow",
        eodhd_column="freeCashFlow",
        eodhd_path="US_Cash_flow.parquet",
        sec_fact_roots=("us-gaap",),
        sec_tags=(),
        yfinance_rows=("Free Cash Flow",),
        simfin_datasets=("derived",),
        simfin_columns=("Free Cash Flow",),
    ),
    MetricSpec(
        statement="shares",
        metric="outstanding_shares",
        eodhd_column="shares",
        eodhd_path="US_share.parquet",
        sec_fact_roots=("dei", "us-gaap"),
        sec_tags=("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
        yfinance_rows=("Ordinary Shares Number", "Share Issued"),
        simfin_datasets=("derived",),
        simfin_columns=("Shares Outstanding",),
    ),
    MetricSpec(
        statement="earnings",
        metric="eps_actual",
        eodhd_column="epsActual",
        eodhd_path="US_Earnings.parquet",
        sec_fact_roots=(),
        sec_tags=(),
        yfinance_rows=(),
    ),
    MetricSpec(
        statement="earnings",
        metric="eps_estimate",
        eodhd_column="epsEstimate",
        eodhd_path="US_Earnings.parquet",
        sec_fact_roots=(),
        sec_tags=(),
        yfinance_rows=(),
    ),
    MetricSpec(
        statement="earnings",
        metric="surprise_percent",
        eodhd_column="surprisePercent",
        eodhd_path="US_Earnings.parquet",
        sec_fact_roots=(),
        sec_tags=(),
        yfinance_rows=(),
    ),
)


def specs_for_statement(statement: str) -> tuple[MetricSpec, ...]:
    return tuple(spec for spec in METRIC_SPECS if spec.statement == statement)

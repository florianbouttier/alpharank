from typing import List,Optional
import numpy as np
import pandas as pd

def data_quality(df):
    """
    - Plot a bar chart of the percentage of NA per column.
    - Plot a line chart of the number of distinct tickers per Year-Month (x axis).
    Returns a dict with:
      - 'na_counts': Series of NA percentage per column
      - 'tickers_per_month': Series of distinct tickers per month
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 1) Percentage of NA per column
    na_counts = (df.isna().mean().sort_values(ascending=False) * 100.0)

    # 2) Detect useful columns
    ticker_col = next((c for c in ["ticker", "Ticker", "symbol", "Symbol"] if c in df.columns), None)
    if ticker_col is None:
        raise ValueError("Ticker column not found. Try 'ticker' or 'symbol'.")

    ym_col = next((c for c in [
        "year_month", "YearMonth", "yearmonth", "ym", "YM",
        "date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"
    ] if c in df.columns), None)

    if ym_col is None:
        # Fallback: any datetime/period-like column
        for c in df.columns:
            if isinstance(df[c].dtype, pd.PeriodDtype) or np.issubdtype(df[c].dtype, np.datetime64):
                ym_col = c
                break

    if ym_col is None:
        raise ValueError("No date/Year-Month column found (e.g., 'year_month', 'date').")

    # Robust conversion to Period[M]
    s = df[ym_col]
    if isinstance(s.dtype, pd.PeriodDtype):
        ym = s.dt.asfreq("M")
    else:
        s_dt = pd.to_datetime(s, errors="coerce")
        ym = s_dt.dt.to_period("M")

    mask = ym.notna() & df[ticker_col].notna()
    base = pd.DataFrame({
        "ym": ym[mask],
        "ticker": df.loc[mask, ticker_col].astype(str)
    }).dropna(subset=["ym", "ticker"])

    # Ensure 'ym' is Period[M]
    if isinstance(base["ym"].dtype, pd.PeriodDtype):
        base["ym"] = base["ym"].dt.asfreq("M")
    else:
        base["ym"] = pd.to_datetime(base["ym"], errors="coerce").dt.to_period("M")

    tickers_per_month = (
        base.groupby("ym")["ticker"]
        .nunique()
        .sort_index()
    )

    # Matplotlib: create figures
    rows = 2
    _fig, axes = plt.subplots(rows, 1, figsize=(12, max(6, min(12, 3 + 0.25 * len(na_counts)))))
    if rows == 1:
        axes = [axes]

    # Plot 1: NA percentage per column
    axes[0].barh(na_counts.index.astype(str), na_counts.values, color="tab:orange")
    axes[0].invert_yaxis()
    axes[0].set_title("Percentage of missing values (NA) per column")
    axes[0].set_xlabel("% NA")
    axes[0].set_ylabel("Columns")
    axes[0].grid(axis="x", alpha=0.3)

    # Plot 2: Distinct tickers per Year-Month
    if not tickers_per_month.empty:
        # Safer conversion for PeriodIndex -> DatetimeIndex for plotting
        plot_df = tickers_per_month.reset_index(name="n_tickers")
        if isinstance(plot_df["ym"].dtype, pd.PeriodDtype):
            x = plot_df["ym"].dt.to_timestamp()
        else:
            x = pd.to_datetime(plot_df["ym"], errors="coerce")
        axes[1].plot(x, plot_df["n_tickers"].values, marker="o", linestyle="-", color="tab:blue")
        axes[1].set_title("Number of distinct tickers per Year-Month")
        axes[1].set_xlabel("Year-Month")
        axes[1].set_ylabel("Nb tickers")
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    else:
        axes[1].text(0.5, 0.5, "No valid data for Year-Month / ticker", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()

    return {"na_counts": na_counts, "tickers_per_month": tickers_per_month}
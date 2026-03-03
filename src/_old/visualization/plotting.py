from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

class StockComparisonPlotter:
    """
    Plotting helper for comparing KPIs across a list of tickers using the output
    of FundamentalProcessor.calculate_all_ratios (monthly, one row per ticker/year_month).

    Key features:
    - HTML figures (Plotly) with a clean design.
    - Robust outlier handling (IQR-based clipping) to suppress extreme values.
    - Optional normalization (index to 100) and smoothing for time series.
    - Convenience methods to build standalone HTML or a combined HTML report.
    """

    def __init__(self, ratios_df: pd.DataFrame):
        """
        Args:
            ratios_df: DataFrame as returned by calculate_all_ratios with columns:
                       ['ticker', 'year_month', 'monthly_return', ... KPIs ...]
        """
        self.df = ratios_df.copy()
        # Ensure time index is timestamp (handle Period safely)
        if 'year_month' in self.df.columns:
            ym = self.df['year_month']
            if pd.api.types.is_period_dtype(ym):
                self.df['year_month'] = ym.dt.to_timestamp()
            elif not pd.api.types.is_datetime64_any_dtype(ym):
                self.df['year_month'] = pd.to_datetime(ym)
        # Common Plotly defaults
        pio.templates.default = "plotly_white"
        self._default_color_discrete = px.colors.qualitative.D3

    # ------------- helpers -------------
    @staticmethod
    def _iqr_clip(series: pd.Series, iqr_multiplier: float = 2.5) -> pd.Series:
        """Clip series to [Q1 - k*IQR, Q3 + k*IQR] to suppress extreme outliers."""
        if series.dropna().empty:
            return series
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or not np.isfinite(iqr):
            return series
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def _ema(series: pd.Series, span: int = 3) -> pd.Series:
        """Simple EMA smoothing for nicer lines."""
        if series.dropna().empty:
            return series
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _index_to_100(series: pd.Series) -> pd.Series:
        """Normalize a series to start at 100 (first non-null)."""
        first = series.dropna().iloc[0] if series.dropna().size else np.nan
        if pd.isna(first) or first == 0:
            return series
        return 100 * series / first

    def _validate_inputs(self, tickers: List[str], kpis: List[str]) -> List[str]:
        missing_cols = [c for c in kpis if c not in self.df.columns]
        if missing_cols:
            # Only keep existing columns; silently drop missing ones
            kpis = [c for c in kpis if c in self.df.columns]
        # Filter to requested tickers if provided
        return kpis

    def _subset(self,
                tickers: Optional[List[str]] = None,
                start: Optional[Union[str, pd.Timestamp]] = None,
                end: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        d = self.df.copy()
        if tickers:
            d = d[d['ticker'].isin(tickers)]
        if start is not None:
            d = d[d['year_month'] >= pd.to_datetime(start)]
        if end is not None:
            d = d[d['year_month'] <= pd.to_datetime(end)]
        return d.sort_values(['ticker', 'year_month'])

    # ------------- public API -------------

    def plot_time_series(
        self,
        tickers: List[str],
        kpis: List[str],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        normalize: bool = False,
        smooth_span: Optional[int] = 3,
        iqr_multiplier: float = 2.5,
        height: int = 420
    ) -> Dict[str, str]:
        """
        Line charts per KPI across requested tickers.

        Returns:
            Dict[kpi, html_str]
        """
        kpis = self._validate_inputs(tickers, kpis)
        d = self._subset(tickers, start, end)

        figs_html: Dict[str, str] = {}
        for kpi in kpis:
            dd = d[['ticker', 'year_month', kpi]].dropna().copy()
            if dd.empty:
                figs_html[kpi] = ""
                continue

            # Outlier suppression per KPI (cross-sectionally across all tickers/time in the window)
            dd[kpi] = self._iqr_clip(dd[kpi], iqr_multiplier=iqr_multiplier)

            # Optional smoothing per ticker
            if smooth_span and smooth_span > 1:
                dd[kpi] = dd.groupby('ticker')[kpi].transform(lambda s: self._ema(s, span=smooth_span))

            # Optional normalization per ticker
            if normalize:
                dd[kpi] = dd.groupby('ticker')[kpi].transform(self._index_to_100)
                y_title = f"{kpi} (Indexed to 100)"
            else:
                y_title = kpi

            fig = px.line(
                dd,
                x='year_month',
                y=kpi,
                color='ticker',
                color_discrete_sequence=self._default_color_discrete,
                markers=False,
                title=f"{kpi} - Time Series",
            )
            fig.update_layout(
                height=height,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig.update_xaxes(title="Date", showgrid=True)
            fig.update_yaxes(title=y_title, showgrid=True, zeroline=False)

            figs_html[kpi] = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        return figs_html

    def plot_latest_bars(
        self,
        tickers: List[str],
        kpis: List[str],
        iqr_multiplier: float = 2.5,
        height: int = 420
    ) -> Dict[str, str]:
        """
        Bar charts for the latest available value per KPI and ticker.

        Returns:
            Dict[kpi, html_str]
        """
        kpis = self._validate_inputs(tickers, kpis)
        d = self._subset(tickers)
        if d.empty:
            return {k: "" for k in kpis}

        # Latest per ticker
        latest_idx = d.groupby('ticker')['year_month'].idxmax()
        latest = d.loc[latest_idx, ['ticker', 'year_month'] + kpis].copy()

        figs_html: Dict[str, str] = {}
        for kpi in kpis:
            dd = latest[['ticker', kpi]].dropna().copy()
            if dd.empty:
                figs_html[kpi] = ""
                continue
            dd[kpi] = self._iqr_clip(dd[kpi], iqr_multiplier=iqr_multiplier)

            fig = px.bar(
                dd,
                x='ticker',
                y=kpi,
                color='ticker',
                color_discrete_sequence=self._default_color_discrete,
                title=f"{kpi} - Latest values",
            )
            fig.update_layout(
                height=height,
                showlegend=False,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig.update_xaxes(title="")
            fig.update_yaxes(title=kpi, zeroline=False, showgrid=True)

            figs_html[kpi] = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        return figs_html

    def plot_scatter_matrix(
        self,
        tickers: List[str],
        kpis: List[str],
        iqr_multiplier: float = 2.5,
        height: int = 700
    ) -> str:
        """
        Scatter-matrix (at latest date) across KPIs for requested tickers.

        Returns:
            html_str
        """
        kpis = self._validate_inputs(tickers, kpis)
        d = self._subset(tickers)
        if d.empty or not kpis:
            return ""

        latest_idx = d.groupby('ticker')['year_month'].idxmax()
        latest = d.loc[latest_idx, ['ticker'] + kpis].copy()
        # Clip each KPI
        for k in kpis:
            latest[k] = self._iqr_clip(latest[k], iqr_multiplier=iqr_multiplier)

        fig = px.scatter_matrix(
            latest,
            dimensions=kpis,
            color='ticker',
            color_discrete_sequence=self._default_color_discrete,
            title="KPI Scatter Matrix (latest)"
        )
        fig.update_layout(height=height, margin=dict(l=40, r=20, t=60, b=40))
        return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

    def make_report(
        self,
        tickers: List[str],
        kpis: List[str],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        normalize: bool = False,
        smooth_span: Optional[int] = 3,
        iqr_multiplier: float = 2.5,
        include_scatter_matrix: bool = True
    ) -> str:
        """
        Build a single HTML string containing:
        - Time-series per KPI
        - Latest-value bars per KPI
        - Optional scatter-matrix for latest cross-section
        """
        ts_html = self.plot_time_series(
            tickers=tickers,
            kpis=kpis,
            start=start,
            end=end,
            normalize=normalize,
            smooth_span=smooth_span,
            iqr_multiplier=iqr_multiplier,
        )
        bars_html = self.plot_latest_bars(
            tickers=tickers,
            kpis=kpis,
            iqr_multiplier=iqr_multiplier,
        )
        scatter_html = self.plot_scatter_matrix(
            tickers=tickers,
            kpis=kpis,
            iqr_multiplier=iqr_multiplier,
        ) if include_scatter_matrix else ""

        # Simple combined HTML with minimal CSS
        blocks = []
        style = """
        <style>
        body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
        .section { margin-bottom: 36px; }
        .grid { display: grid; grid-template-columns: 1fr; gap: 24px; }
        .title { font-size: 20px; font-weight: 600; margin: 6px 0 12px; }
        .subtitle { font-size: 16px; font-weight: 500; margin: 18px 0 8px; color: #444; }
        hr { border: none; border-top: 1px solid #eee; margin: 20px 0; }
        </style>
        """

        # Header
        blocks.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'>{style}</head><body>")
        blocks.append(f"<div class='section'><div class='title'>Stock KPI Comparison</div>")
        blocks.append(f"<div>Tickers: {', '.join(tickers)}</div>")
        blocks.append(f"<div>KPIs: {', '.join(kpis)}</div></div>")

        # Time series
        blocks.append("<div class='section'><div class='subtitle'>Time Series</div><div class='grid'>")
        for k, html in ts_html.items():
            if html:
                blocks.append(html)
        blocks.append("</div></div><hr/>")

        # Latest bars
        blocks.append("<div class='section'><div class='subtitle'>Latest Values</div><div class='grid'>")
        for k, html in bars_html.items():
            if html:
                blocks.append(html)
        blocks.append("</div></div>")

        # Scatter matrix
        if scatter_html:
            blocks.append("<hr/><div class='section'><div class='subtitle'>Scatter Matrix (Latest)</div>")
            blocks.append(scatter_html)
            blocks.append("</div>")

        blocks.append("</body></html>")
        return "\n".join(blocks)

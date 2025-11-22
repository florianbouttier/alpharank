import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from typing import List, Optional, Union, Dict
import datetime

class FinancialReportGenerator:
    """
    Generates premium HTML reports for comparing financial KPIs across multiple tickers.
    
    Features:
    - Robust outlier handling (Winsorization)
    - Missing data visualization
    - Interactive Plotly charts (Radar, Heatmap, Time Series)
    - Premium CSS styling (Dark mode, Glassmorphism)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame containing 'ticker', 'year_month', and KPI columns.
        """
        self.df = df.copy()
        
        # Ensure date format
        if 'year_month' in self.df.columns:
            if pd.api.types.is_period_dtype(self.df['year_month']):
                self.df['year_month'] = self.df['year_month'].dt.to_timestamp()
            else:
                self.df['year_month'] = pd.to_datetime(self.df['year_month'])
        
        # Set Plotly theme
        pio.templates.default = "plotly_dark"
        self.colors = px.colors.qualitative.Pastel

    def _filter_data(self, tickers: List[str], start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Filters data by ticker and date range."""
        d = self.df[self.df['ticker'].isin(tickers)].copy()
        
        if start_date:
            d = d[d['year_month'] >= pd.to_datetime(start_date)]
        if end_date:
            d = d[d['year_month'] <= pd.to_datetime(end_date)]
            
        return d.sort_values(['ticker', 'year_month'])

    def _handle_outliers(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """ robust winsorization using IQR."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return series.clip(lower=lower, upper=upper)

    # KPI Configuration: Name mapping and format type
    KPI_CONFIG = {
        'pe': {'name': 'P/E Ratio', 'format': 'number'},
        'pnetresult': {'name': 'Price / Net Income', 'format': 'number'},
        'ps_ratio': {'name': 'Price / Sales', 'format': 'number'},
        'pb_ratio': {'name': 'Price / Book', 'format': 'number'},
        'ev_ebitda_ratio': {'name': 'EV / EBITDA', 'format': 'number'},
        'gross_margin': {'name': 'Gross Margin', 'format': 'percent'},
        'netmargin': {'name': 'Net Margin', 'format': 'percent'},
        'ebitmargin': {'name': 'EBIT Margin', 'format': 'percent'},
        'ebitdamargin': {'name': 'EBITDA Margin', 'format': 'percent'},
        'return_on_equity': {'name': 'ROE', 'format': 'percent'},
        'return_on_assets': {'name': 'ROA', 'format': 'percent'},
        'roic': {'name': 'ROIC', 'format': 'percent'},
        'net_income_growth': {'name': 'Net Income Growth (TTM)', 'format': 'percent'},
        'asset_turnover': {'name': 'Asset Turnover', 'format': 'number'},
        'debt_to_equity': {'name': 'Debt / Equity', 'format': 'number'},
        'market_cap': {'name': 'Market Cap', 'format': 'currency'},
        'enterprise_value': {'name': 'Enterprise Value', 'format': 'currency'},
    }

    def _get_kpi_info(self, kpi: str) -> Dict:
        """Returns name and format for a KPI, with defaults."""
        return self.KPI_CONFIG.get(kpi, {'name': kpi, 'format': 'number'})

    def generate_heatmap(self, latest_df: pd.DataFrame, kpis: List[str]) -> str:
        """Generates a Heatmap of latest KPI values."""
        # Pivot for heatmap format: Index=Ticker, Columns=KPI
        pivot_df = latest_df.set_index('ticker')[kpis]
        
        # Z-score normalization for color scale (per KPI)
        z_scores = (pivot_df - pivot_df.mean()) / pivot_df.std()
        
        # Prepare display text
        display_text = pivot_df.copy().astype(str)
        x_labels = []
        
        for kpi in kpis:
            info = self._get_kpi_info(kpi)
            x_labels.append(info['name'])
            if info['format'] == 'percent':
                display_text[kpi] = pivot_df[kpi].apply(lambda x: f"{x:.1%}")
            elif info['format'] == 'currency':
                display_text[kpi] = pivot_df[kpi].apply(lambda x: f"${x:,.0f}")
            else:
                display_text[kpi] = pivot_df[kpi].apply(lambda x: f"{x:.2f}")

        fig = go.Figure(data=go.Heatmap(
            z=z_scores.values,
            x=x_labels,
            y=pivot_df.index,
            colorscale='Viridis',
            text=display_text.values, # Show formatted values
            texttemplate="%{text}", # Show actual values in cells
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Relative Strength Heatmap (Z-Score)",
            height=400 + (len(pivot_df) * 20)
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    def generate_time_series(self, df: pd.DataFrame, kpis: List[str]) -> str:
        """Generates interactive time series charts for each KPI."""
        html_parts = []
        
        for kpi in kpis:
            info = self._get_kpi_info(kpi)
            nice_name = info['name']
            fmt = info['format']
            
            # Handle outliers for plotting
            plot_df = df.copy()
            plot_df[kpi] = self._handle_outliers(plot_df[kpi])
            
            fig = px.line(
                plot_df, 
                x='year_month', 
                y=kpi, 
                color='ticker',
                color_discrete_sequence=self.colors,
                title=f"{nice_name} Evolution"
            )
            
            # Determine tick format
            tick_fmt = ""
            if fmt == 'percent':
                tick_fmt = ".1%"
            
            fig.update_layout(
                hovermode="x unified",
                xaxis=dict(showgrid=False, title="Date"),
                yaxis=dict(
                    showgrid=True, 
                    tickformat=tick_fmt,
                    title=nice_name
                ),
                height=350
            )
            html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
            
        return "".join([f"<div class='chart-container'>{h}</div>" for h in html_parts])

    def generate_missing_data_viz(self, df: pd.DataFrame, kpis: List[str]) -> str:
        """Visualizes missing data patterns."""
        # Heatmap of % missing per ticker/month across all KPIs
        missing_agg = df.set_index(['ticker', 'year_month'])[kpis].isna().mean(axis=1).reset_index()
        missing_agg.columns = ['ticker', 'year_month', 'missing_pct']
        
        fig = px.density_heatmap(
            missing_agg,
            x='year_month',
            y='ticker',
            z='missing_pct',
            nbinsx=50,
            title="Data Gaps Heatmap (Brighter = More Missing Data)",
            color_continuous_scale='Magma',
            text_auto=False
        )
        
        fig.update_layout(
            height=300,
            coloraxis_colorbar=dict(
                title="Missing %",
                tickformat=".0%"
            )
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    def generate_report(
        self, 
        tickers: List[str], 
        kpis: List[str], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        title: str = "Financial Comparison Report"
    ) -> str:
        """
        Generates the full HTML report.
        """
        # 1. Filter Data
        filtered_df = self._filter_data(tickers, start_date, end_date)
        
        if filtered_df.empty:
            return "<h1>No data found for the specified criteria.</h1>"

        # 2. Get Latest Data
        latest_df = filtered_df.groupby('ticker').last().reset_index()

        # 3. Generate Visualizations
        # Removed Radar Chart as requested
        heatmap_html = self.generate_heatmap(latest_df, kpis)
        timeseries_html = self.generate_time_series(filtered_df, kpis)
        missing_html = self.generate_missing_data_viz(filtered_df, kpis)
        
        # 4. Build HTML
        css = """
        <style>
            :root {
                --bg-color: #f5f5f5;
                --card-bg: #ffffff;
                --text-color: #333333;
                --accent-color: #6200EE;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                margin: 0;
                padding: 20px;
            }
            .container {
                max_width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 { color: var(--text-color); }
            .header {
                text-align: center;
                margin-bottom: 40px;
                padding: 20px;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }
            .grid-1 {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: var(--card-bg);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            .chart-container {
                margin-bottom: 20px;
                background: var(--card-bg);
                border-radius: 16px;
                padding: 15px;
            }
        </style>
        """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            {css}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Comparing: {", ".join(tickers)}</p>
                    <p>Period: {start_date or 'Start'} to {end_date or 'End'}</p>
                </div>
                
                <h2>Market Position</h2>
                <div class="grid-1">
                    <div class="card">
                        <h3>Relative Strength Heatmap</h3>
                        <p style="font-size: 0.9em; opacity: 0.7;">Colors indicate Z-Score (relative deviation from group mean). Values shown are actuals.</p>
                        {heatmap_html}
                    </div>
                </div>
                
                <h2>Historical Evolution</h2>
                <div class="card">
                    {timeseries_html}
                </div>
                
                <h2>Data Quality Analysis</h2>
                <div class="card">
                    {missing_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

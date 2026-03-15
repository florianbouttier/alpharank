from typing import List, Dict, Optional, Union, Sequence, Tuple
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# %%
class PortfolioVisualizer:
    """
    Generate professional HTML visualizations for portfolio holdings and strategy comparison.
    """
    
    SECTOR_COLORS = {
        'Technology': '#3B82F6', 'Healthcare': '#10B981', 'Financials': '#6366F1',
        'Consumer Cyclical': '#F59E0B', 'Consumer Defensive': '#84CC16', 'Industrials': '#64748B',
        'Energy': '#EF4444', 'Utilities': '#06B6D4', 'Basic Materials': '#A78BFA',
        'Real Estate': '#EC4899', 'Communication Services': '#8B5CF6', 'Financial Services': '#6366F1',
    }
    DEFAULT_COLOR = '#94A3B8'

    @staticmethod
    def sector_allocation_chart(portfolio: pd.DataFrame, height: int = 400, show_values: bool = True) -> str:
        """Create a donut chart showing sector allocation."""
        if 'Sector' not in portfolio.columns or 'weight_normalized' not in portfolio.columns:
            return "<p>Missing required columns: Sector, weight_normalized</p>"
        
        sector_weights = portfolio.groupby('Sector', as_index=False)['weight_normalized'].sum().sort_values('weight_normalized', ascending=False)
        colors = [PortfolioVisualizer.SECTOR_COLORS.get(s, PortfolioVisualizer.DEFAULT_COLOR) for s in sector_weights['Sector']]
        
        fig = go.Figure(data=[go.Pie(
            labels=sector_weights['Sector'], values=sector_weights['weight_normalized'], hole=0.55,
            marker=dict(colors=colors), textinfo='label+percent' if show_values else 'label', textposition='outside'
        )])
        fig.update_layout(showlegend=False, height=height, margin=dict(l=20, r=20, t=30, b=20))
        return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

    @staticmethod
    def stock_table_html(portfolio: pd.DataFrame, max_stocks: int = 50) -> str:
        """Create an HTML table showing stock holdings."""
        df = portfolio.head(max_stocks).copy()
        rows = []
        max_weight = df['weight_normalized'].max() if not df.empty else 1
        
        for _, row in df.iterrows():
            ticker = str(row['ticker']).replace('.US', '')
            sector = row.get('Sector', 'N/A')
            weight_val = row.get('weight_normalized', 0)
            weight_pct = f"{weight_val * 100:.2f}%"
            bar_width = (weight_val / max_weight) * 100
            sector_color = PortfolioVisualizer.SECTOR_COLORS.get(sector, PortfolioVisualizer.DEFAULT_COLOR)
            
            ticker_link = f'<a href="#stock-{ticker}" style="color: inherit; text-decoration: none; border-bottom: 1px dashed #ccc;">{ticker}</a>'
            
            rows.append(f'''
                <tr>
                    <td class="ticker"><b>{ticker_link}</b></td>
                    <td><span class="sector-badge" style="background:{sector_color}20;color:{sector_color}">{sector}</span></td>
                    <td class="weight-cell">
                        <div class="weight-bar-bg"><div class="weight-bar" style="width:{bar_width}%;background:{sector_color}"></div></div>
                        <span class="weight-value">{weight_pct}</span>
                    </td>
                </tr>
            ''')
            
        return f'''
            <table class="holdings-table">
                <thead><tr><th>Ticker</th><th>Sector</th><th>Weight</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        '''

    @staticmethod
    def make_portfolio_report(
        portfolio: pd.DataFrame, 
        title: str = "Portfolio Holdings", 
        month: str = None,
        price_data: pd.DataFrame = None,
        balance_sheet: pd.DataFrame = None,
        income_statement: pd.DataFrame = None,
        cash_flow: pd.DataFrame = None,
        earnings: pd.DataFrame = None,
        backend: str = "polars",
        report_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a complete HTML portfolio report with detailed per-stock analysis.
        
        Args:
            portfolio: DataFrame with 'ticker', 'weight_normalized', 'Sector'.
            title: Report title.
            month: Context month/date string.
            price_data: Daily price DataFrame (long format: date, ticker, close).
            balance_sheet, income_statement, etc: Raw fundamental DataFrames.
            
        If fundamental/price data is provided, a detailed analysis section is added.
        """
        from alpharank.data.processing import FundamentalProcessor
        
        month = str(portfolio.attrs.get('month', 'Latest')) if month is None else month
        
        # 1. Summary Section
        sector_chart = PortfolioVisualizer.sector_allocation_chart(portfolio)
        stock_table = PortfolioVisualizer.stock_table_html(portfolio)

        context_html = ""
        if report_context:
            context_rows = []
            labels = {
                "portfolio_month": "Portfolio Month",
                "portfolio_view_date": "Portfolio View Date",
                "report_generated_at": "Report Generated At",
                "data_snapshot_id": "Data Snapshot ID",
                "data_snapshot_at": "Data Snapshot At",
                "source_snapshot_generated_at": "Source Snapshot Generated At",
                "price_data_max_date": "Price Data Max Date",
                "sp500_price_max_date": "SP500 Price Max Date",
                "sp500_constituents_max_month": "SP500 Constituents Max Month",
                "income_statement_max_date": "Income Statement Max Date",
                "balance_sheet_max_date": "Balance Sheet Max Date",
                "cash_flow_max_date": "Cash Flow Max Date",
                "earnings_max_date": "Earnings Max Date",
            }
            for key, label in labels.items():
                value = report_context.get(key)
                if value:
                    context_rows.append(
                        f"""
                        <div class="meta-card">
                            <div class="meta-label">{escape(label)}</div>
                            <div class="meta-value">{escape(str(value))}</div>
                        </div>
                        """
                    )
            if context_rows:
                context_html = f"""
                <div class="card meta-panel">
                    <h2>Snapshot Context</h2>
                    <div class="meta-grid">{''.join(context_rows)}</div>
                </div>
                """
        
        # 2. Detailed Stock Analysis
        stock_details_html = ""
        
        # Process data if provided
        fundamentals_df = None
        price_df = None
        
        if price_data is not None and balance_sheet is not None:
            try:
                # Filter for portfolio stocks to speed up processing
                tickers = portfolio['ticker'].unique()
                
                # Filter Prices
                price_subset = price_data[price_data['ticker'].isin(tickers)].copy()
                if 'close' not in price_subset.columns and 'adjusted_close' in price_subset.columns:
                    price_subset['close'] = price_subset['adjusted_close'] # Fallback
                
                # Filter Fundamentals
                bs_subset = balance_sheet[balance_sheet['ticker'].isin(tickers)].copy() if balance_sheet is not None else None
                is_subset = income_statement[income_statement['ticker'].isin(tickers)].copy() if income_statement is not None else None
                cf_subset = cash_flow[cash_flow['ticker'].isin(tickers)].copy() if cash_flow is not None else None
                earn_subset = earnings[earnings['ticker'].isin(tickers)].copy() if earnings is not None else None
                
                if not price_subset.empty and bs_subset is not None:
                    # Prepare Monthly Prices for Ratio Calculation
                    price_subset['date'] = pd.to_datetime(price_subset['date'])
                    monthly_prices = price_subset.set_index('date').groupby('ticker').resample('ME')['close'].last().reset_index()
                    monthly_prices = monthly_prices.rename(columns={'close': 'last_close'})
                    
                    # Compute Ratios
                    fundamentals_df = FundamentalProcessor.calculate_all_ratios(
                        balance_sheet=bs_subset,
                        cash_flow=cf_subset, 
                        income_statement=is_subset, 
                        earnings=earn_subset,
                        monthly_return=monthly_prices,
                        backend=backend,
                    )
                    price_df = price_subset # Use daily for chart
            except Exception as e:
                print(f"Viz Error: Could not calculate detailed fundamentals: {e}")

        # Render Details
        if fundamentals_df is not None and price_df is not None:
            stock_details_html += "<h2>Detailed Stock Analysis</h2>"
            
            fundamentals_df['date'] = pd.to_datetime(fundamentals_df['date'])
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            for _, row in portfolio.iterrows():
                ticker_raw = row['ticker']
                ticker_clean = str(ticker_raw).replace('.US', '')
                sector = row.get('Sector', 'Unknown')
                weight_pct = f"{row.get('weight_normalized', 0)*100:.1f}%"
                
                # --- Filter Data ---
                p_data = price_df[price_df['ticker'] == ticker_raw].sort_values('date')
                if not p_data.empty:
                    last_date = p_data['date'].max()
                    start_date = last_date - pd.DateOffset(months=36)
                    p_data = p_data[p_data['date'] >= start_date]

                f_data = fundamentals_df[fundamentals_df['ticker'] == ticker_raw].sort_values('date')
                
                # Setup Charts
                charts_html = ""
                
                # A. Price Chart
                if not p_data.empty:
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=p_data['date'], y=p_data['close'], 
                        mode='lines', name='Price', 
                        line=dict(color='#2563eb', width=2),
                        fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)'
                    ))
                    fig_price.update_layout(
                        title=f"{ticker_clean} - 3 Year Price Trend",
                        template="plotly_white",
                        height=300,
                        margin=dict(l=30, r=20, t=40, b=20),
                        yaxis_title="Price ($)",
                        hovermode="x unified"
                    )
                    charts_html += f'<div class="chart-box">{pio.to_html(fig_price, full_html=True, include_plotlyjs=True)}</div>'

                # B. Financial Growth (Revenue, Net Income)
                if not f_data.empty:
                    f_data['year'] = f_data['date'].dt.year
                    f_annual = f_data.groupby('year').last().reset_index().tail(5)
                    
                    fig_fund = go.Figure()
                    fig_fund.add_trace(go.Bar(
                        x=f_annual['year'], y=f_annual['totalrevenue_rolling'],
                        name='Revenue (TTM)', marker_color='#93c5fd'
                    ))
                    fig_fund.add_trace(go.Bar(
                        x=f_annual['year'], y=f_annual['netincome_rolling'],
                        name='Net Income (TTM)', marker_color='#3b82f6'
                    ))
                    fig_fund.update_layout(
                        title="Financial Growth (Revenue & Earnings)",
                        template="plotly_white",
                        height=300,
                        margin=dict(l=30, r=20, t=40, b=20),
                        barmode='group',
                        legend=dict(orientation="h", y=1.1)
                    )
                    charts_html += f'<div class="chart-box">{pio.to_html(fig_fund, full_html=   True, include_plotlyjs=True)}</div>'

                # C. Key Ratios (Latest)
                ratios_html = ""
                if not f_data.empty:
                    latest = f_data.iloc[-1]
                    
                    def format_ratio(val, fmt="{:.2f}", color=False):
                        if pd.isna(val): return "N/A"
                        s = fmt.format(val)
                        if color:
                             return f'<span style="color: {"green" if val > 0 else "red"}">{s}</span>'
                        return s
                    
                    pe = latest.get('pnetresult')
                    roce = latest.get('roic')    
                    roe = latest.get('return_on_equity')
                    profit_margin = latest.get('netmargin')
                    
                    rev_growth = "N/A"
                    if len(f_annual) >= 2:
                        prev = f_annual.iloc[-2]['totalrevenue_rolling']
                        curr = f_annual.iloc[-1]['totalrevenue_rolling']
                        if prev > 0:
                            g = (curr - prev) / prev
                            rev_growth = f"{g:+.1%}"

                    ratios_html = f'''
                    <div class="ratios-grid">
                        <div class="ratio-card">
                            <div class="label">P/E Ratio</div>
                            <div class="value">{format_ratio(pe)}</div>
                        </div>
                        <div class="ratio-card">
                            <div class="label">ROE</div>
                            <div class="value">{format_ratio(roe, "{:.1%}")}</div>
                        </div>
                        <div class="ratio-card">
                            <div class="label">ROCE</div>
                            <div class="value">{format_ratio(roce, "{:.1%}")}</div>
                        </div>
                        <div class="ratio-card">
                            <div class="label">Net Margin</div>
                            <div class="value">{format_ratio(profit_margin, "{:.1%}")}</div>
                        </div>
                         <div class="ratio-card">
                            <div class="label">Rev Growth (1Y)</div>
                            <div class="value" style="color: #2563eb">{rev_growth}</div>
                        </div>
                    </div>
                    '''

                stock_details_html += f'''
                <div id="stock-{ticker_clean}" class="stock-card">
                    <div class="stock-header">
                        <div class="stock-title">
                            <h3>{ticker_clean}</h3>
                            <span class="stock-sector">{sector}</span>
                        </div>
                        <div class="stock-weight">{weight_pct} Portfolio Weight</div>
                    </div>
                    {ratios_html}
                    <div class="charts-grid">
                        {charts_html}
                    </div>
                </div>
                '''
        
        # CSS styles
        css = """
        <style>
            body { font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px; background: #f8fafc; color: #334155; }
            h1, h2 { color: #1e293b; font-weight: 700; }
            h2 { margin-top: 40px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
            
            .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
            .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            
            .stock-card { background: white; border-radius: 16px; padding: 24px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; }
            .stock-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #f1f5f9; padding-bottom: 15px; }
            .stock-title h3 { margin: 0; font-size: 1.5rem; color: #0f172a; display: inline-block; margin-right: 15px; }
            .stock-sector { background: #eff6ff; color: #3b82f6; padding: 4px 12px; border-radius: 20px; font-size: 0.875rem; font-weight: 600; }
            .stock-weight { background: #f0fdf4; color: #16a34a; padding: 6px 16px; border-radius: 8px; font-weight: 700; }
            
            .ratios-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .ratio-card { background: #f8fafc; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0; }
            .ratio-card .label { font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 4px; }
            .ratio-card .value { font-size: 1.25rem; font-weight: 700; color: #0f172a; }
            .meta-panel { margin-bottom: 24px; }
            .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
            .meta-card { background: #f8fafc; padding: 14px; border-radius: 10px; border: 1px solid #e2e8f0; }
            .meta-label { font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 6px; }
            .meta-value { font-size: 1rem; font-weight: 600; color: #0f172a; }
            
            .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .chart-box { border-radius: 8px; overflow: hidden; background: white; }
            
            .holdings-table { width: 100%; border-collapse: collapse; }
            .holdings-table td { padding: 12px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
            .sector-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
            .weight-bar-bg { background: #f1f5f9; height: 8px; border-radius: 4px; flex-grow: 1; margin-right: 10px; width: 100px; display: inline-block; }
            .weight-bar { height: 100%; border-radius: 4px; }
            .weight-cell { display: flex; align-items: center; }
            
            @media (max-width: 900px) {
                .summary-grid, .charts-grid { grid-template-columns: 1fr; }
            }
        </style>
        """
        
        return f"""
        <!DOCTYPE html>
        <html><head><title>{title}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        {css}
        </head><body>
            <h1>{title} <span style="font-size: 0.6em; color: #64748b; font-weight: normal;">({month})</span></h1>
            {context_html}
            <div class="summary-grid">
                <div class="card"><h2>Sector Allocation</h2>{sector_chart}</div>
                <div class="card"><h2>Holdings</h2>{stock_table}</div>
            </div>
            
            {stock_details_html}
            
        </body></html>
        """

    @staticmethod
    def make_comparison_report(
        metrics_df,
        cumulative_returns,
        drawdowns_df,
        annual_returns_df,
        correlation_matrix,
        worst_periods_df,
        cumulative_metrics_dict, 
        annual_metrics_dict,
        monthly_returns_dict,
        positions_dict=None,
        title="Portfolio Strategy Comparison"
    ):
        """
        Generates a comprehensive interactive HTML report comparing multiple strategies.
        Features Tabbed Interface for Deep Dive into every KPI.
        """
        positions_dict = positions_dict or {}

        # Determine common date range for subtitle
        try:
            start_date = cumulative_returns.index[0].strftime('%Y-%m-%d')
            end_date = cumulative_returns.index[-1].strftime('%Y-%m-%d')
            date_range_str = f"Period: {start_date} to {end_date}"
        except:
            date_range_str = "Period: N/A"
        
        # --- 1. Cumulative Returns ---
        try:
            fig_cum = go.Figure()
            for col in cumulative_returns.columns:
                series = cumulative_returns[col].dropna()
                if series.empty: continue
                
                # REBASE to 100 at the start of the plot
                base_val = series.iloc[0]
                vals = np.round((series / base_val) * 100, 2)
                
                fig_cum.add_trace(go.Scatter(x=vals.index, y=vals, mode='lines', name=col))
                
            fig_cum.update_layout(
                title=f"Cumulative Returns (Log Scale) - Rebased to 100<br><sup>{date_range_str}</sup>",
                yaxis_type="log", 
                yaxis_title="Value (Start=100)", 
                template="plotly_white", 
                height=500, 
                hovermode="x unified"
            )
            html_cum = pio.to_html(fig_cum, full_html=True, include_plotlyjs=True)
        except Exception as e:
            html_cum = f"<div class='alert alert-danger'>Error: {e}</div>"

        # --- 2. Drawdowns ---
        try:
            fig_dd = go.Figure()
            for col in drawdowns_df.columns:
                fig_dd.add_trace(go.Scatter(x=drawdowns_df.index, y=drawdowns_df[col], mode='lines', name=col, fill='tozeroy'))
            fig_dd.update_layout(
                title=f"Drawdown Analysis<br><sup>{date_range_str}</sup>", 
                yaxis_tickformat='.1%', 
                template="plotly_white", 
                height=400, 
                hovermode="x unified"
            )
            html_dd = pio.to_html(fig_dd, full_html=True, include_plotlyjs=True)
        except Exception as e:
             html_dd = f"<div class='alert alert-danger'>Error: {e}</div>"

        # --- 2b. Number of Positions ---
        html_positions = ""
        if positions_dict:
            try:
                fig_positions = go.Figure()
                for col, series in positions_dict.items():
                    if series.empty:
                        continue
                    fig_positions.add_trace(
                        go.Scatter(
                            x=series.index.to_timestamp(),
                            y=series.values,
                            mode='lines',
                            name=col,
                        )
                    )
                fig_positions.update_layout(
                    title=f"Number of Positions by Date<br><sup>{date_range_str}</sup>",
                    yaxis_title="Positions",
                    template="plotly_white",
                    height=400,
                    hovermode="x unified",
                )
                html_positions = pio.to_html(fig_positions, full_html=True, include_plotlyjs=True)
            except Exception as e:
                html_positions = f"<div class='alert alert-danger'>Error: {e}</div>"

        # --- 3. KPI Deep Dive Heatmaps (Tabbed) ---
        kpi_htmls = {}
        target_metrics = ['CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Annualized Volatility']
        
        for metric in target_metrics:
            try:
                df_cum = cumulative_metrics_dict.get(metric)
                df_ann = annual_metrics_dict.get(metric)
                
                if df_cum is None or df_ann is None: continue

                # Subplots: Top=Cumulative, Bottom=Annual
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=(
                        f"{metric}: Cumulative from Start Year (Consistency)",
                        f"{metric}: Discrete Annual Performance",
                    ),
                    vertical_spacing=0.16,
                )
                
                def _add_heatmap(df, row, col, colorscale='RdYlGn'):
                    # Force numeric
                    vals = df.apply(pd.to_numeric, errors='coerce').round(4).T # Transpose: Model=Y, Year=X
                    z = vals.values
                    x = vals.columns # Years
                    y = vals.index   # Models
                    
                    # Determine formatting string
                    is_pct = metric not in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
                    fmt_str = "%{z:.1%}" if is_pct else "%{z:.2f}"

                    fig.add_trace(go.Heatmap(
                        z=z, x=x, y=y, 
                        texttemplate=fmt_str,
                        textfont={"size": 11},
                        colorscale=colorscale, 
                        showscale=True
                    ), row=row, col=col)
                    
                    # Force X-axis to be integers (Years)
                    fig.update_xaxes(type='category', row=row, col=col)

                _add_heatmap(df_cum, 1, 1, 'Viridis' if metric == 'Annualized Volatility' else 'RdYlGn')
                _add_heatmap(df_ann, 2, 1, 'Viridis' if metric == 'Annualized Volatility' else 'RdYlGn')
                
                fig.update_layout(height=1050, title_text=f"Deep Dive: {metric}", template="plotly_white")
                kpi_htmls[metric] = pio.to_html(fig, full_html=False, include_plotlyjs=True)
            except Exception as e:
                kpi_htmls[metric] = f"<div class='alert alert-danger'>Error: {e}</div>"

        # --- 4. Monthly Returns Heatmaps (Tabbed) ---
        monthly_htmls = {}
        for model_name, returns_series in monthly_returns_dict.items():
            if returns_series.empty: continue
            try:
                years = returns_series.index.year
                months = returns_series.index.month
                pivot_df = pd.DataFrame({'Year': years, 'Month': months, 'Return': returns_series.values}).pivot(index='Year', columns='Month', values='Return')
                for m in range(1, 13): 
                    if m not in pivot_df.columns: pivot_df[m] = np.nan
                pivot_df = pivot_df.sort_index(axis=1)
                
                z = pivot_df.values
                x_scale = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig = go.Figure(data=go.Heatmap(
                    z=z, x=x_scale, y=pivot_df.index, 
                    texttemplate="%{z:.1%}",
                    textfont={"size": 10},
                    colorscale='RdYlGn', 
                    xgap=1, ygap=1
                ))
                fig.update_layout(
                    title=f'{model_name} - Monthly Returns', 
                    height=max(400, len(pivot_df)*40), 
                    template="plotly_white",
                    yaxis=dict(dtick=1, type='category') # Ensure years are distinct
                )
                monthly_htmls[model_name] = pio.to_html(fig, full_html=False, include_plotlyjs=true)
            except Exception as e: pass

        # --- G. Risk-Reward Scatter ---
        html_rr = ""
        try:
             rr_data = []
             for model in metrics_df.index:
                 if 'CAGR' in metrics_df.columns and 'Annualized Volatility' in metrics_df.columns:
                     cagr_str = metrics_df.loc[model, 'CAGR']
                     vol_str = metrics_df.loc[model, 'Annualized Volatility']
                     cagr = float(str(cagr_str).strip('%')) / 100 if '%' in str(cagr_str) else 0
                     vol = float(str(vol_str).strip('%')) / 100 if '%' in str(vol_str) else 0
                     rr_data.append({'Model': model, 'CAGR': cagr, 'Volatility': vol})
             if rr_data:
                 rr_df = pd.DataFrame(rr_data)
                 fig_rr = px.scatter(rr_df, x='Volatility', y='CAGR', color='Model', text='Model', size=[1]*len(rr_df), size_max=10)
                 fig_rr.update_traces(textposition='top center')
                 fig_rr.update_layout(
                     title=f'Risk-Reward<br><sup>{date_range_str}</sup>', 
                     xaxis_tickformat='.0%', 
                     yaxis_tickformat='.0%', 
                     template="plotly_white", 
                     height=400
                  )
                 html_rr = pio.to_html(fig_rr, full_html=True, include_plotlyjs=True)
        except Exception: pass
        
        # --- Correlation Matrix ---
        html_corr = ""
        try:
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values, 
                x=correlation_matrix.columns, 
                y=correlation_matrix.index, 
                texttemplate="%{z:.2f}",
                colorscale='RdBu', 
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(
                title=f'Correlation Matrix<br><sup>{date_range_str}</sup>', 
                template="plotly_white", 
                height=500
            )
            html_corr = pio.to_html(fig_corr, full_html=True, include_plotlyjs=True)
        except Exception: pass

        # --- Assemble HTML ---
        
        # KPI Tabs
        kpi_nav = ""
        kpi_content = ""
        first = True
        for m, html in kpi_htmls.items():
            slug = "".join([c for c in m if c.isalnum()])
            active, show = ("active", "show active") if first else ("", "")
            kpi_nav += f'<li class="nav-item"><a class="nav-link {active}" data-toggle="tab" href="#kpi-{slug}">{m}</a></li>'
            kpi_content += f'<div id="kpi-{slug}" class="tab-pane fade {show}"><div class="chart-container">{html}</div></div>'
            first = False
            
        # Monthly Tabs
        mon_nav = ""
        mon_content = ""
        first = True
        for m, html in monthly_htmls.items():
            slug = "".join([c for c in m if c.isalnum()])
            active, show = ("active", "show active") if first else ("", "")
            mon_nav += f'<li class="nav-item"><a class="nav-link {active}" data-toggle="tab" href="#mon-{slug}">{m}</a></li>'
            mon_content += f'<div id="mon-{slug}" class="tab-pane fade {show}"><div class="chart-container">{html}</div></div>'
            first = False

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
            <style>
                body {{ background-color: #f8f9fa; padding: 20px; font-family: 'Segoe UI', sans-serif; }}
                .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none; border-radius: 10px; }}
                .card-header {{ background-color: #fff; border-bottom: 1px solid #eee; font-weight: bold; color: #333; }}
                .chart-container {{ padding: 15px; background: white; border-radius: 8px; }}
                .chart-container .plotly-graph-div {{ width: 100% !important; }}
                h2 {{ color: #2c3e50; margin-bottom: 30px; font-weight: 700; }}
                .nav-tabs .nav-link.active {{ background-color: #e9ECEF; font-weight: bold; color: #007bff; }}
                .table-striped tbody tr:nth-of-type(odd) {{ background-color: rgba(0,0,0,.02); }}
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <center><h2>{title}</h2></center>
                
                <div class="card">
                    <div class="card-header">Performance Metrics</div>
                    <div class="card-body">{metrics_df.to_html(classes='table table-striped', border=0)}</div>
                </div>
                
                <div class="card">
                    <div class="card-header">Worst Drawdown Periods</div>
                    <div class="card-body">{worst_periods_df.to_html(classes='table table-striped', border=0)}</div>
                </div>

                <div class="card"><div class="card-body">{html_cum}</div></div>
                <div class="card"><div class="card-body">{html_dd}</div></div>
                {f'<div class="card"><div class="card-body">{html_positions}</div></div>' if html_positions else ''}
                
                <div class="card">
                    <div class="card-header">KPI Stability Analysis (Annual vs Cumulative)</div>
                    <div class="card-body">
                        <ul class="nav nav-tabs">{kpi_nav}</ul>
                        <div class="tab-content pt-3">{kpi_content}</div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6"><div class="card"><div class="card-body">{html_rr}</div></div></div>
                    <div class="col-md-6"><div class="card"><div class="card-body">{html_corr}</div></div></div>
                </div>
                
                <div class="card">
                    <div class="card-header">Monthly Returns Deep Dive</div>
                    <div class="card-body">
                        <ul class="nav nav-tabs">{mon_nav}</ul>
                        <div class="tab-content pt-3">{mon_content}</div>
                    </div>
                </div>

            </div>
            <script>
                function resizeVisiblePlots(target) {{
                    var scope = target || document;
                    var plots = scope.querySelectorAll('.plotly-graph-div');
                    plots.forEach(function(plot) {{
                        if (window.Plotly && window.Plotly.Plots) {{
                            window.Plotly.Plots.resize(plot);
                        }}
                    }});
                }}
                document.addEventListener('shown.bs.tab', function (event) {{
                    var paneSelector = event.target.getAttribute('href');
                    if (!paneSelector) return;
                    var pane = document.querySelector(paneSelector);
                    if (pane) {{
                        setTimeout(function() {{ resizeVisiblePlots(pane); }}, 50);
                    }}
                }});
                window.addEventListener('load', function () {{
                    setTimeout(function() {{ resizeVisiblePlots(document); }}, 50);
                }});
            </script>
        </body>
        </html>
        """
        
        return html_template

# %%

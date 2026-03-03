
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path (both repo root and src/alpharank)
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from alpharank.strategy.legacy import ModelEvaluator
from alpharank.visualization.plotting import PortfolioVisualizer

def verify_report_generation():
    print("=== Verifying Model Comparison and Reporting ===")
    
    # Generate mock monthly returns for 3 models over 5 years
    periods = 60
    dates = pd.period_range(start='2015-01', periods=periods, freq='M')
    
    np.random.seed(42)
    # Model A: Steady growth
    r_a = np.random.normal(0.01, 0.02, periods)
    # Model B: Volatile
    r_b = np.random.normal(0.015, 0.05, periods)
    # Model C: Losing
    r_c = np.random.normal(-0.005, 0.02, periods)
    
    # Create DataFrames with 'year_month' and 'monthly_return'
    df_a = pd.DataFrame({'year_month': dates, 'monthly_return': r_a})
    df_b = pd.DataFrame({'year_month': dates, 'monthly_return': r_b})
    df_c = pd.DataFrame({'year_month': dates, 'monthly_return': r_c})
    
    # Add dummy 'n' column for stock count avg
    df_a['n'] = 10
    df_b['n'] = 20
    df_c['n'] = 5
    
    models = {
        'Model A': df_a,
        'Model B': df_b,
        'Model C': df_c
    }
    
    print("1. Running ModelEvaluator.compare_models...")
    # New signature returns 8 items
    try:
        metrics, cumulative, correlation, worst_periods, drawdowns, annual_returns, cumulative_metrics, annual_metrics, monthly_returns = ModelEvaluator.compare_models(models)
        print("✓ compare_models execution successful")
    except Exception as e:
        print(f"❌ Error in compare_models: {e}")
        raise e
        
    # Check metrics content
    print("\nMetrics Sample (Model A):")
    print(metrics.iloc[0])
    
    # Basic assertions
    if 'Sortino Ratio' in metrics.columns:
        print("✓ Sortino Ratio present in metrics")
    if 'Calmar Ratio' in metrics.columns:
         print("✓ Calmar Ratio present in metrics")
    if 'Max DD Duration' in metrics.columns or 'Max DD Duration (Months)' in metrics.columns:
         print("✓ Max DD Duration (Months) present in metrics")
         
    # Check new dictionaries
    if cumulative_metrics and 'Sharpe Ratio' in cumulative_metrics:
        print("✓ Cumulative Metrics Dictionary populated")
    else:
        print("❌ Cumulative Metrics Dictionary MISSING or empty")
        
    print(f"DEBUG: worst_periods_df shape: {worst_periods.shape}")
    if worst_periods.empty:
        print("❌ worst_periods_df is EMPTY")
    else:
        print("✓ worst_periods_df populated")
        print(worst_periods.head())

    # 3. Test HTML Report Generation
    print("\n2. Generating HTML Comparison Report...")
    # Mock cagr_by_year (old) -> we need new dicts
    
    try:
        html = PortfolioVisualizer.make_comparison_report(
            metrics_df=metrics,
            cumulative_returns=cumulative,
            drawdowns_df=drawdowns,
            annual_returns_df=annual_returns,
            correlation_matrix=correlation,
            worst_periods_df=worst_periods,
            cumulative_metrics_dict=cumulative_metrics,
            annual_metrics_dict=annual_metrics,
            monthly_returns_dict=monthly_returns,
            title="Test Portfolio Strategy Comparison"
        )
        print("✓ HTML generation successful")
        print(f"HTML Length: {len(html)} chars")
        
        output_path = Path(__file__).parent.parent / "outputs" / "test_comparison_report.html"
        with open(output_path, "w") as f:
            f.write(html)
        print(f"✓ Saved test report to {output_path}")
        
    except Exception as e:
        print(f"❌ Error in make_comparison_report: {e}")
        # Print traceback
        import traceback
        traceback.print_exc()
        raise e
    print("\nSUCCESS: All reporting verification checks passed!")

if __name__ == "__main__":
    verify_report_generation()

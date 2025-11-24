import optuna
import optuna.visualization as vis
import plotly.io as pio
from typing import Optional

def generate_optuna_report(study: optuna.Study, output_path: str) -> None:
    """
    Generates an HTML report containing standard Optuna visualizations.
    
    Args:
        study: The Optuna study object.
        output_path: Path to save the HTML report.
    """
    try:
        # Generate plots
        figs = []
        
        # Optimization History
        try:
            figs.append(vis.plot_optimization_history(study))
        except Exception as e:
            print(f"Could not plot optimization history: {e}")
            
        # Parameter Importances
        try:
            # Requires more than one parameter to be useful, and completed trials
            if len(study.trials) > 1:
                figs.append(vis.plot_param_importances(study))
        except Exception as e:
            print(f"Could not plot param importances: {e}")
            
        # Parallel Coordinate
        try:
            figs.append(vis.plot_parallel_coordinate(study))
        except Exception as e:
            print(f"Could not plot parallel coordinate: {e}")
            
        # Slice Plot
        try:
            figs.append(vis.plot_slice(study))
        except Exception as e:
            print(f"Could not plot slice: {e}")

        if not figs:
            print("No figures generated for Optuna report.")
            return

        # Combine into a single HTML file
        with open(output_path, 'w') as f:
            f.write("<html><head><title>Optuna Optimization Report</title></head><body>")
            f.write("<h1>Optuna Optimization Report</h1>")
            
            for i, fig in enumerate(figs):
                # Convert plotly figure to HTML div
                # full_html=False ensures we don't get a full HTML document for each plot, just the div
                # include_plotlyjs='cdn' ensures we load plotly.js from CDN once (or we can handle it differently)
                # simpler: use to_html
                plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)
                f.write(f"<div>{plot_html}</div>")
                f.write("<hr>")
                
            f.write("</body></html>")
            
        print(f"Optuna report saved to {output_path}")

    except Exception as e:
        print(f"Failed to generate Optuna report: {e}")

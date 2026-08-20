import optuna
import optuna.visualization as vis
import plotly.io as pio
from typing import Optional

from alpharank.observability import get_run_logger


LOGGER = get_run_logger(__name__)

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
        except (RuntimeError, TypeError, ValueError) as e:
            LOGGER.warning(
                "Optuna optimization-history plot unavailable",
                extra={"result": "skipped", "error": str(e)},
            )
            
        # Parameter Importances
        try:
            # Requires more than one parameter to be useful, and completed trials
            if len(study.trials) > 1:
                figs.append(vis.plot_param_importances(study))
        except (RuntimeError, TypeError, ValueError) as e:
            LOGGER.warning(
                "Optuna parameter-importance plot unavailable",
                extra={"result": "skipped", "error": str(e)},
            )
            
        # Parallel Coordinate
        try:
            figs.append(vis.plot_parallel_coordinate(study))
        except (RuntimeError, TypeError, ValueError) as e:
            LOGGER.warning(
                "Optuna parallel-coordinate plot unavailable",
                extra={"result": "skipped", "error": str(e)},
            )
            
        # Slice Plot
        try:
            figs.append(vis.plot_slice(study))
        except (RuntimeError, TypeError, ValueError) as e:
            LOGGER.warning(
                "Optuna slice plot unavailable",
                extra={"result": "skipped", "error": str(e)},
            )

        if not figs:
            LOGGER.warning("No Optuna figures were generated", extra={"result": "empty"})
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
            
        LOGGER.info(
            "Optuna report saved",
            extra={"output_path": output_path, "result": "completed"},
        )

    except (OSError, RuntimeError, TypeError, ValueError) as e:
        LOGGER.exception(
            "Optuna report generation failed",
            extra={"output_path": output_path, "result": "failed", "error": str(e)},
        )

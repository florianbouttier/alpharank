import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any
import seaborn as sns

# --- Main Functions ---

def compute_shap_values(
    model: Any, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both SHAP values and SHAP interaction values for a given model.
    
    Args:
        model: A trained tree-based model (like XGBoost, LightGBM, etc.).
        X: The input data (Pandas DataFrame).
        
    Returns:
        A tuple containing:
        - shap_values (np.ndarray): The contribution of each feature to each prediction.
        - shap_interaction_values (np.ndarray): The interaction effects between features.
    """
    # Create an explainer for the model. TreeExplainer is highly optimized for tree-based models.
    explainer = shap.TreeExplainer(model)
    
    # Calculate the SHAP values
    shap_values = explainer.shap_values(X)
    
    # Calculate the SHAP interaction values
    # shap_interaction_values = explainer.shap_interaction_values(X)
    
    return shap_values #, shap_interaction_values

def run_shap_analysis(model: Any, X: pd.DataFrame, top_n_heatmap: int = 10):
    """
    Runs a complete SHAP analysis and displays plots directly in a Jupyter environment.
    
    This function will display:
    1. A global beeswarm summary plot.
    2. A global interaction heatmap for the top N features.
    3. 2D interaction scatter plots for all features, ordered by importance.
    
    Args:
        model: A trained tree-based model.
        X: The input data (Pandas DataFrame).
        top_n_heatmap: The number of top features to display on the heatmap.
    """
    print("Starting SHAP analysis for Jupyter...")
    
    # 1. Create an explainer and compute all SHAP values
    print("Step 1/4: Computing SHAP values and interactions (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    # Main effects wrapped in an Explanation object
    shap_values_exp = explainer(X)
    # Interaction effects as a numpy array
    # shap_interaction_values = explainer.shap_interaction_values(X)
    
    # 2. Display the beeswarm summary plot
    print("Step 2/4: Displaying beeswarm plot...")
    shap.plots.beeswarm(shap_values_exp, max_display=top_n_heatmap)
    
    # 3. Display the global interaction heatmap
    # print(f"Step 3/4: Displaying global interaction heatmap for top {top_n_heatmap} features...")
    # Aggregate the absolute interaction values across all samples
    # mean_abs_interaction = np.abs(shap_interaction_values).mean(axis=0)
    # Get the top N features based on the mean absolute main effect
    # main_effects = np.abs(shap_values_exp.values).mean(axis=0)
    # top_indices = np.argsort(-main_effects)[:top_n_heatmap]
    
    # Slice the interaction matrix to keep only top features
    # top_features_interaction = mean_abs_interaction[top_indices, :][:, top_indices]
    # top_feature_names = X.columns[top_indices]
    
    # Plot the heatmap using seaborn
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(
    #     top_features_interaction,
    #     xticklabels=top_feature_names,
    #     yticklabels=top_feature_names,
    #     annot=True,
    #     fmt=".3f",
    #     cmap="viridis"
    # )
    # plt.title(f"Heatmap of Mean Absolute SHAP Interaction Values (Top {top_n_heatmap} Features)")
    # plt.show()

    # 4. Display interaction scatter plots for each feature
    # print("Step 4/4: Displaying interaction scatter plots (ordered by importance)...")
    # Use the same feature order as the heatmap for consistency
    # for idx in top_indices:
    #     shap.plots.scatter(shap_values_exp[:, idx])

    print("SHAP analysis complete.")

def save_waterfall_for_row(
    model, shap_row: np.ndarray, base_value: float, x_row: pd.Series, out_path: str, max_display: int = 20
):
    """Saves a waterfall plot for a single prediction."""
    explainer = shap.TreeExplainer(model) # This needs a model passed to it to get base_value properly
    base_value = explainer.expected_value[0] if hasattr(explainer.expected_value, '__iter__') else explainer.expected_value
    
    exp = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=x_row.values,
        feature_names=x_row.index
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def expit(x: np.ndarray) -> np.ndarray:
    """
    Computes the logistic function (expit) for an array of values.
    
    Args:
        x (np.ndarray): Input array of values.

    Returns:
        np.ndarray: The computed expit values.
    """
    return 1 / (1 + np.exp(-x))

def compute_best_shap(shap_object: Any, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Computes the best SHAP values and returns a DataFrame with the top N features.
    
    Args:
        shap_object (Any): The SHAP object containing the values.
        df (pd.DataFrame): The original DataFrame with feature values.
        top_n (int): The number of top features to return.

    Returns:
        pd.DataFrame: A DataFrame with the top N features and their SHAP values.
    """
    # Handle different SHAP object types (array vs Explanation object)
    if isinstance(shap_object, np.ndarray):
         values = shap_object
         # We need base values if it's just an array, but usually it's an Explanation object
         # If it's an array, we might assume it's just the interaction values or similar
         # For now, let's assume it's an Explanation object or has .values
         pass
    
    if hasattr(shap_object, 'values'):
        values = shap_object.values
        base_values = shap_object.base_values
    else:
        # Fallback if it's a list or something else, though unlikely with standard SHAP usage
        return pd.DataFrame()

    # Sum of base value + sum of SHAP values = Prediction (in link space)
    # For classification (logit), this is the log-odds.
    if len(values.shape) > 1:
        sum_values = base_values + values.sum(axis=1)
    else:
        sum_values = base_values + values.sum()

    top_n_index = np.argsort(sum_values)[-top_n:][::-1]
    tickers = []
    
    # Check if 'ticker' column exists in df
    if 'ticker' not in df.columns:
        return pd.DataFrame()

    for best_prediction_index in top_n_index:
        ticker_loop = df['ticker'].iloc[int(best_prediction_index)]
        shap_val = sum_values[best_prediction_index]
        prediction = expit(shap_val)
        
        df_tickers_loop = pd.DataFrame({
            'ticker': [ticker_loop],
            'shap_value': [shap_val],
            'prediction': [prediction]
        })
        tickers.append(df_tickers_loop)
        
    if not tickers:
        return pd.DataFrame()
        
    return pd.concat(tickers, ignore_index=True)

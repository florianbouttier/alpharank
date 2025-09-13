import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost
from typing import List

# Assuming this utility function exists
from ..data_processor.utils import convert_to_category

class ShapAnalyzer:
    """
    A robust SHAP analyzer. This final version uses a custom jitter plot
    for categorical features to ensure reliability and the standard scatter
    plot for numeric features.
    """
    def __init__(self, model: any, framework: str, cat_features: list = None):
        if not hasattr(model, 'predict'):
            raise TypeError("Model must have a 'predict' method.")
        if framework not in ['xgb', 'cat', 'lgb']:
            raise ValueError("Framework must be 'xgb', 'cat', or 'lgb'.")

        self.model = model
        self.framework = framework
        self.cat_features = cat_features if cat_features is not None else []
        self.explainer = shap.TreeExplainer(self.model)
        self._shap_values = None
        self._data_for_shap = None

    def _prepare_data_for_model(self, data: pd.DataFrame) -> pd.DataFrame:
        df_prepared = data.copy()
        if self.framework == 'xgb' and self.cat_features:
            df_prepared = convert_to_category(df_prepared, self.cat_features)
        return df_prepared

    def _calculate_shap_values(self, data: pd.DataFrame):
        if self._shap_values is None or not self._data_for_shap.equals(data):
            self._data_for_shap = data.copy()
            X_prepared_for_model = self._prepare_data_for_model(self._data_for_shap)
            self._shap_values = self.explainer(X_prepared_for_model)

    def get_feature_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        self._calculate_shap_values(data)
        vals = np.abs(self._shap_values.values).mean(axis=0)
        return pd.DataFrame({'feature': data.columns, 'importance': vals}).sort_values('importance', ascending=False).reset_index(drop=True)

    def plot_numeric_feature(self, feature: str, data: pd.DataFrame):
        """Plots a SHAP scatter plot for a single numeric feature."""
        self._calculate_shap_values(data)
        shap.plots.scatter(self._shap_values[:, feature], color=self._shap_values[:, feature], show=False)
        plt.show()

    def plot_categorical_jitter(self, feature: str, data: pd.DataFrame):
        """
        Creates a custom jitter plot for a single categorical feature to
        reliably visualize SHAP values without errors.
        """
        self._calculate_shap_values(data)
        shap_values = self._shap_values[:, feature].values
        feature_data = self._shap_values[:, feature].data

        plot_df = pd.DataFrame({
            'feature_values': feature_data,
            'shap_values': shap_values
        })
        
        # Order categories by median SHAP value for a cleaner plot
        order = plot_df.groupby('feature_values')['shap_values'].median().sort_values().index

        plt.figure(figsize=(10, 6))
        sns.stripplot(
            x='feature_values',
            y='shap_values',
            data=plot_df,
            order=order,
            jitter=0.3,
            alpha=0.5,
            size=4
        )
        
        plt.ylabel(f"SHAP value for {feature}")
        plt.xlabel(feature)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"SHAP Value Distribution for {feature}")
        plt.tight_layout()
        plt.show()

    def run_analysis(self, data: pd.DataFrame, top_n: int = 10):
        """
        Runs a complete analysis using the appropriate plot for each feature type.
        """
        print("Calculating SHAP values and feature importance...")
        importance_df = self.get_feature_importance(data)
        
        print("\n" + "="*80)
        print("STEP 1: Global Feature Importance (Beeswarm Plot)")
        print("="*80)
        shap.plots.beeswarm(self._shap_values, max_display=top_n)
        plt.show()
        
        print("\n" + "="*80)
        print(f"STEP 2: Detailed Dependency Plots for Top {top_n} Features")
        print("="*80)
        
        top_features = importance_df['feature'].head(top_n).tolist()
        
        for feature in top_features:
            print(f"\n--- Plotting for feature: {feature} ---")
            
            # --- ROUTING LOGIC ---
            # Use the custom jitter plot for categorical features.
            if feature in self.cat_features:
                print(f"'{feature}' is categorical. Using custom Jitter Plot.")
                self.plot_categorical_jitter(feature, data)
            else:
                print(f"'{feature}' is numeric. Using standard Scatter Plot.")
                self.plot_numeric_feature(feature, data)
        
        print("\nAnalysis complete.")
        return importance_df
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies in AlphaRank.
    """

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Train the strategy model.
        
        Args:
            data: Training data.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate predictions/signals.
        
        Args:
            data: Input data for prediction.
            **kwargs: Additional arguments.
        
        Returns:
            DataFrame with predictions (e.g., scores, ranks, or target values).
        """
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the strategy.
        
        Args:
            data: Data for optimization.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary of optimized parameters.
        """
        pass

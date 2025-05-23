"""
Emission prediction model for the Carbon Emission Calculator.
This module implements the specific ML model for predicting carbon emissions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class EmissionModel(BaseModel):
    """
    Model for predicting carbon emissions from flights.
    
    Attributes:
        model (RandomForestRegressor): The actual ML model
        feature_importance (pd.Series): Feature importance scores
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the emission model.
        
        Args:
            model_params (Dict, optional): Model parameters
        """
        super().__init__(model_params)
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Update default parameters with any provided parameters
        if model_params:
            default_params.update(model_params)
            
        self.model = RandomForestRegressor(**default_params)
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the emission prediction model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable (carbon emissions)
        """
        logger.info("Training emission prediction model...")
        self.model.fit(X, y)
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        logger.info("Model training completed")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict carbon emissions.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted carbon emissions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True carbon emissions
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        
        Returns:
            pd.Series: Feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained")
        return self.feature_importance 
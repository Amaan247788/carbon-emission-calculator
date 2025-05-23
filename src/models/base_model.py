"""
Base model module for the Carbon Emission Calculator.
This module defines the interface for all ML models in the project.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    Attributes:
        model: The actual ML model
        model_params (Dict): Model parameters
        model_path (Path): Path to save/load the model
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            model_params (Dict, optional): Model parameters
        """
        self.model = None
        self.model_params = model_params or {}
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Model predictions
        """
        pass
    
    def save_model(self, filename: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filename (str): Name of the file to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        filepath = self.model_path / filename
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filename: str) -> None:
        """
        Load the model from disk.
        
        Args:
            filename (str): Name of the file to load the model from
        """
        filepath = self.model_path / filename
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True target values
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        pass 
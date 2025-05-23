"""
Model trainer module for the Carbon Emission Calculator.
This module handles the training pipeline, cross-validation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import logging
import json
from pathlib import Path

from .emission_model import EmissionModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the training pipeline for the emission prediction model.
    
    Attributes:
        model (EmissionModel): The emission prediction model
        param_grid (Dict): Grid of hyperparameters to search
        cv_folds (int): Number of cross-validation folds
    """
    
    def __init__(self, param_grid: Dict[str, List[Any]] = None, cv_folds: int = 5):
        """
        Initialize the model trainer.
        
        Args:
            param_grid (Dict, optional): Grid of hyperparameters to search
            cv_folds (int): Number of cross-validation folds
        """
        self.model = EmissionModel()
        self.param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.cv_folds = cv_folds
        self.best_params = None
        self.best_score = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Create scoring metrics
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'r2': 'r2'
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model.model,
            self.param_grid,
            cv=self.cv_folds,
            scoring=scoring,
            refit='neg_mse',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Store best parameters and score
        self.best_params = grid_search.best_params_
        self.best_score = -grid_search.best_score_  # Convert back to positive MSE
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best MSE score: {self.best_score:.4f}")
        
        return self.best_params
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   tune_hyperparameters: bool = True) -> EmissionModel:
        """
        Train the emission prediction model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            EmissionModel: Trained model
        """
        if tune_hyperparameters:
            best_params = self.tune_hyperparameters(X, y)
            self.model = EmissionModel(model_params=best_params)
        
        # Train the model
        self.model.train(X, y)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model.model,
            X,
            y,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error'
        )
        
        logger.info(f"Cross-validation MSE scores: {-cv_scores}")
        logger.info(f"Average CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def save_training_results(self, output_dir: str = "training_results") -> None:
        """
        Save training results to disk.
        
        Args:
            output_dir (str): Directory to save results
        """
        if self.best_params is None:
            logger.warning("No training results to save")
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            'best_parameters': self.best_params,
            'best_score': float(self.best_score),
            'cv_folds': self.cv_folds
        }
        
        with open(output_path / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Training results saved to {output_path / 'training_results.json'}") 
"""
Visualization module for the Carbon Emission Calculator.
This module handles plotting model results and feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """
    Handles visualization of model results and feature importance.
    """
    
    def __init__(self, output_dir: str = "training_results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_feature_importance(self, feature_importance: pd.Series, top_n: int = 10) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance (pd.Series): Feature importance scores
            top_n (int): Number of top features to plot
        """
        plt.figure(figsize=(12, 6))
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        sns.barplot(x=top_features.values, y=top_features.index)
        
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
        
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Predicted vs Actual Carbon Emissions')
        plt.xlabel('Actual Emissions (kg)')
        plt.ylabel('Predicted Emissions (kg)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'prediction_vs_actual.png')
        plt.close()
        
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot error distribution.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate errors
        errors = y_pred - y_true
        
        # Create histogram with fixed number of bins
        plt.hist(errors, bins=50, density=True, alpha=0.7)
        
        # Add KDE
        sns.kdeplot(data=errors, color='red', linewidth=2)
        
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Prediction Error (kg)')
        plt.ylabel('Density')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'error_distribution.png')
        plt.close()
        
    def plot_learning_curves(self, train_sizes: np.ndarray, train_scores: np.ndarray, 
                           test_scores: np.ndarray) -> None:
        """
        Plot learning curves.
        
        Args:
            train_sizes (np.ndarray): Training set sizes
            train_scores (np.ndarray): Training scores
            test_scores (np.ndarray): Test scores
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'learning_curves.png')
        plt.close() 
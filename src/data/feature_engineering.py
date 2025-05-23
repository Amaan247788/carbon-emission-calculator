"""
Feature engineering module for the Carbon Emission Calculator.
This module handles the creation and transformation of features for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    A class to handle feature engineering for flight data.
    
    Attributes:
        df (pd.DataFrame): Input DataFrame containing flight data
        scaler (StandardScaler): Scaler for numerical features
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngineer.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing flight data
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        
    def create_features(self) -> pd.DataFrame:
        """
        Create and transform features for ML model.
        
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        # Create basic features
        self._create_distance_features()
        self._create_categorical_features()
        self._create_derived_features()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Scale numerical features
        self._scale_numerical_features()
        
        return self.df
    
    def _create_distance_features(self) -> None:
        """Create features based on distance."""
        # Log transform distance to handle skewness
        self.df['log_distance'] = np.log1p(self.df['distance_km'])
        
        # Create distance bins
        self.df['distance_bin'] = pd.qcut(
            self.df['distance_km'],
            q=5,
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )
    
    def _create_categorical_features(self) -> None:
        """Create categorical features."""
        # One-hot encode categorical variables
        categorical_cols = ['distance_bin', 'is_short_haul']
        self.df = pd.get_dummies(self.df, columns=categorical_cols, prefix=categorical_cols)
    
    def _create_derived_features(self) -> None:
        """Create derived features."""
        # Create route identifier
        self.df['route'] = self.df['departure_iata'] + '_' + self.df['arrival_iata']
        
        # Calculate average emissions per route
        route_emissions = self.df.groupby('route')['carbon_emission'].mean()
        self.df['avg_route_emission'] = self.df['route'].map(route_emissions)
        
        # Calculate emission efficiency (emissions per km)
        self.df['emission_efficiency'] = self.df['carbon_emission'] / self.df['distance_km']
    
    def _handle_missing_values(self) -> None:
        """Handle missing values in the dataset."""
        # Fill missing values in numerical columns with median
        numerical_cols = ['distance_km', 'log_distance', 'carbon_emission', 
                         'avg_route_emission', 'emission_efficiency']
        
        for col in numerical_cols:
            if col in self.df.columns:
                median_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_value)
        
        # Fill missing values in categorical columns with mode
        categorical_cols = [col for col in self.df.columns if col.startswith(('distance_bin_', 'is_short_haul_'))]
        for col in categorical_cols:
            if col in self.df.columns:
                mode_value = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_value)
    
    def _scale_numerical_features(self) -> None:
        """Scale numerical features."""
        numerical_cols = ['distance_km', 'log_distance', 'carbon_emission', 
                         'avg_route_emission', 'emission_efficiency']
        
        # Scale numerical features
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
        
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List[str]: List of feature names
        """
        return [col for col in self.df.columns if col not in 
                ['departure_iata', 'arrival_iata', 'route']] 
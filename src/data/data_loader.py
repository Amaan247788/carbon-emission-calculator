"""
Data loader module for the Carbon Emission Calculator.
This module handles loading and initial preprocessing of flight data.
"""

import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightDataLoader:
    """
    A class to handle loading and preprocessing of flight data.
    
    Attributes:
        file_path (str): Path to the JSON file containing flight data
        data (Dict): Raw flight data loaded from JSON
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the FlightDataLoader.
        
        Args:
            file_path (str): Path to the JSON file containing flight data
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> Dict:
        """
        Load flight data from JSON file.
        
        Returns:
            Dict: Raw flight data
        """
        try:
            with open(self.file_path, 'r') as file:
                self.data = json.load(file)
            logger.info(f"Successfully loaded data from {self.file_path}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the raw flight data into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Processed flight data
        """
        if self.data is None:
            self.load_data()
            
        processed_data = []
        
        for departure_iata, airport_data in self.data.items():
            routes = airport_data.get('routes', [])
            for route in routes:
                processed_route = {
                    'departure_iata': departure_iata,
                    'arrival_iata': route.get('iata'),
                    'distance_km': route.get('km'),
                    'is_short_haul': route.get('km', 0) < 1500,
                    'carbon_emission': self._calculate_carbon_emission(route.get('km', 0))
                }
                processed_data.append(processed_route)
                
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(df)} flight routes")
        return df
    
    def _calculate_carbon_emission(self, distance_km: float) -> float:
        """
        Calculate carbon emission for a given distance.
        
        Args:
            distance_km (float): Distance in kilometers
            
        Returns:
            float: Calculated carbon emission in kg
        """
        if distance_km < 1500:
            return round(distance_km * 0.175, 1)  # Short haul emission factor
        return round(distance_km * 0.135, 1)  # Long haul emission factor 